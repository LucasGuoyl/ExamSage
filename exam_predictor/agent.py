"""ExamSage's user-facing agent orchestration."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from .cloud_analyzer import CloudNormalizationResult, normalize_course_uploads
from .generator import generate_for_knowledge_point, sample_style_examples
from .ingest import discover_past_papers
from .pipeline import ExamPredictor
from .providers import BaseProvider
from .schema import Chunk, KnowledgeTreeNode, PredictionReport, WebEvidence
from .security import validate_public_https_url
from .state import CourseStore


class ExamSageAgent:
    """One provider-backed agent that can build a report and continue chatting."""

    def __init__(
        self,
        provider: BaseProvider,
        config: dict[str, Any],
        data_dir: str | Path,
    ):
        self.provider = provider
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = CourseStore(self.data_dir)

    def build_course(
        self,
        upload_paths: list[str | Path],
        user_request: str,
        *,
        course_name: str | None = None,
        source_urls: list[str] | None = None,
    ) -> tuple[str, PredictionReport, CloudNormalizationResult]:
        run_id = uuid.uuid4().hex
        workspace = self.data_dir / "courses" / run_id
        prepared_paths = [Path(item) for item in upload_paths]
        url_evidence: list[WebEvidence] = []
        url_dir = workspace / "url_inputs"
        for index, raw_url in enumerate(source_urls or [], 1):
            url = validate_public_https_url(raw_url)
            result = self.provider.web_search(
                url,
                prompt=(
                    "Read and summarize the academic content at this exact public URL for use as "
                    "course material. Preserve definitions, formulas, section structure and source context."
                ),
            )
            url_evidence.append(WebEvidence(
                query=url,
                summary=result.summary,
                citations=result.citations,
                limitations=["This public webpage was supplied by the user and processed through provider-native web grounding."],
            ))
            url_dir.mkdir(parents=True, exist_ok=True)
            source_path = url_dir / f"webpage_{index:03d}.md"
            source_path.write_text(
                f"# Web course source\n\nOriginal URL: {url}\n\n{result.summary}",
                encoding="utf-8",
            )
            prepared_paths.append(source_path)
        normalized = normalize_course_uploads(
            self.provider,
            prepared_paths,
            workspace / "normalized",
            user_request=user_request,
        )

        pipeline_config = _cloud_pipeline_config(self.config)
        predictor = ExamPredictor(pipeline_config, provider=self.provider)
        resolved_name = course_name or normalized.course_name or "Untitled course"
        report = predictor.predict(
            normalized.course_dir,
            course_name=resolved_name,
            course_context=user_request,
        )
        report.provider = self.provider.name
        report.source_language = normalized.detected_language
        report.warnings.extend(normalized.warnings)
        report.warnings.extend(finding.message for finding in normalized.findings)

        report.web_evidence = url_evidence
        if self._needs_web_research(report):
            report.web_evidence.extend(self._research_sparse_topics(report))
            self._add_grounded_variants(report, normalized)
        report.knowledge_tree = self._build_knowledge_tree(report)
        report.study_guide = self._build_study_guide(report, user_request)

        output_dir = workspace / "output"
        predictor.save_report(report, output_dir)
        course_id = self.store.save_course(
            name=report.course_name,
            provider=self.provider.name,
            workspace_path=workspace,
            report=report.model_dump(),
            course_id=run_id,
        )
        return course_id, report, normalized

    @staticmethod
    def _needs_web_research(report: PredictionReport) -> bool:
        return (
            report.n_past_questions < 10
            or report.overall_confidence < 0.60
            or any(point.confidence < 0.55 for point in report.predictions[:8])
        )

    def _research_sparse_topics(self, report: PredictionReport) -> list[WebEvidence]:
        max_queries = int(self.config.get("research", {}).get("max_queries", 6))
        evidence: list[WebEvidence] = []
        for point in report.predictions[:max_queries]:
            query = (
                f"{report.course_name} {point.title} university course exam practice "
                "syllabus lecture notes"
            )
            result = self.provider.web_search(
                query,
                prompt=(
                    "Use reliable global university or academic sources. Clarify the concept, "
                    "describe common assessment patterns, and suggest original practice directions. "
                    "Do not reproduce a complete copyrighted exam. Include citations."
                ),
            )
            evidence.append(WebEvidence(
                knowledge_point_id=point.knowledge_point_id,
                query=query,
                summary=result.summary,
                citations=result.citations,
                limitations=[
                    "External material supports understanding and practice only; it does not prove what this instructor will test."
                ],
            ))
        return evidence

    def _add_grounded_variants(
        self,
        report: PredictionReport,
        normalized: CloudNormalizationResult,
    ) -> None:
        """Replace a small tail of sparse-topic questions with cited external variants."""

        past_questions = []
        from .chunker import questions_from_records

        past_questions = questions_from_records(discover_past_papers(normalized.course_dir))
        by_point = {item.knowledge_point_id: item for item in report.web_evidence}
        for point in report.predictions:
            evidence = by_point.get(point.knowledge_point_id)
            if not evidence or not evidence.summary:
                continue
            existing = [q for q in report.generated_questions if q.knowledge_point_id == point.knowledge_point_id]
            if not existing:
                continue
            replacement_count = min(2, len(existing))
            source_urls = [citation.url for citation in evidence.citations[:3]]
            chunks = [Chunk(
                id=f"web_{point.knowledge_point_id}",
                source="grounded web research",
                text=evidence.summary,
                metadata={"kp_id": point.knowledge_point_id},
            )]
            variants = generate_for_knowledge_point(
                self.provider.chat_client,
                self.provider.models.reasoning,
                kp_title=point.title,
                kp_chunks=chunks,
                style_examples=sample_style_examples(past_questions, n=3),
                n_candidates=replacement_count,
                max_tokens=4096,
                report_language=self.config.get("generation", {}).get("report_language", "auto"),
                avoid_questions=[q.text for q in existing],
            )
            if not variants:
                continue
            for variant in variants:
                variant.source_kind = "generated_variant"
                variant.source_reference = ", ".join(source_urls) or evidence.query
            remove_ids = {id(item) for item in existing[-len(variants):]}
            report.generated_questions = [
                item for item in report.generated_questions if id(item) not in remove_ids
            ] + variants

    def _build_knowledge_tree(self, report: PredictionReport) -> list[KnowledgeTreeNode]:
        compact = [
            {
                "id": point.knowledge_point_id,
                "title": point.title,
                "summary": point.description,
                "importance": round(point.score, 3),
            }
            for point in report.predictions
        ]
        prompt = """Build a clear chapter tree for this course. Group every supplied
knowledge-point id exactly once under sensible chapters and optional subchapters.
Prerequisites must refer to supplied ids only. Return ONLY JSON:
{{"chapters":[{{"id":"chapter-1","title":"...","summary":"...","knowledge_point_ids":[],"prerequisites":[],"children":[]}}]}}.
Course: {course}\nKnowledge points: {points}""".format(
            course=report.course_name,
            points=json.dumps(compact, ensure_ascii=False),
        )
        try:
            response = self.provider.chat_client.chat.completions.create(
                model=self.provider.models.reasoning,
                messages=[
                    {"role": "system", "content": "You are an expert curriculum architect. Output strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=5000,
            )
            raw = response.choices[0].message.content or ""
            start, end = raw.find("{"), raw.rfind("}")
            parsed = json.loads(raw[start:end + 1])
            return [KnowledgeTreeNode.model_validate(item) for item in parsed.get("chapters", [])]
        except Exception:
            return [KnowledgeTreeNode(
                id="chapter-1",
                title=report.course_name,
                summary="Course knowledge points ordered by estimated exam importance.",
                knowledge_point_ids=[point.knowledge_point_id for point in report.predictions],
            )]

    def _build_study_guide(self, report: PredictionReport, user_request: str) -> str:
        facts = [
            {
                "title": point.title,
                "importance": round(point.score, 2),
                "confidence": round(point.confidence, 2),
                "exam_directions": point.exam_directions,
            }
            for point in report.predictions[:20]
        ]
        response = self.provider.chat_client.chat.completions.create(
            model=self.provider.models.reasoning,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a university revision coach. Produce a practical Markdown study "
                        "guide in the source language. Clearly distinguish relative likelihood from "
                        "certainty and never promise that a topic will appear."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Course: {report.course_name}\nStudent request: {user_request}\n"
                        f"Evidence summary: {json.dumps(facts, ensure_ascii=False)}\n"
                        "Include: a fast revision route, prerequisite order, common mistakes, "
                        "active-recall plan, and how to use the practice set."
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=3500,
        )
        return response.choices[0].message.content or ""

    def chat(self, course_id: str, message: str) -> str:
        course = self.store.get_course(course_id)
        if course is None:
            raise ValueError("Course not found.")
        report = course["report"]
        history = self.store.messages(course_id, limit=30)
        extra_sources = ""
        if re.search(r"\b(search|source|citation|latest|current|other universit|web)\b", message, re.I):
            result = self.provider.web_search(
                f"{report.get('course_name', '')} {message}",
                prompt="Answer the student's question with trustworthy academic sources and citations.",
            )
            extra_sources = "\n\nFresh grounded web research:\n" + result.summary
            if result.citations:
                extra_sources += "\nSources:\n" + "\n".join(
                    f"- {item.title}: {item.url}" for item in result.citations
                )
        context = {
            "course_name": report.get("course_name"),
            "predictions": report.get("predictions", [])[:20],
            "study_guide": report.get("study_guide"),
            "web_evidence": report.get("web_evidence", [])[:8],
        }
        messages = [{
            "role": "system",
            "content": (
                "You are the continuing ExamSage tutor. Answer clearly and patiently until the "
                "student is satisfied. Use the course report as evidence, say when something is "
                "uncertain, and never claim an exam prediction is guaranteed.\n\n"
                + json.dumps(context, ensure_ascii=False)
            ),
        }]
        messages.extend({"role": item["role"], "content": item["content"]} for item in history)
        messages.append({"role": "user", "content": message + extra_sources})
        response = self.provider.chat_client.chat.completions.create(
            model=self.provider.models.balanced,
            messages=messages,
            temperature=0.25,
            max_tokens=3000,
        )
        answer = response.choices[0].message.content or ""
        self.store.add_message(course_id, "user", message)
        self.store.add_message(course_id, "assistant", answer)
        return answer


def _cloud_pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    """Force cloud embeddings without mutating caller-owned configuration."""

    copied = json.loads(json.dumps(config))
    copied.setdefault("embedding", {})["backend"] = "provider"
    copied.setdefault("generation", {}).setdefault("all_knowledge_points", True)
    copied["generation"].setdefault("question_batch_size", 3)
    return copied
