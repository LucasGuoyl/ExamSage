"""End-to-end pipeline.

Reads config, runs the four stages, returns a PredictionReport.

  Stage 1 — Ingest:  PDF / PPTX / MD → raw units
  Stage 2 — Align:   chunks + past questions → heat map  (Route C)
  Stage 3 — Fuse:    multi-signal features → per-chunk score  (Route D)
  Stage 4 — Generate: top-K knowledge points → candidate questions → rerank  (Route E)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI
from rich.console import Console
from rich.progress import track

from .aligner import (
    AlignmentResult,
    aggregate_chunks_to_knowledge_points,
    align_questions_to_chunks,
)
from .chunker import chunk_units, questions_from_records
from .embedder import build_embedder
from .fusion import confidence_from_data, fuse
from .generator import generate_for_knowledge_point, sample_style_examples, summarise_knowledge_points
from .ingest import discover_past_papers, ingest_directory, ingest_markdown
from .scorer import (
    extract_structural_signals,
    is_low_value_chunk,
    score_chunks_pedagogically,
)
from .reranker import rerank
from .schema import (
    Chunk,
    ChunkFeatures,
    ExamQuestion,
    GeneratedQuestion,
    KnowledgePointScore,
    PredictionReport,
)


log = logging.getLogger(__name__)
console = Console()


class ExamPredictor:
    """Top-level orchestrator. Stateless across runs."""

    def __init__(self, config: dict):
        self.config = config
        self.embedder = build_embedder(config)
        llm_cfg = config["llm"]
        self.llm = OpenAI(
            api_key=llm_cfg["api_key"],
            base_url=llm_cfg.get("base_url"),
            timeout=llm_cfg.get("timeout", 60),
            max_retries=llm_cfg.get("max_retries", 3),
        )
        self.llm_model = llm_cfg["model"]

    # ---------- public API ----------
    @classmethod
    def from_config_file(cls, path: str | Path) -> "ExamPredictor":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    def predict(
        self,
        course_dir: str | Path,
        course_name: Optional[str] = None,
        course_context: Optional[str] = None,
    ) -> PredictionReport:
        """Run the full pipeline on one course directory."""
        course_dir = Path(course_dir)
        course_name = course_name or course_dir.name
        console.rule(f"[bold cyan]Predicting: {course_name}")

        # ---- Stage 1: Ingest ----
        console.print("[bold]1. Ingesting course materials...[/bold]")
        slide_units = ingest_directory(course_dir, "slides")
        tutorial_units = ingest_directory(course_dir, "tutorials")
        past_records = discover_past_papers(course_dir)
        syllabus_text: Optional[str] = None
        syllabus_path = course_dir / "syllabus.md"
        if syllabus_path.exists():
            syllabus_text = syllabus_path.read_text(encoding="utf-8")
        console.print(f"   slides units: {len(slide_units)}, "
                      f"tutorial units: {len(tutorial_units)}, "
                      f"past records: {len(past_records)}, "
                      f"syllabus: {'yes' if syllabus_text else 'no'}")

        if not slide_units:
            raise ValueError(f"No course materials found in {course_dir}/slides")

        # ---- Chunk ----
        chunk_cfg = self.config.get("chunking", {})
        chunks = chunk_units(
            slide_units,
            chunk_size=chunk_cfg.get("chunk_size", 800),
            overlap=chunk_cfg.get("chunk_overlap", 100),
        )
        tutorial_texts = [t[0] for t in tutorial_units]
        past_questions = questions_from_records(past_records)
        console.print(f"   chunks: {len(chunks)}, past questions: {len(past_questions)}")

        warnings: list[str] = []
        if len(past_questions) < 3:
            warnings.append(
                f"Only {len(past_questions)} past questions available — "
                "predictions will have very wide uncertainty."
            )

        # ---- Stage 2: Align (Route C) ----
        console.print("[bold]2. Aligning past questions to chunks...[/bold]")
        align_cfg = self.config.get("alignment", {})
        alignment = align_questions_to_chunks(
            past_questions,
            chunks,
            self.embedder,
            top_k=align_cfg.get("top_k_chunks_per_question", 5),
            similarity_threshold=align_cfg.get("similarity_threshold", 0.55),
        )

        chunk_by_id = {c.id: c for c in chunks}

        # ---- Stage 2.5: Pedagogical scoring (cross-course signals) ----
        scorer_cfg    = self.config.get("scorer", {})
        ctx           = course_context or scorer_cfg.get("course_context", "") or course_name or ""
        max_to_score  = scorer_cfg.get("max_chunks_to_score", 80)
        score_batch   = scorer_cfg.get("batch_size", 5)

        # Structural signals — fast, no API
        structural_scores = {c.id: extract_structural_signals(c) for c in chunks}

        # LLM pedagogy scores — batched API calls (skippable when disabled)
        llm_scores: dict[str, float] = {}
        if scorer_cfg.get("enable_llm_scoring", True):
            console.print("[bold]2.5. Scoring chunks with LLM pedagogy prior...[/bold]")
            llm_scores = score_chunks_pedagogically(
                chunks,
                client=self.llm,
                model=self.llm_model,
                course_context=ctx,
                batch_size=score_batch,
                max_chunks=max_to_score,
            )
            console.print(f"   scored {len(llm_scores)} chunks")

        # ---- Stage 2.6: Filter out non-knowledge chunks ----
        # Titles, agendas, "review of last week", references, admin text are NOT
        # knowledge points — drop them BEFORE clustering so they never surface.
        console.print("[bold]2.6. Filtering non-knowledge content...[/bold]")
        min_chars = scorer_cfg.get("min_chunk_chars", 60)
        min_ped   = scorer_cfg.get("min_pedagogy_score", 0.30)
        teaching_chunks: list[Chunk] = []
        drop_counts: dict[str, int] = {}
        for c in chunks:
            low, reason = is_low_value_chunk(c, min_chars=min_chars)
            if low:
                drop_counts[reason] = drop_counts.get(reason, 0) + 1
                continue
            # LLM semantic gate (only when scoring enabled; neutral 0.5 default never trips)
            if llm_scores and llm_scores.get(c.id, 0.5) < min_ped:
                drop_counts["low_pedagogy"] = drop_counts.get("low_pedagogy", 0) + 1
                continue
            teaching_chunks.append(c)

        # Safety fallback: if filtering is too aggressive, keep everything
        if len(teaching_chunks) < max(3, int(len(chunks) * 0.15)):
            console.print(
                f"   [yellow]filter kept only {len(teaching_chunks)}/{len(chunks)} "
                f"chunks — too aggressive, reverting to no filter[/yellow]"
            )
            teaching_chunks = chunks
            drop_counts = {}
        elif drop_counts:
            dropped = sum(drop_counts.values())
            detail = ", ".join(f"{k}:{v}" for k, v in drop_counts.items())
            console.print(
                f"   dropped {dropped} non-knowledge chunks ({detail}); "
                f"{len(teaching_chunks)} remain"
            )

        # ---- Cluster ONLY the teaching chunks into knowledge points ----
        kps = aggregate_chunks_to_knowledge_points(
            teaching_chunks, alignment.chunk_heat, self.embedder,
        )
        console.print(f"   knowledge points found: {len(kps)}")

        # ---- Stage 3: Fuse (Route D) ----
        console.print("[bold]3. Fusing multi-signal features...[/bold]")
        features = fuse(
            chunks=teaching_chunks,
            alignment=alignment,
            config=self.config,
            n_past_papers=len(past_questions),
            tutorial_items=tutorial_texts,
            syllabus_text=syllabus_text,
            embedder=self.embedder,
            llm_scores=llm_scores or None,
            structural_scores=structural_scores,
        )

        # ---- Aggregate to knowledge-point level scores ----
        scored_kps: list[KnowledgePointScore] = []
        for kp in kps:
            member_features = [features[cid] for cid in kp["chunk_ids"] if cid in features]
            if not member_features:
                continue
            # average features across member chunks
            n = len(member_features)
            def _avg(attr): return float(sum(getattr(f, attr) for f in member_features) / n)
            avg = ChunkFeatures(
                chunk_id=kp["id"],
                exam_frequency=_avg("exam_frequency"),
                explicit_emphasis=_avg("explicit_emphasis"),
                chunk_size=_avg("chunk_size"),
                tutorial_overlap=_avg("tutorial_overlap"),
                syllabus_emphasis=_avg("syllabus_emphasis"),
                llm_pedagogy_score=_avg("llm_pedagogy_score"),
                structural_signals=_avg("structural_signals"),
            )
            final_score = float(sum(f.evidence.get("final_score", 0.0) for f in member_features) / n)
            regime = member_features[0].evidence.get("weight_regime", "normal")
            avg.evidence = {
                "final_score": final_score,
                "n_chunks": n,
                "weight_regime": regime,
                "total_questions_matched": sum(
                    alignment.raw_counts.get(cid, 0) for cid in kp["chunk_ids"]
                ),
            }
            rep_text = chunk_by_id[kp["chunk_ids"][0]].text[:200]
            scored_kps.append(KnowledgePointScore(
                knowledge_point_id=kp["id"],
                title=kp["title"],
                chunk_ids=kp["chunk_ids"],
                score=final_score,
                confidence=confidence_from_data(len(past_questions), len(chunks)),
                features=avg,
                representative_text=rep_text,
            ))

        scored_kps.sort(key=lambda k: -k.score)

        # ---- Stage 3.5: Summarise knowledge points ----
        console.print("[bold]3.5. Summarising knowledge points with LLM...[/bold]")
        gen_cfg = self.config.get("generation", {})
        summarise_knowledge_points(
            self.llm, self.llm_model,
            kps=scored_kps[: gen_cfg.get("top_k_knowledge_points", 10)],
            chunk_by_id=chunk_by_id,
            course_context=course_context or ctx,
            batch_size=gen_cfg.get("summarise_batch_size", 4),
        )

        # ---- Stage 4: Generate (Route E) ----
        console.print("[bold]4. Generating candidate questions...[/bold]")
        top_kps = scored_kps[: gen_cfg.get("top_k_knowledge_points", 10)]
        n_candidates = gen_cfg.get("candidates_per_point", 5)
        n_few_shot = gen_cfg.get("few_shot_examples", 3)
        keep_top = gen_cfg.get("rerank_keep_top_n", 2)

        all_questions: list[GeneratedQuestion] = []
        for kp in track(top_kps, description="generating"):
            style_examples = sample_style_examples(past_questions, n=n_few_shot, seed=hash(kp.knowledge_point_id) & 0xFFFFFFFF)
            kp_chunks = [chunk_by_id[cid] for cid in kp.chunk_ids if cid in chunk_by_id]
            # attach kp_id so downstream knows the owner
            for c in kp_chunks:
                c.metadata["kp_id"] = kp.knowledge_point_id

            candidates = generate_for_knowledge_point(
                self.llm, self.llm_model,
                kp_title=kp.title,
                kp_chunks=kp_chunks,
                style_examples=style_examples,
                n_candidates=n_candidates,
                temperature=self.config["llm"].get("temperature", 0.5),
                max_tokens=self.config["llm"].get("max_tokens", 2048),
            )
            if not candidates:
                continue
            top = rerank(
                self.llm, self.llm_model,
                candidates=candidates,
                past_questions=past_questions,
                embedder=self.embedder,
                keep_top_n=keep_top,
            )
            all_questions.extend(top)

        # ---- Build report ----
        overall_conf = confidence_from_data(len(past_questions), len(chunks))
        report = PredictionReport(
            course_name=course_name,
            n_chunks=len(chunks),
            n_past_questions=len(past_questions),
            predictions=scored_kps[:30],
            generated_questions=all_questions,
            overall_confidence=overall_conf,
            warnings=warnings,
        )
        console.rule(f"[bold green]Done. Overall confidence: {overall_conf:.2f}")
        return report

    # ---------- output helpers ----------
    def save_report(self, report: PredictionReport, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON: full structured output (for programmatic use)
        (out_dir / "predictions.json").write_text(
            json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Markdown: student-readable report
        md = self._format_markdown_report(report)
        (out_dir / "report.md").write_text(md, encoding="utf-8")

        console.print(f"[green]Saved:[/green] {out_dir / 'predictions.json'}")
        console.print(f"[green]Saved:[/green] {out_dir / 'report.md'}")

        # PDF: print-ready report (optional — skip gracefully if reportlab missing)
        try:
            from .exporter import report_to_pdf_bytes
            (out_dir / "report.pdf").write_bytes(report_to_pdf_bytes(report))
            console.print(f"[green]Saved:[/green] {out_dir / 'report.pdf'}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]PDF export skipped: {exc}[/yellow]")

    @staticmethod
    def _format_txt_report(r: PredictionReport) -> str:
        sep  = "=" * 60
        thin = "-" * 60

        lines = [
            sep,
            f"考试预测报告: {r.course_name}",
            sep,
            f"课件 chunks  : {r.n_chunks}",
            f"往年真题     : {r.n_past_questions}",
            f"综合置信度   : {r.overall_confidence:.2f}",
            "",
        ]

        if r.warnings:
            lines += ["[警告]"]
            lines += [f"  ! {w}" for w in r.warnings]
            lines += [""]

        lines += [thin, "知识点优先级 (Top 30)", thin, ""]
        for i, p in enumerate(r.predictions, 1):
            f = p.features
            title = p.title[:80].strip()
            lines += [
                f"{i:2d}. {title}",
                f"    综合得分: {p.score:.3f}  置信度: {p.confidence:.2f}  命中真题: {f.evidence.get('total_questions_matched', 0)}",
                f"    考频: {f.exam_frequency:.2f}  显式强调: {f.explicit_emphasis:.2f}  "
                f"教学量: {f.chunk_size:.2f}  tutorial重叠: {f.tutorial_overlap:.2f}  大纲: {f.syllabus_emphasis:.2f}",
                "",
            ]

        lines += [thin, "生成的练习题", thin, ""]

        by_kp: dict[str, list[GeneratedQuestion]] = {}
        for q in r.generated_questions:
            by_kp.setdefault(q.knowledge_point_id, []).append(q)
        kp_titles = {p.knowledge_point_id: p.title for p in r.predictions}

        for kp_id, qs in by_kp.items():
            title = kp_titles.get(kp_id, kp_id)[:80].strip()
            lines += [f"[ {title} ]", ""]
            for idx, q in enumerate(qs, 1):
                diff_str  = f"{q.estimated_difficulty:.2f}" if q.estimated_difficulty is not None else "?"
                score_str = f"{q.overall_score:.2f}"        if q.overall_score         is not None else "?"
                lines += [
                    f"Q{idx} (难度: {diff_str}  得分: {score_str})",
                    q.text,
                    "",
                ]
                if q.answer_sketch:
                    lines += ["参考答案:", q.answer_sketch, ""]
            lines += [thin, ""]

        return "\n".join(lines)

    # ── Markdown report helpers ──────────────────────────────────────────

    @staticmethod
    def _clean_kp_title(raw: str, max_len: int = 60) -> str:
        """Strip Markdown header markers; return only the clean heading text.

        Input examples (first 80 chars of a slide chunk):
          '## Section 1: Power Flow Analysis  Power flow analysis ...'
          '# Sample Power Systems lecture notes (demo data)'
        """
        t = re.sub(r"^#+\s*", "", raw.strip())
        # double-space = boundary between heading and body snippet → keep heading only
        dbl = t.find("  ")
        if dbl > 5:
            t = t[:dbl]
        # strip trailing parenthetical annotations like "(demo data)"
        t = re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()
        if len(t) > max_len:
            t = t[:max_len].rsplit(" ", 1)[0] + "…"
        return t

    @staticmethod
    def _heat_bar(score: float, width: int = 8) -> str:
        """Return a Unicode block progress bar, e.g. '█████░░░'."""
        filled = round(max(0.0, min(score, 1.0)) * width)
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def _diff_stars(diff: float | None, total: int = 5) -> str:
        """Convert 0-1 difficulty to a star string, e.g. '★★★☆☆'."""
        if diff is None:
            return "☆☆☆☆☆"
        filled = round(max(0.0, min(diff, 1.0)) * total)
        return "★" * filled + "☆" * (total - filled)

    @staticmethod
    def _diff_label(diff: float | None) -> str:
        if diff is None:
            return ""
        if diff < 0.35:
            return "基础"
        if diff < 0.55:
            return "中等"
        if diff < 0.75:
            return "进阶"
        return "挑战"

    @staticmethod
    def _format_markdown_report(r: PredictionReport) -> str:
        """Student-friendly integrated report:
        - Overview table with description snippets
        - One section per knowledge point: description → exam directions → questions → answers
        - Answers as Markdown blockquotes (renders in every viewer, PDF-friendly)
        """
        _bar   = ExamPredictor._heat_bar
        _stars = ExamPredictor._diff_stars
        _label = ExamPredictor._diff_label

        lines: list[str] = []

        # ── 封面 ────────────────────────────────────────────────────────
        lines += [
            "# 🎓 考试重点预测报告",
            f"## {r.course_name}",
            "",
            "| | |",
            "|---|---|",
            f"| 📄 分析课件块数 | **{r.n_chunks}** |",
            f"| 📚 参考历年真题 | **{r.n_past_questions}** 道 |",
            f"| 🎯 预测置信度   | **{r.overall_confidence:.0%}** |",
            "",
        ]
        if r.warnings:
            for w in r.warnings:
                lines += [f"> ⚠️ {w}"]
            lines += [""]
        lines += ["---", ""]

        # ── 一、总览表 ────────────────────────────────────────────────────
        lines += [
            "## 一、核心知识点总览",
            "",
            "| 排名 | 知识点 | 重要度 | 概述 |",
            "|:---:|--------|:------:|------|",
        ]
        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        for i, p in enumerate(r.predictions, 1):
            m    = medals.get(i, str(i))
            bar  = _bar(p.score)
            hits = p.features.evidence.get("total_questions_matched", 0)
            # description snippet or fallback
            desc = p.description or ""
            snippet = (desc[:55] + "…") if len(desc) > 55 else desc
            lines.append(
                f"| {m} | **{p.title}** | `{bar}` {p.score:.2f} · {hits}题 | {snippet} |"
            )
        lines += ["", "---", ""]

        # ── 二、知识点详解与押题 ───────────────────────────────────────────
        lines += [
            "## 二、知识点详解与押题",
            "",
            "每节依次包含：概念说明 → 常考方向 → 练习题（附参考答案）。",
            "",
        ]

        by_kp: dict[str, list[GeneratedQuestion]] = {}
        for q in r.generated_questions:
            by_kp.setdefault(q.knowledge_point_id, []).append(q)

        kp_rank_map = {p.knowledge_point_id: i for i, p in enumerate(r.predictions, 1)}
        MEDAL_HDR   = {1: "🥇 第一考点", 2: "🥈 第二考点", 3: "🥉 第三考点"}
        CN          = "一二三四五六七八九十"

        for i, p in enumerate(r.predictions, 1):
            qs   = by_kp.get(p.knowledge_point_id, [])
            cn   = CN[i - 1] if i <= len(CN) else str(i)
            hdr  = MEDAL_HDR.get(i, f"第{cn}考点")
            hits = p.features.evidence.get("total_questions_matched", 0)

            # ── 节标题 ──────────────────────────────────────────────────
            lines += [
                f"### {hdr}：{p.title}",
                "",
                f"**重要度** `{_bar(p.score)}` {p.score:.2f} &emsp; "
                f"**历年命中** {hits} 题 &emsp; "
                f"**权重模式** {'少样本' if p.features.evidence.get('weight_regime') == 'sparse' else '正常'}",
                "",
            ]

            # ── 概念描述 ─────────────────────────────────────────────────
            if p.description:
                lines += [
                    "**📖 概念说明**",
                    "",
                    p.description,
                    "",
                ]

            # ── 常考方向 ─────────────────────────────────────────────────
            if p.exam_directions:
                lines += ["**📌 常考方向**", ""]
                for j, d in enumerate(p.exam_directions, 1):
                    lines.append(f"{j}. {d}")
                lines += [""]

            # ── 练习题 ───────────────────────────────────────────────────
            if qs:
                lines += ["**🖊️ 练习题**", ""]
                for q_i, q in enumerate(qs, 1):
                    d      = q.estimated_difficulty
                    stars  = _stars(d)
                    label  = _label(d)
                    qtype  = f"  ·  {q.question_type}" if q.question_type else ""

                    lines += [
                        f"**第 {q_i} 题**　{stars}　{label}{qtype}",
                        "",
                        q.text,
                        "",
                    ]
                    if q.answer_sketch:
                        lines += ["**💡 参考答案要点**", ""]
                        # blockquote — renders in all Markdown viewers and PDF
                        for al in q.answer_sketch.strip().splitlines():
                            lines.append(f"> {al}" if al.strip() else ">")
                        lines += [""]
            else:
                lines += ["*（本知识点暂无生成练习题）*", ""]

            lines += ["---", ""]

        lines += [
            "*📌 本报告由 押题宝 Exam Predictor 自动生成，仅供参考，请以课堂教材及教师说明为准。*",
        ]
        return "\n".join(lines)
