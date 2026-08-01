"""ExamSage local web application.

Run with: streamlit run app.py
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import streamlit as st

from exam_predictor.agent import ExamSageAgent
from exam_predictor.budget import estimate_run_cost
from exam_predictor.exporter import report_to_pdf_bytes
from exam_predictor.legacy_intake import (
    LegacyIntakeError,
    acquire_legacy_intake_lease,
    cleanup_legacy_intake,
)
from exam_predictor.pipeline import ExamPredictor
from exam_predictor.providers import BudgetExceeded, ProviderError, create_provider
from exam_predictor.schema import KnowledgeTreeNode, PredictionReport
from exam_predictor.security import (
    UploadSecurityError,
    redact_secrets,
    safe_filename,
    validate_public_https_url,
)
from exam_predictor.state import CourseStore


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("EXAMSAGE_DATA_DIR", Path.home() / ".examsage"))
INTAKE_DIR = DATA_DIR / "intake"
COURSE_STORE = CourseStore(DATA_DIR)


def default_config(provider_name: str, api_key: str, approved_max: float, **advanced) -> dict:
    llm = {
        "provider": provider_name,
        "api_key": api_key,
        "approved_max_usd": approved_max,
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    for key in ("base_url", "fast_model", "balanced_model", "reasoning_model", "embedding_model"):
        if advanced.get(key):
            llm[key] = advanced[key]
    return {
        "llm": llm,
        "embedding": {"backend": "provider", "batch_size": 96},
        "chunking": {"chunk_size": 900, "chunk_overlap": 120},
        "alignment": {"top_k_chunks_per_question": 5, "similarity_threshold": 0.52},
        "scorer": {
            "enable_llm_scoring": True,
            "batch_size": 5,
            "max_chunks_to_score": 120,
            "min_chunk_chars": 50,
            "min_pedagogy_score": 0.28,
        },
        "fusion": {
            "low_data_threshold": 10,
            "weights": {
                "exam_frequency": 0.35, "explicit_emphasis": 0.10,
                "chunk_size": 0.05, "tutorial_overlap": 0.15,
                "syllabus_emphasis": 0.10, "llm_pedagogy_score": 0.15,
                "structural_signals": 0.10,
            },
            "sparse_weights": {
                "exam_frequency": 0.10, "explicit_emphasis": 0.08,
                "chunk_size": 0.03, "tutorial_overlap": 0.10,
                "syllabus_emphasis": 0.14, "llm_pedagogy_score": 0.45,
                "structural_signals": 0.10,
            },
            "emphasis_keywords": {
                "zh": ["重点", "考点", "考试", "必考", "重要", "记住", "经典", "高频"],
                "en": ["important", "remember", "exam", "key", "essential", "frequently asked"],
            },
            "syllabus_verb_weights": {
                "derive": 1.0, "prove": 1.0, "compute": 0.8, "apply": 0.8,
                "explain": 0.6, "describe": 0.4, "aware": 0.2,
            },
        },
        "generation": {
            "all_knowledge_points": True,
            "top_k_knowledge_points": 12,
            "few_shot_examples": 3,
            "question_batch_size": 3,
            "summarise_batch_size": 4,
            "report_language": advanced.get("report_language", "auto"),
        },
        "research": {"max_queries": int(advanced.get("max_queries", 6))},
    }


def likelihood_band(score: float) -> str:
    if score >= 0.80:
        return "Very high"
    if score >= 0.60:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Lower"


def confidence_band(score: float) -> str:
    if score >= 0.75:
        return "Strong"
    if score >= 0.50:
        return "Moderate"
    return "Limited"


def save_uploads(uploaded_files) -> list[Path]:
    session_id = st.session_state.setdefault("intake_id", uuid.uuid4().hex)
    destination = INTAKE_DIR / session_id
    destination.mkdir(parents=True, exist_ok=True)
    if "intake_lease" not in st.session_state:
        st.session_state["intake_lease"] = acquire_legacy_intake_lease(
            DATA_DIR,
            session_id,
        )
    paths: list[Path] = []
    for index, upload in enumerate(uploaded_files):
        name = safe_filename(upload.name)
        target = destination / f"{index:03d}_{name}"
        target.write_bytes(upload.getvalue())
        paths.append(target)
    return paths


def render_tree(nodes: list[KnowledgeTreeNode], depth: int = 0) -> None:
    for node in nodes:
        label = f"{'↳ ' * depth}{node.title}"
        with st.expander(label, expanded=depth == 0):
            if node.summary:
                st.write(node.summary)
            if node.knowledge_point_ids:
                st.caption("Knowledge points: " + ", ".join(node.knowledge_point_ids))
            if node.prerequisites:
                st.caption("Prerequisites: " + ", ".join(node.prerequisites))
            if node.children:
                render_tree(node.children, depth + 1)


def report_markdown(report: PredictionReport) -> str:
    return ExamPredictor._format_markdown_report(report)


def render_report(report: PredictionReport, course_id: str | None) -> None:
    st.divider()
    st.header(report.course_name)
    a, b, c, d = st.columns(4)
    a.metric("Knowledge points", len(report.predictions))
    b.metric("Past questions found", report.n_past_questions)
    c.metric("Overall confidence", f"{report.overall_confidence:.0%}")
    d.metric("Practice questions", len(report.generated_questions))
    st.caption(
        "Likelihood is a relative study-priority score, not the probability that a question is guaranteed to appear."
    )
    for warning in dict.fromkeys(report.warnings):
        st.warning(warning)

    tabs = st.tabs([
        "Focus map", "Chapter tree", "Practice", "Study guide",
        "Web evidence", "Ask ExamSage", "Export",
    ])

    with tabs[0]:
        rows = []
        for index, point in enumerate(report.predictions, 1):
            rows.append({
                "Rank": index,
                "Knowledge point": point.title,
                "Likelihood": likelihood_band(point.score),
                "Relative score": round(point.score * 100),
                "Confidence": confidence_band(point.confidence),
                "Similar past questions": point.features.evidence.get("total_questions_matched", 0),
                "Why it matters": point.description or point.representative_text,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.info(
            "How to read this: a high score with limited confidence means ‘revise this early, but treat the ranking cautiously’."
        )

    with tabs[1]:
        if report.knowledge_tree:
            render_tree(report.knowledge_tree)
        else:
            st.info("No chapter tree is available for this saved report.")

    with tabs[2]:
        by_kp: dict[str, list] = {}
        for question in report.generated_questions:
            by_kp.setdefault(question.knowledge_point_id, []).append(question)
        for point in report.predictions:
            questions = by_kp.get(point.knowledge_point_id, [])
            if not questions:
                continue
            with st.expander(f"{point.title} · {len(questions)} questions"):
                st.caption(
                    f"Study priority: {likelihood_band(point.score)} · confidence: {confidence_band(point.confidence)}"
                )
                for index, question in enumerate(questions, 1):
                    marks = f" · {question.suggested_marks} marks" if question.suggested_marks else ""
                    st.markdown(f"#### Question {index}{marks}")
                    st.write(question.text)
                    if question.question_type:
                        st.caption(
                            f"{question.question_type} · difficulty {question.estimated_difficulty or 0.5:.0%} · {question.source_kind}"
                        )
                    with st.expander("Worked answer and marking scheme"):
                        st.markdown(question.answer_sketch or "No worked answer was returned.")
                        if question.marking_scheme:
                            st.table([
                                {
                                    "Step / knowledge point": item.criterion,
                                    "Marks": item.marks,
                                    "How marks are earned": item.explanation or "",
                                }
                                for item in question.marking_scheme
                            ])
                        if question.source_reference:
                            st.caption("Grounding/style reference: " + question.source_reference)

    with tabs[3]:
        if report.study_guide:
            st.markdown(report.study_guide)
        else:
            st.info("No study guide is available for this saved report.")

    with tabs[4]:
        if not report.web_evidence:
            st.success("Uploaded evidence was sufficient; no supplementary web research was needed.")
        for evidence in report.web_evidence:
            point = next(
                (item for item in report.predictions if item.knowledge_point_id == evidence.knowledge_point_id),
                None,
            )
            with st.expander(point.title if point else evidence.query):
                st.write(evidence.summary)
                st.caption(evidence.limitations[0] if evidence.limitations else "Supporting evidence only.")
                for citation in evidence.citations:
                    st.markdown(f"- [{citation.title}]({citation.url})")

    with tabs[5]:
        if not course_id:
            st.info("Run or load a course to continue the conversation.")
        else:
            for message in COURSE_STORE.messages(course_id):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            question = st.chat_input("Ask for an explanation, a new example, or a web-backed answer…")
            if question:
                api_key = st.session_state.get("active_api_key", "")
                provider_cfg = st.session_state.get("active_provider_config")
                if not api_key or not provider_cfg:
                    st.error("Enter your API key above to continue chatting. It is not stored in the course database.")
                else:
                    with st.chat_message("user"):
                        st.write(question)
                    try:
                        cfg = dict(provider_cfg)
                        cfg["api_key"] = api_key
                        provider = create_provider(cfg)
                        agent = ExamSageAgent(
                            provider,
                            st.session_state["active_config"],
                            DATA_DIR,
                        )
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking…"):
                                answer = agent.chat(course_id, question)
                            st.markdown(answer)
                    except Exception as exc:  # user-facing boundary; keys are never included
                        message = redact_secrets(f"{type(exc).__name__}: {exc}")
                        st.error(f"The provider could not answer: {message}")

    with tabs[6]:
        json_bytes = json.dumps(report.model_dump(), ensure_ascii=False, indent=2).encode("utf-8")
        markdown_bytes = report_markdown(report).encode("utf-8")
        st.download_button("Download JSON", json_bytes, "examsage-report.json", "application/json")
        st.download_button("Download Markdown", markdown_bytes, "examsage-report.md", "text/markdown")
        try:
            pdf_bytes = report_to_pdf_bytes(report)
            st.download_button("Download PDF", pdf_bytes, "examsage-report.pdf", "application/pdf")
        except Exception as exc:
            st.warning(f"PDF export is unavailable: {exc}")


st.set_page_config(page_title="ExamSage", page_icon="🎓", layout="wide")
if os.environ.get("EXAMSAGE_AGENT_V2", "0") == "1":
    from exam_predictor.ui.agent_view import render_agent_kernel

    render_agent_kernel()
    st.stop()

st.title("🎓 ExamSage")
st.subheader("Turn course materials into a structured, evidence-aware revision agent")
st.caption("Local app · your API key · no ExamSage server · no telemetry")

with st.sidebar:
    st.header("Saved courses")
    courses = COURSE_STORE.list_courses()
    if courses:
        selected = st.selectbox(
            "Open a previous course",
            options=[None] + [item["id"] for item in courses],
            format_func=lambda value: "Choose…" if value is None else next(
                item["name"] for item in courses if item["id"] == value
            ),
        )
        if selected and st.button("Open course", use_container_width=True):
            saved = COURSE_STORE.get_course(selected)
            st.session_state["course_id"] = selected
            st.session_state["report"] = PredictionReport.model_validate(saved["report"])
            st.rerun()
    st.divider()
    st.markdown("**Privacy controls**")
    st.caption("Keys live in this browser session only. Reports and chat are stored locally on this computer.")
    if st.session_state.get("course_id") and st.button("Delete this local course"):
        course_id = st.session_state["course_id"]
        path = COURSE_STORE.delete_course(course_id)
        if path:
            resolved_data = DATA_DIR.resolve()
            resolved_path = path.resolve()
            if resolved_path != resolved_data and resolved_data in resolved_path.parents:
                shutil.rmtree(resolved_path, ignore_errors=True)
        st.session_state.pop("course_id", None)
        st.session_state.pop("report", None)
        st.rerun()

st.markdown("### 1. Connect one AI provider")
provider_label = st.radio(
    "Provider",
    ["OpenAI", "Google Gemini", "Custom OpenAI-compatible (experimental)"],
    horizontal=True,
)
provider_name = {
    "OpenAI": "openai",
    "Google Gemini": "gemini",
    "Custom OpenAI-compatible (experimental)": "custom",
}[provider_label]
api_key = st.text_input(
    "API key",
    type="password",
    help="Used directly from this local app to the selected provider. ExamSage does not log or store it.",
)
st.session_state["active_api_key"] = api_key

advanced_values = {}
with st.expander("Advanced provider settings"):
    if provider_name == "custom":
        st.warning(
            "Custom endpoints vary. Native file vision, cited web search and no-retention controls are not guaranteed; "
            "the full agent workflow therefore requires OpenAI or Gemini."
        )
        advanced_values["base_url"] = st.text_input("Base URL", placeholder="https://provider.example/v1")
    advanced_values["fast_model"] = st.text_input("Fast model override (optional)")
    advanced_values["balanced_model"] = st.text_input("Balanced model override (optional)")
    advanced_values["reasoning_model"] = st.text_input("Reasoning model override (optional)")
    advanced_values["embedding_model"] = st.text_input("Embedding model override (optional)")
    advanced_values["report_language"] = st.selectbox(
        "Report language", ["auto", "en", "zh"], format_func=lambda x: {
            "auto": "Match source material", "en": "English", "zh": "Simplified Chinese"
        }[x],
    )
    advanced_values["max_queries"] = st.slider("Maximum web research queries", 0, 12, 6)

# Keep enough non-secret connection metadata to chat with newly opened saved courses.
st.session_state["active_provider_config"] = {
    "provider": provider_name,
    "base_url": advanced_values.get("base_url", ""),
    "fast_model": advanced_values.get("fast_model", ""),
    "balanced_model": advanced_values.get("balanced_model", ""),
    "reasoning_model": advanced_values.get("reasoning_model", ""),
    "embedding_model": advanced_values.get("embedding_model", ""),
    "approved_max_usd": 25.0,
}
st.session_state["active_config"] = default_config(
    provider_name, api_key, 25.0, **advanced_values
)

st.markdown("### 2. Upload a course and describe what you need")
uploads = st.file_uploader(
    "Course files",
    type=[
        "pdf", "ppt", "pptx", "doc", "docx", "xls", "xlsx", "csv", "tsv",
        "png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff",
        "md", "markdown", "txt", "html", "htm", "json", "yaml", "yml", "zip",
    ],
    accept_multiple_files=True,
    help="Up to 1 GB per course. Audio and video are intentionally not supported yet.",
)
source_urls_text = st.text_area(
    "Public HTTPS course webpages (optional, one per line)",
    placeholder="https://university.example.edu/course/syllabus",
    height=80,
)
source_urls = [line.strip() for line in source_urls_text.splitlines() if line.strip()]
request = st.text_area(
    "Your goal",
    placeholder=(
        "Example: I have a final in three weeks. Build the chapter tree, prioritize likely calculation and proof "
        "questions, and give me a detailed practice plan."
    ),
    height=110,
)
course_name = st.text_input("Course name (optional — ExamSage can infer it)")

if st.button(
    "Estimate cost",
    type="primary",
    disabled=(not uploads and not source_urls) or not request.strip(),
):
    try:
        source_urls = [validate_public_https_url(url) for url in source_urls]
        paths = save_uploads(uploads)
        estimate = estimate_run_cost(
            provider_name,
            paths,
            knowledge_points=12,
            practice_questions_per_point=12,
            web_queries=int(advanced_values["max_queries"]) + len(source_urls),
        )
        st.session_state["upload_paths"] = [str(path) for path in paths]
        st.session_state["estimate"] = estimate
        st.session_state["estimate_request"] = request
        st.session_state["estimate_urls"] = source_urls
        st.session_state["estimate_provider"] = provider_name
        st.session_state["estimate_uploads"] = [
            (upload.name, upload.size) for upload in uploads
        ]
    except Exception as exc:
        st.error(f"Could not prepare the estimate: {exc}")

estimate = st.session_state.get("estimate")
if estimate:
    st.markdown("### 3. Review the estimate and approve a limit")
    st.metric("Estimated provider cost", f"${estimate.estimated_min:.2f} – ${estimate.estimated_max:.2f} USD")
    st.table([
        {
            "Planned work": item.label,
            "Estimated range": f"${item.estimated_min:.2f} – ${item.estimated_max:.2f}",
            "Assumptions": "; ".join(item.assumptions),
        }
        for item in estimate.breakdown
    ])
    for assumption in estimate.assumptions:
        st.caption("• " + assumption)
    approved_max = st.number_input(
        "Hard run limit (USD)",
        min_value=0.01,
        value=max(0.01, float(estimate.estimated_max)),
        step=0.25,
    )
    confirmed = st.checkbox(
        "I understand this is an estimate and authorize ExamSage to start, up to the limit above."
    )

    if st.button("Build my ExamSage agent", type="primary", disabled=not confirmed):
        if not api_key.strip():
            st.error("Enter the selected provider's API key first.")
        elif provider_name == "custom":
            st.error("The full multimodal + web-search build currently requires OpenAI or Google Gemini.")
        elif provider_name != st.session_state.get("estimate_provider"):
            st.error("Your provider changed after the estimate. Estimate again before starting.")
        elif request != st.session_state.get("estimate_request"):
            st.error("Your request changed after the estimate. Estimate again before starting.")
        elif source_urls != st.session_state.get("estimate_urls", []):
            st.error("Your webpage list changed after the estimate. Estimate again before starting.")
        elif [(upload.name, upload.size) for upload in uploads] != st.session_state.get("estimate_uploads", []):
            st.error("Your uploaded files changed after the estimate. Estimate again before starting.")
        else:
            try:
                config = default_config(
                    provider_name,
                    api_key,
                    float(approved_max),
                    **advanced_values,
                )
                provider = create_provider(config["llm"])
                agent = ExamSageAgent(provider, config, DATA_DIR)
                st.session_state["active_config"] = config
                st.session_state["active_provider_config"] = {
                    key: value for key, value in config["llm"].items() if key != "api_key"
                }
                with st.status("Building your course agent…", expanded=True) as status:
                    st.write("Validating files and sending them directly to your selected provider…")
                    st.write("Reading text, scans, handwriting, formulas, tables and embedded visuals…")
                    course_id, report, normalization = agent.build_course(
                        st.session_state["upload_paths"],
                        request,
                        course_name=course_name.strip() or None,
                        source_urls=source_urls,
                    )
                    st.write("Building the chapter tree, focus ranking, practice set and study guide…")
                    if report.web_evidence:
                        st.write("Course evidence was sparse, so cited web research was added…")
                    status.update(label="Your ExamSage agent is ready", state="complete")
                st.session_state["course_id"] = course_id
                st.session_state["report"] = report
                intake_session_id = st.session_state.get("intake_id", "")
                intake_lease = st.session_state.pop("intake_lease", None)
                if intake_lease is not None:
                    intake_lease.close()
                try:
                    cleanup_legacy_intake(
                        DATA_DIR,
                        session_ids=(intake_session_id,),
                    )
                except LegacyIntakeError:
                    pass
            except (BudgetExceeded, UploadSecurityError, ProviderError, ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                message = redact_secrets(f"{type(exc).__name__}: {exc}")
                st.error(f"The build stopped safely: {message}")

if st.session_state.get("report"):
    render_report(st.session_state["report"], st.session_state.get("course_id"))

st.divider()
st.caption(
    "ExamSage predicts study priorities, not actual exam questions. Always follow your instructor, syllabus and academic-integrity rules."
)
