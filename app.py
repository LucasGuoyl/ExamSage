"""ExamSage — Streamlit Web UI

Run with:
    streamlit run app.py

The first run downloads the BGE embedding model (~1.3 GB); later runs reuse it.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import streamlit as st

# ── Make the exam_predictor package importable ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Page config (must precede every other st call) ───────────────────────────
st.set_page_config(
    page_title="ExamSage · Exam Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.stTabs [role="tab"]   { font-size: 1rem; padding: .4rem 1.2rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_uploads(files, dest: Path) -> None:
    """Write Streamlit uploaded-file objects to disk."""
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        (dest / f.name).write_bytes(f.getvalue())


def build_config(
    api_key: str,
    base_url: str,
    model: str,
    course_context: str,
    top_k: int,
    n_candidates: int,
    keep_top: int,
    enable_llm_scoring: bool,
    filter_noise: bool = True,
    report_language: str = "auto",
) -> dict:
    """Build a pipeline config dict from UI inputs (no config.yaml needed)."""
    return {
        "llm": {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "temperature": 0.3,
            "max_tokens": 2048,
            "timeout": 60,
            "max_retries": 2,
        },
        "embedding": {
            "backend": "local",
            "model_name": "BAAI/bge-large-zh-v1.5",
            "device": "cpu",
            "batch_size": 32,
        },
        "scorer": {
            "enable_llm_scoring": enable_llm_scoring,
            "course_context": course_context,
            "batch_size": 5,
            "max_chunks_to_score": 80,
            # Smart filter: drop title/agenda/background/admin content when on
            "min_chunk_chars": 60 if filter_noise else 0,
            "min_pedagogy_score": 0.30 if filter_noise else 0.0,
        },
        "chunking": {"chunk_size": 800, "chunk_overlap": 100},
        "alignment": {"top_k_chunks_per_question": 5, "similarity_threshold": 0.55},
        "fusion": {
            "low_data_threshold": 10,
            "weights": {
                "exam_frequency": 0.35, "explicit_emphasis": 0.10,
                "chunk_size": 0.05,    "tutorial_overlap": 0.15,
                "syllabus_emphasis": 0.10, "llm_pedagogy_score": 0.15,
                "structural_signals": 0.10,
            },
            "sparse_weights": {
                "exam_frequency": 0.10, "explicit_emphasis": 0.08,
                "chunk_size": 0.03,    "tutorial_overlap": 0.10,
                "syllabus_emphasis": 0.14, "llm_pedagogy_score": 0.45,
                "structural_signals": 0.10,
            },
            "emphasis_keywords": {
                "zh": ["重点", "考点", "考试", "必考", "重要", "记住", "经典", "高频"],
                "en": ["important", "remember", "exam", "key", "essential"],
            },
            "syllabus_verb_weights": {
                "derive": 1.0, "prove": 1.0, "compute": 0.8, "apply": 0.8,
                "explain": 0.6, "describe": 0.4, "aware": 0.2,
            },
        },
        "generation": {
            "top_k_knowledge_points": top_k,
            "candidates_per_point": n_candidates,
            "few_shot_examples": 3,
            "rerank_keep_top_n": keep_top,
            "report_language": report_language,
        },
    }


def heat_bar(v: float, w: int = 8) -> str:
    n = round(max(0.0, min(v, 1.0)) * w)
    return "█" * n + "░" * (w - n)


def diff_stars(d: float | None) -> str:
    if d is None:
        return "☆☆☆☆☆"
    n = round(max(0.0, min(d, 1.0)) * 5)
    return "★" * n + "☆" * (5 - n)


def diff_label(d: float | None) -> str:
    if d is None:     return ""
    if d < 0.35:      return "Basic"
    if d < 0.55:      return "Medium"
    if d < 0.75:      return "Advanced"
    return "Challenge"


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar: configuration
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ Settings")

    # —— API
    st.subheader("🔑 API")
    api_key = st.text_input(
        "API Key *",
        type="password",
        placeholder="sk-...",
        help="Works with DeepSeek / OpenAI / any OpenAI-compatible service",
    )
    base_url = st.text_input(
        "API Base URL",
        value="https://api.deepseek.com/v1",
        help="DeepSeek: https://api.deepseek.com/v1\nOpenAI: https://api.openai.com/v1",
    )
    llm_model = st.text_input("Model name", value="deepseek-chat")

    st.divider()

    # —— Course info
    st.subheader("📖 Course")
    course_name = st.text_input(
        "Course name *",
        placeholder="e.g. Power Systems Analysis",
    )
    course_ctx = st.text_area(
        "Course description (optional)",
        placeholder="e.g. Undergraduate power systems course; emphasis on "
                    "Newton-Raphson power flow and economic dispatch derivations",
        height=90,
        help="A specific description improves the LLM's pedagogical scoring and cross-course accuracy",
    )

    st.divider()

    # —— Advanced
    st.subheader("🔧 Advanced")
    top_k        = st.slider("Top-K knowledge points", 3, 20, 10)
    n_candidates = st.slider("Candidates / point",      2,  8,  5)
    keep_top_n   = st.slider("Keep / point",            1,  4,  2)

    lang_choice = st.selectbox(
        "Report language",
        options=["Auto (match materials)", "English", "Chinese"],
        index=0,
        help="Auto generates questions/answers in the source material's language. "
             "Force English/Chinese to override.",
    )
    report_language = {"Auto (match materials)": "auto",
                       "English": "en", "Chinese": "zh"}[lang_choice]

    enable_llm_scoring = st.checkbox(
        "Enable LLM pedagogy scoring",
        value=True,
        help="Greatly improves accuracy on few-shot / new courses; uses a little extra API quota",
    )
    filter_noise = st.checkbox(
        "Smart-filter non-exam content",
        value=True,
        help="Drops title pages, agendas, recaps, background, and admin text so "
             "they aren't mistaken for knowledge points",
    )

    st.divider()

    # Run button pinned to the bottom of the sidebar, always visible
    _can_run_sidebar = bool(api_key and course_name)
    if not _can_run_sidebar:
        _miss = [l for l, ok in [("API Key", api_key), ("Course name", course_name)] if not ok]
        st.caption(f"⚠️ Still needed: {', '.join(_miss)}")

    run_sidebar_btn = st.button(
        "🚀  Run Analysis",
        type="primary",
        disabled=not _can_run_sidebar,
        use_container_width=True,
        key="run_sidebar",
    )

    st.divider()
    st.caption("ExamSage v0.2 · [GitHub](https://github.com/LucasGuoyl/ExamSage)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main area
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🎓 ExamSage · Exam Predictor")
st.caption("Upload slides and past papers → AI ranks exam topics → auto-generates practice questions")

# ── Upload area ──────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.subheader("📄 Course materials (required)")
    slides_files = st.file_uploader(
        "Supports **PDF / PPTX / Markdown**, multiple files allowed",
        type=["pdf", "pptx", "md"],
        accept_multiple_files=True,
        key="slides",
    )
    if slides_files:
        total_kb = sum(f.size for f in slides_files) // 1024
        st.success(f"✅ {len(slides_files)} file(s) selected · {total_kb} KB total")
        with st.expander(f"View file list ({len(slides_files)})"):
            for f in slides_files:
                st.caption(f"· `{f.name}`  {f.size // 1024} KB")

with col_r:
    st.subheader("📚 Past papers (optional — more is better)")
    papers_files = st.file_uploader(
        "Supports **JSON / PDF / Markdown**, multiple files allowed",
        type=["json", "pdf", "md"],
        accept_multiple_files=True,
        key="papers",
    )
    if papers_files:
        st.success(f"✅ {len(papers_files)} file(s) selected")
        with st.expander(f"View file list ({len(papers_files)})"):
            for f in papers_files:
                st.caption(f"· `{f.name}`")

    # JSON template download
    TEMPLATE = [
        {
            "id": "2023_q1",
            "year": 2023,
            "text": "Put the full question text here (required)",
            "type": "computation",
            "answer": "Reference answer key points (optional)",
        },
        {
            "id": "2022_q1",
            "year": 2022,
            "text": "Put the full question text here",
            "type": "derivation",
        },
    ]
    st.download_button(
        "📥 Download past-papers JSON template",
        data=json.dumps(TEMPLATE, ensure_ascii=False, indent=2),
        file_name="questions_template.json",
        mime="application/json",
        use_container_width=True,
    )

# Optional materials
with st.expander("➕ Tutorials & syllabus (optional — improves accuracy)"):
    c1, c2 = st.columns(2)
    with c1:
        st.caption("**Tutorials / problem sets** (PDF or Markdown)")
        tut_files = st.file_uploader(
            "Tutorials",
            type=["pdf", "md"],
            accept_multiple_files=True,
            key="tutorials",
            label_visibility="collapsed",
        )
    with c2:
        st.caption("**Syllabus** (.md or .txt)")
        syl_file = st.file_uploader(
            "Syllabus",
            type=["md", "txt"],
            key="syllabus",
            label_visibility="collapsed",
        )

st.divider()

# ── Run button & validation ──────────────────────────────────────────────────
can_run = bool(api_key and slides_files and course_name)

if not can_run:
    missing = [label for label, ok in [
        ("API Key (sidebar)", api_key),
        ("Course name (sidebar)", course_name),
        ("Course materials",     slides_files),
    ] if not ok]
    st.warning("⚠️ Still need: " + ", ".join(missing))

# Main-area button (page center)
run_btn = st.button(
    "🚀  Run Analysis",
    type="primary",
    disabled=not can_run,
    use_container_width=True,
    key="run_main",
)

# Either button can trigger the run
run_btn = run_btn or (run_sidebar_btn and can_run)

# ═══════════════════════════════════════════════════════════════════════════════
# Run the pipeline
# ═══════════════════════════════════════════════════════════════════════════════

if run_btn:
    # Clear previous results
    for key in ("report", "md", "pdf", "run_course"):
        st.session_state.pop(key, None)

    from exam_predictor.pipeline import ExamPredictor

    cfg = build_config(
        api_key=api_key,
        base_url=base_url,
        model=llm_model,
        course_context=course_ctx,
        top_k=top_k,
        n_candidates=n_candidates,
        keep_top=keep_top_n,
        enable_llm_scoring=enable_llm_scoring,
        filter_noise=filter_noise,
        report_language=report_language,
    )

    with tempfile.TemporaryDirectory() as tmp:
        course_dir = Path(tmp) / "course"

        # Write uploaded files
        save_uploads(slides_files, course_dir / "slides")
        if papers_files:
            save_uploads(papers_files, course_dir / "past_papers")
        if tut_files:
            save_uploads(tut_files, course_dir / "tutorials")
        if syl_file:
            (course_dir / "syllabus.md").write_bytes(syl_file.getvalue())

        # Run the pipeline with a live status panel
        with st.status("🔄  Running the prediction engine…", expanded=True) as status:
            try:
                st.write("**Step 1 / 2** ⚙️  Loading the embedding model…")
                st.caption("First run downloads the BGE model (~1.3 GB); cached afterwards.")
                predictor = ExamPredictor(cfg)

                st.write("**Step 2 / 2** 🚀  Running the prediction stages (~1–3 min)…")
                st.caption(
                    "Stage 1 parse → Stage 2 align → Stage 2.5 LLM scoring → "
                    "Stage 3 fuse → Stage 4 generate"
                )
                report = predictor.predict(
                    course_dir,
                    course_name=course_name,
                    course_context=course_ctx or None,
                )
                md = ExamPredictor._format_markdown_report(report)

                # Build the PDF once and cache in session_state
                pdf_bytes = None
                try:
                    from exam_predictor.exporter import report_to_pdf_bytes
                    pdf_bytes = report_to_pdf_bytes(report)
                except Exception as pdf_exc:  # noqa: BLE001
                    st.warning(f"PDF generation failed (Markdown/JSON still available): {pdf_exc}")

                # Stash for display
                st.session_state["report"]     = report
                st.session_state["md"]         = md
                st.session_state["pdf"]        = pdf_bytes
                st.session_state["run_course"] = course_name

                status.update(label="✅  Done!", state="complete", expanded=False)

            except Exception as exc:
                status.update(label="❌  Run failed", state="error")
                st.error(f"Error: {exc}")
                with st.expander("🐛 Full traceback"):
                    st.code(traceback.format_exc())
                st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Show results
# ═══════════════════════════════════════════════════════════════════════════════

if "report" in st.session_state:
    import pandas as pd
    from exam_predictor.pipeline import ExamPredictor as EP

    report = st.session_state["report"]
    md     = st.session_state["md"]
    name   = st.session_state.get("run_course", "course")

    st.success(
        f"🎉  Analyzed **{report.n_chunks}** chunks · "
        f"referenced **{report.n_past_questions}** past papers · "
        f"generated **{len(report.generated_questions)}** practice questions · "
        f"overall confidence **{report.overall_confidence:.0%}**"
    )

    # Warnings
    for w in report.warnings:
        st.warning(w)

    tab_rank, tab_qs, tab_dl = st.tabs([
        "📊 Topic Ranking",
        "📝 Practice Questions",
        "📥 Download",
    ])

    # ── Tab 1: ranking table ─────────────────────────────────────────────────
    with tab_rank:
        st.markdown("#### Importance ranking (higher = more frequently examined)")

        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        rows = []
        for i, p in enumerate(report.predictions, 1):
            f = p.features
            rows.append({
                "Rank":       medals.get(i, str(i)),
                "Knowledge Point": EP._clean_kp_title(p.title),
                "Importance": f"{heat_bar(p.score)}  {p.score:.2f}",
                "LLM score":  f"{f.llm_pedagogy_score:.2f}",
                "Structural": f"{f.structural_signals:.2f}",
                "Past hits":  f.evidence.get("total_questions_matched", 0),
                "Data mode": (
                    "📉 Few-shot" if f.evidence.get("weight_regime") == "sparse"
                    else "📈 Normal"
                ),
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Knowledge Point": st.column_config.TextColumn(width="large"),
                "Importance":      st.column_config.TextColumn(width="medium"),
            },
        )

        st.caption(
            "💡 **Data mode**: with fewer than 10 past papers the system switches to "
            "**Few-shot** mode — the LLM pedagogy score's weight rises to 45% to "
            "compensate for sparse history."
        )

    # ── Tab 2: practice questions ────────────────────────────────────────────
    with tab_qs:
        if not report.generated_questions:
            st.info(
                "No practice questions were generated.\n\n"
                "Possible causes: insufficient API balance · Top-K set too high · "
                "course materials too short."
            )
        else:
            kp_title_map = {
                p.knowledge_point_id: EP._clean_kp_title(p.title)
                for p in report.predictions
            }
            kp_rank_map = {
                p.knowledge_point_id: i
                for i, p in enumerate(report.predictions, 1)
            }

            # Group by knowledge point
            by_kp: dict = {}
            for q in report.generated_questions:
                by_kp.setdefault(q.knowledge_point_id, []).append(q)

            ordered_kps = sorted(by_kp, key=lambda k: kp_rank_map.get(k, 999))

            for sec_i, kp_id in enumerate(ordered_kps, 1):
                qs    = by_kp[kp_id]
                title = kp_title_map.get(kp_id, kp_id)
                rank  = kp_rank_map.get(kp_id, "?")

                st.markdown(f"### {sec_i}. {title}")
                st.caption(f"Importance rank #{rank} · {len(qs)} practice question(s)")

                for q_i, q in enumerate(qs, 1):
                    d      = q.estimated_difficulty
                    stars  = diff_stars(d)
                    label  = diff_label(d)
                    qtype  = f"  ·  {q.question_type}" if q.question_type else ""
                    header = f"Question {q_i}　{stars}　{label}{qtype}"

                    # Expand the very first question by default
                    with st.expander(header, expanded=(q_i == 1 and sec_i == 1)):
                        st.markdown(q.text)
                        if q.answer_sketch:
                            st.info(f"💡 **Reference answer (key points)**\n\n{q.answer_sketch}")

    # ── Tab 3: download ──────────────────────────────────────────────────────
    with tab_dl:
        pdf_bytes = st.session_state.get("pdf")

        st.markdown("#### Choose a download format")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            if pdf_bytes:
                st.download_button(
                    "📕  Download PDF (recommended)",
                    data=pdf_bytes,
                    file_name=f"{name}_exam_prediction.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.button(
                    "📕  PDF generation failed",
                    disabled=True,
                    use_container_width=True,
                )

        with dl_col2:
            st.download_button(
                "📝  Download Markdown",
                data=md.encode("utf-8"),
                file_name=f"{name}_exam_prediction.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with dl_col3:
            st.download_button(
                "🗂️  Download JSON",
                data=json.dumps(
                    report.model_dump(), ensure_ascii=False, indent=2
                ).encode("utf-8"),
                file_name=f"{name}_predictions.json",
                mime="application/json",
                use_container_width=True,
            )

        st.caption(
            "💡 **PDF** is best for printing and sharing; **Markdown** for further editing; "
            "**JSON** holds all structured data for programmatic use."
        )

        st.divider()
        st.caption("📄 Report preview")
        st.markdown(md)
