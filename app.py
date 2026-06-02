"""押题宝 · Exam Predictor — Streamlit Web UI

启动方式:
    streamlit run app.py

首次运行会自动下载 BGE Embedding 模型 (~1.3 GB)，后续运行无需重新下载。
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import streamlit as st

# ── 确保 exam_predictor 包可以被导入 ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── 页面配置（必须在所有其他 st 调用之前）────────────────────────────────────
st.set_page_config(
    page_title="押题宝 · Exam Predictor",
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
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def save_uploads(files, dest: Path) -> None:
    """把 Streamlit 上传的文件对象写入磁盘。"""
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
) -> dict:
    """从 UI 参数构建 pipeline config 字典（无需 config.yaml）。"""
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
            # 智能过滤：开启时剔除标题/目录/背景/行政内容；关闭时保留全部
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
    if d < 0.35:      return "基础"
    if d < 0.55:      return "中等"
    if d < 0.75:      return "进阶"
    return "挑战"


# ═══════════════════════════════════════════════════════════════════════════════
# 侧边栏：配置区
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ 配置")

    # —— API 设置
    st.subheader("🔑 API 设置")
    api_key = st.text_input(
        "API Key *",
        type="password",
        placeholder="sk-...",
        help="支持 DeepSeek / OpenAI / 任何兼容 OpenAI 格式的服务",
    )
    base_url = st.text_input(
        "API Base URL",
        value="https://api.deepseek.com/v1",
        help="DeepSeek: https://api.deepseek.com/v1\nOpenAI: https://api.openai.com/v1",
    )
    llm_model = st.text_input("模型名称", value="deepseek-chat")

    st.divider()

    # —— 课程信息
    st.subheader("📖 课程信息")
    course_name = st.text_input(
        "课程名称 *",
        placeholder="例：电力系统分析",
    )
    course_ctx = st.text_area(
        "课程描述（可选）",
        placeholder="例：本科电力系统课程，重点考牛顿-拉夫逊潮流计算和经济调度推导",
        height=90,
        help="描述越具体，LLM 教学评分越准确，跨课程效果越好",
    )

    st.divider()

    # —— 高级参数
    st.subheader("🔧 高级参数")
    top_k        = st.slider("预测 Top-K 知识点", 3, 20, 10)
    n_candidates = st.slider("候选题数 / 知识点",  2,  8,  5)
    keep_top_n   = st.slider("最终保留 / 知识点",  1,  4,  2)
    enable_llm_scoring = st.checkbox(
        "启用 LLM 教学评分",
        value=True,
        help="少样本/新课程场景下大幅提升准确度；每次多消耗少量 API 额度",
    )
    filter_noise = st.checkbox(
        "智能过滤非考点内容",
        value=True,
        help="剔除标题页、目录、课程回顾、背景介绍、行政信息等，"
             "避免它们被误判为知识点",
    )

    st.divider()

    # 运行按钮放在侧边栏最底部，始终可见
    _can_run_sidebar = bool(api_key and course_name)
    if not _can_run_sidebar:
        _miss = [l for l, ok in [("API Key", api_key), ("课程名称", course_name)] if not ok]
        st.caption(f"⚠️ 还需填写：{'、'.join(_miss)}")

    run_sidebar_btn = st.button(
        "🚀  开始分析",
        type="primary",
        disabled=not _can_run_sidebar,
        use_container_width=True,
        key="run_sidebar",
    )

    st.divider()
    st.caption("押题宝 v0.2 · [GitHub](https://github.com/your-repo/exam-predictor)")


# ═══════════════════════════════════════════════════════════════════════════════
# 主区域
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🎓 押题宝 · 考试预测系统")
st.caption("上传课件和历年真题 → AI 分析考点热度 → 自动生成押题练习")

# ── 文件上传区 ────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.subheader("📄 课件材料（必填）")
    slides_files = st.file_uploader(
        "支持 **PDF / PPTX / Markdown**，可同时上传多个",
        type=["pdf", "pptx", "md"],
        accept_multiple_files=True,
        key="slides",
    )
    if slides_files:
        total_kb = sum(f.size for f in slides_files) // 1024
        st.success(f"✅ 已选择 **{len(slides_files)}** 个文件 · 共 {total_kb} KB")
        with st.expander(f"查看文件列表（{len(slides_files)} 个）"):
            for f in slides_files:
                st.caption(f"· `{f.name}`  {f.size // 1024} KB")

with col_r:
    st.subheader("📚 历年真题（可选，越多越准）")
    papers_files = st.file_uploader(
        "支持 **JSON / PDF / Markdown**，可同时上传多个",
        type=["json", "pdf", "md"],
        accept_multiple_files=True,
        key="papers",
    )
    if papers_files:
        st.success(f"✅ 已选择 **{len(papers_files)}** 个文件")
        with st.expander(f"查看文件列表（{len(papers_files)} 个）"):
            for f in papers_files:
                st.caption(f"· `{f.name}`")

    # JSON 模板下载
    TEMPLATE = [
        {
            "id": "2023_q1",
            "year": 2023,
            "text": "在此填写题目原文（必填）",
            "type": "computation",
            "answer": "参考答案要点（可选）",
        },
        {
            "id": "2022_q1",
            "year": 2022,
            "text": "在此填写题目原文",
            "type": "derivation",
        },
    ]
    st.download_button(
        "📥 下载历年真题 JSON 模板",
        data=json.dumps(TEMPLATE, ensure_ascii=False, indent=2),
        file_name="questions_template.json",
        mime="application/json",
        use_container_width=True,
    )

# 可选材料
with st.expander("➕ 习题集 & 教学大纲（可选，提升准确度）"):
    c1, c2 = st.columns(2)
    with c1:
        st.caption("**习题集 / Tutorial**（PDF 或 Markdown）")
        tut_files = st.file_uploader(
            "习题集",
            type=["pdf", "md"],
            accept_multiple_files=True,
            key="tutorials",
            label_visibility="collapsed",
        )
    with c2:
        st.caption("**教学大纲**（.md 或 .txt）")
        syl_file = st.file_uploader(
            "大纲",
            type=["md", "txt"],
            key="syllabus",
            label_visibility="collapsed",
        )

st.divider()

# ── 运行按钮 & 校验提示 ───────────────────────────────────────────────────────
can_run = bool(api_key and slides_files and course_name)

if not can_run:
    missing = [label for label, ok in [
        ("左侧 API Key", api_key),
        ("左侧课程名称", course_name),
        ("课件材料",     slides_files),
    ] if not ok]
    st.warning("⚠️ 还需要填写/上传：" + "、".join(missing))

# 主区域按钮（页面中央）
run_btn = st.button(
    "🚀  开始分析",
    type="primary",
    disabled=not can_run,
    use_container_width=True,
    key="run_main",
)

# 侧边栏按钮也可触发（二者任一点击即可）
run_btn = run_btn or (run_sidebar_btn and can_run)

# ═══════════════════════════════════════════════════════════════════════════════
# 执行 Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

if run_btn:
    # 清除上次结果
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
    )

    with tempfile.TemporaryDirectory() as tmp:
        course_dir = Path(tmp) / "course"

        # 写入上传的文件
        save_uploads(slides_files, course_dir / "slides")
        if papers_files:
            save_uploads(papers_files, course_dir / "past_papers")
        if tut_files:
            save_uploads(tut_files, course_dir / "tutorials")
        if syl_file:
            (course_dir / "syllabus.md").write_bytes(syl_file.getvalue())

        # 运行 pipeline，用 st.status 显示进度
        with st.status("🔄  正在运行预测引擎…", expanded=True) as status:
            try:
                st.write("**Step 1 / 2** ⚙️  加载 Embedding 模型…")
                st.caption("首次运行需下载 BGE 模型 (~1.3 GB)，已下载的课程秒开。")
                predictor = ExamPredictor(cfg)

                st.write("**Step 2 / 2** 🚀  运行四阶段预测（约 1–3 分钟）…")
                st.caption(
                    "Stage 1 解析课件 → Stage 2 对齐真题 → "
                    "Stage 2.5 LLM 教学评分 → Stage 3 融合 → Stage 4 生成押题"
                )
                report = predictor.predict(
                    course_dir,
                    course_name=course_name,
                    course_context=course_ctx or None,
                )
                md = ExamPredictor._format_markdown_report(report)

                # 生成 PDF（一次性，缓存进 session_state）
                pdf_bytes = None
                try:
                    from exam_predictor.exporter import report_to_pdf_bytes
                    pdf_bytes = report_to_pdf_bytes(report)
                except Exception as pdf_exc:  # noqa: BLE001
                    st.warning(f"PDF 生成失败（仍可下载 Markdown/JSON）：{pdf_exc}")

                # 存入 session_state 供展示用
                st.session_state["report"]     = report
                st.session_state["md"]         = md
                st.session_state["pdf"]        = pdf_bytes
                st.session_state["run_course"] = course_name

                status.update(label="✅  预测完成！", state="complete", expanded=False)

            except Exception as exc:
                status.update(label="❌  运行失败", state="error")
                st.error(f"错误：{exc}")
                with st.expander("🐛 详细错误信息"):
                    st.code(traceback.format_exc())
                st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# 展示结果
# ═══════════════════════════════════════════════════════════════════════════════

if "report" in st.session_state:
    import pandas as pd
    from exam_predictor.pipeline import ExamPredictor as EP

    report = st.session_state["report"]
    md     = st.session_state["md"]
    name   = st.session_state.get("run_course", "课程")

    st.success(
        f"🎉  已分析 **{report.n_chunks}** 个课件块 · "
        f"参考 **{report.n_past_questions}** 道历年真题 · "
        f"生成 **{len(report.generated_questions)}** 道练习题 · "
        f"综合置信度 **{report.overall_confidence:.0%}**"
    )

    # 警告提示
    for w in report.warnings:
        st.warning(w)

    tab_rank, tab_qs, tab_dl = st.tabs([
        "📊 知识点热度排行",
        "📝 押题练习",
        "📥 下载报告",
    ])

    # ── Tab 1：知识点排行表 ──────────────────────────────────────────────────
    with tab_rank:
        st.markdown("#### 综合热度排行（热度越高 = 历年考察频率越高）")

        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        rows = []
        for i, p in enumerate(report.predictions, 1):
            f = p.features
            rows.append({
                "排名":     medals.get(i, str(i)),
                "知识点":   EP._clean_kp_title(p.title),
                "热度":     f"{heat_bar(p.score)}  {p.score:.2f}",
                "LLM评分":  f"{f.llm_pedagogy_score:.2f}",
                "结构信号": f"{f.structural_signals:.2f}",
                "命中真题": f.evidence.get("total_questions_matched", 0),
                "数据模式": (
                    "📉 少样本" if f.evidence.get("weight_regime") == "sparse"
                    else "📈 正常"
                ),
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "知识点": st.column_config.TextColumn(width="large"),
                "热度":   st.column_config.TextColumn(width="medium"),
            },
        )

        st.caption(
            "💡 **数据模式**：历年真题 < 10 道时自动切换为「少样本」模式，"
            "LLM 教学评分权重上升至 45%，以补偿历史数据不足。"
        )

    # ── Tab 2：押题练习 ──────────────────────────────────────────────────────
    with tab_qs:
        if not report.generated_questions:
            st.info(
                "暂无生成的练习题。\n\n"
                "可能原因：API 余额不足 · Top-K 设置过大 · 课件内容过短。"
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

            # 按知识点分组
            by_kp: dict = {}
            for q in report.generated_questions:
                by_kp.setdefault(q.knowledge_point_id, []).append(q)

            ordered_kps = sorted(by_kp, key=lambda k: kp_rank_map.get(k, 999))
            CN = "一二三四五六七八九十"

            for sec_i, kp_id in enumerate(ordered_kps):
                qs    = by_kp[kp_id]
                title = kp_title_map.get(kp_id, kp_id)
                rank  = kp_rank_map.get(kp_id, "?")
                cn    = CN[sec_i] if sec_i < len(CN) else str(sec_i + 1)

                st.markdown(f"### {cn}、{title}")
                st.caption(f"热度排名第 {rank} 位 · {len(qs)} 道练习题")

                for q_i, q in enumerate(qs, 1):
                    d      = q.estimated_difficulty
                    stars  = diff_stars(d)
                    label  = diff_label(d)
                    qtype  = f"  ·  {q.question_type}" if q.question_type else ""
                    header = f"第 {q_i} 题　{stars}　{label}{qtype}"

                    # 第一道题默认展开
                    with st.expander(header, expanded=(q_i == 1 and sec_i == 0)):
                        st.markdown(q.text)
                        if q.answer_sketch:
                            st.info(f"💡 **参考答案要点**\n\n{q.answer_sketch}")

    # ── Tab 3：下载报告 ──────────────────────────────────────────────────────
    with tab_dl:
        pdf_bytes = st.session_state.get("pdf")

        st.markdown("#### 选择下载格式")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            if pdf_bytes:
                st.download_button(
                    "📕  下载 PDF 报告（推荐）",
                    data=pdf_bytes,
                    file_name=f"{name}_押题报告.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.button(
                    "📕  PDF 生成失败",
                    disabled=True,
                    use_container_width=True,
                )

        with dl_col2:
            st.download_button(
                "📝  下载 Markdown",
                data=md.encode("utf-8"),
                file_name=f"{name}_押题报告.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with dl_col3:
            st.download_button(
                "🗂️  下载 JSON 数据",
                data=json.dumps(
                    report.model_dump(), ensure_ascii=False, indent=2
                ).encode("utf-8"),
                file_name=f"{name}_predictions.json",
                mime="application/json",
                use_container_width=True,
            )

        st.caption(
            "💡 **PDF** 适合打印复习与分享；**Markdown** 适合二次编辑；"
            "**JSON** 含全部结构化数据，便于程序处理。"
        )

        st.divider()
        st.caption("📄 报告预览")
        st.markdown(md)
