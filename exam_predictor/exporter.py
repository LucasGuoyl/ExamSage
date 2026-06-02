"""PDF exporter — render a PredictionReport to a print-ready PDF.

Built directly from the structured report object (not Markdown) so layout is
fully controlled and there are no broken glyphs. Uses a CJK TrueType font for
Chinese text and avoids emoji / block-drawing characters that don't exist in
standard fonts.

Public API:
    report_to_pdf_bytes(report) -> bytes
"""

from __future__ import annotations

import io
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .schema import GeneratedQuestion, PredictionReport


# ── Font registration (cross-platform with built-in fallback) ────────────────

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
]

_FONT_NAME: str | None = None


def _cjk_font() -> str:
    """Register and return a usable CJK font name (cached)."""
    global _FONT_NAME
    if _FONT_NAME:
        return _FONT_NAME
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("CJK", path))
                _FONT_NAME = "CJK"
                return _FONT_NAME
            except Exception:
                continue
    # Always-available built-in CID font (ships with reportlab)
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    _FONT_NAME = "STSong-Light"
    return _FONT_NAME


# ── Helpers ──────────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    """Escape XML special chars for reportlab Paragraph markup."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _diff_label(d: float | None) -> str:
    if d is None:    return "未知"
    if d < 0.35:     return "基础"
    if d < 0.55:     return "中等"
    if d < 0.75:     return "进阶"
    return "挑战"


def _importance_label(score: float) -> str:
    if score >= 0.6:  return "极高"
    if score >= 0.45: return "高"
    if score >= 0.30: return "中"
    return "低"


def _styles(font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s: dict[str, ParagraphStyle] = {}
    s["title"] = ParagraphStyle(
        "T", parent=base["Title"], fontName=font, fontSize=22, leading=28,
        textColor=colors.HexColor("#1a3c6e"), spaceAfter=4,
    )
    s["subtitle"] = ParagraphStyle(
        "ST", fontName=font, fontSize=14, leading=18,
        textColor=colors.HexColor("#555555"), spaceAfter=10,
    )
    s["h2"] = ParagraphStyle(
        "H2", fontName=font, fontSize=15, leading=20, spaceBefore=10, spaceAfter=8,
        textColor=colors.HexColor("#1a3c6e"),
    )
    s["kp"] = ParagraphStyle(
        "KP", fontName=font, fontSize=13, leading=18, spaceBefore=6, spaceAfter=4,
        textColor=colors.white, backColor=colors.HexColor("#2a6099"),
        borderPadding=(5, 6, 5, 6), leftIndent=0,
    )
    s["meta"] = ParagraphStyle(
        "M", fontName=font, fontSize=9, leading=13,
        textColor=colors.HexColor("#777777"), spaceAfter=6,
    )
    s["label"] = ParagraphStyle(
        "L", fontName=font, fontSize=10.5, leading=15, spaceBefore=4, spaceAfter=2,
        textColor=colors.HexColor("#1a3c6e"),
    )
    s["body"] = ParagraphStyle(
        "B", fontName=font, fontSize=10.5, leading=16, alignment=TA_LEFT, spaceAfter=4,
    )
    s["q"] = ParagraphStyle(
        "Q", fontName=font, fontSize=10.5, leading=16, spaceBefore=2, spaceAfter=3,
        leftIndent=4,
    )
    s["qhead"] = ParagraphStyle(
        "QH", fontName=font, fontSize=10.5, leading=15, spaceBefore=6, spaceAfter=2,
        textColor=colors.HexColor("#b8500a"),
    )
    s["ans"] = ParagraphStyle(
        "A", fontName=font, fontSize=9.5, leading=14.5,
        textColor=colors.HexColor("#333333"),
    )
    s["cell"] = ParagraphStyle(
        "C", fontName=font, fontSize=9.5, leading=13,
    )
    s["cellb"] = ParagraphStyle(
        "CB", fontName=font, fontSize=9.5, leading=13, textColor=colors.white,
    )
    s["foot"] = ParagraphStyle(
        "F", fontName=font, fontSize=8.5, leading=12,
        textColor=colors.HexColor("#999999"), spaceBefore=12,
    )
    return s


# ── Main builder ─────────────────────────────────────────────────────────────

def report_to_pdf_bytes(report: PredictionReport) -> bytes:
    """Render the report to PDF and return the raw bytes."""
    font = _cjk_font()
    st = _styles(font)
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        title=f"{report.course_name} 考试重点预测报告",
    )
    usable_w = doc.width
    flow: list = []

    # —— 封面 ——
    flow.append(Paragraph("考试重点预测报告", st["title"]))
    flow.append(Paragraph(_esc(report.course_name), st["subtitle"]))

    info = [
        [Paragraph("分析课件块数", st["cell"]), Paragraph(str(report.n_chunks), st["cell"]),
         Paragraph("参考历年真题", st["cell"]), Paragraph(f"{report.n_past_questions} 道", st["cell"])],
        [Paragraph("预测置信度", st["cell"]), Paragraph(f"{report.overall_confidence:.0%}", st["cell"]),
         Paragraph("知识点数", st["cell"]), Paragraph(str(len(report.predictions)), st["cell"])],
    ]
    info_tbl = Table(info, colWidths=[usable_w * 0.25] * 4)
    info_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f4f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    flow.append(info_tbl)
    flow.append(Spacer(1, 6))

    for w in report.warnings:
        flow.append(Paragraph("⚠ " + _esc(w), st["meta"]))

    flow.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc"),
                           spaceBefore=8, spaceAfter=10))

    # —— 一、总览表 ——
    flow.append(Paragraph("一、核心知识点总览", st["h2"]))
    header = [
        Paragraph("排名", st["cellb"]), Paragraph("知识点", st["cellb"]),
        Paragraph("重要度", st["cellb"]), Paragraph("命中真题", st["cellb"]),
        Paragraph("概述", st["cellb"]),
    ]
    rows = [header]
    for i, p in enumerate(report.predictions, 1):
        hits = p.features.evidence.get("total_questions_matched", 0)
        desc = p.description or p.representative_text or ""
        snippet = (desc[:48] + "…") if len(desc) > 48 else desc
        rows.append([
            Paragraph(str(i), st["cell"]),
            Paragraph(_esc(p.title), st["cell"]),
            Paragraph(f"{p.score:.2f} ({_importance_label(p.score)})", st["cell"]),
            Paragraph(f"{hits} 题", st["cell"]),
            Paragraph(_esc(snippet), st["cell"]),
        ])
    overview = Table(
        rows,
        colWidths=[usable_w * x for x in (0.08, 0.28, 0.16, 0.12, 0.36)],
        repeatRows=1,
    )
    overview.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a6099")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fc")]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    flow.append(overview)
    flow.append(Spacer(1, 10))

    # —— 二、知识点详解 ——
    flow.append(Paragraph("二、知识点详解与押题", st["h2"]))
    flow.append(Paragraph(
        "每节依次包含：概念说明 → 常考方向 → 练习题（附参考答案）。", st["meta"]))

    by_kp: dict[str, list[GeneratedQuestion]] = {}
    for q in report.generated_questions:
        by_kp.setdefault(q.knowledge_point_id, []).append(q)

    CN = "一二三四五六七八九十"
    for i, p in enumerate(report.predictions, 1):
        qs   = by_kp.get(p.knowledge_point_id, [])
        cn   = CN[i - 1] if i <= len(CN) else str(i)
        hits = p.features.evidence.get("total_questions_matched", 0)

        # 节标题（蓝底白字）
        flow.append(Paragraph(f"第{cn}考点：{_esc(p.title)}", st["kp"]))
        flow.append(Paragraph(
            f"重要度 {p.score:.2f}（{_importance_label(p.score)}）　|　"
            f"历年命中 {hits} 题　|　"
            f"权重模式 {'少样本' if p.features.evidence.get('weight_regime') == 'sparse' else '正常'}",
            st["meta"],
        ))

        if p.description:
            flow.append(Paragraph("概念说明", st["label"]))
            flow.append(Paragraph(_esc(p.description), st["body"]))

        if p.exam_directions:
            flow.append(Paragraph("常考方向", st["label"]))
            for j, d in enumerate(p.exam_directions, 1):
                flow.append(Paragraph(f"{j}. {_esc(d)}", st["body"]))

        if qs:
            flow.append(Paragraph("练习题", st["label"]))
            for q_i, q in enumerate(qs, 1):
                qtype = f" · {q.question_type}" if q.question_type else ""
                flow.append(Paragraph(
                    f"第 {q_i} 题　[{_diff_label(q.estimated_difficulty)}{qtype}]",
                    st["qhead"],
                ))
                flow.append(Paragraph(_esc(q.text), st["q"]))
                if q.answer_sketch:
                    ans = Paragraph(
                        "<b>参考答案要点：</b>" + _esc(q.answer_sketch), st["ans"])
                    ans_tbl = Table([[ans]], colWidths=[usable_w])
                    ans_tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f5f0")),
                        ("LINEBEFORE", (0, 0), (0, -1), 2, colors.HexColor("#d0a040")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]))
                    flow.append(ans_tbl)
                    flow.append(Spacer(1, 4))
        else:
            flow.append(Paragraph("（本知识点暂无生成练习题）", st["meta"]))

        flow.append(HRFlowable(width="100%", thickness=0.5,
                               color=colors.HexColor("#dddddd"),
                               spaceBefore=8, spaceAfter=8))

    flow.append(Paragraph(
        "本报告由 押题宝 Exam Predictor 自动生成，仅供参考，请以课堂教材及教师说明为准。",
        st["foot"],
    ))

    doc.build(flow)
    return buf.getvalue()
