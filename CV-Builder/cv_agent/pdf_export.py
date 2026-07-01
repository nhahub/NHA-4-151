"""
cv_agent.pdf_export — Markdown-to-PDF export using ReportLab + optional Mistune.

Supports 3 visual templates: classic, modern, monochrome.
"""
from __future__ import annotations
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict
from cv_agent.config import _REPORTLAB_AVAILABLE, _MISTUNE_AVAILABLE
from cv_agent.utils import md_to_rl

# Available template names
TEMPLATES = ("classic", "modern", "monochrome")

def _require_reportlab():
    if not _REPORTLAB_AVAILABLE:
        raise ImportError("PDF export requires reportlab: pip install reportlab")


# ==============================================================================
# TEMPLATE 1: CLASSIC — Centered name, blue headers, slate body
# ==============================================================================

def _build_classic_styles() -> Dict[str, Any]:
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    C_INK  = colors.HexColor("#1e293b")
    C_NAME = colors.HexColor("#0f172a")
    C_BLUE = colors.HexColor("#2563eb")
    C_RULE = colors.HexColor("#cbd5e1")
    C_GRAY = colors.HexColor("#475569")
    base = getSampleStyleSheet()["Normal"]
    return {
        "name":    ParagraphStyle("s_name", parent=base, fontName="Helvetica-Bold", fontSize=26, leading=30, textColor=C_NAME, spaceAfter=4, alignment=1),
        "contact": ParagraphStyle("s_contact", parent=base, fontName="Helvetica", fontSize=10, leading=14, textColor=C_GRAY, spaceAfter=12, alignment=1),
        "section": ParagraphStyle("s_section", parent=base, fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=C_BLUE, spaceBefore=14, spaceAfter=4),
        "sub":     ParagraphStyle("s_sub", parent=base, fontName="Helvetica-Bold", fontSize=11.5, leading=16, textColor=C_NAME, spaceBefore=8, spaceAfter=2),
        "body":    ParagraphStyle("s_body", parent=base, fontName="Helvetica", fontSize=10, leading=14, textColor=C_INK, spaceAfter=4),
        "bullet":  ParagraphStyle("s_bullet", parent=base, fontName="Helvetica", fontSize=10, leading=14, textColor=C_INK, leftIndent=15, firstLineIndent=-10, spaceAfter=3),
        "_C_ACCENT": C_BLUE, "_C_RULE": C_RULE,
    }


# ==============================================================================
# TEMPLATE 2: MODERN — Teal accent, left-aligned name, compact
# ==============================================================================

def _build_modern_styles() -> Dict[str, Any]:
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    C_INK   = colors.HexColor("#1a1a2e")
    C_NAME  = colors.HexColor("#0f3460")
    C_TEAL  = colors.HexColor("#0d9488")
    C_RULE  = colors.HexColor("#99f6e4")
    C_GRAY  = colors.HexColor("#4b5563")
    base = getSampleStyleSheet()["Normal"]
    return {
        "name":    ParagraphStyle("s_name_m", parent=base, fontName="Helvetica-Bold", fontSize=28, leading=32, textColor=C_NAME, spaceAfter=2, alignment=0),
        "contact": ParagraphStyle("s_contact_m", parent=base, fontName="Helvetica", fontSize=9.5, leading=13, textColor=C_TEAL, spaceAfter=10, alignment=0),
        "section": ParagraphStyle("s_section_m", parent=base, fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=C_TEAL, spaceBefore=12, spaceAfter=3, textTransform="uppercase"),
        "sub":     ParagraphStyle("s_sub_m", parent=base, fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=C_NAME, spaceBefore=6, spaceAfter=2),
        "body":    ParagraphStyle("s_body_m", parent=base, fontName="Helvetica", fontSize=9.5, leading=13, textColor=C_INK, spaceAfter=3),
        "bullet":  ParagraphStyle("s_bullet_m", parent=base, fontName="Helvetica", fontSize=9.5, leading=13, textColor=C_INK, leftIndent=14, firstLineIndent=-10, spaceAfter=2),
        "_C_ACCENT": C_TEAL, "_C_RULE": C_RULE,
    }


# ==============================================================================
# TEMPLATE 3: MONOCHROME — All black, zero colors
# ==============================================================================

def _build_monochrome_styles() -> Dict[str, Any]:
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    C_BLACK = colors.HexColor("#000000")
    C_DARK  = colors.HexColor("#1a1a1a")
    C_RULE  = colors.HexColor("#333333")
    C_GRAY  = colors.HexColor("#2a2a2a")
    base = getSampleStyleSheet()["Normal"]
    return {
        "name":    ParagraphStyle("s_name_bw", parent=base, fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=C_BLACK, spaceAfter=4, alignment=1),
        "contact": ParagraphStyle("s_contact_bw", parent=base, fontName="Helvetica", fontSize=10, leading=14, textColor=C_GRAY, spaceAfter=12, alignment=1),
        "section": ParagraphStyle("s_section_bw", parent=base, fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=C_BLACK, spaceBefore=14, spaceAfter=4),
        "sub":     ParagraphStyle("s_sub_bw", parent=base, fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=C_DARK, spaceBefore=8, spaceAfter=2),
        "body":    ParagraphStyle("s_body_bw", parent=base, fontName="Helvetica", fontSize=10, leading=14, textColor=C_DARK, spaceAfter=4),
        "bullet":  ParagraphStyle("s_bullet_bw", parent=base, fontName="Helvetica", fontSize=10, leading=14, textColor=C_DARK, leftIndent=15, firstLineIndent=-10, spaceAfter=3),
        "_C_ACCENT": C_BLACK, "_C_RULE": C_RULE,
    }


# ==============================================================================
# TEMPLATE SELECTOR
# ==============================================================================

_TEMPLATE_BUILDERS = {
    "classic":    _build_classic_styles,
    "modern":     _build_modern_styles,
    "monochrome": _build_monochrome_styles,
}

def _build_pdf_styles(template: str = "classic") -> Dict[str, Any]:
    builder = _TEMPLATE_BUILDERS.get(template, _build_classic_styles)
    return builder()


# ==============================================================================
# STORY BUILDERS (unchanged — they use the style dict generically)
# ==============================================================================

def _story_from_mistune(md_text: str, styles: Dict[str, Any]) -> list:
    import mistune
    from reportlab.platypus import Paragraph, Spacer, HRFlowable
    C_ACCENT, C_RULE = styles["_C_ACCENT"], styles["_C_RULE"]
    md = mistune.create_markdown(renderer="ast")
    ast = md(md_text)
    story: list = []
    after_name = False

    def _inline_to_text(children):
        out = ""
        for child in children:
            raw = child.get("raw", ""); ch = child.get("children", []); t = child.get("type", "")
            if t == "link":
                url = child.get("link", "") or child.get("attrs", {}).get("url", "") or child.get("attrs", {}).get("link", "")
                link_text = _inline_to_text(ch) if ch else (raw or url)
                out += f'<a href="{url}" color="blue">{link_text}</a>'
            elif t == "strong": out += f"<b>{_inline_to_text(ch)}</b>"
            elif t == "emphasis": out += f"<i>{_inline_to_text(ch)}</i>"
            elif t == "codespan": out += f"<font name='Courier'>{md_to_rl(raw)}</font>"
            else: out += md_to_rl(raw or "".join(c.get("raw", "") for c in ch))
        return out
    def _token_to_flowables(token):
        nonlocal after_name
        t = token.get("type", ""); ch = token.get("children", [])
        if t == "heading":
            level = token.get("attrs", {}).get("level", 2); text = _inline_to_text(ch)
            if level == 1:
                story.append(Paragraph(text, styles["name"]))
                story.append(Spacer(1, 6))
                after_name = True
            elif level == 2:
                story.append(Spacer(1, 10)); story.append(Paragraph(text.upper(), styles["section"]))
                story.append(HRFlowable(width="100%", thickness=1, color=C_ACCENT, spaceAfter=8))
                after_name = False
            else: 
                story.append(Spacer(1, 6)); story.append(Paragraph(text, styles["sub"]))
                after_name = False
        elif t == "paragraph":
            text = _inline_to_text(ch)
            if text.strip():
                if after_name:
                    story.append(Paragraph(text, styles["contact"]))
                    after_name = False
                else:
                    story.append(Paragraph(text, styles["body"]))
            else:
                story.append(Spacer(1, 3))
        elif t == "list":
            after_name = False
            for item in ch:
                for sub in item.get("children", []):
                    item_text = _inline_to_text(sub.get("children", []))
                    if item_text.strip(): story.append(Paragraph(f"• {item_text}", styles["bullet"]))
        elif t == "block_code": 
            after_name = False
            story.append(Paragraph(md_to_rl(token.get("raw", "")), styles["body"]))
        elif t == "thematic_break": 
            after_name = False
            story.append(HRFlowable(width="100%", thickness=0.5, color=C_RULE, spaceAfter=3))
        elif t == "blank_line": 
            story.append(Spacer(1, 3))
        else:
            for child in ch: _token_to_flowables(child)
    for token in ast: _token_to_flowables(token)
    return story

def _story_from_lines(md_text: str, styles: Dict[str, Any]) -> list:
    from reportlab.platypus import Paragraph, Spacer, HRFlowable
    C_ACCENT, C_RULE = styles["_C_ACCENT"], styles["_C_RULE"]
    story: list = []
    after_name = False
    for line in md_text.splitlines():
        stripped = line.strip()
        if not stripped: story.append(Spacer(1, 3)); continue
        if stripped.startswith("# "):
            story.append(Paragraph(md_to_rl(stripped[2:]), styles["name"]))
            story.append(Spacer(1, 6))
            after_name = True
        elif stripped.startswith("## "):
            after_name = False
            story.append(Spacer(1, 10)); story.append(Paragraph(md_to_rl(stripped[3:]).upper(), styles["section"]))
            story.append(HRFlowable(width="100%", thickness=1, color=C_ACCENT, spaceAfter=8))
        elif stripped.startswith("### "): 
            after_name = False
            story.append(Spacer(1, 6)); story.append(Paragraph(md_to_rl(stripped[4:]), styles["sub"]))
        elif stripped.startswith(("- ", "• ", "* ")): 
            after_name = False
            story.append(Paragraph(f"• {md_to_rl(stripped[2:])}", styles["bullet"]))
        elif re.match(r"^\d+\.", stripped):
            after_name = False
            _num_text = md_to_rl(re.sub(r'^\d+\.', '', stripped).strip())
            story.append(Paragraph(f"• {_num_text}", styles["bullet"]))
        elif stripped.startswith("---"): 
            after_name = False
            story.append(HRFlowable(width="100%", thickness=0.5, color=C_RULE, spaceAfter=3))
        else: 
            if after_name:
                story.append(Paragraph(md_to_rl(stripped), styles["contact"]))
                after_name = False
            else:
                story.append(Paragraph(md_to_rl(stripped), styles["body"]))
    return story


# ==============================================================================
# PUBLIC API
# ==============================================================================

def export_pdf_bytes(md_text: str, candidate_name: str = "CV", template: str = "classic") -> bytes:
    _require_reportlab()
    from reportlab.platypus import SimpleDocTemplate
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = _build_pdf_styles(template)
    story = _story_from_mistune(md_text, styles) if _MISTUNE_AVAILABLE else _story_from_lines(md_text, styles)
    doc.build(story)
    return buf.getvalue()

def export_pdf_file(md_text: str, candidate_name: str, out_path: str, template: str = "classic") -> None:
    pdf_bytes = export_pdf_bytes(md_text, candidate_name, template)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(pdf_bytes)
