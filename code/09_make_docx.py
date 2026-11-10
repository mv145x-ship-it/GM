"""Markdown → DOCX через pandoc с настроенными стилями и постобработкой таблиц.

* формулы LaTeX преобразуются pandoc в нативные формулы Word (OMML);
* шаблон стилей paper/reference.docx генерируется из стандартного шаблона pandoc
  (Times New Roman 12 pt, интервал 1,15, поля 2 см, нумерация страниц);
* таблицы получают «трёхлинейное» оформление (верхняя и нижняя линии, линия под шапкой).

Запуск:  python code/09_make_docx.py --md paper/article_ru.md --out paper/article_ru.docx
"""
from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from gravity_common import PAPER

FONT = "Times New Roman"


def set_style_font(style, size=None, bold=None, italic=None, color=(0, 0, 0), name=FONT):
    f = style.font
    f.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for att in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(att), name)
    for att in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rfonts.get(qn(att)) is not None:
            del rfonts.attrib[qn(att)]
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color is not None:
        f.color.rgb = RGBColor(*color)


def build_reference(path: Path) -> None:
    subprocess.run(["pandoc", "-o", str(path), "--print-default-data-file", "reference.docx"], check=True)
    doc = Document(str(path))
    st = {s.name: s for s in doc.styles}   # python-docx не находит часть стилей pandoc по имени через __getitem__
    for name in ["Normal", "Body Text", "First Paragraph", "Compact", "Block Text", "Footnote Text", "Bibliography"]:
        if name in st:
            s = st[name]
            set_style_font(s, size=12)
            pf = s.paragraph_format
            pf.line_spacing = 1.15
            pf.space_before = Pt(0)
            pf.space_after = Pt(4)
            if name in ("Body Text", "First Paragraph"):
                pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                pf.first_line_indent = Cm(1.0)
            if name == "Compact":
                pf.space_after = Pt(2)
                pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    set_style_font(st["Footnote Text"], size=10)
    cfg = {"Title": (16, True, WD_ALIGN_PARAGRAPH.CENTER, 6, 6), "Subtitle": (12, False, WD_ALIGN_PARAGRAPH.CENTER, 2, 6),
           "Author": (12, False, WD_ALIGN_PARAGRAPH.CENTER, 2, 2), "Date": (11, False, WD_ALIGN_PARAGRAPH.CENTER, 2, 2),
           "Abstract": (11, False, WD_ALIGN_PARAGRAPH.JUSTIFY, 2, 4), "Heading 1": (13, True, WD_ALIGN_PARAGRAPH.LEFT, 14, 6),
           "Heading 2": (12, True, WD_ALIGN_PARAGRAPH.LEFT, 10, 4), "Heading 3": (12, False, WD_ALIGN_PARAGRAPH.LEFT, 8, 3),
           "Heading 4": (12, False, WD_ALIGN_PARAGRAPH.LEFT, 6, 2)}
    for name, (size, bold, al, sb, sa) in cfg.items():
        if name in st:
            s = st[name]
            set_style_font(s, size=size, bold=bold, italic=(name == "Heading 3"))
            s.paragraph_format.alignment = al
            s.paragraph_format.space_before = Pt(sb)
            s.paragraph_format.space_after = Pt(sa)
            s.paragraph_format.first_line_indent = Cm(0)
            s.paragraph_format.keep_with_next = True
    for name, al in [("Table Caption", WD_ALIGN_PARAGRAPH.LEFT), ("Image Caption", WD_ALIGN_PARAGRAPH.CENTER)]:
        if name in st:
            s = st[name]
            set_style_font(s, size=10.5, italic=False, bold=False)
            s.paragraph_format.alignment = al
            s.paragraph_format.first_line_indent = Cm(0)
            s.paragraph_format.space_before = Pt(6)
            s.paragraph_format.space_after = Pt(6)
            s.paragraph_format.keep_with_next = (name == "Table Caption")
    if "Captioned Figure" in st:
        s = st["Captioned Figure"]
        s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        s.paragraph_format.first_line_indent = Cm(0)
        s.paragraph_format.keep_with_next = True
    if "Figure" in st:
        st["Figure"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        st["Figure"].paragraph_format.first_line_indent = Cm(0)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    doc.save(str(path))


def set_cell_border(cell, **kwargs):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge in ("top", "left", "bottom", "right"):
        spec = kwargs.get(edge)
        if spec is None:
            continue
        el = borders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}")
            borders.append(el)
        for k, val in spec.items():
            el.set(qn(f"w:{k}"), str(val))


def format_tables(doc: Document, font_size: float = 8.5) -> None:
    for tbl in doc.tables:
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        # убрать стилевое оформление pandoc, задать собственные границы
        tblPr = tbl._tbl.tblPr
        for el in tblPr.findall(qn("w:tblBorders")):
            tblPr.remove(el)
        nrows = len(tbl.rows)
        # шапка повторяется на каждой странице
        trPr = tbl.rows[0]._tr.get_or_add_trPr()
        hdr = OxmlElement("w:tblHeader")
        hdr.set(qn("w:val"), "true")
        trPr.append(hdr)
        for ri, row in enumerate(tbl.rows):
            cant = OxmlElement("w:cantSplit")
            row._tr.get_or_add_trPr().append(cant)
            for ci, cell in enumerate(row.cells):
                top = {"val": "single", "sz": 8, "color": "000000"} if ri == 0 else None
                bottom = None
                if ri == 0:
                    bottom = {"val": "single", "sz": 4, "color": "000000"}
                if ri == nrows - 1:
                    bottom = {"val": "single", "sz": 8, "color": "000000"}
                set_cell_border(cell, top=top, bottom=bottom, left={"val": "nil"}, right={"val": "nil"})
                tcPr = cell._tc.get_or_add_tcPr()
                mar = OxmlElement("w:tcMar")
                for edge, w in (("top", 30), ("left", 50), ("bottom", 30), ("right", 50)):   # порядок по схеме OOXML
                    e = OxmlElement(f"w:{edge}")
                    e.set(qn("w:w"), str(w)); e.set(qn("w:type"), "dxa")
                    mar.append(e)
                tcPr.append(mar)
                for p in cell.paragraphs:
                    # формулы (OMML) в ячейках: размер шрифта как у текста таблицы
                    for mr in p._p.iter(qn("m:r")):
                        rpr = mr.find(qn("w:rPr"))
                        if rpr is None:
                            rpr = OxmlElement("w:rPr")
                            mrpr = mr.find(qn("m:rPr"))      # порядок по схеме: m:rPr, затем w:rPr
                            if mrpr is not None:
                                mrpr.addnext(rpr)
                            else:
                                mr.insert(0, rpr)
                        for tag in ("w:sz", "w:szCs"):
                            el = rpr.find(qn(tag))
                            if el is None:
                                el = OxmlElement(tag)
                                rpr.append(el)
                            el.set(qn("w:val"), str(int(font_size * 2)))
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.first_line_indent = Cm(0)
                    p.paragraph_format.line_spacing = 1.0
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:
                        run.font.size = Pt(font_size)
                        run.font.name = FONT
                        run.font.bold = True if ri == 0 else run.font.bold


def add_page_numbers(doc: Document) -> None:
    for sec in doc.sections:
        p = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        run = p.add_run()
        for typ, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            if typ:
                fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), typ); run._r.append(fc)
            else:
                it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; run._r.append(it)
        run.font.size = Pt(10)
        run.font.name = FONT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--font-size", type=float, default=8.5)
    args = ap.parse_args()
    ref = PAPER / "reference.docx"
    build_reference(ref)
    cmd = ["pandoc", args.md, "-o", args.out, "--reference-doc", str(ref), "-f",
           "markdown-smart+tex_math_dollars+pipe_tables+fenced_divs+bracketed_spans+yaml_metadata_block+raw_attribute",
           "--resource-path", str(PAPER), "--wrap=none"]
    subprocess.run(cmd, check=True)
    doc = Document(args.out)
    format_tables(doc, args.font_size)
    add_page_numbers(doc)
    doc.save(args.out)
    print("DOCX:", args.out)


if __name__ == "__main__":
    main()
