# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

操作手册 docx 导出脚本（B2）。

功能意图：
    把 docs/操作手册.md（Markdown 源文件）真实转换为软著提交用的
    copyright/操作手册.docx。脚本自带一个轻量 Markdown 解析器，
    支持本手册用到的全部语法：标题（#/##/###）、普通段落、有序/
    无序列表、管道表格、 fenced 代码块、行内 **加粗** 与 `代码`、
    图片引用 ![图注](相对路径)——图片按手册所在目录解析为绝对路径
    后真实嵌入 docx（宽 14.5cm 居中，图注小字居中）。

    版式约定（软著材料风格，与 60 页源代码文档一致）：
        - 页眉左侧为"鹰眼巡校——无人机校园安全智能巡检系统 V1.0 操作手册"，
          右侧为 PAGE 页码域（Word 打开/重排版时自动显示页码）；
        - 每个一级章节（##）前手动分页符，做确定性分页；
        - 正文中文宋体小四、代码 Consolas 9pt。

用法示例：
    python tools/export_manual_docx.py            # 转换并自检
    python tools/export_manual_docx.py --check    # 只对已生成 docx 做结构自检
"""

import argparse
import os
import re
import sys

# 脚本位于 tools/ 下，直接运行时 sys.path 不含项目根目录，这里手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt

from src.logger import get_logger
from src.config import Config

# 手册源文件与导出目标（相对项目根目录）
MANUAL_MD = os.path.join("docs", "操作手册.md")
OUTPUT_DOCX = os.path.join("copyright", "操作手册.docx")

# 页眉文字与版式参数
HEADER_TITLE = "鹰眼巡校——无人机校园安全智能巡检系统 V1.0 操作手册"
IMAGE_WIDTH_CM = 14.5     # 版心内图片宽度（A4 默认页边距下版心约 15.9cm）
BODY_FONT_SIZE = Pt(12)   # 正文小四
CODE_FONT_SIZE = Pt(9)    # 代码块字号

# 自检达标线（验收标准：图片嵌入数 >= 10）
MIN_EMBEDDED_IMAGES = 10

# 行内样式切分：**加粗** 与 `代码`
INLINE_RE = re.compile(r"(\*\*.+?\*\*|`.+?`)")
# 图片引用：![图注](路径)
IMAGE_RE = re.compile(r"^!\[(.*?)\]\((.+?)\)\s*$")
# 表格行：以 | 起始
TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
# 有序列表：数字. 开头
ORDERED_RE = re.compile(r"^\d+\.\s+(.*)$")


def _set_cjk_font(run, ascii_font, cjk_font, size, bold=False):
    """统一设置 run 的中西文字体与字号（python-docx 中文字体需设 eastAsia）。"""
    run.font.name = ascii_font
    run.font.size = size
    run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), cjk_font)


def _add_runs(paragraph, text, base_size=BODY_FONT_SIZE, base_bold=False):
    """把一段含行内标记的文本按 **加粗** / `代码` 切分为多个 run 写入段落。"""
    for token in INLINE_RE.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            run = paragraph.add_run(token[2:-2])
            _set_cjk_font(run, "Times New Roman", "宋体", base_size, bold=True)
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1])
            _set_cjk_font(run, "Consolas", "宋体", base_size,
                          bold=base_bold)
        else:
            run = paragraph.add_run(token)
            _set_cjk_font(run, "Times New Roman", "宋体", base_size,
                          bold=base_bold)


def _build_header(section):
    """页眉：左侧手册名，右侧 PAGE 页码域（右对齐制表位定位）。"""
    header = section.header
    paragraph = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
    paragraph.text = ""
    run = paragraph.add_run(HEADER_TITLE)
    _set_cjk_font(run, "Times New Roman", "宋体", Pt(9))
    # 右对齐制表位（版心右缘），页码域紧随其后
    tabs = paragraph.paragraph_format.tab_stops
    tabs.add_tab_stop(Cm(15.9), WD_ALIGN_PARAGRAPH.RIGHT)
    paragraph.add_run("\t")
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), " PAGE ")
    paragraph._element.append(fld)


def _add_code_block(doc, lines):
    """代码块：逐行等宽字体段落，前后紧凑。"""
    for line in lines:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(line if line else " ")
        _set_cjk_font(run, "Consolas", "宋体", CODE_FONT_SIZE)


def _add_table(doc, rows):
    """把管道表格行（含分隔行）渲染为 docx 表格，首行作表头加粗。"""
    # 去掉分隔行（|---|---|）
    rows = [r for r in rows if not re.match(r"^\|[\s:\-|]+\|$", r)]
    if not rows:
        return
    cells = [[c.strip() for c in TABLE_ROW_RE.match(r).group(1).split("|")]
             for r in rows]
    table = doc.add_table(rows=len(cells), cols=len(cells[0]))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(cells):
        for j, text in enumerate(row):
            cell = table.cell(i, j)
            p = cell.paragraphs[0]
            _add_runs(p, text, base_size=Pt(10.5), base_bold=(i == 0))


def _add_picture(doc, alt, rel_path, md_dir, logger):
    """嵌入图片：相对路径按手册所在目录解析；文件不存在时报错（不静默跳过，
    保证"手册引用与真实产物一一对应"可被自检发现）。"""
    abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(
        md_dir, rel_path)
    abs_path = os.path.normpath(abs_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError("手册引用的图片不存在: %s" % abs_path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(abs_path, width=Cm(IMAGE_WIDTH_CM))
    logger.info("嵌入图片: %s", abs_path)
    if alt:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_runs(cap, alt, base_size=Pt(9))


def convert(md_path=MANUAL_MD, out_path=OUTPUT_DOCX):
    """执行 Markdown -> docx 转换，返回导出文件绝对路径。"""
    config = Config()
    logger = get_logger(__name__, config)
    root = config.project_root()
    md_abs = os.path.join(root, md_path)
    out_abs = os.path.join(root, out_path)
    md_dir = os.path.dirname(md_abs)
    with open(md_abs, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    doc = Document()
    _build_header(doc.sections[0])

    in_code = False
    code_buf = []
    table_buf = []
    started = False  # 是否已有正文内容（首个 ## 前不加分页符）
    image_count = 0

    def flush_table():
        if table_buf:
            _add_table(doc, table_buf)
            table_buf.clear()

    for raw in lines:
        line = raw.rstrip()
        # fenced 代码块进出
        if line.strip().startswith("```"):
            if in_code:
                _add_code_block(doc, code_buf)
                code_buf = []
                in_code = False
            else:
                flush_table()
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue
        # 表格行累积
        if TABLE_ROW_RE.match(line):
            table_buf.append(line)
            continue
        flush_table()
        # 空行与分隔线
        if not line.strip() or line.strip() == "---":
            continue
        # 标题
        if line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=2)
            started = True
            continue
        if line.startswith("## "):
            if started:
                doc.add_page_break()  # 确定性分页：每个一级章节另起一页
            doc.add_heading(line[3:].strip(), level=1)
            started = True
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=0)
            started = True
            continue
        # 图片引用
        m = IMAGE_RE.match(line.strip())
        if m:
            _add_picture(doc, m.group(1), m.group(2), md_dir, logger)
            image_count += 1
            continue
        # 列表
        if line.lstrip().startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            _add_runs(p, line.lstrip()[2:])
            continue
        m = ORDERED_RE.match(line.strip())
        if m:
            p = doc.add_paragraph(style="List Number")
            _add_runs(p, m.group(1))
            continue
        # 普通段落
        p = doc.add_paragraph()
        _add_runs(p, line)
    flush_table()

    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    doc.save(out_abs)
    logger.info("操作手册已导出: %s（嵌入图片 %d 张）", out_abs, image_count)
    return out_abs, image_count


def check(out_path=OUTPUT_DOCX):
    """结构自检：重开生成的 docx，统计段落/表格/图片/逻辑页数并判定达标项。

    docx 为流式格式，脚本侧以"手动分页符数 + 1"作为逻辑页数下限；
    真实渲染页数可用 Word COM ComputeStatistics(2) 交叉核对（见验收记录）。
    """
    config = Config()
    root = config.project_root()
    out_abs = os.path.join(root, out_path)
    doc = Document(out_abs)
    n_paragraphs = len(doc.paragraphs)
    n_tables = len(doc.tables)
    n_images = len(doc.inline_shapes)
    n_headings = sum(1 for p in doc.paragraphs
                     if p.style.name.startswith("Heading") or p.style.name == "Title")
    # 手动分页符计数（w:br w:type="page"）
    n_breaks = len(doc.element.body.findall(
        ".//" + qn("w:br") + "[@" + qn("w:type") + "='page']"))
    logical_pages = n_breaks + 1
    print("docx 结构自检: %s" % out_abs)
    print("  段落数: %d" % n_paragraphs)
    print("  表格数: %d" % n_tables)
    print("  标题数: %d" % n_headings)
    print("  嵌入图片数: %d（达标线 >= %d） -> %s" % (
        n_images, MIN_EMBEDDED_IMAGES,
        "PASS" if n_images >= MIN_EMBEDDED_IMAGES else "FAIL"))
    print("  手动分页符: %d（逻辑页数下限 %d）" % (n_breaks, logical_pages))
    ok = n_images >= MIN_EMBEDDED_IMAGES
    print("  自检结论: %s" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    parser = argparse.ArgumentParser(description="操作手册 Markdown -> docx 导出")
    parser.add_argument("--check", action="store_true",
                        help="只对已生成的 docx 做结构自检，不重新转换")
    args = parser.parse_args()
    if args.check:
        return 0 if check() else 1
    convert()
    return 0 if check() else 1


if __name__ == "__main__":
    sys.exit(main())
