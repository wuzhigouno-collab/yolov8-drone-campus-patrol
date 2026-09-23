# -*- coding: utf-8 -*-
"""鹰眼巡校——B1 软著 60 页源代码文档导出脚本。

功能意图：
    将 src/ + tools/ 下全部 .py 源代码（plan.md 规定的软著代码量统计
    范围；reference/、tests/、archive/、weights/ 一律排除）按确定性规则
    排版导出为 Word 文档 copyright/源代码文档.docx，满足软著源代码文档
    的形式合规要求：

        - 页眉含系统全称"鹰眼巡校——无人机校园安全智能巡检系统 V1.0"；
        - 每页 >= 50 行代码（LINES_PER_PAGE）；
        - 页码位于页面右上角（PAGE 域 + 右对齐制表位）；
        - 总页数 = 60：代码超量（可排满页数 >= 60）时取前 30 页 + 后 30
          页连续拼接（软著惯例）；代码不足时全量导出且总页数 >= 30。

    文件排序规则（软著审阅惯例：入口在前、基础模块其次、工具脚本最后）：
        1. src/app.py —— GUI 入口模块置顶；
        2. src/ 其余模块按系统数据流依赖顺序排列：配置 -> 日志 -> 视频源
           -> 检测 -> 跟踪 -> 切片 -> 分区阈值 -> 多帧确认 -> 越线计数
           -> 徘徊检测 -> 告警归档 -> 热力图；
        3. tools/ 脚本按文件名升序排列；
        4. 上述清单之外的新增 .py 自动按文件名升序附于对应分组末尾，
           保证脚本对未来新增文件仍确定性可用。

    分页策略（确定性分页，不依赖 Word 自动排版）：
        全部源代码行按上述顺序拼接为连续行流，均分为 P 页
        （P = 总行数 // 50，余数依次分摊到前若干页，故每页 50 或 51 行，
        保证每页 >= 50 行），每页末尾插入手动分页符。版式参数（A4、四边
        2cm 页边距、正文 9pt、固定行距 13pt）下单页可容纳 56 个视觉行，
        大于单页上限（51 行代码 + 个别超长行折行），内容不会溢出到下
        一页，逻辑页数即 Word 渲染页数（--check 会用本机 Word COM 实测
        渲染页数复核，无 Word 环境时跳过并注明）。

    行数统计口径：按 Python splitlines() 统计逻辑行（含空行与注释行，
    与 wc -l 口径基本一致，仅对"文件末尾无换行"的文件多计 1 行）。

用法：
    python tools/export_code_doc.py            # 仅导出 copyright/源代码文档.docx
    python tools/export_code_doc.py --check    # 导出并自检，逐项输出 PASS/FAIL

退出码：0 = 导出成功且（--check 时）全部自检项 PASS；1 = 存在 FAIL 项。
"""

import os
import subprocess
import sys
import tempfile

from docx import Document
from docx.enum.text import WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

# 本文件位于 tools/ 下，直接运行时 sys.path 不含项目根目录，手动补上
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---------------- 软著形式要求常量 ----------------
DOC_TITLE = "鹰眼巡校——无人机校园安全智能巡检系统 V1.0"  # 页眉系统全称
LINES_PER_PAGE = 50      # 软著要求：每页不少于 50 行代码
TOTAL_PAGES = 60         # 软著惯例：源代码文档共 60 页
FRONT_PAGES = 30         # 超量时取前 30 页
BACK_PAGES = 30          # 超量时取后 30 页
MIN_TOTAL_PAGES = 30     # 不足 60 页时：全量导出且总页数不得少于 30

# 软著代码量统计范围（仅这两个目录下的 .py 进入导出与行数统计）
INCLUDE_DIRS = ("src", "tools")
# 明确排除的目录（仅用于自检交叉核对，证明无混入）
EXCLUDE_DIRS = ("reference", "tests", "archive", "weights")

# src/ 模块的固定排列顺序（入口在前、基础模块其次，见模块 docstring）
SRC_FIXED_ORDER = [
    "app.py",            # GUI 入口
    "config.py",         # 集中配置
    "logger.py",         # 日志
    "video_source.py",   # 视频源抽象
    "detector.py",       # YOLO 推理封装
    "tracker.py",        # ByteTrack 跟踪
    "slicer.py",         # 切片推理
    "zone_threshold.py", # 分区域阈值
    "frame_confirm.py",  # 多帧确认
    "line_counter.py",   # 越线计数
    "loitering.py",      # 徘徊检测
    "archiver.py",       # 告警归档
    "heatmap.py",        # 热力图
]

# 导出文件默认路径
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "copyright", "源代码文档.docx")


def collect_inventory(root):
    """收集统计范围内全部 .py 文件，按既定排序规则返回相对路径列表。

    排序规则：src/app.py 置顶 -> src/ 固定序模块 -> src/ 清单外文件升序
    -> tools/ 全部脚本按文件名升序。
    """
    inventory = []
    src_dir = os.path.join(root, "src")
    src_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".py"))
    ordered_src = [f for f in SRC_FIXED_ORDER if f in src_files]
    ordered_src += [f for f in src_files if f not in SRC_FIXED_ORDER]
    inventory += [os.path.join("src", f) for f in ordered_src]

    tools_dir = os.path.join(root, "tools")
    tools_files = sorted(f for f in os.listdir(tools_dir) if f.endswith(".py"))
    inventory += [os.path.join("tools", f) for f in tools_files]
    return inventory


def read_lines(root, rel_path):
    """读取单个源文件的逻辑行（splitlines 口径，含空行/注释行）。"""
    with open(os.path.join(root, rel_path), "r", encoding="utf-8") as f:
        return f.read().splitlines()


def load_corpus(root, inventory):
    """按清单顺序加载全部源代码行，返回 (逐文件行数, 连续行流)。"""
    per_file = []
    all_lines = []
    for rel in inventory:
        lines = read_lines(root, rel)
        per_file.append((rel, len(lines)))
        all_lines.extend(lines)
    return per_file, all_lines


def paginate(all_lines):
    """把连续行流均分为每页 >= LINES_PER_PAGE 的满页列表。

    页数 P = 总行数 // 50（整除），余数依次分摊到前若干页（每页多 1 行），
    因此每页恰为 50 或 51 行；极端情况（总行数 < 50*30）保底排 30 页，
    此时每页不足 50 行，--check 的"每页行数"项会如实 FAIL。
    """
    total = len(all_lines)
    if total >= LINES_PER_PAGE * MIN_TOTAL_PAGES:
        n_pages = total // LINES_PER_PAGE
    else:
        n_pages = MIN_TOTAL_PAGES  # 兜底分支：页数达标、行数如实暴露
    base, extra = divmod(total, n_pages)
    pages, idx = [], 0
    for i in range(n_pages):
        size = base + (1 if i < extra else 0)
        pages.append(all_lines[idx:idx + size])
        idx += size
    return pages


def select_pages(pages):
    """按软著惯例选取导出页，返回 (选中页列表, 适用分支说明)。"""
    if len(pages) >= TOTAL_PAGES:
        # 代码超量：取前 30 页 + 后 30 页，各自连续
        selected = pages[:FRONT_PAGES] + pages[len(pages) - BACK_PAGES:]
        branch = ("代码超量（可排满 %d 页 >= 60）：按软著惯例取前 %d 页 + "
                  "后 %d 页连续导出，共 60 页" % (len(pages), FRONT_PAGES, BACK_PAGES))
    else:
        # 代码不足：全量导出，总页数 >= 30
        selected = pages
        branch = ("代码不足 60 页容量：全量导出，共 %d 页（要求 >= 30 页）"
                  % len(pages))
    return selected, branch


def _set_run_font(run, size_pt):
    """统一设置 run 字体：西文等宽 Consolas，中文宋体（软著代码文档惯例）。"""
    run.font.name = "Consolas"
    run.font.size = Pt(size_pt)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:eastAsia"), "宋体")


def _format_code_paragraph(paragraph):
    """代码行段落版式：无段距、固定行距 13pt（9pt 字下单页容量 56 行）。"""
    pf = paragraph.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(13)


def _build_header(section):
    """页眉：左侧系统全称，右侧页码（PAGE 域），右对齐制表位定位。"""
    header = section.header
    paragraph = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
    paragraph.text = ""
    # 右对齐制表位设在版心右缘（页宽 21cm - 左右边距各 2cm = 17cm）
    paragraph.paragraph_format.tab_stops.add_tab_stop(Cm(17), WD_TAB_ALIGNMENT.RIGHT)
    paragraph.paragraph_format.space_after = Pt(0)

    title_run = paragraph.add_run(DOC_TITLE + "\t")
    _set_run_font(title_run, 9)

    # PAGE 域：Word 打开/重排版时自动显示当前页码
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), " PAGE ")
    fld_run = OxmlElement("w:r")
    fld_text = OxmlElement("w:t")
    fld_text.text = "1"
    fld_run.append(fld_text)
    fld.append(fld_run)
    paragraph._p.append(fld)


def build_document(pages, out_path):
    """按选中页构建 Word 文档：每页末尾手动分页符，页数即选中页数。"""
    doc = Document()
    section = doc.sections[0]
    # A4 页面、四边 2cm 页边距、页眉距边界 1.2cm
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    section.header_distance = Cm(1.2)
    _build_header(section)

    for page_idx, page_lines in enumerate(pages):
        for line in page_lines:
            p = doc.add_paragraph()
            _format_code_paragraph(p)
            run = p.add_run(line)
            _set_run_font(run, 9)
        # 页末插入手动分页符（附加在末行段落的 run 上，不占新行）
        if page_idx < len(pages) - 1:
            doc.paragraphs[-1].runs[-1].add_break(WD_BREAK.PAGE)

    doc.core_properties.title = DOC_TITLE
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc.save(out_path)
    return out_path


def word_render_page_count(docx_path):
    """调用本机 Word COM 实测渲染页数；无 Word 环境或调用失败返回 None。

    通过临时 .ps1（UTF-8 BOM，保证中文路径正确解析）驱动 Word 打开文档
    并执行 ComputeStatistics(wdStatisticPages=2) 触发真实重排版。
    """
    ps_script = (
        "$ErrorActionPreference = 'Stop'\n"
        "$path = '%s'\n"
        "$w = New-Object -ComObject Word.Application\n"
        "$w.Visible = $false\n"
        "try {\n"
        "    $d = $w.Documents.Open($path, $false, $true)\n"
        "    $n = $d.ComputeStatistics(2)\n"
        "    $d.Close($false)\n"
        "    Write-Output $n\n"
        "} finally {\n"
        "    $w.Quit()\n"
        "}\n" % str(docx_path).replace("'", "''")
    )
    fd, tmp = tempfile.mkstemp(suffix=".ps1")
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8-sig") as f:
            f.write(ps_script)
        r = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", tmp],
            capture_output=True, text=True, timeout=180)
        out = (r.stdout or "").strip().splitlines()
        if r.returncode == 0 and out:
            return int(out[-1])
        return None
    except Exception:
        return None
    finally:
        os.unlink(tmp)  # 临时脚本用完即删，不混入交付物


def run_checks(root, out_path, inventory, per_file, pages, selected, branch):
    """逐项自检并打印 PASS/FAIL，返回是否全部通过。"""
    results = []

    def record(name, ok, detail):
        results.append(ok)
        print("[%s] %s：%s" % ("PASS" if ok else "FAIL", name, detail))

    # 1) 行数统计范围：仅 src/ + tools/，且排除目录无任何文件混入清单
    in_scope = all(rel.split(os.sep)[0] in INCLUDE_DIRS for rel in inventory)
    inv_set = set(os.path.normpath(rel) for rel in inventory)
    excluded_count = 0
    leaked = []  # 排除目录下却混入统计清单的 .py（正常应为空）
    for ex in EXCLUDE_DIRS:
        ex_dir = os.path.join(root, ex)
        if not os.path.isdir(ex_dir):
            continue
        for dirpath, _dirnames, filenames in os.walk(ex_dir):
            for fn in filenames:
                if fn.endswith(".py"):
                    excluded_count += 1
                    rel = os.path.relpath(os.path.join(dirpath, fn), root)
                    if os.path.normpath(rel) in inv_set:
                        leaked.append(rel)
    print("  统计范围逐文件行数清单：")
    for rel, n in per_file:
        print("    %6d 行  %s" % (n, rel.replace(os.sep, "/")))
    record("行数统计范围", in_scope and not leaked,
           "仅含 src/ + tools/ 共 %d 个 .py；reference/tests/archive/weights "
           "下共 %d 个 .py，混入清单 %d 个" % (len(inventory), excluded_count, len(leaked)))

    # 2) 总行数与适用分支
    total = sum(n for _rel, n in per_file)
    record("总行数与排版分支", len(pages) >= MIN_TOTAL_PAGES,
           "src/ + tools/ 实测总行数 %d，可排满页数 %d；%s" % (total, len(pages), branch))
    if not 2500 <= total <= 4000:
        print("  说明：总行数 %d 超出 plan.md 目标区间 2500-4000（阶段 A 定型"
              "代码现状），本任务按超量分支规则排版，不影响本项判定。" % total)

    # 3) 每页行数：全部导出页 >= 50 行
    page_sizes = [len(p) for p in selected]
    record("每页行数", min(page_sizes) >= LINES_PER_PAGE,
           "导出页逐页行数最小 %d、最大 %d（要求每页 >= %d）"
           % (min(page_sizes), max(page_sizes), LINES_PER_PAGE))

    # 4) 总页数：逻辑分页 +（有 Word 时）实测渲染页数一致
    logical_ok = (len(selected) == TOTAL_PAGES) if len(pages) >= TOTAL_PAGES \
        else (len(selected) >= MIN_TOTAL_PAGES)
    rendered = word_render_page_count(out_path)
    if rendered is None:
        record("总页数", logical_ok,
               "逻辑分页 %d 页（要求 %s）；本机无 Word，渲染页数以确定性分页"
               "逻辑为准" % (len(selected), "= 60" if len(pages) >= TOTAL_PAGES else ">= 30"))
    else:
        record("总页数", logical_ok and rendered == len(selected),
               "逻辑分页 %d 页，Word 实测渲染页数 %d 页，二者一致"
               % (len(selected), rendered))

    # 5) 页眉含系统全称
    doc = Document(out_path)
    header = doc.sections[0].header
    header_text = "".join(p.text for p in header.paragraphs)
    record("页眉", DOC_TITLE in header_text, "页眉文字：%s" % header_text.strip())

    # 6) 页码位于右上角：PAGE 域存在 + 右对齐制表位
    has_page_field = any(
        "PAGE" in (fld.get(qn("w:instr")) or "")
        for fld in header._element.iter(qn("w:fldSimple")))
    has_right_tab = any(
        tab.get(qn("w:val")) == "right"
        for p in header.paragraphs
        for tab in p._p.iter(qn("w:tab")))
    record("页码右上角", has_page_field and has_right_tab,
           "PAGE 域 %s，右对齐制表位 %s"
           % ("存在" if has_page_field else "缺失",
              "存在" if has_right_tab else "缺失"))

    # 7) 内容一致性：导出正文逐行与统计范围内源代码一致（无外部内容混入）
    body_lines = [p.text for p in doc.paragraphs]
    expected = [line for page in selected for line in page]
    record("内容一致性", body_lines == expected,
           "导出正文 %d 行与 src/ + tools/ 源代码逐行比对%s"
           % (len(body_lines), "完全一致" if body_lines == expected else "存在差异"))

    # 8) 分页符数量：导出页间手动分页符 = 页数 - 1
    n_breaks = sum(1 for br in doc.element.body.iter(qn("w:br"))
                   if br.get(qn("w:type")) == "page")
    record("分页符", n_breaks == len(selected) - 1,
           "手动分页符 %d 个（页数 %d - 1）" % (n_breaks, len(selected)))

    # 9) 导出文件存在且非空
    exists = os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    record("导出文件", exists,
           "%s（%d 字节）" % (out_path, os.path.getsize(out_path) if exists else 0))

    return all(results)


def main():
    check_mode = "--check" in sys.argv[1:]
    out_path = DEFAULT_OUT

    inventory = collect_inventory(PROJECT_ROOT)
    per_file, all_lines = load_corpus(PROJECT_ROOT, inventory)
    pages = paginate(all_lines)
    selected, branch = select_pages(pages)
    build_document(selected, out_path)
    print("导出完成：%s（%s）" % (out_path, branch))

    if check_mode:
        print("---- 合规自检 ----")
        ok = run_checks(PROJECT_ROOT, out_path, inventory, per_file,
                        pages, selected, branch)
        print("---- 自检结论：%s ----" % ("全部 PASS" if ok else "存在 FAIL"))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
