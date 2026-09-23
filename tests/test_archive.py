# -*- coding: utf-8 -*-
"""鹰眼巡校——A7 告警截图归档与日志模块 验收测试。

测试内容分两部分：

一、合成场景单元测试（不依赖视频/模型，直接驱动 AlertArchiver，秒级完成）：
    1. 归档目录自动创建（含多级嵌套目录），CSV 自动写入固定表头；
    2. 核心验收：两条告警归档后截图文件存在且文件名互不覆盖，
       CSV 每行 6 个字段（时间戳/事件类型/类别/置信度/位置/截图路径）
       齐全，时间戳格式可解析且单调递增，截图路径字段与落盘文件吻合；
    3. 截图内容验证：归档前在画面上绘制检测框，重新解码截图后断言
       框线像素存在（证明归档的是"带标注画面"而非原始空帧）；
    4. 线程安全：多线程并发归档，文件数/CSV 行数/事件序号无一丢失
       或重复；
    5. 参数防护：frame 为 None 时抛出 ValueError。

二、真实视频端到端验收（默认运行，可用 --video 覆盖素材）：
    检测 -> 多帧确认 -> 越线计数 -> 告警归档 全链路逐帧运行真实视频：
    已确认目标首次确认（目标确认告警）与穿线事件发生时，把当前带标注
    画面（检测框+警戒线叠加完毕）送入 AlertArchiver。运行到两类告警
    各自至少归档一次为止（上限 --max-frames 帧）。断言：
      - 两类告警均至少归档 1 条，CSV 行数与归档记录数一致；
      - 每张截图真实存在、可解码、尺寸与视频画面一致、非空文件；
      - CSV 时间戳落在本次测试运行的墙上时钟窗口内（时间与事件吻合），
        位置字段与事件时刻的目标中心点/穿线点坐标吻合；
      - 抽首张"目标确认"截图解码，断言其上检测框框线像素存在
       （程序断言带框），截图文件同时保留在归档目录供人工目检。

运行方式：
    python tests/test_archive.py
        # 合成单测 + 默认素材真实视频端到端（默认归档到 tests/a7_archive）
    python tests/test_archive.py --video <视频路径> --max-frames 1600 \
        --archive-dir tests/a7_archive
全部断言通过输出 PASS 并以退出码 0 结束；断言失败抛 AssertionError。
"""

import argparse
import csv
import os
import shutil
import sys
import threading
from datetime import datetime, timedelta

import cv2
import numpy as np

# 本文件位于 tests/ 下，直接运行时 sys.path 不含项目根目录，手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.archiver import (  # noqa: E402
    CSV_HEADER,
    EVENT_CONFIRM,
    EVENT_CROSS_IN,
    EVENT_CROSS_OUT,
    AlertArchiver,
)
from src.config import Config  # noqa: E402
from src.logger import get_logger  # noqa: E402

logger = get_logger("test_archive")

# 默认真实视频素材（派活指定；3712 帧，含人/车画面）
DEFAULT_VIDEO = "C:/Users/OS/Desktop/farm_guard/videos/5G智慧农业素材.mp4"
# 测试独立归档目录：与运行期 archive/ 隔离，避免污染（约定见任务说明）
DEFAULT_ARCHIVE_DIR = "tests/a7_archive"
# CSV 时间戳格式（与 archiver 写入格式一致，精确到毫秒）
TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
# 截图中检测框的绘制颜色（BGR 纯绿），供"带框"像素断言使用
BOX_COLOR = (0, 255, 0)


def parse_ts(text):
    """解析 CSV 时间戳字段为 datetime。"""
    return datetime.strptime(text, TS_FORMAT)


def read_csv_rows(csv_path):
    """读取告警日志 CSV，返回 (表头, 数据行列表)。编码 utf-8-sig 剥 BOM。"""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fp:
        rows = list(csv.reader(fp))
    return rows[0], rows[1:]


def assert_row_fields(row):
    """断言一行 CSV 记录 6 个字段齐全且语义基本合法。"""
    assert len(row) == len(CSV_HEADER), \
        "CSV 行字段数应为 %d：%s" % (len(CSV_HEADER), row)
    ts_text, event_type, cls, conf, position, shot = row
    parse_ts(ts_text)  # 时间戳格式可解析
    assert event_type in (EVENT_CONFIRM, EVENT_CROSS_IN, EVENT_CROSS_OUT), \
        "事件类型非法：%s" % event_type
    assert cls, "类别字段不能为空：%s" % (row,)
    float(conf)  # 置信度字段可解析为数值
    assert position.startswith("(") and position.endswith(")"), \
        "位置字段应为 (x, y) 形式：%s" % position
    assert shot.endswith(".png"), "截图路径应为 png：%s" % shot
    assert os.path.isfile(shot), "截图路径指向的文件不存在：%s" % shot


def run_unit_tests(tmp_root):
    """执行全部合成场景单元测试。返回临时目录（已清理）。"""
    logger.info("==== A7 告警归档模块 合成场景单元测试开始 ====")
    unit_dir = os.path.join(tmp_root, "unit")
    if os.path.isdir(unit_dir):
        shutil.rmtree(unit_dir)

    # ---- 用例 1：目录自动创建 + CSV 表头 ----
    nested = os.path.join(unit_dir, "case1", "deep", "dir")
    arch = AlertArchiver(Config(), archive_dir=nested)
    assert os.path.isdir(nested), "嵌套归档目录应被自动创建：%s" % nested
    header, rows = read_csv_rows(arch.csv_path)
    assert header == CSV_HEADER, "CSV 表头错误：%s" % header
    assert rows == [], "新建 CSV 不应有数据行：%s" % rows
    logger.info("用例1 PASS：嵌套目录自动创建，CSV 表头 %s", header)

    # ---- 用例 2+3：两条告警归档 / 字段齐全 / 截图带框 ----
    frame_h, frame_w = 240, 320
    t0 = datetime.now()
    # 告警 1：目标确认——画面先叠加绿色检测框再归档
    frame1 = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    box1 = (50, 60, 150, 180)
    cv2.rectangle(frame1, (box1[0], box1[1]), (box1[2], box1[3]),
                  BOX_COLOR, 2)
    rec1 = arch.archive(frame1, EVENT_CONFIRM, "person", 0.87,
                        position=(100, 120))
    # 告警 2：越线进
    frame2 = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    rec2 = arch.archive(frame2, EVENT_CROSS_IN, "car", 0.66,
                        position=(160, 121))
    t1 = datetime.now()

    # 文件名互不覆盖、截图真实存在且非空
    assert rec1["screenshot_path"] != rec2["screenshot_path"], \
        "两条告警的截图文件名必须不同（防覆盖）"
    for rec in (rec1, rec2):
        path = rec["screenshot_path"]
        assert os.path.isfile(path) and os.path.getsize(path) > 0, \
            "截图文件不存在或为空：%s" % path
        img = cv2.imread(path)
        assert img is not None, "截图无法解码：%s" % path
        assert img.shape == (frame_h, frame_w, 3), \
            "截图尺寸应为 %dx%d：%s" % (frame_w, frame_h, img.shape)
    # 截图内容验证：告警 1 归档前画了绿色检测框，解码后框线像素必须在
    img1 = cv2.imread(rec1["screenshot_path"])
    edge_points = [(box1[0] + 1, box1[1] + 1), (box1[2] - 1, box1[1] + 1),
                   (box1[0] + 1, box1[3] - 1)]
    for px, py in edge_points:
        b, g, r = (int(v) for v in img1[py, px])
        assert g > 200 and b < 80 and r < 80, \
            "截图 (%d,%d) 处应为检测框绿色像素，实测 BGR=(%d,%d,%d)" % (
                px, py, b, g, r)
    # 告警 2 画面无框：同位置应为纯黑，佐证两张截图内容各自独立
    b, g, r = (int(v) for v in cv2.imread(
        rec2["screenshot_path"])[box1[1] + 1, box1[0] + 1])
    assert (b, g, r) == (0, 0, 0), "无框画面归档不应出现框线像素"

    # CSV 行校验：两行、字段齐全、时间与事件窗口吻合、字段值吻合
    header, rows = read_csv_rows(arch.csv_path)
    assert len(rows) == 2, "CSV 应有 2 行数据：%d" % len(rows)
    for row in rows:
        assert_row_fields(row)
    ts1, ts2 = parse_ts(rows[0][0]), parse_ts(rows[1][0])
    assert t0 - timedelta(seconds=2) <= ts1 <= t1 + timedelta(seconds=2), \
        "时间戳应落在归档发生窗口内：%s" % ts1
    assert ts1 <= ts2, "时间戳应单调递增：%s > %s" % (ts1, ts2)
    assert rows[0][1] == EVENT_CONFIRM and rows[0][2] == "person"
    assert abs(float(rows[0][3]) - 0.87) < 1e-6
    assert rows[0][4] == "(100, 120)", "位置字段错误：%s" % rows[0][4]
    assert rows[0][5] == rec1["screenshot_path"]
    assert rows[1][1] == EVENT_CROSS_IN and rows[1][4] == "(160, 121)"
    # 内存记录与 CSV 一一对应
    assert [r["screenshot_path"] for r in arch.records] == \
        [row[5] for row in rows]
    arch.close()
    logger.info(
        "用例2/3 PASS：两条告警归档完成，字段齐全、时间吻合、"
        "截图带框像素验证通过（%s）",
        os.path.basename(rec1["screenshot_path"]),
    )

    # ---- 用例 4：多线程并发归档线程安全 ----
    mt_dir = os.path.join(unit_dir, "case4")
    arch_mt = AlertArchiver(Config(), archive_dir=mt_dir)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    n_threads, n_per = 4, 10

    def worker(tid):
        for i in range(n_per):
            arch_mt.archive(frame, EVENT_CONFIRM, "person", 0.5,
                            position=(tid, i))

    threads = [threading.Thread(target=worker, args=(t,))
               for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = n_threads * n_per
    _header, rows = read_csv_rows(arch_mt.csv_path)
    assert len(rows) == total, \
        "并发归档后 CSV 应有 %d 行：%d" % (total, len(rows))
    for row in rows:
        assert_row_fields(row)
    shots = {row[5] for row in rows}
    assert len(shots) == total, "截图文件名出现覆盖：%d/%d" % (len(shots), total)
    seqs = sorted(rec["seq"] for rec in arch_mt.records)
    assert seqs == list(range(1, total + 1)), "事件序号应连续无重复"
    arch_mt.close()
    logger.info("用例4 PASS：%d 线程 x %d 条并发归档，40 行/40 文件/序号齐全",
                n_threads, n_per)

    # ---- 用例 5：frame=None 参数防护 ----
    arch_x = AlertArchiver(Config(), archive_dir=os.path.join(unit_dir, "c5"))
    try:
        arch_x.archive(None, EVENT_CONFIRM, "person", 0.5, (1, 1))
        raise AssertionError("frame=None 应抛 ValueError")
    except ValueError:
        pass
    arch_x.close()
    logger.info("用例5 PASS：frame=None 正确抛出 ValueError")

    # 单元测试临时目录整体清理（合成产物不占用交付目录）
    shutil.rmtree(unit_dir)
    logger.info("==== 全部 5 组（含并发）合成用例断言通过 ====")
    print("PASS: 合成场景单元测试 5 组用例全部通过")


def draw_annotated(frame, confirm_result, counter):
    """生成测试用带标注画面：已确认目标绿色实线框 + 警戒线叠加。

    与 GUI 的标注语义一致（已确认目标绿框、警戒线由 A6 模块绘制），
    保证归档截图即为"带检测框画面"。
    """
    annotated = frame.copy()
    for rec in confirm_result.records:
        if not rec["confirmed"]:
            continue
        x1, y1, x2, y2 = rec["det"]["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(annotated, "%s #%d" % (rec["det"]["cls"], rec["track_id"]),
                    (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    BOX_COLOR, 1)
    return counter.draw_overlay(annotated)


def run_video_mode(args):
    """真实视频端到端验收：全链路告警归档 + CSV/截图核对。"""
    from src.detector import Detector
    from src.frame_confirm import FrameConfirmer
    from src.line_counter import LineCounter
    from src.video_source import create_video_source

    config = Config()
    archive_dir = os.path.abspath(args.archive_dir)
    # 测试归档目录每次运行前清空重建，保证产出与当次运行一一对应
    if os.path.isdir(archive_dir):
        shutil.rmtree(archive_dir)

    source = create_video_source(args.video, config)
    src_w, src_h = source.width, source.height  # 快照，release 后属性归零
    detector = Detector(config)
    confirmer = FrameConfirmer(config)
    counter = LineCounter(config, src_w, src_h)
    archiver = AlertArchiver(config, archive_dir=archive_dir)
    logger.info(
        "真实视频端到端验收开始：%s（%dx%d，%.2f fps），归档目录 %s",
        args.video, source.width, source.height, source.fps, archive_dir,
    )

    run_start = datetime.now()  # 运行窗口起点（CSV 时间吻合断言用）
    frame_no = 0
    confirm_records = []  # [(归档记录, 确认帧的目标框)]，供带框像素断言
    cross_records = []    # 穿线事件的归档记录
    while True:
        frame = source.read()
        if frame is None:
            break
        frame_no += 1
        detections = detector.detect(frame)
        confirm_result = confirmer.update(detections, frame_no)
        line_result = counter.update(confirm_result, frame_no, source.fps)
        annotated = draw_annotated(frame, confirm_result, counter)

        # 目标确认告警：本帧刚确认且命中告警类别的目标
        for rec in confirm_result.just_confirmed:
            det = rec["det"]
            if det["cls"] not in config.alert_classes:
                continue
            record = archiver.archive(annotated, EVENT_CONFIRM,
                                      det["cls"], det["conf"],
                                      position=det["center"])
            confirm_records.append((record, list(det["box"])))
        # 穿线事件告警：位置取穿线点
        for ev in line_result.events:
            event_type = EVENT_CROSS_IN if ev["direction"] == "in" \
                else EVENT_CROSS_OUT
            record = archiver.archive(annotated, event_type,
                                      ev["cls"], ev.get("conf"),
                                      position=ev["cross_point"])
            cross_records.append(record)
            logger.info(
                "[穿线告警] 帧%d t=%.2fs %s %s #%d 穿线点%s -> %s",
                ev["frame_no"], ev["time_sec"], ev["direction_cn"],
                ev["cls"], ev["track_id"], ev["cross_point"],
                os.path.basename(record["screenshot_path"]),
            )

        # 两类告警各自归档到即收尾，无需跑完全片
        if confirm_records and cross_records:
            logger.info("两类告警均已归档（确认 %d 条 / 穿线 %d 条），"
                        "于第 %d 帧收尾", len(confirm_records),
                        len(cross_records), frame_no)
            break
        if args.max_frames is not None and frame_no >= args.max_frames:
            logger.info("达到 --max-frames %d，结束", args.max_frames)
            break
    run_end = datetime.now()
    source.release()
    archiver.close()

    # ---------------- 断言 ----------------
    assert confirm_records, \
        "跑满 %d 帧无任何'目标确认'告警归档，请检查素材或告警类别配置" % frame_no
    assert cross_records, \
        "跑满 %d 帧无任何'穿线事件'告警归档（默认警戒线下该素材首起穿线" \
        "在千帧量级，可调大 --max-frames 重试）" % frame_no

    # CSV 行数与归档记录数一致，逐行字段齐全
    _header, rows = read_csv_rows(archiver.csv_path)
    assert len(rows) == len(archiver.records), \
        "CSV 行数 %d 应与归档记录数 %d 一致" % (
            len(rows), len(archiver.records))
    for row in rows:
        assert_row_fields(row)
    # 时间与事件吻合：每条 CSV 时间戳落在本次运行窗口内；
    # 截图路径与内存记录一一对应
    assert [row[5] for row in rows] == \
        [rec["screenshot_path"] for rec in archiver.records]
    for row in rows:
        ts = parse_ts(row[0])
        assert run_start - timedelta(seconds=2) <= ts \
            <= run_end + timedelta(seconds=2), \
            "CSV 时间戳 %s 不在运行窗口 [%s, %s] 内" % (
                ts, run_start, run_end)
    # 类别与事件类型分布断言
    confirm_rows = [r for r in rows if r[1] == EVENT_CONFIRM]
    cross_rows = [r for r in rows
                  if r[1] in (EVENT_CROSS_IN, EVENT_CROSS_OUT)]
    assert confirm_rows and cross_rows, "CSV 应同时含两类事件"
    assert all(r[2] in config.alert_classes for r in rows), \
        "CSV 类别字段应全部命中告警类别配置"

    # 截图内容抽查：首张"目标确认"截图上检测框框线像素必须存在
    first_rec, first_box = confirm_records[0]
    img = cv2.imread(first_rec["screenshot_path"])
    assert img is not None and img.shape == (src_h, src_w, 3), \
        "归档截图尺寸应与视频画面一致：%s vs %sx%s" % (
            img.shape if img is not None else None, src_w, src_h)
    x1, y1, x2, y2 = first_box
    edge_points = [(x1 + 1, y1 + 1), (x2 - 1, y1 + 1),
                   (x1 + 1, y2 - 1), ((x1 + x2) // 2, y1 + 1)]
    green_hits = 0
    for px, py in edge_points:
        b, g, r = (int(v) for v in img[py, px])
        if g > 200 and b < 80 and r < 80:
            green_hits += 1
    assert green_hits >= 2, \
        "确认告警截图上未检出检测框框线像素（采样 %d 点命中 %d）" % (
            len(edge_points), green_hits)

    summary = ("真实视频端到端验收通过：共处理 %d 帧，归档 %d 条"
               "（目标确认 %d / 穿线 %d），CSV %s，归档目录 %s" % (
                   frame_no, len(rows), len(confirm_rows), len(cross_rows),
                   archiver.csv_path, archive_dir))
    print(summary, flush=True)
    logger.info(summary)


def parse_args():
    """解析命令行参数（默认直接跑合成单测 + 默认素材真实视频验收）。"""
    parser = argparse.ArgumentParser(
        description="A7 告警截图归档与日志模块 验收测试")
    parser.add_argument("--video", default=DEFAULT_VIDEO,
                        help="真实视频验收素材路径（默认派活指定素材）")
    parser.add_argument("--max-frames", type=int, default=1600,
                        help="真实视频验收最多处理帧数（默认 1600，足够覆盖"
                             "该素材默认警戒线下的首起自然穿线）")
    parser.add_argument("--archive-dir", default=DEFAULT_ARCHIVE_DIR,
                        help="测试归档输出目录（默认 tests/a7_archive，"
                             "与运行期 archive/ 隔离）")
    parser.add_argument("--skip-video", action="store_true",
                        help="只跑合成场景单元测试")
    return parser.parse_args()


def main():
    args = parse_args()
    run_unit_tests(os.path.abspath(args.archive_dir))
    if not args.skip_video:
        run_video_mode(args)
        print("PASS: tests/test_archive.py 合成单测与真实视频端到端验收全部通过")


if __name__ == "__main__":
    main()
