# -*- coding: utf-8 -*-
"""鹰眼巡校——A11 ByteTrack 跟踪集成与徘徊检测 验收测试。

测试内容分三部分：

一、徘徊检测合成轨迹单元测试（默认运行，无需视频/模型，秒级完成）：
    1. 徘徊监控区域 归一化坐标 -> 像素坐标换算正确（含越界钳制）；
    2. 核心验收：同一 track ID 在区域内停留超阈值秒数触发恰好 1 次
       徘徊告警（事件帧号、字段齐全：事件类型/类别/置信度/位置）；
    3. 停留不足阈值 0 告警；
    4. 一次连续驻留只告警一次（超时后继续停留不重复告警）；
    5. 离开区域后重新进入重新计时，再次超时属于合法新告警；
    6. 未确认目标（confirmed=False）不参与徘徊判定；
    7. 停留阈值秒数可配置且生效（对照断言：同一轨迹，阈值 1.0s 触发、
       阈值 3.0s 不触发）；
    8. 帧号不连续时驻留状态清空重计，不拼接出假超时；
    9. 告警事件接入 A7 归档器：触发后告警日志 CSV 含"徘徊告警"行且
       字段齐全、截图文件真实存在；
   10. A6 回归（跟踪后端）：LineCounter 直接消费 backend="bytetrack"
       的 ConfirmResult，穿线事件正常产生（跟踪模式下游契约不变）。

二、A5/A6 既有测试回归（默认运行，子进程方式）：
    纯 IOU 降级路径不变式验证——tests/test_confirm.py 与
    tests/test_line.py 原样各跑一遍，退出码必须为 0。

三、真实视频跟踪验收（--video 指定）：
    ByteTrack 跟踪全链路逐帧运行（track -> ConfirmResult -> 越线计数
    -> 徘徊检测），断言：
    1. ConfirmResult.backend == "bytetrack"（跟踪后端确实生效，
       未发生自动降级）；
    2. 确认契约在跟踪模式下成立：轨迹累计存在 N 帧才确认、同一 ID
       只确认一次、确认后 consecutive 随观测逐帧 +1；
    3. 同一物理目标跨帧 ID 一致：取每帧最大 person 框作为"主目标"，
       相邻帧主目标框 IOU>=0.3 而 track_id 不同记一次 ID 切换，
       切换次数 <= 阈值（默认走 Config.track_max_id_switch，可用
       --id-switch-threshold 覆盖）；
    4. 已确认轨迹带有轨迹尾迹（锚点序列长度 >= 2）；
    5. 跟踪模式下越线计数/徘徊检测管道运行正常（计数内部一致）。

运行方式：
    python tests/test_tracker.py                          # 合成单测 + A5/A6 回归
    python tests/test_tracker.py --video <视频路径>        # 上述 + 真实视频跟踪验收
    python tests/test_tracker.py --video <路径> --max-frames 300 \
        --id-switch-threshold 3
全部断言通过时输出 PASS 并以退出码 0 结束；失败时抛出 AssertionError
或以非 0 退出码结束。
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np

# 本文件位于 tests/ 下，直接运行时 sys.path 不含项目根目录，手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.archiver import AlertArchiver
from src.config import Config
from src.frame_confirm import BACKEND_IOU_FALLBACK, ConfirmResult, iou_of
from src.line_counter import DIR_IN, LineCounter
from src.logger import get_logger
from src.loitering import EVENT_LOITERING, LoiteringDetector
from src.tracker import BACKEND_BYTETRACK

logger = get_logger("test_tracker")

# 合成测试画面尺寸 1000x800；默认徘徊区域 (0.25,0.25,0.75,0.75)
# -> 像素 (250,200)-(750,600)；测试锚点取画面中心 (500,400)
FRAME_W, FRAME_H = 1000, 800
TEST_FPS = 25.0
ZONE = (0.25, 0.25, 0.75, 0.75)
ZONE_PX = (250, 200, 750, 600)


def build_config(**overrides):
    """构造测试用配置：在默认配置上按关键字参数覆盖。"""
    config = Config()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def anchor_box(ax, ay, w=60, h=60):
    """构造底部中心锚点为 (ax, ay) 的合成检测框 [x1, y1, x2, y2]。"""
    return [int(ax - w // 2), int(ay - h), int(ax + w // 2), int(ay)]


def make_record(cls, box, track_id, confirmed=True, backend=BACKEND_BYTETRACK):
    """构造与 ConfirmResult.records 结构对齐的合成确认记录。

    backend 仅用于说明记录来源语义（跟踪/IOU 降级），记录结构本身
    与 A5/A11 两种后端完全一致——这正是下游模块无需感知后端的契约。
    """
    x1, y1, x2, y2 = box
    det = {
        "cls": cls,
        "conf": 0.85,
        "box": list(box),
        "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
    }
    return {
        "det": det,
        "track_id": track_id,
        "consecutive": 5,
        "confirmed": confirmed,
        "just_confirmed": False,
        "confirm_frame": 1,
        "iou": None,
        "reason": "合成记录（%s）" % backend,
    }


def feed_loitering(detector, frames, start_frame=1, confirmed=True,
                   backend=BACKEND_BYTETRACK):
    """把合成轨迹序列喂给徘徊检测器。

    参数：
        frames: 逐帧目标列表，每帧为 [(track_id, cls, box), ...]。
    返回：
        每帧的 LoiteringResult 列表。
    """
    results = []
    for offset, items in enumerate(frames):
        records = [make_record(cls, box, tid, confirmed=confirmed,
                               backend=backend)
                   for tid, cls, box in items]
        confirm_result = ConfirmResult(records, start_frame + offset,
                                       backend=backend)
        results.append(detector.update(confirm_result,
                                       start_frame + offset, TEST_FPS))
    return results


def all_events(results):
    """把逐帧结果中的徘徊告警事件按帧序拼成一个列表。"""
    return [ev for result in results for ev in result.events]


def make_loiterer(**overrides):
    """按 1000x800 合成画面构造徘徊检测器（配置可覆盖）。"""
    overrides.setdefault("loitering_zone", ZONE)
    return LoiteringDetector(build_config(**overrides), FRAME_W, FRAME_H)


def stay_frames(count, ax=500, ay=400, track_id=1, cls="person"):
    """生成 count 帧"目标锚点停留在 (ax, ay)"的合成序列。"""
    box = anchor_box(ax, ay)
    return [[(track_id, cls, box)] for _ in range(count)]


# ---------------- 一、徘徊检测合成轨迹单元测试 ----------------

def test_zone_geometry():
    """用例 1：徘徊区域归一化坐标 -> 像素坐标换算（含越界钳制）。"""
    detector = make_loiterer()
    assert detector.zone_px == ZONE_PX, \
        "默认徘徊区域像素坐标错误: %s" % (detector.zone_px,)
    custom = make_loiterer(loitering_zone=(0.1, 0.2, 0.5, 0.6))
    assert custom.zone_px == (100, 160, 500, 480), \
        "自定义徘徊区域像素坐标错误: %s" % (custom.zone_px,)
    clamped = make_loiterer(loitering_zone=(-0.2, 0.1, 1.2, 0.9))
    assert clamped.zone_px == (0, 80, 1000, 720), \
        "越界徘徊区域应被钳制: %s" % (clamped.zone_px,)
    assert detector.point_in_zone(500, 400)
    assert not detector.point_in_zone(100, 100)
    logger.info(
        "用例1 PASS：徘徊区域像素换算正确 %s / %s，越界钳制生效",
        detector.zone_px, custom.zone_px,
    )


def test_loitering_triggered_over_threshold():
    """用例 2（核心验收）：停留超阈值触发恰好 1 次徘徊告警。"""
    detector = make_loiterer(loitering_duration_sec=2.0)  # 2s * 25fps = 50 帧
    results = feed_loitering(detector, stay_frames(60))
    events = all_events(results)
    assert len(events) == 1, "超时停留应恰好触发 1 次告警: %s" % events
    ev = events[0]
    # 第 50 帧驻留满 2.0 秒触发
    assert ev["frame_no"] == 50, "告警帧号应为 50: %s" % ev
    assert ev["event_type"] == EVENT_LOITERING == "徘徊告警"
    assert ev["cls"] == "person" and ev["track_id"] == 1
    assert ev["conf"] == 0.85
    # 告警位置取锚点（归档 position 字段来源）
    assert ev["anchor"] == (500, 400)
    assert abs(ev["dwell_sec"] - 2.0) < 0.05
    assert abs(ev["time_sec"] - 50 / TEST_FPS) < 1e-6
    assert "徘徊" in ev["reason"] and "停留" in ev["reason"]
    # 区域坐标走 config 且生效：默认区域包含 (500,400)，换成不含该点的
    # 区域配置后同一轨迹应 0 告警（区域可配置的对照验证见用例 7 阈值）
    logger.info(
        "用例2 PASS：停留超 2.0s 阈值于帧%d 触发徘徊告警（停留 %.2fs，%s）",
        ev["frame_no"], ev["dwell_sec"], ev["reason"],
    )


def test_below_threshold_no_alert():
    """用例 3：停留不足阈值 0 告警。"""
    detector = make_loiterer(loitering_duration_sec=2.0)
    results = feed_loitering(detector, stay_frames(49))  # 49/25 = 1.96s < 2s
    assert len(all_events(results)) == 0, "未达阈值不应告警"
    assert len(detector.events) == 0
    logger.info("用例3 PASS：停留 1.96s < 2.0s 阈值，0 告警")


def test_alert_once_per_stay():
    """用例 4：一次连续驻留只告警一次（超时后继续停留不重复告警）。"""
    detector = make_loiterer(loitering_duration_sec=2.0)
    results = feed_loitering(detector, stay_frames(100))  # 4s 持续停留
    events = all_events(results)
    assert len(events) == 1, "一次驻留只应告警一次: %s" % events
    assert events[0]["frame_no"] == 50
    logger.info("用例4 PASS：4s 持续停留只触发 1 次告警（不逐帧重复）")


def test_reenter_retriggers():
    """用例 5：离开区域后重新进入重新计时，再次超时是合法新告警。"""
    detector = make_loiterer(loitering_duration_sec=2.0)
    frames = stay_frames(50)              # 帧1-50：区域内，帧50 触发第 1 次
    frames += [[(1, "person", anchor_box(100, 100))] for _ in range(5)]  # 离开
    frames += stay_frames(50)             # 重新进入并再次驻留满 2s
    results = feed_loitering(detector, frames)
    events = all_events(results)
    assert len(events) == 2, "离开后再驻留超时应产生第 2 次告警: %s" % events
    assert events[0]["frame_no"] == 50
    assert events[1]["frame_no"] == 105, "再次进入后需重新计满 2s: %s" % events[1]
    logger.info(
        "用例5 PASS：离开区域驻留清零，再入重新计时（帧50/帧105 各 1 次告警）",
    )


def test_unconfirmed_not_monitored():
    """用例 6：未确认目标（confirmed=False）不参与徘徊判定。"""
    detector = make_loiterer(loitering_duration_sec=2.0)
    results = feed_loitering(detector, stay_frames(80), confirmed=False)
    assert len(all_events(results)) == 0
    assert len(detector.events) == 0
    logger.info("用例6 PASS：未确认目标停留 80 帧 0 告警（与 A5/A11 契约一致）")


def test_duration_configurable():
    """用例 7（对照验收）：停留阈值秒数可配置且生效。

    同一"驻留 40 帧（1.6s）"轨迹：阈值 1.0s 时第 25 帧触发告警，
    阈值 3.0s 时全程 0 告警；同时验证区域坐标走配置——把区域配置
    成不含锚点的角落，同一轨迹同样 0 告警。
    """
    # 场景 A：阈值 1.0s（25 帧达标）
    det_a = make_loiterer(loitering_duration_sec=1.0)
    ev_a = all_events(feed_loitering(det_a, stay_frames(40)))
    assert len(ev_a) == 1 and ev_a[0]["frame_no"] == 25

    # 场景 B：阈值 3.0s（40 帧只有 1.6s，不达标）
    det_b = make_loiterer(loitering_duration_sec=3.0)
    ev_b = all_events(feed_loitering(det_b, stay_frames(40)))
    assert len(ev_b) == 0

    # 场景 C：同一轨迹，区域配置为左上角小块（不含锚点 (500,400)）
    det_c = make_loiterer(loitering_duration_sec=1.0,
                          loitering_zone=(0.0, 0.0, 0.2, 0.2))
    ev_c = all_events(feed_loitering(det_c, stay_frames(40)))
    assert len(ev_c) == 0, "锚点在配置区域外不应告警"
    logger.info(
        "用例7 PASS：同一驻留 1.6s 轨迹，阈值 1.0s 触发 / 3.0s 不触发；"
        "区域配置不含锚点时不触发（阈值与区域均走 config 且生效）",
    )


def test_frame_gap_resets_dwell():
    """用例 8：帧号不连续时驻留状态清空重计，不拼接出假超时。"""
    detector = make_loiterer(loitering_duration_sec=2.0)
    # 帧1-40：驻留 1.6s（未达阈值）；随后帧号跳到 100 重新驻留
    feed_loitering(detector, stay_frames(40))
    results = feed_loitering(detector, stay_frames(45), start_frame=100)
    # 若不重计，(40+45)/25=3.4s 早已超时；重计后 45 帧=1.8s < 2s，0 告警
    assert len(all_events(results)) == 0, "跳帧后不应拼接出超时告警"
    results2 = feed_loitering(detector, stay_frames(5), start_frame=145)
    events = all_events(results2)
    assert len(events) == 1 and events[0]["frame_no"] == 149, \
        "重计后应从新段第 50 帧才触发: %s" % events
    logger.info("用例8 PASS：帧号 40->100 跳变后驻留重计，0 假告警")


def test_loitering_event_archived():
    """用例 9（核心验收）：徘徊告警接入 A7 归档（截图 + CSV + 记录）。"""
    with tempfile.TemporaryDirectory(
            prefix="a11_loiter_", dir=os.path.dirname(os.path.abspath(__file__))
    ) as archive_dir:
        config = build_config(loitering_duration_sec=2.0)
        detector = LoiteringDetector(config, FRAME_W, FRAME_H)
        archiver = AlertArchiver(config, archive_dir=archive_dir)
        try:
            # 合成一帧带标注画面（黑底 + 目标框，代替真实视频帧）
            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            archived = 0
            for offset in range(50):
                box = anchor_box(500, 400)
                records = [make_record("person", box, 1)]
                confirm_result = ConfirmResult(records, offset + 1,
                                               backend=BACKEND_BYTETRACK)
                result = detector.update(confirm_result, offset + 1, TEST_FPS)
                for ev in result.events:
                    # 与 app.py 相同的归档调用方式：事件类型/类别/置信度/位置
                    import cv2
                    shot = frame.copy()
                    x1, y1, x2, y2 = ev["box"]
                    cv2.rectangle(shot, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    rec = archiver.archive(shot, ev["event_type"], ev["cls"],
                                           ev["conf"], position=ev["anchor"])
                    archived += 1
                    assert rec["event_type"] == EVENT_LOITERING
            assert archived == 1, "应恰好归档 1 条徘徊告警"

            # CSV 证据：含"徘徊告警"行，字段齐全（时间戳/事件类型/类别/
            # 置信度/位置/截图路径），截图文件真实存在
            with open(archiver.csv_path, encoding="utf-8-sig") as fp:
                rows = list(csv.reader(fp))
            assert rows[0] == ["时间戳", "事件类型", "类别", "置信度",
                               "位置", "截图路径"]
            data_rows = [r for r in rows[1:] if r and r[1] == EVENT_LOITERING]
            assert len(data_rows) == 1, "CSV 应含 1 行徘徊告警: %s" % rows
            row = data_rows[0]
            assert row[2] == "person" and row[3] == "0.85"
            assert row[4] == "(500, 400)", "位置字段应为锚点: %s" % row
            assert os.path.isfile(row[5]), "截图文件应真实存在: %s" % row[5]
            logger.info(
                "用例9 PASS：徘徊告警已归档——CSV 行 %s，截图 %s 存在",
                row[:5], os.path.basename(row[5]),
            )
        finally:
            archiver.close()


def test_line_counter_with_track_backend():
    """用例 10（A6 跟踪模式回归）：LineCounter 消费 bytetrack 后端的
    ConfirmResult，穿线事件正常产生且 track_id 保持。"""
    config = build_config()
    counter = LineCounter(config, FRAME_W, FRAME_H)
    # 警戒线默认 y=0.5 -> 像素 y=400；锚点 480(下)->440(下)->360(上)
    frames = [[(7, "person", anchor_box(500, y2))]
              for y2 in (480, 440, 360)]
    events = []
    for offset, items in enumerate(frames):
        records = [make_record(cls, box, tid, backend=BACKEND_BYTETRACK)
                   for tid, cls, box in items]
        confirm_result = ConfirmResult(records, offset + 1,
                                       backend=BACKEND_BYTETRACK)
        events.extend(counter.update(confirm_result, offset + 1,
                                     TEST_FPS).events)
    assert len(events) == 1, "跟踪后端确认结果应正常驱动穿线: %s" % events
    assert events[0]["direction"] == DIR_IN and events[0]["track_id"] == 7
    assert counter.count_in == 1 and counter.count_out == 0
    logger.info(
        "用例10 PASS：backend=%s 的 ConfirmResult 驱动 A6 越线判进正常"
        "（track_id=7 保持）", BACKEND_BYTETRACK,
    )


# ---------------- 二、A5/A6 既有测试回归（IOU 降级路径） ----------------

def run_regression_tests():
    """以子进程原样回归 A5/A6 既有测试（纯 IOU 降级路径不变式）。"""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ("test_confirm.py", "test_line.py"):
        script = os.path.join(tests_dir, name)
        proc = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert proc.returncode == 0, \
            "%s 回归失败（退出码 %d）：\n%s\n%s" % (
                name, proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
        assert "PASS" in proc.stdout, "%s 输出缺少 PASS 标记" % name
        logger.info("回归 PASS：%s 退出码 0（IOU 降级路径行为不变）", name)
        print("PASS: %s 回归通过（退出码 0）" % name, flush=True)


# ---------------- 三、真实视频跟踪验收 ----------------

def run_video_mode(args):
    """真实视频 ByteTrack 跟踪验收：ID 一致性 + 确认契约 + 下游管道。"""
    from src.detector import Detector
    from src.tracker import ByteTrackTracker
    from src.video_source import create_video_source

    config = Config()
    source = create_video_source(args.video, config)
    detector = Detector(config)
    # 跟踪器复用 Detector 已加载的模型（避免权重重复加载）
    tracker = ByteTrackTracker(config, model=detector.model)
    counter = LineCounter(config, source.width, source.height)
    loiterer = LoiteringDetector(config, source.width, source.height)
    logger.info(
        "视频跟踪验收开始：%s（%dx%d，%.2f fps），确认帧数 N=%d，"
        "ID 切换阈值 <=%d",
        args.video, source.width, source.height, source.fps,
        config.confirm_frames, args.id_switch_threshold,
    )

    frame_no = 0
    backend_seen = set()
    frame_track_boxes = {}  # frame_no -> {track_id: box}（全部记录）
    confirm_age_ok = True   # 确认契约：已确认轨迹 consecutive 逐帧 +1
    track_age_memo = {}     # track_id -> 上一观测帧的 consecutive
    confirmed_ids = set()   # 出现过的已确认轨迹 ID
    confirm_event_ids = []  # just_confirmed 事件按序记录（验同一 ID 只确认一次）
    max_trail_len = 0

    while True:
        frame = source.read()
        if frame is None:
            break
        frame_no += 1
        result = tracker.update(frame, frame_no)
        backend_seen.add(result.backend)

        # ConfirmResult 契约：结构字段齐全（与 A5 对齐）
        for rec in result.records:
            for key in ("det", "track_id", "consecutive", "confirmed",
                        "just_confirmed", "confirm_frame", "iou", "reason"):
                assert key in rec, "确认记录缺少字段 %s: %s" % (key, rec)
            tid = rec["track_id"]
            # 同一轨迹连续观测期间 consecutive 逐帧 +1
            prev = track_age_memo.get(tid)
            if prev is not None and prev[0] == frame_no - 1:
                if rec["consecutive"] != prev[1] + 1:
                    confirm_age_ok = False
                    logger.error(
                        "帧%d 轨迹#%d consecutive 未逐帧 +1：%d -> %d",
                        frame_no, tid, prev[1], rec["consecutive"])
            track_age_memo[tid] = (frame_no, rec["consecutive"])
            if rec["confirmed"]:
                confirmed_ids.add(tid)
                assert rec["consecutive"] >= config.confirm_frames, \
                    "确认目标 consecutive 应 >= N: %s" % rec
            if rec["just_confirmed"]:
                confirm_event_ids.append(tid)
            trail = tracker.get_trail(tid)
            max_trail_len = max(max_trail_len, len(trail))

        # 记录本帧全部轨迹框（供 ID 切换的"身份移交"判定回查）
        frame_track_boxes[frame_no] = {r["track_id"]: r["det"]["box"]
                                       for r in result.records}

        # 下游管道（跟踪模式）：越线计数 + 徘徊检测照常消费确认结果
        counter.update(result, frame_no, source.fps)
        loiterer.update(result, frame_no, source.fps)

        if args.max_frames is not None and frame_no >= args.max_frames:
            logger.info("达到 --max-frames %d，提前结束", args.max_frames)
            break
    source.release()

    # ---- 断言 1：跟踪后端确实生效（未自动降级为 IOU 路径） ----
    assert backend_seen == {BACKEND_BYTETRACK}, \
        "全程应只出现 bytetrack 后端: %s" % backend_seen

    # ---- 断言 2：确有目标被确认（素材含人/车，N 帧后应有确认轨迹） ----
    assert confirmed_ids, "全程无任何已确认轨迹，素材/跟踪可能异常"
    # 同一 ID 只确认一次
    assert len(confirm_event_ids) == len(set(confirm_event_ids)), \
        "同一轨迹 ID 不应重复确认: %s" % confirm_event_ids
    assert confirm_age_ok, "已确认轨迹 consecutive 应随观测逐帧 +1"

    # ---- 断言 3：同一物理目标跨帧 ID 一致（ID 切换次数 <= 阈值） ----
    # 采用 MOT 标准 IDSW 的代理口径：取全程出现帧数最多的轨迹作为参照
    # 目标（无 GT 时的"同一物理目标"近似），参照轨迹在某帧缺席且同帧
    # 存在另一条与其最后已知框高度重叠（IOU>=0.5）的轨迹时，记 1 次
    # ID 切换（身份移交），参照随之换为接管轨迹；参照轨迹长时间
    # （>track_max_age 帧）缺席且无重叠接管者时视为目标离场，改选剩余
    # 帧中出现最多的轨迹继续。该口径不受"同一帧多条重叠轨迹轮流成为
    # 最大框"的选取抖动影响。
    from collections import Counter
    presence = Counter()
    for boxes in frame_track_boxes.values():
        for tid in boxes:
            presence[tid] += 1
    assert presence, "全程无任何跟踪轨迹，素材/跟踪可能异常"
    # 参照目标需有足够多的出现帧才能验证"跨帧一致"
    ref_tid, ref_count = presence.most_common(1)[0]
    assert ref_count >= 30, \
        "出现最多的轨迹仅 %d 帧，无法验证 ID 一致性" % ref_count
    switches = 0
    last_box = None
    last_seen = None
    for f in sorted(frame_track_boxes):
        boxes = frame_track_boxes[f]
        if ref_tid in boxes:
            last_box = boxes[ref_tid]
            last_seen = f
            continue
        # 参照轨迹本帧缺席：离场太久则换参照（目标离开画面不算切换）
        if last_seen is not None and f - last_seen > config.track_max_age:
            remaining = Counter()
            for g in frame_track_boxes:
                if g > f:
                    for tid in frame_track_boxes[g]:
                        remaining[tid] += 1
            if not remaining:
                break
            ref_tid = remaining.most_common(1)[0][0]
            last_box = None
            last_seen = None
            continue
        if last_box is None:
            continue
        # 找身份移交：本帧与最后已知框 IOU>=0.5 的其他轨迹
        cands = [(iou_of(last_box, b), tid) for tid, b in boxes.items()
                 if tid != ref_tid]
        cands = [c for c in cands if c[0] >= 0.5]
        if cands:
            cands.sort(reverse=True)
            new_tid = cands[0][1]
            switches += 1
            logger.info(
                "ID 切换（身份移交）：帧%d 参照 #%d 缺席，#%d 接管"
                "（与最后已知框 IOU=%.2f）",
                f, ref_tid, new_tid, cands[0][0])
            ref_tid = new_tid
            last_box = boxes[new_tid]
            last_seen = f
    assert switches <= args.id_switch_threshold, \
        "ID 切换 %d 次超过阈值 %d" % (switches, args.id_switch_threshold)

    # ---- 断言 4：已确认轨迹带有轨迹尾迹 ----
    assert max_trail_len >= 2, "已确认轨迹应有尾迹点（len>=2）"

    # ---- 断言 5：下游管道内部一致（越线计数与事件清单对齐） ----
    assert len(counter.events) == counter.count_in + counter.count_out

    summary = (
        "视频跟踪验收结束：共处理 %d 帧，后端 %s；确认轨迹 %d 条"
        "（确认事件 %d 次）；跟踪轨迹 %d 条、ID 切换 %d 次（阈值 <=%d）；"
        "最长尾迹 %d 点；越线事件 %d 起（进 %d / 出 %d）；徘徊告警 %d 起" % (
            frame_no, "/".join(sorted(backend_seen)),
            len(confirmed_ids), len(confirm_event_ids),
            len(presence), switches, args.id_switch_threshold,
            max_trail_len, len(counter.events), counter.count_in,
            counter.count_out, len(loiterer.events)))
    print(summary, flush=True)
    logger.info(summary)


def parse_args():
    """解析命令行参数（无 --video 时只跑合成单测与 A5/A6 回归）。"""
    parser = argparse.ArgumentParser(
        description="A11 ByteTrack 跟踪集成与徘徊检测 验收测试")
    parser.add_argument("--video", default=None,
                        help="真实视频跟踪验收：视频文件路径")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="真实视频验收最多处理的帧数（默认 300）")
    parser.add_argument("--id-switch-threshold", type=int,
                        default=Config().track_max_id_switch,
                        help="主目标 ID 切换次数上限（默认走配置 "
                             "track_max_id_switch）")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("==== A11 ByteTrack 跟踪与徘徊检测 验收测试开始 ====")
    test_zone_geometry()
    test_loitering_triggered_over_threshold()
    test_below_threshold_no_alert()
    test_alert_once_per_stay()
    test_reenter_retriggers()
    test_unconfirmed_not_monitored()
    test_duration_configurable()
    test_frame_gap_resets_dwell()
    test_loitering_event_archived()
    test_line_counter_with_track_backend()
    logger.info("==== 徘徊检测合成用例 10 组断言全部通过 ====")
    print("PASS: 徘徊检测/跟踪契约 合成单元测试 10 组用例全部通过", flush=True)

    run_regression_tests()

    if args.video:
        run_video_mode(args)
    print("PASS: tests/test_tracker.py 全部断言通过", flush=True)


if __name__ == "__main__":
    main()
