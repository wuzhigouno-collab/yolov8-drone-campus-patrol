# -*- coding: utf-8 -*-
"""鹰眼巡校——A6 越线方向计数模块 验收测试。

测试内容分两部分：

一、合成轨迹单元测试（默认运行，无需视频/模型，秒级完成）：
    1. 警戒线归一化坐标 -> 像素坐标换算正确（含越界钳制）；
    2. 核心验收：底部中心锚点 下->上 穿过水平线判"进"（默认
       line_enter_direction=True），上->下 判"出"；
    3. 未穿线（同侧移动 / 沿线平移）0 事件；
    4. 同目标重复计数防护：压线抖动（死区内）不产生事件，
       穿到对侧后才允许下一次计数；
    5. 未确认目标（confirmed=False）不参与计数；
    6. 方向语义配置可翻转（line_enter_direction=False 时进出互换）；
    7. 非水平线（垂直警戒线）的穿线判定；
    8. 帧号不连续时轨迹侧记忆清空，不会拼接出假穿线；
    9. 同一帧两个目标反向穿线，进出各计一次；
    10. reset() 清空计数与事件；
    11. 与真实 A5 FrameConfirmer 联用：已确认目标穿线产生事件，
        事件的 track_id 与确认器轨迹号一致。

二、真实视频验收（--video 指定）：
    检测 -> 多帧确认 -> 越线计数 全链路逐帧运行，输出穿线事件列表
    （帧号、视频内时间、方向、类别、轨迹号、穿线位置），并把每个事件
    的穿线帧画面（含警戒线、目标框、锚点与穿线点标注）及其前 5 帧的
    画面前照保存到佐证目录，供人工抽查核实事件真实。

运行方式：
    python tests/test_line.py                          # 仅合成单元测试
    python tests/test_line.py --video <视频路径>        # 单测 + 真实视频
    python tests/test_line.py --video <路径> --line 0.0,0.5,1.0,0.5 \
        --max-frames 500 --evidence-dir tests/line_evidence
全部断言通过且视频模式产出事件时输出 PASS 并以退出码 0 结束；
断言失败抛 AssertionError；视频模式 0 事件时以退出码 1 结束。
"""

import argparse
import json
import os
import sys

# 本文件位于 tests/ 下，直接运行时 sys.path 不含项目根目录，手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.frame_confirm import ConfirmResult, FrameConfirmer
from src.line_counter import (
    DIR_IN,
    DIR_OUT,
    LineCounter,
)
from src.logger import get_logger

logger = get_logger("test_line")

# 合成测试画面尺寸 1000x200：默认水平警戒线 y=0.5 -> 像素 y=100
FRAME_W, FRAME_H = 1000, 200
LINE_Y = 100
TEST_FPS = 25.0


def build_config(**overrides):
    """构造测试用配置：在默认配置上按关键字参数覆盖。"""
    config = Config()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def vbox(y2, cx=500, w=60, h=60):
    """构造底部中心锚点为 (cx, y2) 的合成检测框 [x1, y1, x2, y2]。"""
    return [cx - w // 2, y2 - h, cx + w // 2, y2]


def make_record(cls, box, track_id, confirmed=True):
    """构造与 A5 ConfirmResult.records 结构对齐的合成确认记录。"""
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
        "reason": "合成记录",
    }


def feed(counter, frames, start_frame=1, confirmed=True):
    """把合成轨迹序列喂给计数器。

    参数：
        frames: 逐帧目标列表，每帧为 [(track_id, cls, box), ...]。
    返回：
        每帧的 LineCountResult 列表。
    """
    results = []
    for offset, items in enumerate(frames):
        records = [make_record(cls, box, tid, confirmed=confirmed)
                   for tid, cls, box in items]
        confirm_result = ConfirmResult(records, start_frame + offset)
        results.append(
            counter.update(confirm_result, start_frame + offset, TEST_FPS))
    return results


def feed_anchor_track(counter, y2_list, track_id=1, cls="person",
                      start_frame=1, confirmed=True):
    """便捷方法：单目标锚点纵向移动序列（水平位置固定画面中央）。"""
    return feed(counter,
                [[(track_id, cls, vbox(y2))] for y2 in y2_list],
                start_frame=start_frame, confirmed=confirmed)


def all_events(results):
    """把逐帧结果中的穿线事件按帧序拼成一个列表。"""
    return [ev for result in results for ev in result.events]


def make_counter(**overrides):
    """按 1000x200 合成画面构造计数器（配置可覆盖）。"""
    return LineCounter(build_config(**overrides), FRAME_W, FRAME_H)


def test_line_geometry():
    """用例 1：警戒线归一化坐标 -> 像素坐标换算（含越界钳制）。"""
    counter = make_counter()
    assert counter.line_px == (0, 100, 1000, 100), \
        "默认警戒线像素坐标错误: %s" % (counter.line_px,)
    custom = make_counter(warning_line=(0.1, 0.2, 0.9, 0.8))
    assert custom.line_px == (100, 40, 900, 160), \
        "自定义警戒线像素坐标错误: %s" % (custom.line_px,)
    # 越界坐标钳制到画面范围内
    clamped = make_counter(warning_line=(-0.2, 0.5, 1.2, 0.5))
    assert clamped.line_px == (0, 100, 1000, 100), \
        "越界警戒线应被钳制: %s" % (clamped.line_px,)
    logger.info(
        "用例1 PASS：警戒线像素换算正确 %s / %s，越界钳制生效",
        counter.line_px, custom.line_px,
    )


def test_down_to_up_is_in():
    """用例 2（核心验收）：下->上 穿过水平线判"进"。"""
    counter = make_counter()  # 默认 line_enter_direction=True
    # 锚点 y2：160(线下) -> 120(线下) -> 80(线上)，第 3 帧穿线
    results = feed_anchor_track(counter, [160, 120, 80])
    events = all_events(results)
    assert len(events) == 1, "应恰好产生 1 次穿线事件: %s" % events
    ev = events[0]
    assert ev["direction"] == DIR_IN and ev["direction_cn"] == "进"
    assert ev["frame_no"] == 3, "穿线应判定在第 3 帧: %s" % ev
    assert ev["cls"] == "person" and ev["track_id"] == 1
    assert abs(ev["time_sec"] - 3 / TEST_FPS) < 1e-6, "时间戳应=帧号/fps"
    # 穿线点应落在警戒线上（y≈100）且在轨迹 x 位置上
    assert abs(ev["cross_point"][1] - LINE_Y) <= 1, \
        "穿线点应在警戒线上: %s" % (ev["cross_point"],)
    assert abs(ev["cross_point"][0] - 500) <= 1
    assert counter.count_in == 1 and counter.count_out == 0
    assert "进" in ev["reason"]
    logger.info(
        "用例2 PASS：下->上穿线判'进'（帧%d，穿线点%s，累计进%d出%d）",
        ev["frame_no"], ev["cross_point"], counter.count_in,
        counter.count_out,
    )


def test_up_to_down_is_out():
    """用例 3（核心验收）：上->下 穿过水平线判"出"。"""
    counter = make_counter()
    # 锚点 y2：40(线上) -> 80(线上) -> 140(线下)，第 3 帧穿线
    results = feed_anchor_track(counter, [40, 80, 140])
    events = all_events(results)
    assert len(events) == 1
    ev = events[0]
    assert ev["direction"] == DIR_OUT and ev["direction_cn"] == "出"
    assert ev["frame_no"] == 3
    assert counter.count_in == 0 and counter.count_out == 1
    logger.info(
        "用例3 PASS：上->下穿线判'出'（帧%d，累计进%d出%d）",
        ev["frame_no"], counter.count_in, counter.count_out,
    )


def test_no_cross_no_event():
    """用例 4（核心验收）：未穿线 0 事件（同侧移动 / 沿线平移）。"""
    counter = make_counter()
    # 场景 A：始终在线下方移动（远离线）
    res_a = feed_anchor_track(counter, [130, 150, 170, 160])
    # 场景 B：在线下方沿线水平平移（x 从 100 走到 800，y 不变）
    res_b = feed(counter, [[(2, "car", vbox(160, cx=x))]
                           for x in (100, 300, 500, 800)],
                 start_frame=10)
    assert len(all_events(res_a)) == 0 and len(all_events(res_b)) == 0
    assert counter.count_in == 0 and counter.count_out == 0
    logger.info("用例4 PASS：同侧移动与沿线平移均 0 穿线事件")


def test_deadzone_repeat_guard():
    """用例 5（核心验收）：同目标重复计数防护（死区防抖）。

    序列：下方 -> 上方（进）-> 死区内来回抖动（不应计数）-> 继续上方
    -> 回到下方（出）。全程应恰好 2 个事件：1 进 1 出。
    """
    counter = make_counter()  # 默认死区 3px，线 y=100
    y2_seq = [160, 120, 80,      # 帧1-3：下->上，帧3 穿线判"进"
              99, 101, 98, 100,  # 帧4-7：死区(100±3)内抖动，维持上方侧
              70,                # 帧8：上方侧，无事件
              110, 140]          # 帧9-10：上->下，帧9 穿线判"出"
    results = feed_anchor_track(counter, y2_seq)
    events = all_events(results)
    assert len(events) == 2, \
        "死区抖动不应产生额外事件，应恰好 2 个: %s" % events
    assert events[0]["direction"] == DIR_IN and events[0]["frame_no"] == 3
    assert events[1]["direction"] == DIR_OUT and events[1]["frame_no"] == 9
    assert counter.count_in == 1 and counter.count_out == 1
    logger.info(
        "用例5 PASS：压线抖动 0 事件，穿回对侧后才再计数（1进1出共2事件）",
    )


def test_legitimate_recross_counted():
    """用例 6：目标穿回对侧后再次穿出属于合法新事件（往复计数）。"""
    counter = make_counter()
    # 锚点在 160(下)/80(上) 之间往复 4 次穿线
    results = feed_anchor_track(counter, [160, 80, 160, 80, 160])
    events = all_events(results)
    assert len(events) == 4
    assert [ev["direction"] for ev in events] == \
        [DIR_IN, DIR_OUT, DIR_IN, DIR_OUT]
    assert counter.count_in == 2 and counter.count_out == 2
    logger.info("用例6 PASS：往复穿线 4 次均被计数（进2出2）")


def test_unconfirmed_not_counted():
    """用例 7：未确认目标（confirmed=False）不参与穿线计数。"""
    counter = make_counter()
    results = feed_anchor_track(counter, [160, 120, 80], confirmed=False)
    assert len(all_events(results)) == 0
    assert counter.count_in == 0 and counter.count_out == 0
    logger.info("用例7 PASS：未确认目标穿线 0 事件（与 A5 契约一致）")


def test_enter_direction_flippable():
    """用例 8：方向语义走配置——line_enter_direction=False 时进出互换。"""
    counter = make_counter(line_enter_direction=False)
    results = feed_anchor_track(counter, [160, 120, 80])  # 下->上
    events = all_events(results)
    assert len(events) == 1
    assert events[0]["direction"] == DIR_OUT, \
        "配置翻转后 下->上 应判'出': %s" % events[0]
    assert counter.count_out == 1 and counter.count_in == 0
    logger.info("用例8 PASS：line_enter_direction=False 时 下->上 判'出'")


def test_vertical_line():
    """用例 9：非水平线（垂直警戒线）的穿线判定。

    垂直线 x=0.5（像素 x=500），有向线段自上而下 (500,0)->(500,200)；
    锚点从左向右（x 300->450->600）穿越，有向距离由正转负，
    按默认配置等价于"下->上"的符号跳变，判为"进"。
    """
    counter = make_counter(warning_line=(0.5, 0.0, 0.5, 1.0))
    assert counter.line_px == (500, 0, 500, 200)
    frames = [[(1, "person", [x - 30, 60, x + 30, 120])]
              for x in (300, 450, 600)]
    results = feed(counter, frames)
    events = all_events(results)
    assert len(events) == 1
    ev = events[0]
    assert ev["direction"] == DIR_IN
    assert abs(ev["cross_point"][0] - 500) <= 1, \
        "穿线点应在垂直线 x=500 上: %s" % (ev["cross_point"],)
    logger.info(
        "用例9 PASS：垂直线左->右穿越判'进'（穿线点%s）", ev["cross_point"],
    )


def test_frame_gap_clears_memory():
    """用例 10：帧号不连续时轨迹侧记忆清空，不拼接出假穿线。"""
    counter = make_counter()
    feed_anchor_track(counter, [160, 150])           # 帧1-2：下方侧
    # 帧号 2 -> 10 跳变：即使锚点已在线上方，也不应判定穿线
    res_gap = feed_anchor_track(counter, [80, 70], start_frame=10)
    assert len(all_events(res_gap)) == 0, "跳帧后不应拼接出穿线事件"
    # 随后正常连续穿线仍可被检出
    res_after = feed_anchor_track(counter, [120, 160], start_frame=12)
    events = all_events(res_after)
    assert len(events) == 1 and events[0]["direction"] == DIR_OUT
    logger.info("用例10 PASS：帧号 2->10 跳变后 0 假事件，后续穿线正常")


def test_multi_target_same_frame():
    """用例 11：同一帧两个目标反向穿线，进/出各计一次。"""
    counter = make_counter()
    frames = [
        [(1, "person", vbox(160, cx=300)), (2, "car", vbox(40, cx=700))],
        [(1, "person", vbox(80, cx=300)), (2, "car", vbox(140, cx=700))],
    ]
    results = feed(counter, frames)
    events = all_events(results)
    assert len(events) == 2
    by_track = {ev["track_id"]: ev for ev in events}
    assert by_track[1]["direction"] == DIR_IN   # 目标1 下->上
    assert by_track[2]["direction"] == DIR_OUT  # 目标2 上->下
    assert counter.count_in == 1 and counter.count_out == 1
    logger.info("用例11 PASS：同帧双目标反向穿线，进/出各计一次")


def test_reset():
    """用例 12：reset() 清空双向计数、事件列表与轨迹侧记忆。"""
    counter = make_counter()
    feed_anchor_track(counter, [160, 120, 80])
    assert counter.count_in == 1 and len(counter.events) == 1
    counter.reset()
    assert counter.count_in == 0 and counter.count_out == 0
    assert len(counter.events) == 0
    # 重置后同一轨迹重新穿线应被重新计数
    results = feed_anchor_track(counter, [160, 120, 80])
    assert len(all_events(results)) == 1 and counter.count_in == 1
    logger.info("用例12 PASS：reset() 后计数清零，可重新计数")


def test_integration_with_frame_confirmer():
    """用例 13：与真实 A5 多帧确认器联用（端到端合成轨迹）。

    目标连续 5 帧 下->上 穿线（相邻帧 IOU≈0.41，确认帧数 N=2）：
    第 2 帧确认，第 4 帧穿线判"进"，事件 track_id 与确认器一致。
    """
    config = build_config(confirm_frames=2, iou_threshold=0.30)
    confirmer = FrameConfirmer(config)
    counter = LineCounter(config, FRAME_W, FRAME_H)
    y2_seq = [170, 145, 120, 95, 70]
    events = []
    for offset, y2 in enumerate(y2_seq):
        det = {
            "cls": "person", "conf": 0.85, "box": vbox(y2),
            "center": [500, y2 - 30],
        }
        confirm_result = confirmer.update([det], offset + 1)
        result = counter.update(confirm_result, offset + 1, TEST_FPS)
        events.extend(result.events)
    assert len(confirmer.confirmed_events) == 1
    assert confirmer.confirmed_events[0]["frame_no"] == 2, "第 2 帧应确认"
    assert len(events) == 1, "端到端应恰好 1 次穿线事件: %s" % events
    ev = events[0]
    assert ev["direction"] == DIR_IN and ev["frame_no"] == 4
    assert ev["track_id"] == confirmer.confirmed_events[0]["track_id"]
    assert counter.count_in == 1 and counter.count_out == 0
    logger.info(
        "用例13 PASS：A5确认(帧2)+A6穿线(帧4) 端到端贯通，"
        "track_id=%d 一致", ev["track_id"],
    )


def run_unit_tests():
    """执行全部合成轨迹单元测试。"""
    logger.info("==== A6 越线方向计数模块 合成轨迹单元测试开始 ====")
    test_line_geometry()
    test_down_to_up_is_in()
    test_up_to_down_is_out()
    test_no_cross_no_event()
    test_deadzone_repeat_guard()
    test_legitimate_recross_counted()
    test_unconfirmed_not_counted()
    test_enter_direction_flippable()
    test_vertical_line()
    test_frame_gap_clears_memory()
    test_multi_target_same_frame()
    test_reset()
    test_integration_with_frame_confirmer()
    logger.info("==== 全部 13 组合成用例断言通过 ====")
    print("PASS: 合成轨迹单元测试 13 组用例全部通过")


def run_video_mode(args):
    """真实视频验收：全链路逐帧运行并输出穿线事件与佐证帧画面。"""
    from collections import deque

    import cv2

    from src.detector import Detector
    from src.video_source import create_video_source

    config = Config()
    if args.line:
        values = [float(v) for v in args.line.split(",")]
        assert len(values) == 4, "--line 需要 4 个归一化坐标 x1,y1,x2,y2"
        config.warning_line = tuple(values)

    source = create_video_source(args.video, config)
    detector = Detector(config)
    confirmer = FrameConfirmer(config)
    counter = LineCounter(config, source.width, source.height)
    logger.info(
        "视频验收开始：%s（%dx%d，%.2f fps），警戒线 %s -> 像素 %s，"
        "方向语义 %s",
        args.video, source.width, source.height, source.fps,
        config.warning_line, counter.line_px,
        "下->上=进、上->下=出" if config.line_enter_direction
        else "上->下=进、下->上=出",
    )

    evidence_dir = os.path.abspath(args.evidence_dir)
    os.makedirs(evidence_dir, exist_ok=True)
    # 最近若干帧缓存（帧号、原始画面、该帧各轨迹号的目标框）：
    # 事件发生时同时导出"穿线前"画面对照帧
    recent = deque(maxlen=8)

    frame_no = 0
    while True:
        frame = source.read()
        if frame is None:
            break
        frame_no += 1
        detections = detector.detect(frame)
        confirm_result = confirmer.update(detections, frame_no)
        line_result = counter.update(confirm_result, frame_no, source.fps)
        # 记录本帧各轨迹号的目标框，供事件对照帧回查
        track_boxes = {rec["track_id"]: rec["det"]["box"]
                       for rec in confirm_result.records}
        for ev in line_result.events:
            text = ("帧%d | t=%.2fs | 越线%s | %s #%d | 穿线点(%d,%d) "
                    "| 累计 进%d 出%d") % (
                ev["frame_no"], ev["time_sec"], ev["direction_cn"],
                ev["cls"], ev["track_id"],
                ev["cross_point"][0], ev["cross_point"][1],
                line_result.count_in, line_result.count_out)
            print("[穿线事件] %s" % text, flush=True)
            logger.info("[穿线事件] %s", text)
            _save_evidence_frames(evidence_dir, ev, frame, recent,
                                  counter, source.fps)
        recent.append((frame_no, frame, track_boxes))
        if args.max_frames is not None and frame_no >= args.max_frames:
            logger.info("达到 --max-frames %d，提前结束", args.max_frames)
            break
    source.release()

    # 事件清单落盘（佐证材料之一，字段与事件字典一致）
    events_path = os.path.join(evidence_dir, "cross_events.json")
    with open(events_path, "w", encoding="utf-8") as fp:
        json.dump(counter.events, fp, ensure_ascii=False, indent=2)

    summary = ("视频验收结束：共处理 %d 帧，穿线事件 %d 起（进 %d / 出 %d），"
               "事件清单 %s，佐证帧目录 %s" % (
                   frame_no, len(counter.events), counter.count_in,
                   counter.count_out, events_path, evidence_dir))
    print(summary, flush=True)
    logger.info(summary)
    if not counter.events:
        # 0 事件视为验收不通过：提示可换用 --line 指定适合该视频的线位置
        print("FAIL: 全程未检出穿线事件。若该视频在默认警戒线位置无自然"
              "穿线，可用 --line x1,y1,x2,y2 指定适合该视频的线位置重跑",
              flush=True)
        sys.exit(1)


def _save_evidence_frames(evidence_dir, ev, frame, recent, counter, fps):
    """保存穿线事件的佐证帧画面。

    每个事件保存两张图：
        1. 穿线帧：叠加警戒线、双向计数、事件目标框（红色）、锚点与
           穿线点标记及事件文字；
        2. 穿线前 5 帧的画面前照（若缓存中有）：叠加警戒线、该轨迹号
           当时的目标框（黄色）与锚点，供对照"穿线前锚点位于线的
           另一侧"。
    """
    import cv2

    def annotate(img, title):
        annotated = counter.draw_overlay(img.copy())
        x1, y1, x2, y2 = ev["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        anchor = ((x1 + x2) // 2, y2)
        cv2.circle(annotated, anchor, 5, (0, 0, 255), -1)
        cv2.drawMarker(annotated, ev["cross_point"], (0, 255, 255),
                       cv2.MARKER_CROSS, 24, 2)
        cv2.putText(annotated, title, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return annotated

    title = "CROSS %s #%d %s f%d" % (
        ev["direction"].upper(), ev["track_id"], ev["cls"], ev["frame_no"])
    name = "cross_f%05d_%s_%s_id%d" % (
        ev["frame_no"], ev["direction"], ev["cls"], ev["track_id"])
    cv2.imwrite(os.path.join(evidence_dir, name + ".png"),
                annotate(frame, title))
    # 穿线前对照帧（约 5 帧前）：画出该轨迹当时的目标框与锚点，
    # 可直接看出锚点当时位于警戒线的另一侧
    before = [item for item in recent if item[0] <= ev["frame_no"] - 5]
    if before:
        prev_no, prev_frame, prev_boxes = before[0]
        prev = counter.draw_overlay(prev_frame.copy())
        if ev["track_id"] in prev_boxes:
            bx1, by1, bx2, by2 = prev_boxes[ev["track_id"]]
            cv2.rectangle(prev, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            cv2.circle(prev, ((bx1 + bx2) // 2, by2), 5, (0, 255, 255), -1)
            cv2.putText(prev, "#%d BEFORE" % ev["track_id"],
                        (bx1, max(20, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(prev, "BEFORE f%d (t=%.2fs)" % (prev_no, prev_no / fps),
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)
        cv2.imwrite(os.path.join(
            evidence_dir, name + "_before_f%05d.png" % prev_no), prev)


def parse_args():
    """解析命令行参数（无 --video 时只跑合成单元测试）。"""
    parser = argparse.ArgumentParser(
        description="A6 越线方向计数模块 验收测试")
    parser.add_argument("--video", default=None,
                        help="真实视频验收：视频文件路径")
    parser.add_argument("--line", default=None,
                        help="覆盖警戒线：归一化坐标 x1,y1,x2,y2（默认走配置）")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="真实视频验收最多处理的帧数（默认全程）")
    parser.add_argument("--evidence-dir", default="tests/line_evidence",
                        help="穿线事件佐证帧输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    run_unit_tests()
    if args.video:
        run_video_mode(args)
        print("PASS: tests/test_line.py 合成单测与真实视频验收全部通过")


if __name__ == "__main__":
    main()
