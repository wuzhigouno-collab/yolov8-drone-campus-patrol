# -*- coding: utf-8 -*-
"""鹰眼巡校——A13 推理加速 单元测试（无需视频与真实模型，秒级完成）。

测试内容：
    一、FP16 半精度降级判定（src.detector.resolve_fp16）：
        1. 配置 fp16=False 时不启用半精度（FP32）；
        2. 配置 fp16=True 且有 CUDA 时启用半精度；
        3. 核心降级分支：配置 fp16=True 但无 CUDA 时自动降级 FP32，
           且原因文案含"降级"说明（验收要求：降级检测逻辑有单测覆盖）；
        4. 不显式传 cuda_available 时，判定结果与 torch 实时探测一致；
        5. 跟踪器初始化时的精度判定与 resolve_fp16 结论一致。

    二、跳帧轨迹外推（src.tracker.ByteTrackTracker.extrapolate）：
        6. 外推位置正确：按最近一次实测速度线性平移，steps 随帧号递增；
        7. 外推记录标记 extrapolated=True，且不推进轨迹确认计数
           （确认只认真实观测帧，确认事件发生在下一个真实检测帧）；
        8. 最近一次检测帧未再被观测到的轨迹不参与外推（无幽灵框）；
        9. 外推框越界时钳制到画面范围内；
       10. 外推锚点追加进轨迹尾迹（跳帧期间尾迹连线连续）。

    外推测试用 DummyModel 伪造 model.track() 返回（固定框与跟踪 ID），
    驱动真实 update() 路径产出轨迹状态，不依赖真实 YOLO 权重与视频。

运行方式：
    python tests/test_accel.py
全部断言通过时输出 PASS 并以退出码 0 结束；失败时抛出 AssertionError。
"""

import os
import sys

import numpy as np

# 本文件位于 tests/ 下，直接运行时 sys.path 不含项目根目录，手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.detector import resolve_fp16
from src.tracker import BACKEND_BYTETRACK, ByteTrackTracker

# 合成测试画面尺寸 640x480
FRAME_W, FRAME_H = 640, 480


def build_config(**overrides):
    """构造测试用配置：在默认配置上按关键字参数覆盖。"""
    config = Config()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_frame():
    """生成一帧纯黑合成画面（仅提供形状，内容不参与断言）。"""
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


class FakeBoxItem:
    """伪造 ultralytics 单框结果项（供 _collect_tracks 消费）。"""

    def __init__(self, cls_id, conf, xyxy, track_id):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]
        self.id = [track_id]


class FakeBoxes:
    """伪造 ultralytics boxes 容器：支持迭代且 id 非 None。"""

    def __init__(self, items):
        self._items = items
        # _collect_tracks 以 boxes.id is None 判空，这里只需非 None
        self.id = 1 if items else None

    def __iter__(self):
        return iter(self._items)


class FakeResults:
    def __init__(self, items):
        self.boxes = FakeBoxes(items)


class DummyModel:
    """伪造 YOLO 模型：track() 按预设脚本逐次返回固定跟踪结果。

    names 与真实模型同结构（cls_id -> 类别名）；track() 每次调用
    弹出 script 首元素返回，script 为空时返回空结果（模拟目标消失）。
    """

    names = {0: "person", 1: "car"}

    def __init__(self, script):
        self.script = list(script)
        self.predictor = None
        self.track_calls = 0  # 真实推理调用次数（供断言外推帧不推理）

    def track(self, frame, **kwargs):
        self.track_calls += 1
        items = self.script.pop(0) if self.script else []
        return [FakeResults(items)]


def make_tracker(script, **overrides):
    """用伪造模型构造跟踪器（跳过真实权重加载）。"""
    config = build_config(**overrides)
    return ByteTrackTracker(config, model=DummyModel(script))


# ---------------- 一、FP16 降级判定 ----------------

def test_fp16_config_off():
    """fp16=False 时不启用半精度。"""
    enabled, reason = resolve_fp16(build_config(fp16=False),
                                   cuda_available=True)
    assert enabled is False, "fp16=False 时不应启用半精度"
    assert "FP32" in reason, "原因文案应说明使用 FP32：%s" % reason


def test_fp16_enabled_with_cuda():
    """fp16=True 且有 CUDA 时启用半精度。"""
    enabled, reason = resolve_fp16(build_config(fp16=True),
                                   cuda_available=True)
    assert enabled is True, "fp16=True 且有 CUDA 时应启用半精度"
    assert "FP16" in reason, "原因文案应说明启用 FP16：%s" % reason


def test_fp16_fallback_without_cuda():
    """核心降级分支：fp16=True 但无 CUDA 时自动降级 FP32 并说明。"""
    config = build_config(fp16=True)
    enabled, reason = resolve_fp16(config, cuda_available=False)
    assert enabled is False, "无 CUDA 时必须降级为 FP32"
    assert "降级" in reason, "降级原因文案应含“降级”说明：%s" % reason
    # 降级不修改配置值本身（仅本次运行生效）
    assert config.fp16 is True, "降级不应改写配置值 fp16"


def test_fp16_auto_probe_consistent():
    """不显式传 cuda_available 时，结论与 torch 实时探测一致。"""
    import torch
    expected = bool(torch.cuda.is_available())
    enabled, _ = resolve_fp16(build_config(fp16=True))
    assert enabled == expected, \
        "实时探测结论(%s)应与 torch.cuda.is_available()(%s)一致" % (
            enabled, expected)


def test_tracker_fp16_consistent():
    """跟踪器初始化的精度判定与 resolve_fp16 结论一致。"""
    config = build_config(fp16=True)
    tracker = ByteTrackTracker(config, model=DummyModel([]))
    expected, _ = resolve_fp16(config)
    assert tracker.use_fp16 == expected, \
        "跟踪器 use_fp16(%s) 应与 resolve_fp16(%s) 一致" % (
            tracker.use_fp16, expected)


# ---------------- 二、跳帧轨迹外推 ----------------

def test_extrapolate_position():
    """外推位置按最近一次实测速度线性平移，steps 随帧号递增。"""
    # 检测帧1 框(100,100,160,160) 中心(130,130)；
    # 检测帧2 框(120,100,180,160) 中心(150,130) -> 速度 (20,0) 像素/帧
    tracker = make_tracker([
        [FakeBoxItem(0, 0.90, (100, 100, 160, 160), 7)],
        [FakeBoxItem(0, 0.91, (120, 100, 180, 160), 7)],
    ])
    tracker.update(make_frame(), 1)
    tracker.update(make_frame(), 2)
    calls_after_detect = tracker.model.track_calls

    # 外推帧3（steps=1）：中心 (170,130)，框 (140,100,200,160)
    result3 = tracker.extrapolate(make_frame(), 3)
    assert len(result3.records) == 1, "外推帧应产出 1 条轨迹记录"
    rec3 = result3.records[0]
    assert rec3["det"]["center"] == [170, 130], \
        "外推帧3 中心应为 (170,130)，实际 %s" % rec3["det"]["center"]
    assert rec3["det"]["box"] == [140, 100, 200, 160], \
        "外推帧3 框应整体平移 20px，实际 %s" % rec3["det"]["box"]
    assert rec3["track_id"] == 7, "外推应沿用原轨迹 ID"
    assert rec3["extrapolated"] is True, "外推记录应标记 extrapolated=True"

    # 外推帧4（steps=2）：中心 (190,130)
    result4 = tracker.extrapolate(make_frame(), 4)
    assert result4.records[0]["det"]["center"] == [190, 130], \
        "外推帧4 中心应为 (190,130)，实际 %s" % result4.records[0]["det"]["center"]
    # 外推帧不触发任何推理调用
    assert tracker.model.track_calls == calls_after_detect, \
        "外推帧不应调用 model.track（不应发生推理）"


def test_extrapolate_does_not_advance_confirm():
    """外推帧不推进确认计数：确认事件发生在下一个真实检测帧。"""
    # confirm_frames=3：检测帧1、2 后 age=2 未确认；外推帧3 不计数；
    # 检测帧4（真实观测）age=3 才确认，确认帧号应为 4
    tracker = make_tracker([
        [FakeBoxItem(0, 0.90, (100, 100, 160, 160), 7)],
        [FakeBoxItem(0, 0.91, (110, 100, 170, 160), 7)],
        [FakeBoxItem(0, 0.92, (130, 100, 190, 160), 7)],
    ], confirm_frames=3)
    tracker.update(make_frame(), 1)
    tracker.update(make_frame(), 2)
    result3 = tracker.extrapolate(make_frame(), 3)
    rec3 = result3.records[0]
    assert rec3["confirmed"] is False and rec3["consecutive"] == 2, \
        "外推帧不应推进确认计数（consecutive 应保持 2），实际 %d" % \
        rec3["consecutive"]
    assert not tracker.confirmed_events, "外推帧不应产生确认事件"

    result4 = tracker.update(make_frame(), 4)
    rec4 = result4.records[0]
    assert rec4["confirmed"] is True and rec4["just_confirmed"] is True, \
        "第 3 次真实观测帧应触发确认"
    assert rec4["confirm_frame"] == 4, \
        "确认帧号应为真实检测帧 4，实际 %s" % rec4["confirm_frame"]
    assert rec4["extrapolated"] is False, "真实检测帧不应标记外推"


def test_extrapolate_keeps_confirmed_state():
    """已确认轨迹在外推期间保持已确认状态（下游告警/计数不中断）。"""
    tracker = make_tracker([
        [FakeBoxItem(0, 0.90, (100, 100, 160, 160), 7)],
        [FakeBoxItem(0, 0.91, (110, 100, 170, 160), 7)],
        [FakeBoxItem(0, 0.92, (120, 100, 180, 160), 7)],
    ], confirm_frames=3)
    tracker.update(make_frame(), 1)
    tracker.update(make_frame(), 2)
    tracker.update(make_frame(), 3)  # age=3，本帧确认
    result4 = tracker.extrapolate(make_frame(), 4)
    assert result4.records[0]["confirmed"] is True, \
        "已确认轨迹在外推帧应保持已确认状态"
    assert result4.records[0]["just_confirmed"] is False, \
        "外推帧不应重复产生确认事件"


def test_vanished_track_not_extrapolated():
    """最近一次检测帧未再被观测到的轨迹不参与外推（无幽灵框）。"""
    # 检测帧1、2 有目标，检测帧3 目标消失（空结果）
    tracker = make_tracker([
        [FakeBoxItem(0, 0.90, (100, 100, 160, 160), 7)],
        [FakeBoxItem(0, 0.91, (110, 100, 170, 160), 7)],
        [],  # 帧3 目标消失
    ])
    tracker.update(make_frame(), 1)
    tracker.update(make_frame(), 2)
    tracker.update(make_frame(), 3)
    result4 = tracker.extrapolate(make_frame(), 4)
    assert len(result4.records) == 0, \
        "目标在最近检测帧已消失，外推帧不应产出幽灵框，实际 %d 条" % \
        len(result4.records)


def test_extrapolate_clamped_to_frame():
    """外推框越界时钳制到画面范围内。"""
    # 速度 (100,0)：检测帧1 中心(530,240)，检测帧2 中心(630,240)
    tracker = make_tracker([
        [FakeBoxItem(0, 0.90, (500, 200, 560, 280), 7)],
        [FakeBoxItem(0, 0.91, (600, 200, 660, 280), 7)],
    ])
    tracker.update(make_frame(), 1)
    tracker.update(make_frame(), 2)  # 帧2 框已部分越出 640 宽画面
    # 外推帧3：按速度中心将到 (730,240)，框宽 60 -> 钳制到 x1=580
    result3 = tracker.extrapolate(make_frame(), 3)
    box = result3.records[0]["det"]["box"]
    assert box[2] <= FRAME_W and box[3] <= FRAME_H, \
        "外推框应钳制在画面 %dx%d 内，实际 %s" % (FRAME_W, FRAME_H, box)
    assert box[0] >= 0 and box[1] >= 0, \
        "外推框坐标不应为负，实际 %s" % box


def test_extrapolate_extends_trail():
    """外推锚点追加进轨迹尾迹（跳帧期间尾迹连线连续）。"""
    tracker = make_tracker([
        [FakeBoxItem(0, 0.90, (100, 100, 160, 160), 7)],
        [FakeBoxItem(0, 0.91, (120, 100, 180, 160), 7)],
    ])
    tracker.update(make_frame(), 1)
    tracker.update(make_frame(), 2)
    trail_before = len(tracker.get_trail(7))
    tracker.extrapolate(make_frame(), 3)
    trail_after = len(tracker.get_trail(7))
    assert trail_after == trail_before + 1, \
        "外推帧应向尾迹追加 1 个外推锚点（%d -> %d）" % (
            trail_before, trail_after)
    # 外推锚点为外推框底部中心 (170, 160)
    assert tracker.get_trail(7)[-1] == (170, 160), \
        "外推锚点应为外推框底部中心 (170,160)，实际 %s" % (
            tracker.get_trail(7)[-1],)


def test_extrapolate_backend_unchanged():
    """外推结果的确认后端标识保持 bytetrack（与实测帧一致）。"""
    tracker = make_tracker([
        [FakeBoxItem(0, 0.90, (100, 100, 160, 160), 7)],
    ])
    tracker.update(make_frame(), 1)
    result = tracker.extrapolate(make_frame(), 2)
    assert result.backend == BACKEND_BYTETRACK, \
        "外推结果 backend 应保持 bytetrack，实际 %s" % result.backend


def main():
    """依次执行全部断言，全部通过输出 PASS。"""
    tests = [
        test_fp16_config_off,
        test_fp16_enabled_with_cuda,
        test_fp16_fallback_without_cuda,
        test_fp16_auto_probe_consistent,
        test_tracker_fp16_consistent,
        test_extrapolate_position,
        test_extrapolate_does_not_advance_confirm,
        test_extrapolate_keeps_confirmed_state,
        test_vanished_track_not_extrapolated,
        test_extrapolate_clamped_to_frame,
        test_extrapolate_extends_trail,
        test_extrapolate_backend_unchanged,
    ]
    for test in tests:
        test()
        print("PASS %s" % test.__name__)
    print("全部 %d 项断言通过：PASS" % len(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())
