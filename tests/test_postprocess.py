# -*- coding: utf-8 -*-
"""鹰眼巡校——阶段 C 公共后处理模块（src.postprocess）验收测试。

测试内容（纯单测，伪造 YOLO 结果对象，无需真实模型/视频，秒级完成）：
    1. 类别过滤：仅保留 target_classes 关注类别；配置为空/None 时不过滤；
    2. 字段结构：键集合精确为 {cls, conf, box, center}，置信度 round(4)，
       框坐标 int 取整，中心点由原始浮点坐标计算（与重构前逐值一致），
       输出顺序与输入框顺序一致；
    3. 空结果：无框 / 全部被过滤时返回空列表；
    4. 切片语义：box_transform 坐标映射生效、中心点由映射后整数框计算、
       extra_fields（slice 追溯字段）原样合并且键序与重构前一致；
    5. 跟踪结果收集：boxes 为 None / boxes.id 为 None / 单框 id 为 None
       三种跳过路径，返回 (检测项, 跟踪ID) 列表；
    6. 类别映射漏斗：映射函数改名后过滤作用于新名，映射返回 None 的
       类别被丢弃；
    7. 三条路径接线：Detector.detect / ByteTrackTracker._collect_tracks /
       SlicedDetector.detect 实际调用公共实现，输出与直接调用
       collect_detections / collect_track_detections 逐值一致。

运行方式：
    python tests/test_postprocess.py
全部断言通过时输出 PASS 并以退出码 0 结束；失败时抛出 AssertionError。
"""

import os
import sys

import numpy as np

# 本文件位于 tests/ 下，直接运行时 sys.path 不含项目根目录，手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.class_map import identity_class_name
from src.config import Config
from src.logger import get_logger
from src.postprocess import collect_detections, collect_track_detections

logger = get_logger("test_postprocess")

# 伪造模型的类别名表（含一个非关注类别 dog 用于过滤断言）
NAMES = {0: "person", 1: "car", 2: "dog", 3: "bicycle"}


def build_config(**overrides):
    """构造测试用配置：在默认配置上按关键字参数覆盖。"""
    config = Config()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class FakeBoxItem:
    """伪造 ultralytics 单框结果项（cls/conf/xyxy/id 四个属性）。"""

    def __init__(self, cls_id, conf, xyxy, track_id=0):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]
        # track_id 为 None 时表示该框未分配跟踪 ID（track 路径跳过）
        self.id = None if track_id is None else [track_id]


class FakeBoxes:
    """伪造 ultralytics boxes 容器：可迭代，id 属性模拟整体有无跟踪 ID。"""

    def __init__(self, items, has_ids=True):
        self._items = list(items)
        self.id = 1 if (has_ids and self._items) else None

    def __iter__(self):
        return iter(self._items)


class FakeResults:
    def __init__(self, items, has_ids=True):
        self.boxes = FakeBoxes(items, has_ids=has_ids)


class DummyModel:
    """伪造 YOLO 模型：predict/track 按预设结果返回，记录调用次数。"""

    names = NAMES

    def __init__(self, predict_items=None, track_items=None):
        self.predict_items = predict_items or []
        self.track_items = track_items or []
        self.predictor = None
        self.predict_calls = 0
        self.track_calls = 0

    def predict(self, frame, **kwargs):
        self.predict_calls += 1
        return [FakeResults(self.predict_items)]

    def track(self, frame, **kwargs):
        self.track_calls += 1
        return [FakeResults(self.track_items)]


def make_frame(h=480, w=640):
    """生成一帧纯黑合成画面（仅提供形状，内容不参与断言）。"""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------- 1. 类别过滤 ----------------

def test_filter_by_target_classes():
    """用例 1：仅保留关注类别；target_classes 为空/None 时不过滤。"""
    items = [
        FakeBoxItem(0, 0.9, (10.0, 10.0, 50.0, 60.0)),   # person
        FakeBoxItem(2, 0.8, (70.0, 10.0, 110.0, 60.0)),  # dog（非关注）
        FakeBoxItem(1, 0.7, (10.0, 70.0, 50.0, 120.0)),  # car
    ]
    dets = collect_detections(FakeBoxes(items), NAMES,
                              ["person", "car", "bicycle"])
    assert [d["cls"] for d in dets] == ["person", "car"], \
        "非关注类别 dog 应被过滤：%s" % dets

    # 配置为空列表：不过滤（与重构前 `if target_classes and ...` 语义一致）
    all_dets = collect_detections(FakeBoxes(items), NAMES, [])
    assert [d["cls"] for d in all_dets] == ["person", "dog", "car"]
    # 配置为 None：同样不过滤
    none_dets = collect_detections(FakeBoxes(items), NAMES, None)
    assert len(none_dets) == 3
    logger.info("用例1 PASS：dog 被过滤；target_classes 空/None 时不过滤")


# ---------------- 2. 字段结构 ----------------

def test_field_structure():
    """用例 2：键集合精确、conf round(4)、坐标取整规则与重构前一致。"""
    # 取整判别样本：x1=10.9, x2=21.9 —— 中心点按原始浮点算为
    # int((10.9+21.9)/2)=16；若误按取整框算则为 (10+21)//2=15
    items = [FakeBoxItem(0, 0.123456789, (10.9, 20.9, 21.9, 42.9))]
    dets = collect_detections(FakeBoxes(items), NAMES, ["person"])
    assert len(dets) == 1
    det = dets[0]
    assert set(det.keys()) == {"cls", "conf", "box", "center"}, \
        "键集合应为契约四键：%s" % det
    assert list(det.keys()) == ["cls", "conf", "box", "center"], \
        "键插入顺序应与重构前一致：%s" % list(det.keys())
    assert det["cls"] == "person"
    assert det["conf"] == 0.1235, "置信度应 round(4)：%s" % det["conf"]
    assert det["box"] == [10, 20, 21, 42], "框坐标应 int 取整：%s" % det["box"]
    assert det["center"] == [16, 31], \
        "中心点应由原始浮点坐标计算（非取整框）：%s" % det["center"]
    for value in det["box"] + det["center"]:
        assert isinstance(value, int), "坐标应为 int：%s" % det

    # 输出顺序与输入框顺序一致（不做任何排序）
    ordered = [
        FakeBoxItem(1, 0.5, (0.0, 0.0, 9.0, 9.0)),
        FakeBoxItem(0, 0.9, (0.0, 0.0, 9.0, 9.0)),
        FakeBoxItem(3, 0.7, (0.0, 0.0, 9.0, 9.0)),
    ]
    dets = collect_detections(FakeBoxes(ordered), NAMES,
                              ["person", "car", "bicycle"])
    assert [d["cls"] for d in dets] == ["car", "person", "bicycle"]
    logger.info(
        "用例2 PASS：四键结构与键序精确；conf=0.1235(round4)；"
        "框 [10,20,21,42](int 取整)；中心 [16,31](浮点口径)；顺序保持")


# ---------------- 3. 空结果 ----------------

def test_empty_results():
    """用例 3：无框与全部被过滤两种空结果路径。"""
    assert collect_detections(FakeBoxes([]), NAMES, ["person"]) == []
    items = [FakeBoxItem(2, 0.9, (0.0, 0.0, 9.0, 9.0))]  # 仅 dog
    assert collect_detections(FakeBoxes(items), NAMES, ["person"]) == []
    # 跟踪路径：boxes 为 None / boxes.id 为 None 同样返回空列表
    assert collect_track_detections(None, NAMES, ["person"]) == []
    no_ids = FakeBoxes([FakeBoxItem(0, 0.9, (0.0, 0.0, 9.0, 9.0))],
                       has_ids=False)
    assert collect_track_detections(no_ids, NAMES, ["person"]) == []
    logger.info("用例3 PASS：无框/全过滤/boxes None/无跟踪ID 均返回空列表")


# ---------------- 4. 切片语义 ----------------

def test_box_transform_and_extra_fields():
    """用例 4：box_transform 映射、中心点按映射后整数框、extra 合并。"""
    items = [FakeBoxItem(0, 0.87654, (10.6, 20.4, 110.2, 120.8))]

    def transform(box):
        # 模拟切片偏移 (853, 480) 映射回原图（同 map_box_to_frame 取整）
        return [int(box[0]) + 853, int(box[1]) + 480,
                int(box[2]) + 853, int(box[3]) + 480]

    dets = collect_detections(FakeBoxes(items), NAMES, ["person"],
                              box_transform=transform,
                              extra_fields={"slice": 1})
    assert len(dets) == 1
    det = dets[0]
    assert det["box"] == [863, 500, 963, 600], "变换后框：%s" % det["box"]
    # 中心点由映射后的整数框计算（切片路径口径）
    assert det["center"] == [int((863 + 963) / 2), int((500 + 600) / 2)]
    assert det["slice"] == 1, "extra_fields 应原样合并：%s" % det
    assert list(det.keys()) == ["cls", "conf", "box", "center", "slice"], \
        "键序应与重构前 SlicedDetector 输出一致：%s" % list(det.keys())
    assert det["conf"] == 0.8765
    logger.info("用例4 PASS：box_transform 偏移映射正确、中心按整数框、"
                "slice 字段并入且键序 cls/conf/box/center/slice")


# ---------------- 5. 跟踪结果收集 ----------------

def test_collect_track_detections():
    """用例 5：跳过未分配 ID 的框，返回 (检测项, 跟踪ID) 并做类别过滤。"""
    items = [
        FakeBoxItem(0, 0.9, (10.0, 10.0, 50.0, 60.0), track_id=7),
        FakeBoxItem(2, 0.8, (70.0, 10.0, 110.0, 60.0), track_id=8),  # dog
        FakeBoxItem(1, 0.7, (10.0, 70.0, 50.0, 120.0), track_id=None),
        FakeBoxItem(1, 0.6, (60.0, 70.0, 100.0, 120.0), track_id=9),
    ]
    collected = collect_track_detections(FakeBoxes(items), NAMES,
                                         ["person", "car", "bicycle"])
    assert len(collected) == 2, "dog 被过滤、无 ID 框被跳过：%s" % collected
    det0, tid0 = collected[0]
    det1, tid1 = collected[1]
    assert (tid0, tid1) == (7, 9), "跟踪 ID 应为 int：%s" % ((tid0, tid1),)
    assert det0["cls"] == "person" and det1["cls"] == "car"
    assert set(det0.keys()) == {"cls", "conf", "box", "center"}
    assert det0["box"] == [10, 10, 50, 60] and det0["center"] == [30, 35]
    logger.info("用例5 PASS：dog 过滤、无ID框跳过；保留 (person,#7) "
                "与 (car,#9)，检测项结构与整图路径一致")


# ---------------- 6. 类别映射漏斗 ----------------

def test_map_cls_funnel():
    """用例 6：映射改名后过滤作用于新名；映射返回 None 的类别被丢弃。"""
    visdrone_names = {0: "pedestrian", 1: "van", 2: "truck"}
    items = [
        FakeBoxItem(0, 0.90, (0.0, 0.0, 9.0, 9.0)),   # pedestrian->person
        FakeBoxItem(1, 0.80, (0.0, 0.0, 9.0, 9.0)),   # van->car
        FakeBoxItem(2, 0.70, (0.0, 0.0, 9.0, 9.0)),   # truck->丢弃
    ]

    def mapper(name):
        return {"pedestrian": "person", "van": "car"}.get(name)

    dets = collect_detections(FakeBoxes(items), visdrone_names,
                              ["person", "car"], map_cls=mapper)
    assert [d["cls"] for d in dets] == ["person", "car"], \
        "truck 应被映射丢弃，pedestrian/van 归并后保留：%s" % dets

    # 过滤作用于映射后的新名：只关注 person 时 van->car 也被滤掉
    only_person = collect_detections(FakeBoxes(items), visdrone_names,
                                     ["person"], map_cls=mapper)
    assert [d["cls"] for d in only_person] == ["person"]

    # 恒等映射（map_cls=None）下按原始名过滤：pedestrian 不在关注集
    raw = collect_detections(FakeBoxes(items), visdrone_names,
                             ["person", "car"], map_cls=None)
    assert raw == []
    logger.info("用例6 PASS：pedestrian->person、van->car 保留；truck 丢弃；"
                "过滤作用于映射后类别名；恒等路径按原始名过滤")


# ---------------- 7. 三条路径接线 ----------------

def test_detector_delegates():
    """用例 7a：Detector.detect 实际走公共实现，输出逐值一致。"""
    from src.detector import Detector
    items = [
        FakeBoxItem(0, 0.912345678, (10.9, 20.9, 21.9, 42.9)),
        FakeBoxItem(2, 0.80, (0.0, 0.0, 9.0, 9.0)),  # dog 被过滤
    ]
    config = build_config()
    # 跳过 __init__ 的权重加载，手工装配与 __init__ 相同的属性
    det = Detector.__new__(Detector)
    det.config = config
    det.names = NAMES
    det.use_fp16 = False
    det.map_cls = identity_class_name
    det.model = DummyModel(predict_items=items)

    got = det.detect(make_frame())
    expect = collect_detections(FakeBoxes(items), NAMES,
                                config.target_classes,
                                map_cls=identity_class_name)
    assert got == expect, "Detector.detect 输出应与公共实现逐值一致"
    assert [d["cls"] for d in got] == ["person"]
    assert got[0]["center"] == [16, 31]
    logger.info("用例7a PASS：Detector.detect 委托公共实现，"
                "输出与直接调用 collect_detections 逐值一致")


def test_tracker_delegates():
    """用例 7b：ByteTrackTracker 的 track/predict 两路径均走公共实现。"""
    from src.tracker import BACKEND_BYTETRACK, ByteTrackTracker
    items = [
        FakeBoxItem(0, 0.912345678, (10.9, 20.9, 21.9, 42.9), track_id=7),
        FakeBoxItem(2, 0.80, (0.0, 0.0, 9.0, 9.0), track_id=8),  # dog
    ]
    config = build_config(confirm_frames=1)
    tracker = ByteTrackTracker(config, model=DummyModel(track_items=items))
    assert tracker.backend == BACKEND_BYTETRACK

    result = tracker.update(make_frame(), 0)
    assert len(result.records) == 1, "dog 应被过滤：%s" % result.records
    record = result.records[0]
    assert record["track_id"] == 7 and record["confirmed"] is True
    expect = collect_track_detections(FakeBoxes(items), NAMES,
                                      config.target_classes,
                                      map_cls=identity_class_name)
    assert record["det"] == expect[0][0], \
        "track 路径检测项应与公共实现逐值一致：%s" % record["det"]
    assert record["det"]["center"] == [16, 31]

    # 降级路径（_predict_detections）：输出同样走公共实现
    predict_items = [FakeBoxItem(1, 0.654321, (1.0, 2.0, 30.5, 40.5))]
    tracker.model = DummyModel(predict_items=predict_items)
    dets = tracker._predict_detections(make_frame())
    expect_dets = collect_detections(FakeBoxes(predict_items), NAMES,
                                     config.target_classes,
                                     map_cls=identity_class_name)
    assert dets == expect_dets and dets[0]["cls"] == "car"
    assert dets[0]["conf"] == 0.6543 and dets[0]["box"] == [1, 2, 30, 40]
    logger.info("用例7b PASS：track 路径与降级 predict 路径检测项"
                "均与公共实现逐值一致（轨迹#7 确认、dog 过滤）")


def test_slicer_delegates():
    """用例 7c：SlicedDetector.detect 走公共实现（含 slice 字段与 NMS）。"""
    from src.slicer import SlicedDetector
    items = [FakeBoxItem(0, 0.91234, (10.0, 10.0, 50.0, 60.0))]
    config = build_config()
    slicer = SlicedDetector(config, model=DummyModel(predict_items=items))
    dets = slicer.detect(make_frame(h=80, w=100))
    # 2x2 切片、每片返回同一框：映射回原图后位置各异，NMS 不互杀
    assert slicer.model.predict_calls == 4, "2x2 切片应推理 4 次"
    assert len(dets) == 4
    for det in dets:
        assert list(det.keys()) == ["cls", "conf", "box", "center", "slice"]
        assert det["cls"] == "person" and det["conf"] == 0.9123
        assert 0 <= det["slice"] < 4
    # 左上限切片（序号 0，偏移 0,0）框位置与单片检出一致
    first = [d for d in dets if d["slice"] == 0][0]
    assert first["box"] == [10, 10, 50, 60] and first["center"] == [30, 35]
    logger.info("用例7c PASS：SlicedDetector 4 片推理均走公共实现，"
                "输出含 slice 追溯字段、键序与重构前一致")


def main():
    logger.info("==== 阶段 C 公共后处理模块 验收测试开始 ====")
    test_filter_by_target_classes()
    test_field_structure()
    test_empty_results()
    test_box_transform_and_extra_fields()
    test_collect_track_detections()
    test_map_cls_funnel()
    test_detector_delegates()
    test_tracker_delegates()
    test_slicer_delegates()
    logger.info("==== 全部 9 组用例断言通过 ====")
    print("PASS: tests/test_postprocess.py 全部断言通过")


if __name__ == "__main__":
    main()
