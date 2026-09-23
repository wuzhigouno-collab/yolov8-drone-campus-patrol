# -*- coding: utf-8 -*-
"""鹰眼巡校——阶段 C VisDrone 10->5 类别映射模块（src.class_map）验收测试。

测试内容（纯单测，无需真实模型/视频，秒级完成）：
    1. 常量表完整性：VisDrone 10 类、3 个丢弃类、映射值域为 5 类业务类别；
    2. 10 类全量映射：7 个保留类逐一映射正确，3 个丢弃类返回 None；
    3. 生效判定——开关开启 + VisDrone 10 类名：映射生效（dict 与 list
       两种 names 形式一致）；
    4. 不生效判定——COCO 类名（开关开启但模型为 COCO 80 类名）：映射
       完全不改变行为（恒等透传，连与 VisDrone 同名的类也不变）；
    5. 不生效判定——开关关闭（即使模型为 VisDrone 类名）：恒等透传；
    6. 端到端漏斗：生效态经 src.postprocess 公共后处理后，类别名归并、
       丢弃类消失、过滤作用于映射后类别名；不生效态输出与无映射一致；
    7. 防御性兜底：映射表外的未知类别名原样返回；Config 默认开关为 True。

运行方式：
    python tests/test_class_map.py
全部断言通过时输出 PASS 并以退出码 0 结束；失败时抛出 AssertionError。
"""

import os
import sys

# 本文件位于 tests/ 下，直接运行时 sys.path 不含项目根目录，手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.class_map import (
    VISDRONE_CLASSES,
    VISDRONE_DROP_CLASSES,
    VISDRONE_TO_BUSINESS,
    build_class_mapper,
    map_visdrone_class,
)
from src.config import Config
from src.logger import get_logger

logger = get_logger("test_class_map")

# 模型 names 为 VisDrone 10 类时的 dict 形式（cls_id -> 类别名）
VISDRONE_NAMES_DICT = {i: name for i, name in enumerate(VISDRONE_CLASSES)}

# 当前 COCO 预训练权重的代表性类别名（含大量 VisDrone 集合外的类名，
# 与 yolov8n.pt 实际 names 前若干类一致）
COCO_NAMES_DICT = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog",
}

# VisDrone 10 类 -> 业务类别 的期望映射（None 表示丢弃）
EXPECTED_MAPPING = {
    "pedestrian": "person",
    "people": "person",
    "bicycle": "bicycle",
    "car": "car",
    "van": "car",
    "truck": None,
    "tricycle": None,
    "awning-tricycle": None,
    "bus": "bus",
    "motor": "motorcycle",
}


def build_config(class_map_enabled=True):
    """构造测试用配置：仅覆盖类别映射开关。"""
    config = Config()
    config.class_map_enabled = class_map_enabled
    return config


# ---------------- 1. 常量表完整性 ----------------

def test_constants():
    """用例 1：10 类全、3 个丢弃类、映射值域恰为 5 类业务类别。"""
    assert len(VISDRONE_CLASSES) == 10, "VisDrone 应为 10 类"
    assert len(set(VISDRONE_CLASSES)) == 10, "10 类不应有重名"
    assert sorted(VISDRONE_DROP_CLASSES) == \
        ["awning-tricycle", "tricycle", "truck"], "丢弃类应为 3 个"
    # 丢弃类与保留类互斥，且并集恰为 10 类全集
    assert not set(VISDRONE_DROP_CLASSES) & set(VISDRONE_TO_BUSINESS)
    assert set(VISDRONE_DROP_CLASSES) | set(VISDRONE_TO_BUSINESS) \
        == set(VISDRONE_CLASSES)
    assert set(VISDRONE_TO_BUSINESS.values()) == \
        {"person", "bicycle", "motorcycle", "car", "bus"}, "应为 5 类业务类别"
    logger.info("用例1 PASS：10 类全集 = 7 保留类 + 3 丢弃类；"
                "映射值域为 person/bicycle/motorcycle/car/bus 五类")


# ---------------- 2. 10 类全量映射 ----------------

def test_full_mapping_10_classes():
    """用例 2：10 类逐一映射正确（含 3 个丢弃类返回 None）。"""
    for name in VISDRONE_CLASSES:
        got = map_visdrone_class(name)
        assert got == EXPECTED_MAPPING[name], \
            "%s 映射应为 %s，实际 %s" % (name, EXPECTED_MAPPING[name], got)
    logger.info("用例2 PASS：%s",
                "；".join("%s->%s" % (k, v) for k, v in EXPECTED_MAPPING.items()))


def test_drop_classes():
    """用例 2b：三个丢弃类单独断言——返回 None 且不进入映射表。"""
    for name in ("truck", "tricycle", "awning-tricycle"):
        assert map_visdrone_class(name) is None, "%s 应被丢弃" % name
        assert name not in VISDRONE_TO_BUSINESS
    logger.info("用例2b PASS：truck/tricycle/awning-tricycle 均返回 None（丢弃）")


# ---------------- 3. 生效判定：开关开启 + VisDrone 类名 ----------------

def test_active_with_visdrone_names():
    """用例 3：开关开启且模型类名为 VisDrone 10 类时映射生效。"""
    mapper, active, reason = build_class_mapper(
        build_config(True), VISDRONE_NAMES_DICT)
    assert active is True, "VisDrone 类名下映射应生效"
    assert "启用" in reason and "10->5" in reason
    for name in VISDRONE_CLASSES:
        assert mapper(name) == EXPECTED_MAPPING[name]

    # names 为 list 形式（按类别序）时判定结果一致
    mapper_list, active_list, _ = build_class_mapper(
        build_config(True), list(VISDRONE_CLASSES))
    assert active_list is True
    assert mapper_list("pedestrian") == "person"
    assert mapper_list("truck") is None
    logger.info("用例3 PASS：开关开启 + VisDrone 10 类名映射生效"
                "（dict/list 两种 names 形式一致）")


# ---------------- 4. 不生效判定：COCO 类名 ----------------

def test_inactive_with_coco_names():
    """用例 4：开关开启但模型为 COCO 类名时映射完全不生效（恒等）。"""
    mapper, active, reason = build_class_mapper(
        build_config(True), COCO_NAMES_DICT)
    assert active is False, "COCO 类名下映射不得生效"
    assert "恒等" in reason, "原因文案应说明恒等透传：%s" % reason
    # 恒等透传：全部 COCO 类名原样返回、永不返回 None——包括与
    # VisDrone 同名的 car/bicycle/bus/truck（同名也不触发映射/丢弃）
    for name in COCO_NAMES_DICT.values():
        assert mapper(name) == name, "COCO 类名 %s 应原样透传" % name
    assert mapper("car") == "car" and mapper("truck") == "truck"
    logger.info("用例4 PASS：COCO 类名（含与 VisDrone 同名类）全部恒等透传，"
                "映射完全不改变现有行为")


# ---------------- 5. 不生效判定：开关关闭 ----------------

def test_inactive_when_switch_off():
    """用例 5：开关关闭时即使模型为 VisDrone 类名，映射也不生效。"""
    mapper, active, reason = build_class_mapper(
        build_config(False), VISDRONE_NAMES_DICT)
    assert active is False, "开关关闭时映射不得生效"
    assert "恒等" in reason
    # 细粒度类名原样透传，丢弃类也不丢
    for name in VISDRONE_CLASSES:
        assert mapper(name) == name, "开关关闭时 %s 应原样透传" % name
    assert mapper("pedestrian") == "pedestrian"
    assert mapper("truck") == "truck"
    logger.info("用例5 PASS：开关关闭 + VisDrone 类名仍恒等透传"
                "（pedestrian 不归并、truck 不丢弃）")


# ---------------- 6. 端到端漏斗（经公共后处理） ----------------

class FakeBoxItem:
    """伪造 ultralytics 单框结果项（供 postprocess 漏斗消费）。"""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]
        self.id = None


def test_end_to_end_funnel():
    """用例 6：生效/不生效两种形态下，公共后处理输出与预期一致。"""
    from src.postprocess import collect_detections
    boxes = [
        FakeBoxItem(0, 0.90, (0.0, 0.0, 9.0, 9.0)),   # pedestrian
        FakeBoxItem(4, 0.80, (0.0, 0.0, 9.0, 9.0)),   # van
        FakeBoxItem(5, 0.70, (0.0, 0.0, 9.0, 9.0)),   # truck
        FakeBoxItem(9, 0.60, (0.0, 0.0, 9.0, 9.0)),   # motor
    ]
    config = build_config(True)

    # 生效态：pedestrian->person、van->car、motor->motorcycle 归并保留，
    # truck 丢弃；过滤作用于映射后类别名（motorcycle 不在关注集被滤）
    mapper, active, _ = build_class_mapper(config, VISDRONE_NAMES_DICT)
    assert active is True
    dets = collect_detections(boxes, VISDRONE_NAMES_DICT,
                              ["person", "car", "bicycle"], map_cls=mapper)
    assert [d["cls"] for d in dets] == ["person", "car"], \
        "生效态输出应为归并后类别且 truck/motorcycle 不出现：%s" % dets

    # 不生效态（COCO 类名恒等映射）：输出与 map_cls=None 完全一致
    coco_mapper, coco_active, _ = build_class_mapper(config, COCO_NAMES_DICT)
    assert coco_active is False
    coco_boxes = [
        FakeBoxItem(0, 0.90, (0.0, 0.0, 9.0, 9.0)),   # person
        FakeBoxItem(2, 0.80, (0.0, 0.0, 9.0, 9.0)),   # car
        FakeBoxItem(7, 0.70, (0.0, 0.0, 9.0, 9.0)),   # truck（COCO 同名不丢）
        FakeBoxItem(16, 0.60, (0.0, 0.0, 9.0, 9.0)),  # dog
    ]
    with_map = collect_detections(coco_boxes, COCO_NAMES_DICT,
                                  ["person", "car", "bicycle"],
                                  map_cls=coco_mapper)
    without_map = collect_detections(coco_boxes, COCO_NAMES_DICT,
                                     ["person", "car", "bicycle"],
                                     map_cls=None)
    assert with_map == without_map, \
        "COCO 类名下加不加映射输出必须完全一致：%s vs %s" % (with_map, without_map)
    assert [d["cls"] for d in with_map] == ["person", "car"]
    logger.info("用例6 PASS：生效态 truck 丢弃、pedestrian/van 归并；"
                "COCO 恒等态输出与无映射逐值一致")


# ---------------- 7. 防御性兜底与默认配置 ----------------

def test_fallback_and_default():
    """用例 7：未知类别名原样返回；Config 默认开关为 True。"""
    assert map_visdrone_class("unknown-thing") == "unknown-thing"
    assert Config().class_map_enabled is True, \
        "默认配置应开启类别映射开关（COCO 权重下靠类名集合判定保持恒等）"
    # names 为空表时不生效（防御性分支，不抛异常）
    _, active, _ = build_class_mapper(build_config(True), {})
    assert active is False
    logger.info("用例7 PASS：未知类名原样返回；默认开关 True；空 names 不生效")


def main():
    logger.info("==== 阶段 C VisDrone 类别映射模块 验收测试开始 ====")
    test_constants()
    test_full_mapping_10_classes()
    test_drop_classes()
    test_active_with_visdrone_names()
    test_inactive_with_coco_names()
    test_inactive_when_switch_off()
    test_end_to_end_funnel()
    test_fallback_and_default()
    logger.info("==== 全部 8 组用例断言通过 ====")
    print("PASS: tests/test_class_map.py 全部断言通过")


if __name__ == "__main__":
    main()
