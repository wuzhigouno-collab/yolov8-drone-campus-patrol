# -*- coding: utf-8 -*-
"""鹰眼巡校——阶段 C 统一检测引擎（src.engine.DetectionEngine）验收测试。

测试内容：
    1. 单例语义（真实默认权重 weights/yolov8n.pt）：shared() 反复调用
       返回同一实例；Detector/ByteTrackTracker/SlicedDetector 不传模型
       时经共享引擎持有同一 model 对象（is 判定）、同一套 FP16/类别
       映射判定结论，且全程只有一条权重加载日志；
    2. FP16 判定委托（伪造模型，无需真实权重与 GPU）：引擎
       use_fp16/fp16_reason 与 resolve_fp16 直接调用逐值一致（含
       注入 cuda_available 的启用/降级分支与 fp16=False 分支）；
    3. 类别映射判定委托（伪造模型）：引擎 map_cls/class_map_active/
       class_map_reason 与 build_class_mapper 直接调用一致
       （VisDrone 生效 / COCO 恒等 / 开关关闭三组）；
    4. names 来源：引擎 names 即其所持模型的 names（伪造模型与真实
       权重两种形态），三条路径的 names 与引擎同源；
    5. E6 换权重就绪（真实第二权重 weights/yolo26n.pt）：Config 仅改
       weights_path，三条路径经共享引擎同学到新权重——model 为同一
       对象、names 一致且与直接加载该权重文件得到的 names 逐值一致；
       使用既有权重文件，不产生临时文件，测完清空共享缓存。

运行方式：
    python tests/test_engine.py
全部断言通过时输出 PASS 并以退出码 0 结束；失败时抛出 AssertionError。
"""

import logging
import os
import sys

# 本文件位于 tests/ 下，直接运行时 sys.path 不含项目根目录，手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.class_map import VISDRONE_CLASSES, build_class_mapper
from src.config import Config
from src.detector import Detector
from src.engine import DetectionEngine, resolve_fp16
from src.logger import get_logger
from src.slicer import SlicedDetector
from src.tracker import ByteTrackTracker

logger = get_logger("test_engine")

# 伪造模型的 COCO 风格类别名表（不满足 VisDrone 10 类集合，映射应恒等）
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle"}
# 伪造模型的 VisDrone 10 类名表（满足生效条件，映射应启用）
VISDRONE_NAMES = {i: name for i, name in enumerate(VISDRONE_CLASSES)}

# E6 换权重验收用的第二权重（A10 对比实验既有权重，非临时文件）
ALT_WEIGHTS = os.path.join("weights", "yolo26n.pt")


def build_config(**overrides):
    """构造测试用配置：在默认配置上按关键字参数覆盖。"""
    config = Config()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class DummyModel:
    """伪造 YOLO 模型：仅提供引擎判定所需的 names/predictor 属性。"""

    def __init__(self, names):
        self.names = names
        self.predictor = None


# ---------------- 1. 单例语义（真实默认权重） ----------------

def test_singleton_shared_engine():
    """用例 1：三路径共享同一引擎/模型，全程只有一条权重加载日志。"""
    DetectionEngine.reset_shared()  # 测试隔离：清空共享缓存
    records = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    # 挂到 root logger：各命名 logger 默认向上传播，不影响其自有处理器
    root_logger = logging.getLogger()
    handler = ListHandler()
    root_logger.addHandler(handler)
    try:
        config = Config()
        engine = DetectionEngine.shared(config)
        # 另一个 Config 实例、同一权重路径：仍返回同一引擎实例
        assert DetectionEngine.shared(Config()) is engine, \
            "同一权重路径的 shared() 应返回同一实例"

        detector = Detector(config)
        tracker = ByteTrackTracker(config)
        slicer = SlicedDetector(config)
        # 三路径共用同一引擎实例、同一模型实例（is 判定）
        assert detector.engine is engine, "Detector 应持有共享引擎"
        assert tracker.engine is engine, "ByteTrackTracker 应持有共享引擎"
        assert slicer.engine is engine, "SlicedDetector 应持有共享引擎"
        assert detector.model is tracker.model is slicer.model, \
            "三条路径持有的 model 应为同一对象"
        # names 与判定结论同源
        assert detector.names is tracker.names is slicer.names is engine.names
        assert detector.use_fp16 == tracker.use_fp16 == engine.use_fp16
        assert detector.map_cls is tracker.map_cls is slicer.map_cls
        assert detector.class_map_active == tracker.class_map_active \
            == slicer.class_map_active == engine.class_map_active

        # 权重加载日志全程只有一条，且指向默认权重
        loads = [m for m in records if "正在加载 YOLO 模型权重" in m]
        assert len(loads) == 1, \
            "三路径装配全程权重加载日志应只有一条，实际 %d 条：%s" % (
                len(loads), loads)
        assert "yolov8n.pt" in loads[0], "加载的应为默认权重：%s" % loads[0]
    finally:
        root_logger.removeHandler(handler)
    logger.info(
        "用例1 PASS：shared() 单例成立；Detector/ByteTrackTracker/"
        "SlicedDetector 共用同一引擎与模型（is 判定）；"
        "names/FP16/类别映射判定同源；权重加载日志仅 1 条（%s）",
        engine.weights_path,
    )


# ---------------- 2. FP16 判定委托 ----------------

def test_fp16_delegation():
    """用例 2：引擎 FP16 判定与 resolve_fp16 直接调用逐值一致。"""
    config = build_config(fp16=True)
    # 注入有 CUDA：启用 FP16，结论与直接调用一致
    eng = DetectionEngine(config, model=DummyModel(COCO_NAMES),
                          cuda_available=True)
    exp_enabled, exp_reason = resolve_fp16(config, cuda_available=True)
    assert eng.use_fp16 is True and exp_enabled is True
    assert eng.fp16_reason == exp_reason, "判定文案应与 resolve_fp16 一致"

    # 注入无 CUDA：自动降级 FP32，文案含"降级"且与直接调用一致
    eng_fb = DetectionEngine(config, model=DummyModel(COCO_NAMES),
                             cuda_available=False)
    exp_enabled, exp_reason = resolve_fp16(config, cuda_available=False)
    assert eng_fb.use_fp16 is False and exp_enabled is False
    assert "降级" in eng_fb.fp16_reason
    assert eng_fb.fp16_reason == exp_reason

    # fp16=False：不启用，使用 FP32
    eng_off = DetectionEngine(build_config(fp16=False),
                              model=DummyModel(COCO_NAMES))
    assert eng_off.use_fp16 is False and "FP32" in eng_off.fp16_reason

    # 路径级委托：跟踪器经引擎判定，结论与 resolve_fp16 实时结论一致
    tracker = ByteTrackTracker(config, model=DummyModel(COCO_NAMES))
    expected, _ = resolve_fp16(config)
    assert tracker.use_fp16 == expected
    logger.info(
        "用例2 PASS：引擎 FP16 判定委托 resolve_fp16——有CUDA启用/"
        "无CUDA降级/fp16=False 三分支结论与文案逐值一致；"
        "跟踪器结论与引擎同源",
    )


# ---------------- 3. 类别映射判定委托 ----------------

def test_class_map_delegation():
    """用例 3：引擎类别映射判定与 build_class_mapper 直接调用一致。"""
    config = build_config(class_map_enabled=True)
    # VisDrone 10 类名：映射生效，结论与直接调用一致
    eng = DetectionEngine(config, model=DummyModel(VISDRONE_NAMES))
    _, exp_active, exp_reason = build_class_mapper(config, VISDRONE_NAMES)
    assert eng.class_map_active is True and exp_active is True
    assert eng.class_map_reason == exp_reason
    assert eng.map_cls("pedestrian") == "person"
    assert eng.map_cls("van") == "car"
    assert eng.map_cls("truck") is None

    # COCO 类名：恒等透传不生效
    eng_coco = DetectionEngine(config, model=DummyModel(COCO_NAMES))
    assert eng_coco.class_map_active is False
    assert eng_coco.map_cls("car") == "car"
    assert eng_coco.map_cls("truck") == "truck"  # COCO 同名类也不丢弃

    # 开关关闭：即使 VisDrone 类名也不生效
    eng_off = DetectionEngine(build_config(class_map_enabled=False),
                              model=DummyModel(VISDRONE_NAMES))
    assert eng_off.class_map_active is False
    assert eng_off.map_cls("pedestrian") == "pedestrian"

    # 路径级委托：切片器经引擎判定，结论与直接调用一致
    slicer = SlicedDetector(config, model=DummyModel(VISDRONE_NAMES))
    assert slicer.class_map_active is True
    assert slicer.map_cls("people") == "person"
    logger.info(
        "用例3 PASS：引擎类别映射判定委托 build_class_mapper——VisDrone "
        "生效/COCO 恒等/开关关闭三组一致；切片器结论与引擎同源",
    )


# ---------------- 4. names 来源 ----------------

def test_names_source():
    """用例 4：引擎 names 即其所持模型的 names，三路径与引擎同源。"""
    # 伪造模型形态：引擎 names 就是传入模型的 names 对象本身
    dummy = DummyModel(COCO_NAMES)
    eng = DetectionEngine(build_config(), model=dummy)
    assert eng.names is dummy.names, "引擎 names 应取自所持模型"

    # 真实权重形态（共享引擎）：names 与模型自身 names 逐值一致
    shared_eng = DetectionEngine.shared(Config())
    assert shared_eng.names == shared_eng.model.names
    assert len(shared_eng.names) > 0
    detector = Detector(Config())
    assert detector.names is shared_eng.names, "路径 names 应与引擎同源"
    logger.info(
        "用例4 PASS：引擎 names 取自所持模型（伪造模型 is 判定、真实权重 "
        "%d 类逐值一致）；Detector.names 与引擎同源", len(shared_eng.names),
    )


# ---------------- 5. E6 换权重就绪（真实第二权重） ----------------

def test_switch_weights_e6_ready():
    """用例 5：仅改 Config.weights_path，三路径同学到新权重。"""
    alt_abs = Config().abs_path(ALT_WEIGHTS)
    assert os.path.isfile(alt_abs), "第二权重文件缺失：%s" % alt_abs
    DetectionEngine.reset_shared()  # 测试隔离：清空共享缓存

    default_engine = DetectionEngine.shared(Config())  # 默认权重引擎
    config2 = Config()
    config2.weights_path = ALT_WEIGHTS  # E6 场景：只改权重路径配置项

    detector = Detector(config2)
    tracker = ByteTrackTracker(config2)
    slicer = SlicedDetector(config2)
    eng2 = detector.engine
    # 三路径共享新引擎/新模型，且与默认权重引擎不是同一实例
    assert eng2 is tracker.engine is slicer.engine
    assert eng2 is not default_engine, "权重路径不同应各自缓存引擎"
    assert detector.model is tracker.model is slicer.model
    assert detector.model is not default_engine.model
    assert eng2.weights_path == alt_abs, \
        "引擎应按新权重路径加载：%s" % eng2.weights_path

    # names 三路径一致，且与直接加载该权重文件得到的 names 逐值一致
    assert detector.names is tracker.names is slicer.names is eng2.names
    from ultralytics import YOLO  # 测试内直接加载同一权重做参照
    ref_names = YOLO(alt_abs).names
    assert dict(detector.names) == dict(ref_names), \
        "三路径 names 应与新权重 names 逐值一致"
    logger.info(
        "用例5 PASS：Config.weights_path 指向 %s 后，三路径共享新引擎/"
        "新模型（与默认权重引擎不同实例）；names 一致且与新权重（%d 类）"
        "逐值相同——E6 换权重只改配置项即就绪",
        ALT_WEIGHTS, len(ref_names),
    )
    DetectionEngine.reset_shared()  # 收尾清理共享缓存（未产生临时文件）


def main():
    logger.info("==== 阶段 C 统一检测引擎 验收测试开始 ====")
    test_singleton_shared_engine()
    test_fp16_delegation()
    test_class_map_delegation()
    test_names_source()
    test_switch_weights_e6_ready()
    logger.info("==== 全部 5 组用例断言通过 ====")
    print("PASS: tests/test_engine.py 全部断言通过")


if __name__ == "__main__":
    main()
