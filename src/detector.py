# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

YOLO 目标检测封装模块。

功能意图：
    封装 YOLO 单帧推理，对外提供统一的检测结果结构，屏蔽底层模型
    调用细节。GUI 主界面、演示脚本、评估脚本等上层模块只需调用
    detect() 即可拿到格式一致的检测列表，便于后续分区域阈值、
    多帧确认、越线计数等模块统一消费。

    R2 检测引擎抽象：模型生命周期（权重加载）、权重路径解析、
    FP16 半精度判定、VisDrone 类别映射判定已统一收敛到
    src.engine.DetectionEngine；本类不再自行加载模型，改为从
    （按权重路径共享的）检测引擎获取模型与判定结果，与跟踪/切片
    两条推理路径同源同学——更换部署权重只需修改 Config.weights_path。
"""

from src.config import Config
# resolve_fp16 已迁入 src.engine（判定唯一归属）；此处仅作兼容转出，
# 供既有调用方（tests/test_accel.py、tools/benchmark.py 等）按原路径导入
from src.engine import DetectionEngine, resolve_fp16
from src.logger import get_logger
from src.postprocess import collect_detections

__all__ = ["Detector", "resolve_fp16"]


class Detector:
    """YOLO 推理封装类。

    职责：
        1. 从共享检测引擎（src.engine.DetectionEngine）取得已加载的
           YOLO 模型与三项判定结果（权重路径解析/FP16/类别映射）；
        2. 对单帧画面执行推理（A13：fp16 配置开启且具备 CUDA GPU 时
           以半精度执行，无 GPU 环境自动降级 FP32 并记日志）；
        3. 将原始推理结果整理为统一结构返回：
           [{cls, conf, box, center}, ...]
           cls    —— 类别名（字符串，如 person/car/bicycle）
           conf   —— 置信度（0~1 浮点数）
           box    —— 目标框 xyxy 像素坐标 [x1, y1, x2, y2]
           center —— 目标框中心点像素坐标 [cx, cy]
    """

    def __init__(self, config=None, engine=None):
        """初始化检测器：接入共享检测引擎并取得模型与判定结果。

        参数：
            config: Config 实例；为 None 时内部新建默认配置。
                    权重路径、推理尺寸、目标类别均从配置读取。
            engine: 已建立的 src.engine.DetectionEngine 实例；为 None
                    时按 Config.weights_path 取共享引擎单例（同一进程
                    同一权重只加载一次，与跟踪/切片路径共享同一模型）。
        """
        self.config = config if config is not None else Config()
        self.logger = get_logger(__name__, self.config)

        # R2：模型与三项判定（权重路径解析/FP16/类别映射）统一来自检测
        # 引擎；未显式传入时取按权重路径键控的共享单例
        self.engine = engine if engine is not None \
            else DetectionEngine.shared(self.config)
        self.model = self.engine.model
        # 类别 id -> 类别名 映射表，取自引擎持有的模型（兼容自训权重）
        self.names = self.engine.names
        self.use_fp16 = self.engine.use_fp16
        self.map_cls = self.engine.map_cls
        self.class_map_active = self.engine.class_map_active

    def detect(self, frame):
        """对单帧画面执行推理，返回统一结构的检测结果列表。

        参数：
            frame: OpenCV 读取的 BGR 图像（numpy.ndarray）。

        返回：
            检测结果列表，每个元素为字典：
                {"cls": 类别名, "conf": 置信度,
                 "box": [x1, y1, x2, y2], "center": [cx, cy]}
            仅保留 Config.target_classes 中配置的关注类别；
            画面无目标时返回空列表。阶段 C：配置开启且模型为
            VisDrone 权重时类别名先经 10->5 映射再过滤（COCO
            权重下映射为恒等，行为不变，见 src.class_map）。
        """
        results = self.model.predict(
            frame, imgsz=self.config.imgsz, verbose=False,
            half=self.use_fp16,
        )[0]

        # 后处理（类别映射 -> 关注类别过滤 -> 组装统一结构）走
        # src.postprocess 公共实现，与跟踪/切片路径保持一致
        return collect_detections(
            results.boxes, self.names, self.config.target_classes,
            map_cls=self.map_cls)
