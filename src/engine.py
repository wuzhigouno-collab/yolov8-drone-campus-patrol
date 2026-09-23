# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

统一检测引擎模块（阶段 C 结构重构 R2：检测引擎抽象）。

功能意图：
    检测（src.detector.Detector）、跟踪（src.tracker.ByteTrackTracker）、
    切片推理（src.slicer.SlicedDetector）三条推理路径原本各自管理
    YOLO 模型生命周期，并各自重复执行三项初始化判定——权重路径解析
    （Config.abs_path(Config.weights_path)）、FP16 半精度判定
    （resolve_fp16）、VisDrone 10->5 类别映射判定（build_class_mapper）。
    本模块把"模型加载 + 三项判定 + model.names"收敛为唯一检测引擎
    DetectionEngine，三条路径改为从引擎获取模型与判定结果，保证：

    1. 同源：全项目 src/ 内权重加载（YOLO(...)）、权重路径解析、
       FP16 判定、类别映射判定各只有本模块一处实现，行为调整只改一处；
    2. 同学：shared() 工厂提供按权重路径键控的单例——同一进程内同一
       权重文件只加载一次，三条路径共享同一模型实例与同一套判定结论，
       启动日志只出现一次权重加载记录；
    3. 可替换：E6 更换部署权重（如 COCO yolov8n.pt -> 自训
       weights/best.pt）只需修改 Config.weights_path，三条路径经引擎
       自动同学到新权重，model.names 一律取自新权重本身。

    延迟导入取舍保留：ultralytics 仅在引擎真正加载权重时才导入
    （构造方法内部），无 GPU / 无 ultralytics 环境下仍可导入本模块及
    各纯逻辑模块并运行其单测；外部传入模型实例时引擎不加载权重，
    仅基于该模型完成三项判定（测试注入伪造模型走此路径）。
"""

import os

from src.class_map import build_class_mapper
from src.config import Config
from src.logger import get_logger


def resolve_fp16(config, cuda_available=None):
    """判定 FP16 半精度推理是否实际启用（A13；R2 起归属检测引擎）。

    判定规则：
        1. 配置 fp16=False —— 不启用，使用 FP32 推理；
        2. 配置 fp16=True 但无可用 CUDA GPU —— 半精度推理仅 GPU 支持，
           自动降级为 FP32（配置值不被修改，仅本次运行生效）；
        3. 配置 fp16=True 且检测到 CUDA GPU —— 启用半精度推理。

    参数：
        config: Config 实例（读取 fp16 配置项）。
        cuda_available: 是否检测到 CUDA；为 None 时实时调用
            torch.cuda.is_available() 探测。测试可显式传入布尔值，
            无需真实 GPU 即可覆盖降级分支。

    返回：
        (enabled, reason)：enabled 为最终是否启用半精度；
        reason 为文字原因说明（供启动日志与测试断言）。
    """
    if not getattr(config, "fp16", False):
        return False, "配置 fp16=False：使用 FP32 全精度推理"
    if cuda_available is None:
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            # torch 缺失或探测异常时按无 CUDA 处理，保证不启用半精度
            cuda_available = False
    if not cuda_available:
        return False, ("配置 fp16=True，但当前环境无可用 CUDA GPU："
                       "半精度推理仅 GPU 支持，自动降级为 FP32 全精度推理")
    return True, "配置 fp16=True 且检测到 CUDA GPU：启用 FP16 半精度推理"


class DetectionEngine:
    """统一检测引擎：YOLO 模型生命周期与三项初始化判定的唯一归属。

    职责：
        1. 把 Config.weights_path 解析为项目根目录下的绝对路径并加载
           YOLO 模型（ultralytics 延迟导入，仅此处真正加载权重）；
        2. FP16 半精度推理判定（resolve_fp16；配置开启但无 CUDA GPU
           时自动降级 FP32 并记日志，配置值本身不修改）；
        3. VisDrone 10->5 类别映射判定（build_class_mapper；COCO
           权重下自动为恒等映射，行为与未开启时一致）；
        4. 对外暴露 model / names / use_fp16 / map_cls /
           class_map_active，三条推理路径统一取用。

    单例语义：
        shared(config) 按"权重绝对路径"键控缓存引擎实例：同一进程内
        同一权重文件只加载一次（启动日志只有一条权重加载记录），三条
        路径共享同一模型；权重路径不同则各自缓存、互不干扰。判定结果
        以首次创建该权重引擎时的 Config 快照为准。reset_shared() 清空
        缓存，供测试隔离使用（正常运行无需调用）。
    """

    # 共享引擎缓存：{规范化权重绝对路径: DetectionEngine}
    _shared_engines = {}

    def __init__(self, config=None, model=None, cuda_available=None):
        """初始化检测引擎：加载（或接入）模型并完成三项判定。

        参数：
            config: Config 实例；为 None 时内部新建默认配置。
            model:  外部已加载的模型实例（调用方自带模型或测试伪造
                    模型）；传入时引擎不再加载权重，仅基于该模型的
                    names 完成判定。为 None 时按 Config.weights_path
                    加载权重（首次使用 ultralytics 会自动下载缺失的
                    权重文件到该路径）。
            cuda_available: 测试注入用 CUDA 探测值，原样透传给
                    resolve_fp16；为 None 时实时探测。
        """
        self.config = config if config is not None else Config()
        self.logger = get_logger(__name__, self.config)

        # 权重路径走配置项：相对路径统一换算为项目根目录下的绝对路径
        # （全项目 src/ 内仅此一处解析 Config.weights_path）
        self.weights_path = self._weights_abs(self.config)
        if model is None:
            # 延迟导入 ultralytics：该库加载较慢，仅在真正加载权重时引入
            from ultralytics import YOLO
            self.logger.info("正在加载 YOLO 模型权重：%s", self.weights_path)
            model = YOLO(self.weights_path)
            self.logger.info(
                "模型加载完成，可识别类别数：%d，关注类别：%s",
                len(model.names), ",".join(self.config.target_classes),
            )
        else:
            self.logger.info(
                "检测引擎接入外部模型实例，可识别类别数：%d，关注类别：%s",
                len(model.names), ",".join(self.config.target_classes),
            )
        self.model = model
        # 类别 id -> 类别名 映射表，取自模型本身（兼容自训权重）：
        # 三条推理路径的类别名一律经引擎取自该模型，换权重后自动同源
        self.names = self.model.names

        # A13：FP16 半精度推理判定——配置开启但无 CUDA GPU 时自动降级
        # 为 FP32 并记日志说明（配置值本身不修改，仅本次运行生效）
        self.use_fp16, self.fp16_reason = resolve_fp16(
            self.config, cuda_available)
        self.logger.info("推理精度判定：%s", self.fp16_reason)
        # 阶段 C：VisDrone 10->5 类别映射判定——仅当配置开关开启且模型
        # 类别名全部属于 VisDrone 10 类集合时生效；当前 COCO 权重下自动
        # 为恒等映射，行为与未开启时一致（详见 src.class_map 模块说明）
        self.map_cls, self.class_map_active, self.class_map_reason = \
            build_class_mapper(self.config, self.names)
        self.logger.info("类别映射判定：%s", self.class_map_reason)

    # ---------------- 共享单例 ----------------

    @staticmethod
    def _weights_abs(config):
        """把 Config.weights_path 解析为项目根目录下的绝对路径。"""
        return config.abs_path(config.weights_path)

    @classmethod
    def shared(cls, config=None):
        """返回按权重路径键控的共享引擎实例（同一进程同一权重只加载一次）。

        参数：
            config: Config 实例；为 None 时内部新建默认配置。

        返回：
            DetectionEngine 实例；同一进程内对同一 weights_path 反复
            调用返回同一对象（is 判定成立），权重加载日志只出现一次。
        """
        config = config if config is not None else Config()
        # 键做大小写/分隔符规范化，避免 Windows 下同一路径写法不同而重复加载
        key = os.path.normcase(os.path.normpath(cls._weights_abs(config)))
        engine = cls._shared_engines.get(key)
        if engine is None:
            engine = cls(config)
            cls._shared_engines[key] = engine
            engine.logger.info(
                "检测引擎共享实例已建立：本进程内检测/跟踪/切片三条推理"
                "路径复用同一模型与判定结果，权重不再重复加载"
            )
        return engine

    @classmethod
    def reset_shared(cls):
        """清空共享引擎缓存（测试隔离用；正常运行无需调用）。"""
        cls._shared_engines = {}
