# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

VisDrone 10->5 类别映射模块（阶段 C）。

功能意图：
    论文实验阶段训练与评估一律使用 VisDrone2019 原始 10 类（保证与
    文献可比），部署到巡检系统时则在推理输出后把细粒度类别归并为
    5 类业务类别：
        pedestrian + people -> person
        bicycle             -> bicycle
        motor               -> motorcycle
        car + van           -> car
        bus                 -> bus
    truck / tricycle / awning-tricycle 三类与校园巡检业务无关，
    在映射时直接丢弃（不再进入后续过滤、确认、告警与计数）。
    映射只发生在 src/ 推理侧（检测/跟踪/切片三条路径统一经
    src.postprocess 应用本模块构造的映射函数），不改动任何训练标签。

生效条件（两个条件缺一不可，由 build_class_mapper 统一判定）：
    1. 配置开关 Config.class_map_enabled 开启；
    2. 模型实际类别名（model.names）全部属于 VisDrone 10 类集合。
    当前开发联调使用 COCO 预训练权重（person/car 等 80 类名），不
    满足条件 2，映射自动退化为恒等映射——类别名原样透传，系统行为
    与未开启映射时完全一致。
"""


# VisDrone2019-DET 原始 10 类（训练/评估口径，与数据集类别序一致）
VISDRONE_CLASSES = (
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
)
VISDRONE_CLASS_SET = frozenset(VISDRONE_CLASSES)

# VisDrone -> 业务类别 的映射表（仅列出被保留的 7 个细粒度类别）
VISDRONE_TO_BUSINESS = {
    "pedestrian": "person",
    "people": "person",
    "bicycle": "bicycle",
    "car": "car",
    "van": "car",
    "bus": "bus",
    "motor": "motorcycle",
}

# 映射时丢弃的 VisDrone 类别（与校园巡检业务无关，不进入后续流程）
VISDRONE_DROP_CLASSES = ("truck", "tricycle", "awning-tricycle")


def identity_class_name(name):
    """恒等映射：类别名原样透传（映射未生效时使用）。"""
    return name


def map_visdrone_class(name):
    """把单个 VisDrone 细粒度类别名映射为业务类别名。

    参数：
        name: 模型输出的原始类别名（VisDrone 10 类之一）。

    返回：
        归并后的业务类别名（person/bicycle/motorcycle/car/bus）；
        truck/tricycle/awning-tricycle 返回 None 表示丢弃；
        不在 VisDrone 10 类集合内的名字原样返回（防御性兜底：
        正常生效路径下由 build_class_mapper 保证不会出现）。
    """
    if name in VISDRONE_DROP_CLASSES:
        return None
    return VISDRONE_TO_BUSINESS.get(name, name)


def _model_class_names(names):
    """从 model.names 提取类别名集合（兼容 dict {id: name} 与 list）。"""
    if isinstance(names, dict):
        return set(names.values())
    return set(names)


def build_class_mapper(config, names):
    """按配置开关与模型类别名构造类别映射函数（三条推理路径统一调用）。

    判定规则：
        1. Config.class_map_enabled=False —— 映射未启用，恒等透传；
        2. 开关开启但模型类别名不全属于 VisDrone 10 类集合（如当前
           COCO 权重）—— 映射不生效，恒等透传，行为与未开启一致；
        3. 开关开启且模型类别名全部属于 VisDrone 10 类集合 —— 启用
           VisDrone 10->5 映射（map_visdrone_class）。

    参数：
        config: Config 实例（读取 class_map_enabled 配置项）；
        names:  模型类别名表（ultralytics model.names，dict 或 list）。

    返回：
        (mapper, active, reason)：
        mapper —— 类别名映射函数，输入原始类别名，返回业务类别名；
                  返回 None 表示该类别被丢弃；未生效时为恒等函数；
        active —— 映射是否实际生效（布尔）；
        reason —— 文字原因说明（供启动日志与测试断言）。
    """
    if not getattr(config, "class_map_enabled", False):
        return identity_class_name, False, (
            "配置 class_map_enabled=False：VisDrone 类别映射未启用，"
            "按模型原始类别名输出（恒等映射）"
        )
    model_classes = _model_class_names(names)
    if not model_classes or not model_classes.issubset(VISDRONE_CLASS_SET):
        # 当前 COCO 权重走此分支：80 类名大多不在 VisDrone 10 类集合内
        outsider = sorted(model_classes - VISDRONE_CLASS_SET)
        sample = outsider[0] if outsider else "(空类别表)"
        return identity_class_name, False, (
            "配置 class_map_enabled=True，但模型类别名不全属于 VisDrone "
            "10 类集合（如 %s）：类别映射不生效，按模型原始类别名输出"
            "（恒等映射）" % sample
        )
    return map_visdrone_class, True, (
        "配置 class_map_enabled=True 且模型类别名全部属于 VisDrone 10 "
        "类集合：启用 VisDrone 10->5 类别映射（pedestrian+people->person、"
        "car+van->car、motor->motorcycle，truck/tricycle/awning-tricycle 丢弃）"
    )
