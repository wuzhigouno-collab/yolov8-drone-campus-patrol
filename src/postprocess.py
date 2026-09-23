# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

推理结果公共后处理模块（阶段 C 结构重构）。

功能意图：
    检测（src.detector.Detector.detect）、跟踪（src.tracker
    .ByteTrackTracker 正常 track 路径与降级 predict 路径）、切片推理
    （src.slicer.SlicedDetector.detect）三条推理路径的"遍历 YOLO
    results.boxes -> cls_id 查类别名 -> 可选 VisDrone 类别映射 ->
    按 Config.target_classes 过滤 -> 组装 {cls, conf, box, center}
    统一检测字典"后处理段，原本在三处各自重复实现；本模块将其抽取为
    唯一公共实现，三条路径统一调用，保证对外结构逐字段一致，后续
    行为调整只改一处。

    输出契约（与重构前逐字段一致）：
        cls    —— 类别名（字符串；VisDrone 映射生效时为归并后的业务
                  类别名，未生效时为模型原始类别名）
        conf   —— 置信度（0~1 浮点数，round 到 4 位小数）
        box    —— 目标框 xyxy 像素坐标 [x1, y1, x2, y2]（整数）
        center —— 目标框中心点像素坐标 [cx, cy]（整数）：
                  整图/跟踪路径由原始浮点坐标取中点后取整；
                  切片路径由映射回原图的整数框取中点后取整
"""


def _build_det(item, names, target_classes, map_cls, box_transform):
    """把单个 YOLO box 项整理为统一检测字典；被丢弃/过滤时返回 None。

    参数：
        item: ultralytics 单框结果项（提供 cls/conf/xyxy 属性）；
        names: 模型类别名表（cls_id -> 类别名）；
        target_classes: 关注类别名列表；为空列表/None 时不过滤；
        map_cls: 类别名映射函数（src.class_map.build_class_mapper 构造），
            返回 None 表示该类别被映射表丢弃；为 None 时不做映射；
        box_transform: 可选的框坐标变换函数（切片路径把切片内坐标映射
            回原图坐标系）；为 None 时框直接取整。

    返回：
        {"cls", "conf", "box", "center"} 字典；类别被映射丢弃或不命中
        关注类别时返回 None。
    """
    cls_name = names[int(item.cls[0])]
    if map_cls is not None:
        cls_name = map_cls(cls_name)
        if cls_name is None:
            return None  # VisDrone 映射表中标记丢弃的类别
    # 只保留配置中关注的类别（人/车/自行车等），过滤无关目标
    if target_classes and cls_name not in target_classes:
        return None
    conf = float(item.conf[0])
    x1, y1, x2, y2 = (float(v) for v in item.xyxy[0])
    if box_transform is not None:
        # 切片路径：切片内坐标先映射回原图（越界钳制为整数框），
        # 中心点由映射后的整数框计算（与 SlicedDetector 原行为一致）
        box = box_transform([x1, y1, x2, y2])
        center = [int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)]
    else:
        # 整图/跟踪路径：框坐标取整，中心点由原始浮点坐标计算
        # （与 Detector/ByteTrackTracker 原行为逐值一致）
        box = [int(x1), int(y1), int(x2), int(y2)]
        center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
    return {
        "cls": cls_name,
        "conf": round(conf, 4),
        "box": box,
        "center": center,
    }


def collect_detections(boxes, names, target_classes, map_cls=None,
                       box_transform=None, extra_fields=None):
    """遍历 predict() 结果的 boxes，整理为统一检测字典列表。

    参数：
        boxes: ultralytics results.boxes（可迭代单框结果项）；
        names / target_classes / map_cls / box_transform: 同 _build_det；
        extra_fields: 可选字典，原样合并进每个检测项（切片路径用于
            附带 {"slice": 切片序号} 追溯字段）。

    返回：
        [{cls, conf, box, center, ...}, ...]，保持输入框的先后顺序；
        无检出或全部被过滤时返回空列表。
    """
    detections = []
    for item in boxes:
        det = _build_det(item, names, target_classes, map_cls, box_transform)
        if det is None:
            continue
        if extra_fields:
            det.update(extra_fields)
        detections.append(det)
    return detections


def collect_track_detections(boxes, names, target_classes, map_cls=None):
    """遍历 track() 结果的 boxes，整理为 (检测项, 跟踪ID) 列表。

    与 collect_detections 的差别仅在跟踪 ID 处理：个别框未分配跟踪
    ID（id 为 None，例如首帧尚未关联成功）时跳过，不参与轨迹确认；
    boxes 整体无 ID（非跟踪结果）时返回空列表。

    返回：
        [(det, track_id), ...]，det 结构与 collect_detections 一致。
    """
    if boxes is None or boxes.id is None:
        return []
    collected = []
    for item in boxes:
        if item.id is None:
            continue
        det = _build_det(item, names, target_classes, map_cls, None)
        if det is None:
            continue
        track_id = int(item.id[0])
        collected.append((det, track_id))
    return collected
