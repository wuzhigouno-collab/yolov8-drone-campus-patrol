# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

切片推理模块（A12，自研实现，不依赖 sahi 等第三方切片库）。

功能意图：
    无人机高空俯视画面分辨率大、目标像素占比小，整图缩放到 YOLO 推理
    尺寸（imgsz，默认 640）后小目标特征被进一步压缩，容易漏检。本模块
    把整帧切分为 rows x cols（默认 2x2）个带重叠的切片，逐片独立推理
    ——相当于把局部画面放大后检测，提升小目标召回；再把各片检出框的
    切片内坐标加上切片偏移映射回原图坐标系；最后对全部检出做类别感知
    的合并 NMS 去重（同一目标在相邻切片重叠区会被重复检出，同类框 IOU
    达到 Config.slice_nms_iou 阈值即抑制低置信度框）。

    对外接口与 src.detector.Detector.detect() 完全对齐（输入单帧 BGR
    画面，返回 [{cls, conf, box, center}, ...]），可作为检测路径的
    替换项；每个返回项额外附带 "slice" 字段记录检出切片序号，供日志
    与测试追溯。切片行列数、重叠率、NMS 阈值均走 Config 配置项
    （slice_rows / slice_cols / slice_overlap_ratio / slice_nms_iou）。
"""

from src.config import Config
from src.engine import DetectionEngine
from src.frame_confirm import iou_of
from src.logger import get_logger
from src.postprocess import collect_detections


def _axis_spans(length, n, overlap_ratio):
    """沿单一轴（宽或高）计算 n 个切片的起止坐标 [(start, end), ...]。

    重叠率定义为 相邻切片重叠长度 / 切片长度（SAHI 风格）：
    由 length = tile * (n - (n - 1) * r) 解出切片长度
    tile = ceil(length / (n - (n - 1) * r))；首片贴 0、末片顶到
    length，中间均匀分布，保证全覆盖且实际重叠率不低于配置值。

    参数：
        length: 该轴总长度（像素）；n: 切片数量（>=1）；
        overlap_ratio: 重叠率（0~1）。

    返回：
        [(start, end), ...]，坐标均在 [0, length] 内；n==1 或切片长度
        不小于总长时退化为单片 [(0, length)]。
    """
    if n <= 1:
        return [(0, length)]
    ratio = min(max(float(overlap_ratio), 0.0), 0.9)  # 钳制非法重叠率
    # 切片长度：重叠率越大，同样总长下需要的单片越长
    tile = int(-(-length / (n - (n - 1) * ratio) // 1))  # ceil 除法
    if tile >= length:
        return [(0, length)]
    # 首片 start=0、末片 end=length，中间均匀插值，保证末端对齐画面边缘
    step = (length - tile) / (n - 1)
    spans = []
    for i in range(n):
        start = int(round(i * step))
        spans.append((start, min(start + tile, length)))
    spans[-1] = (length - tile, length)  # 末片强制顶边，消除舍入误差
    return spans


def compute_slice_regions(width, height, rows, cols, overlap_ratio):
    """把 width x height 的画面切分为 rows x cols 个重叠切片区域。

    参数：
        width / height: 画面尺寸（像素）；rows / cols: 切片行列数；
        overlap_ratio: 相邻切片重叠率（0~1，重叠宽度占切片宽/高比例）。

    返回：
        [(x1, y1, x2, y2), ...]，按行优先顺序（先上后下、先左后右），
        全部为原图坐标系下的闭区间像素范围，可直接用于 numpy 切片
        frame[y1:y2, x1:x2]（end 越界由 numpy 自动钳制，此处已保证
        end <= 画面尺寸）。
    """
    x_spans = _axis_spans(width, cols, overlap_ratio)
    y_spans = _axis_spans(height, rows, overlap_ratio)
    regions = []
    for y1, y2 in y_spans:
        for x1, x2 in x_spans:
            regions.append((x1, y1, x2, y2))
    return regions


def map_box_to_frame(box, offset_x, offset_y, frame_w, frame_h):
    """把切片内坐标的目标框映射回原图坐标系（越界钳制到画面内）。

    参数：
        box: 切片内 xyxy 坐标 [x1, y1, x2, y2]；
        offset_x / offset_y: 切片左上角在原图中的偏移（像素）；
        frame_w / frame_h: 原图尺寸，用于钳制越界坐标。

    返回：
        [x1, y1, x2, y2] 原图坐标（整数，保证 0 <= x1 < x2 <= frame_w，
        y 方向同理；退化框被钳成最小 1 像素宽避免后续除零）。
    """
    x1 = min(max(int(box[0]) + offset_x, 0), frame_w - 1)
    y1 = min(max(int(box[1]) + offset_y, 0), frame_h - 1)
    x2 = min(max(int(box[2]) + offset_x, x1 + 1), frame_w)
    y2 = min(max(int(box[3]) + offset_y, y1 + 1), frame_h)
    return [x1, y1, x2, y2]


def nms_merge(detections, iou_threshold):
    """类别感知的贪心 NMS 合并去重。

    按置信度降序（同分按输入下标，保证确定性可复现）依次取框；
    与已保留框同类且 IOU >= 阈值的框被抑制（视为同一目标的重复
    检出），不同类的框即使重叠也各自保留（类别是目标身份的强约束，
    与 A5/A11 匹配逻辑一致）。

    参数：
        detections: 检测列表，每项为含 cls/conf/box 键的字典（不修改
            原字典，返回项为原对象引用）；
        iou_threshold: 同类框 IOU 达到该值即抑制低置信度框。

    返回：
        保留的检测列表，按置信度降序排列。由此保证同类保留框两两
        IOU 均严格小于阈值。
    """
    # 置信度降序 + 同分按下标升序，排序键完全确定
    order = sorted(range(len(detections)),
                   key=lambda i: (-detections[i]["conf"], i))
    kept = []
    for i in order:
        det = detections[i]
        suppressed = False
        for other in kept:
            if other["cls"] != det["cls"]:
                continue  # 类别不同不互相抑制
            if iou_of(det["box"], other["box"]) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(det)
    return kept


class SlicedDetector:
    """重叠切片推理检测器（接口与 Detector.detect() 对齐的替换项）。

    职责：
        1. 把输入帧按 Config.slice_rows x slice_cols 切分为重叠切片
           （重叠率走 Config.slice_overlap_ratio）；
        2. 逐片调用 YOLO 模型推理，仅保留 Config.target_classes 关注
           类别（过滤逻辑与 Detector 一致）；
        3. 把各片检出框的切片内坐标映射回原图坐标系（越界钳制）；
        4. 对全部检出做类别感知合并 NMS 去重（阈值走 Config
           .slice_nms_iou），返回与 Detector.detect() 结构一致的
           检测列表，每项额外带 "slice" 字段记录检出切片序号。

    参数在构造时从 Config 读取快照；运行期改配置后重新构造即可生效。
    """

    def __init__(self, config=None, model=None, engine=None):
        """初始化切片推理检测器。

        参数：
            config: Config 实例；为 None 时内部新建默认配置。
            model:  已加载的 ultralytics YOLO 模型实例（传入
                    Detector.model 可避免权重重复加载）；为 None 时
                    取共享检测引擎持有的模型。
            engine: 已建立的 src.engine.DetectionEngine 实例；为 None
                    时按 model 参数推导——传了 model 则基于该模型新建
                    引擎（不加载权重），否则取按 Config.weights_path
                    键控的共享引擎单例（与检测/跟踪路径共享同一模型，
                    R2 检测引擎抽象）。
        """
        self.config = config if config is not None else Config()
        self.logger = get_logger(__name__, self.config)

        # R2：模型与类别映射判定统一来自检测引擎；切片器不再自行加载
        # 权重（模型与检测/跟踪路径同源，保证切片检出类别口径一致）
        if engine is None:
            engine = DetectionEngine(self.config, model=model) \
                if model is not None else DetectionEngine.shared(self.config)
        self.engine = engine
        self.model = engine.model
        # 类别 id -> 类别名 映射表（经引擎取自模型本身，兼容自训权重）
        self.names = engine.names

        # 参数快照：切片行列数、重叠率、合并 NMS 的 IOU 阈值
        self.rows = int(self.config.slice_rows)
        self.cols = int(self.config.slice_cols)
        self.overlap_ratio = float(self.config.slice_overlap_ratio)
        self.nms_iou = float(self.config.slice_nms_iou)
        # 阶段 C：VisDrone 10->5 类别映射判定结论取自检测引擎（判定
        # 过程与日志见 src.engine；COCO 权重下自动为恒等映射，行为与
        # 未开启时一致）
        self.map_cls = engine.map_cls
        self.class_map_active = engine.class_map_active
        self.logger.info(
            "切片推理器初始化：%dx%d 切片、重叠率 %.2f、合并 NMS IOU 阈值 %.2f",
            self.rows, self.cols, self.overlap_ratio, self.nms_iou,
        )

    def detect(self, frame):
        """对单帧画面执行切片推理，返回统一结构的检测结果列表。

        参数：
            frame: OpenCV 读取的 BGR 图像（numpy.ndarray）。

        返回：
            检测结果列表，每个元素为字典：
                {"cls": 类别名, "conf": 置信度,
                 "box": [x1, y1, x2, y2]（原图坐标）,
                 "center": [cx, cy], "slice": 检出切片序号}
            结构与 Detector.detect() 对齐（多一个 slice 追溯字段）；
            画面无目标时返回空列表。
        """
        frame_h, frame_w = frame.shape[:2]
        regions = compute_slice_regions(
            frame_w, frame_h, self.rows, self.cols, self.overlap_ratio)

        raw = []
        for index, (x1, y1, x2, y2) in enumerate(regions):
            tile = frame[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            results = self.model.predict(
                tile, imgsz=self.config.imgsz, verbose=False)[0]
            # 后处理走 src.postprocess 公共实现（类别映射 -> 关注类别
            # 过滤 -> 组装统一结构）；box_transform 把切片内坐标映射回
            # 原图（越界钳制），extra_fields 附带切片序号追溯字段
            raw.extend(collect_detections(
                results.boxes, self.names, self.config.target_classes,
                map_cls=self.map_cls,
                box_transform=lambda b: map_box_to_frame(
                    b, x1, y1, frame_w, frame_h),
                extra_fields={"slice": index}))

        # 合并 NMS 去重：相邻切片重叠区的重复检出只留高置信度框
        merged = nms_merge(raw, self.nms_iou)
        self.logger.debug(
            "切片推理：%d 片原始检出 %d 个，合并去重后 %d 个",
            len(regions), len(raw), len(merged),
        )
        return merged
