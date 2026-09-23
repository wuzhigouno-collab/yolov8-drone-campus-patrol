# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

ByteTrack 跟踪集成模块（A11）。

功能意图：
    A5 的多帧 IOU 确认是无跟踪器条件下的降级方案：候选目标号只在相邻
    帧间有意义，目标被短暂遮挡或漏检即断档重计。本模块封装 ultralytics
    内置的 ByteTrack 跟踪器（model.track()），把检测升级为带持久 ID 的
    目标轨迹——同一物理目标在视频中跨帧保持同一跟踪 ID，轨迹被短暂
    遮挡后仍能接续，为越线计数、徘徊检测等下游模块提供稳定的身份依据。

    确认逻辑与 A5 保持同一对外契约：轨迹从首次出现起累计存在 N 帧
    （Config.confirm_frames）才标记为"已确认"，未确认目标不进入告警
    与计数。输出复用 src.frame_confirm.ConfirmResult，records /
    confirmed / just_confirmed 结构与 A5 完全一致，GUI 与下游模块
    无需感知后端差异；当前生效后端通过 ConfirmResult.backend 区分
    （本模块为 BACKEND_BYTETRACK）。

降级路径：
    A5 的 IOU 匹配实现（src.frame_confirm.FrameConfirmer）保留为无
    跟踪器时的降级路径：
    1. 配置降级——Config.tracking_enabled=False 时上层直接构造
       FrameConfirmer，不经过本模块；
    2. 自动降级——model.track() 运行期抛异常（跟踪器配置缺失、推理
       后端异常等）时，本模块自动切换到内部持有的 FrameConfirmer，
       用 model.predict() 的检测结果继续提供确认服务，ConfirmResult
       .backend 随之变为 BACKEND_IOU_FALLBACK，全程只记一次日志。

轨迹尾迹：
    本模块同时维护每条轨迹最近若干帧（Config.track_tail_length）的
    锚点序列（底部中心点，与 A4/A6 锚点定义一致），供 GUI 叠加显示
    轨迹尾迹与徘徊检测取点使用。

跳帧外推（A13）：
    Config.detect_interval > 1 时，上层只在检测帧调用 update()（模型
    推理只发生在检测帧），中间帧调用 extrapolate()：不执行任何推理，
    按每条轨迹最近一次实测速度（相邻两个检测帧的框中心位移 / 帧间隔）
    线性外推目标位置，产出 record["extrapolated"]=True 的
    ConfirmResult，供跳帧期间渲染中间帧目标位置。外推帧不推进轨迹
    的"累计存在帧数"（确认判定只认真实观测），已确认状态在外推期间
    保持不变；最近一次检测帧未再被观测到的轨迹不做外推（防止目标
    消失后外推出幽灵框）。
"""

from collections import deque

from src.config import Config
from src.engine import DetectionEngine
from src.frame_confirm import (
    BACKEND_IOU_FALLBACK,
    ConfirmResult,
    FrameConfirmer,
)
from src.logger import get_logger
from src.postprocess import collect_detections, collect_track_detections
from src.zone_threshold import anchor_point

# 确认后端标识：本模块正常工作时为 ByteTrack 跟踪模式
BACKEND_BYTETRACK = "bytetrack"


class ByteTrackTracker:
    """ByteTrack 轨迹跟踪与多帧确认器（跟踪后端）。

    职责：
        1. 逐帧调用 ultralytics model.track(persist=True) 执行检测 +
           ByteTrack 关联，获得带持久跟踪 ID 的目标框；
        2. 按 Config.target_classes 过滤关注类别，把跟踪结果整理为与
           Detector 输出对齐的检测项 {cls, conf, box, center}；
        3. 维护每条轨迹的生命周期状态（累计存在帧数、是否已确认、
           确认帧号）：轨迹累计存在满 Config.confirm_frames 帧才确认，
           确认状态在轨迹存活期间保持（同一 ID 只产生一次确认事件）；
        4. 维护每条轨迹的锚点尾迹（deque，长度走 Config.track_tail_
           length），供 GUI 轨迹叠加与徘徊检测使用；
        5. model.track() 异常时自动降级为 A5 IOU 确认路径，对外契约
           （ConfirmResult）不变；
        6. A13 跳帧加速：中间帧调用 extrapolate() 按轨迹实测速度线性
           外推目标位置（不执行推理），外推帧不推进确认计数。

    本类不修改 Config，参数在构造时从 Config 读取快照；运行期改
    配置后重新构造即可生效。切换视频源/重新开播时应调用 reset()
    清空轨迹状态（含 ultralytics 内部跟踪器状态）。
    """

    def __init__(self, config=None, model=None, engine=None):
        """初始化跟踪器。

        参数：
            config: Config 实例；为 None 时内部新建默认配置。
            model:  已加载的 ultralytics YOLO 模型实例（传入
                    Detector.model 可避免权重重复加载）；为 None 时
                    取共享检测引擎持有的模型。
            engine: 已建立的 src.engine.DetectionEngine 实例；为 None
                    时按 model 参数推导——传了 model 则基于该模型新建
                    引擎（不加载权重），否则取按 Config.weights_path
                    键控的共享引擎单例（与检测/切片路径共享同一模型，
                    R2 检测引擎抽象）。
        """
        self.config = config if config is not None else Config()
        self.logger = get_logger(__name__, self.config)

        # R2：模型与三项判定（权重路径解析/FP16/类别映射）统一来自检测
        # 引擎；跟踪器不再自行加载权重
        if engine is None:
            engine = DetectionEngine(self.config, model=model) \
                if model is not None else DetectionEngine.shared(self.config)
        self.engine = engine
        self.model = engine.model
        # 类别 id -> 类别名 映射表（经引擎取自模型本身，兼容自训权重）
        self.names = engine.names

        # 参数快照：确认帧数 N、跟踪器名、尾迹长度、轨迹状态遗忘帧数
        self.confirm_frames = int(self.config.confirm_frames)
        self.tracker_name = str(self.config.tracker_name)
        self.tail_length = int(self.config.track_tail_length)
        self.track_max_age = int(self.config.track_max_age)
        # ultralytics 跟踪器配置参数要求 *.yaml 形式
        self._tracker_arg = self.tracker_name \
            if self.tracker_name.endswith(".yaml") \
            else self.tracker_name + ".yaml"

        # A13 FP16 判定与阶段 C 类别映射判定：结论取自检测引擎（判定
        # 过程与日志见 src.engine；COCO 权重下映射自动为恒等，行为与
        # 未开启时一致）
        self.use_fp16 = engine.use_fp16
        self.map_cls = engine.map_cls
        self.class_map_active = engine.class_map_active
        self.logger.info(
            "跟踪器已接入检测引擎：模型与 FP16/类别映射判定结果共享自 "
            "src.engine.DetectionEngine（权重路径 %s）", engine.weights_path)

        # 确认后端标识：正常为 bytetrack，自动降级后变为 iou-fallback
        self.backend = BACKEND_BYTETRACK
        # 降级路径持有的 A5 IOU 确认器（未降级时为 None）
        self._fallback = None

        # 轨迹生命周期状态：{track_id: {cls, age, confirmed,
        #   confirm_frame, last_frame, center, velocity}}；
        # age 为累计被观测到的帧数；center 为最近一次实测框中心点，
        # velocity 为最近一次实测速度（(vx, vy)，单位 像素/帧，
        # 由相邻两个检测帧的中心位移除以帧间隔得到，A13 跳帧外推用）
        self._tracks = {}
        # 轨迹锚点尾迹：{track_id: deque([(x, y), ...], maxlen=尾迹长度)}
        self.trails = {}
        self._last_frame_no = None
        # A13：最近一次真实检测帧（update 调用）的帧号；外推时只有
        # 在该帧仍被观测到的轨迹才参与，防止目标消失后外推幽灵框
        self._last_detect_frame_no = None
        # A13：最近一帧画面尺寸 (w, h)，外推框越界钳制用
        self._last_frame_size = None
        # 累计确认事件列表：{track_id, cls, frame_no}（与 A5 契约一致）
        self.confirmed_events = []

    # ---------------- 状态管理 ----------------

    def reset(self):
        """清空全部轨迹状态与尾迹（切换视频源/重新开播时调用）。

        同时重置 ultralytics 内部跟踪器状态：丢弃当前 predictor 使其
        在下一次 track() 调用时重建，避免上一段视频的轨迹 ID 接续到
        新视频。降级路径的 IOU 确认器一并重置。
        """
        self._tracks = {}
        self.trails = {}
        self._last_frame_no = None
        self._last_detect_frame_no = None
        self._last_frame_size = None
        self.confirmed_events = []
        if self._fallback is not None:
            self._fallback.reset()
        # 让 ultralytics 重建 predictor（其内部持有 ByteTrack 轨迹缓存）
        try:
            if getattr(self.model, "predictor", None) is not None:
                self.model.predictor = None
        except Exception as exc:  # 重置失败不影响主流程，仅记日志
            self.logger.warning("重置 ultralytics 内部跟踪器状态失败：%s", exc)

    # ---------------- 降级路径 ----------------

    def _degrade_to_iou(self, exc):
        """model.track() 失败后的自动降级：切换到 A5 IOU 确认路径。

        只在首次失败时执行一次：创建内部 FrameConfirmer、切换 backend
        标识并记录日志；此后所有帧都走降级路径，不再尝试 track()。
        """
        self._fallback = FrameConfirmer(self.config)
        self.backend = BACKEND_IOU_FALLBACK
        self.logger.warning(
            "ByteTrack 跟踪调用失败（%s），自动降级为 A5 多帧 IOU 确认路径"
            "（backend=%s）；目标 ID 只在相邻帧间有效，轨迹尾迹不再更新",
            exc, self.backend,
        )

    def _predict_detections(self, frame):
        """用 model.predict() 取一帧检测列表（降级路径的数据来源）。

        后处理走 src.postprocess 公共实现（类别映射 -> 关注类别过滤
        -> 组装统一结构），与 src.detector.Detector.detect 一致：
        仅保留 Config.target_classes 中的关注类别。
        """
        results = self.model.predict(
            frame, imgsz=self.config.imgsz, verbose=False,
            half=self.use_fp16)[0]
        return collect_detections(
            results.boxes, self.names, self.config.target_classes,
            map_cls=self.map_cls)

    # ---------------- 主接口 ----------------

    def update(self, frame, frame_no):
        """输入一帧原始画面，执行 ByteTrack 跟踪并更新轨迹确认状态。

        参数：
            frame:    OpenCV 读取的 BGR 图像（numpy.ndarray）。
            frame_no: 当前帧号（应逐帧连续递增；不连续时清空轨迹重计）。

        返回：
            src.frame_confirm.ConfirmResult，结构与 A5 完全一致：
            records 中 track_id 为 ByteTrack 持久 ID、consecutive 为该
            轨迹累计存在帧数、iou 恒为 None（跟踪模式不做相邻帧 IOU
            匹配）；result.confirmed 为本帧可进入告警与计数的目标。
            已自动降级时返回降级路径的 ConfirmResult（backend 为
            BACKEND_IOU_FALLBACK）。
        """
        # 帧号不连续（跳帧、重新开播）：轨迹"连续存在"的前提被破坏，
        # 清空轨迹状态与尾迹重计（ultralytics 内部状态由 reset 语义
        # 交由上层切换视频源时处理，这里只清本模块状态）
        if self._last_frame_no is not None \
                and frame_no != self._last_frame_no + 1:
            self.logger.info(
                "帧号不连续（%d -> %d），跟踪轨迹状态清空重计",
                self._last_frame_no, frame_no,
            )
            self._tracks = {}
            self.trails = {}
        self._last_frame_no = frame_no

        # 已降级：直接走 A5 IOU 确认路径（检测结果由 predict 提供）
        if self._fallback is not None:
            return self._fallback.update(
                self._predict_detections(frame), frame_no)

        # 正常路径：ByteTrack 检测 + 关联（persist=True 保持跨帧轨迹）
        try:
            results = self.model.track(
                frame, persist=True, tracker=self._tracker_arg,
                imgsz=self.config.imgsz, verbose=False,
                half=self.use_fp16)[0]
        except Exception as exc:
            # 跟踪不可用（跟踪器配置缺失、后端异常等）：自动降级，
            # 本帧起走 IOU 路径，对外 ConfirmResult 契约不变
            self._degrade_to_iou(exc)
            return self._fallback.update(
                self._predict_detections(frame), frame_no)

        # A13：记录最近一次真实检测帧号与画面尺寸，供跳帧外推使用
        self._last_detect_frame_no = frame_no
        self._last_frame_size = (frame.shape[1], frame.shape[0])

        detections = self._collect_tracks(results)
        records = []
        for det, track_id in detections:
            state = self._tracks.get(track_id)
            if state is None:
                # 新轨迹：累计存在帧数从 1 计起
                state = {
                    "cls": det["cls"],
                    "age": 0,
                    "confirmed": False,
                    "confirm_frame": None,
                    "last_frame": frame_no,
                    "center": None,
                    "velocity": (0.0, 0.0),
                    "box": None,
                    "conf": det["conf"],
                }
                self._tracks[track_id] = state
                self.trails[track_id] = deque(maxlen=self.tail_length)
            # A13：由相邻两个真实检测帧的框中心位移估计轨迹速度
            # （像素/帧），供跳帧期间线性外推；首次观测速度记 0
            cx, cy = det["center"]
            if state["center"] is not None:
                gap = frame_no - state["last_frame"]
                if gap > 0:
                    state["velocity"] = (
                        (cx - state["center"][0]) / gap,
                        (cy - state["center"][1]) / gap,
                    )
            state["center"] = (cx, cy)
            state["box"] = det["box"]
            state["conf"] = det["conf"]
            state["age"] += 1
            state["last_frame"] = frame_no
            # 轨迹尾迹追加本帧锚点（底部中心点，与 A4/A6 语义一致）
            ax, ay = anchor_point(det)
            self.trails[track_id].append((int(ax), int(ay)))

            # 轨迹累计存在满 N 帧首次达标：本帧为确认事件帧
            just_confirmed = False
            if not state["confirmed"] and state["age"] >= self.confirm_frames:
                state["confirmed"] = True
                state["confirm_frame"] = frame_no
                just_confirmed = True
                self.confirmed_events.append({
                    "track_id": track_id,
                    "cls": det["cls"],
                    "frame_no": frame_no,
                })
            reason = "%s：轨迹#%d %s 累计存在%d帧（ByteTrack 持久ID）%s" % (
                "确认" if state["confirmed"] else "待确认",
                track_id, det["cls"], state["age"],
                "，本帧达标确认" if just_confirmed else "",
            )
            records.append({
                "det": det,
                "track_id": track_id,
                "consecutive": state["age"],
                "confirmed": state["confirmed"],
                "just_confirmed": just_confirmed,
                "confirm_frame": state["confirm_frame"],
                "iou": None,
                "extrapolated": False,
                "reason": reason,
            })

        # 遗忘长期未再出现的轨迹：防止状态随运行时间无界增长
        stale = [tid for tid, st in self._tracks.items()
                 if frame_no - st["last_frame"] > self.track_max_age]
        for tid in stale:
            self.logger.debug(
                "帧%d 轨迹#%d 超过%d帧未出现，清除其确认状态与尾迹",
                frame_no, tid, self.track_max_age,
            )
            del self._tracks[tid]
            self.trails.pop(tid, None)

        return ConfirmResult(records, frame_no, backend=self.backend)

    # ---------------- 跳帧外推（A13） ----------------

    def extrapolate(self, frame, frame_no):
        """跳帧期间的外推更新：不做模型推理，按轨迹实测速度线性外推位置。

        Config.detect_interval > 1 时，上层在中间帧（非检测帧）调用本
        方法替代 update()：对"最近一次真实检测帧仍被观测到"的每条轨迹，
        按 state["velocity"]（像素/帧）把框中心平移 steps 帧的距离
        （steps = 当前帧号 - 最近一次实测帧号），框宽高不变，越界钳制
        到画面范围内；产出 record["extrapolated"]=True 的 ConfirmResult，
        供 GUI 渲染中间帧目标位置与下游模块（越线/徘徊/热力）连续消费。

        语义约定：
            1. 外推帧不推进轨迹 age、不改变确认状态、不产生确认事件
               ——确认判定只认真实观测帧；
            2. 最近一次检测帧未再被观测到的轨迹不参与外推（目标已消失
               或被漏检），防止外推出幽灵框；
            3. 已自动降级到 IOU 路径时，外推无轨迹速度可依，退化为
               用 predict 做一次真实检测（行为与降级 update 一致）。

        参数：
            frame:    当前帧 BGR 画面（仅降级路径与画面尺寸钳制使用）。
            frame_no: 当前帧号（须与 update 的帧号序列保持连续递增）。

        返回：
            ConfirmResult；正常路径 records 全部 extrapolated=True。
        """
        # 帧号连续性保护与 update 一致：不连续时清空轨迹状态重计
        if self._last_frame_no is not None \
                and frame_no != self._last_frame_no + 1:
            self.logger.info(
                "帧号不连续（%d -> %d），跟踪轨迹状态清空重计",
                self._last_frame_no, frame_no,
            )
            self._tracks = {}
            self.trails = {}
        self._last_frame_no = frame_no
        if frame is not None:
            self._last_frame_size = (frame.shape[1], frame.shape[0])

        # 已降级到 IOU 路径：外推不可用，退化为一次真实检测确认
        if self._fallback is not None:
            return self._fallback.update(
                self._predict_detections(frame), frame_no)

        records = []
        for track_id, state in self._tracks.items():
            # 只有最近一次真实检测帧仍被观测到的轨迹才外推
            if self._last_detect_frame_no is None \
                    or state["last_frame"] != self._last_detect_frame_no:
                continue
            steps = frame_no - state["last_frame"]
            if steps <= 0 or state["center"] is None:
                continue
            vx, vy = state.get("velocity", (0.0, 0.0))
            cx, cy = state["center"]
            ncx, ncy = cx + vx * steps, cy + vy * steps
            # 框整体按中心位移平移，宽高不变；越界钳制到画面范围内
            det = self._shift_det(state, ncx - cx, ncy - cy)
            # 外推锚点追加进尾迹，保证跳帧期间轨迹连线连续
            ax, ay = anchor_point(det)
            if track_id in self.trails:
                self.trails[track_id].append((int(ax), int(ay)))
            reason = ("外推：轨迹#%d %s 按实测速度(%.1f,%.1f)像素/帧外推"
                      "%d帧（跳帧中间帧，非实测观测）") % (
                track_id, det["cls"], vx, vy, steps,
            )
            records.append({
                "det": det,
                "track_id": track_id,
                "consecutive": state["age"],
                "confirmed": state["confirmed"],
                "just_confirmed": False,
                "confirm_frame": state["confirm_frame"],
                "iou": None,
                "extrapolated": True,
                "reason": reason,
            })
        return ConfirmResult(records, frame_no, backend=self.backend)

    def _shift_det(self, state, dx, dy):
        """把轨迹最近一次实测框按 (dx, dy) 平移并钳制到画面内（A13）。

        框宽/高保持不变；置信度沿用最近一次实测值。返回与 Detector
        输出结构一致的检测项字典。
        """
        x1, y1, x2, y2 = state["box"]
        w, h = x2 - x1, y2 - y1
        nx1, ny1 = x1 + dx, y1 + dy
        if self._last_frame_size is not None:
            fw, fh = self._last_frame_size
            nx1 = min(max(nx1, 0.0), max(fw - w, 0.0))
            ny1 = min(max(ny1, 0.0), max(fh - h, 0.0))
        nx2, ny2 = nx1 + w, ny1 + h
        return {
            "cls": state["cls"],
            "conf": state.get("conf", 0.0),
            "box": [int(nx1), int(ny1), int(nx2), int(ny2)],
            "center": [int((nx1 + nx2) / 2), int((ny1 + ny2) / 2)],
        }

    def _collect_tracks(self, results):
        """把 ultralytics track() 结果整理为 (检测项, 跟踪ID) 列表。

        检测项结构与 Detector 输出对齐（{cls, conf, box, center}），
        后处理走 src.postprocess 公共实现（类别映射 -> 关注类别过滤
        -> 组装统一结构）；仅保留 Config.target_classes 关注类别。
        个别框未分配跟踪 ID（id 为 None，例如首帧尚未关联成功）时
        跳过，不参与轨迹确认。
        """
        return collect_track_detections(
            results.boxes, self.names, self.config.target_classes,
            map_cls=self.map_cls)

    # ---------------- 轨迹查询 ----------------

    def get_trail(self, track_id):
        """返回指定轨迹的锚点尾迹列表 [(x, y), ...]（旧->新顺序）。"""
        trail = self.trails.get(track_id)
        return list(trail) if trail is not None else []

    def active_tracks(self):
        """返回当前存活的轨迹状态快照 {track_id: 状态字典拷贝}。"""
        return {tid: dict(st) for tid, st in self._tracks.items()}
