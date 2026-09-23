# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

徘徊检测模块（A11）。

功能意图：
    校园安全巡检中，"人员长时间滞留某一区域"是需要重点关注的异常
    行为。本模块在巡检画面中划定一个徘徊监控区域（Config.loitering_
    zone，归一化坐标），对确认模块输出的已确认目标轨迹进行停留监测：
    同一跟踪 ID 的锚点（检测框底部中心点，与 A4/A6 一致）连续停留
    在区域内的时间超过阈值秒数（Config.loitering_duration_sec）时，
    触发一次徘徊告警。

    告警事件结构与 A7 告警归档器（src.archiver.AlertArchiver.archive）
    兼容：事件字典携带事件类型（EVENT_LOITERING）、类别、置信度、
    位置（锚点像素坐标）与目标框，调用方直接把事件字段传给
    archiver.archive(frame, event_type, cls, conf, position) 即可形成
    "截图 + CSV 日志"证据链。

触发与防抖规则：
    1. 只监测已确认目标（ConfirmResult.records 中 confirmed=True 且
       携带 track_id 的记录），未确认目标不参与徘徊判定，与 A5/A11
       "未确认目标不进入告警与计数"的契约一致；
    2. 锚点连续位于区域内才累计停留时长；锚点离开区域即清零，
       重新进入重新计时；
    3. 每次"连续驻留"只触发一次告警（触发后该次驻留不再重复告警），
       目标离开区域后再次驻留超时属于合法的新告警；
    4. 帧号不连续（跳帧/重新开播）时清空全部驻留状态，避免把两段
       不相邻的轨迹拼接成一次"长时间停留"；
    5. 目标消失超过 Config.track_max_age 帧后其驻留状态被遗忘，
       防止长时间运行时状态无界增长（与跟踪模块的遗忘语义一致）。

锚点定义（与 A4 分区域阈值模块一致）：
    徘徊判定取检测框"底部中心点" ((x1+x2)/2, y2) 作为锚点，俯视画面
    中底边更接近目标的地面实际位置，区域归属与停留判定更稳定。
    锚点计算复用 src.zone_threshold.anchor_point。
"""

import time

import cv2

from src.config import Config
from src.logger import get_logger
from src.zone_threshold import anchor_point, point_in_rect

# 徘徊告警事件类型（告警日志 CSV"事件类型"字段值）
EVENT_LOITERING = "徘徊告警"


class LoiteringResult:
    """单帧徘徊检测结果。

    属性：
        events: 本帧新产生的徘徊告警事件列表，每项为字典：
            frame_no     —— 告警触发帧号
            time_sec     —— 视频内时间（秒，frame_no/fps；fps 未知为 None）
            wall_time    —— 墙上时钟时间字符串（系统本地时间）
            event_type   —— 事件类型（EVENT_LOITERING）
            cls          —— 目标类别名
            conf         —— 目标置信度
            track_id     —— 确认记录中的轨迹号（跟踪模式下为持久 ID）
            box          —— 触发帧目标框 xyxy 像素坐标
            anchor       —— 触发帧锚点像素坐标 (x, y)（告警位置）
            dwell_sec    —— 触发时已连续停留时长（秒）
            reason       —— 文字原因说明（供日志与测试断言）
        frame_no: 本帧帧号
        zone_px: 徘徊监控区域像素坐标 (x1, y1, x2, y2)
        dwell: 截至本帧处于区域内目标的停留时长 {track_id: 秒}
    """

    def __init__(self, events, frame_no, zone_px, dwell):
        self.events = events
        self.frame_no = frame_no
        self.zone_px = zone_px
        self.dwell = dwell

    def __len__(self):
        return len(self.events)


class LoiteringDetector:
    """徘徊检测器：区域内停留超时触发徘徊告警。

    职责：
        1. 把 Config.loitering_zone 的归一化坐标换算为当前画面尺寸下
           的像素矩形（越界钳制到画面内）；
        2. 逐帧消费确认结果（ConfirmResult），对已确认目标维护"锚点在
           区域内的连续驻留时长"；
        3. 驻留时长达到 Config.loitering_duration_sec 且本次驻留尚未
           告警时，产生徘徊告警事件（每次连续驻留只告警一次）；
        4. 提供徘徊监控区域与实时停留时长的画面叠加绘制接口。

    本类不修改 Config，参数在构造时从 Config 读取快照；运行期改
    配置后重新构造即可生效。构造时必须给定画面尺寸（像素），
    切换视频源后应重新构造或调用 reset()。
    """

    def __init__(self, config=None, frame_w=0, frame_h=0):
        """初始化徘徊检测器。

        参数：
            config: Config 实例；为 None 时内部新建默认配置。
            frame_w / frame_h: 画面尺寸（像素），用于把归一化区域
                坐标换算为像素坐标；必须为正数。
        """
        self.config = config if config is not None else Config()
        self.logger = get_logger(__name__, self.config)
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError(
                "LoiteringDetector 需要有效的画面尺寸: %sx%s"
                % (frame_w, frame_h))
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        # 参数快照：停留阈值秒数、驻留状态遗忘帧数（复用跟踪遗忘语义）
        self.duration_sec = float(self.config.loitering_duration_sec)
        self.track_max_age = int(self.config.track_max_age)
        # 归一化徘徊区域 -> 像素矩形（钳制到画面范围内）
        self.zone_px = self._zone_to_pixels(self.config.loitering_zone)
        # 驻留状态：{track_id: {cls, enter_frame, dwell_frames,
        #   alerted, last_frame, anchor}}
        self._stays = {}
        self._last_frame_no = None
        # 最近一次 update 使用的有效帧率（驻留帧数 -> 秒换算用）
        self._eff_fps = 25.0
        # 全部徘徊告警事件（供统计与测试断言）
        self.events = []

    # ---------------- 坐标换算 ----------------

    def _zone_to_pixels(self, norm_zone):
        """把归一化徘徊区域 (x1, y1, x2, y2) 换算为像素坐标并钳制。"""
        nx1, ny1, nx2, ny2 = (float(v) for v in norm_zone)
        x1 = int(round(min(max(nx1, 0.0), 1.0) * self.frame_w))
        y1 = int(round(min(max(ny1, 0.0), 1.0) * self.frame_h))
        x2 = int(round(min(max(nx2, 0.0), 1.0) * self.frame_w))
        y2 = int(round(min(max(ny2, 0.0), 1.0) * self.frame_h))
        return (x1, y1, x2, y2)

    def point_in_zone(self, px, py):
        """判断像素点是否位于徘徊监控区域内（边界含入内）。"""
        return point_in_rect(px, py, self.zone_px)

    # ---------------- 状态管理 ----------------

    def reset(self):
        """清空全部驻留状态与告警事件（切换视频源时调用）。"""
        self._stays = {}
        self._last_frame_no = None
        self.events = []

    # ---------------- 主接口 ----------------

    def update(self, confirm_result, frame_no, fps=0.0):
        """输入一帧确认结果，执行徘徊判定并更新驻留状态。

        参数：
            confirm_result: src.frame_confirm.ConfirmResult，仅其中
                confirmed=True 的记录（携带 track_id）参与徘徊判定；
                未确认目标不进入徘徊监测（与 A5/A11 契约一致）。
            frame_no: 当前帧号（应逐帧连续递增；不连续时自动清空
                驻留状态，重新计）。
            fps: 视频帧率，用于把驻留帧数换算为停留秒数；取不到时
                按 25fps 折算并在日志说明一次。

        返回：
            LoiteringResult，含本帧新产生的徘徊告警事件列表与当前
            区域内各目标的停留时长。
        """
        # 帧号不连续（跳帧、重新开播）：连续驻留的前提被破坏，清空重计
        if self._last_frame_no is not None \
                and frame_no != self._last_frame_no + 1:
            self.logger.info(
                "帧号不连续（%d -> %d），徘徊驻留状态清空重计",
                self._last_frame_no, frame_no,
            )
            self._stays = {}
        self._last_frame_no = frame_no

        # 帧率兜底：取不到真实 fps 时按 25fps 折算停留秒数
        eff_fps = float(fps) if fps and fps > 0 else 25.0
        self._eff_fps = eff_fps

        frame_events = []
        seen_in_zone = set()
        for rec in confirm_result.records:
            # 只监测已确认目标：未确认目标不进入告警与计数
            if not rec["confirmed"]:
                continue
            det = rec["det"]
            track_id = rec["track_id"]
            ax, ay = anchor_point(det)

            if not self.point_in_zone(ax, ay):
                # 锚点离开区域：本次驻留结束，状态清除（再进重新计时）
                if track_id in self._stays:
                    self.logger.debug(
                        "帧%d 目标#%d 锚点离开徘徊区域，驻留清零",
                        frame_no, track_id,
                    )
                    del self._stays[track_id]
                continue

            seen_in_zone.add(track_id)
            stay = self._stays.get(track_id)
            if stay is None:
                # 新进入区域：开始计时
                stay = {
                    "cls": det["cls"],
                    "enter_frame": frame_no,
                    "dwell_frames": 0,
                    "alerted": False,
                    "last_frame": frame_no,
                    "anchor": (ax, ay),
                }
                self._stays[track_id] = stay
            # 驻留帧数 +1（同一帧同一 ID 只出现一次，逐帧累计）
            stay["dwell_frames"] += 1
            stay["last_frame"] = frame_no
            stay["anchor"] = (ax, ay)
            dwell_sec = stay["dwell_frames"] / eff_fps

            # 停留超阈值且本次驻留尚未告警：触发徘徊告警（一次驻留一次）
            if not stay["alerted"] and dwell_sec >= self.duration_sec:
                stay["alerted"] = True
                event = {
                    "frame_no": frame_no,
                    "time_sec": (round(frame_no / eff_fps, 2)
                                 if fps and fps > 0 else None),
                    "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event_type": EVENT_LOITERING,
                    "cls": det["cls"],
                    "conf": det.get("conf"),
                    "track_id": track_id,
                    "box": list(det["box"]),
                    "anchor": (int(ax), int(ay)),
                    "dwell_sec": round(dwell_sec, 2),
                    "reason": "徘徊告警：目标#%d %s 锚点(%.0f,%.0f) 在监控区域"
                              "内已连续停留 %.1f 秒（阈值 %.1f 秒，进入帧 %d）" % (
                                  track_id, det["cls"], ax, ay,
                                  dwell_sec, self.duration_sec,
                                  stay["enter_frame"]),
                }
                self.events.append(event)
                frame_events.append(event)
                self.logger.info(
                    "帧%d %s", frame_no, event["reason"],
                )

        # 遗忘长期消失的目标：防止驻留状态随运行时间无界增长
        stale = [tid for tid, st in self._stays.items()
                 if frame_no - st["last_frame"] > self.track_max_age]
        for tid in stale:
            self.logger.debug(
                "帧%d 目标#%d 超过%d帧未出现，遗忘其徘徊驻留状态",
                frame_no, tid, self.track_max_age,
            )
            del self._stays[tid]

        # 当前区域内各目标停留时长（秒），供 GUI 实时叠加显示
        dwell = {tid: round(st["dwell_frames"] / eff_fps, 2)
                 for tid, st in self._stays.items()
                 if tid in seen_in_zone}

        return LoiteringResult(frame_events, frame_no, self.zone_px, dwell)

    # ---------------- 画面叠加 ----------------

    def draw_overlay(self, frame, confirm_result=None):
        """在画面上叠加徘徊监控区域与实时停留时长（原地绘制）。

        绘制内容：
            1. 徘徊监控区域（品红色矩形 + LOITER ZONE 标注）；
            2. 区域内各目标的实时停留时长 "DWELL x.xs"（锚点旁）；
               停留已超阈值的用时长的告警红显示。
        OpenCV putText 不支持中文，画面标注统一用英文短词；
        中文文案由 tkinter 控件层展示。返回绘制后的画面（同一对象）。
        """
        x1, y1, x2, y2 = self.zone_px
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, "LOITER ZONE", (x1 + 8, max(y1 + 22, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        # 各在区域目标的实时停留时长标注（已告警的用红色，区别于常态品红）
        for tid, stay in self._stays.items():
            if stay["last_frame"] != self._last_frame_no:
                continue
            ax, ay = stay["anchor"]
            dwell_sec = stay["dwell_frames"] / self._eff_fps
            color = (0, 0, 255) if stay["alerted"] else (255, 0, 255)
            text_pos = (int(ax) + 8, max(int(ay) - 8, 16))
            cv2.putText(frame, "#%d STAY %.1fs" % (tid, dwell_sec), text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
