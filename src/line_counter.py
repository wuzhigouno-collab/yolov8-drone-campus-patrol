# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

越线方向计数模块（A6）。

功能意图：
    在巡检画面中设置一条虚拟警戒线（Config.warning_line，归一化坐标），
    对多帧确认模块（A5）输出的已确认目标轨迹进行越线监测：当某个已确认
    目标的轨迹锚点在相邻帧之间从警戒线一侧穿越到另一侧时，判定为一次
    穿线事件，并按穿越方向分别累计"进/出"双向计数。穿线事件携带时间戳
    （视频内时间）、帧号、方向、类别、轨迹号与穿线位置，供 GUI 告警列表、
    日志追溯与测试断言使用。

    本模块只消费 src.frame_confirm.ConfirmResult 的确认记录（records 中
    confirmed=True 的项携带 track_id），不自行做帧间匹配；未确认目标
    不进入计数，与 A5"未确认目标不进入告警与计数"的契约一致。

锚点定义（与 A4 分区域阈值模块一致）：
    轨迹穿线判定取检测框"底部中心点" ((x1+x2)/2, y2) 作为锚点。
    俯视画面中目标底边更接近其地面实际位置，且底部中心对框高抖动
    不敏感，穿线时机判定更稳定。锚点计算直接复用
    src.zone_threshold.anchor_point，保证两处语义一致。

方向语义（务必结合 Config.line_enter_direction 理解）：
    警戒线按 (x1, y1) -> (x2, y2) 视为有向线段。锚点 p 相对线段的有向
    距离 d = cross(p2-p1, p-p1) / |p2-p1|（屏幕坐标系，y 轴向下）。
    对默认的自左向右水平警戒线：d > 0 一侧为画面"下方侧"，
    d < 0 一侧为画面"上方侧"；对非水平线，"下方侧/上方侧"推广为
    "有向距离为正/负的一侧"。
    Config.line_enter_direction = True（默认）：
        锚点由 下方侧 -> 上方侧（下->上）穿线记为"进"(in)，
        由 上方侧 -> 下方侧（上->下）穿线记为"出"(out)；
    False 时两种方向的进出语义互换。
    对非水平线，进出按同样的"有向距离符号跳变"规则映射：
    正->负 的跳变等价于上述"下->上"，负->正 等价于"上->下"。

防抖与防护：
    1. 死区（Config.line_deadzone_px）：锚点距线距离不超过该值时视为
       "压线"，所在侧维持上一次判定不变，防止目标沿线抖动造成
       同一目标在同一条线上反复计数；
    2. 方向翻转才计数：只有相邻帧所在侧发生 正<->负 跳变才产生事件，
       同侧持续移动不产生事件；目标穿回对侧后再穿出属于合法的新事件；
    3. 帧号不连续（跳帧/重新开播）时清空全部轨迹侧记忆，避免把两段
       不相邻的轨迹拼接成一次"穿线"；
    4. 目标消失超过 Config.line_track_max_age 帧后，其所在侧记忆被
       遗忘，防止长期运行时内存无界增长。
"""

import time

import cv2

from src.config import Config
from src.logger import get_logger
from src.zone_threshold import anchor_point

# 方向标识常量：进 / 出
DIR_IN = "in"
DIR_OUT = "out"
# 方向中文名映射（告警列表与日志展示用；OpenCV 画面叠加仍用英文）
DIR_NAME_CN = {DIR_IN: "进", DIR_OUT: "出"}

# 锚点所在侧标识：有向距离为正 / 为负 / 死区内（压线）
SIDE_POSITIVE = 1   # 默认水平线下对应"画面下方侧"
SIDE_NEGATIVE = -1  # 默认水平线下对应"画面上方侧"
SIDE_ON_LINE = 0    # 死区内，维持上一次所在侧


class LineCountResult:
    """单帧越线计数结果。

    属性：
        events: 本帧新产生的穿线事件列表，每项为字典：
            frame_no     —— 穿线发生的帧号
            time_sec     —— 视频内时间（秒，frame_no/fps；fps 未知为 None）
            wall_time    —— 墙上时钟时间字符串（系统本地时间）
            direction    —— 穿线方向（"in"/"out"）
            direction_cn —— 方向中文名（"进"/"出"）
            cls          —— 目标类别名
            conf         —— 目标置信度
            track_id     —— A5 确认记录中的轨迹号
            box          —— 穿线帧目标框 xyxy 像素坐标
            cross_point  —— 轨迹与警戒线的交点像素坐标 (x, y)
            from_side    —— 穿线前所在侧（SIDE_POSITIVE/SIDE_NEGATIVE）
            to_side      —— 穿线后所在侧
            reason       —— 文字原因说明（供日志与测试断言）
        frame_no: 本帧帧号
        count_in / count_out: 截至本帧的累计进/出计数
        line_px: 警戒线像素坐标 (x1, y1, x2, y2)
    """

    def __init__(self, events, frame_no, count_in, count_out, line_px):
        self.events = events
        self.frame_no = frame_no
        self.count_in = count_in
        self.count_out = count_out
        self.line_px = line_px

    def __len__(self):
        return len(self.events)


class LineCounter:
    """越线方向计数器。

    职责：
        1. 把 Config.warning_line 的归一化坐标换算为当前画面尺寸下的
           像素线段；
        2. 逐帧消费多帧确认结果（ConfirmResult），对已确认目标维护
           "锚点位于警戒线哪一侧"的记忆；
        3. 相邻帧所在侧发生正<->负跳变时产生穿线事件，按
           Config.line_enter_direction 判定进/出并双向计数；
        4. 提供警戒线、进出方向箭头与双向计数的画面叠加绘制接口。

    本类不修改 Config，参数在构造时从 Config 读取快照；运行期改
    配置后重新构造即可生效。构造时必须给定画面尺寸（像素），
    切换视频源后应重新构造或调用 reset()。
    """

    def __init__(self, config=None, frame_w=0, frame_h=0):
        """初始化计数器。

        参数：
            config: Config 实例；为 None 时内部新建默认配置。
            frame_w / frame_h: 画面尺寸（像素），用于把归一化警戒线
                坐标换算为像素坐标；必须为正数。
        """
        self.config = config if config is not None else Config()
        self.logger = get_logger(__name__, self.config)
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError(
                "LineCounter 需要有效的画面尺寸: %sx%s" % (frame_w, frame_h))
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        # 参数快照：方向语义、死区像素、轨迹侧记忆保留帧数
        self.enter_direction = bool(self.config.line_enter_direction)
        self.deadzone_px = float(self.config.line_deadzone_px)
        self.track_max_age = int(self.config.line_track_max_age)
        # 归一化警戒线 -> 像素线段（钳制到画面范围内）
        self.line_px = self._line_to_pixels(self.config.warning_line)
        x1, y1, x2, y2 = self.line_px
        # 线段方向向量与长度（长度<=0 的退化线段按点处理，判定恒为压线）
        self._vx = float(x2 - x1)
        self._vy = float(y2 - y1)
        self._vlen = (self._vx ** 2 + self._vy ** 2) ** 0.5
        # 轨迹侧记忆：{track_id: {"side", "last_frame", "anchor"}}
        self._tracks = {}
        self._last_frame_no = None
        # 双向累计计数与全部穿线事件（供统计与测试断言）
        self.count_in = 0
        self.count_out = 0
        self.events = []

    # ---------------- 坐标与几何 ----------------

    def _line_to_pixels(self, norm_line):
        """把归一化警戒线 (x1, y1, x2, y2) 换算为像素坐标并钳制到画面内。"""
        nx1, ny1, nx2, ny2 = (float(v) for v in norm_line)
        x1 = int(round(min(max(nx1, 0.0), 1.0) * self.frame_w))
        y1 = int(round(min(max(ny1, 0.0), 1.0) * self.frame_h))
        x2 = int(round(min(max(nx2, 0.0), 1.0) * self.frame_w))
        y2 = int(round(min(max(ny2, 0.0), 1.0) * self.frame_h))
        return (x1, y1, x2, y2)

    def _signed_distance(self, px, py):
        """计算锚点相对警戒线的有向距离（像素）。

        返回正值表示锚点位于线段有向距离为正的一侧（默认水平线的
        画面下方），负值表示另一侧；线段退化为点时返回 0.0。
        """
        if self._vlen <= 0:
            return 0.0
        x1, y1, _x2, _y2 = self.line_px
        # 二维叉积 cross(v, p-p1)，再除以线段长度得到垂直距离
        cross = self._vx * (py - y1) - self._vy * (px - x1)
        return cross / self._vlen

    def _side_of(self, px, py):
        """判定锚点位于警戒线哪一侧（带死区）。

        有向距离绝对值不超过死区（Config.line_deadzone_px）时返回
        SIDE_ON_LINE，表示"压线"——所在侧维持上一次判定不变。
        """
        dist = self._signed_distance(px, py)
        if dist > self.deadzone_px:
            return SIDE_POSITIVE
        if dist < -self.deadzone_px:
            return SIDE_NEGATIVE
        return SIDE_ON_LINE

    def _cross_point(self, prev_xy, cur_xy):
        """计算轨迹段（上一帧锚点 -> 本帧锚点）与警戒线的交点。

        用参数方程求解两线交点；轨迹与警戒线近似平行（分母为 0）时
        退化为本帧锚点。交点坐标钳制到警戒线段的包围盒内，便于绘制。
        """
        ax, ay = prev_xy
        bx, by = cur_xy
        x1, y1, x2, y2 = self.line_px
        denom = self._vx * (ay - by) - self._vy * (ax - bx)
        if abs(denom) < 1e-9:
            cx, cy = bx, by
        else:
            t = (self._vx * (ay - y1) - self._vy * (ax - x1)) / denom
            t = min(max(t, 0.0), 1.0)
            cx = ax + t * (bx - ax)
            cy = ay + t * (by - ay)
        # 显示用：钳制到警戒线段的包围盒范围
        cx = min(max(cx, min(x1, x2)), max(x1, x2))
        cy = min(max(cy, min(y1, y2)), max(y1, y2))
        return (int(round(cx)), int(round(cy)))

    def _direction_of(self, from_side, to_side):
        """按 Config.line_enter_direction 把侧跳变映射为进/出方向。

        正->负 跳变（默认水平线的 下->上）在 enter_direction=True 时
        记为"进"，False 时记为"出"；负->正 跳变反之。
        """
        if from_side == SIDE_POSITIVE and to_side == SIDE_NEGATIVE:
            return DIR_IN if self.enter_direction else DIR_OUT
        return DIR_OUT if self.enter_direction else DIR_IN

    # ---------------- 状态管理 ----------------

    def reset(self):
        """清空轨迹侧记忆、双向计数与事件列表（切换视频源时调用）。"""
        self._tracks = {}
        self._last_frame_no = None
        self.count_in = 0
        self.count_out = 0
        self.events = []

    # ---------------- 主接口 ----------------

    def update(self, confirm_result, frame_no, fps=0.0):
        """输入一帧多帧确认结果，执行越线判定并更新双向计数。

        参数：
            confirm_result: src.frame_confirm.ConfirmResult，仅其中
                confirmed=True 的记录（携带 track_id）参与穿线判定；
                未确认目标不进入计数（与 A5 契约一致）。
            frame_no: 当前帧号（应逐帧连续递增；不连续时自动清空
                轨迹侧记忆，重新计）。
            fps: 视频帧率，用于把帧号换算为视频内时间戳；未知传 0。

        返回：
            LineCountResult，含本帧新产生的穿线事件列表与截至本帧的
            双向累计计数。
        """
        # 帧号不连续（跳帧、重新开播）：相邻帧的前提被破坏，清空侧记忆
        if self._last_frame_no is not None \
                and frame_no != self._last_frame_no + 1:
            self.logger.info(
                "帧号不连续（%d -> %d），越线计数轨迹侧记忆清空重计",
                self._last_frame_no, frame_no,
            )
            self._tracks = {}
        self._last_frame_no = frame_no

        frame_events = []
        for rec in confirm_result.records:
            # 只统计已确认目标：未确认目标不进入告警与计数
            if not rec["confirmed"]:
                continue
            det = rec["det"]
            track_id = rec["track_id"]
            ax, ay = anchor_point(det)
            side = self._side_of(ax, ay)
            state = self._tracks.get(track_id)

            if state is not None and side != SIDE_ON_LINE \
                    and state["side"] != SIDE_ON_LINE \
                    and side != state["side"]:
                # 相邻帧所在侧发生 正<->负 跳变：判定为一次穿线事件
                direction = self._direction_of(state["side"], side)
                cross_xy = self._cross_point(state["anchor"], (ax, ay))
                if direction == DIR_IN:
                    self.count_in += 1
                else:
                    self.count_out += 1
                event = {
                    "frame_no": frame_no,
                    "time_sec": (round(frame_no / float(fps), 2)
                                 if fps and fps > 0 else None),
                    "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "direction": direction,
                    "direction_cn": DIR_NAME_CN[direction],
                    "cls": det["cls"],
                    "conf": det.get("conf"),
                    "track_id": track_id,
                    "box": list(det["box"]),
                    "cross_point": cross_xy,
                    "from_side": state["side"],
                    "to_side": side,
                    "reason": "%s：目标#%d %s 锚点(%.0f,%.0f) 由%s穿至%s，"
                              "判为越线%s" % (
                                  DIR_NAME_CN[direction], track_id,
                                  det["cls"], ax, ay,
                                  "正侧" if state["side"] == SIDE_POSITIVE
                                  else "负侧",
                                  "正侧" if side == SIDE_POSITIVE else "负侧",
                                  DIR_NAME_CN[direction]),
                }
                self.events.append(event)
                frame_events.append(event)
                self.logger.info(
                    "帧%d 越线%s：目标#%d %s 穿线于(%d,%d)（视频内时间 %s），"
                    "累计 进%d 出%d",
                    frame_no, DIR_NAME_CN[direction], track_id, det["cls"],
                    cross_xy[0], cross_xy[1],
                    ("%.2fs" % event["time_sec"])
                    if event["time_sec"] is not None else "未知",
                    self.count_in, self.count_out,
                )

            # 更新侧记忆：压线（死区内）时维持上一次所在侧不变，
            # 仅刷新锚点与最后出现帧号
            if state is None:
                self._tracks[track_id] = {
                    "side": side, "last_frame": frame_no,
                    "anchor": (ax, ay),
                }
            else:
                if side != SIDE_ON_LINE:
                    state["side"] = side
                state["last_frame"] = frame_no
                state["anchor"] = (ax, ay)

        # 遗忘长期消失的轨迹：防止侧记忆随运行时间无界增长
        stale = [tid for tid, st in self._tracks.items()
                 if frame_no - st["last_frame"] > self.track_max_age]
        for tid in stale:
            self.logger.debug(
                "帧%d 轨迹#%d 超过%d帧未出现，遗忘其所在侧",
                frame_no, tid, self.track_max_age,
            )
            del self._tracks[tid]

        return LineCountResult(
            frame_events, frame_no, self.count_in, self.count_out,
            self.line_px)

    # ---------------- 画面叠加 ----------------

    def draw_overlay(self, frame):
        """在画面上叠加警戒线、进/出方向箭头与双向计数（原地绘制）。

        绘制内容：
            1. 警戒线（青色实线 + 两端点圆点），线起点旁标注 WARNING LINE；
            2. 线段中点处的"进"方向箭头（指向 enter 语义的目标侧）；
            3. 画面左上角叠加双向计数 "IN: n  OUT: m"。
        OpenCV putText 不支持中文，画面标注统一用英文短词；
        中文文案由 tkinter 控件层展示。返回绘制后的画面（同一对象）。
        """
        x1, y1, x2, y2 = self.line_px
        # 警戒线（青色）与两端点
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.circle(frame, (x1, y1), 5, (255, 255, 0), -1)
        cv2.circle(frame, (x2, y2), 5, (255, 255, 0), -1)
        # 警戒线名称标注（放在线起点内侧，避免出画）
        cv2.putText(frame, "WARNING LINE", (min(x1 + 8, self.frame_w - 150),
                                            max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # "进"方向箭头：指向 enter 语义的目标侧。
        # enter=True 时"进"为 正侧->负侧，箭头指向负侧法向 (-(-vy), -(vx))；
        # 即单位法向 n=(-(-vy), -(vx))/L 中使有向距离减小的方向。
        if self._vlen > 0:
            ux, uy = self._vx / self._vlen, self._vy / self._vlen
            # 有向距离沿法向 (-uy, ux) 增大；enter=True 时进方向为其反向
            nx, ny = -uy, ux
            if self.enter_direction:
                nx, ny = -nx, -ny
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            arrow_len = 40.0
            tip = (int(round(mx + nx * arrow_len)),
                   int(round(my + ny * arrow_len)))
            cv2.arrowedLine(frame, (int(round(mx)), int(round(my))), tip,
                            (255, 255, 0), 2, tipLength=0.3)
            cv2.putText(frame, "IN", (tip[0] + 6, tip[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 左上角双向计数叠加（带底色条保证可读；y=48 避开区域标注）
        text = "IN: %d  OUT: %d" % (self.count_in, self.count_out)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (4, 30), (12 + tw, 30 + th + 10),
                      (255, 255, 0), -1)
        cv2.putText(frame, text, (8, 30 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return frame
