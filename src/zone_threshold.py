# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

分区域差异化阈值过滤模块（A4）。

功能意图：
    无人机巡检画面中，中央区域多为俯拍远处的小目标，置信度天然偏低；
    边缘区域目标较大、畸变较小，低置信度目标更多是误检。
    本模块把画面按 Config 配置的比例划分为"中央区"与"边缘区"，
    按目标所在区域应用不同的置信度阈值：中央区用较低阈值保召回，
    边缘区用较高阈值压误检，实现分区域差异化过滤。

    过滤结果保留每个目标的完整判定记录（所在区域、适用阈值、
    保留/过滤结论与文字原因），供 GUI 渲染、日志输出与测试断言使用。
    输入结构与 src.detector.Detector 输出结构完全对齐：
    [{cls, conf, box, center}, ...]，box 为 xyxy 像素坐标。

锚点定义说明：
    目标区域判定取检测框"底部中心点" ((x1+x2)/2, y2) 作为锚点，
    而不是检测框几何中心。原因：俯视画面中目标可见部分的底边更接
    近其在地面上的实际站立/停放位置，用底部中心判定"目标处于哪个
    区域"更贴合一一对应的地面语义；同时底部中心对框高抖动（目标
    被遮挡、截断导致框高变化）不如几何中心敏感，区域归属更稳定。
"""

import cv2

from src.config import Config
from src.logger import get_logger

# 区域标识常量：中央区 / 边缘区
ZONE_CENTER = "center"
ZONE_EDGE = "edge"

# 区域中文名映射（用于原因说明与画面标注之外的日志展示）
ZONE_NAME_CN = {ZONE_CENTER: "中央区", ZONE_EDGE: "边缘区"}


def calc_center_zone(frame_w, frame_h, ratio_w, ratio_h):
    """计算中央区矩形（像素坐标 xyxy），相对画面居中对称。

    参数：
        frame_w: 画面宽度（像素）
        frame_h: 画面高度（像素）
        ratio_w: 中央区宽度占画面宽度的比例（0~1）
        ratio_h: 中央区高度占画面高度的比例（0~1）

    返回：
        (x1, y1, x2, y2) 整数像素坐标，中央区矩形的左上与右下角。
        比例超出 (0, 1] 范围时会被钳制到合法区间，避免非法矩形。
    """
    ratio_w = min(max(float(ratio_w), 0.0), 1.0)
    ratio_h = min(max(float(ratio_h), 0.0), 1.0)
    zone_w = frame_w * ratio_w
    zone_h = frame_h * ratio_h
    x1 = int(round((frame_w - zone_w) / 2.0))
    y1 = int(round((frame_h - zone_h) / 2.0))
    x2 = int(round((frame_w + zone_w) / 2.0))
    y2 = int(round((frame_h + zone_h) / 2.0))
    return (x1, y1, x2, y2)


def anchor_point(det):
    """返回目标区域判定的锚点：检测框底部中心点 ((x1+x2)/2, y2)。

    参数：
        det: 单个检测项字典，至少含 "box": [x1, y1, x2, y2]。

    返回：
        (px, py) 浮点像素坐标。选用底部中心的理由见模块 docstring。
    """
    x1, _y1, x2, y2 = det["box"]
    return ((x1 + x2) / 2.0, float(y2))


def point_in_rect(px, py, rect):
    """判断点是否在矩形内（边界含入内，即落在分界线上算中央区）。"""
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


class ZoneFilterResult:
    """单帧分区域阈值过滤结果。

    属性：
        records: 判定记录列表，每项为字典：
            det       —— 原检测项字典（{cls, conf, box, center}）
            zone      —— 目标所在区域（"center"/"edge"）
            threshold —— 该区域适用的置信度阈值
            kept      —— True 保留 / False 过滤
            reason    —— 文字原因说明（供日志与测试断言）
        zone_rect: 本帧使用的中央区矩形 (x1, y1, x2, y2)
        frame_w / frame_h: 判定所用的画面尺寸
    """

    def __init__(self, records, zone_rect, frame_w, frame_h):
        self.records = records
        self.zone_rect = zone_rect
        self.frame_w = frame_w
        self.frame_h = frame_h

    @property
    def kept(self):
        """被保留的检测项列表（保持原输入顺序）。"""
        return [r["det"] for r in self.records if r["kept"]]

    @property
    def filtered(self):
        """被过滤掉的检测项列表（保持原输入顺序）。"""
        return [r["det"] for r in self.records if not r["kept"]]

    def __len__(self):
        return len(self.records)


class ZoneThresholdFilter:
    """分区域差异化阈值过滤器。

    职责：
        1. 按 Config 中的区域比例与两区置信度阈值，
           对 Detector 输出的检测列表逐目标判定 保留/过滤；
        2. 每个目标生成含区域、阈值与原因的判定记录，
           供日志输出与测试断言；
        3. 提供中央区分界线叠加绘制接口，供 GUI 画面渲染使用。

    本类不修改 Config，阈值与比例均在构造时或 filter() 调用时
    从 Config 读取，运行期改配置后重新构造即可生效。
    """

    def __init__(self, config=None):
        """初始化过滤器。

        参数：
            config: Config 实例；为 None 时内部新建默认配置。
        """
        self.config = config if config is not None else Config()
        self.logger = get_logger(__name__, self.config)

    def center_zone_rect(self, frame_w, frame_h):
        """按当前配置计算给定画面尺寸下的中央区矩形。"""
        return calc_center_zone(
            frame_w, frame_h,
            self.config.center_zone_ratio_w,
            self.config.center_zone_ratio_h,
        )

    def locate_zone(self, det, frame_w, frame_h):
        """判定单个目标所在区域：返回 "center" 或 "edge"。"""
        px, py = anchor_point(det)
        rect = self.center_zone_rect(frame_w, frame_h)
        return ZONE_CENTER if point_in_rect(px, py, rect) else ZONE_EDGE

    def filter(self, detections, frame_w, frame_h):
        """对单帧检测列表执行分区域差异化阈值过滤。

        参数：
            detections: Detector 输出的检测列表 [{cls, conf, box, center}, ...]
            frame_w / frame_h: 画面尺寸（像素），用于计算中央区范围。

        返回：
            ZoneFilterResult，含每个目标的判定记录；
            result.kept 为保留列表，result.filtered 为过滤列表。
        """
        zone_rect = self.center_zone_rect(frame_w, frame_h)
        records = []
        for det in detections:
            px, py = anchor_point(det)
            zone = ZONE_CENTER if point_in_rect(px, py, zone_rect) else ZONE_EDGE
            # 按所在区域取对应阈值：中央区低阈值保召回，边缘区高阈值压误检
            threshold = (self.config.center_conf_threshold
                         if zone == ZONE_CENTER
                         else self.config.edge_conf_threshold)
            kept = det["conf"] >= threshold
            reason = "%s：%s conf=%.2f %s 阈值%.2f（锚点(%.0f,%d)）" % (
                "保留" if kept else "过滤",
                ZONE_NAME_CN[zone], det["conf"],
                ">=" if kept else "<", threshold, px, int(py),
            )
            records.append({
                "det": det,
                "zone": zone,
                "threshold": threshold,
                "kept": kept,
                "reason": reason,
            })
        return ZoneFilterResult(records, zone_rect, frame_w, frame_h)

    def draw_zone_border(self, frame):
        """在画面上叠加中央区/边缘区分界线，返回标注后的 BGR 图像。

        绘制内容：中央区矩形边框（黄色）、矩形左上角标注 "CENTER ZONE"、
        画面左上角标注 "EDGE ZONE"，便于直观区分两个区域。
        输入画面不被修改（内部先拷贝）。
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        x1, y1, x2, y2 = self.center_zone_rect(w, h)
        # 中央区分界线（黄色矩形）
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # 区域文字标注（OpenCV putText 不支持中文，区域名用英文标注）
        cv2.putText(annotated, "CENTER ZONE", (x1 + 6, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated, "EDGE ZONE", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return annotated
