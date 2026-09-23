# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

人流车流热力图模块（A8）。

功能意图：
    巡检过程中"目标经常出现在哪里"是评估校园重点区域（路口、
    广场、出入口）人流车流密度的核心依据。本模块在巡检播放期间
    持续累积"已确认目标"（A5 多帧确认结果中 confirmed=True 的
    记录）的锚点位置，形成一张与画面同尺寸的热度累积图；需要时
    以高斯累积渲染为伪彩热力图，叠加到当前底帧或黑底上导出 PNG，
    供事后分析与汇报使用。

    锚点定义与 A4 完全一致：取检测框"底部中心点" ((x1+x2)/2, y2)，
    直接复用 src.zone_threshold.anchor_point，保证"目标在哪个
    地面位置"的语义在区域判定、越线计数与热力累积三处一致。

    累积方式：每个轨迹点向累积图盖一个二维高斯"印章"——核半径
    走 Config.heatmap_kernel_radius，峰值强度走 Config.
    heatmap_intensity；同一点位被反复踩到时热度线性叠加。渲染时
    按当前最大值归一化到 0~255 再套 JET 伪彩（蓝冷红热），
    叠加底帧时只对热度>0 的像素按 Config.heatmap_overlay_alpha
    混合，无热区域保持底帧原样。

    只有"已确认"目标才进热力累积：未确认的瞬时误检（反光、
    背景抖动）不污染热力分布，与告警/计数的准入口径一致。

接口：
    update(confirm_result, frame_w, frame_h) —— 累积一帧的已确认目标锚点
    add_point(x, y, frame_w, frame_h)        —— 直接累积单个点（测试用）
    render(base_frame=None)                  —— 渲染为 BGR 热力图
    export_png(path, base_frame=None)        —— 渲染并导出 PNG 文件
    get_points() / point_count               —— 累积点查询
    reset()                                  —— 清空累积
"""

import os

import cv2
import numpy as np

from src.config import Config
from src.logger import get_logger
from src.zone_threshold import anchor_point


class HeatmapAccumulator:
    """人流车流热力累积器。

    职责：
        1. 维护一张与画面同尺寸的 float32 热度累积图，逐帧把已确认
           目标的底部中心锚点以高斯核"印章"累加进去；
        2. 渲染时按累积最大值归一化并套 JET 伪彩，可叠加到指定底帧
           （按透明度混合，仅热区生效）或黑底；
        3. 导出 PNG（imencode+tofile，规避 Windows 下 imwrite 对
           非 ASCII 路径静默失败的问题）；
        4. 提供累积点查询（get_points / point_count）与重置（reset）
           接口，供 GUI 状态展示与测试断言使用。

    本类不修改 Config，核半径/强度/叠加透明度在构造时读取快照；
    运行期改配置后重新构造即可生效。画面尺寸在首次累积时确定，
    之后尺寸变化（切换视频源）会自动清空重计。
    """

    def __init__(self, config=None):
        """初始化热力累积器。

        参数：
            config: Config 实例；为 None 时内部新建默认配置。
        """
        self.config = config if config is not None else Config()
        self.logger = get_logger(__name__, self.config)
        # 高斯核半径与单点峰值强度（构造时取配置快照）
        self.kernel_radius = max(1, int(self.config.heatmap_kernel_radius))
        self.intensity = float(self.config.heatmap_intensity)
        # 叠加底帧时的混合透明度（钳制到 0~1）
        self.overlay_alpha = min(max(float(self.config.heatmap_overlay_alpha),
                                     0.0), 1.0)
        # 预生成二维高斯"印章"：(2r+1) 见方，sigma 取半径的一半，
        # 峰值归一后乘以单点强度——半高处约为峰值的 0.5，半径外趋近 0
        ksize = 2 * self.kernel_radius + 1
        sigma = self.kernel_radius / 2.0
        k1d = cv2.getGaussianKernel(ksize, sigma)
        kernel = (k1d @ k1d.T).astype(np.float32)
        kernel /= kernel.max()
        self._kernel = kernel * self.intensity
        # 热度累积图（float32，与画面同尺寸；首次累积时创建）
        self._accum = None
        self._frame_size = None  # (w, h)
        # 全部累积点列表 [(x, y), ...]，供查询与测试断言
        self._points = []

    # ---------------- 累积 ----------------

    def _ensure_size(self, frame_w, frame_h):
        """确保累积图与画面尺寸一致；尺寸变化时自动清空重计。"""
        size = (int(frame_w), int(frame_h))
        if self._accum is not None and self._frame_size == size:
            return
        if self._accum is not None:
            self.logger.info(
                "画面尺寸变化 %s -> %s，热力累积清空重计",
                self._frame_size, size,
            )
            self._points = []
        self._frame_size = size
        self._accum = np.zeros((size[1], size[0]), dtype=np.float32)

    def _stamp(self, x, y):
        """在 (x, y) 处盖一个高斯"印章"（越界部分自动裁剪）。"""
        w, h = self._frame_size
        r = self.kernel_radius
        cx, cy = int(round(x)), int(round(y))
        # 印章在画面内的有效范围（中心越界时对应裁剪核的边缘）
        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
        y1, y2 = max(0, cy - r), min(h, cy + r + 1)
        if x1 >= x2 or y1 >= y2:
            return  # 锚点完全落在画面外（异常检测框），忽略
        kx1, ky1 = x1 - (cx - r), y1 - (cy - r)
        self._accum[y1:y2, x1:x2] += \
            self._kernel[ky1:ky1 + (y2 - y1), kx1:kx1 + (x2 - x1)]

    def add_point(self, x, y, frame_w, frame_h):
        """直接累积单个轨迹点（像素坐标）。

        参数：
            x / y: 轨迹点像素坐标（浮点亦可，内部取整）。
            frame_w / frame_h: 画面尺寸（像素），首次调用时确定累积图
                尺寸；与既有尺寸不一致时自动清空重计。
        """
        self._ensure_size(frame_w, frame_h)
        self._stamp(x, y)
        self._points.append((int(round(x)), int(round(y))))

    def update(self, confirm_result, frame_w, frame_h):
        """累积一帧多帧确认结果中"已确认"目标的锚点。

        参数：
            confirm_result: src.frame_confirm.ConfirmResult；
                只有 records 中 confirmed=True 的记录参与累积
                （未确认目标不进入热力，与告警/计数准入口径一致）；
                传 None 时本帧不累积（多帧确认开关关闭等场景）。
            frame_w / frame_h: 当前画面尺寸（像素）。

        返回：
            本帧实际累积的点数。
        """
        if confirm_result is None:
            return 0
        self._ensure_size(frame_w, frame_h)
        added = 0
        for rec in confirm_result.records:
            if not rec["confirmed"]:
                continue
            # 锚点定义与 A4/A6 一致：检测框底部中心
            px, py = anchor_point(rec["det"])
            self._stamp(px, py)
            self._points.append((int(round(px)), int(round(py))))
            added += 1
        return added

    # ---------------- 查询与重置 ----------------

    @property
    def point_count(self):
        """已累积的轨迹点总数。"""
        return len(self._points)

    @property
    def frame_size(self):
        """累积图尺寸 (w, h)；尚无累积时为 None。"""
        return self._frame_size

    def get_points(self):
        """返回全部累积点的副本列表 [(x, y), ...]。"""
        return list(self._points)

    @property
    def accum(self):
        """热度累积图（float32 只读视图；无累积时为 None）。"""
        return self._accum

    def reset(self):
        """清空热度累积图与累积点（切换视频源/重新开播时调用）。"""
        self._accum = None
        self._frame_size = None
        self._points = []

    # ---------------- 渲染与导出 ----------------

    def render(self, base_frame=None):
        """把当前热度累积渲染为 BGR 热力图。

        渲染流程：累积图按最大值归一化到 0~255 -> JET 伪彩
        （蓝冷红热）-> 叠加到底帧或黑底：
            - base_frame 为 None：黑底，只有热度>0 的像素显示彩色；
            - base_frame 非 None：热度>0 的像素按 overlay_alpha 与
              底帧混合，无热区域保持底帧原样。

        参数：
            base_frame: 底帧 BGR 图像（尺寸须与累积图一致），或 None。

        返回：
            BGR uint8 热力图。

        异常：
            尚无任何累积点时抛出 ValueError（调用方应提示"先累积"）。
        """
        if not self._points or self._accum is None:
            raise ValueError(
                "热力累积为空：尚无已确认目标轨迹点，无法渲染热力图；"
                "请先运行巡检累积目标后再导出")
        w, h = self._frame_size
        max_v = float(self._accum.max())
        if max_v <= 0:
            raise ValueError("热力累积图全零（强度配置异常），无法渲染热力图")
        norm = (self._accum / max_v * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        hot_mask = norm > 0  # 热度>0 的像素才参与叠加，无热区保持原样

        if base_frame is None:
            out = np.zeros((h, w, 3), dtype=np.uint8)
            out[hot_mask] = color[hot_mask]
            return out
        if base_frame.shape[:2] != (h, w):
            raise ValueError(
                "底帧尺寸 %s 与热力累积尺寸 %s 不一致" % (
                    base_frame.shape[:2][::-1], (w, h)))
        out = base_frame.copy()
        blended = cv2.addWeighted(base_frame, 1.0 - self.overlay_alpha,
                                  color, self.overlay_alpha, 0)
        out[hot_mask] = blended[hot_mask]
        return out

    def export_png(self, path, base_frame=None):
        """渲染当前热力图并导出为 PNG 文件。

        参数：
            path: 导出文件路径（目录不存在时自动创建）。
            base_frame: 底帧 BGR 图像或 None（黑底）。

        返回：
            导出记录字典：{path, point_count, frame_size}。

        异常：
            无任何累积点或 PNG 编码失败时抛出 ValueError。
        """
        heat_img = self.render(base_frame)  # 空累积时在此抛 ValueError
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        # 用 imencode + tofile 而非 cv2.imwrite，规避 Windows 下
        # imwrite 对非 ASCII 路径静默失败的问题
        ok, buf = cv2.imencode(".png", heat_img)
        if not ok:
            raise ValueError("热力图 PNG 编码失败")
        buf.tofile(path)
        record = {
            "path": os.path.abspath(path),
            "point_count": self.point_count,
            "frame_size": self._frame_size,
        }
        self.logger.info(
            "热力图已导出：%s（累积 %d 个轨迹点，底帧：%s）",
            record["path"], record["point_count"],
            "黑底" if base_frame is None else "当前画面",
        )
        return record
