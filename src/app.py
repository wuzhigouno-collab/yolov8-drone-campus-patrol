# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

GUI 主框架模块（系统入口）。

功能意图：
    基于 tkinter 搭建系统主界面，提供视频画面区（检测框、类别、
    置信度实时叠加）、视频源选择（文件路径输入/浏览或摄像头设备号）、
    开始/停止控制按钮、底部状态栏（当前帧号/状态信息）与告警列表区
    （本阶段显示逐帧检测摘要）。检测推理复用 src.detector.Detector，
    视频输入复用 src.video_source 的统一 read() 接口，参数取值
    复用 src.config.Config，保证 GUI 与后续功能模块（分区阈值、
    多帧确认、越线计数、告警归档、热力图）数据来源一致。

    A4 增量：功能开关区提供"显示区域分界线"与"启用分区域置信度阈值
    过滤"两个开关（复选框）。前者开启时在画面上叠加中央区/边缘区分
    界线；后者开启时检测渲染接入 src.zone_threshold 的分区域差异
    化阈值过滤（被过滤目标绘为灰色并标注 filtered）。

    A5 增量：功能开关区新增"启用多帧确认降噪"开关（默认开启）。
    开启时检测渲染接入 src.frame_confirm 的多帧 IOU 确认机制：
    目标需连续 N 帧（Config.confirm_frames）稳定出现才标记为"已
    确认"；已确认目标绘为绿色实线框，未确认目标绘为黄色虚线框并
    标注连续帧数进度；告警列表区只记录已确认目标。开关关闭时，
    画面渲染与告警列表行为与 A4 完全一致。

    A6 增量：功能开关区新增"启用越线方向计数"开关（默认开启，
    默认值走 Config.line_count_enabled）。开启时画面叠加 Config
    .warning_line 定义的警戒线、"进"方向箭头与 IN/OUT 双向累计
    计数；已确认目标（A5 轨迹号）的底部中心锚点跨越警戒线时产生
    穿线事件（默认 下->上=进、上->下=出，由 Config.line_enter_
    direction 控制），事件写入告警列表并记日志。越线计数依赖多帧
    确认提供的轨迹号，关闭多帧确认时越线计数自动跳过并提示。
    开关关闭时，画面渲染与告警列表行为与 A5 完全一致。

    A7 增量：功能开关区新增"启用告警归档"开关（默认开启）。开启时
    接入 src.archiver.AlertArchiver：已确认目标首次进入告警范围
    （目标确认）与 A6 穿线事件发生时，自动把当前带标注画面（检测框、
    警戒线等叠加完毕的帧）截图保存到归档目录（Config.archive_dir，
    默认 archive/），文件名含毫秒时间戳与事件序号避免覆盖；同时向
    告警日志 CSV（Config.alert_log_name）追加一行：时间戳、事件类型、
    类别、置信度、位置、截图路径；告警列表同步显示归档条目。开关关闭
    时不截图、不写 CSV，告警列表行为回到 A6 状态。

    A11 增量：功能开关区新增"启用 ByteTrack 目标跟踪"（默认值走
    Config.tracking_enabled）与"启用徘徊检测告警"（默认值走
    Config.loitering_enabled）两个开关。跟踪开启时（依赖多帧确认开关
    同时开启）：检测/确认走 src.tracker.ByteTrackTracker（ultralytics
    model.track()），目标带持久 ID，画面叠加目标 ID 与最近若干帧
    （Config.track_tail_length）的轨迹尾迹；确认契约与 A5 一致
    （轨迹累计存在 N 帧才确认），跟踪调用失败时自动降级为 A5 IOU
    路径。跟踪关闭时回到 A5 多帧 IOU 确认行为。徘徊检测开启时：
    画面叠加徘徊监控区域（Config.loitering_zone），已确认目标在
    区域内停留超过 Config.loitering_duration_sec 秒触发徘徊告警，
    告警接入归档（截图+CSV+告警列表，事件类型"徘徊告警"）。

    A12 增量：功能开关区新增"启用切片推理"开关（默认值走
    Config.slice_enabled，默认关闭）。开启时检测路径从
    Detector.detect 整图直检替换为 src.slicer.SlicedDetector——
    帧切分为 2x2 重叠切片（重叠率走 Config.slice_overlap_ratio）
    逐片检测，坐标映射回原图后按 Config.slice_nms_iou 做合并
    NMS 去重，提升高空俯视小目标召回；检测框渲染、分区过滤、
    多帧确认等下游链路不变。注意：跟踪后端 model.track 内部为
    整图推理、不与切片叠加，切片开启期间 ByteTrack 跟踪自动让位，
    多帧确认走 A5 IOU 降级路径。开关关闭时保持整图直检原行为。

    A13 增量：推理加速。其一，FP16 半精度推理开关走配置
    （Config.fp16，默认开启）：检测/跟踪推理在具备 CUDA GPU 时以
    半精度执行，无 GPU 环境自动降级 FP32 并在启动日志说明。其二，
    功能开关区新增"启用跳帧检测加速"开关（默认值由
    Config.detect_interval > 1 决定）：开启且 ByteTrack 跟踪生效
    时，每隔 Config.detect_interval 帧执行一次检测+跟踪（模型推理
    只发生在检测帧），中间帧由跟踪器按轨迹实测速度线性外推目标
    位置渲染，外推目标绘为青色虚线框并标注 extrap（区别于实测
    检测框），确认计数只认实测帧。跳帧依赖跟踪后端：跟踪未开启
    或已自动降级（IOU 路径）时跳帧不生效，按逐帧检测处理并记
    日志说明。

用法示例：
    python src/app.py                                  # 正常启动 GUI
    python src/app.py --video 测试视频路径              # 启动并直接加载视频
    python src/app.py --video 测试视频路径 --max-frames 60 --dump-frame tests/a3_gui_frame.png
        # 自动化验收：播放到第 60 帧后自动退出，退出前保存带标注画面
"""

import argparse
import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

# 本文件位于 src/ 下，直接运行 python src/app.py 时 sys.path 不含项目
# 根目录，这里手动补上，保证 src 包内各模块可被正常导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.archiver import (
    EVENT_CONFIRM,
    EVENT_CROSS_IN,
    EVENT_CROSS_OUT,
    AlertArchiver,
)
from src.config import Config
from src.detector import Detector
from src.engine import DetectionEngine
from src.frame_confirm import ConfirmResult, FrameConfirmer
from src.heatmap import HeatmapAccumulator
from src.line_counter import LineCounter
from src.logger import get_logger
from src.loitering import EVENT_LOITERING, LoiteringDetector
from src.slicer import SlicedDetector
from src.tracker import BACKEND_BYTETRACK, ByteTrackTracker
from src.video_source import create_video_source
from src.zone_threshold import ZoneThresholdFilter

# 窗口标题（验收标准要求的固定标题）
WINDOW_TITLE = "鹰眼巡校——无人机校园安全智能巡检系统 V1.0"


def parse_args():
    """解析命令行参数。

    --video       启动后直接加载的视频源（文件路径或摄像头设备号）
    --max-frames  自动化验收用：播放到第 N 帧后自动退出主循环并正常关闭
    --dump-frame  自动化验收用：退出前把当前带标注画面保存为图片文件
    """
    parser = argparse.ArgumentParser(description=WINDOW_TITLE)
    parser.add_argument(
        "--video", default=None,
        help="启动后直接加载的视频源：视频文件路径或摄像头设备号（如 0）",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="播放到第 N 帧后自动退出（自动化验收用，不影响正常 GUI 行为）",
    )
    parser.add_argument(
        "--dump-frame", default=None,
        help="退出前将当前带标注画面保存为图片（自动化验收用）",
    )
    return parser.parse_args()


class EagleEyeApp:
    """鹰眼巡校主界面应用类。

    职责：
        1. 搭建 tkinter 主界面（视频画面区 / 视频源选择 / 控制按钮 /
           告警列表区 / 状态栏）；
        2. 通过 after() 定时驱动逐帧读取 -> YOLO 检测 -> 标注 -> 显示
           的播放循环，全部在主线程执行，避免 tkinter 跨线程操作；
        3. 维护帧号、检测摘要等运行状态并实时刷新状态栏与告警列表。
    """

    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.config = Config()
        self.logger = get_logger(__name__, self.config)

        # ---------- 运行状态变量 ----------
        self.source = None          # 当前视频源（FileVideoSource/CameraVideoSource）
        self.is_running = False     # 播放循环开关
        self.frame_count = 0        # 当前已处理帧号
        self.current_annotated = None  # 当前带标注画面（BGR，供截图/导出用）
        self._after_id = None       # after() 定时任务句柄，停止时取消

        # ---------- A4 分区域阈值：界面开关变量 ----------
        # 区域边框显示开关（默认值取配置项 show_zone_border）
        self.show_zone_border_var = tk.BooleanVar(value=self.config.show_zone_border)
        # 分区阈值过滤开关（默认关闭：未开启时检测渲染行为与 A3 完全一致）
        self.enable_zone_filter_var = tk.BooleanVar(value=False)

        # ---------- A5 多帧确认降噪：界面开关变量 ----------
        # 多帧确认开关（默认开启）：目标需连续 N 帧稳定出现才确认，
        # 未确认目标以黄色虚线框显示且不进入告警列表
        self.enable_confirm_var = tk.BooleanVar(value=True)

        # ---------- A6 越线方向计数：界面开关变量 ----------
        # 越线计数开关（默认值走配置项 line_count_enabled）：开启时画面
        # 叠加警戒线与 IN/OUT 双向计数，已确认目标穿线事件写入告警列表
        self.enable_line_count_var = tk.BooleanVar(
            value=self.config.line_count_enabled)
        # 越线计数器在 start_patrol 打开视频源后按实际画面尺寸创建；
        # 未创建前为 None
        self.line_counter = None
        # "越线计数依赖多帧确认"提示只记一次日志，避免逐帧刷屏
        self._line_needs_confirm_warned = False

        # ---------- A7 告警归档：界面开关变量与归档器 ----------
        # 告警归档开关（默认开启）：开启时已确认目标告警与穿线事件
        # 自动截图存盘并向告警日志 CSV 追加记录；关闭时回到 A6 行为
        self.enable_archive_var = tk.BooleanVar(value=True)
        # 告警归档器：构造时自动创建归档目录与 CSV 表头；
        # 归档目录/日志文件名走 Config（archive_dir / alert_log_name）
        self.archiver = AlertArchiver(self.config)

        # ---------- A8 人流车流热力图：累积器 ----------
        # 热力累积器：播放期间持续累积"已确认目标"的底部中心锚点
        # （锚点定义与 A4 一致），核半径/强度/叠加透明度走 Config
        # （heatmap_kernel_radius / heatmap_intensity / heatmap_overlay_alpha）；
        # "导出热力图"按钮把当前累积渲染导出到归档目录
        self.heatmap = HeatmapAccumulator(self.config)

        # ---------- A11 ByteTrack 跟踪与徘徊检测：开关变量与模块 ----------
        # ByteTrack 跟踪开关（默认值走配置项 tracking_enabled）：开启时
        # 检测/确认走跟踪后端（目标带持久 ID + 轨迹尾迹叠加）；关闭时
        # 回到 A5 多帧 IOU 确认降级路径。该开关在多帧确认开关之上生效
        self.enable_tracking_var = tk.BooleanVar(
            value=self.config.tracking_enabled)
        # 徘徊检测开关（默认值走配置项 loitering_enabled）：开启时已确认
        # 目标在徘徊监控区域内停留超时触发徘徊告警并接入告警归档
        self.enable_loitering_var = tk.BooleanVar(
            value=self.config.loitering_enabled)
        # 跟踪器在共享检测引擎建立后创建（与检测/切片路径复用引擎持有
        # 的同一模型，权重不重复加载）；徘徊检测器在 start_patrol 打开
        # 视频源后按实际画面尺寸创建，未创建前为 None
        self.tracker = None
        self.loitering = None
        # "徘徊检测依赖多帧确认/跟踪"提示只记一次日志，避免逐帧刷屏
        self._loiter_needs_confirm_warned = False

        # ---------- A12 切片推理：界面开关变量 ----------
        # 切片推理开关（默认值走配置项 slice_enabled，默认关闭）：开启时
        # 检测路径从 Detector.detect 整图直检替换为 SlicedDetector
        # 重叠切片推理（逐片检测 + 坐标映射回原图 + 合并 NMS 去重），
        # 提升高空俯视小目标召回；关闭时保持整图直检原行为
        self.enable_slice_var = tk.BooleanVar(value=self.config.slice_enabled)

        # ---------- A13 推理加速：界面开关变量 ----------
        # 跳帧检测加速开关（默认值由配置项 detect_interval > 1 决定）：
        # 开启且 ByteTrack 跟踪生效时，每隔 Config.detect_interval 帧
        # 执行一次检测+跟踪，中间帧由跟踪器按轨迹实测速度外推渲染
        # （外推目标绘为青色虚线框）；跟踪未开启时跳帧不生效
        self.enable_skip_var = tk.BooleanVar(
            value=self.config.detect_interval > 1)
        # "跳帧依赖跟踪"提示只记一次日志，避免逐帧刷屏
        self._skip_needs_tracking_warned = False

        # 窗口基础设置
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1280x800")
        self.root.configure(bg="#2c3e50")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._setup_theme()
        self._build_layout()

        # 加载统一检测引擎（R2）：YOLO 权重加载耗时数秒，启动时一次性
        # 完成；检测/跟踪/切片三条推理路径共享同一 DetectionEngine 实例
        # ——同一模型、同一套 FP16/类别映射判定，权重只加载一次
        self._set_status("正在加载检测模型，请稍候……")
        self.root.update_idletasks()
        self.engine = DetectionEngine.shared(self.config)
        self.detector = Detector(self.config, engine=self.engine)
        # 分区域差异化阈值过滤器（A4），供播放循环按需启用
        self.zone_filter = ZoneThresholdFilter(self.config)
        # 多帧 IOU 确认器（A5）：无跟踪器时的降级确认路径，
        # 供播放循环在"启用多帧确认降噪"开关开启时使用
        self.frame_confirmer = FrameConfirmer(self.config)
        # A11：ByteTrack 跟踪器（共享同一检测引擎实例，权重不重复
        # 加载）；供播放循环在"启用 ByteTrack 目标跟踪"开关开启时使用
        self.tracker = ByteTrackTracker(self.config, engine=self.engine)
        # A12：切片推理检测器（同样共享同一检测引擎实例）；供播放
        # 循环在"启用切片推理"开关开启时替换 Detector.detect 检测路径
        self.slicer = SlicedDetector(self.config, engine=self.engine)
        self._set_status("模型加载完成，就绪——请选择视频源后点击“开始巡检”")
        self.logger.info("GUI 初始化完成：%s", WINDOW_TITLE)
        # A5 行为变化说明：多帧确认默认开启，未确认目标不进入告警列表
        self.logger.info(
            "A5 多帧确认降噪默认开启：目标需连续 %d 帧稳定出现"
            "（相邻帧同类框 IOU>=%.2f）才标记为已确认；未确认目标以黄色"
            "虚线框显示、不进入告警列表；已确认目标恢复绿色实线框。"
            "如需恢复逐帧直显旧行为，可在功能开关中关闭该选项",
            self.config.confirm_frames, self.config.iou_threshold,
        )
        # A6 行为变化说明：越线方向计数默认开启，画面新增警戒线/双向计数
        # 叠加，穿线事件写入告警列表
        self.logger.info(
            "A6 越线方向计数默认%s：已确认目标（多帧确认轨迹号）的底部中心"
            "跨越警戒线时按方向计数，方向语义 %s（由配置项 "
            "line_enter_direction 控制）；画面叠加警戒线与 IN/OUT 双向计数，"
            "穿线事件写入告警列表。相比 A5 的可观察变化：画面多出青色警戒线"
            "与计数文字、告警列表多出越线事件条目。如需恢复 A5 行为，可在"
            "功能开关中关闭“启用越线方向计数”",
            "开启" if self.config.line_count_enabled else "关闭",
            "下->上=进、上->下=出" if self.config.line_enter_direction
            else "上->下=进、下->上=出",
        )
        # A7 行为变化说明：告警归档默认开启——相比 A6 的可观察变化：
        # 归档目录下新增告警截图与告警日志 CSV，告警列表新增归档条目
        self.logger.info(
            "A7 告警归档默认开启：已确认目标首次进入告警范围（目标确认）"
            "与越线穿线事件发生时，自动把当前带标注画面截图保存到归档目录 "
            "%s（文件名含毫秒时间戳+事件序号，避免覆盖），并向告警日志 %s "
            "追加记录（时间戳/事件类型/类别/置信度/位置/截图路径）；告警列表"
            "同步显示归档条目。相比 A6 的可观察变化：归档目录新增截图与 CSV "
            "文件、告警列表新增“已归档”条目。如需恢复 A6 行为（不截图、"
            "不写 CSV），可在功能开关中关闭“启用告警归档”",
            self.archiver.archive_dir, self.archiver.csv_path,
        )
        # A8 行为变化说明：热力累积默认进行（依赖多帧确认，未确认目标
        # 不进入热力），不产生画面/列表变化；新增"导出热力图"按钮
        self.logger.info(
            "A8 人流车流热力图默认累积：巡检期间持续累积已确认目标的"
            "底部中心锚点（与 A4 锚点定义一致；多帧确认关闭时无已确认"
            "目标、热力不累积），高斯核半径 %d 像素、单点强度 %.2f、"
            "叠加透明度 %.2f（均由配置项控制）。点击右侧“导出热力图”"
            "按钮可把当前累积渲染导出到 %s。相比 A7 的可观察变化："
            "控制面板新增“热力图”操作区，画面渲染不变",
            self.heatmap.kernel_radius, self.heatmap.intensity,
            self.heatmap.overlay_alpha,
            os.path.join(self.config.abs_path(self.config.archive_dir),
                         self.config.heatmap_export_name),
        )
        # A11 行为变化说明：ByteTrack 跟踪与徘徊检测默认可由配置控制——
        # 相比 A8 的可观察变化：跟踪开启时目标标签带持久 ID、画面新增
        # 轨迹尾迹连线与徘徊监控区域框，告警列表/归档新增"徘徊告警"条目
        self.logger.info(
            "A11 ByteTrack 目标跟踪默认%s：检测/确认走 ultralytics "
            "model.track()（跟踪器 %s），目标带持久 ID 并叠加最近 %d 帧轨迹"
            "尾迹；确认契约与 A5 一致（轨迹累计存在 %d 帧才确认），跟踪调用"
            "失败时自动降级为 A5 IOU 路径（backend 变为 iou-fallback）。"
            "徘徊检测默认%s：监控区域 %s，已确认目标在区域内停留超过 %.1f 秒"
            "触发徘徊告警（事件类型“徘徊告警”，接入截图归档与告警列表）。"
            "如需恢复 A8 行为，可在功能开关中关闭“启用 ByteTrack 目标跟踪”"
            "与“启用徘徊检测告警”",
            "开启" if self.config.tracking_enabled else "关闭",
            self.config.tracker_name, self.config.track_tail_length,
            self.config.confirm_frames,
            "开启" if self.config.loitering_enabled else "关闭",
            self.config.loitering_zone, self.config.loitering_duration_sec,
        )
        # A12 行为变化说明：切片推理默认关闭，关闭时行为与之前完全一致；
        # 开启时的可观察变化：检测路径走切片管线（单帧推理耗时约为 4 片
        # 之和、小目标检出增多），跟踪自动让位（画面无 ID/轨迹尾迹，
        # 确认框号变为 IOU 降级路径的临时轨迹号）
        self.logger.info(
            "A12 切片推理默认%s：开启时检测路径从 Detector.detect 整图直检"
            "替换为 SlicedDetector 切片推理（%dx%d 重叠切片，重叠率 %.2f，"
            "逐片检测后坐标映射回原图，按 IOU>=%.2f 做类别感知合并 NMS 去重），"
            "提升高空俯视小目标召回；跟踪后端 model.track 内部为整图推理、"
            "不与切片叠加，切片开启期间 ByteTrack 跟踪自动让位，多帧确认走 "
            "A5 IOU 降级路径。默认关闭，关闭时保持整图直检原行为；如需启用"
            "可在功能开关中勾选“启用切片推理”",
            "开启" if self.config.slice_enabled else "关闭",
            self.config.slice_rows, self.config.slice_cols,
            self.config.slice_overlap_ratio, self.config.slice_nms_iou,
        )
        # A13 行为变化说明：FP16 半精度走配置（Detector/ByteTrackTracker
        # 初始化时已各自记录精度判定日志）；跳帧检测默认由
        # Config.detect_interval 决定，开启时的可观察变化：中间帧目标
        # 以青色虚线框 + extrap 标注渲染（非实测），实测帧渲染不变
        self.logger.info(
            "A13 推理加速：FP16 半精度开关 fp16=%s（实际精度判定见检测引擎"
            "初始化日志，无 CUDA 环境自动降级 FP32）；跳帧检测间隔 "
            "detect_interval=%d（1=不跳帧；GUI“启用跳帧检测加速”开关默认%s）："
            "跳帧生效时每隔 %d 帧执行一次检测+ByteTrack 跟踪，中间帧按轨迹"
            "实测速度线性外推目标位置（青色虚线框 + extrap 标注，确认计数"
            "只认实测帧）；跳帧依赖跟踪后端，跟踪未开启或已降级时按逐帧"
            "检测处理",
            self.config.fp16, self.config.detect_interval,
            "开启" if self.config.detect_interval > 1 else "关闭",
            max(self.config.detect_interval, 1),
        )

        # 命令行指定了视频源：填入输入框并自动开始播放
        if self.args.video:
            self.source_var.set(self.args.video)
            self.root.after(200, self.start_patrol)

    # ---------------- 界面搭建 ----------------

    def _setup_theme(self):
        """设置现代化深色主题样式（clam 主题 + 自定义配色）。"""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Main.TFrame", background="#34495e")
        style.configure("Title.TLabel", background="#2c3e50",
                        foreground="#ecf0f1", font=("微软雅黑", 15, "bold"))
        style.configure("Sub.TLabel", background="#34495e",
                        foreground="#bdc3c7", font=("微软雅黑", 10))
        style.configure("Status.TLabel", background="#2c3e50",
                        foreground="#ecf0f1", font=("微软雅黑", 9))

    def _build_layout(self):
        """搭建主界面布局：顶部标题、左侧视频区、右侧控制面板、底部状态栏。"""
        main = ttk.Frame(self.root, style="Main.TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 顶部标题栏
        title_bar = ttk.Frame(main, style="Main.TFrame")
        title_bar.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(title_bar, text="鹰眼巡校 · 无人机校园安全智能巡检系统",
                  style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(title_bar, text="V1.0 | YOLO 实时检测",
                  style="Sub.TLabel").pack(side=tk.RIGHT)

        # 中部内容区：左侧视频画面 + 右侧控制面板
        content = ttk.Frame(main, style="Main.TFrame")
        content.pack(fill=tk.BOTH, expand=True)
        self._build_video_area(content)
        self._build_control_panel(content)

        # 底部状态栏
        status_bar = ttk.Frame(main, style="Main.TFrame")
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(8, 0))
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_bar, textvariable=self.status_var,
                  style="Status.TLabel").pack(side=tk.LEFT, padx=5)
        self.frame_var = tk.StringVar(value="帧号: 0")
        ttk.Label(status_bar, textvariable=self.frame_var,
                  style="Status.TLabel").pack(side=tk.RIGHT, padx=5)

    def _build_video_area(self, parent):
        """搭建左侧视频画面区：Canvas 显示视频帧，下方为开始/停止按钮。"""
        video_frame = ttk.Frame(parent, style="Main.TFrame")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        ttk.Label(video_frame, text="巡检实时画面（检测框 / 类别 / 置信度实时叠加）",
                  style="Sub.TLabel").pack(pady=(0, 6))

        # 视频显示画布（深色底，无高亮边框）
        self.video_canvas = tk.Canvas(video_frame, bg="#1a252f",
                                      highlightthickness=0,
                                      width=860, height=600)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # 视频控制按钮
        btn_frame = ttk.Frame(video_frame, style="Main.TFrame")
        btn_frame.pack(fill=tk.X, pady=(8, 0))
        self.start_btn = ttk.Button(btn_frame, text="开始巡检",
                                    command=self.start_patrol)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_btn = ttk.Button(btn_frame, text="停止",
                                   command=self.stop_patrol, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)

    def _build_control_panel(self, parent):
        """搭建右侧控制面板：视频源选择 + 告警列表区。"""
        panel = ttk.Frame(parent, style="Main.TFrame", width=340)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)  # 固定右侧面板宽度

        # ----- 视频源选择区 -----
        src_frame = ttk.LabelFrame(panel, text="视频源选择", padding=8)
        src_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(src_frame, text="文件路径或摄像头设备号：",
                  style="Sub.TLabel").pack(anchor=tk.W)
        self.source_var = tk.StringVar(value=self.config.video_source)
        src_row = ttk.Frame(src_frame, style="Main.TFrame")
        src_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Entry(src_row, textvariable=self.source_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(src_row, text="浏览…", width=8,
                   command=self.browse_file).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(src_frame, text="提示：填纯数字（如 0）表示摄像头设备号",
                  style="Sub.TLabel").pack(anchor=tk.W, pady=(4, 0))

        # ----- 功能开关区（A4：区域边框显示 / 分区阈值过滤；A5：多帧确认） -----
        switch_frame = ttk.LabelFrame(panel, text="功能开关", padding=8)
        switch_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Checkbutton(
            switch_frame, text="显示区域分界线（中央区/边缘区）",
            variable=self.show_zone_border_var,
            bg="#34495e", fg="#ecf0f1", selectcolor="#1a252f",
            activebackground="#34495e", activeforeground="#ecf0f1",
            font=("微软雅黑", 9), anchor=tk.W,
        ).pack(fill=tk.X)
        tk.Checkbutton(
            switch_frame, text="启用分区域置信度阈值过滤",
            variable=self.enable_zone_filter_var,
            bg="#34495e", fg="#ecf0f1", selectcolor="#1a252f",
            activebackground="#34495e", activeforeground="#ecf0f1",
            font=("微软雅黑", 9), anchor=tk.W,
        ).pack(fill=tk.X)
        tk.Checkbutton(
            switch_frame, text="启用多帧确认降噪（未确认目标不告警）",
            variable=self.enable_confirm_var,
            bg="#34495e", fg="#ecf0f1", selectcolor="#1a252f",
            activebackground="#34495e", activeforeground="#ecf0f1",
            font=("微软雅黑", 9), anchor=tk.W,
        ).pack(fill=tk.X)
        tk.Checkbutton(
            switch_frame, text="启用越线方向计数（警戒线+进出计数）",
            variable=self.enable_line_count_var,
            bg="#34495e", fg="#ecf0f1", selectcolor="#1a252f",
            activebackground="#34495e", activeforeground="#ecf0f1",
            font=("微软雅黑", 9), anchor=tk.W,
        ).pack(fill=tk.X)
        tk.Checkbutton(
            switch_frame, text="启用告警归档（告警截图+CSV日志）",
            variable=self.enable_archive_var,
            bg="#34495e", fg="#ecf0f1", selectcolor="#1a252f",
            activebackground="#34495e", activeforeground="#ecf0f1",
            font=("微软雅黑", 9), anchor=tk.W,
        ).pack(fill=tk.X)
        tk.Checkbutton(
            switch_frame, text="启用 ByteTrack 目标跟踪（目标ID+轨迹尾迹）",
            variable=self.enable_tracking_var,
            bg="#34495e", fg="#ecf0f1", selectcolor="#1a252f",
            activebackground="#34495e", activeforeground="#ecf0f1",
            font=("微软雅黑", 9), anchor=tk.W,
        ).pack(fill=tk.X)
        tk.Checkbutton(
            switch_frame, text="启用徘徊检测告警（区域内停留超时）",
            variable=self.enable_loitering_var,
            bg="#34495e", fg="#ecf0f1", selectcolor="#1a252f",
            activebackground="#34495e", activeforeground="#ecf0f1",
            font=("微软雅黑", 9), anchor=tk.W,
        ).pack(fill=tk.X)
        tk.Checkbutton(
            switch_frame, text="启用切片推理（2×2重叠切片+合并去重）",
            variable=self.enable_slice_var,
            bg="#34495e", fg="#ecf0f1", selectcolor="#1a252f",
            activebackground="#34495e", activeforeground="#ecf0f1",
            font=("微软雅黑", 9), anchor=tk.W,
        ).pack(fill=tk.X)
        tk.Checkbutton(
            switch_frame, text="启用跳帧检测加速（间隔走配置 detect_interval）",
            variable=self.enable_skip_var,
            bg="#34495e", fg="#ecf0f1", selectcolor="#1a252f",
            activebackground="#34495e", activeforeground="#ecf0f1",
            font=("微软雅黑", 9), anchor=tk.W,
        ).pack(fill=tk.X)

        # ----- 热力图操作区（A8：导出当前累积的人流车流热力图） -----
        heat_frame = ttk.LabelFrame(panel, text="热力图（人流车流分布）", padding=8)
        heat_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(heat_frame, text="导出热力图",
                   command=self._export_heatmap).pack(fill=tk.X)
        ttk.Label(heat_frame,
                  text="导出当前累积热力图到归档目录 %s" %
                       self.config.heatmap_export_name,
                  style="Sub.TLabel", wraplength=300,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

        # ----- 告警列表区（多帧确认开启时只记录已确认目标，A5） -----
        alert_frame = ttk.LabelFrame(panel, text="告警 / 检测摘要", padding=8)
        alert_frame.pack(fill=tk.BOTH, expand=True)
        list_row = ttk.Frame(alert_frame, style="Main.TFrame")
        list_row.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_row, orient=tk.VERTICAL)
        self.alert_list = tk.Listbox(list_row, height=18,
                                     bg="#1a252f", fg="#ecf0f1",
                                     font=("微软雅黑", 9),
                                     yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.alert_list.yview)
        self.alert_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # ---------------- 视频源与控制 ----------------

    def browse_file(self):
        """弹出文件选择对话框，把选中的视频文件路径填入视频源输入框。"""
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                       ("所有文件", "*.*")],
        )
        if path:
            self.source_var.set(path)

    def start_patrol(self):
        """开始巡检：按输入的视频源打开源并启动逐帧播放循环。"""
        if self.is_running:
            return
        source_desc = self.source_var.get().strip()
        if not source_desc:
            self._set_status("请先填写视频源（文件路径或摄像头设备号）")
            return
        try:
            # 通过工厂函数创建视频源：数字按摄像头处理，其余按文件处理
            self.source = create_video_source(source_desc, self.config)
        except IOError as exc:
            self._set_status("打开视频源失败：%s" % exc)
            self.logger.error("打开视频源失败：%s", exc)
            return

        self.is_running = True
        self.frame_count = 0
        # 切换/重开视频源时清空多帧确认候选，避免跨视频错误接续
        self.frame_confirmer.reset()
        # A11：跟踪器与徘徊检测器同步重置，避免跨视频接续轨迹/驻留状态
        if self.tracker is not None:
            self.tracker.reset()
        # A8：热力累积同步清空重计，避免跨视频混合两段素材的热度分布
        self.heatmap.reset()
        # A6：按新视频源的实际画面尺寸重建越线计数器（警戒线归一化坐标
        # 换算为像素需要画面尺寸；重建同时完成计数与轨迹侧记忆清零）
        self.line_counter = LineCounter(
            self.config, self.source.width, self.source.height)
        self._line_needs_confirm_warned = False
        # A11：按新视频源的实际画面尺寸重建徘徊检测器（区域归一化坐标
        # 换算为像素需要画面尺寸；重建同时完成驻留状态与告警清零）
        self.loitering = LoiteringDetector(
            self.config, self.source.width, self.source.height)
        self._loiter_needs_confirm_warned = False
        # A13：跳帧提示标志随开播重置（跳帧开关状态在开播时记日志）
        self._skip_needs_tracking_warned = False
        if self.enable_skip_var.get() and self.config.detect_interval > 1:
            self.logger.info(
                "跳帧检测加速已启用：每隔 %d 帧执行一次检测+跟踪，中间帧"
                "按轨迹实测速度外推渲染（青色虚线框 extrap 标注）",
                self.config.detect_interval,
            )
        if self.enable_loitering_var.get():
            self.logger.info(
                "徘徊检测已启用：监控区域 %s（像素 %s），停留阈值 %.1f 秒",
                self.config.loitering_zone, self.loitering.zone_px,
                self.config.loitering_duration_sec,
            )
        if self.enable_line_count_var.get():
            self.logger.info(
                "越线计数已启用：警戒线 %s（像素 %s），方向语义 %s",
                self.config.warning_line, self.line_counter.line_px,
                "下->上=进、上->下=出" if self.config.line_enter_direction
                else "上->下=进、下->上=出",
            )
        # A12：切片推理开启时写一条日志说明检测路径变化（含参数快照）；
        # 切片开启期间跟踪让位的提示只记一次，避免逐帧刷屏
        if self.enable_slice_var.get():
            self.logger.info(
                "切片推理已启用：检测路径走 SlicedDetector（%dx%d 重叠切片，"
                "重叠率 %.2f，合并 NMS IOU 阈值 %.2f）",
                self.config.slice_rows, self.config.slice_cols,
                self.config.slice_overlap_ratio, self.config.slice_nms_iou,
            )
            if self.enable_confirm_var.get() and self.enable_tracking_var.get():
                self.logger.info(
                    "提示：切片推理开启期间 ByteTrack 跟踪自动让位"
                    "（model.track 内部为整图推理，不与切片叠加），"
                    "多帧确认走 A5 IOU 降级路径",
                )
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._set_status("巡检运行中…… 视频源：%s" % source_desc)
        self.logger.info("开始巡检，视频源：%s", source_desc)
        # 启动 after() 驱动的播放循环（主线程逐帧处理）
        self._after_id = self.root.after(0, self._play_loop)

    def stop_patrol(self):
        """停止巡检：关闭播放循环并释放视频源。"""
        self.is_running = False
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if self.source is not None:
            self.source.release()
            self.source = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._set_status("巡检已停止")
        self.logger.info("巡检已停止，共处理 %d 帧", self.frame_count)

    # ---------------- 播放循环 ----------------

    def _play_loop(self):
        """逐帧播放主循环：读帧 -> 检测 -> 标注 -> 显示 -> 调度下一帧。"""
        if not self.is_running or self.source is None:
            return

        frame = self.source.read()
        if frame is None:
            # 视频文件播放结束（或摄像头读取失败）：停止并提示
            self._set_status("视频播放结束，共处理 %d 帧" % self.frame_count)
            self.logger.info("视频播放结束，共处理 %d 帧", self.frame_count)
            self._finish_and_exit()
            return

        self.frame_count += 1

        # A12：跟踪模式判定——"多帧确认"与"ByteTrack 跟踪"同时开启时，
        # 检测 + ByteTrack 关联 + 轨迹确认由跟踪器一次完成（model.track
        # 内部已含推理，不再重复调用 detector.detect），目标带持久 ID；
        # 跟踪关闭时走原 A3/A5 路径（detector.detect + 多帧 IOU 确认）。
        # A12 切片推理开启时：检测路径替换为 SlicedDetector 切片推理；
        # 跟踪后端 model.track 内部为整图推理、不与切片叠加，故切片
        # 开启期间跟踪自动让位，确认走 A5 IOU 降级路径
        use_slicing = self.enable_slice_var.get()
        use_tracking = self.enable_confirm_var.get() \
            and self.enable_tracking_var.get() and self.tracker is not None \
            and not use_slicing
        # A13：跳帧检测判定——跳帧开关开启、间隔>1 且跟踪后端可用时，
        # 只在 (帧号-1) % 间隔 == 0 的检测帧执行推理+跟踪，其余帧走
        # 跟踪器外推（不推理）；跟踪未开启时跳帧不生效并一次性提示
        skip_interval = int(self.config.detect_interval) \
            if self.enable_skip_var.get() else 1
        if skip_interval > 1 and not use_tracking:
            if not self._skip_needs_tracking_warned:
                self._skip_needs_tracking_warned = True
                self.logger.info(
                    "跳帧检测加速依赖 ByteTrack 跟踪后端（需同时开启多帧"
                    "确认与目标跟踪、且未开启切片推理）：当前条件不满足，"
                    "跳帧不生效，按逐帧检测处理")
            skip_interval = 1
        if use_tracking:
            if skip_interval > 1 \
                    and (self.frame_count - 1) % skip_interval != 0 \
                    and self.tracker.backend == BACKEND_BYTETRACK:
                # A13 跳帧中间帧：不推理，按轨迹实测速度线性外推目标
                # 位置（records 全部 extrapolated=True，渲染为青色虚线框）
                track_result = self.tracker.extrapolate(
                    frame, self.frame_count)
            else:
                track_result = self.tracker.update(frame, self.frame_count)
            detections = [r["det"] for r in track_result.records]
        else:
            track_result = None
            # YOLO 推理并叠加检测框、类别、置信度；切片开关开启时
            # 检测路径走切片推理（逐片检测 + 坐标映射 + 合并 NMS 去重），
            # 返回结构与 Detector.detect() 一致，下游链路无需感知差异
            if use_slicing:
                detections = self.slicer.detect(frame)
            else:
                detections = self.detector.detect(frame)

        # A4：勾选"启用分区域置信度阈值过滤"时按所在区域应用差异化阈值；
        # 未开启时 zone_result 为 None，检测渲染行为与之前完全一致
        zone_result = None
        if self.enable_zone_filter_var.get():
            fh, fw = frame.shape[:2]
            zone_result = self.zone_filter.filter(detections, fw, fh)
            # 有目标被过滤时写日志（含逐目标原因），无过滤不刷屏
            for rec in zone_result.records:
                if not rec["kept"]:
                    self.logger.info("帧%d 分区过滤：%s %.2f -> %s",
                                     self.frame_count,
                                     rec["det"]["cls"], rec["det"]["conf"],
                                     rec["reason"])

        # A5：勾选"启用多帧确认降噪"时，把（分区过滤后的）检测结果送入
        # 多帧 IOU 确认器；被分区过滤的目标不参确认累计。确认器输出
        # 逐目标状态记录，已确认目标才可进入告警列表。
        # A11：跟踪模式下确认结果已由跟踪器产出（ConfirmResult 契约与
        # A5 一致）；分区过滤开启时把被过滤目标从下游确认结果中剔除，
        # 保持"被过滤目标不进入告警与计数"的行为一致
        confirm_result = None
        if self.enable_confirm_var.get():
            if use_tracking:
                confirm_result = track_result
                if zone_result is not None:
                    confirm_result = self._filter_confirm_result(
                        track_result, zone_result)
            else:
                confirm_input = (zone_result.kept
                                 if zone_result is not None else detections)
                confirm_result = self.frame_confirmer.update(
                    confirm_input, self.frame_count)
            # 新产生的确认事件写日志（确认帧号、连续帧数，供追溯）
            for rec in confirm_result.just_confirmed:
                self.logger.info(
                    "帧%d 目标确认：#%d %s 连续%d帧达标，进入告警范围（后端 %s）",
                    self.frame_count, rec["track_id"], rec["det"]["cls"],
                    rec["consecutive"], confirm_result.backend,
                )
            # A8：把本帧已确认目标的底部中心锚点累积进热力图
            # （多帧确认关闭时无已确认目标，热力不累积）
            self.heatmap.update(confirm_result, frame.shape[1],
                                frame.shape[0])

        # A6：勾选"启用越线方向计数"时，把多帧确认结果送入越线计数器；
        # 已确认目标锚点跨越警戒线时产生穿线事件（进/出双向计数）。
        # 越线计数依赖多帧确认提供的轨迹号，确认开关关闭时跳过并提示
        line_result = None
        if self.enable_line_count_var.get() and self.line_counter is not None:
            if confirm_result is not None:
                line_result = self.line_counter.update(
                    confirm_result, self.frame_count,
                    fps=self.source.fps if self.source is not None else 0.0)
            elif not self._line_needs_confirm_warned:
                self._line_needs_confirm_warned = True
                self.logger.info(
                    "越线计数依赖多帧确认提供的轨迹号：当前多帧确认已关闭，"
                    "越线计数自动跳过；如需计数请同时开启多帧确认开关")

        # A11：勾选"启用徘徊检测告警"时，把确认结果送入徘徊检测器；
        # 已确认目标在监控区域内停留超时触发徘徊告警。徘徊检测同样
        # 依赖确认结果中的轨迹号，确认开关关闭时跳过并提示
        loiter_result = None
        if self.enable_loitering_var.get() and self.loitering is not None:
            if confirm_result is not None:
                loiter_result = self.loitering.update(
                    confirm_result, self.frame_count,
                    fps=self.source.fps if self.source is not None else 0.0)
            elif not self._loiter_needs_confirm_warned:
                self._loiter_needs_confirm_warned = True
                self.logger.info(
                    "徘徊检测依赖多帧确认/跟踪提供的轨迹号：当前多帧确认"
                    "已关闭，徘徊检测自动跳过；如需检测请同时开启多帧确认开关")

        annotated = self._draw_detections(frame, detections, zone_result,
                                          confirm_result)
        # A6：越线计数开启时叠加警戒线、进方向箭头与 IN/OUT 双向计数
        if line_result is not None:
            annotated = self.line_counter.draw_overlay(annotated)
        # A11：徘徊检测开启时叠加监控区域与实时停留时长
        if loiter_result is not None:
            annotated = self.loitering.draw_overlay(annotated)
        self.current_annotated = annotated

        # A7：告警归档开启时，把本帧新产生的告警（目标确认 / 穿线事件 /
        # 徘徊告警）连当前带标注画面一起归档：截图存盘 + CSV 日志 + 列表条目
        if self.enable_archive_var.get():
            self._archive_alerts(annotated, confirm_result, line_result,
                                 loiter_result)

        # 刷新画面、状态栏与告警列表
        self._show_frame(annotated)
        self.frame_var.set("帧号: %d" % self.frame_count)
        self._update_alert_list(self.frame_count, detections, zone_result,
                                confirm_result)
        # A6：本帧新产生的穿线事件写入告警列表
        if line_result is not None:
            self._append_cross_events(line_result)
        # A11：本帧新产生的徘徊告警写入告警列表
        if loiter_result is not None:
            self._append_loiter_events(loiter_result)

        # 自动化验收：播放到指定帧数后自动退出主循环并正常关闭
        if self.args.max_frames is not None \
                and self.frame_count >= self.args.max_frames:
            self.logger.info("已达到最大帧数 %d，自动退出", self.args.max_frames)
            self._finish_and_exit()
            return

        # 按配置的帧间隔调度下一帧
        self._after_id = self.root.after(
            self.config.frame_interval_ms, self._play_loop)

    def _draw_detections(self, frame, detections, zone_result=None,
                         confirm_result=None):
        """在画面上叠加检测框、类别与置信度，返回标注后的 BGR 图像。

        参数：
            frame: 原始 BGR 画面
            detections: Detector 输出的检测列表
            zone_result: 分区过滤结果（ZoneFilterResult）或 None。
                非 None 时被过滤目标改绘为灰色并标注 "filtered"。
            confirm_result: 多帧确认结果（ConfirmResult）或 None。
                非 None 时未确认目标绘为黄色虚线框并标注连续帧数进度，
                已确认目标绘为绿色实线框并标注内部目标号；
                为 None 时（多帧确认未开启）所有检测按原样绿色绘制，
                行为与之前完全一致。
        绘制优先级：分区过滤（灰）> 外推中间帧（青虚线，A13）>
        未确认（黄虚线）> 已确认（绿实线）。
        """
        # A4：勾选"显示区域分界线"时先叠加中央区/边缘区分界线
        if self.show_zone_border_var.get():
            annotated = self.zone_filter.draw_zone_border(frame)
        else:
            annotated = frame.copy()

        # 建立 id(det) -> 判定记录 的映射，便于渲染时查区域与确认状态
        record_map = {}
        if zone_result is not None:
            record_map = {id(r["det"]): r for r in zone_result.records}
        confirm_map = {}
        if confirm_result is not None:
            confirm_map = {id(r["det"]): r for r in confirm_result.records}

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            rec = record_map.get(id(det))
            crec = confirm_map.get(id(det))
            if rec is not None and not rec["kept"]:
                # 被分区阈值过滤的目标：灰色框 + filtered 标注
                color = (128, 128, 128)
                label = "%s %.2f filtered" % (det["cls"], det["conf"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            elif crec is not None and crec.get("extrapolated"):
                # A13 跳帧中间帧的外推目标：青色虚线框 + extrap 标注
                # （非实测观测，与实测检测框明确区分）
                color = (255, 255, 0)
                label = "%s %.2f #%d extrap" % (
                    det["cls"], det["conf"], crec["track_id"])
                self._draw_dashed_rect(annotated, (x1, y1), (x2, y2),
                                       color, 2)
            elif crec is not None and not crec["confirmed"]:
                # 未确认目标：黄色虚线框 + 连续帧数进度（confirm x/N）
                color = (0, 255, 255)
                label = "%s %.2f #%d confirm %d/%d" % (
                    det["cls"], det["conf"], crec["track_id"],
                    crec["consecutive"],
                    self.frame_confirmer.confirm_frames)
                self._draw_dashed_rect(annotated, (x1, y1), (x2, y2),
                                       color, 2)
            else:
                color = (0, 255, 0)
                label = "%s %.2f" % (det["cls"], det["conf"])
                # 分区过滤开启时，在标签上追加区域标识，便于核对判定
                if rec is not None:
                    label += " [%s]" % rec["zone"]
                # 已确认目标追加内部目标号，便于与确认日志对照
                if crec is not None:
                    label += " #%d" % crec["track_id"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            # 检测框 + 框上方的 类别 置信度 文本（带底色条保证可读）
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)),
                          (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # A11：跟踪模式下叠加目标轨迹尾迹——每条轨迹最近若干帧
        # （Config.track_tail_length）的锚点连线 + 末端锚点圆点；
        # 品红色与徘徊监控区域配色一致，区别于检测框绿/黄/灰
        if confirm_result is not None and self.tracker is not None \
                and confirm_result.backend == BACKEND_BYTETRACK:
            for rec in confirm_result.records:
                trail = self.tracker.get_trail(rec["track_id"])
                for i in range(1, len(trail)):
                    cv2.line(annotated, trail[i - 1], trail[i],
                             (255, 0, 255), 2)
                if trail:
                    cv2.circle(annotated, trail[-1], 3, (255, 0, 255), -1)
        return annotated

    @staticmethod
    def _filter_confirm_result(confirm_result, zone_result):
        """跟踪模式下按分区过滤结论裁剪确认结果（A4 + A11 组合行为）。

        跟踪器产出 ConfirmResult 后，分区过滤才把部分目标判为"过滤"；
        为保持与 A5 路径一致的行为（被过滤目标不进入告警与计数），
        这里把被过滤目标对应的确认记录从下游结果中剔除，返回仅含
        保留目标记录的 ConfirmResult（backend 标识保持不变）。
        渲染仍依据原 zone_result 把被过滤目标绘为灰色。
        """
        kept_dets = {id(r["det"]) for r in zone_result.records if r["kept"]}
        records = [r for r in confirm_result.records
                   if id(r["det"]) in kept_dets]
        return ConfirmResult(records, confirm_result.frame_no,
                             backend=confirm_result.backend)

    @staticmethod
    def _draw_dashed_rect(img, pt1, pt2, color, thickness,
                          dash_len=8, gap_len=6):
        """绘制虚线矩形（OpenCV 无原生虚线框，用短线段拼出）。

        参数：
            img: 目标画面（原地绘制）
            pt1 / pt2: 矩形左上、右下角 (x, y)
            color: BGR 颜色；thickness: 线宽
            dash_len / gap_len: 虚线段长与间隔（像素）
        """
        x1, y1 = pt1
        x2, y2 = pt2

        def dashed_line(start, end):
            # 沿 start->end 方向按 dash_len/gap_len 间隔画实线段
            sx, sy = start
            ex, ey = end
            length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
            if length <= 0:
                return
            dx, dy = (ex - sx) / length, (ey - sy) / length
            pos = 0.0
            while pos < length:
                seg_end = min(pos + dash_len, length)
                cv2.line(
                    img,
                    (int(sx + dx * pos), int(sy + dy * pos)),
                    (int(sx + dx * seg_end), int(sy + dy * seg_end)),
                    color, thickness,
                )
                pos += dash_len + gap_len

        dashed_line((x1, y1), (x2, y1))  # 上边
        dashed_line((x2, y1), (x2, y2))  # 右边
        dashed_line((x2, y2), (x1, y2))  # 下边
        dashed_line((x1, y2), (x1, y1))  # 左边

    def _show_frame(self, annotated):
        """把标注后的 BGR 画面缩放适配画布并刷新显示。"""
        canvas_w = max(self.video_canvas.winfo_width(), 320)
        canvas_h = max(self.video_canvas.winfo_height(), 240)
        h, w = annotated.shape[:2]
        # 等比缩放至画布范围内，避免画面变形
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        if scale < 1.0:
            annotated = cv2.resize(annotated,
                                   (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_canvas.delete("all")
        self.video_canvas.create_image(
            canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=photo)
        self.video_canvas.image = photo  # 保持引用，防止被垃圾回收

    def _update_alert_list(self, frame_no, detections, zone_result=None,
                           confirm_result=None):
        """把本帧检测摘要追加到告警列表。

        多帧确认开启时（confirm_result 非 None）：只记录已确认目标
        （未确认目标不进入告警与计数），并标注内部目标号；
        未开启时保持旧行为：逐帧记录全部检测摘要。
        分区过滤开启时（zone_result 非 None），摘要追加区域与结论。
        """
        if not detections:
            return
        record_map = {}
        if zone_result is not None:
            record_map = {id(r["det"]): r for r in zone_result.records}
        confirm_map = {}
        if confirm_result is not None:
            confirm_map = {id(r["det"]): r for r in confirm_result.records}
        for det in detections:
            cx, cy = det["center"]
            rec = record_map.get(id(det))
            crec = confirm_map.get(id(det))
            # 多帧确认开启时，未确认目标不进入告警列表
            if confirm_result is not None and crec is None:
                continue  # 被分区过滤掉的目标不参与确认，也不进告警
            if crec is not None and not crec["confirmed"]:
                continue
            if rec is not None:
                text = "帧%d | %s %.2f @ (%d,%d) | %s %s" % (
                    frame_no, det["cls"], det["conf"], cx, cy,
                    rec["zone"], "保留" if rec["kept"] else "过滤")
            else:
                text = "帧%d | %s %.2f @ (%d,%d)" % (
                    frame_no, det["cls"], det["conf"], cx, cy)
            if crec is not None:
                text += " | 已确认 #%d（确认于帧%d）" % (
                    crec["track_id"], crec["confirm_frame"])
            self.alert_list.insert(tk.END, text)
        # 列表有界增长：超出 300 条时从头删除旧记录，避免长时间运行膨胀
        if self.alert_list.size() > 300:
            self.alert_list.delete(0, self.alert_list.size() - 300)
        self.alert_list.see(tk.END)  # 滚动到最新一条

    def _append_cross_events(self, line_result):
        """把本帧新产生的越线穿线事件追加到告警列表（A6）。

        每条事件一行：帧号、视频内时间、方向（进/出）、类别、轨迹号、
        穿线位置与截至目前的双向累计计数。
        """
        for ev in line_result.events:
            time_text = ("%.2fs" % ev["time_sec"]) \
                if ev["time_sec"] is not None else "时间未知"
            text = "帧%d | %s | 越线%s | %s #%d @ (%d,%d) | 累计 进%d 出%d" % (
                ev["frame_no"], time_text, ev["direction_cn"], ev["cls"],
                ev["track_id"], ev["cross_point"][0], ev["cross_point"][1],
                line_result.count_in, line_result.count_out)
            self.alert_list.insert(tk.END, text)
        if line_result.events:
            # 与检测摘要共用一个列表，同样做有界增长控制
            if self.alert_list.size() > 300:
                self.alert_list.delete(0, self.alert_list.size() - 300)
            self.alert_list.see(tk.END)

    def _append_loiter_events(self, loiter_result):
        """把本帧新产生的徘徊告警事件追加到告警列表（A11）。

        每条事件一行：帧号、视频内时间、事件类型、类别、轨迹号、
        告警位置（锚点）与触发时已停留时长。
        """
        for ev in loiter_result.events:
            time_text = ("%.2fs" % ev["time_sec"]) \
                if ev["time_sec"] is not None else "时间未知"
            text = "帧%d | %s | %s | %s #%d @ (%d,%d) | 已停留 %.1fs" % (
                ev["frame_no"], time_text, ev["event_type"], ev["cls"],
                ev["track_id"], ev["anchor"][0], ev["anchor"][1],
                ev["dwell_sec"])
            self.alert_list.insert(tk.END, text)
        if loiter_result.events:
            # 与检测摘要共用一个列表，同样做有界增长控制
            if self.alert_list.size() > 300:
                self.alert_list.delete(0, self.alert_list.size() - 300)
            self.alert_list.see(tk.END)

    def _archive_alerts(self, annotated, confirm_result, line_result,
                        loiter_result=None):
        """把本帧新产生的告警归档：截图存盘 + CSV 记录 + 告警列表条目（A7）。

        归档时机（只在事件产生的当帧触发一次，避免逐帧重复截图）：
            1. 目标确认：多帧确认结果中"本帧刚确认"且类别命中
               Config.alert_classes 的目标，位置取目标框中心点；
            2. 穿线事件：A6 本帧新产生的越线进/出事件，位置取穿线点；
            3. 徘徊告警：A11 本帧新产生的徘徊事件，位置取目标锚点。
        截图使用传入的带标注画面（检测框、警戒线等已叠加完毕），
        保证归档图与操作员当时看到的画面一致。
        """
        archived = []
        # 1. 目标确认告警（已确认目标首次进入告警范围）
        if confirm_result is not None:
            for rec in confirm_result.just_confirmed:
                det = rec["det"]
                # 只归档告警触发类别（Config.alert_classes）内的目标
                if det["cls"] not in self.config.alert_classes:
                    continue
                record = self.archiver.archive(
                    annotated, EVENT_CONFIRM, det["cls"], det["conf"],
                    position=det["center"])
                archived.append((record, "目标#%d 确认" % rec["track_id"]))
        # 2. 越线穿线事件告警
        if line_result is not None:
            for ev in line_result.events:
                event_type = EVENT_CROSS_IN if ev["direction"] == "in" \
                    else EVENT_CROSS_OUT
                record = self.archiver.archive(
                    annotated, event_type, ev["cls"], ev.get("conf"),
                    position=ev["cross_point"])
                archived.append((record, "越线%s 目标#%d" % (
                    ev["direction_cn"], ev["track_id"])))
        # 3. 徘徊告警（A11）：区域停留超时事件
        if loiter_result is not None:
            for ev in loiter_result.events:
                record = self.archiver.archive(
                    annotated, EVENT_LOITERING, ev["cls"], ev.get("conf"),
                    position=ev["anchor"])
                archived.append((record, "徘徊 目标#%d 停留 %.1fs" % (
                    ev["track_id"], ev["dwell_sec"])))
        # 归档条目同步显示到告警列表（含截图文件名，便于按图追溯）
        for record, desc in archived:
            text = "帧%d | 已归档 | %s %s -> %s" % (
                self.frame_count, record["event_type"], desc,
                os.path.basename(record["screenshot_path"]))
            self.alert_list.insert(tk.END, text)
        if archived:
            if self.alert_list.size() > 300:
                self.alert_list.delete(0, self.alert_list.size() - 300)
            self.alert_list.see(tk.END)

    # ---------------- 退出处理 ----------------

    def _export_heatmap(self):
        """"导出热力图"按钮回调：把当前累积热力图导出到归档目录（A8）。

        导出路径 = Config.archive_dir / Config.heatmap_export_name
        （默认 archive/heatmap.png）；有当前画面时以当前带标注画面为
        底帧叠加，无画面时黑底导出。导出结果同步到状态栏与告警列表；
        累积为空时导出失败并提示先运行巡检。
        """
        export_path = os.path.join(
            self.config.abs_path(self.config.archive_dir),
            self.config.heatmap_export_name)
        try:
            record = self.heatmap.export_png(
                export_path, base_frame=self.current_annotated)
        except ValueError as exc:
            # 空累积防护：状态栏 + 告警列表 + 日志三处提示
            self._set_status("热力图导出失败：%s" % exc)
            self.alert_list.insert(tk.END, "热力图导出失败：%s" % exc)
            self.alert_list.see(tk.END)
            self.logger.info("热力图导出失败：%s", exc)
            return
        text = "热力图已导出：%s（累积 %d 个轨迹点）" % (
            record["path"], record["point_count"])
        self._set_status(text)
        self.alert_list.insert(tk.END, "帧%d | %s" % (self.frame_count, text))
        if self.alert_list.size() > 300:
            self.alert_list.delete(0, self.alert_list.size() - 300)
        self.alert_list.see(tk.END)
        self.logger.info(text)

    def _set_status(self, text):
        """更新底部状态栏文本。"""
        self.status_var.set(text)

    def _finish_and_exit(self):
        """结束流程：按需保存带标注画面，释放资源并销毁窗口（退出码 0）。"""
        self.is_running = False
        # 自动化验收：退出前把当前带标注画面保存为图片文件
        if self.args.dump_frame and self.current_annotated is not None:
            dump_path = os.path.abspath(self.args.dump_frame)
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            cv2.imwrite(dump_path, self.current_annotated)
            self.logger.info("带标注画面已保存：%s", dump_path)
            print("带标注画面已保存：%s" % dump_path)
        if self.source is not None:
            self.source.release()
            self.source = None
        # 关闭告警归档器（CSV 句柄），保证日志文件句柄及时释放
        self.archiver.close()
        self.root.destroy()

    def on_closing(self):
        """窗口关闭事件：停止播放循环、释放视频源后销毁窗口。"""
        self.is_running = False
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if self.source is not None:
            self.source.release()
            self.source = None
        # 关闭告警归档器（CSV 句柄）
        self.archiver.close()
        self.logger.info("用户关闭窗口，系统退出，共处理 %d 帧", self.frame_count)
        self.root.destroy()


def main():
    """主入口：解析命令行参数，创建主窗口并进入 tkinter 主循环。"""
    args = parse_args()
    root = tk.Tk()
    EagleEyeApp(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
