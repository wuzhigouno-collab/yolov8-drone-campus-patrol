# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

集中配置模块。

功能意图：
    将系统运行所需的全部可调参数集中在一个 Config 类中，
    供检测引擎、分区阈值、多帧确认、越线计数、告警归档、
    热力图等各功能模块统一读取，避免参数散落各处。
    所有路径类配置均为相对项目根目录的相对路径，
    通过 project_root() 可换算为绝对路径。
"""

import os


class Config:
    """系统集中配置类。

    属性按功能分组：
        1. 基础运行参数（权重路径、视频源、检测类别等）
        2. 分区域差异化阈值参数（A4）
        3. 多帧 IOU 确认参数（A5）
        4. 越线方向计数参数（A6）
        5. 告警归档与日志参数（A7）
        6. 热力图参数（A8）
        7. ByteTrack 跟踪与徘徊检测参数（A11）
        8. 切片推理参数（A12）
        9. 日志模块参数
        10. 推理加速参数（A13：FP16 半精度、跳帧检测）
        11. VisDrone 类别映射参数（阶段 C：推理侧 10->5 类映射）
    """

    def __init__(self):
        # ---------- 1. 基础运行参数 ----------
        # YOLO 模型权重路径（相对项目根目录；开发联调阶段使用
        # ultralytics 自动下载的 yolov8n.pt，统一归置在 weights/ 目录下；
        # A9 自训权重产出后同样放入 weights/ 并修改本项指向）
        self.weights_path = "weights/yolov8n.pt"
        # 默认视频源：可为视频文件路径或摄像头编号（整数字符串）
        self.video_source = "0"
        # 推理输入图像尺寸（YOLO imgsz 参数）
        self.imgsz = 640
        # 需要关注的目标类别（COCO 类别名：人、汽车、自行车）
        self.target_classes = ["person", "car", "bicycle"]
        # GUI 播放帧间隔（毫秒），控制视频播放速度
        self.frame_interval_ms = 30

        # ---------- 2. 分区域差异化阈值参数（A4） ----------
        # 中央区域宽度占画面宽度的比例（0~1，居中对称）
        self.center_zone_ratio_w = 0.5
        # 中央区域高度占画面高度的比例（0~1，居中对称）
        self.center_zone_ratio_h = 0.5
        # 中央区域置信度阈值（无人机俯拍远处小目标较多，阈值适当放低）
        self.center_conf_threshold = 0.40
        # 边缘区域置信度阈值（边缘画面畸变小、目标较大，阈值略高以抑制误检）
        self.edge_conf_threshold = 0.50
        # GUI 是否默认显示区域边框（A4 要求提供显示开关）
        self.show_zone_border = True

        # ---------- 3. 多帧 IOU 确认参数（A5） ----------
        # 确认帧数 N：目标需连续 N 帧稳定出现才标记为"已确认"
        self.confirm_frames = 3
        # 相邻帧目标匹配的 IOU 阈值，低于该值视为不同目标
        self.iou_threshold = 0.30

        # ---------- 4. 越线方向计数参数（A6） ----------
        # 警戒线两端点坐标（归一化到 0~1 的画面比例坐标：(x1, y1, x2, y2)）
        # 默认横贯画面中部的一条水平线
        self.warning_line = (0.0, 0.5, 1.0, 0.5)
        # 越线方向定义：目标锚点（检测框底部中心，与 A4 区域判定一致）
        # 由下向上穿线记为"进"时的判定开关
        # True 表示 下->上 为"进"、上->下 为"出"；False 反之
        self.line_enter_direction = True
        # 越线计数总开关（GUI"启用越线计数"复选框的默认值）：
        # 开启时画面叠加警戒线与双向计数，穿线事件写入告警列表；
        # 关闭时画面渲染与告警行为回到 A5 状态
        self.line_count_enabled = True
        # 穿线判定死区（像素）：锚点距警戒线的距离小于该值时视为"压线"，
        # 所在侧维持上一次判定不变，防止目标沿线抖动造成同一目标反复计数
        self.line_deadzone_px = 3
        # 轨迹所在侧记忆的最大保留帧数：目标消失超过该帧数后遗忘，
        # 防止长时间运行时侧记忆无界增长
        self.line_track_max_age = 30

        # ---------- 5. 告警归档参数（A7） ----------
        # 运行期告警归档输出目录（截图与告警日志均落在此目录下）
        self.archive_dir = "archive"
        # 告警日志 CSV 文件名（位于归档目录内）
        self.alert_log_name = "alert_log.csv"
        # 告警触发类别集合：命中这些类别的已确认目标才触发告警
        self.alert_classes = ["person", "car", "bicycle"]

        # ---------- 6. 热力图参数（A8） ----------
        # 热力图高斯累积核半径（像素），控制单个轨迹点的影响范围
        self.heatmap_kernel_radius = 25
        # 单个轨迹点累积强度（高斯核峰值）；多点叠加达到该值的若干倍后
        # 在渲染归一化时饱和为最热色，值越大需要越密集的点才显热
        self.heatmap_intensity = 1.0
        # 热力图叠加到底帧时的混合透明度（0~1）：仅对热度>0 的像素生效，
        # 无热区城保持底帧原样；黑底导出时该参数不生效
        self.heatmap_overlay_alpha = 0.6
        # 热力图导出文件名（位于归档目录内）
        self.heatmap_export_name = "heatmap.png"

        # ---------- 7. ByteTrack 跟踪与徘徊检测参数（A11） ----------
        # ByteTrack 跟踪开关（GUI"启用 ByteTrack 目标跟踪"复选框默认值）：
        # 开启时检测/确认走 ultralytics model.track()（跟踪后端），目标带
        # 持久 ID 并叠加轨迹尾迹；关闭时回到 A5 多帧 IOU 确认降级路径
        self.tracking_enabled = True
        # ultralytics 跟踪器配置名（内置 bytetrack/botsort，交付主线用 bytetrack）
        self.tracker_name = "bytetrack"
        # 轨迹尾迹长度（帧）：GUI 叠加显示最近若干帧的锚点轨迹连线
        self.track_tail_length = 30
        # 跟踪轨迹状态遗忘帧数：轨迹超过该帧数未再出现则清除其确认状态，
        # 防止长时间运行时状态无界增长（语义同 A6 line_track_max_age）
        self.track_max_age = 30
        # 验收用 ID 一致性阈值：同一物理目标的跟踪 ID 切换次数上限
        # （tests/test_tracker.py 视频验收断言使用）
        self.track_max_id_switch = 3
        # 徘徊检测开关（GUI"启用徘徊检测告警"复选框默认值）：
        # 开启时已确认目标在徘徊监控区域内停留超时触发徘徊告警
        self.loitering_enabled = True
        # 徘徊停留阈值（秒）：同一跟踪 ID 锚点连续停留在区域内超过该时长
        # 触发一次徘徊告警（离开后重新进入重新计时）
        self.loitering_duration_sec = 5.0
        # 徘徊监控区域（归一化 0~1 画面比例坐标 xyxy，默认画面中央一半区域）
        self.loitering_zone = (0.25, 0.25, 0.75, 0.75)

        # ---------- 8. 切片推理参数（A12） ----------
        # 切片推理开关（GUI"启用切片推理"复选框默认值，默认关闭保持整图
        # 直检原行为）：开启时检测路径走 src.slicer.SlicedDetector——
        # 帧切分为重叠切片逐片检测、坐标映射回原图、合并 NMS 去重，
        # 提升高空俯视小目标召回（切片相当于把局部画面放大推理）
        self.slice_enabled = False
        # 切片行列数（默认 2x2 共 4 片；1x1 时退化为整图直检）
        self.slice_rows = 2
        self.slice_cols = 2
        # 相邻切片重叠率（0~1，重叠宽度占切片宽/高的比例）：
        # 保证跨切片边界目标至少在某个切片中完整出现、可被稳定检出，
        # 重复检出部分由合并 NMS 去重
        self.slice_overlap_ratio = 0.2
        # 切片合并 NMS 的 IOU 阈值：同一目标在相邻切片重叠区被重复
        # 检出时，同类框 IOU 达到该阈值即去重，保留高置信度框
        self.slice_nms_iou = 0.5

        # ---------- 9. 日志模块参数 ----------
        # 日志文件输出目录（src/logger.py 使用，可配置）
        self.log_dir = "logs"
        # 日志级别：DEBUG / INFO / WARNING / ERROR
        self.log_level = "INFO"
        # 日志文件名前缀（实际文件名会追加日期）
        self.log_file_prefix = "eagle_eye"

        # ---------- 10. 推理加速参数（A13） ----------
        # FP16 半精度推理开关（GUI 不提供开关，走配置）：开启且环境具备
        # CUDA GPU 时，YOLO 检测/跟踪推理以半精度（half=True）执行，
        # 降低显存占用并提升推理速度；无可用 CUDA 环境时由
        # src.detector.resolve_fp16 自动降级为 FP32 并记日志说明，
        # 配置值本身不被修改
        self.fp16 = True
        # 跳帧检测间隔（默认 1 = 每帧都检测，不跳帧）：N>1 时每隔 N 帧
        # 执行一次"检测+ByteTrack 关联"（模型推理只发生在检测帧），
        # 中间帧不推理，由跟踪器按轨迹最近一次实测速度线性外推目标
        # 位置进行渲染（src.tracker.ByteTrackTracker.extrapolate），
        # 以少量精度为代价换取整体处理帧率提升。跳帧依赖跟踪后端，
        # 跟踪关闭或已降级时本项不生效（GUI"启用跳帧检测加速"开关
        # 的默认值由本项是否大于 1 决定）
        self.detect_interval = 1

        # ---------- 11. VisDrone 类别映射参数（阶段 C） ----------
        # VisDrone 10->5 类别映射开关（GUI 不提供开关，走配置）：
        # 论文实验阶段训练/评估使用 VisDrone 原始 10 类，部署到巡检
        # 系统时在推理输出后把细粒度类别归并为 5 类业务类别
        # （pedestrian+people->person、bicycle->bicycle、motor->motorcycle、
        # car+van->car、bus->bus；truck/tricycle/awning-tricycle 丢弃），
        # 映射只发生在 src/ 推理侧（由 src.class_map.build_class_mapper
        # 构造映射函数，src.postprocess 统一应用），不影响训练标签。
        # 仅当模型实际类别名（model.names）全部属于 VisDrone 10 类集合
        # 时映射才真正生效；当前开发联调使用 COCO 权重（person/car 等
        # 80 类名），不满足生效条件，映射自动为恒等映射，系统行为与
        # 未开启时完全一致
        self.class_map_enabled = True

    @staticmethod
    def project_root():
        """返回项目根目录的绝对路径（本文件位于 src/ 下，取上一级）。"""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def abs_path(self, rel_path):
        """将相对项目根目录的路径换算为绝对路径；已是绝对路径则原样返回。"""
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(self.project_root(), rel_path)

    def to_dict(self):
        """将当前全部实例配置项导出为字典，便于打印检查与序列化。"""
        return dict(self.__dict__)
