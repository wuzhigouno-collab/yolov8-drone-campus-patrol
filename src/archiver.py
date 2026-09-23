# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

告警截图归档与日志模块（A7）。

功能意图：
    告警触发瞬间的现场画面是事后追溯的核心证据。本模块在告警产生时
    自动把"带标注画面"（检测框、警戒线等叠加完成后的帧）截取保存到
    归档目录，同时向告警日志 CSV 追加一条结构化记录，形成
    "一条日志 <-> 一张现场截图"的完整证据链。

    归档文件命名含墙上时钟时间戳（精确到毫秒）与全局递增事件序号，
    同一毫秒内多条告警也不会互相覆盖；告警日志 CSV 字段固定为：
    时间戳、事件类型、类别、置信度、位置、截图路径。

    模块内部对截图写盘与 CSV 追加使用同一把互斥锁，多线程并发触发
    告警时事件序号、文件命名与日志行均不会交错错乱；归档目录与 CSV
    表头在首次使用时自动创建，GUI（src/app.py）与验收测试
    （tests/test_archive.py）通过同一接口复用本模块。

事件类型约定：
    EVENT_CONFIRM   —— 新目标确认告警（已确认目标首次进入告警范围）
    EVENT_CROSS_IN  —— 越线"进"方向穿线事件（A6）
    EVENT_CROSS_OUT —— 越线"出"方向穿线事件（A6）
    调用方也可传入自定义事件类型字符串，日志原样记录。
"""

import csv
import os
import threading
from datetime import datetime

import cv2

from src.config import Config
from src.logger import get_logger

# 事件类型常量：新目标确认 / 越线进 / 越线出
EVENT_CONFIRM = "目标确认"
EVENT_CROSS_IN = "越线进"
EVENT_CROSS_OUT = "越线出"

# 事件类型 -> 文件名用英文短标识（文件名保持 ASCII，规避个别环境
# 对非 ASCII 文件名的兼容问题；事件类型中文全称只写入 CSV）
_EVENT_SLUG = {
    EVENT_CONFIRM: "confirm",
    EVENT_CROSS_IN: "cross_in",
    EVENT_CROSS_OUT: "cross_out",
}

# 告警日志 CSV 固定表头（验收字段：时间戳、事件类型、类别、置信度、
# 位置、截图路径）
CSV_HEADER = ["时间戳", "事件类型", "类别", "置信度", "位置", "截图路径"]


class AlertArchiver:
    """告警归档器：告警截图落盘 + 告警日志 CSV 追加。

    职责：
        1. 初始化时自动创建归档目录与告警日志 CSV（含表头）；
        2. archive() 接收一帧带标注画面与告警元信息，原子地完成
           "保存截图 -> 追加 CSV 行"，全程持锁保证线程安全；
        3. 维护全局递增事件序号，配合毫秒级时间戳生成不重复的文件名；
        4. 内存中保留全部归档记录（records 属性），供 GUI 展示与
           测试断言直接核对。

    本类不修改 Config，归档目录与日志文件名在构造时从 Config 读取
    快照；构造后运行期改配置不影响本实例。
    """

    def __init__(self, config=None, archive_dir=None):
        """初始化归档器。

        参数：
            config: Config 实例；为 None 时内部新建默认配置。
            archive_dir: 归档输出目录；为 None 时使用
                Config.archive_dir（相对路径按项目根目录换算）。
                测试可传入独立目录，避免污染运行期 archive/ 输出。
        """
        self.config = config if config is not None else Config()
        self.logger = get_logger(__name__, self.config)
        if archive_dir is None:
            archive_dir = self.config.abs_path(self.config.archive_dir)
        self.archive_dir = os.path.abspath(archive_dir)
        # 归档目录自动创建（含多级父目录），已存在时静默跳过
        os.makedirs(self.archive_dir, exist_ok=True)
        # 告警日志 CSV 路径（固定文件名，追加写入）
        self.csv_path = os.path.join(
            self.archive_dir, self.config.alert_log_name)
        # 截图写盘 + CSV 追加 + 事件序号自增的互斥锁（线程安全）
        self._lock = threading.Lock()
        # 全局递增事件序号：保证同一毫秒内多条告警文件名不冲突
        self._seq = 0
        # 全部归档记录（与 CSV 行一一对应），供 GUI 与测试复用
        self.records = []
        # 追加模式打开 CSV；文件为空（新建）时先写表头。
        # 编码用 utf-8-sig：Excel 打开含中文表头的 CSV 时不乱码，
        # 程序读取侧用同一编码即可自动剥掉 BOM。
        need_header = (not os.path.exists(self.csv_path)
                       or os.path.getsize(self.csv_path) == 0)
        self._csv_file = open(self.csv_path, "a", newline="",
                              encoding="utf-8-sig")
        self._writer = csv.writer(self._csv_file)
        if need_header:
            self._writer.writerow(CSV_HEADER)
            self._csv_file.flush()
        self.logger.info(
            "告警归档器就绪：归档目录 %s，告警日志 %s",
            self.archive_dir, self.csv_path,
        )

    # ---------------- 主接口 ----------------

    def archive(self, frame, event_type, cls="", conf=None, position=None):
        """归档一条告警：保存带标注截图并追加 CSV 日志行。

        参数：
            frame: 告警触发时刻的带标注画面（BGR numpy 数组，
                检测框/警戒线等应已叠加完毕）；
            event_type: 事件类型字符串（建议用 EVENT_* 常量）；
            cls: 告警目标类别名（如 person/car）；
            conf: 告警目标置信度（0~1），无则传 None；
            position: 告警位置像素坐标 (x, y)——确认告警取目标中心点，
                穿线事件取穿线点；无则传 None。

        返回：
            归档记录字典：{seq, timestamp, event_type, cls, conf,
            position, screenshot_path}，与 CSV 行一一对应。

        异常：
            frame 为 None 或截图编码失败时抛出 ValueError。
        """
        if frame is None:
            raise ValueError("告警归档需要带标注画面，frame 不能为 None")

        with self._lock:
            self._seq += 1
            seq = self._seq
            now = datetime.now()
            # 墙上时钟时间戳精确到毫秒：既是 CSV 时间字段，也进文件名
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S.") + \
                "%03d" % (now.microsecond // 1000)
            slug = _EVENT_SLUG.get(event_type, "alert")
            # 文件名 = 时间戳(毫秒) + 事件序号 + 事件短标识，天然防覆盖
            filename = "alert_%s_%04d_%s.png" % (
                now.strftime("%Y%m%d_%H%M%S")
                + "_%03d" % (now.microsecond // 1000), seq, slug)
            screenshot_path = os.path.join(self.archive_dir, filename)

            # 截图写盘：用 imencode + tofile 而非 cv2.imwrite，
            # 规避 Windows 下 imwrite 对非 ASCII 路径静默失败的问题
            ok, buf = cv2.imencode(".png", frame)
            if not ok:
                raise ValueError("告警截图编码失败（画面数据异常）")
            buf.tofile(screenshot_path)

            # 位置格式化为 "(x, y)" 文本，无位置信息留空
            pos_text = ""
            if position is not None:
                pos_text = "(%d, %d)" % (int(position[0]), int(position[1]))
            conf_text = ("%.2f" % conf) if conf is not None else ""
            row = [timestamp, event_type, cls, conf_text, pos_text,
                   screenshot_path]
            self._writer.writerow(row)
            # 每行立即落盘：程序异常退出时日志不丢行，外部可实时读取
            self._csv_file.flush()

            record = {
                "seq": seq,
                "timestamp": timestamp,
                "event_type": event_type,
                "cls": cls,
                "conf": conf,
                "position": position,
                "screenshot_path": screenshot_path,
            }
            self.records.append(record)

        self.logger.info(
            "告警归档 #%d：%s %s %s -> %s",
            seq, event_type, cls, pos_text or "(无位置)", filename,
        )
        return record

    # ---------------- 资源管理 ----------------

    def close(self):
        """关闭告警日志文件句柄（截图随写随关，无需额外处理）。"""
        with self._lock:
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
