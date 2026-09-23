# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

视频源抽象模块。

功能意图：
    将"视频文件"与"USB/板载摄像头"两类输入源抽象为统一接口，
    上层模块（GUI、演示脚本、评估脚本）无需关心画面来自文件
    还是实时摄像头，统一通过 read() 取帧、通过 fps/width/height
    读取元信息、通过 release() 释放资源。
"""

import cv2

from src.config import Config
from src.logger import get_logger


class BaseVideoSource:
    """视频源基类，定义统一接口。

    子类必须实现 open()；对外统一接口为：
        read()    —— 读取下一帧，成功返回 BGR 图像，结束/失败返回 None
        fps       —— 帧率（摄像头取不到时为 0.0）
        width     —— 画面宽度（像素）
        height    —— 画面高度（像素）
        is_opened —— 源当前是否处于打开状态
        release() —— 释放底层视频资源
    """

    def __init__(self, config=None):
        self.config = config if config is not None else Config()
        self.logger = get_logger(__name__, self.config)
        self.cap = None

    def open(self):
        """打开视频源，子类实现；打开失败应抛出异常。"""
        raise NotImplementedError

    def read(self):
        """读取下一帧。

        返回：
            成功返回 OpenCV BGR 图像（numpy.ndarray）；
            视频播放结束或读取失败时返回 None。
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    @property
    def fps(self):
        """视频帧率；摄像头或取不到时返回 0.0。"""
        if self.cap is None:
            return 0.0
        return float(self.cap.get(cv2.CAP_PROP_FPS))

    @property
    def width(self):
        """画面宽度（像素）。"""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        """画面高度（像素）。"""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def is_opened(self):
        """视频源是否处于可读取状态。"""
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        """释放视频源占用的底层资源，可重复调用。"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.logger.info("视频源已释放")


class FileVideoSource(BaseVideoSource):
    """视频文件源：按文件路径打开本地 mp4 等视频文件。"""

    def __init__(self, path, config=None):
        """参数：path 为视频文件路径。"""
        super().__init__(config)
        self.path = path
        self.open()

    def open(self):
        """打开视频文件，文件不存在或无法解码时抛出 IOError。"""
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise IOError("无法打开视频文件：%s" % self.path)
        self.logger.info(
            "已打开视频文件：%s（%dx%d，%.2f fps）",
            self.path, self.width, self.height, self.fps,
        )


class CameraVideoSource(BaseVideoSource):
    """摄像头源：按设备编号打开摄像头（0 通常为默认摄像头）。"""

    def __init__(self, device_id=0, config=None):
        """参数：device_id 为摄像头设备编号（整数）。"""
        super().__init__(config)
        self.device_id = int(device_id)
        self.open()

    def open(self):
        """打开摄像头设备，设备不存在或被占用时抛出 IOError。"""
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise IOError("无法打开摄像头设备：%d" % self.device_id)
        self.logger.info(
            "已打开摄像头：设备号 %d（%dx%d）",
            self.device_id, self.width, self.height,
        )


def create_video_source(source, config=None):
    """视频源工厂函数。

    根据传入的源描述自动选择具体实现：
        纯数字字符串或整数 —— 按摄像头设备号处理（如 "0"、1）；
        其余               —— 按视频文件路径处理。

    参数：
        source: 视频文件路径（字符串）或摄像头设备号（整数/数字字符串）。
        config: Config 实例；为 None 时内部新建默认配置。

    返回：
        FileVideoSource 或 CameraVideoSource 实例。
    """
    text = str(source).strip()
    if text.isdigit():
        return CameraVideoSource(int(text), config=config)
    return FileVideoSource(text, config=config)
