# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

统一日志模块。

功能意图：
    为系统各功能模块提供统一的日志获取入口 get_logger()。
    每个 logger 同时输出到控制台与日志文件，日志文件落在
    Config 配置的日志目录下（默认 logs/），按日期命名，
    便于运行期问题排查与告警事件追溯。
"""

import logging
import os
import time

from src.config import Config


def get_logger(name="eagle_eye", config=None):
    """获取一个同时输出到控制台与文件的 logger。

    参数：
        name:   logger 名称，通常传入调用方模块名（__name__）。
        config: Config 实例；为 None 时内部新建默认配置。

    返回：
        logging.Logger 实例。重复调用同名 logger 不会重复添加处理器。
    """
    if config is None:
        config = Config()

    logger = logging.getLogger(name)
    # 已初始化过的 logger 直接返回，避免重复挂处理器导致日志重复输出
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, str(config.log_level).upper(), logging.INFO))

    # 统一日志格式：时间 | 级别 | 模块名 | 内容
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出：日志目录由配置项指定，按日期命名日志文件
    log_dir = config.abs_path(config.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        "%s_%s.log" % (config.log_file_prefix, time.strftime("%Y%m%d")),
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
