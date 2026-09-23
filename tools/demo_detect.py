# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

检测引擎演示脚本。

功能意图：
    不启动 GUI，直接对指定视频逐帧执行 YOLO 检测，将每帧的
    检测结果列表写入 JSON 文件。用于检测引擎核心（A2）的联调
    验证，也可作为后续评估脚本的最小调用样例。

用法示例：
    python tools/demo_detect.py --video 测试视频路径 --out tests/out_frames.json
    python tools/demo_detect.py --video 0 --out tests/out_camera.json --max-frames 50
"""

import argparse
import json
import os
import sys

# 脚本位于 tools/ 下，直接运行时 sys.path 不含项目根目录，
# 这里手动补上，保证 src 包可被正常导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.detector import Detector
from src.logger import get_logger
from src.video_source import create_video_source


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="鹰眼巡校检测引擎演示：逐帧推理并导出 JSON 结果"
    )
    parser.add_argument(
        "--video", required=True,
        help="视频源：视频文件路径或摄像头设备号（如 0）",
    )
    parser.add_argument(
        "--out", required=True,
        help="检测结果 JSON 输出路径（如 tests/out_frames.json）",
    )
    parser.add_argument(
        "--max-frames", type=int, default=100,
        help="最多处理的帧数，默认 100",
    )
    return parser.parse_args()


def main():
    """主流程：打开视频源 -> 逐帧推理 -> 汇总写出 JSON。"""
    args = parse_args()
    config = Config()
    logger = get_logger(__name__, config)

    detector = Detector(config)
    source = create_video_source(args.video, config)

    # 每帧一条记录：frame 为帧序号（从 0 起），detections 为该帧检测结果列表
    frames_result = []
    frame_count = 0
    try:
        while frame_count < args.max_frames:
            frame = source.read()
            if frame is None:
                # 视频播完或读取失败，正常结束循环
                break
            detections = detector.detect(frame)
            frames_result.append({
                "frame": frame_count,
                "detections": detections,
            })
            frame_count += 1
    finally:
        source.release()

    # 输出目录不存在时自动创建（如 tests/）
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(frames_result, f, ensure_ascii=False, indent=2)

    total_dets = sum(len(item["detections"]) for item in frames_result)
    logger.info(
        "演示完成：共处理 %d 帧，检出目标 %d 个，结果已写入 %s",
        frame_count, total_dets, args.out,
    )
    print("处理帧数：%d，检出目标总数：%d，输出：%s"
          % (frame_count, total_dets, args.out))


if __name__ == "__main__":
    main()
