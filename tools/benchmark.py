# -*- coding: utf-8 -*-
"""鹰眼巡校——A13 推理加速基准测试脚本。

功能意图：
    对同一视频分别跑"优化前"与"优化后"两组巡检管线（检测 +
    ByteTrack 跟踪 + 多帧确认，与 src/app.py 跟踪路径同源），
    输出两组实测处理帧率（FPS）与检出数对比，量化 A13 推理加速
    优化项（FP16 半精度、跳帧检测+轨迹外推）的真实收益：

        优化前（基线组）：detect_interval=1 逐帧检测、FP32 全精度；
        优化后（加速组）：detect_interval=N（--interval，默认 3），
                         FP16 按环境可用性自动启用（无 CUDA 自动
                         降级 FP32，与运行时行为一致）。

    计时口径：逐帧"读帧 + 检测/跟踪/外推"全链路墙钟时间（模型加载
    与首帧跟踪器初始化预热不计入）；检出数口径：全部处理帧的目标
    记录总数与帧均检出（基线组全部为实测；加速组检测帧为实测、
    中间帧为轨迹外推），另单列"仅检测帧实测检出"便于核对。

达标判定（plan.md A13 验收标准）：
    综合 FPS 提升 ≥ 30%（无 CUDA 环境只看跳帧单项提升 ≥ 50%），
    且帧均检出数下降 ≤ 10%。判定结论随结果一并输出。

用法示例：
    python tools/benchmark.py --video <测试视频路径>
    python tools/benchmark.py --video <路径> --max-frames 300 \\
        --interval 3 --out tests/a13_benchmark.json
退出码：0 = 运行完成且达标；2 = 运行完成但未达标；1 = 运行失败。
"""

import argparse
import json
import os
import sys
import time

# 本文件位于 tools/ 下，直接运行时 sys.path 不含项目根目录，手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.config import Config
from src.detector import resolve_fp16
from src.logger import get_logger
from src.tracker import ByteTrackTracker

logger = get_logger("benchmark")


def parse_args():
    """解析命令行参数。

    --video       测试视频路径（必填）
    --max-frames  最多处理帧数（默认 300；限定帧数保证可重复对比）
    --interval    优化组跳帧间隔 N（默认 3；每隔 N 帧检测一次）
    --out         可选：把两组实测结果与对比结论落盘为 JSON
    """
    parser = argparse.ArgumentParser(
        description="鹰眼巡校 A13 推理加速基准测试（优化前后 FPS/检出数对比）")
    parser.add_argument("--video", required=True, help="测试视频文件路径")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="最多处理帧数（默认 300，保证可重复对比）")
    parser.add_argument("--interval", type=int, default=3,
                        help="优化组跳帧检测间隔 N（默认 3，1 表示不跳帧）")
    parser.add_argument("--out", default=None,
                        help="可选：实测结果 JSON 落盘路径")
    return parser.parse_args()


def build_config(fp16, detect_interval):
    """按基准分组参数构造配置（其余配置项保持系统默认值）。"""
    config = Config()
    config.fp16 = fp16
    config.detect_interval = detect_interval
    return config


def run_group(name, video, config, model, max_frames):
    """跑一组基准：逐帧 读帧 -> 检测/跟踪（或跳帧外推），统计 FPS 与检出数。

    参数：
        name:        组名（"优化前"/"优化后"，仅用于结果标识）；
        video:       视频文件路径；
        config:      本组配置（fp16 与 detect_interval 已按需覆盖）；
        model:       已加载的 YOLO 模型（两组共享，加载耗时不计入）；
        max_frames:  最多处理帧数。

    返回：
        结果字典：组名、实际生效配置、帧数、耗时、FPS、检出数指标、
        确认事件数。
    """
    tracker = ByteTrackTracker(config, model=model)
    # 清理共享模型的 predictor，避免上一组的 ByteTrack 轨迹接续到本组
    tracker.reset()

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError("无法打开视频：%s" % video)
    src_fps = cap.get(cv2.CAP_PROP_FPS)

    # 预热：首帧跑一遍真实跟踪（跟踪器首次初始化的固定开销两组相同，
    # 不计入计时），随后重置状态并把读取位置倒回开头，保证两组处理的
    # 帧序列完全一致
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise IOError("视频无帧可读：%s" % video)
    tracker.update(frame, 1)
    tracker.reset()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    interval = max(int(config.detect_interval), 1)
    frames = 0
    total_records = 0   # 全部帧目标记录总数（实测帧 + 外推帧）
    real_frames = 0     # 真实检测帧数（推理只发生在这些帧）
    real_records = 0    # 仅检测帧的实测检出总数
    confirmed_events = 0
    t0 = time.perf_counter()
    while frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        if interval > 1 and (frames - 1) % interval != 0:
            # 跳帧中间帧：不推理，按轨迹实测速度线性外推目标位置
            result = tracker.extrapolate(frame, frames)
        else:
            result = tracker.update(frame, frames)
            real_frames += 1
            real_records += len(result.records)
        total_records += len(result.records)
        confirmed_events += len(result.just_confirmed)
    elapsed = time.perf_counter() - t0
    cap.release()

    fps = frames / elapsed if elapsed > 0 else 0.0
    result = {
        "name": name,
        "fp16_config": config.fp16,
        "fp16_effective": tracker.use_fp16,
        "detect_interval": interval,
        "frames": frames,
        "elapsed_sec": round(elapsed, 3),
        "fps": round(fps, 2),
        "total_records": total_records,
        "avg_records_per_frame": round(total_records / frames, 4)
        if frames else 0.0,
        "real_detect_frames": real_frames,
        "real_detect_records": real_records,
        "confirmed_events": confirmed_events,
        "backend": tracker.backend,
        "video_fps": round(src_fps, 2),
    }
    logger.info(
        "%s 组完成：%d 帧耗时 %.2fs，FPS=%.2f，帧均检出 %.3f"
        "（fp16 生效=%s，跳帧间隔=%d，检测帧 %d 实测检出 %d，确认事件 %d）",
        name, frames, elapsed, fps, result["avg_records_per_frame"],
        tracker.use_fp16, interval, real_frames, real_records,
        confirmed_events)
    return result


def print_env():
    """探测并打印运行环境（torch/CUDA/GPU 型号），写入结果与日志。"""
    import torch
    env = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else None,
    }
    print("运行环境：torch %s | CUDA 可用：%s | GPU：%s" % (
        env["torch_version"], env["cuda_available"],
        env["gpu_name"] or "无"))
    return env


def print_report(base, opt, env):
    """打印优化前后对比表与达标判定，返回 (是否达标, 对比结论字典)。"""
    speedup = (opt["fps"] - base["fps"]) / base["fps"] * 100 \
        if base["fps"] > 0 else 0.0
    drop = (base["avg_records_per_frame"] - opt["avg_records_per_frame"]) \
        / base["avg_records_per_frame"] * 100 \
        if base["avg_records_per_frame"] > 0 else 0.0

    # 达标线：有 CUDA 时看综合提升 ≥30%；无 CUDA 时只看跳帧单项 ≥50%
    # （FP16 在无 GPU 环境本就降级不生效，综合与跳帧单项为同一数字）
    fps_threshold = 30.0 if env["cuda_available"] else 50.0
    fps_pass = speedup >= fps_threshold
    drop_pass = drop <= 10.0

    print()
    print("=" * 78)
    print("A13 推理加速基准对比（同一视频、同一帧数上限，计时含读帧全链路）")
    print("-" * 78)
    print("%-8s %-22s %6s %9s %8s %8s %10s" % (
        "组别", "配置", "帧数", "耗时(s)", "FPS", "检出总数", "帧均检出"))
    print("%-8s %-22s %6d %9.2f %8.2f %8d %10.3f" % (
        base["name"], "FP32 + 逐帧检测(间隔1)", base["frames"],
        base["elapsed_sec"], base["fps"], base["total_records"],
        base["avg_records_per_frame"]))
    opt_desc = "%s + 跳帧间隔%d" % (
        "FP16" if opt["fp16_effective"] else "FP32(无CUDA降级)",
        opt["detect_interval"])
    print("%-8s %-22s %6d %9.2f %8.2f %8d %10.3f" % (
        opt["name"], opt_desc, opt["frames"], opt["elapsed_sec"],
        opt["fps"], opt["total_records"], opt["avg_records_per_frame"]))
    print("-" * 78)
    print("优化后检测帧 %d 帧实测检出 %d；确认事件：优化前 %d / 优化后 %d" % (
        opt["real_detect_frames"], opt["real_detect_records"],
        base["confirmed_events"], opt["confirmed_events"]))
    print("FPS 提升：%+.1f%%（达标线 ≥%.0f%%，%s）：%s" % (
        speedup, fps_threshold,
        "有 CUDA 看综合提升" if env["cuda_available"] else "无 CUDA 只看跳帧单项",
        "达标" if fps_pass else "未达标"))
    print("帧均检出数变化：%+.1f%%（允许下降 ≤10%%）：%s" % (
        -drop, "达标" if drop_pass else "未达标"))
    print("=" * 78)

    passed = fps_pass and drop_pass
    conclusion = {
        "fps_speedup_pct": round(speedup, 2),
        "fps_threshold_pct": fps_threshold,
        "fps_pass": fps_pass,
        "avg_records_drop_pct": round(drop, 2),
        "drop_pass": drop_pass,
        "overall_pass": passed,
    }
    return passed, conclusion


def main():
    """主入口：加载模型一次，依次跑优化前/优化后两组并输出对比。"""
    args = parse_args()
    if not os.path.isfile(args.video):
        print("视频文件不存在：%s" % args.video)
        return 1

    env = print_env()

    # 模型只加载一次，两组共享（加载耗时不计入任何一组计时）
    from ultralytics import YOLO
    config = Config()
    weights = config.abs_path(config.weights_path)
    logger.info("基准测试加载模型权重：%s", weights)
    model = YOLO(weights)

    # 优化前（基线组）：FP32 + 逐帧检测
    base = run_group("优化前", args.video,
                     build_config(fp16=False, detect_interval=1),
                     model, args.max_frames)
    # 优化后（加速组）：跳帧间隔 N，FP16 按环境自动判定（无 CUDA 降级）
    opt = run_group("优化后", args.video,
                    build_config(fp16=True, detect_interval=args.interval),
                    model, args.max_frames)

    passed, conclusion = print_report(base, opt, env)

    if args.out:
        payload = {
            "video": os.path.abspath(args.video),
            "max_frames": args.max_frames,
            "interval": args.interval,
            "environment": env,
            "baseline": base,
            "optimized": opt,
            "conclusion": conclusion,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                    exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print("实测结果已落盘：%s" % os.path.abspath(args.out))

    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
