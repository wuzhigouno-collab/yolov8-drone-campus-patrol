# -*- coding: utf-8 -*-
"""鹰眼巡校——A10 系统集成与多高度评估脚本。

功能意图：
    按 YAML 配置中的"飞行高度档位 -> 测试视频"分组，对每个视频运行
    与系统运行时同源的完整巡检管线（检测 -> ByteTrack 跟踪 ->
    多帧轨迹确认，复用 src.tracker.ByteTrackTracker），统计检测
    成功率 / 误检率 / 漏检率三项指标，并按高度档位汇总输出，
    为 docs/测试报告.md 提供真实运行数据。

指标口径（无人工标注真值时的代理口径，诚实且可复现）
================================================================
本系统测试素材未附带人工标注真值，无法计算经典意义下的
Precision/Recall。本脚本采用如下**代理口径**，全部从真实运行
的管线状态推导，可复现、可交叉核对：

  1. 检测成功率（轨迹确认率）
       = 确认轨迹数 / 候选轨迹总数
     "候选轨迹"：评估期间被跟踪器分配过 ID 的全部轨迹（出现 >=1 帧）；
     "确认轨迹"：累计存在满 Config.confirm_frames（默认 3）帧、
     被系统标记为"已确认"的轨迹。物理含义：出现过的目标中，被
     系统稳定锁定为真实目标的比例。

  2. 误检率（疑似误检占比，两个层级）
       轨迹级 = 未确认即消失的轨迹数 / 候选轨迹总数
       帧级   = 未确认目标记录数 / 全部目标记录数
     只出现一两帧就消失的轨迹 / 记录被视为疑似误检（瞬时抖动、
     背景干扰等），即"被多帧确认机制压制掉的部分"。轨迹级误检率
     与检测成功率互补（两者相加恒为 1），帧级口径提供观察噪声
     密度的独立视角。

  3. 漏检率（轨迹缺口率）
       = 确认轨迹生命周期内缺席帧数 / 确认轨迹生命周期总帧数
     对每条确认轨迹：生命周期 = 末次出现帧号 - 首次出现帧号 + 1，
     缺席帧 = 生命周期 - 实际被观测帧数。缺席说明目标处于画面中
     却在该帧未被检出/关联，作为漏检的代理度量。

  4. 人工标注比对（可选，优先级高于代理口径）
     配置中为某视频指定 gt_file 时，改用真值比对口径：
     标注文件为 CSV（表头 frame,count），逐帧给出画面内真实目标
     总数；系统输出取该帧"已确认目标数"。对 gt>0 的帧累计：
       检测成功率 = Σmin(系统数, 真值数) / Σ真值数
       漏检率     = Σmax(真值数-系统数, 0) / Σ真值数
       误检率     = Σmax(系统数-真值数, 0) / Σ真值数
     （分母为真值目标总帧次；误检率可大于 1 的情形即系统输出
      系统性多于真值。）

配置格式（tests/eval_config.yaml 示例）：
    weights: weights/yolov8n.pt      # 可选，默认走 Config.weights_path
    out: tests/a10_eval.json         # 可选，结果 JSON 落盘路径
    groups:
      - height_m: 10                 # 高度档位（米）
        note: 素材归挡说明（原样写入报告，便于溯源）
        videos:
          - path: C:/path/to/video.mp4
            max_frames: 3712         # 可选，缺省跑完整视频
            gt_file: null            # 可选，人工标注 CSV

用法示例：
    python tools/evaluate.py --config tests/eval_config.yaml
    python tools/evaluate.py --config tests/eval_config.yaml \\
        --weights weights/yolo26n.pt --out tests/a10_eval_yolo26n.json
退出码：0 = 评估完成；1 = 配置/素材错误导致无法评估。
"""

import argparse
import csv
import json
import os
import sys
import time

import yaml

# 本文件位于 tools/ 下，直接运行时 sys.path 不含项目根目录，手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.config import Config
from src.logger import get_logger
from src.tracker import ByteTrackTracker

logger = get_logger("evaluate")


def parse_args():
    """解析命令行参数。

    --config   评估配置 YAML 路径（必填，按高度档位分组列出视频）
    --weights  可选：覆盖配置中的权重路径（用于模型对比实验）
    --out      可选：覆盖配置中的结果 JSON 落盘路径
    """
    parser = argparse.ArgumentParser(
        description="鹰眼巡校 A10 多高度分组评估（检测成功率/误检率/漏检率）")
    parser.add_argument("--config", required=True, help="评估配置 YAML 路径")
    parser.add_argument("--weights", default=None,
                        help="可选：覆盖配置中的模型权重路径")
    parser.add_argument("--out", default=None,
                        help="可选：覆盖结果 JSON 落盘路径")
    return parser.parse_args()


def load_eval_config(path):
    """读取并校验评估配置 YAML，返回规范化后的配置字典。

    校验项：groups 必须为列表、每组必须有数值 height_m 与列表
    videos（允许为空列表，表示该档位暂无素材）；视频项必须有 path。
    """
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("groups"), list):
        raise ValueError("配置文件缺少 groups 列表：%s" % path)
    for gi, group in enumerate(cfg["groups"]):
        if not isinstance(group, dict) or "height_m" not in group:
            raise ValueError("第 %d 个分组缺少 height_m 字段" % gi)
        videos = group.get("videos")
        if videos is None:
            group["videos"] = []
        if not isinstance(group["videos"], list):
            raise ValueError("第 %d 个分组的 videos 必须是列表" % gi)
        for vi, video in enumerate(group["videos"]):
            if not isinstance(video, dict) or not video.get("path"):
                raise ValueError("第 %d 组第 %d 个视频缺少 path 字段" % (gi, vi))
    return cfg


def load_gt_counts(gt_file):
    """读取人工标注 CSV（表头 frame,count），返回 {帧号: 真值目标数}。"""
    counts = {}
    with open(gt_file, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            counts[int(row["frame"])] = int(row["count"])
    return counts


def eval_video(video_cfg, model, config, src_root):
    """对单个视频运行完整巡检管线并统计代理指标。

    参数：
        video_cfg: 配置中的视频项（path / max_frames / gt_file）；
        model:     已加载的 YOLO 模型（同一次评估内多视频共享）；
        config:    Config 实例（权重路径已被调用方按需覆盖）；
        src_root:  项目根目录，用于把相对 gt_file 路径换算为绝对路径。

    返回：
        结果字典：视频信息、帧数与耗时、轨迹级/帧级代理指标、
        （可选）真值比对指标。所有数字均为本次真实运行产出。
    """
    video_path = video_cfg["path"]
    tracker = ByteTrackTracker(config, model=model)
    # 清理共享模型的 predictor，避免上一段视频的轨迹接续到本段
    tracker.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("无法打开视频：%s" % video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = video_cfg.get("max_frames") or total_frames
    max_frames = min(int(max_frames), total_frames)

    # 逐轨迹观测统计：track_id -> {cls, first, last, seen, confirmed}
    tracks = {}
    frames = 0
    total_records = 0        # 全部帧目标记录总数
    unconfirmed_records = 0  # 帧级未确认记录数（疑似误检帧级口径）
    per_frame_confirmed = {} # 帧号 -> 已确认目标数（真值比对用）
    t0 = time.perf_counter()
    while frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        result = tracker.update(frame, frames)
        total_records += len(result.records)
        unconfirmed_records += len(result.unconfirmed)
        per_frame_confirmed[frames] = len(result.confirmed)
        for rec in result.records:
            tid = rec["track_id"]
            stat = tracks.get(tid)
            if stat is None:
                stat = {"cls": rec["det"]["cls"], "first": frames,
                        "last": frames, "seen": 0, "confirmed": False}
                tracks[tid] = stat
            stat["last"] = frames
            stat["seen"] += 1
            if rec["confirmed"]:
                stat["confirmed"] = True
    elapsed = time.perf_counter() - t0
    cap.release()

    candidate = len(tracks)
    confirmed = sum(1 for s in tracks.values() if s["confirmed"])
    # 漏检代理：确认轨迹生命周期内的缺席帧占比
    lifespan_sum = 0
    gap_sum = 0
    for s in tracks.values():
        if not s["confirmed"]:
            continue
        lifespan = s["last"] - s["first"] + 1
        lifespan_sum += lifespan
        gap_sum += lifespan - s["seen"]

    result = {
        "video": os.path.abspath(video_path),
        "video_total_frames": total_frames,
        "video_fps": round(src_fps, 2),
        "frames_processed": frames,
        "elapsed_sec": round(elapsed, 3),
        "process_fps": round(frames / elapsed, 2) if elapsed > 0 else 0.0,
        "backend": tracker.backend,
        "candidate_tracks": candidate,
        "confirmed_tracks": confirmed,
        "unconfirmed_tracks": candidate - confirmed,
        "confirmed_events": len(tracker.confirmed_events),
        "total_records": total_records,
        "unconfirmed_records": unconfirmed_records,
        "avg_records_per_frame": round(total_records / frames, 4)
        if frames else 0.0,
        # 三项代理指标；分母为 0（全程无目标）时记 None 并在报告注明
        "detection_success_rate": round(confirmed / candidate, 4)
        if candidate else None,
        "false_rate_track": round((candidate - confirmed) / candidate, 4)
        if candidate else None,
        "false_rate_frame": round(unconfirmed_records / total_records, 4)
        if total_records else None,
        "miss_rate_track_gap": round(gap_sum / lifespan_sum, 4)
        if lifespan_sum else None,
        "confirmed_track_lifespan_frames": lifespan_sum,
        "confirmed_track_gap_frames": gap_sum,
    }

    # 可选：人工标注真值比对（指定 gt_file 时启用，口径见模块 docstring）
    gt_file = video_cfg.get("gt_file")
    if gt_file:
        gt_path = gt_file if os.path.isabs(gt_file) \
            else os.path.join(src_root, gt_file)
        gt_counts = load_gt_counts(gt_path)
        gt_total = 0
        hit = 0
        miss = 0
        false = 0
        for frame_no, gt_n in gt_counts.items():
            if gt_n <= 0 or frame_no > frames:
                continue
            sys_n = per_frame_confirmed.get(frame_no, 0)
            gt_total += gt_n
            hit += min(sys_n, gt_n)
            miss += max(gt_n - sys_n, 0)
            false += max(sys_n - gt_n, 0)
        result["gt_file"] = gt_path
        result["gt_target_frames"] = gt_total
        result["gt_detection_success_rate"] = round(hit / gt_total, 4) \
            if gt_total else None
        result["gt_miss_rate"] = round(miss / gt_total, 4) \
            if gt_total else None
        result["gt_false_rate"] = round(false / gt_total, 4) \
            if gt_total else None
        logger.info(
            "真值比对：%s 真值目标总帧次 %d，成功率 %.4f / 漏检率 %.4f / "
            "误检率 %.4f", os.path.basename(video_path), gt_total,
            result["gt_detection_success_rate"] or 0.0,
            result["gt_miss_rate"] or 0.0,
            result["gt_false_rate"] or 0.0)

    logger.info(
        "%s 评估完成：%d/%d 帧，候选轨迹 %d（确认 %d），成功率 %s，"
        "帧级误检率 %s，轨迹缺口率 %s，确认事件 %d",
        os.path.basename(video_path), frames, total_frames, candidate,
        confirmed,
        _fmt(result["detection_success_rate"]),
        _fmt(result["false_rate_frame"]),
        _fmt(result["miss_rate_track_gap"]),
        result["confirmed_events"])
    return result


def _fmt(value):
    """指标格式化：None 显示为'无目标'，避免日志出现裸 None。"""
    return "%.4f" % value if value is not None else "无目标(分母0)"


def aggregate_group(group, video_results):
    """按高度档位汇总组内各视频指标（计数合并后重算比率）。"""
    frames = sum(r["frames_processed"] for r in video_results)
    candidate = sum(r["candidate_tracks"] for r in video_results)
    confirmed = sum(r["confirmed_tracks"] for r in video_results)
    total_records = sum(r["total_records"] for r in video_results)
    unconfirmed_records = sum(r["unconfirmed_records"] for r in video_results)
    lifespan = sum(r["confirmed_track_lifespan_frames"] for r in video_results)
    gap = sum(r["confirmed_track_gap_frames"] for r in video_results)
    elapsed = sum(r["elapsed_sec"] for r in video_results)
    summary = {
        "height_m": group["height_m"],
        "note": group.get("note", ""),
        "video_count": len(video_results),
        "frames_processed": frames,
        "elapsed_sec": round(elapsed, 3),
        "candidate_tracks": candidate,
        "confirmed_tracks": confirmed,
        "total_records": total_records,
        "detection_success_rate": round(confirmed / candidate, 4)
        if candidate else None,
        "false_rate_track": round((candidate - confirmed) / candidate, 4)
        if candidate else None,
        "false_rate_frame": round(unconfirmed_records / total_records, 4)
        if total_records else None,
        "miss_rate_track_gap": round(gap / lifespan, 4)
        if lifespan else None,
    }
    # 组内真值比对汇总（仅当组内存在带 gt_file 的视频时输出）
    gt_items = [r for r in video_results if "gt_target_frames" in r]
    if gt_items:
        gt_total = sum(r["gt_target_frames"] for r in gt_items)
        hit = sum(round(r["gt_detection_success_rate"] * r["gt_target_frames"])
                  for r in gt_items)
        summary["gt_target_frames"] = gt_total
        summary["gt_detection_success_rate"] = round(hit / gt_total, 4) \
            if gt_total else None
    return summary


def print_group_report(summary, video_results):
    """打印单个高度档位的指标表（视频明细 + 档位汇总）。"""
    print()
    print("=" * 78)
    print("高度档位：%s 米（%s）" % (summary["height_m"],
                                 summary["note"] or "无说明"))
    if not video_results:
        print("  本档位暂无素材，未评估。")
        print("=" * 78)
        return
    print("-" * 78)
    print("%-24s %6s %7s %7s %9s %9s %9s" % (
        "视频", "帧数", "候选", "确认", "成功率", "误检率F", "漏检率"))
    for r in video_results:
        print("%-24s %6d %7d %7d %9s %9s %9s" % (
            os.path.basename(r["video"])[:24], r["frames_processed"],
            r["candidate_tracks"], r["confirmed_tracks"],
            _fmt(r["detection_success_rate"]),
            _fmt(r["false_rate_frame"]),
            _fmt(r["miss_rate_track_gap"])))
    print("-" * 78)
    print("%-24s %6d %7d %7d %9s %9s %9s" % (
        "【档位汇总】", summary["frames_processed"],
        summary["candidate_tracks"], summary["confirmed_tracks"],
        _fmt(summary["detection_success_rate"]),
        _fmt(summary["false_rate_frame"]),
        _fmt(summary["miss_rate_track_gap"])))
    print("  （误检率F=帧级疑似误检占比；漏检率=确认轨迹生命周期缺口占比；"
          "轨迹级误检率=%s，与成功率互补）" % _fmt(summary["false_rate_track"]))
    print("=" * 78)


def print_env():
    """探测并打印运行环境（torch/CUDA/GPU 型号），写入结果与日志留痕。"""
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


def main():
    """主入口：按配置逐档位逐视频评估，打印报告并落盘 JSON。"""
    args = parse_args()
    if not os.path.isfile(args.config):
        print("配置文件不存在：%s" % args.config)
        return 1
    try:
        eval_cfg = load_eval_config(args.config)
    except (ValueError, yaml.YAMLError) as exc:
        print("配置解析失败：%s" % exc)
        return 1

    config = Config()
    src_root = Config.project_root()
    # 权重优先级：命令行 --weights > 配置 weights > Config 默认
    weights_rel = args.weights or eval_cfg.get("weights") \
        or config.weights_path
    config.weights_path = weights_rel
    weights_abs = config.abs_path(weights_rel)
    if not os.path.isfile(weights_abs):
        print("权重文件不存在：%s" % weights_abs)
        return 1
    out_path = args.out or eval_cfg.get("out") \
        or os.path.join("tests", "a10_eval.json")
    if not os.path.isabs(out_path):
        out_path = os.path.join(src_root, out_path)

    env = print_env()
    print("评估权重：%s" % weights_abs)

    # 模型只加载一次，全部档位/视频共享（每组之间 tracker.reset）
    from ultralytics import YOLO
    logger.info("评估加载模型权重：%s", weights_abs)
    model = YOLO(weights_abs)

    # 逐档位评估；素材路径在运行前统一校验，避免跑到一半才发现缺失
    groups = eval_cfg["groups"]
    for group in groups:
        for video in group["videos"]:
            if not os.path.isfile(video["path"]):
                print("素材文件不存在：%s" % video["path"])
                return 1

    group_summaries = []
    for group in groups:
        video_results = []
        for video in group["videos"]:
            video_results.append(eval_video(video, model, config, src_root))
        summary = aggregate_group(group, video_results)
        summary["videos"] = video_results
        group_summaries.append(summary)
        print_group_report(summary, video_results)

    payload = {
        "config_file": os.path.abspath(args.config),
        "weights": weights_abs,
        "metric_definitions": "代理口径见 tools/evaluate.py 模块 docstring",
        "environment": env,
        "groups": group_summaries,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print()
    print("评估结果已落盘：%s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
