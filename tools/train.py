# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

模型迁移学习微调脚本（A9 训练管线入口）。

功能意图：
    以 COCO 预训练权重（默认 weights/yolov8n.pt，可配置）为起点，
    在校园巡检数据集（VisDrone2019 主数据集 + 自采校园数据）上做
    迁移学习微调。训练轮数、学习率、输入尺寸与数据增强参数均可
    通过命令行配置；正式训练强制 imgsz >= 960 并开启多尺度训练，
    以保障无人机俯视小目标（行人/车辆）的召回率（plan.md 训练
    数据策略要求）。训练过程中 ultralytics 自动按验证集 mAP
    （fitness）保存最优权重 best.pt，正式训练结束后将其复制到
    weights/best.pt 供系统主线替换使用。

冒烟模式（--smoke）：
    无真实数据集时，自动在项目内合成一个迷你数据集（随机绘制的
    矩形目标 + YOLO 标注，固定随机种子可复现），用小 imgsz、少
    epoch 走完"加载权重 -> 训练 -> 验证 -> 保存权重"全流程，
    仅用于验证训练管线可用，不产出任何有实际意义的模型精度。

用法示例：
    # 冒烟自验（自动合成迷你数据，--data 参数被忽略）
    python tools/train.py --data data/campus.yaml --epochs 1 --imgsz 320 --smoke

    # 正式微调（需先按 tools/README_TRAIN.md 备好数据集并填写 campus.yaml）
    python tools/train.py --data data/campus.yaml --epochs 100 --imgsz 960
"""

import argparse
import os
import random
import shutil
import sys

# 脚本位于 tools/ 下，直接运行时 sys.path 不含项目根目录，
# 这里手动补上，保证 src 包可被正常导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.logger import get_logger

# 正式训练允许的最小输入尺寸：俯视小目标召回要求 imgsz >= 960（plan.md 约定）
TRAIN_MIN_IMGSZ = 960

# 冒烟模式自动合成的迷你数据集目录（相对项目根目录）
SMOKE_DATASET_DIR = os.path.join("data", "smoke_dataset")
# 冒烟数据集类别（与系统关注类别一致）
SMOKE_CLASSES = ["person", "car", "bicycle"]
# 冒烟数据集规模：训练 8 张、验证 4 张，够跑通流程即可
SMOKE_TRAIN_COUNT = 8
SMOKE_VAL_COUNT = 4
# 冒烟图像尺寸与随机种子（固定种子保证可复现）
SMOKE_IMG_W = 640
SMOKE_IMG_H = 480
SMOKE_SEED = 42


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="鹰眼巡校 YOLO 迁移学习微调脚本（--smoke 为流程冒烟模式）"
    )
    parser.add_argument(
        "--data", default="data/campus.yaml",
        help="数据集配置 YAML 路径（相对项目根目录）；--smoke 模式下忽略",
    )
    parser.add_argument(
        "--weights", default=None,
        help="预训练权重路径，默认读取 Config().weights_path（weights/yolov8n.pt）",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数，默认 100")
    parser.add_argument(
        "--imgsz", type=int, default=960,
        help="训练输入图像尺寸；正式训练必须 >= %d（俯视小目标召回要求）" % TRAIN_MIN_IMGSZ,
    )
    parser.add_argument("--batch", type=int, default=8, help="批大小，默认 8")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率，默认 0.01")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率系数（lr0 * lrf），默认 0.01")
    # ---------- 数据增强参数（与 ultralytics 训练参数同名） ----------
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV 色调增强幅度")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV 饱和度增强幅度")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV 明度增强幅度")
    parser.add_argument("--degrees", type=float, default=0.0, help="随机旋转角度范围（度）")
    parser.add_argument("--translate", type=float, default=0.1, help="随机平移比例")
    parser.add_argument("--scale", type=float, default=0.5, help="随机缩放比例")
    parser.add_argument("--fliplr", type=float, default=0.5, help="左右翻转概率")
    parser.add_argument("--mosaic", type=float, default=1.0, help="马赛克增强概率")
    # ---------- 运行控制 ----------
    parser.add_argument(
        "--smoke", action="store_true",
        help="冒烟模式：自动合成迷你数据集，仅验证训练流程走通，不产出有效精度",
    )
    parser.add_argument("--project", default="runs", help="训练输出根目录，默认 runs/")
    parser.add_argument("--name", default=None, help="本次运行子目录名，默认按模式自动命名")
    parser.add_argument("--device", default=None, help="训练设备（如 0 / cpu），默认自动选择")
    parser.add_argument("--seed", type=int, default=42, help="训练随机种子，默认 42")
    return parser.parse_args()


def build_smoke_dataset(config, logger):
    """自动合成冒烟模式迷你数据集，返回其 YAML 配置绝对路径。

    数据内容：固定随机种子下在纯色背景上绘制若干带颜色的矩形框，
    模拟 person/car/bicycle 三类目标，同时生成 YOLO 格式标注与
    数据集 YAML。仅用于走通训练流程，目标无可学习语义，精度无意义。
    重复运行时若数据集已存在则直接复用，保证幂等。
    """
    # 延迟导入 cv2：仅冒烟合成时需要
    import cv2
    import numpy as np

    root = config.abs_path(SMOKE_DATASET_DIR)
    yaml_path = os.path.join(root, "smoke.yaml")
    if os.path.exists(yaml_path):
        logger.info("冒烟数据集已存在，直接复用：%s", root)
        return yaml_path

    rng = random.Random(SMOKE_SEED)
    # 各类别的颜色（BGR）与框体宽高范围，模拟不同长宽比的目标
    class_style = {
        0: {"color": (60, 160, 60), "w": (20, 45), "h": (50, 110)},    # person：高瘦
        1: {"color": (200, 120, 40), "w": (70, 140), "h": (35, 70)},   # car：宽扁
        2: {"color": (40, 60, 200), "w": (50, 90), "h": (30, 60)},     # bicycle：中等
    }

    def gen_split(split, count):
        """生成一个数据划分（train/val）的图像与 YOLO 标注。"""
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for idx in range(count):
            # 亮灰底背景加少量噪点，避免全黑/全白退化输入
            img = np.full((SMOKE_IMG_H, SMOKE_IMG_W, 3), 220, dtype=np.uint8)
            noise = np.random.default_rng(SMOKE_SEED + idx).integers(
                0, 25, img.shape, dtype=np.uint8)
            img = cv2.subtract(img, noise)
            lines = []
            for _ in range(rng.randint(2, 5)):
                cls_id = rng.choice(list(class_style.keys()))
                style = class_style[cls_id]
                w = rng.randint(*style["w"])
                h = rng.randint(*style["h"])
                x1 = rng.randint(0, SMOKE_IMG_W - w)
                y1 = rng.randint(0, SMOKE_IMG_H - h)
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), style["color"], -1)
                # 像素框转 YOLO 归一化中心坐标
                cx = (x1 + w / 2) / SMOKE_IMG_W
                cy = (y1 + h / 2) / SMOKE_IMG_H
                lines.append("%d %.6f %.6f %.6f %.6f" % (
                    cls_id, cx, cy, w / SMOKE_IMG_W, h / SMOKE_IMG_H))
            cv2.imwrite(os.path.join(img_dir, "smoke_%s_%03d.jpg" % (split, idx)), img)
            with open(os.path.join(lbl_dir, "smoke_%s_%03d.txt" % (split, idx)),
                      "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    gen_split("train", SMOKE_TRAIN_COUNT)
    gen_split("val", SMOKE_VAL_COUNT)

    # 数据集 YAML：path 用绝对路径，避免受 ultralytics datasets_dir 设置影响
    names = "\n".join("  %d: %s" % (i, n) for i, n in enumerate(SMOKE_CLASSES))
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# 冒烟模式自动合成的迷你数据集（tools/train.py 生成，仅验证训练流程）\n")
        f.write("path: %s\n" % root.replace("\\", "/"))
        f.write("train: images/train\nval: images/val\nnames:\n%s\n" % names)

    logger.info(
        "冒烟迷你数据集已合成：%s（train %d 张 / val %d 张，类别 %s）",
        root, SMOKE_TRAIN_COUNT, SMOKE_VAL_COUNT, "/".join(SMOKE_CLASSES),
    )
    return yaml_path


def main():
    """主流程：解析参数 -> 模式校验 -> 加载预训练权重 -> 微调训练 -> 汇总产物。"""
    args = parse_args()
    config = Config()
    logger = get_logger(__name__, config)

    # 预训练权重：命令行未指定时读集中配置（默认 weights/yolov8n.pt）
    weights = config.abs_path(args.weights) if args.weights else \
        config.abs_path(config.weights_path)
    if not os.path.exists(weights):
        logger.error("预训练权重不存在：%s", weights)
        sys.exit(2)

    if args.smoke:
        # 冒烟模式：忽略 --data，自动合成迷你数据集；小尺寸少轮数只验证流程，
        # 不开启多尺度训练（多尺度是正式训练保障小目标召回的要求，冒烟无需）
        data_yaml = build_smoke_dataset(config, logger)
        multi_scale = False
        run_name = args.name or "smoke_train"
        logger.warning(
            "冒烟模式：使用自动合成迷你数据集（忽略 --data），"
            "imgsz=%d epochs=%d，仅验证流程，精度无意义",
            args.imgsz, args.epochs,
        )
    else:
        # 正式训练：数据集配置必须存在；imgsz 不得低于 960；强制多尺度训练
        data_yaml = config.abs_path(args.data)
        if not os.path.exists(data_yaml):
            logger.error(
                "数据集配置不存在：%s（请先按 tools/README_TRAIN.md 准备数据集"
                "并填写 data/campus.yaml，或用 --smoke 走流程自验）", data_yaml)
            sys.exit(2)
        if args.imgsz < TRAIN_MIN_IMGSZ:
            logger.error(
                "正式训练 imgsz=%d 低于下限 %d：俯视小目标召回要求大输入尺寸；"
                "流程自验请使用 --smoke 模式", args.imgsz, TRAIN_MIN_IMGSZ)
            sys.exit(2)
        # 多尺度取值依据（参数语义修正）：ultralytics 8.4.21 起 multi_scale 由
        # bool 改 float 语义（PR #23284，缩放比例 fraction）；True 会被按 1.0
        # 解释，preprocess_batch 缩放下界 int(imgsz*(1-1.0))=0，randrange 抽出
        # 0 后 interpolate(size=[0,0]) 确定性崩溃（issue #23480）。0.5 等价旧版
        # True 的 ±50% 抖动，与 plan.md"开启多尺度训练"原意一致。
        multi_scale = 0.5
        run_name = args.name or "campus_train"
        logger.info(
            "正式迁移学习：data=%s weights=%s imgsz=%d epochs=%d lr0=%g "
            "多尺度训练=开启（plan.md 训练数据策略）",
            args.data, weights, args.imgsz, args.epochs, args.lr0,
        )

    # 延迟导入 ultralytics：该库加载较慢，仅在真正训练时才引入
    from ultralytics import YOLO

    # 以预训练权重为起点做迁移学习微调
    model = YOLO(weights)
    train_kwargs = dict(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        # 数据增强参数（均可命令行配置）
        hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
        degrees=args.degrees, translate=args.translate,
        scale=args.scale, fliplr=args.fliplr, mosaic=args.mosaic,
        multi_scale=multi_scale,
        # save=True：按验证集 mAP（fitness）自动保存最优权重 best.pt 与末轮 last.pt
        save=True,
        project=config.abs_path(args.project),
        name=run_name,
        seed=args.seed,
        pretrained=True,
        val=True,
        # 冒烟模式禁用多进程数据加载，避免 Windows 下 spawn 开销拖慢小数据集；
        # 同时关闭 AMP 检查（检查会尝试联网下载临时权重，离线环境会空等重试）
        workers=0 if args.smoke else 8,
        amp=False if args.smoke else True,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device
    results = model.train(**train_kwargs)

    # 训练产物目录（runs/<name>），其中 weights/best.pt 为验证集最优权重
    save_dir = str(getattr(results, "save_dir", "")) or \
        os.path.join(config.abs_path(args.project), run_name)
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    last_pt = os.path.join(save_dir, "weights", "last.pt")
    logger.info("训练完成，产物目录：%s", save_dir)
    logger.info("验证集最优权重：%s；末轮权重：%s", best_pt, last_pt)

    if not os.path.exists(best_pt):
        logger.error("未找到最优权重文件：%s", best_pt)
        sys.exit(1)

    if not args.smoke:
        # 正式训练：把验证集最优权重归置到 weights/best.pt，供系统主线替换
        dst = config.abs_path(os.path.join("weights", "best.pt"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(best_pt, dst)
        logger.info("最优权重已复制到 %s（可将 config.weights_path 指向它以启用自训模型）", dst)
    else:
        logger.info("冒烟模式产物仅用于流程验证，不复制到 weights/，可整体删除 runs/ 目录")

    print("训练完成：%s" % save_dir)
    print("最优权重：%s" % best_pt)


if __name__ == "__main__":
    main()
