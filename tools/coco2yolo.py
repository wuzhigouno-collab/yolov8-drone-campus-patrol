# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

COCO 标注格式转 YOLO 标注格式转换脚本。

功能意图：
    将 COCO 格式的数据集标注（单个 JSON 文件汇总全部图像与目标框）
    转换为 YOLO 训练所需的逐图 .txt 标注（每行一个目标，类别 id +
    归一化中心坐标）。用于把 VisDrone2019 等公开数据集（经官方或
    第三方工具整理为 COCO 格式后）以及标注工具导出的 COCO 结果，
    接入本项目训练管线（tools/train.py + data/campus.yaml）。

    与参考实现（reference/hw5-code）相比，本脚本按本项目需要重写，
    额外支持：
      1. 类别筛选（--classes）：只保留指定类别，输出类别 id 按
         --classes 顺序从 0 重新编号，与 data/campus.yaml 的 names
         顺序对应；
      2. 类别改名映射（--class-map）：如 VisDrone 的 pedestrian/people
         统一映射为 person；
      3. 小目标筛选（--min-size）：按宽/高像素下限丢弃过小目标，
         对应采集标注规范中的 32×32 像素筛选标准。

格式说明：
    COCO：bbox 为 [x 左上角, y 左上角, 宽, 高] 的像素绝对坐标；
    YOLO：每行 <class_id> <x中心> <y中心> <宽> <高>，均为相对图像
          宽高的归一化值（0~1），标注文件与图像同名、扩展名 .txt。

用法示例：
    python tools/coco2yolo.py \
        --coco-json datasets/visdrone/annotations/train.json \
        --out datasets/campus/labels/train \
        --classes person car bicycle \
        --class-map '{"pedestrian": "person", "people": "person"}' \
        --min-size 32
"""

import argparse
import json
import os
import sys
from collections import defaultdict

# 脚本位于 tools/ 下，直接运行时 sys.path 不含项目根目录，手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.logger import get_logger


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="COCO 标注转 YOLO 标注（支持类别映射/筛选与小目标过滤）"
    )
    parser.add_argument("--coco-json", required=True, help="COCO 标注 JSON 文件路径")
    parser.add_argument(
        "--out", required=True,
        help="YOLO 标注输出目录（通常为 <数据集>/labels/<train|val>）",
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="只保留这些类别（按给定顺序重新编号 0,1,2...）；不指定则保留全部类别",
    )
    parser.add_argument(
        "--class-map", default=None,
        help='类别改名映射 JSON 字符串，如 \'{"pedestrian": "person"}\'，'
             "映射在 --classes 筛选之前生效",
    )
    parser.add_argument(
        "--min-size", type=float, default=0.0,
        help="目标宽或高的最小像素值，任一维度小于该值的目标被丢弃（32 对应 32×32 筛选标准）",
    )
    return parser.parse_args()


def convert_coco_to_yolo(coco_json_path, out_dir, keep_classes=None,
                         class_map=None, min_size=0.0, logger=None):
    """执行单个 COCO 标注文件到 YOLO 标注目录的转换，返回统计信息字典。

    参数：
        coco_json_path: COCO 标注 JSON 路径；
        out_dir:        YOLO 标注输出目录；
        keep_classes:   要保留的类别名列表（顺序即新类别 id），None 表示全保留；
        class_map:      类别改名映射字典 {原类别名: 新类别名}；
        min_size:       目标宽/高像素下限，小于则丢弃；
        logger:         日志器，可为 None。
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    # 类别改名映射先行（如 pedestrian/people -> person），再按保留名单筛选
    class_map = class_map or {}
    mapped_names = {}  # 原 category_id -> 映射后的类别名
    for cat in categories:
        mapped_names[cat["id"]] = class_map.get(cat["name"], cat["name"])

    if keep_classes:
        # 新类别 id 按 keep_classes 给定顺序从 0 编号，与 campus.yaml names 对齐
        new_id = {name: idx for idx, name in enumerate(keep_classes)}
    else:
        ordered = []  # 按 COCO categories 出现顺序保留全部（去重后）
        for cat in categories:
            name = mapped_names[cat["id"]]
            if name not in ordered:
                ordered.append(name)
        new_id = {name: idx for idx, name in enumerate(ordered)}
        keep_classes = ordered

    # 图像 id -> 尺寸/文件名；图像 id -> 该图全部标注
    image_info = {img["id"]: img for img in images}
    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    os.makedirs(out_dir, exist_ok=True)

    n_targets = 0        # 实际写出的目标数
    n_skip_class = 0     # 因类别筛选丢弃的目标数
    n_skip_small = 0     # 因小目标过滤丢弃的目标数
    n_skip_bad = 0       # 因坐标非法/退化丢弃的目标数

    for img in images:
        img_w = img["width"]
        img_h = img["height"]
        lines = []
        for ann in anns_by_image.get(img["id"], []):
            name = mapped_names.get(ann["category_id"])
            if name not in new_id:
                n_skip_class += 1
                continue
            x, y, w, h = ann["bbox"]  # COCO：左上角 + 宽高（像素）
            if w < min_size or h < min_size:
                n_skip_small += 1
                continue
            if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
                n_skip_bad += 1
                continue
            # 转归一化中心坐标并裁剪到 [0,1]，防止越界框使训练报错
            cx = min(max((x + w / 2) / img_w, 0.0), 1.0)
            cy = min(max((y + h / 2) / img_h, 0.0), 1.0)
            nw = min(max(w / img_w, 0.0), 1.0)
            nh = min(max(h / img_h, 0.0), 1.0)
            lines.append("%d %.6f %.6f %.6f %.6f" % (new_id[name], cx, cy, nw, nh))
            n_targets += 1
        # 每张图（含无保留目标的图）都生成同名 .txt，空文件表示无目标
        base = os.path.splitext(os.path.basename(img["file_name"]))[0]
        with open(os.path.join(out_dir, base + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    # 类别清单写到输出目录的上一级（<数据集>/classes.txt），与 hw5 参考布局一致
    classes_txt = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "classes.txt")
    with open(classes_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(keep_classes) + "\n")

    stats = {
        "images": len(images),
        "annotations": len(annotations),
        "written": n_targets,
        "skip_class": n_skip_class,
        "skip_small": n_skip_small,
        "skip_bad": n_skip_bad,
        "classes": keep_classes,
        "classes_txt": classes_txt,
    }
    if logger:
        logger.info(
            "转换完成：图像 %d 张，原始标注 %d 条 -> 写出 %d 条"
            "（类别筛选丢弃 %d，小目标丢弃 %d，非法框丢弃 %d）；类别 %s；清单 %s",
            stats["images"], stats["annotations"], stats["written"],
            stats["skip_class"], stats["skip_small"], stats["skip_bad"],
            "/".join(keep_classes), classes_txt,
        )
    return stats


def main():
    """主流程：解析参数 -> 读 COCO JSON -> 逐图写出 YOLO 标注 -> 打印统计。"""
    args = parse_args()
    config = Config()
    logger = get_logger(__name__, config)

    if not os.path.exists(args.coco_json):
        logger.error("COCO 标注文件不存在：%s", args.coco_json)
        sys.exit(2)

    class_map = None
    if args.class_map:
        try:
            class_map = json.loads(args.class_map)
        except json.JSONDecodeError as exc:
            logger.error("--class-map 不是合法 JSON：%s", exc)
            sys.exit(2)

    stats = convert_coco_to_yolo(
        coco_json_path=args.coco_json,
        out_dir=args.out,
        keep_classes=args.classes,
        class_map=class_map,
        min_size=args.min_size,
        logger=logger,
    )
    print("转换完成：写出目标 %d 条（图像 %d 张），输出目录 %s" % (
        stats["written"], stats["images"], args.out))


if __name__ == "__main__":
    main()
