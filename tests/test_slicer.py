# -*- coding: utf-8 -*-
"""鹰眼巡校——A12 切片推理模块 验收测试。

测试内容：
    一、纯单测（无需模型/视频，秒级完成）：
        1. 切片区域划分：2x2 重叠率 0.2 的区域坐标精确正确、全覆盖、
           相邻切片重叠率达标；重叠率 0 时切片恰好相接；1x1 退化为整图；
        2. 切片内坐标 -> 原图坐标映射正确（含越界钳制）；
        3. 跨切片边界目标合并去重专项：构造恰好跨切片边界的检测（同一
           目标被左右两片分别检出），验证合并 NMS 去重正确（只留高置
           信框）；IOU 恰好等于阈值的对照场景验证阈值可配置且生效；
           同类才抑制、异类不互杀（类别感知）；
    二、真实图片验收（tests/assets/a12_slice_frame.jpg，取自
        5G智慧农业素材.mp4 第 2225 帧，含 4 个高置信 person）：
        4. 切片模式检出数 >= 整图模式检出数；合并后无重复框（任意两框
           IOU < 0.5）；返回结构与 Detector.detect() 对齐；切片参数
           （行列数/重叠率/NMS 阈值）走 config 快照生效。

运行方式：
    python tests/test_slicer.py
全部断言通过时输出 PASS 并以退出码 0 结束；失败时抛出 AssertionError。
"""

import os
import sys

import cv2

# 本文件位于 tests/ 下，直接运行时 sys.path 不含项目根目录，手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.frame_confirm import iou_of
from src.logger import get_logger
from src.slicer import (
    SlicedDetector,
    compute_slice_regions,
    map_box_to_frame,
    nms_merge,
)

logger = get_logger("test_slicer")

# 真实验收素材：5G智慧农业素材.mp4 第 2225 帧（1920x1080，4 个高置信
# person），由开发期探测脚本截取保存
TEST_IMAGE = os.path.join("tests", "assets", "a12_slice_frame.jpg")


def make_det(cls, conf, box, slice_index=0):
    """构造与 Detector 输出结构对齐的合成检测项（附带切片序号字段）。"""
    x1, y1, x2, y2 = box
    return {
        "cls": cls,
        "conf": conf,
        "box": list(box),
        "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
        "slice": slice_index,
    }


def test_slice_regions():
    """用例 1：切片区域划分——坐标精确、全覆盖、重叠率达标、可走配置。"""
    # 1920x1080 按 2x2、重叠率 0.2 切分：
    # 切片宽 = ceil(1920 / (2 - 0.2)) = 1067，x 起点 [0, 853]
    # 切片高 = ceil(1080 / 1.8) = 600，y 起点 [0, 480]（行优先排列）
    regions = compute_slice_regions(1920, 1080, 2, 2, 0.2)
    assert regions == [
        (0, 0, 1067, 600),
        (853, 0, 1920, 600),
        (0, 480, 1067, 1080),
        (853, 480, 1920, 1080),
    ], "2x2 切片区域坐标错误: %s" % (regions,)

    # 全覆盖：四角区域并集必须覆盖整幅画面（四象限拼合无缝无溢）
    assert regions[0][:2] == (0, 0) and regions[-1][2:] == (1920, 1080)
    # 相邻切片实际重叠率不低于配置值（水平向 214/1067≈0.2006，
    # 垂直向 120/600=0.2）
    overlap_x = regions[0][2] - regions[1][0]
    overlap_y = regions[0][3] - regions[2][1]
    assert overlap_x >= 0.2 * 1067 - 1, "水平重叠不足: %d" % overlap_x
    assert overlap_y >= 0.2 * 600 - 1, "垂直重叠不足: %d" % overlap_y

    # 重叠率 0：切片恰好相接（无重叠无间隙）
    regions0 = compute_slice_regions(1920, 1080, 2, 2, 0.0)
    assert regions0[0] == (0, 0, 960, 540)
    assert regions0[1] == (960, 0, 1920, 540)
    assert regions0[0][2] == regions0[1][0] == 960

    # 1x1 退化为整图单片
    assert compute_slice_regions(1920, 1080, 1, 1, 0.2) == \
        [(0, 0, 1920, 1080)]

    # 行列数走配置：3x2 切 6 片，行优先排列（上排 3 片、下排 3 片）
    regions32 = compute_slice_regions(1920, 1080, 2, 3, 0.2)
    assert len(regions32) == 6
    assert regions32[0][1] == 0 and regions32[-1][3] == 1080
    logger.info(
        "用例1 PASS：2x2/重叠0.2 区域坐标精确、全覆盖、重叠率达标"
        "（水平重叠 %dpx、垂直 %dpx）；重叠0 相接、1x1 整图、3x2 六片",
        overlap_x, overlap_y,
    )


def test_coordinate_mapping():
    """用例 2：切片内坐标 -> 原图坐标映射正确（含越界钳制）。"""
    # 右下角切片（偏移 853, 480）内的局部框 (10,20,110,120)
    # 映射到原图 (863,500,963,600)
    mapped = map_box_to_frame([10, 20, 110, 120], 853, 480, 1920, 1080)
    assert mapped == [863, 500, 963, 600], "坐标映射错误: %s" % mapped

    # 局部框超出切片范围（模型输出可能略越界）时钳制到原图边缘：
    # (0,0,1100,650) + 偏移 (853,480) -> x2/y2 越界，钳到 1920/1080
    clamped = map_box_to_frame([0, 0, 1100, 650], 853, 480, 1920, 1080)
    assert clamped == [853, 480, 1920, 1080], "越界钳制错误: %s" % clamped

    # 负坐标（越左/上界）钳到 0
    neg = map_box_to_frame([-5, -8, 50, 60], 0, 0, 1920, 1080)
    assert neg == [0, 0, 50, 60], "负坐标钳制错误: %s" % neg
    logger.info(
        "用例2 PASS：切片内 (10,20,110,120)+偏移(853,480) -> 原图 %s；"
        "越界/负坐标钳制正确", mapped,
    )


def test_cross_boundary_merge():
    """用例 3（专项）：跨切片边界目标的合并去重逻辑。

    在 1000x800 画面、2x2、重叠率 0.2 下，切片 x 区间为
    [0,556) 与 [444,1000)，重叠区 [444,556)。构造一个恰好跨垂直
    切片边界的目标框 (400,100,560,200)：左片（slice 0，偏移 0,0）
    完整检出（conf 0.90），右片（slice 1，偏移 444,0）在重叠区
    重复检出同一目标（conf 0.70，框略有抖动）。
    """
    # 右片检出框：原图坐标 (404,102,558,202)，经右片偏移 (-444,0)
    # 换算回切片内局部坐标后，再走 map_box_to_frame 映射回原图，
    # 验证"局部 -> 原图"映射与合并链路的一致性与正确性
    local_in_tile1 = [404 - 444, 102, 558 - 444, 202]
    mapped_back = map_box_to_frame(local_in_tile1, 444, 0, 1000, 800)
    assert mapped_back == [404, 102, 558, 202]

    det_left = make_det("person", 0.90, [400, 100, 560, 200], slice_index=0)
    det_right = make_det("person", 0.70, mapped_back, slice_index=1)
    # 两片重复检出的同类框 IOU≈0.9254，达到默认阈值 0.5 -> 去重
    pair_iou = iou_of(det_left["box"], det_right["box"])
    assert pair_iou >= 0.5, "构造前提不成立：重复框 IOU=%.4f" % pair_iou
    merged = nms_merge([det_right, det_left], 0.5)
    assert len(merged) == 1, "跨边界重复检出应合并为 1 框: %s" % merged
    assert merged[0]["conf"] == 0.90, "合并应保留高置信度框"
    assert merged[0]["box"] == [400, 100, 560, 200]
    assert merged[0]["slice"] == 0, "合并应保留左片检出（置信度更高）"

    # 阈值对照场景：IOU 恰好 = 0.5 的一对框（(0,0,120,100) 与
    # (40,0,160,100)：交 80*100=8000，并 16000）
    det_a = make_det("car", 0.80, [0, 0, 120, 100])
    det_b = make_det("car", 0.60, [40, 0, 160, 100])
    assert abs(iou_of(det_a["box"], det_b["box"]) - 0.5) < 1e-6
    # 阈值 0.5：IOU >= 阈值即抑制 -> 合并为 1
    assert len(nms_merge([det_a, det_b], 0.5)) == 1
    # 阈值 0.6：IOU < 阈值 -> 两框都保留（阈值配置生效）
    assert len(nms_merge([det_a, det_b], 0.6)) == 2

    # 类别感知：同位置不同类别（人/车）即使 IOU=1 也不互相抑制
    det_person = make_det("person", 0.90, [100, 100, 220, 240])
    det_car = make_det("car", 0.70, [100, 100, 220, 240])
    both = nms_merge([det_person, det_car], 0.5)
    assert len(both) == 2, "类别不同不应合并去重"

    # 确定性：同分按下标，输入顺序不影响结果内容
    m1 = nms_merge([det_left, det_right], 0.5)
    m2 = nms_merge([det_right, det_left], 0.5)
    assert [d["box"] for d in m1] == [d["box"] for d in m2]
    logger.info(
        "用例3 PASS：跨切片边界目标（IOU=%.4f）合并去重保留高置信框；"
        "IOU=0.5 对照场景阈值 0.5 合并 / 0.6 保留；类别感知不互杀",
        pair_iou,
    )


def test_real_image_slice_vs_whole():
    """用例 4（核心验收）：真实高空测试图 切片模式 vs 整图模式。"""
    assert os.path.exists(TEST_IMAGE), "测试素材缺失: %s" % TEST_IMAGE
    image = cv2.imread(TEST_IMAGE)
    assert image is not None, "测试图片读取失败: %s" % TEST_IMAGE
    frame_h, frame_w = image.shape[:2]
    logger.info("测试图 %s（%dx%d）", TEST_IMAGE, frame_w, frame_h)

    # 延迟导入检测器（模型加载耗时数秒，仅本用例需要）
    from src.detector import Detector
    config = Config()
    detector = Detector(config)
    # 切片器复用 Detector 已加载的模型，避免权重重复加载
    slicer = SlicedDetector(config, model=detector.model)
    # 切片参数走 config 快照：行列数/重叠率/NMS 阈值与配置一致
    assert slicer.rows == config.slice_rows == 2
    assert slicer.cols == config.slice_cols == 2
    assert abs(slicer.overlap_ratio - config.slice_overlap_ratio) < 1e-9
    assert abs(slicer.nms_iou - config.slice_nms_iou) < 1e-9

    whole = detector.detect(image)
    sliced = slicer.detect(image)
    logger.info(
        "整图模式检出 %d 个：%s",
        len(whole), [(d["cls"], d["conf"]) for d in whole],
    )
    logger.info(
        "切片模式检出 %d 个：%s",
        len(sliced), [(d["cls"], d["conf"], d["slice"]) for d in sliced],
    )

    # 核心断言 1：切片模式检出数 >= 整图模式检出数
    assert len(sliced) >= len(whole), \
        "切片模式检出数(%d)应 >= 整图模式(%d)" % (len(sliced), len(whole))
    assert len(sliced) > 0, "测试图上切片模式应有检出（素材含4个高置信人）"

    # 核心断言 2：合并后无重复框——任意两框 IOU < 0.5
    max_pair_iou = 0.0
    for i in range(len(sliced)):
        for j in range(i + 1, len(sliced)):
            iou = iou_of(sliced[i]["box"], sliced[j]["box"])
            max_pair_iou = max(max_pair_iou, iou)
            assert iou < 0.5, \
                "合并后仍存在重复框：%s 与 %s IOU=%.4f" % (
                    sliced[i], sliced[j], iou)

    # 接口对齐：返回项含 Detector 契约的四个键，坐标均在画面内，
    # center 与 box 一致；附带的 slice 字段为合法切片序号
    n_regions = len(compute_slice_regions(
        frame_w, frame_h, slicer.rows, slicer.cols, slicer.overlap_ratio))
    for det in sliced:
        for key in ("cls", "conf", "box", "center"):
            assert key in det, "返回项缺少契约键 %s" % key
        x1, y1, x2, y2 = det["box"]
        assert 0 <= x1 < x2 <= frame_w and 0 <= y1 < y2 <= frame_h, \
            "检出框越出画面: %s" % det
        cx, cy = det["center"]
        assert abs(cx - (x1 + x2) / 2) <= 1 and abs(cy - (y1 + y2) / 2) <= 1
        assert 0 <= det["slice"] < n_regions, "切片序号越界: %s" % det

    logger.info(
        "用例4 PASS：切片 %d 个 >= 整图 %d 个；合并后任意两框 "
        "IOU<0.5（最大 pair IOU=%.4f）；接口与 Detector.detect() 对齐",
        len(sliced), len(whole), max_pair_iou,
    )


def main():
    logger.info("==== A12 切片推理模块 验收测试开始 ====")
    test_slice_regions()
    test_coordinate_mapping()
    test_cross_boundary_merge()
    test_real_image_slice_vs_whole()
    logger.info("==== 全部 4 组用例断言通过 ====")
    print("PASS: tests/test_slicer.py 全部断言通过")


if __name__ == "__main__":
    main()
