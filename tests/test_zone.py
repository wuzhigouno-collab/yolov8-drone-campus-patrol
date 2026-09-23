# -*- coding: utf-8 -*-
"""鹰眼巡校——A4 分区域差异化阈值模块 验收测试。

测试内容（全部使用合成检测数据，结构与 Detector 输出对齐）：
    1. 区域划分几何正确性：中央区矩形居中对称、比例符合配置；
    2. 锚点定义：区域判定取检测框底部中心点；
    3. 核心验收场景：边缘区置信度 0.35 的目标
       - 边缘阈值 0.5 时被过滤；
       - 边缘阈值 0.3 时被保留；
       - 两种边缘阈值下，中央区目标的判定结果完全一致
         （中央区判定不受边缘配置影响）；
    4. 过滤结果携带每个目标的区域/阈值/原因记录（供断言与日志证据）。

运行方式：
    python tests/test_zone.py
全部断言通过时输出 PASS 并以退出码 0 结束；失败时抛出 AssertionError。
"""

import os
import sys

# 本文件位于 tests/ 下，直接运行时 sys.path 不含项目根目录，手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.logger import get_logger
from src.zone_threshold import (
    ZONE_CENTER,
    ZONE_EDGE,
    ZoneThresholdFilter,
    anchor_point,
    calc_center_zone,
)

logger = get_logger("test_zone")


def make_det(cls, conf, box):
    """构造与 Detector 输出结构对齐的合成检测项 {cls, conf, box, center}。"""
    x1, y1, x2, y2 = box
    return {
        "cls": cls,
        "conf": conf,
        "box": list(box),
        "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
    }


def build_config(edge_threshold):
    """构造测试用配置：固定中央区阈值，边缘阈值由参数指定。"""
    config = Config()
    config.center_zone_ratio_w = 0.5
    config.center_zone_ratio_h = 0.5
    config.center_conf_threshold = 0.40
    config.edge_conf_threshold = edge_threshold
    return config


# 测试画面尺寸 1000x800：中央区比例 0.5 -> 中央区矩形 (250,200)-(750,600)
FRAME_W, FRAME_H = 1000, 800

# 合成检测集：
#   det_edge_low    边缘区（底部中心 (80,160)，x<250），conf 0.35
#   det_center_mid  中央区（底部中心 (480,430)），conf 0.45
#   det_edge_high   边缘区（底部中心 (900,700)），conf 0.60
DETECTIONS = [
    make_det("person", 0.35, [40, 40, 120, 160]),
    make_det("person", 0.45, [430, 330, 530, 430]),
    make_det("car", 0.60, [860, 620, 940, 700]),
]


def test_zone_geometry():
    """用例 1：中央区矩形几何正确性（居中、对称、比例正确）。"""
    rect = calc_center_zone(FRAME_W, FRAME_H, 0.5, 0.5)
    assert rect == (250, 200, 750, 600), "中央区矩形计算错误: %s" % (rect,)
    # 非 0.5 比例：0.4 x 0.25 -> (300,300)-(700,500)
    rect2 = calc_center_zone(FRAME_W, FRAME_H, 0.4, 0.25)
    assert rect2 == (300, 300, 700, 500), "中央区矩形计算错误: %s" % (rect2,)
    logger.info("用例1 PASS：中央区矩形 %s / %s 居中对称、比例正确", rect, rect2)


def test_anchor_point():
    """用例 2：区域判定锚点为检测框底部中心点 ((x1+x2)/2, y2)。"""
    det = make_det("person", 0.9, [100, 200, 300, 400])
    px, py = anchor_point(det)
    assert (px, py) == (200.0, 400.0), "锚点应为底部中心: %s" % ((px, py),)
    logger.info("用例2 PASS：锚点为底部中心 ((%s, %s))", px, py)


def run_scenario(edge_threshold):
    """按指定边缘阈值执行过滤，返回 {目标键: 判定记录}。"""
    config = build_config(edge_threshold)
    ztf = ZoneThresholdFilter(config)
    result = ztf.filter(DETECTIONS, FRAME_W, FRAME_H)
    logger.info("---- 场景：edge_conf_threshold=%.2f ----", edge_threshold)
    for rec in result.records:
        logger.info(
            "判定记录 | %s conf=%.2f | 区域=%s | 阈值=%.2f | %s | %s",
            rec["det"]["cls"], rec["det"]["conf"], rec["zone"],
            rec["threshold"], "保留" if rec["kept"] else "过滤", rec["reason"],
        )
    # 按 conf 值索引记录，便于断言（本测试各目标 conf 互不相同）
    return {rec["det"]["conf"]: rec for rec in result.records}, result


def test_zone_threshold_scenario():
    """用例 3（核心验收）：边缘目标随边缘阈值变化，中央目标判定不变。"""
    # --- 场景 A：边缘阈值 0.50 ---
    recs_a, result_a = run_scenario(0.50)
    # 边缘区 conf=0.35 目标应被过滤（0.35 < 0.50）
    assert recs_a[0.35]["zone"] == ZONE_EDGE
    assert recs_a[0.35]["kept"] is False, "边缘阈值0.5时 0.35 目标应被过滤"
    assert recs_a[0.35]["threshold"] == 0.50
    # 中央区 conf=0.45 目标应被保留（0.45 >= 中央阈值 0.40）
    assert recs_a[0.45]["zone"] == ZONE_CENTER
    assert recs_a[0.45]["kept"] is True, "中央区 0.45 目标应被保留"
    assert recs_a[0.45]["threshold"] == 0.40
    # 边缘区 conf=0.60 目标应被保留
    assert recs_a[0.60]["kept"] is True
    assert len(result_a.kept) == 2 and len(result_a.filtered) == 1

    # --- 场景 B：边缘阈值 0.30 ---
    recs_b, result_b = run_scenario(0.30)
    # 边缘区 conf=0.35 目标应被保留（0.35 >= 0.30）
    assert recs_b[0.35]["zone"] == ZONE_EDGE
    assert recs_b[0.35]["kept"] is True, "边缘阈值0.3时 0.35 目标应被保留"
    assert recs_b[0.35]["threshold"] == 0.30
    assert len(result_b.kept) == 3 and len(result_b.filtered) == 0

    # --- 关键不变量：中央区判定结果不受边缘配置影响 ---
    assert recs_a[0.45]["kept"] == recs_b[0.45]["kept"] is True
    assert recs_a[0.45]["zone"] == recs_b[0.45]["zone"] == ZONE_CENTER
    assert recs_a[0.45]["threshold"] == recs_b[0.45]["threshold"] == 0.40
    assert recs_a[0.45]["reason"] == recs_b[0.45]["reason"], \
        "边缘阈值变化不应影响中央区目标的判定原因"
    logger.info(
        "用例3 PASS：边缘 0.35 目标 阈值0.5->过滤 / 阈值0.3->保留；"
        "中央区 0.45 目标两场景判定一致（均保留，阈值0.40，原因相同）",
    )


def test_boundary_and_reason():
    """用例 4：分界线上的点算中央区（边界含入）；原因字段非空且含区域名。"""
    config = build_config(0.50)
    ztf = ZoneThresholdFilter(config)
    # 底部中心恰好落在中央区左边界 x=250 上的目标
    det = make_det("person", 0.45, [230, 300, 270, 400])  # 锚点 (250,400)
    result = ztf.filter([det], FRAME_W, FRAME_H)
    rec = result.records[0]
    assert rec["zone"] == ZONE_CENTER, "分界线上的点应判为中央区"
    assert rec["kept"] is True
    assert "中央区" in rec["reason"] and "0.40" in rec["reason"]
    logger.info("用例4 PASS：边界点判为中央区，原因字段=%s", rec["reason"])


def main():
    logger.info("==== A4 分区域差异化阈值模块 验收测试开始 ====")
    test_zone_geometry()
    test_anchor_point()
    test_zone_threshold_scenario()
    test_boundary_and_reason()
    logger.info("==== 全部 4 组用例断言通过 ====")
    print("PASS: tests/test_zone.py 全部断言通过")


if __name__ == "__main__":
    main()
