# -*- coding: utf-8 -*-
"""鹰眼巡校——A5 多帧 IOU 确认降噪模块 验收测试。

测试内容（全部使用合成检测序列，结构与 Detector 输出对齐）：
    1. IOU 计算正确性：同框为 1、不相交为 0、已知几何关系取理论值；
    2. 单帧闪现目标：只出现 1 帧的目标 0 条确认记录（0 条告警）；
    3. 连续稳定目标：连续出现 3 帧的目标在第 3 帧产生确认记录，
       第 4 帧起保持已确认但不再重复产生确认事件；
    4. IOU 阈值可配置且生效（对照断言）：同一组相邻帧 IOU≈0.43 的
       序列，阈值 0.30 时第 3 帧确认、阈值 0.50 时全程不确认；
    5. 断档重计：连续 2 帧后漏 1 帧，再次出现需重新累计 3 帧才确认；
    6. 类别感知匹配：同位置不同类别（人->车）不视为同一目标；
    7. 帧号不连续（跳帧/重新开播）时候选清空重计。

运行方式：
    python tests/test_confirm.py
全部断言通过时输出 PASS 并以退出码 0 结束；失败时抛出 AssertionError。
"""

import os
import sys

# 本文件位于 tests/ 下，直接运行时 sys.path 不含项目根目录，手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.frame_confirm import (
    BACKEND_IOU_FALLBACK,
    FrameConfirmer,
    iou_of,
)
from src.logger import get_logger

logger = get_logger("test_confirm")


def make_det(cls, conf, box):
    """构造与 Detector 输出结构对齐的合成检测项 {cls, conf, box, center}。"""
    x1, y1, x2, y2 = box
    return {
        "cls": cls,
        "conf": conf,
        "box": list(box),
        "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
    }


def build_config(confirm_frames=3, iou_threshold=0.30):
    """构造测试用配置：确认帧数与 IOU 阈值由参数指定。"""
    config = Config()
    config.confirm_frames = confirm_frames
    config.iou_threshold = iou_threshold
    return config


def feed_sequence(confirmer, sequences, start_frame=1):
    """把逐帧检测序列依次喂给确认器，返回每帧的 ConfirmResult 列表。"""
    results = []
    for offset, dets in enumerate(sequences):
        results.append(confirmer.update(dets, start_frame + offset))
    return results


def test_iou_of():
    """用例 1：IOU 计算正确性。"""
    # 完全相同的框 IOU = 1
    assert abs(iou_of([0, 0, 100, 100], [0, 0, 100, 100]) - 1.0) < 1e-6
    # 不相交的框 IOU = 0
    assert iou_of([0, 0, 100, 100], [200, 200, 300, 300]) == 0.0
    # 水平平移 40px 的 100x100 框：交 60*100，并 2*10000-6000
    # IOU = 6000/14000 ≈ 0.4286（用例 4 对照序列的几何基础）
    iou = iou_of([0, 0, 100, 100], [40, 0, 140, 100])
    assert abs(iou - 6000.0 / 14000.0) < 1e-6, "IOU 计算错误: %s" % iou
    logger.info("用例1 PASS：IOU 同框=1、不相交=0、平移40px≈%.4f", iou)


def test_flash_target_no_confirm():
    """用例 2（核心验收）：单帧闪现目标 0 条确认记录（0 条告警）。"""
    confirmer = FrameConfirmer(build_config())
    flash = make_det("person", 0.60, [300, 300, 380, 430])
    # 闪现目标只在第 2 帧出现一次，前后帧均无目标
    results = feed_sequence(confirmer, [[], [flash], [], []])
    # 全程没有任何已确认目标 -> 可进入告警的目标数为 0
    total_confirmed = sum(len(r.confirmed) for r in results)
    assert total_confirmed == 0, "单帧闪现目标不应产生任何已确认目标"
    assert len(confirmer.confirmed_events) == 0, "单帧闪现目标不应产生确认记录"
    # 闪现那一帧的状态记录：新建候选、连续 1 帧、未确认
    rec = results[1].records[0]
    assert rec["consecutive"] == 1 and rec["confirmed"] is False
    assert rec["just_confirmed"] is False and rec["confirm_frame"] is None
    logger.info(
        "用例2 PASS：单帧闪现目标 0 条确认记录（出现帧状态：%s）",
        rec["reason"],
    )


def test_stable_target_confirmed_at_frame3():
    """用例 3（核心验收）：连续 3 帧稳定出现的目标在第 3 帧确认。"""
    confirmer = FrameConfirmer(build_config(confirm_frames=3,
                                            iou_threshold=0.30))
    # 带轻微抖动的稳定目标（相邻帧 IOU 远高于阈值）
    boxes = [
        [100, 100, 200, 220],
        [102, 101, 202, 221],
        [100, 100, 200, 220],
        [101, 102, 201, 222],
    ]
    results = feed_sequence(
        confirmer, [[make_det("person", 0.80, b)] for b in boxes])

    # 第 1、2 帧：未确认，连续帧数 1、2，可告警目标数为 0
    assert results[0].records[0]["consecutive"] == 1
    assert results[0].records[0]["confirmed"] is False
    assert len(results[0].confirmed) == 0
    assert results[1].records[0]["consecutive"] == 2
    assert results[1].records[0]["confirmed"] is False
    assert len(results[1].confirmed) == 0

    # 第 3 帧：产生确认记录（确认事件帧 == 3，连续帧数 == 3）
    rec3 = results[2].records[0]
    assert rec3["confirmed"] is True and rec3["just_confirmed"] is True
    assert rec3["confirm_frame"] == 3, "确认帧号应为 3: %s" % rec3
    assert rec3["consecutive"] == 3
    assert len(results[2].confirmed) == 1, "第 3 帧起目标可进入告警"
    # track_id 跨帧保持一致（同一候选目标接续）
    assert results[0].records[0]["track_id"] == rec3["track_id"]

    # 第 4 帧：保持已确认，但不重复产生确认事件
    rec4 = results[3].records[0]
    assert rec4["confirmed"] is True and rec4["just_confirmed"] is False
    assert rec4["consecutive"] == 4 and rec4["confirm_frame"] == 3
    assert len(confirmer.confirmed_events) == 1
    assert confirmer.confirmed_events[0]["frame_no"] == 3
    logger.info(
        "用例3 PASS：稳定目标第 3 帧确认（确认帧号=%d，track_id=%d，"
        "第 4 帧保持已确认不重复确认）", rec3["confirm_frame"], rec3["track_id"],
    )


def test_iou_threshold_configurable():
    """用例 4（核心验收对照）：IOU 阈值可配置且生效。

    同一合成序列：目标每帧水平平移 40px，相邻帧 IOU≈0.4286。
    阈值 0.30 时相邻帧可匹配 -> 第 3 帧确认；
    阈值 0.50 时相邻帧无法匹配 -> 全程 0 条确认记录。
    """
    shifting = [
        [make_det("car", 0.70, [0, 100, 100, 200])],
        [make_det("car", 0.70, [40, 100, 140, 200])],
        [make_det("car", 0.70, [80, 100, 180, 200])],
    ]
    # 场景 A：阈值 0.30（< 0.4286，匹配成功）
    conf_a = FrameConfirmer(build_config(iou_threshold=0.30))
    res_a = feed_sequence(conf_a, shifting)
    assert res_a[2].records[0]["confirmed"] is True
    assert res_a[2].records[0]["confirm_frame"] == 3
    assert len(conf_a.confirmed_events) == 1
    # 匹配记录中应能查到相邻帧 IOU（≈0.4286）
    assert abs(res_a[1].records[0]["iou"] - 0.4286) < 0.001

    # 场景 B：阈值 0.50（> 0.4286，匹配失败，每帧都新建候选）
    conf_b = FrameConfirmer(build_config(iou_threshold=0.50))
    res_b = feed_sequence(conf_b, shifting)
    assert all(len(r.confirmed) == 0 for r in res_b)
    assert len(conf_b.confirmed_events) == 0, "阈值 0.50 时不应产生确认记录"
    assert all(r.records[0]["consecutive"] == 1 for r in res_b)
    logger.info(
        "用例4 PASS：同一 IOU≈0.43 序列，阈值0.30 第3帧确认 / "
        "阈值0.50 全程未确认（阈值配置生效）",
    )


def test_interrupted_streak_restarts():
    """用例 5：连续 2 帧后漏 1 帧，再次出现需重新累计满 3 帧才确认。"""
    confirmer = FrameConfirmer(build_config(confirm_frames=3))
    det = make_det("person", 0.75, [100, 100, 200, 220])
    # 帧 1、2 出现，帧 3 漏检，帧 4、5、6 重新连续出现
    results = feed_sequence(confirmer, [[det], [det], [], [det], [det], [det]])
    assert results[1].records[0]["consecutive"] == 2
    # 断档后再次出现：从 1 重新计，且是新的内部目标号
    rec4 = results[3].records[0]
    assert rec4["consecutive"] == 1 and rec4["confirmed"] is False
    assert rec4["track_id"] != results[1].records[0]["track_id"]
    # 帧 5 仍未确认（连续 2 帧 < 3）
    assert results[4].records[0]["consecutive"] == 2
    assert results[4].records[0]["confirmed"] is False
    # 帧 6 连续满 3 帧：确认帧号为 6
    rec6 = results[5].records[0]
    assert rec6["confirmed"] is True and rec6["confirm_frame"] == 6
    assert len(confirmer.confirmed_events) == 1
    logger.info(
        "用例5 PASS：断档重计——漏检后再次出现，重新累计至第 6 帧才确认",
    )


def test_class_aware_matching():
    """用例 6：类别感知匹配——同位置不同类别不视为同一目标。"""
    confirmer = FrameConfirmer(build_config())
    person = make_det("person", 0.80, [100, 100, 200, 220])
    car_same_box = make_det("car", 0.80, [100, 100, 200, 220])
    # 帧 1 人、帧 2 同位置车：IOU=1 但类别不同，不允许接续
    results = feed_sequence(confirmer, [[person], [car_same_box]])
    rec2 = results[1].records[0]
    assert rec2["consecutive"] == 1, "类别变化不应接续连续帧数"
    assert rec2["iou"] is None, "类别不同不应发生 IOU 匹配"
    assert rec2["track_id"] != results[0].records[0]["track_id"]
    logger.info("用例6 PASS：同位置 人->车 不匹配，各自独立计数")


def test_frame_gap_resets():
    """用例 7：帧号不连续时候选清空重计；确认帧数 N 走配置。"""
    # 确认帧数 N 可配置：N=2 时第 2 帧即确认
    conf2 = FrameConfirmer(build_config(confirm_frames=2))
    det = make_det("person", 0.75, [100, 100, 200, 220])
    res = feed_sequence(conf2, [[det], [det]])
    assert res[1].records[0]["confirmed"] is True
    assert res[1].records[0]["confirm_frame"] == 2

    # 帧号跳变：帧 1、2 后跳到帧 10，此前候选作废
    confirmer = FrameConfirmer(build_config(confirm_frames=3))
    confirmer.update([det], 1)
    confirmer.update([det], 2)
    result = confirmer.update([det], 10)  # 2 -> 10 不连续
    rec = result.records[0]
    assert rec["consecutive"] == 1 and rec["confirmed"] is False
    # 降级路径标识：供 A11 跟踪版确认实现区分后端
    assert result.backend == BACKEND_IOU_FALLBACK
    logger.info(
        "用例7 PASS：N=2 第 2 帧确认（N 走配置）；帧号 2->10 跳变后重计",
    )


def main():
    logger.info("==== A5 多帧 IOU 确认降噪模块 验收测试开始 ====")
    test_iou_of()
    test_flash_target_no_confirm()
    test_stable_target_confirmed_at_frame3()
    test_iou_threshold_configurable()
    test_interrupted_streak_restarts()
    test_class_aware_matching()
    test_frame_gap_resets()
    logger.info("==== 全部 7 组用例断言通过 ====")
    print("PASS: tests/test_confirm.py 全部断言通过")


if __name__ == "__main__":
    main()
