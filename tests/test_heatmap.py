# -*- coding: utf-8 -*-
"""鹰眼巡校——A8 人流车流热力图模块 验收测试。

测试内容分两部分：

一、合成场景单元测试（不依赖视频/模型，直接驱动 HeatmapAccumulator，
    秒级完成）：
    1. 接口与累积口径：add_point/update 累积点数与坐标查询
       （get_points/point_count）正确；update() 只累积多帧确认结果中
       confirmed=True 的记录，且锚点取检测框底部中心（复用 A4
       zone_threshold.anchor_point，与区域判定/越线计数口径一致）；
    2. 核心验收：已知坐标点集中累积后，渲染图热区中心"热度亮度"
       （BGR 三通道最大值，理由见下）显著高于无点冷区，且冷区为纯黑；
    3. 空累积防护：尚无任何累积点时 render/export_png 抛出 ValueError，
       错误信息含"先累积"类提示；
    4. 底帧叠加：有底帧时无热区像素保持底帧原样、热区像素被伪彩混合；
       底帧尺寸与累积尺寸不一致时抛 ValueError；
    5. reset 清空与画面尺寸变化自动清空重计。

    "热度亮度"度量说明：渲染采用 JET 伪彩（蓝冷红热），灰度值
    （0.299R+0.587G+0.114B）对热度并非单调（纯红灰度反而低于青色），
    因此本测试以 BGR 三通道最大值作为亮度度量——JET 色图中任何
    热度>0 的颜色至少一个通道 >=128，而无热区纯黑为 0，该度量对
    热度严格单调，像素均值差断言基于此。

二、真实视频端到端验收（默认运行，可用 --video/--max-frames 覆盖）：
    检测 -> 多帧确认（A5）-> 热力累积（本模块）全链路逐帧运行真实视频，
    累积到 --target-points 个已确认锚点即收尾（上限 --max-frames 帧），
    然后分别黑底导出与叠加最后一帧底帧导出两张 PNG。断言：
      - 累积点数 >= 下限（素材前 800 帧实测约 140 点，下限取 60 留足
        余量；点数不足说明确认链路或素材异常）；
      - 导出文件真实存在、可解码、尺寸与视频画面一致；
      - "目标密集区域"取累积点中最密集点（半径 r 内邻居最多者）的
        (2r+1) 见方邻域，"无目标区域"取 3x3 网格计数为 0 的整格，
        黑底导出图上密集区亮度均值显著高于无目标区（均值差 > 50：
        无热区像素纯黑接近 0，密集区中心累积值接近最大值、归一化
        后接近满量程 255，通道最大值均值必然远超该阈值）；
      - 叠加底帧导出图上无热区域保持底帧原样（冷区像素与底帧一致）。
    导出的两张 PNG 保留在 --out-dir（默认 tests/a8_heatmap）供人工目检。

运行方式：
    python tests/test_heatmap.py
        # 合成单测 + 默认素材真实视频端到端
    python tests/test_heatmap.py --video <视频路径> --max-frames 800 \
        --target-points 80
    python tests/test_heatmap.py --skip-video   # 只跑合成单测
全部断言通过输出 PASS 并以退出码 0 结束；断言失败抛 AssertionError。
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np

# 本文件位于 tests/ 下，直接运行时 sys.path 不含项目根目录，手动补上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.frame_confirm import ConfirmResult  # noqa: E402
from src.heatmap import HeatmapAccumulator  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.zone_threshold import anchor_point  # noqa: E402

logger = get_logger("test_heatmap")

# 默认真实视频素材（派活指定；3712 帧，前 800 帧实测约 140 个已确认锚点）
DEFAULT_VIDEO = "C:/Users/OS/Desktop/farm_guard/videos/5G智慧农业素材.mp4"
# 测试导出目录：与运行期 archive/ 隔离（沿用 test_archive 的隔离约定）
DEFAULT_OUT_DIR = "tests/a8_heatmap"


def brightness(img):
    """热度亮度图：BGR 三通道最大值（对 JET 热度单调，见模块 docstring）。"""
    return img.max(axis=2)


def make_det(x1, y1, x2, y2, cls="person", conf=0.8):
    """构造一个与 Detector 输出结构一致的检测项字典。"""
    return {"cls": cls, "conf": conf, "box": [x1, y1, x2, y2],
            "center": ((x1 + x2) // 2, (y1 + y2) // 2)}


def make_record(det, confirmed):
    """构造一个 ConfirmResult.records 记录（仅热力累积关心的字段）。"""
    return {"det": det, "track_id": 1, "consecutive": 3 if confirmed else 1,
            "confirmed": confirmed, "just_confirmed": False,
            "confirm_frame": 3 if confirmed else None, "iou": None,
            "reason": "测试构造"}


def run_unit_tests():
    """执行全部合成场景单元测试。"""
    logger.info("==== A8 热力图模块 合成场景单元测试开始 ====")
    frame_w, frame_h = 640, 480

    # ---- 用例 1：接口与累积口径（只收已确认、锚点=底部中心） ----
    acc = HeatmapAccumulator(Config())
    # update()：confirmed 与 unconfirmed 混合，只有 confirmed 参与累积
    det_ok = make_det(100, 200, 140, 300)   # 锚点应为 (120, 300)
    det_no = make_det(400, 100, 440, 200)   # 未确认，不应累积
    result = ConfirmResult(
        [make_record(det_ok, True), make_record(det_no, False)], 1)
    added = acc.update(result, frame_w, frame_h)
    assert added == 1, "只应累积 1 个已确认目标，实际 %d" % added
    assert acc.point_count == 1, "point_count 应为 1：%d" % acc.point_count
    expect_anchor = anchor_point(det_ok)
    assert acc.get_points() == [(int(expect_anchor[0]),
                                 int(expect_anchor[1]))], \
        "累积点应为底部中心锚点 %s，实际 %s" % (
            expect_anchor, acc.get_points())
    assert acc.frame_size == (frame_w, frame_h)
    # update(None)（多帧确认关闭场景）不累积；add_point 直接累积
    assert acc.update(None, frame_w, frame_h) == 0
    acc.add_point(320.4, 240.6, frame_w, frame_h)
    assert acc.point_count == 2 and acc.get_points()[-1] == (320, 241)
    logger.info("用例1 PASS：update 只收已确认目标，锚点=底部中心 %s，"
                "add_point/查询接口正确", expect_anchor)

    # ---- 用例 2：已知点集中累积后，热区亮度显著高于冷区 ----
    acc2 = HeatmapAccumulator(Config())
    # 30 个点集中在 (320, 240) 附近小范围抖动（模拟同一位置反复出现目标）
    rng = np.random.RandomState(42)
    for _ in range(30):
        acc2.add_point(320 + rng.randint(-5, 6), 240 + rng.randint(-5, 6),
                       frame_w, frame_h)
    heat = acc2.render()  # 黑底渲染
    assert heat.shape == (frame_h, frame_w, 3)
    br = brightness(heat)
    r = acc2.kernel_radius
    hot_mean = float(br[240 - r:240 + r + 1, 320 - r:320 + r + 1].mean())
    # 冷区：远离任何累积点的角落（50,50），距热区中心远超核半径
    cold_mean = float(br[20:80, 20:80].mean())
    assert cold_mean == 0.0, "无热区应为纯黑（亮度 0），实际 %.2f" % cold_mean
    # 30 点同址叠加远超归一化饱和点，热区中心应为最热色（通道最大=255）
    assert hot_mean >= 200, "热区中心亮度应接近满量程，实际 %.2f" % hot_mean
    assert hot_mean - cold_mean > 150, \
        "热区/冷区亮度均值差应 >150，实际 %.2f" % (hot_mean - cold_mean)
    logger.info("用例2 PASS：热区亮度 %.1f >> 冷区 %.1f（差 %.1f）",
                hot_mean, cold_mean, hot_mean - cold_mean)

    # ---- 用例 3：空累积防护 ----
    acc3 = HeatmapAccumulator(Config())
    for action in ("render", "export"):
        try:
            if action == "render":
                acc3.render()
            else:
                acc3.export_png("should_not_exist.png")
            raise AssertionError("空累积 %s 应抛 ValueError" % action)
        except ValueError as exc:
            assert "累积" in str(exc), "错误信息应说明累积为空：%s" % exc
    assert not os.path.exists("should_not_exist.png"), "空累积不应落盘文件"
    logger.info("用例3 PASS：空累积 render/export_png 均抛 ValueError 且不落盘")

    # ---- 用例 4：底帧叠加——无热区保持底帧原样、热区被混合、尺寸校验 ----
    acc4 = HeatmapAccumulator(Config())
    for _ in range(20):
        acc4.add_point(320, 240, frame_w, frame_h)
    base = np.full((frame_h, frame_w, 3), (60, 120, 180), dtype=np.uint8)
    over = acc4.render(base)
    # 无热区（角落）保持底帧原样
    assert (over[30, 30] == base[30, 30]).all(), "无热区像素应保持底帧原样"
    # 热区中心像素被伪彩混合，不再等于底帧
    assert not (over[240, 320] == base[240, 320]).all(), \
        "热区中心应被伪彩混合覆盖"
    # 底帧尺寸不一致抛 ValueError
    try:
        acc4.render(np.zeros((100, 100, 3), dtype=np.uint8))
        raise AssertionError("底帧尺寸不一致应抛 ValueError")
    except ValueError:
        pass
    logger.info("用例4 PASS：底帧叠加无热区原样保持、热区混合、尺寸校验生效")

    # ---- 用例 5：reset 清空 + 画面尺寸变化自动清空重计 ----
    acc5 = HeatmapAccumulator(Config())
    acc5.add_point(100, 100, frame_w, frame_h)
    acc5.reset()
    assert acc5.point_count == 0 and acc5.get_points() == []
    assert acc5.accum is None and acc5.frame_size is None
    # 尺寸变化：先以 640x480 累积，再换 320x240 累积，旧点应被清空
    acc5.add_point(100, 100, frame_w, frame_h)
    acc5.add_point(50, 50, 320, 240)
    assert acc5.point_count == 1 and acc5.get_points() == [(50, 50)], \
        "尺寸变化后应只剩新尺寸下累积的点：%s" % acc5.get_points()
    assert acc5.frame_size == (320, 240)
    logger.info("用例5 PASS：reset 清空、尺寸变化自动清空重计均正确")

    logger.info("==== 全部 5 组合成用例断言通过 ====")
    print("PASS: 合成场景单元测试 5 组用例全部通过")


def grid_counts(points, frame_w, frame_h):
    """把累积点按 3x3 网格计数，返回 (3,3) 计数数组。"""
    grid = np.zeros((3, 3), dtype=int)
    for x, y in points:
        grid[min(2, int(y / frame_h * 3)), min(2, int(x / frame_w * 3))] += 1
    return grid


def grid_region(gi, gj, frame_w, frame_h):
    """返回 3x3 网格中第 (gi, gj) 格的像素范围 (y1, y2, x1, x2)。"""
    return (gi * frame_h // 3, (gi + 1) * frame_h // 3,
            gj * frame_w // 3, (gj + 1) * frame_w // 3)


def run_video_mode(args):
    """真实视频端到端验收：检测->多帧确认->热力累积->导出与亮度断言。"""
    from src.detector import Detector
    from src.frame_confirm import FrameConfirmer
    from src.video_source import create_video_source

    config = Config()
    out_dir = os.path.abspath(args.out_dir)
    # 测试导出目录每次运行前清空重建，保证产出与当次运行一一对应
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    source = create_video_source(args.video, config)
    src_w, src_h = source.width, source.height  # 快照，release 后属性归零
    detector = Detector(config)
    confirmer = FrameConfirmer(config)
    heatmap = HeatmapAccumulator(config)
    logger.info(
        "真实视频端到端验收开始：%s（%dx%d，%.2f fps），目标累积点 %d，"
        "帧上限 %d", args.video, src_w, src_h, source.fps,
        args.target_points, args.max_frames)

    frame_no = 0
    last_frame = None
    while True:
        frame = source.read()
        if frame is None:
            break
        frame_no += 1
        last_frame = frame
        detections = detector.detect(frame)
        confirm_result = confirmer.update(detections, frame_no)
        heatmap.update(confirm_result, src_w, src_h)
        # 累积到目标点数即收尾，无需跑完全片
        if heatmap.point_count >= args.target_points:
            logger.info("已累积 %d 个已确认锚点（>= %d），于第 %d 帧收尾",
                        heatmap.point_count, args.target_points, frame_no)
            break
        if args.max_frames is not None and frame_no >= args.max_frames:
            logger.info("达到 --max-frames %d，结束（累积 %d 点）",
                        args.max_frames, heatmap.point_count)
            break
    source.release()

    # ---------------- 断言 ----------------
    # 累积足量已确认目标（素材前 800 帧实测约 140 点，下限 60 留余量）
    assert heatmap.point_count >= 60, \
        "跑满 %d 帧仅累积 %d 个已确认锚点（<60），确认链路或素材异常" % (
            frame_no, heatmap.point_count)

    # 黑底导出 + 叠加底帧导出（底帧用最后一帧原始画面，尺寸一致）
    black_path = os.path.join(out_dir, "heatmap_black.png")
    over_path = os.path.join(out_dir, "heatmap_overlay.png")
    rec_black = heatmap.export_png(black_path)
    rec_over = heatmap.export_png(over_path, base_frame=last_frame)
    for rec in (rec_black, rec_over):
        path = rec["path"]
        assert os.path.isfile(path) and os.path.getsize(path) > 0, \
            "导出文件不存在或为空：%s" % path
        img = cv2.imread(path)
        assert img is not None, "导出 PNG 无法解码：%s" % path
        assert img.shape == (src_h, src_w, 3), \
            "导出图尺寸应与视频画面一致：%s vs %dx%d" % (
                img.shape, src_w, src_h)
        assert rec["point_count"] == heatmap.point_count
        assert rec["frame_size"] == (src_w, src_h)

    # 目标密集区 vs 无目标区：
    # 密集区——对每个累积点统计核半径 r 内的邻居数，取邻居最多的点，
    # 以其 (2r+1) 见方邻域为密集区（该处累积值接近全图最大值）；
    # 无目标区——3x3 网格计数为 0 的整格（该素材顶行整行无目标）
    points = np.array(heatmap.get_points(), dtype=np.float32)
    grid = grid_counts(heatmap.get_points(), src_w, src_h)
    cold_idx = np.unravel_index(grid.argmin(), grid.shape)
    assert grid[cold_idx] == 0, \
        "本素材应存在无点空格作为冷区：\n%s" % grid
    r = heatmap.kernel_radius
    # 两两点距 <= r 视为邻居（向量化计算，百量级点数开销可忽略）
    dist = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    neighbor_counts = (dist <= r).sum(axis=1)
    hot_pt = points[int(neighbor_counts.argmax())]
    hx, hy = int(hot_pt[0]), int(hot_pt[1])
    hy1, hy2 = max(0, hy - r), min(src_h, hy + r + 1)
    hx1, hx2 = max(0, hx - r), min(src_w, hx + r + 1)
    cy1, cy2, cx1, cx2 = grid_region(*cold_idx, src_w, src_h)
    br = brightness(cv2.imread(black_path))
    hot_mean = float(br[hy1:hy2, hx1:hx2].mean())
    cold_mean = float(br[cy1:cy2, cx1:cx2].mean())
    # 阈值理由：黑底导出时无热区像素纯黑（亮度≈0，仅允许核边缘微量
    # 晕入）；密集区中心被反复踩到，累积值接近全图最大、归一化后接近
    # 满量程 255，且 JET 任何非零热度通道最大值 >=128。实测两者相差
    # 一个数量级以上，>50 的均值差是宽松而明确的"显著高于"判据
    assert cold_mean <= 20, \
        "无目标区亮度均值应接近纯黑（<=20），实际 %.2f" % cold_mean
    assert hot_mean - cold_mean > 50, \
        "密集区亮度应显著高于无目标区（差>50），实际 %.2f vs %.2f" % (
            hot_mean, cold_mean)
    logger.info(
        "亮度断言通过：密集区中心 (%d,%d)（半径%d内邻居%d个）亮度均值 "
        "%.1f；无目标格 (%d,%d) 亮度均值 %.1f；差 %.1f > 50",
        hx, hy, r, int(neighbor_counts.max()), hot_mean,
        cold_idx[0], cold_idx[1], cold_mean, hot_mean - cold_mean)

    # 叠加底帧导出图：无目标格（远离任何累积点）应保持底帧原样
    over_img = cv2.imread(over_path)
    diff = np.abs(over_img[cy1:cy2, cx1:cx2].astype(int)
                  - last_frame[cy1:cy2, cx1:cx2].astype(int)).max()
    assert diff <= 2, \
        "叠加导出图的无目标格应与底帧基本一致（允许 PNG 编码级误差），" \
        "实际最大差 %d" % diff

    summary = ("真实视频端到端验收通过：处理 %d 帧，累积 %d 个已确认锚点；"
               "密集格亮度 %.1f >> 无目标格 %.1f；导出 %s 与 %s" % (
                   frame_no, heatmap.point_count, hot_mean, cold_mean,
                   rec_black["path"], rec_over["path"]))
    print(summary, flush=True)
    logger.info(summary)


def parse_args():
    """解析命令行参数（默认直接跑合成单测 + 默认素材真实视频验收）。"""
    parser = argparse.ArgumentParser(
        description="A8 人流车流热力图模块 验收测试")
    parser.add_argument("--video", default=DEFAULT_VIDEO,
                        help="真实视频验收素材路径（默认派活指定素材）")
    parser.add_argument("--max-frames", type=int, default=800,
                        help="真实视频验收最多处理帧数（默认 800：该素材前"
                             " 800 帧实测约 140 个已确认锚点，足量）")
    parser.add_argument("--target-points", type=int, default=80,
                        help="累积到该点数即收尾（默认 80，低于 800 帧实测"
                             " 140 点，留足余量）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="测试导出目录（默认 tests/a8_heatmap，"
                             "与运行期 archive/ 隔离）")
    parser.add_argument("--skip-video", action="store_true",
                        help="只跑合成场景单元测试")
    return parser.parse_args()


def main():
    args = parse_args()
    run_unit_tests()
    if not args.skip_video:
        run_video_mode(args)
        print("PASS: tests/test_heatmap.py 合成单测与真实视频端到端验收全部通过")


if __name__ == "__main__":
    main()
