# -*- coding: utf-8 -*-
"""鹰眼巡校——无人机校园安全智能巡检系统 V1.0

多帧 IOU 确认降噪模块（A5）。

功能意图：
    单帧检测结果中常混入只出现一两帧的抖动误检（反光、背景干扰等），
    直接据此告警会产生大量虚警。本模块在"无跟踪器"条件下提供降级
    确认路径：相邻帧之间按 IOU（交并比）把检测框匹配为同一候选目标，
    候选目标连续 N 帧（默认 3 帧，走 Config.confirm_frames）稳定出现
    才标记为"已确认"；未确认目标不进入告警与计数，从而压制瞬时误检。

    确认结果保留每个目标的完整状态记录（内部目标号、已连续帧数、
    是否已确认、确认帧号、文字原因），供 GUI 渲染、日志输出与测试
    断言使用。输入结构与 src.detector.Detector 输出结构完全对齐：
    [{cls, conf, box, center}, ...]，box 为 xyxy 像素坐标。

接口契约（为 A11 预留）：
    A11 引入 ByteTrack 跟踪器后，确认逻辑将升级为基于轨迹生命周期
    （轨迹连续存在 N 帧才确认）。届时跟踪版实现与本类保持同一对外
    契约——update(detections, frame_no) -> ConfirmResult，ConfirmResult
    的 records / confirmed / just_confirmed 结构不变，GUI 与下游模块
    无需改动；本 IOU 匹配实现保留为无跟踪器时的降级路径，
    通过 backend 属性区分当前生效的确认后端。
"""

from src.config import Config
from src.logger import get_logger

# 确认后端标识：本模块为无跟踪器时的 IOU 降级路径
BACKEND_IOU_FALLBACK = "iou-fallback"


def iou_of(box_a, box_b):
    """计算两个 xyxy 框的交并比（IOU）。

    参数：
        box_a / box_b: [x1, y1, x2, y2] 像素坐标。

    返回：
        0~1 浮点数；两框不相交时返回 0.0。
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    # 交集区域坐标（无交集时宽/高会被钳到 0）
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class ConfirmResult:
    """单帧多帧确认结果。

    属性：
        records: 逐目标确认状态记录列表，每项为字典：
            det           —— 原检测项字典（{cls, conf, box, center}）
            track_id      —— 内部候选目标号（降级路径内自增，仅帧间匹配用）
            consecutive   —— 截至本帧已连续出现的帧数
            confirmed     —— 当前是否已确认（连续帧数达到 N 后为 True）
            just_confirmed —— 本帧是否是"刚确认"的那一帧（确认事件帧）
            confirm_frame —— 确认发生的帧号，未确认为 None
            iou           —— 本帧与上一帧匹配框的 IOU（新建候选为 None）
            reason        —— 文字原因说明（供日志与测试断言）
        frame_no: 本帧帧号
        backend:  确认后端标识（A11 跟踪版实现将使用各自标识）

    下游约定：只有 confirmed 为 True 的目标才允许进入告警与计数。
    """

    def __init__(self, records, frame_no, backend=BACKEND_IOU_FALLBACK):
        self.records = records
        self.frame_no = frame_no
        self.backend = backend

    @property
    def confirmed(self):
        """已确认的检测项列表（保持原输入顺序）。"""
        return [r["det"] for r in self.records if r["confirmed"]]

    @property
    def unconfirmed(self):
        """尚未确认的检测项列表（保持原输入顺序）。"""
        return [r["det"] for r in self.records if not r["confirmed"]]

    @property
    def just_confirmed(self):
        """本帧刚产生确认事件的记录列表（确认帧号 == 本帧帧号）。"""
        return [r for r in self.records if r["just_confirmed"]]

    def __len__(self):
        return len(self.records)


class FrameConfirmer:
    """多帧 IOU 确认器（无跟踪器时的降级路径）。

    职责：
        1. 维护一组"候选目标"（上一帧仍未断档的检测框），相邻帧间
           按 IOU 贪心匹配（同类别、IOU >= Config.iou_threshold）；
        2. 匹配成功的候选连续帧数 +1；达到 Config.confirm_frames 的
           候选在该帧标记为已确认并记录确认帧号；
        3. 未匹配上的检测框新建候选（连续帧数从 1 计起）；本帧未
           匹配上的候选视为断档直接丢弃——"连续 N 帧"要求严格相邻，
           中间漏一帧即重新计数；
        4. 帧号不连续（跳帧/重新开播）时清空全部候选，避免把两段
           不相邻的检测错误接续为同一目标。

    本类不修改 Config，参数在构造时从 Config 读取快照；运行期改
    配置后重新构造即可生效。A11 跟踪版确认实现将复用 ConfirmResult
    输出契约，本类保留为降级路径（backend 属性可用于区分）。
    """

    def __init__(self, config=None):
        """初始化确认器。

        参数：
            config: Config 实例；为 None 时内部新建默认配置。
        """
        self.config = config if config is not None else Config()
        self.logger = get_logger(__name__, self.config)
        # 确认帧数 N 与 IOU 匹配阈值（构造时取配置快照）
        self.confirm_frames = int(self.config.confirm_frames)
        self.iou_threshold = float(self.config.iou_threshold)
        # 确认后端标识：供上层区分当前生效的是 IOU 降级路径还是跟踪版
        self.backend = BACKEND_IOU_FALLBACK
        # 候选目标列表，每项：{id, cls, box, det, consecutive,
        #                       confirmed, confirm_frame, last_frame}
        self._candidates = []
        self._next_id = 1
        self._last_frame_no = None
        # 累计确认事件列表：{track_id, cls, frame_no}，供统计与测试断言
        self.confirmed_events = []

    def reset(self):
        """清空全部候选目标与帧状态（切换视频源/重新开播时调用）。"""
        self._candidates = []
        self._next_id = 1
        self._last_frame_no = None
        self.confirmed_events = []

    def update(self, detections, frame_no):
        """输入一帧检测列表，执行相邻帧 IOU 匹配并更新确认状态。

        参数：
            detections: Detector 输出的检测列表 [{cls, conf, box, center}, ...]
            frame_no:   当前帧号（应逐帧连续递增；不连续时自动清空重计）。

        返回：
            ConfirmResult，含每个输入检测的确认状态记录；
            result.confirmed 为本帧可进入告警与计数的已确认目标。
        """
        # 帧号不连续（跳帧、重新开播）："连续 N 帧"的前提被破坏，清空重计
        if self._last_frame_no is not None \
                and frame_no != self._last_frame_no + 1:
            self.logger.info(
                "帧号不连续（%d -> %d），多帧确认候选全部清空重计",
                self._last_frame_no, frame_no,
            )
            self._candidates = []
        self._last_frame_no = frame_no

        # 相邻帧 IOU 贪心匹配：仅同类别且 IOU 达标的候选-检测对参与
        match_of_det = self._match(detections)

        records = []
        used_candidate_idx = set()
        for di, det in enumerate(detections):
            if di in match_of_det:
                # 匹配成功：接续候选，连续帧数 +1
                ci, iou = match_of_det[di]
                cand = self._candidates[ci]
                used_candidate_idx.add(ci)
                cand["box"] = det["box"]
                cand["det"] = det
                cand["consecutive"] += 1
                cand["last_frame"] = frame_no
                just_confirmed = False
                if not cand["confirmed"] \
                        and cand["consecutive"] >= self.confirm_frames:
                    # 连续帧数首次达标：本帧为确认事件帧
                    cand["confirmed"] = True
                    cand["confirm_frame"] = frame_no
                    just_confirmed = True
                    self.confirmed_events.append({
                        "track_id": cand["id"],
                        "cls": cand["cls"],
                        "frame_no": frame_no,
                    })
                reason = "%s：目标#%d %s 连续%d帧（与上帧IOU=%.2f）%s" % (
                    "确认" if cand["confirmed"] else "待确认",
                    cand["id"], det["cls"], cand["consecutive"], iou,
                    "，本帧达标确认" if just_confirmed else "",
                )
                records.append({
                    "det": det,
                    "track_id": cand["id"],
                    "consecutive": cand["consecutive"],
                    "confirmed": cand["confirmed"],
                    "just_confirmed": just_confirmed,
                    "confirm_frame": cand["confirm_frame"],
                    "iou": round(iou, 4),
                    "reason": reason,
                })
            else:
                # 未匹配：新建候选目标，连续帧数从 1 计起
                cand = {
                    "id": self._next_id,
                    "cls": det["cls"],
                    "box": det["box"],
                    "det": det,
                    "consecutive": 1,
                    "confirmed": False,
                    "confirm_frame": None,
                    "last_frame": frame_no,
                }
                self._next_id += 1
                self._candidates.append(cand)
                used_candidate_idx.add(len(self._candidates) - 1)
                reason = "待确认：目标#%d %s 首次出现（连续1/%d帧）" % (
                    cand["id"], det["cls"], self.confirm_frames,
                )
                records.append({
                    "det": det,
                    "track_id": cand["id"],
                    "consecutive": 1,
                    "confirmed": False,
                    "just_confirmed": False,
                    "confirm_frame": None,
                    "iou": None,
                    "reason": reason,
                })

        # 本帧未匹配上的候选视为断档：连续出现的前提是严格相邻，
        # 漏一帧即丢弃，再次出现时按新目标重新计数
        dropped = [c for i, c in enumerate(self._candidates)
                   if i not in used_candidate_idx]
        for cand in dropped:
            self.logger.debug(
                "帧%d 候选目标#%d %s 断档丢弃（此前连续%d帧，%s）",
                frame_no, cand["id"], cand["cls"], cand["consecutive"],
                "已确认" if cand["confirmed"] else "未确认",
            )
        self._candidates = [c for i, c in enumerate(self._candidates)
                            if i in used_candidate_idx]

        return ConfirmResult(records, frame_no, backend=self.backend)

    def _match(self, detections):
        """相邻帧贪心 IOU 匹配。

        在现有候选目标与本帧检测之间，按"同类别且 IOU >= 阈值"生成
        候选配对，按 IOU 从大到小贪心选取（每个候选/每个检测最多
        配对一次），返回 {检测下标: (候选下标, IOU)}。
        """
        pairs = []
        for ci, cand in enumerate(self._candidates):
            for di, det in enumerate(detections):
                # 类别不同不配对：人框与车框即使重叠也不视为同一目标
                if cand["cls"] != det["cls"]:
                    continue
                iou = iou_of(cand["box"], det["box"])
                if iou >= self.iou_threshold:
                    pairs.append((iou, ci, di))
        # IOU 降序贪心：重叠最大的配对优先锁定
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        match_of_det = {}
        used_c, used_d = set(), set()
        for iou, ci, di in pairs:
            if ci in used_c or di in used_d:
                continue
            used_c.add(ci)
            used_d.add(di)
            match_of_det[di] = (ci, iou)
        return match_of_det
