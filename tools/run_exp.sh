#!/usr/bin/env bash
# ============================================================================
# 鹰眼巡校——实验可复现与登记基础设施(tools/run_exp.sh)
#
# 依据 docs/实验计划.md:§4(实验矩阵与统一训练参数)、§6(实例环境与
# SSH/tmux 纪律)、§7(备份与登记方案)、§11(hw7 模板:每实验 = 一条
# run_exp.sh 命令 + 固定产出口径)。本脚本只做实验基础设施,不改 src/ 推理代码。
#
# ------------------------------ 使用纪律(§6) ------------------------------
# 1. 所有训练必须在 tmux 会话内执行(断线不丢任务):
#       tmux new -s exp
#       bash tools/run_exp.sh e0          # 脚本自动把命令全文/起止时间/退出码落 logs/
#       (Ctrl-b d 脱离;tmux a -t exp 重连)
#    非 tmux 环境发起正式训练时脚本会打印警告(不阻断,便于调试)。
# 2. 实例路径约定(AutoDL 容器,§6 实测):
#       工作区   /root/autodl-tmp/eagle_eye   (git clone 至此;本脚本按自身位置定位项目根)
#       数据集   /root/autodl-tmp/datasets    (yolo settings 已固化 datasets_dir)
#       训练输出  工作区内 runs/ 与 logs/
#    数据集用 ultralytics 内置 visdrone.yaml,首次训练自动下载并转 YOLO 格式
#    (§2,约 2.3GB);weights/ 与数据集不进 git(§12)。
# 3. batch 环境变量覆盖:统一默认 batch=32(§4);OOM 时人工减半重跑:
#       BATCH=16 bash tools/run_exp.sh e0
#    (yolov8s@960 建议 BATCH=16,见 §6 batch 建议;重跑前请先 rsync/删除旧 runs 目录)
# 4. 本机冒烟(离线、合成数据、分钟级,走 tools/train.py 既有 --smoke 机制):
#       SMOKE=1 bash tools/run_exp.sh e0
#    冒烟 runs 目录带 smoke_ 前缀,config_snapshot.yaml 中标注
#    "合成数据/无语义/精度无意义",严禁冒充真实训练结果。
#
# ------------------------------ 实验命令(§4 矩阵) ------------------------------
#   bash tools/run_exp.sh e0                          # E0 基线 -> runs/e0_baseline
#   bash tools/run_exp.sh e1a10                       # E1a degrees=10 -> runs/e1a_deg10
#   bash tools/run_exp.sh e1a30                       # E1a degrees=30 -> runs/e1a_deg30
#   bash tools/run_exp.sh e1b                         # E1b cutout=0.3 -> runs/e1b_cutout
#   bash tools/run_exp.sh e1c                         # E1c InvertImg(p=0.2) -> runs/e1c_invert
#   bash tools/run_exp.sh e1d                         # E1d copy_paste=0.1 -> runs/e1d_copypaste
#   bash tools/run_exp.sh e1e <deg> <cut> <inv> <cp>  # E1e 最优组合(参数见下)
#   bash tools/run_exp.sh e2  <deg> <cut> <inv> <cp>  # E2 yolov8s 参照(建议 BATCH=16)
#   bash tools/run_exp.sh e3 <weights.pt> [fp16文件] [int8文件]   # E3 轻量化测量
#   bash tools/run_exp.sh e4 <weights.pt>             # E4 FP16/INT8(TensorRT engine)导出 + 验证集 mAP 复测
#   bash tools/run_exp.sh e5 <weights.pt>             # E5 算子构成统计
#   bash tools/run_exp.sh e6 <weights.pt>             # E6 tools/evaluate.py 系统级评估
#   bash tools/run_exp.sh register <runs_dir> <实验编号> [备注]   # 登记(§7.4)
#
# E1e/E2 参数:<deg>=degrees 实际值(0/10/30),<cut>=0 或 0.3,<inv>=0 或 1,
#   <cp>=copy_paste 实际值(0/0.1)。**E1e 最优组合尚未实测出来**:组合参数必须
#   等 E1a–E1d 结果出来后由调用方显式传入,本脚本不预设任何"假装确定"的组合。
#
# E1b 实现说明(诚实口径):cutout 不是 ultralytics 8.4.21 的合法训练参数
#   (DEFAULT_CFG 无此键,yolo CLI 传 cutout= 会被直接拒绝)。§4 矩阵
#   cutout=0.3 的语义为"随机遮挡切块,概率 0.3",本脚本以 Albumentations
#   CoarseDropout(p=0.3) 等价实现,经补丁注入训练管线(见 train_combo 内嵌驱动)。
#
# E4 路线说明(R3 修正留痕):8.4.21 的 onnx 导出**不支持 int8**(export_formats
#   表中 ONNX 合法参数无 int8,validate_args() 直接 assert,R3 验证方实测
#   AssertionError/EXIT=1,本机复跑复现);E4 的 FP16/INT8 统一走 TensorRT engine
#   (format=engine,half/int8 均合法,本机 8.4.21 validate_args 源码核对 + CLI
#   干跑实测通过;INT8 为 TensorRT PTQ,校准集=验证集 548 张全量)。选 TensorRT
#   而非 openvino:实例 RTX 4090 原生路线,FP16/INT8 同后端,mAP 掉点为纯量化
#   效应。E4 需实例侧 GPU 与预装 tensorrt(Linux 上 ultralytics 会尝试自动装
#   tensorrt-cuXX,建议 setup_env.sh 预装);engine 绑定生成机 GPU/TensorRT 版本,
#   同机导出同机复测,禁止跨机拷贝。
#
# albumentations 为实例侧预装(§6 setup_env.sh),requirements.txt 不引入;
# tensorrt 同为实例侧预装依赖,亦不写入 requirements.txt。
# ============================================================================

set -uo pipefail

# ---------- 项目根定位:脚本在 tools/ 下,项目根为其上一级 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# ---------- 统一训练参数(docs/实验计划.md §4,除受试变量外全部实验一致) ----------
IMGSZ=960
EPOCHS=100
BATCH="${BATCH:-32}"            # 环境变量覆盖:OOM 时人工减半重跑(§6)
OPTIMIZER=auto
PATIENCE=20
SEED=0
# multi_scale=True / cos_lr=True / AMP 开启:在各训练调用处显式写死,不接受覆盖
DATA="${DATA:-visdrone.yaml}"   # ultralytics 内置,首次自动下载转换(§2)
MODEL_N="${MODEL_N:-yolov8n.pt}"   # COCO 预训练起步权重(§3);实例上首次自动下载
MODEL_S="${MODEL_S:-yolov8s.pt}"   # E2 上限参照
RUNS_DIR=runs
LOGS_DIR=logs
PYTHON="${PYTHON:-python}"
SMOKE="${SMOKE:-0}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"  # 冒烟极小轮数,仅走通流程
SMOKE_BATCH="${SMOKE_BATCH:-4}"

FULL_CMD="bash tools/run_exp.sh${*:+ $*}"
SUB="${1:-help}"
[ $# -gt 0 ] && shift || true

log()  { echo "[run_exp] $*"; }
warn() { echo "[run_exp][警告] $*" >&2; }
die()  { echo "[run_exp][错误] $*" >&2; exit 2; }
smoke_mode() { [ "$SMOKE" = "1" ]; }

tmux_hint() {
    # §6 纪律:训练在 tmux 内执行。非 tmux 环境只警告不阻断(调试/CI 场景)。
    if [ -z "${TMUX:-}" ]; then
        warn "当前不在 tmux 会话内:§6 纪律要求训练在 tmux 内执行(tmux new -s exp),断线会丢任务"
    fi
}

# ============================================================================
# 配置快照:每次经本脚本发起训练后自动落盘到当次 runs 目录(config_snapshot.yaml)
#   $1=name(runs 目录名) $2=实验编号 $3=模式(train|smoke) $4=备注
#   $5=附加 Albumentations 注入说明(无则空) $6=训练开始时刻(epoch 秒)
# 快照取值来源:优先 ultralytics 训练器落盘的 args.yaml(实际生效值);
# args.yaml 缺失时回退为调用参数并在 values_source 中如实标注。
# ============================================================================
snapshot_config() {
    SNAP_NAME="$1" SNAP_EXP="$2" SNAP_MODE="$3" SNAP_NOTE="$4" SNAP_EXTRA="$5" SNAP_START="$6" \
    SNAP_RUNS="$RUNS_DIR" SNAP_CMD="$FULL_CMD" \
    SNAP_F_MODEL="${SNAP_F_MODEL:-}" SNAP_F_DATA="${SNAP_F_DATA:-$DATA}" \
    SNAP_F_IMGSZ="${SNAP_F_IMGSZ:-$IMGSZ}" SNAP_F_EPOCHS="${SNAP_F_EPOCHS:-$EPOCHS}" \
    SNAP_F_BATCH="${SNAP_F_BATCH:-$BATCH}" SNAP_F_OPTIMIZER="${SNAP_F_OPTIMIZER:-$OPTIMIZER}" \
    SNAP_F_PATIENCE="${SNAP_F_PATIENCE:-$PATIENCE}" SNAP_F_MS="${SNAP_F_MS:-True}" \
    SNAP_F_COSLR="${SNAP_F_COSLR:-True}" SNAP_F_AMP="${SNAP_F_AMP:-True}" SNAP_F_SEED="${SNAP_F_SEED:-$SEED}" \
    "$PYTHON" - <<'PYEOF'
# -*- coding: utf-8 -*-
# run_exp.sh 内嵌:生成 config_snapshot.yaml(自动调用,禁止人肉另跑)
import os, sys, glob, json, socket, platform, datetime
try:
    import yaml
except ImportError:
    yaml = None

name = os.environ["SNAP_NAME"]
runs_root = os.environ["SNAP_RUNS"]
start = float(os.environ["SNAP_START"])

# 1) 定位当次 runs 目录:同名前缀 + 目录时间不早于训练开始(ultralytics 同名会加序号)
cands = []
for d in glob.glob(os.path.join(runs_root, name + "*")):
    if os.path.isdir(d):
        mt = os.path.getmtime(d)
        if mt >= start - 5:
            cands.append((mt, d))
if not cands:
    print("快照失败:%s 下找不到本次 %s* 目录" % (runs_root, name), file=sys.stderr)
    sys.exit(1)
run_dir = max(cands)[1]

# 2) 读取 ultralytics 训练器落盘的实际生效参数(args.yaml 优先,args.json 兜底)
eff = {}
src = None
for cand in ("args.yaml", "args.json"):
    p = os.path.join(run_dir, cand)
    if not os.path.exists(p):
        continue
    try:
        with open(p, encoding="utf-8") as fh:
            eff = (yaml.safe_load(fh) if cand.endswith(".yaml") and yaml else json.load(fh)) or {}
        src = cand
        break
    except Exception as exc:
        print("读取 %s 失败:%s" % (p, exc), file=sys.stderr)

# 3) 统一参数(§4)与增强参数:取实际生效值;args.yaml 缺失键回退为调用参数
fb = {  # 调用参数回退值(由 run_exp.sh 按当次实际调用传入)
    "model": os.environ.get("SNAP_F_MODEL") or None,
    "data": os.environ.get("SNAP_F_DATA") or None,
    "imgsz": os.environ.get("SNAP_F_IMGSZ"), "epochs": os.environ.get("SNAP_F_EPOCHS"),
    "batch": os.environ.get("SNAP_F_BATCH"), "optimizer": os.environ.get("SNAP_F_OPTIMIZER"),
    "patience": os.environ.get("SNAP_F_PATIENCE"), "multi_scale": os.environ.get("SNAP_F_MS"),
    "cos_lr": os.environ.get("SNAP_F_COSLR"), "amp": os.environ.get("SNAP_F_AMP"),
    "seed": os.environ.get("SNAP_F_SEED"),
}
UNIFIED = ["model", "data", "imgsz", "epochs", "batch", "optimizer",
           "patience", "multi_scale", "cos_lr", "amp", "seed"]
unified = {}
for k in UNIFIED:
    unified[k] = eff[k] if k in eff else fb.get(k)
values_source = ("runs 目录 %s(ultralytics 训练器落盘的实际生效值)" % src) if src \
    else "调用参数(未找到 args.yaml,训练可能未进入 trainer 阶段)"

AUGS = ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
        "perspective", "flipud", "fliplr", "mosaic", "mixup", "cutmix",
        "copy_paste", "copy_paste_mode", "auto_augment", "erasing"]
augmentation = {k: eff[k] for k in AUGS if k in eff}
extra = os.environ.get("SNAP_EXTRA") or None
augmentation["albumentations_extra"] = extra  # run_exp.sh 补丁注入项(受试变量),无则 null

# 4) 环境可探测项
try:
    from importlib.metadata import version as _v
    ul_ver = _v("ultralytics")
except Exception:
    ul_ver = None
gpu, cuda = None, None
try:
    import torch
    cuda = bool(torch.cuda.is_available())
    gpu = torch.cuda.get_device_name(0) if cuda else None
except Exception:
    pass

snap = {
    "experiment": os.environ["SNAP_EXP"],
    "mode": os.environ["SNAP_MODE"],
    "note": os.environ["SNAP_NOTE"],
    "command": os.environ.get("SNAP_CMD", ""),
    "runs_dir": run_dir.replace("\\", "/"),
    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "values_source": values_source,
    "unified_params": unified,
    "augmentation": augmentation,
    "environment": {
        "ultralytics": ul_ver,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cuda_available": cuda,
        "gpu_name": gpu,
    },
}
if os.environ["SNAP_MODE"] == "smoke":
    snap["smoke_warning"] = ("合成数据冒烟:数据为 tools/train.py 自动合成的随机矩形,"
                             "无语义,精度无意义,严禁冒充真实训练结果")

out = os.path.join(run_dir, "config_snapshot.yaml")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("# config_snapshot.yaml —— tools/run_exp.sh 自动生成(每次训练自动落盘,禁止手工编辑)\n")
    if yaml:
        yaml.safe_dump(snap, fh, allow_unicode=True, sort_keys=False)
    else:
        json.dump(snap, fh, ensure_ascii=False, indent=2)
print("配置快照已落盘:%s" % out)
if src is None:
    print("警告:未找到 args.yaml,快照统一参数取调用参数而非训练器实际值", file=sys.stderr)
PYEOF
}

# ============================================================================
# 组合训练驱动(E1b/E1c/E1e/E2 共用):向 ultralytics Albumentations 注入受试
# 变换后发起训练。cutout 不是 8.4.21 合法参数(见脚本头说明),E1b 以其语义
# 等价 CoarseDropout(p=0.3) 实现;E1c 为 InvertImg(p=0.2)。
# 补丁方式:改写 Albumentations.__init__ 的默认 transforms(8.4.21 默认清单
# 逐项照抄其 augment.py,版本耦合已与实例锁定的 8.4.21 对齐)。
# ============================================================================
train_combo() {
    local name="$1" exp_id="$2" note="$3" degrees="$4" cp="$5" extras="$6" extra_desc="$7" model="$8"
    tmux_hint
    local start rc
    start=$(date +%s)
    EXP_MODEL="$model" EXP_DATA="$DATA" EXP_IMGSZ="$IMGSZ" EXP_EPOCHS="$EPOCHS" \
    EXP_BATCH="$BATCH" EXP_OPTIMIZER="$OPTIMIZER" EXP_PATIENCE="$PATIENCE" EXP_SEED="$SEED" \
    EXP_DEGREES="$degrees" EXP_COPY_PASTE="$cp" EXP_ALBU_EXTRAS="$extras" \
    EXP_PROJECT="$RUNS_DIR" EXP_NAME="$name" \
    "$PYTHON" - <<'PYEOF'
# -*- coding: utf-8 -*-
# run_exp.sh 内嵌:E1b/E1c/E1e/E2 训练驱动(补丁注入 Albumentations 受试变换)
import os, sys
import albumentations as A
from ultralytics.data import augment as ua


def _base_transforms_8421():
    """ultralytics 8.4.21 Albumentations 默认变换清单(逐项照抄其 augment.py)。"""
    return [
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.RandomBrightnessContrast(p=0.0),
        A.RandomGamma(p=0.0),
        A.ImageCompression(quality_range=(75, 100), p=0.0),
    ]


def _coarse_dropout(p):
    """cutout 语义等价:随机遮挡切块。兼容 albumentations 新旧两套参数名。"""
    try:
        # albumentations >=1.4.15 / 2.x 参数名;hole 尺寸取相对比例(约占图 10%)
        return A.CoarseDropout(num_holes_range=(1, 1),
                               hole_height_range=(0.1, 0.1),
                               hole_width_range=(0.1, 0.1),
                               fill=0, p=p)
    except Exception:
        # 旧版参数名(像素尺寸,48~96px 对 imgsz=960 约 5%~10%)
        return A.CoarseDropout(min_holes=1, max_holes=1,
                               min_height=48, max_height=96,
                               min_width=48, max_width=96,
                               fill_value=0, p=p)


_EXTRA_BUILDERS = {
    "cutout": lambda: _coarse_dropout(0.3),   # E1b:切割类增强(§4 矩阵 cutout=0.3)
    "invert": lambda: A.InvertImg(p=0.2),     # E1c:反转色(模拟红外/负片光照)
}
extras = [x for x in os.environ.get("EXP_ALBU_EXTRAS", "").split(",") if x]
extra_transforms = [_EXTRA_BUILDERS[x]() for x in extras]
if extras:
    print("[run_exp] Albumentations 注入受试变换:%s" % ", ".join(extras))

_orig_init = ua.Albumentations.__init__


def _patched_init(self, p=1.0, transforms=None):
    # 仅在调用方未自定义 transforms 时注入:8.4.21 默认清单 + 受试变换
    if transforms is None:
        transforms = _base_transforms_8421() + extra_transforms
    _orig_init(self, p, transforms)


ua.Albumentations.__init__ = _patched_init

from ultralytics import YOLO
model = YOLO(os.environ["EXP_MODEL"])
results = model.train(
    data=os.environ["EXP_DATA"],
    imgsz=int(os.environ["EXP_IMGSZ"]),
    epochs=int(os.environ["EXP_EPOCHS"]),
    batch=int(os.environ["EXP_BATCH"]),
    optimizer=os.environ["EXP_OPTIMIZER"],
    patience=int(os.environ["EXP_PATIENCE"]),
    # §4 统一参数:multi_scale / cos_lr / AMP / seed 全部显式
    multi_scale=True, cos_lr=True, amp=True,
    seed=int(os.environ["EXP_SEED"]),
    # E0 基线默认增强显式锚定;受试变量 degrees/copy_paste 由环境传入实际值
    mosaic=1.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=float(os.environ["EXP_DEGREES"]),
    translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.0, mixup=0.0, cutmix=0.0,
    copy_paste=float(os.environ["EXP_COPY_PASTE"]),
    project=os.environ["EXP_PROJECT"], name=os.environ["EXP_NAME"],
)
print("TRAIN_SAVE_DIR=%s" % getattr(results, "save_dir", ""))
PYEOF
    rc=$?
    snapshot_config "$name" "$exp_id" "train" "$note" "$extra_desc" "$start" \
        || warn "配置快照生成失败(训练退出码 $rc 不变)"
    return $rc
}

# ============================================================================
# 纯 yolo CLI 训练(E0/E1a/E1d):参数全显式写在函数体内
#   $1=name $2=实验编号 $3=备注;调用方以 DEGREES=xx / COPY_PASTE=xx 前缀传受试变量
# ============================================================================
train_yolo() {
    local name="$1" exp_id="$2" note="$3"
    tmux_hint
    command -v yolo >/dev/null 2>&1 \
        || die "未找到 yolo CLI(实例需 pip install ultralytics==8.4.21)"
    local start rc
    start=$(date +%s)
    yolo detect train \
        model="$MODEL_N" data="$DATA" \
        imgsz="$IMGSZ" epochs="$EPOCHS" batch="$BATCH" \
        optimizer="$OPTIMIZER" patience="$PATIENCE" \
        multi_scale=True cos_lr=True amp=True seed="$SEED" \
        mosaic=1.0 fliplr=0.5 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
        degrees="${DEGREES:-0.0}" translate=0.1 scale=0.5 shear=0.0 perspective=0.0 \
        flipud=0.0 mixup=0.0 cutmix=0.0 copy_paste="${COPY_PASTE:-0.0}" \
        project="$RUNS_DIR" name="$name"
    rc=$?
    snapshot_config "$name" "$exp_id" "train" "$note" "" "$start" \
        || warn "配置快照生成失败(训练退出码 $rc 不变)"
    return $rc
}

# ============================================================================
# 冒烟训练(SMOKE=1):走 tools/train.py 既有 --smoke 机制(合成迷你数据集、
# 极小 epochs、amp 关闭、无联网下载),仅验证流程;快照标注合成数据。
#   $1=基础 name(实际 runs 目录加 smoke_ 前缀) $2=实验编号 $3=degrees(可选)
# ============================================================================
smoke_train() {
    local base="$1" exp_id="$2" degrees="${3:-0.0}" name="smoke_$1"
    local start rc
    start=$(date +%s)
    warn "SMOKE 冒烟模式:tools/train.py --smoke 合成迷你数据集(无语义,精度无意义,严禁冒充真实训练)"
    case "$SUB" in
        e1b|e1c) warn "SMOKE 不演练 Albumentations 受试变量($SUB 的 cutout/invert),仅走通训练流程" ;;
        e1d)     warn "SMOKE 不演练 copy_paste 受试变量(train.py 未暴露该参数),仅走通训练流程" ;;
        e1e|e2)  warn "SMOKE 不演练组合增强与模型换装,仅走通训练流程" ;;
    esac
    # 冒烟实际生效值与正式矩阵不同(imgsz/epochs/amp/multi_scale 等),
    # 快照回退值(SNAP_F_*)按冒烟实际调用传入,如实记录
    "$PYTHON" tools/train.py --smoke \
        --epochs "$SMOKE_EPOCHS" --imgsz 320 --batch "$SMOKE_BATCH" \
        --degrees "$degrees" \
        --seed "$SEED" --project "$RUNS_DIR" --name "$name"
    rc=$?
    SNAP_F_MODEL="weights/yolov8n.pt(train.py smoke)" SNAP_F_DATA="合成迷你数据集(train.py smoke)" \
    SNAP_F_IMGSZ=320 SNAP_F_EPOCHS="$SMOKE_EPOCHS" SNAP_F_BATCH="$SMOKE_BATCH" \
    SNAP_F_PATIENCE=100 SNAP_F_MS=False SNAP_F_COSLR=False SNAP_F_AMP=False \
    snapshot_config "$name" "$exp_id" "smoke" \
        "SMOKE 冒烟:合成数据/无语义/精度无意义,不冒充真实训练" "" "$start" \
        || warn "配置快照生成失败(冒烟退出码 $rc 不变)"
    return $rc
}

# ============================================================================
# 实验函数(E0–E6,对照 §4 矩阵;统一参数:imgsz=960 epochs=100 batch=32(可覆盖)
#   optimizer=auto patience=20 multi_scale=True cos_lr=True AMP 开 seed=0)
# ============================================================================

# E0 基线复训:YOLOv8n + 默认增强(mosaic=1.0, fliplr=0.5, 轻 HSV)
cmd_e0() {
    if smoke_mode; then smoke_train e0_baseline E0; return $?; fi
    train_yolo e0_baseline E0 \
        "E0 基线:yolov8n + 默认增强(mosaic=1.0 fliplr=0.5 轻HSV hsv_h=0.015 hsv_s=0.7 hsv_v=0.4)"
}

# E1a 旋转增强 degrees=10(俯视无固定朝向;对照 E0,仅 degrees 不同)
cmd_e1a10() {
    if smoke_mode; then smoke_train e1a_deg10 E1a 10.0; return $?; fi
    DEGREES=10.0 train_yolo e1a_deg10 E1a "E1a 旋转增强:degrees=10(其余同 E0)"
}

# E1a 旋转增强 degrees=30
cmd_e1a30() {
    if smoke_mode; then smoke_train e1a_deg30 E1a 30.0; return $?; fi
    DEGREES=30.0 train_yolo e1a_deg30 E1a "E1a 旋转增强:degrees=30(其余同 E0)"
}

# E1b 切割类增强 cutout=0.3(随机遮挡切块;cutout 非 8.4.21 合法参数,
# 以 Albumentations CoarseDropout(p=0.3) 等价实现,见脚本头说明)
cmd_e1b() {
    if smoke_mode; then smoke_train e1b_cutout E1b; return $?; fi
    train_combo e1b_cutout E1b \
        "E1b 切割增强:cutout=0.3(Albumentations CoarseDropout(p=0.3) 等价实现,其余同 E0)" \
        0.0 0.0 "cutout" "Albumentations CoarseDropout(p=0.3)[等价 cutout=0.3]" "$MODEL_N"
}

# E1c 反转色 Albumentations InvertImg(p=0.2)(模拟红外/负片光照;对照 E0)
cmd_e1c() {
    if smoke_mode; then smoke_train e1c_invert E1c; return $?; fi
    train_combo e1c_invert E1c \
        "E1c 反转色:Albumentations InvertImg(p=0.2)(其余同 E0)" \
        0.0 0.0 "invert" "InvertImg(p=0.2)" "$MODEL_N"
}

# E1d 小目标增强 copy_paste=0.1(对照 E0,仅 copy_paste 不同)
cmd_e1d() {
    if smoke_mode; then smoke_train e1d_copypaste E1d; return $?; fi
    COPY_PASTE=0.1 train_yolo e1d_copypaste E1d "E1d 小目标增强:copy_paste=0.1(其余同 E0)"
}

# E1e/E2 组合参数校验与注入组合串
_combo_args() {
    # $1=degrees $2=cutout(0|0.3) $3=invert(0|1) $4=copy_paste
    [ $# -eq 4 ] || die "用法: bash tools/run_exp.sh $SUB <degrees> <cutout:0|0.3> <invert:0|1> <copy_paste>
       E1e 最优组合尚未实测:组合参数待 E1a–E1d 结果出来后确定,须调用方全显式传入"
    [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "degrees 须为非负数值,收到: $1"
    case "$2" in 0|0.3) ;; *) die "cutout 只接受 0 或 0.3(§4 矩阵受试值),收到: $2" ;; esac
    case "$3" in 0|1)   ;; *) die "invert 只接受 0 或 1,收到: $3" ;; esac
    [[ "$4" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "copy_paste 须为非负数值,收到: $4"
}
_combo_extras() { # $1=cutout $2=invert -> "cutout,invert"
    local e=""
    [ "$1" = "0.3" ] && e="cutout"
    [ "$2" = "1" ] && e="${e:+$e,}invert"
    echo "$e"
}
_combo_desc() {  # $1=cutout $2=invert -> 快照/备注用文字说明
    local d=""
    [ "$1" = "0.3" ] && d="Albumentations CoarseDropout(p=0.3)[等价 cutout=0.3]"
    [ "$2" = "1" ] && d="${d:+$d + }InvertImg(p=0.2)"
    echo "$d"
}

# E1e 最优组合:E1a–E1d 中实测有效的项组合(论文主模型)。
# 注意:最优组合尚未实测出来——组合参数待 E1a–E1d 结果出来后确定,
# 本函数不预设任何组合,degrees/cutout/invert/copy_paste 全部由调用方显式传入。
cmd_e1e() {
    if smoke_mode; then smoke_train e1e_combo E1e; return $?; fi
    _combo_args "$@"
    local deg="$1" cut="$2" inv="$3" cp="$4"
    local name extras desc
    name=$(printf 'e1e_deg%s_cut%s_inv%s_cp%s' "$deg" "$cut" "$inv" "$cp" | tr -d '.')
    extras=$(_combo_extras "$cut" "$inv")
    desc=$(_combo_desc "$cut" "$inv")
    train_combo "$name" E1e \
        "E1e 最优组合(E1a–E1d 实测有效项;调用方传入:degrees=$deg cutout=$cut invert=$inv copy_paste=$cp)" \
        "$deg" "$cp" "$extras" "$desc" "$MODEL_N"
}

# E2 大模型参照:YOLOv8s 套用 E1e 最优组合(精度上限参照行;建议 BATCH=16,§6)
cmd_e2() {
    if smoke_mode; then smoke_train e2_yolov8s E2; return $?; fi
    _combo_args "$@"
    local deg="$1" cut="$2" inv="$3" cp="$4"
    local name extras desc
    name=$(printf 'e2_yolov8s_deg%s_cut%s_inv%s_cp%s' "$deg" "$cut" "$inv" "$cp" | tr -d '.')
    extras=$(_combo_extras "$cut" "$inv")
    desc=$(_combo_desc "$cut" "$inv")
    train_combo "$name" E2 \
        "E2 大模型参照:yolov8s 套用 E1e 组合(degrees=$deg cutout=$cut invert=$inv copy_paste=$cp)" \
        "$deg" "$cp" "$extras" "$desc" "$MODEL_S"
}

# E3 轻量化测量:参数量/GFLOPs(imgsz=960)/FP32/FP16/INT8 三档权重体积/FPS
#   FPS 口径(§5):yolo val,batch=1,imgsz=960,FP32 与 FP16 各一组
cmd_e3() {
    local w="${1:-}"
    [ -n "$w" ] || die "用法: bash tools/run_exp.sh e3 <weights.pt> [fp16权重文件] [int8权重文件]"
    [ -f "$w" ] || die "权重文件不存在: $w"
    EXP_WEIGHTS="$w" EXP_FP16="${2:-}" EXP_INT8="${3:-}" EXP_DATA="$DATA" EXP_IMGSZ="$IMGSZ" \
    "$PYTHON" - <<'PYEOF'
# -*- coding: utf-8 -*-
# run_exp.sh 内嵌:E3 轻量化测量
import os
w = os.environ["EXP_WEIGHTS"]
imgsz = int(os.environ["EXP_IMGSZ"])
import torch
from ultralytics import YOLO
model = YOLO(w)
n_params = sum(p.numel() for p in model.model.parameters())
flops = None
try:
    from ultralytics.utils.torch_utils import get_flops
    flops = get_flops(model.model, imgsz)
except Exception as exc:
    print("GFLOPs 计算失败:%s" % exc)
print("=" * 64)
print("E3 轻量化测量(权重:%s,imgsz=%d)" % (w, imgsz))
print("  参数量: %.3f M" % (n_params / 1e6))
if flops:
    print("  GFLOPs: %.2f" % flops)
print("  权重体积 FP32(%s): %.2f MB" % (os.path.basename(w), os.path.getsize(w) / 1e6))
for tag, envk in (("FP16", "EXP_FP16"), ("INT8", "EXP_INT8")):
    p = os.environ.get(envk)
    if p and os.path.isfile(p):
        print("  权重体积 %s(%s): %.2f MB" % (tag, os.path.basename(p), os.path.getsize(p) / 1e6))
    elif p:
        print("  权重体积 %s: 文件不存在(%s),跳过" % (tag, p))
    else:
        print("  权重体积 %s: 未提供文件(先跑 e4 导出得到),跳过" % tag)
data = os.environ["EXP_DATA"]
wstem = os.path.splitext(os.path.basename(w))[0]
for half in (False, True):
    if half and not torch.cuda.is_available():
        print("  FPS(FP16): 本机无 CUDA,跳过(在实例上执行)")
        continue
    # workers=0:Windows 下 stdin 脚本 spawn 数据加载子进程会崩,0 保证跨平台可跑;
    # 显式 name:复测产物落 runs/detect/e3_val_<模型>_<精度>,便于溯源(不落 valN)
    tag = "fp16" if half else "fp32"
    r = model.val(data=data, imgsz=imgsz, batch=1, half=half, verbose=False, workers=0,
                  name="e3_val_%s_%s" % (wstem, tag))
    infer_ms = float(r.speed.get("inference", 0.0))
    fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
    print("  FPS(%s,batch=1,imgsz=%d): %.1f(单帧推理 %.2f ms)"
          % ("FP16" if half else "FP32", imgsz, fps, infer_ms))
print("=" * 64)
PYEOF
}

# E4 量化鲁棒性:导出 FP16 / INT8(PTQ,校准集=验证集 548 张),并用同一
#   yolo val 口径复测验证集 mAP,对比 FP32 掉点。
# 路线(修正留痕,R3 第一次验证 FAIL 项):8.4.21 的 ONNX 导出**不支持 int8**——
#   export_formats() 表中 ONNX 合法参数仅 batch/dynamic/half/opset/simplify/nms,
#   validate_args() 对 int8 直接 assert(R3 验证方实测 AssertionError/EXIT=1,
#   本机已复跑复现:logs/r3_fix_onnx_int8_repro.log)。故 FP16/INT8 统一走
#   TensorRT engine(format=engine:half/int8/batch 均在其合法参数表内,本机
#   8.4.21 已用 validate_args() 源码调用 + CLI 干跑实测通过,留痕
#   logs/r3_fix_validate_args.log、logs/r3_fix_engine_int8_dryrun.log——干跑
#   越过参数校验与 GPU 断言、ONNX 中间导出成功,仅缺实例侧 tensorrt 包)。
#   选 TensorRT 而非 openvino 的原因:实例为 RTX 4090,TensorRT PTQ 是 GPU 部署
#   原生路线;FP16/INT8 同一后端,mAP 掉点为纯量化效应,不混入后端差异;
#   openvino INT8 面向 Intel CPU 部署,与本项目边缘/FPGA 论证主线无关。
# 注意:① engine 导出强制 GPU(device=0,CPU 会被 assert 拒绝);
#   ② 两腿导出产物同名 <stem>.engine,每腿导完立即改名为 _fp16/_int8.engine;
#   ③ INT8 校准集 = data 的 val 划分全量(visdrone 验证集 548 张,fraction
#   默认 1.0,augment=False,get_int8_calibration_dataloader 行为),batch=8
#   仅加速校准收集,不改变校准集口径;④ engine 绑定生成机 GPU/TensorRT
#   版本,同机导出同机复测,禁止跨机拷贝;⑤ 实例侧需预装 tensorrt(见脚本头
#   依赖说明);⑥ engine 导出内部会先产 <stem>.onnx 中间文件,属正常现象。
cmd_e4() {
    local w="${1:-}"
    [ -n "$w" ] || die "用法: bash tools/run_exp.sh e4 <weights.pt>"
    [ -f "$w" ] || die "权重文件不存在: $w"
    command -v yolo >/dev/null 2>&1 || die "未找到 yolo CLI"
    local stem="${w%.pt}" base rc=0
    base=$(basename "$stem")
    log "E4 FP16 导出:format=engine half=True(TensorRT,需实例 GPU)"
    rm -f "${stem}.engine"
    yolo export model="$w" format=engine half=True imgsz="$IMGSZ" device=0 || rc=$?
    if [ -f "${stem}.engine" ]; then
        mv -f "${stem}.engine" "${stem}_fp16.engine"
        log "FP16 产物: ${stem}_fp16.engine"
    else
        warn "FP16 导出未产出 ${stem}.engine(退出码已记录),FP16 腿失败"
    fi
    log "E4 INT8 PTQ 导出:format=engine int8=True(校准集=验证集 548 张全量)"
    rm -f "${stem}.engine"
    yolo export model="$w" format=engine int8=True data="$DATA" imgsz="$IMGSZ" batch=8 device=0 || rc=$?
    if [ -f "${stem}.engine" ]; then
        mv -f "${stem}.engine" "${stem}_int8.engine"
        log "INT8 产物: ${stem}_int8.engine"
    else
        warn "INT8 导出未产出 ${stem}.engine(退出码已记录),INT8 腿失败"
    fi
    log "E4 验证集 mAP 复测(同一 yolo val 口径,batch=1):FP32 参照"
    yolo val model="$w" data="$DATA" imgsz="$IMGSZ" batch=1 \
        name="e4_val_${base}_fp32" || rc=$?
    if [ -f "${stem}_fp16.engine" ]; then
        yolo val model="${stem}_fp16.engine" data="$DATA" imgsz="$IMGSZ" batch=1 \
            name="e4_val_${base}_fp16" || rc=$?
    else
        warn "缺 ${stem}_fp16.engine,跳过 FP16 复测"
    fi
    if [ -f "${stem}_int8.engine" ]; then
        yolo val model="${stem}_int8.engine" data="$DATA" imgsz="$IMGSZ" batch=1 \
            name="e4_val_${base}_int8" || rc=$?
    else
        warn "缺 ${stem}_int8.engine,跳过 INT8 复测"
    fi
    return $rc
}

# E5 FPGA 友好性分析:算子构成统计(Conv/BN/激活占比)、BN 折叠后参数量、
#   动态 shape 排查。"SiLU→ReLU 重训对照"为 §4 标注的可选项,未排期,不在本函数内。
cmd_e5() {
    local w="${1:-}"
    [ -n "$w" ] || die "用法: bash tools/run_exp.sh e5 <weights.pt>"
    [ -f "$w" ] || die "权重文件不存在: $w"
    EXP_WEIGHTS="$w" "$PYTHON" - <<'PYEOF'
# -*- coding: utf-8 -*-
# run_exp.sh 内嵌:E5 算子构成统计
import os, collections
w = os.environ["EXP_WEIGHTS"]
import torch.nn as nn
from ultralytics import YOLO
net = YOLO(w).model
counts = collections.Counter(type(m).__name__ for m in net.modules())
total = sum(p.numel() for p in net.parameters())
bn_params = 0
bn_count = 0
for m in net.modules():
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        bn_count += 1
        bn_params += sum(p.numel() for p in m.parameters())
print("=" * 64)
print("E5 算子构成统计(权重:%s)" % w)
print("  总参数量: %.3f M" % (total / 1e6))
print("  BN 层数: %d;BN 可吸收参数量: %.4f M" % (bn_count, bn_params / 1e6))
print("  BN 折叠后参数量(估计): %.3f M" % ((total - bn_params) / 1e6))
print("  (口径:BN 折叠把 BN 的 gamma/beta/均值/方差吸收进相邻 Conv 权重与偏置,")
print("   总参数减少量=BN 参数总量;精确值以实际 fold 后模型为准)")
n_modules = sum(counts.values())
print("  模块构成(共 %d 个模块):" % n_modules)
for name, n in counts.most_common():
    print("    %-24s %5d  (%5.1f%%)" % (name, n, 100.0 * n / n_modules))
# 动态 shape 排查:列出已知会引入动态/不定形状的算子;YOLOv8 固定 imgsz 导出时不含
watch = {"Upsample", "PixelShuffle", "Interpolate", "GridSample", "RoiAlign"}
found = {k: v for k, v in counts.items() if k in watch}
if found:
    print("  动态 shape 排查:发现需核对算子 %s(固定 imgsz 导出时通常为静态,以后端导出日志为准)" % found)
else:
    print("  动态 shape 排查:未发现 Upsample/Interpolate 等动态算子(固定 imgsz 导出,动态性以 e4 导出日志为准)")
silu = counts.get("SiLU", 0)
# modules() 对共享对象去重:ultralytics Conv 的 default_act 是类级共享 nn.SiLU,
# 需另按"激活点位"(持有 SiLU 激活的模块数)统计,FPGA 分析看的是点位数
silu_sites = sum(1 for m in net.modules()
                 if isinstance(getattr(m, "act", None), nn.SiLU))
print("  SiLU 激活点位数: %d(modules() 去重后 SiLU 对象 %d 个,Conv 间共享)"
      % (silu_sites, silu))
print("  (FPGA 对 ReLU 支持最好;SiLU→ReLU 替换为 §4 标注的可选项,未排期)")
print("=" * 64)
PYEOF
}

# E6 系统级验证:调用既有 tools/evaluate.py 跑 farm_guard 两段素材,
#   与 A10 报告 COCO 权重结果同口径对比(检测成功率/误检率/漏检率)。
#   注意:素材在本机(C:/Users/OS/Desktop/farm_guard/videos/),E6 在本机执行,
#   E1e 权重需先 rsync 回传。
cmd_e6() {
    local w="${1:-}"
    [ -n "$w" ] || die "用法: bash tools/run_exp.sh e6 <weights.pt>"
    [ -f "$w" ] || die "权重文件不存在: $w(请先将 E1e 权重 rsync 回传本机)"
    [ -f tests/eval_config.yaml ] || die "缺少评估配置 tests/eval_config.yaml"
    local out="tests/e6_eval_$(basename "${w%.pt}").json"
    "$PYTHON" tools/evaluate.py --config tests/eval_config.yaml --weights "$w" --out "$out"
}

# ============================================================================
# 登记(§7.4):从指定 runs 目录的 results.csv 读出最佳 mAP50/mAP50-95,
#   自动追加登记行到 docs/实验登记.csv。数字一律来自文件,无手工填数入口。
#   REGISTER_CSV=<路径> 可覆盖登记目标(验证/试登记用临时副本)。
# ============================================================================
cmd_register() {
    local runs_dir="${1:-}" exp_id="${2:-}" note="${3:-}"
    [ -n "$runs_dir" ] && [ -n "$exp_id" ] \
        || die "用法: bash tools/run_exp.sh register <runs_dir> <实验编号> [备注]  (REGISTER_CSV=<路径> 可覆盖登记目标)"
    [ -d "$runs_dir" ] || die "runs 目录不存在: $runs_dir"
    REG_RUNS="$runs_dir" REG_EXP="$exp_id" REG_NOTE="$note" \
    REG_CSV="${REGISTER_CSV:-docs/实验登记.csv}" \
    "$PYTHON" - <<'PYEOF'
# -*- coding: utf-8 -*-
# run_exp.sh 内嵌:实验登记(数字一律来自 runs 目录内文件)
import os, sys, csv, datetime
try:
    import yaml
except ImportError:
    yaml = None

runs_dir = os.environ["REG_RUNS"].rstrip("/\\")
exp_id = os.environ["REG_EXP"]
note = os.environ["REG_NOTE"]
csv_path = os.environ["REG_CSV"]
HEADER = ["日期", "实例型号", "实验编号", "配置摘要", "runs路径",
          "验证集mAP50", "验证集mAP50-95", "备注"]

results_csv = os.path.join(runs_dir, "results.csv")
if not os.path.isfile(results_csv):
    print("登记失败:找不到 %s" % results_csv, file=sys.stderr)
    sys.exit(1)

# 1) 从 results.csv 逐项读取 mAP:取全列最大值,输出文件内原始字符串(与文件逐项一致)
with open(results_csv, encoding="utf-8", newline="") as fh:
    rows = list(csv.reader(fh))
if len(rows) < 2:
    print("登记失败:results.csv 无数据行", file=sys.stderr)
    sys.exit(1)
header = [h.strip() for h in rows[0]]


def _col(*names):
    for n in names:
        if n in header:
            return header.index(n)
    for i, h in enumerate(header):
        for n in names:
            if h.startswith(n):
                return i
    return None


i50 = _col("metrics/mAP50(B)", "metrics/mAP50")
i5095 = _col("metrics/mAP50-95(B)", "metrics/mAP50-95")
if i50 is None or i5095 is None:
    print("登记失败:results.csv 缺少 mAP 列,表头=%s" % header, file=sys.stderr)
    sys.exit(1)
best50 = best5095 = None
raw50 = raw5095 = ""
for r in rows[1:]:
    if len(r) <= max(i50, i5095):
        continue
    try:
        v50 = float(r[i50])
        v5095 = float(r[i5095])
    except ValueError:
        continue
    if best50 is None or v50 > best50:
        best50, raw50 = v50, r[i50].strip()
    if best5095 is None or v5095 > best5095:
        best5095, raw5095 = v5095, r[i5095].strip()
if best50 is None:
    print("登记失败:results.csv 无可解析的 mAP 数值", file=sys.stderr)
    sys.exit(1)

# 2) 日期/实例型号/配置摘要:优先 config_snapshot.yaml,回退 args.yaml/文件时间
snap = {}
sp = os.path.join(runs_dir, "config_snapshot.yaml")
if os.path.isfile(sp) and yaml:
    with open(sp, encoding="utf-8") as fh:
        snap = yaml.safe_load(fh) or {}
date_s = str(snap.get("created_at") or "")[:10]
if not date_s:
    date_s = datetime.datetime.fromtimestamp(
        os.path.getmtime(results_csv)).strftime("%Y-%m-%d")
instance = (snap.get("environment") or {}).get("gpu_name") \
    or os.environ.get("INSTANCE_MODEL", "")

summary = ""
u = snap.get("unified_params") or {}
a = snap.get("augmentation") or {}
if u:
    parts = ["model=%s" % u.get("model"), "imgsz=%s" % u.get("imgsz"),
             "epochs=%s" % u.get("epochs"), "batch=%s" % u.get("batch"),
             "optimizer=%s" % u.get("optimizer"), "patience=%s" % u.get("patience"),
             "multi_scale=%s" % u.get("multi_scale"), "cos_lr=%s" % u.get("cos_lr"),
             "amp=%s" % u.get("amp"), "seed=%s" % u.get("seed"),
             "degrees=%s" % a.get("degrees"), "mosaic=%s" % a.get("mosaic"),
             "fliplr=%s" % a.get("fliplr"), "copy_paste=%s" % a.get("copy_paste")]
    if a.get("albumentations_extra"):
        parts.append("albu=%s" % a["albumentations_extra"])
    summary = " ".join(str(x) for x in parts)
if not summary:
    ap = os.path.join(runs_dir, "args.yaml")
    if os.path.isfile(ap) and yaml:
        with open(ap, encoding="utf-8") as fh:
            eff = yaml.safe_load(fh) or {}
        summary = " ".join("%s=%s" % (k, eff.get(k))
                           for k in ("model", "imgsz", "epochs", "batch", "optimizer", "seed"))
if not summary:
    summary = "见 runs 目录 args.yaml"

# 3) 追加登记行:目标已存在则校验表头,不存在则按模板创建(utf-8-sig)
if os.path.isfile(csv_path):
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        first = fh.readline().strip("\r\n")
    if first != ",".join(HEADER):
        print("登记失败:%s 表头与 §7.4 模板不一致: %s" % (csv_path, first), file=sys.stderr)
        sys.exit(1)
else:
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerow(HEADER)
    print("登记 CSV 不存在,已按模板创建:%s" % csv_path)

row = [date_s, instance, exp_id, summary, runs_dir.replace("\\", "/"),
       raw50, raw5095, note]
with open(csv_path, "a", encoding="utf-8", newline="") as fh:
    csv.writer(fh).writerow(row)
print("登记完成 -> %s" % csv_path)
print("  日期=%s 实例型号=%s 实验编号=%s" % (date_s, instance or "(未探测到)", exp_id))
print("  验证集mAP50=%s 验证集mAP50-95=%s(来源:%s)" % (raw50, raw5095, results_csv))
PYEOF
}

# ============================================================================
# 入口:使用说明 / 分发 / 日志包装(每次执行落 logs/:命令全文+起止时间+退出码)
# ============================================================================
usage() {
    sed -n '2,/^set -uo pipefail/p' "$0" | sed -e 's/^#\s\?//' -e '/^set -uo pipefail/d'
}

dispatch() {
    case "$SUB" in
        e0)       cmd_e0 "$@" ;;
        e1a10)    cmd_e1a10 "$@" ;;
        e1a30)    cmd_e1a30 "$@" ;;
        e1b)      cmd_e1b "$@" ;;
        e1c)      cmd_e1c "$@" ;;
        e1d)      cmd_e1d "$@" ;;
        e1e)      cmd_e1e "$@" ;;
        e2)       cmd_e2 "$@" ;;
        e3)       cmd_e3 "$@" ;;
        e4)       cmd_e4 "$@" ;;
        e5)       cmd_e5 "$@" ;;
        e6)       cmd_e6 "$@" ;;
        register) cmd_register "$@" ;;
        help|-h|--help) usage ;;
        *) die "未知子命令: $SUB(用 bash tools/run_exp.sh help 查看用法)" ;;
    esac
}

main() {
    mkdir -p "$LOGS_DIR"
    local ts logf rcfile rc
    ts=$(date +%Y%m%d_%H%M%S)
    logf="$LOGS_DIR/run_exp_${SUB}_${ts}.log"
    rcfile="$LOGS_DIR/.rc_${SUB}_${ts}_$$"
    {
        echo "# 命令: $FULL_CMD"
        echo "# 开始: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "# 工作目录: $ROOT"
        # 用 EXIT 陷阱捕获退出码:正常结束与 die/中断路径都能落"结束时间+退出码"
        trap 'trc=$?; echo "$trc" > "$rcfile"; echo "# 结束: $(date "+%Y-%m-%d %H:%M:%S")"; echo "# 退出码: $trc"' EXIT
        dispatch "$@"
    } 2>&1 | tee "$logf"
    rc=$(cat "$rcfile" 2>/dev/null || true)
    rm -f "$rcfile"
    [ -n "$rc" ] || rc=2
    echo "执行记录: $logf"
    return "$rc"
}

main "$@"
exit $?
