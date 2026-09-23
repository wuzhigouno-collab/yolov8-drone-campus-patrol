# 鹰眼巡校 模型训练管线文档（A9）

面向"鹰眼巡校——无人机校园安全智能巡检系统 V1.0"的检测模型训练全流程：
**数据采集 → 标注 → 格式转换 → 训练 → 评估**。
训练策略（plan.md 约定）：以 **VisDrone2019**（无人机俯视行人/车辆公开数据集）为主做迁移学习，再用**自采校园数据二次微调**；正式训练 **imgsz ≥ 960 且开启多尺度训练**，保障俯视小目标召回。

---

## 1. 总体策略与目录约定

- 交付主线模型为 YOLOv8 系列：联调用 `weights/yolov8n.pt`（COCO 预训练），A9 微调产出 `weights/best.pt` 后修改 `src/config.py` 的 `weights_path` 指向它。
- 数据集采用 YOLO 目录结构（图像与标注分目录、同名对应）：

  ```text
  <数据集根>/
  ├── images/train/  xxx.jpg ...
  ├── labels/train/  xxx.txt ...   # 每行：<类别id> <x中心> <y中心> <宽> <高>（0~1 归一化）
  ├── images/val/
  ├── labels/val/
  └── classes.txt                  # 类别清单（转换脚本自动生成）
  ```

- 两套数据两种用法（任选）：
  1. **合并训练**：自采数据与 VisDrone 转换结果合并进同一套目录，`data/campus.yaml` 指向它，一次训练完成；
  2. **二次微调**（推荐）：先在 VisDrone 上训出权重，再以该权重为 `--weights` 起点、用自采数据单独训一轮（YAML 指自采数据集根目录）。二次微调更贴合校园场景。
- 类别表以 `data/campus.yaml` 的 `names` 为准（person/bicycle/car/motorcycle/bus），**转换脚本的 `--classes` 顺序必须与之完全一致**。

## 2. 数据采集规范（自采校园数据）

- 采集设备：无人机航拍，飞行高度覆盖 5/10/15 米三档（与 A10 多高度评估对齐）。
- 画面要求：俯视/大倾角视角为主，覆盖校道、广场、宿舍区、上下课高峰等典型场景；晴天/阴天、不同时段均衡采样。
- 抽帧方式：视频按 1~2 fps 抽帧为 JPG，剔除严重模糊、过曝、画面重复（相邻帧 IoU 过高）的样本。
- 规模建议：首轮每类目标实例 ≥ 500 个，训练/验证按约 8:2 随机划分，**验证集场景不与训练集逐帧重复**。

## 3. 数据标注规范（LabelImg）

- 工具：[LabelImg](https://github.com/tzutalin/labelImg)，格式选 **YOLO**（直接产出 .txt，免转换）；若使用产出 COCO JSON 的工具，走第 4 节转换流程。
- 标注规则：
  - 逐目标拉框，框紧贴目标外缘，类别从 `person / bicycle / car / motorcycle / bus` 中选取；
  - 被遮挡目标按可见部分估计全身框；画面边缘截断目标照常标注；
  - **32×32 像素筛选标准**：宽或高小于 32 像素的目标属于"极小目标"，标注时仍正常标注（不遗漏），但：
    - 训练转换时可用 `tools/coco2yolo.py --min-size 32` 将其滤除（小目标噪声大时可提升收敛稳定性）；
    - 评估时小于 32×32 的目标**不计入漏检统计**（A10 评估口径）。
- 质控：标注完成后抽样 10% 复查框位置与类别；`classes.txt` 类别顺序与 `data/campus.yaml` 一致。

## 4. 格式转换（tools/coco2yolo.py）

适用于 COCO JSON 来源（VisDrone2019 经官方/第三方工具整理为 COCO 格式后，或标注工具导出的 COCO 结果）。LabelImg 直出 YOLO 时跳过本节。

```bash
python tools/coco2yolo.py \
    --coco-json <COCO标注.json> \
    --out <数据集根>/labels/train \
    --classes person bicycle car motorcycle bus \
    --class-map '{"pedestrian": "person", "people": "person", "motor": "motorcycle", "van": "car", "truck": "car", "tricycle": "bicycle", "awning-tricycle": "bicycle"}' \
    --min-size 32
```

- `--classes`：只保留列出的类别，**按给定顺序重新从 0 编号**（与 campus.yaml 的 names 对齐）；不指定则保留全部。
- `--class-map`：类别改名映射（JSON 字符串），在筛选之前生效；上例为 VisDrone2019 → 校园五类的推荐映射。
- `--min-size`：目标宽/高像素下限，任一维度不足即丢弃（对应第 3 节 32×32 标准）。
- 对 train 和 val 各执行一次（分别指向各自 JSON 与输出目录）；转换完成后把图像放入对应 `images/` 目录，并填写 `data/campus.yaml` 的 `path`。

## 5. 训练（tools/train.py）

正式微调（迁移学习，预训练权重起步）：

```bash
python tools/train.py --data data/campus.yaml --epochs 100 --imgsz 960 --batch 8 --lr0 0.01
```

- `--weights` 默认取 `Config().weights_path`（`weights/yolov8n.pt`）；二次微调时传上一轮产出的 `weights/best.pt`。
- 增强参数可配：`--hsv-h/--hsv-s/--hsv-v/--degrees/--translate/--scale/--fliplr/--mosaic`。
- 硬性约束：正式训练 `imgsz < 960` 会被拒绝（plan.md 小目标召回要求）；多尺度训练（`multi_scale=True`）在正式模式强制开启。
- 产物：`runs/<name>/` 下含训练曲线、混淆矩阵、`weights/best.pt`（按验证集 mAP/fitness 自动保存的最优权重）与 `weights/last.pt`；正式模式结束后自动把 `best.pt` 复制为 `weights/best.pt`。

冒烟自验（无真实数据集时仅验证流程）：

```bash
python tools/train.py --data data/campus.yaml --epochs 1 --imgsz 320 --smoke
```

`--smoke` 模式忽略 `--data`，自动在 `data/smoke_dataset/` 下合成迷你数据集（固定随机种子的彩色矩形目标 + YOLO 标注，train 8 张 / val 4 张），跑通"加载权重 → 训练 → 验证 → 保存权重"全流程。**合成数据无可学习语义，产出的精度数字无意义，不复制到 weights/，仅用于确认管线可用**；验证完可整体删除 `runs/` 与 `data/smoke_dataset/`。

## 6. 评估

- 训练自带的验证结果在 `runs/<name>/`（PR 曲线、混淆矩阵、`results.csv` 逐轮 mAP）；
- 对训练出的权重单独评估：

  ```bash
  python -c "from ultralytics import YOLO; print(YOLO('weights/best.pt').val(data='data/campus.yaml', imgsz=960))"
  ```

- 系统级效果评估（检测成功率/误检率/漏检率、按高度分档）走 A10 的 `tools/evaluate.py`，评估口径遵循第 3 节的 32×32 标准。
- 确认效果达标后，把 `src/config.py` 的 `weights_path` 改为 `weights/best.pt`，系统主线即切换到自训模型。

## 7. 常见问题

- **显存不足**：减小 `--batch` 或 `--imgsz`（正式训练 imgsz 不得低于 960，只能减 batch）；
- **类别错位**：检查 `--classes` 顺序与 `campus.yaml` names 是否一致（错位会表现为某类全检成另一类）；
- **Windows 数据加载慢/卡**：`tools/train.py` 冒烟模式已禁用多进程加载；正式训练如异常可加 `--device cpu` 或降 batch 排查。
