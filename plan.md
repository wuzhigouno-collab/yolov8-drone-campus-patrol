# 鹰眼巡校 — 任务计划与验收标准

> 本文件由主 Agent（编排者）维护，是唯一任务状态来源。生产/验证子代理**禁止修改本文件**。
> 状态标记：⏳ 待做 | 🔨 进行中 | ✅ 通过 | ⚠️ 低质量通过（达修正上限）

## 项目目标

开发"鹰眼巡校——无人机校园安全智能巡检系统 V1.0"（Python 桌面软件），完成软著登记全套材料（60 页源代码文档 + 操作手册），用于吉利学院大创项目结项（校级/省级合格标准：学生为第一著作权人，软著 1 项获受理）。

## 技术约定（生产 Agent 必须遵守）

- 语言/框架：Python 3.10+、ultralytics>=8.4（YOLO，内置 ByteTrack/BoT-SORT 跟踪）、opencv-python、tkinter（标准库 GUI）、numpy、Pillow
- 模型权重：开发联调用 ultralytics 自动下载的 `yolov8n.pt`（COCO 预训练，默认可检 person/car/bicycle 等）；自训权重产出后放 `weights/` 目录替换，代码中权重路径走配置项
- 模型选型：申报书为 YOLOv8，交付主线保持 yolov8 系列（联调 `yolov8n.pt` → A9 微调产出 `weights/best.pt`）；`yolo26n` 仅用于 A10 评估报告中的对比实验，不进交付主线
- 训练数据策略：以 VisDrone2019（无人机俯视行人/车辆公开数据集）为主做迁移学习，自采校园数据二次微调；训练输入 imgsz ≥960、开启多尺度训练，保障俯视小目标召回
- 新增配置项（随模块任务加入 `config.py`）：`fp16` 半精度开关、`detect_interval` 跳帧间隔、`slice` 切片开关与重叠率、徘徊检测的停留阈值秒数与区域坐标
- GUI 风格参考（只读，不照搬）：`C:/Users/OS/Desktop/farm_guard/scripts/farm_guard_gui_2.py`
- YOLO 训练管线参考（只读）：`D:/py_workdir/light_work/homework/3/1772785083142-hw5.zip` 内 `hw5/hw5-code/`（train.py、convert_dataset_coco2yolo.ipynb、d2-city.yaml）；参考时解压到本项目 `reference/hw5-code/`，该目录**不计入软著代码量、不进入导出范围**
- 开发测试视频：`C:/Users/OS/Desktop/farm_guard/videos/` 下的 mp4（含人/车画面；`farm_guard.mp4` 仅 50 帧，`5G智慧农业素材.mp4` 3712 帧，帧数要求高的验收用后者）；项目内 `tests/assets/` 只放小段剪辑
- 代码注释与 docstring 用中文、写清功能意图（软著代码文档要给人审）；禁止把模型权重文件、视频文件放进交付目录
- 软著代码量统计范围：`src/` + `tools/`（目标总行数 2500–4000）；`reference/`、`tests/assets/`、`archive/`、`weights/` 一律排除

## 目录约定

```
eagle_eye_patrol/
├── agents/          # 角色提示词（已就绪）
├── plan.md          # 本文件
├── lessons.md       # 经验库
├── main-log.md      # 主日志
├── reports/         # 验证报告
├── reference/       # 只读参考（hw5 解压于此，不进软著）
├── src/             # 系统源代码（软著主体）
│   ├── app.py           # GUI 入口
│   ├── config.py        # 集中配置
│   ├── logger.py        # 日志
│   ├── detector.py      # YOLO 推理封装
│   ├── video_source.py  # 视频源抽象（文件/摄像头）
│   ├── zone_threshold.py    # A4 分区域阈值
│   ├── frame_confirm.py     # A5 多帧确认
│   ├── line_counter.py      # A6 越线计数
│   ├── archiver.py          # A7 截图归档与告警日志
│   ├── heatmap.py           # A8 热力图
│   ├── tracker.py           # A11 ByteTrack 跟踪与轨迹管理
│   ├── loitering.py         # A11 徘徊检测
│   └── slicer.py            # A12 切片推理
├── tools/           # 训练/评估/导出/基准脚本（计入软著代码量）
├── tests/           # 测试脚本与小型素材
├── weights/         # 模型权重（不进版本交付；yolov8n.pt 亦归置于此）
├── archive/         # 运行期告警归档输出
├── docs/            # 操作手册、测试报告、性能测试
└── copyright/       # 软著材料（60页代码文档、清单）
```

---

## 阶段 A：系统开发

### A1 项目骨架与基础设施 ✅（260912 独立验证 PASS）
- 交付：`src/config.py`、`src/logger.py`、`requirements.txt`、上述目录结构
- 验收：`pip install -r requirements.txt` 成功；`python -c "from src.config import Config; c=Config(); print(c.to_dict())"` 正常输出全部配置项（含：中央/边缘区比例、两区置信度阈值、确认帧数 N、警戒线坐标、归档目录、权重路径）

### A2 检测引擎核心 ✅（260912 二轮验证通过）
- 依赖：A1
- 交付：`src/detector.py`（YOLO 推理封装：加载权重、单帧推理、返回统一结构 `[{cls,conf,box,center}, ...]`）、`src/video_source.py`（文件/摄像头两种源，统一 `read() -> frame` 接口）
- 验收：`python tools/demo_detect.py --video C:/Users/OS/Desktop/farm_guard/videos/5G智慧农业素材.mp4 --out tests/out_frames.json` 跑满 100 帧无异常退出，输出 JSON 每帧含检测结果列表。帧数必须与素材实际帧数一致（该素材 3712 帧，farm_guard.mp4 只有 50 帧不可用于本条），输出数据必须由真实运行产生，禁止编造或张冠李戴

### A3 GUI 主框架 ✅（260912 独立验证 PASS）
- 依赖：A2
- 交付：`src/app.py`（tkinter：视频画面区、视频源选择、开始/停止按钮、状态栏、告警列表区、检测框实时叠加显示）
- 验收：`python src/app.py --video <测试视频>` 启动无报错；视频逐帧播放且检测框/类别/置信度实时渲染；界面标题为"鹰眼巡校——无人机校园安全智能巡检系统 V1.0"

### A4 分区域差异化阈值模块 ✅（260912 验证 PASS）
- 依赖：A3
- 交付：`src/zone_threshold.py`；GUI 增加区域边框显示开关
- 验收：`python tests/test_zone.py` 通过——边缘区置信度 0.35 的目标在边缘阈值 0.5 时被过滤、边缘阈值 0.3 时保留，且两种情况下中央区判定结果不受边缘配置影响（断言+日志证据）

### A5 多帧 IOU 确认降噪模块 ✅（260912 验证 PASS）
- 依赖：A3
- 交付：`src/frame_confirm.py`（相邻帧 IOU 匹配，目标连续 N 帧（默认 3，可配置）稳定出现才标记为"已确认"，未确认目标不进入告警与计数）
- 验收：`python tests/test_confirm.py` 通过——单帧闪现目标 0 条告警；连续出现 3 帧的目标在第 3 帧产生确认记录；IOU 阈值可配置且生效

### A6 越线方向计数模块 ✅（260912 验证 PASS）
- 依赖：A5
- 交付：`src/line_counter.py`（配置定义虚拟线段；已确认目标轨迹穿线时判定方向 进/出；画面实时渲染警戒线与双向计数）
- 验收：`python tests/test_line.py --video <测试视频>` 输出的穿线事件（时间、方向、类别）与人工抽查结果一致；GUI 画面计数与日志一致

### A7 告警截图归档与日志 ✅（260912 验证 PASS）
- 依赖：A5
- 交付：`src/archiver.py`（告警触发自动截取带框画面存 `archive/`；告警日志 CSV：时间戳、事件类型、类别、置信度、位置、截图路径）
- 验收：运行测试视频触发告警后，`archive/` 下存在对应截图且图中带检测框；CSV 每行字段齐全、时间与事件吻合

### A8 人流车流热力图模块 ✅（260912 验证 PASS）
- 依赖：A5
- 交付：`src/heatmap.py`（累积已确认目标的底部中心点，高斯累积渲染，导出 PNG；GUI 提供"导出热力图"按钮）
- 验收：跑完测试视频后导出 `archive/heatmap.png`；目标密集区域亮度显著高于无目标区域（程序断言像素均值差 + 人工目检）

### A9 模型训练管线 ✅（260912 验证 PASS）
- 依赖：A1（独立性强，可提前做）
- 交付：`tools/train.py`（迁移学习微调：预训练权重起步、epoch/学习率/增强参数可配、imgsz ≥960、多尺度训练开启、按验证集 mAP 保存最优权重）、`tools/coco2yolo.py`（COCO→YOLO 标注转换，参考 hw5 实现）、`data/campus.yaml` 模板、`tools/README_TRAIN.md`（数据采集标注规范：LabelImg、32×32 像素筛选标准、VisDrone2019 主数据集与自采数据二次微调的完整流程）；收尾杂项：把项目根目录的 `yolov8n.pt` 移入 `weights/` 并同步 `config.py` 默认权重路径
- 验收：`python tools/train.py --data data/campus.yaml --epochs 1 --imgsz 320 --smoke` 冒烟跑通（仅验证流程，产出 runs/ 目录与权重文件）；README 覆盖采集→标注→转换→训练→评估全流程

### A10 系统集成与多高度测试 ✅（260912 验证 PASS）
- 依赖：A4–A8
- 交付：`tools/evaluate.py`（对测试视频统计检测成功率/误检率/漏检率，支持按 5/10/15 米高度分组输入）、`docs/测试报告.md`
- 验收：`python tools/evaluate.py --config tests/eval_config.yaml` 运行成功；`docs/测试报告.md` 含三个高度档位的指标表格与结论，数据来自真实运行输出（禁止编造）
- 可选项：同素材下 yolov8n 与 yolo26n 的对比结果可附入报告，作为评估素材（不进交付主线）

---

## 阶段 A+：性能与功能优化（创新点增强）

### A11 ByteTrack 跟踪集成与徘徊检测 ✅（260912 验证 PASS）
- 依赖：A5
- 交付：`src/tracker.py`（封装 ultralytics `model.track()`，ByteTrack 轨迹管理，输出带持久 ID 的目标轨迹；A5 的确认逻辑升级为基于轨迹生命周期——轨迹连续存在 N 帧才确认，原 IOU 匹配实现保留为无跟踪器时的降级路径）、`src/loitering.py`（同一 ID 在配置区域内停留超过阈值秒数触发徘徊告警，区域与阈值走 config）；GUI 叠加显示目标 ID 与轨迹尾迹
- 验收：`python tests/test_tracker.py --video <测试视频>` 通过——同一目标跨帧 ID 一致（ID 切换次数 ≤ 可配置阈值）；构造停留超阈值用例触发徘徊告警并写入告警日志；A5、A6 既有测试在跟踪模式下仍通过

### A12 切片推理模块 ✅（260912 验证 PASS）
- 依赖：A2
- 交付：`src/slicer.py`（自研，不引入 sahi 依赖：帧切分为 2×2 重叠切片，重叠率可配置；逐片检测后坐标映射回原图，合并 NMS 去重；`config.py` 与 GUI 提供开关）
- 验收：`python tests/test_slicer.py` 通过——同一张高空测试图，切片模式检出数 ≥ 整图模式，且合并后无重复框（任意两框 IOU < 0.5 断言）；跨切片边界目标的合并逻辑有专项单元测试

### A13 推理加速 ✅（260912 验证 PASS）
- 依赖：A11
- 交付：FP16 半精度推理开关（config，无 N 卡自动降级并日志说明）；跳帧检测+跟踪器外推（`detect_interval` 可配置，中间帧用轨迹外推渲染）；`tools/benchmark.py`、`docs/性能测试.md`
- 验收：`python tools/benchmark.py --video <测试视频>` 输出优化前后两组实测 FPS；开启优化后 FPS 提升 ≥30%（无 N 卡环境只看跳帧单项 ≥50%），且检出数下降 ≤10%；`docs/性能测试.md` 数据来自真实运行（禁止编造）

---

## 阶段 B：软著材料

### B1 60 页源代码文档 ✅（260912 验证 PASS）
- 依赖：A10–A13 全部完成（代码定型）
- 交付：`tools/export_code_doc.py`、`copyright/源代码文档.docx`（或 PDF）
- 验收：导出脚本将 `src/`+`tools/` 代码排版为：页眉含"鹰眼巡校——无人机校园安全智能巡检系统 V1.0"、每页 ≥50 行、右上角页码；总页数 = 60（代码超量时取前 30 页+后 30 页连续）或代码不足时全量且 ≥30 页；`python tools/export_code_doc.py --check` 自检输出全部合规项 PASS

### B2 操作手册 ✅（260912 验证 PASS）
- 依赖：A3–A8、A11–A12（需真实运行截图）
- 交付：`docs/操作手册.md` 及导出的 `copyright/操作手册.docx`
- 验收：手册含 软件简介/运行环境/安装部署/10 个功能章节（检测显示、分区阈值、多帧确认与跟踪、越线计数、徘徊检测、告警归档、热力图、切片推理、训练管线、测试评估）/FAQ，每功能章节 ≥1 张真实运行截图，导出 docx ≥15 页

### B3 申请材料清单 ✅（260912 验证 PASS）
- 依赖：B1、B2
- 交付：`copyright/材料清单.md`
- 验收：清单覆盖 申请表填写要点（软件全称、版本号 V1.0、开发完成日期、著作权人=学生本人）/源代码文档/操作手册/身份证明 四项，与 `copyright/` 实际文件一一对应，附中国版权保护中心网上登记流程步骤

---

## 执行备注

- 修正循环上限：每个子任务最多打回 3 轮，仍 FAIL 标 ⚠️ 并在收尾汇总
- BATCH_SIZE 默认 1（串行派活），用户可另行指定
- 可并行性：A4/A5/A6 互相独立；A9、A12 仅依赖已完成任务，可随时插入；A11 依赖 A5，A13 依赖 A11
- 阶段 A 与 A+ 全部 ✅/⚠️ 后才进入阶段 B
