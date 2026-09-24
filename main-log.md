# 主日志（Main Log）

> 主 Agent 维护，每个关键步骤追加一行，时间格式 yymmdd hhmm。

| 时间 | 事件 |
|---|---|
| 260911 1420 | 项目骨架初始化：agents/（orchestrator/developer/verifier）、plan.md、lessons.md、reports/ 就绪，待用户启动执行循环 |
| 260912 0113 | 用户要求验收 A1–A3（注：三者已被标 ✅ 但未经独立验证，不合规）；派验证工程师（agent-0）按 plan.md 验收标准实测 |
| 260912 0125 | 验收结果：A1 PASS、A3 PASS、A2 FAIL（既有 tests/out_frames*.json 声称 100 帧但素材仅 50 帧，系编造/张冠李戴）；修 plan.md：A2 状态回退 🔨，验收标准改用 3712 帧素材并加"禁止编造"条款；派生产 Agent 修正 |
| 260912 0140 | A2 修正完成（agent-1）：删旧文件、5G素材真实重跑 100 帧、lessons.md 沉淀教训；复验（agent-0 第2次）：逐帧比对零差异，PASS。查明旧数据实为对 5G 素材的真实输出，问题实质是跑错素材+验收卡素材指定不当。A1–A3 全部 ✅ |
| 260912 0155 | 用户确认优化方案：plan.md 新增 A+ 阶段（A11 ByteTrack跟踪+徘徊检测 / A12 自研切片推理 / A13 FP16+跳帧加速），技术约定补模型选型（主线 yolov8、yolo26n 仅作对比）与 VisDrone 训练数据策略，目录约定增 tracker/loitering/slicer 三模块；orchestrator.md 同步 B 阶段准入条件 |
| 260912 0210 | 用户要求整体验收。A4–B3 共 13 项均为外部会话标记 ✅ 但未经独立验证，状态一律回退 🔨（验收中），分三批派验证工程师实测：批1 A4–A8、批2 A9–A13、批3 B1–B3 |
| 260912 0240 | 三批验收全部完成，13/13 PASS：批1（agent-2）A4–A8 全过；批2（agent-3）A9–A13 全过（实测 RTX 3060、FP16 生效、加速 +88.9%、切片检出 8≥整图 4、跟踪 300 帧未降级）；批3（agent-4）B1–B3 全过（代码文档 Word COM 实测 60 页、每页 50–51 行、页眉合规；手册 25 页 17 图；清单四项与实物吻合）。遗留轻微项：B2 手册 5.10 节一张图注与图内容不符（建议改图注），A10 报告两处 FPS 墙钟波动（计数硬指标全吻合，非编造）。全部任务 ✅，软著材料齐备可提交登记 |
| 260911 1535 | 环境预检通过：Python 3.14.2、ultralytics 8.4.21、测试视频与 GUI 参考就位；A1 派出
| 260911 1538 | A1 生产完成（agent-0），交付 config.py/logger.py/requirements.txt/目录结构；派验证 |
| 260911 1541 | A1 验证 PASS（agent-1，报告 reports/A1-功能合规.md）；A1 ✅；A2 派出 |
| 260911 1548 | A2 生产完成（agent-2）：detector.py/video_source.py/demo_detect.py；注意 farm_guard.mp4 仅 50 帧且无目标类，100 帧验收用 5G智慧农业素材.mp4 补验；派验证 |
| 260911 1555 | A2 验证 PASS（agent-3，报告 reports/A2-功能合规.md）；A2 ✅；A3 派出 |
| 260911 1604 | A3 生产完成（agent-4）：app.py（含 --max-frames/--dump-frame 验收钩子），渲染证据 tests/a3_gui_frame_dets.png；派验证 |
| 260911 1612 | A3 验证 PASS（agent-5，报告 reports/A3-功能合规.md+截屏证据）；A3 ✅。遗留观察项：①GBK 控制台日志中文乱码（-X utf8 正常）②yolov8n.pt 在根目录待移入 weights/ ③150% DPI 下默认窗口超高，后续做 DPI 自适应。A1–A3 批次完成
| 260912 0010 | 用户更新 plan.md：新增 A+ 阶段（A11 ByteTrack+徘徊、A12 切片推理、A13 推理加速）、A9 修订（VisDrone 主数据集、imgsz≥960、权重归置 weights/）；A4 派出 |
| 260912 0020 | A4 生产完成（agent-6）：zone_threshold.py（底部中心锚点+逐目标判定记录）+app.py 增量（区域边框/过滤开关）+test_zone.py 4 用例全过；派验证 |
| 260912 0028 | A4 验证 PASS（agent-7，报告 reports/A4-功能合规.md）；A4 ✅；A5 派出 |
| 260912 0037 | A5 生产完成（agent-8）：frame_confirm.py（ConfirmResult 契约为 A11 预留降级路径）+app.py 增量+test_confirm.py 7 用例全过；派验证 |
| 260912 0045 | A5 验证 PASS（agent-9，报告 reports/A5-功能合规.md）；A5 ✅；A6 派出 |
| 260912 0058 | A6 生产完成（agent-10）：line_counter.py+app.py 增量+test_line.py；真实视频 4 起穿线事件带佐证帧（tests/line_evidence/），13 组合成单测全过，A4/A5 回归 PASS；派验证 |
| 260912 0110 | A6 验证 PASS（agent-11，报告 reports/A6-功能合规.md）；A6 ✅；A7 派出 |
| 260912 0121 | A7 生产完成（agent-12）：archiver.py+app.py 增量+test_archive.py；端到端 1059 帧归档 24 条告警（穿线帧号与 A6 一致），回归全过；派验证 |
| 260912 0132 | A7 验证 PASS（agent-13，报告 reports/A7-功能合规.md）；A7 ✅；A8 派出 |
| 260912 0140 | A8 首次派出因 API 5 小时配额上限失败（agent-14，未产生任何改动）；用户确认后续跑，重新派出 A8 |
| 260912 0205 | A8 生产完成（agent-15）：发现 agent-14 中断前已写出 heatmap.py 与 app.py 增量（未验证），本轮补建 test_heatmap.py 并全面验证通过；端到端 568 帧 80 锚点，热区亮度 242.8 vs 冷区 0.0；派验证（重点核查既有代码真实性与目检结论） |
| 260912 0218 | A8 验证 PASS（agent-16，报告 reports/A8-功能合规.md）；A8 ✅；A9 派出 |
| 260912 0235 | A9 生产完成（agent-17）：train.py（smoke 合成数据离线跑通、正式模式强制 imgsz≥960+多尺度）/coco2yolo.py/campus.yaml/README_TRAIN.md；yolov8n.pt 已移入 weights/ 并同步 config；派验证 |
| 260912 0248 | A9 验证 PASS（agent-18，报告 reports/A9-功能合规.md）；A9 ✅；A11 派出 |
| 260912 0310 | A11 生产完成（agent-19）：tracker.py（ByteTrack，ConfirmResult 契约）+loitering.py+app.py 增量（ID/尾迹叠加）；test_tracker.py 10 用例全过、A5/A6 跟踪模式回归过；注意 model.track() 首跑联网装了 lap==0.5.13；派验证 |
| 260912 0325 | A11 验证 PASS（agent-20，报告 reports/A11-功能合规.md）；A11 ✅；A12 派出 |
| 260912 0342 | A12 生产完成（agent-21）：slicer.py（自研 2×2 切片+NMS 合并，无 sahi 依赖）+config/app.py 增量+test_slicer.py；切片 8 ≥ 整图 4、合并后最大 pair IOU 0.2416；设计取舍：切片开启时 ByteTrack 让位走 IOU 降级（track 内部整图推理无法注入切片）；派验证 |
| 260912 0355 | A12 验证 PASS（agent-22，报告 reports/A12-功能合规.md）；A12 ✅。遗留：①切片增量源于半身影拆分非小目标补检，创新点叙事需真航拍小目标佐证（A10/B 阶段处理）②项目根目录出现 yolo26n.pt（A10 对比实验用品，A10 收尾时归置）；A13 派出 |
| 260912 0412 | A13 生产完成（agent-23）：fp16 开关+跳帧外推+benchmark.py+性能测试.md；实测环境为 RTX 3060 Laptop（CUDA 可用，纠正主 Agent 无卡假设），综合 +82.6% FPS、检出降 9.5% 达标；派验证 |
| 260912 0425 | A13 验证 PASS（agent-24，报告 reports/A13-功能合规.md）；A13 ✅；A10 派出 |
| 260912 0445 | A10 生产完成（agent-25）：evaluate.py+eval_config.yaml+测试报告.md；15米档实测成功率 0.8254/误检 0.0786/漏检 0.1247；5米档无素材留空、10米档 0 检出系类别语义不匹配（探测留痕，待自训权重补测）；yolo26n.pt 已归置 weights/；全量回归过；派验证 |
| 260912 0502 | A10 验证 PASS（agent-26，报告 reports/A10-功能合规.md）；A10 ✅。**阶段 A/A+ 全部通过（A1–A13 共 10 项，0 打回）**，满足进入阶段 B 条件；B1 派出 |
| 260912 0518 | B1 生产完成（agent-27）：export_code_doc.py+源代码文档.docx（60 页：前 30+后 30，自检 9 项 PASS）；注意 src/+tools/ 实测 5773 行超 plan 目标区间 2500–4000，已按超量分支排版；派验证 |
| 260912 0530 | B1 验证 PASS（agent-28，报告 reports/B1-功能合规.md；docx XML 独立复核页眉/PAGE 域/59 分页符/逐行比对一致）；B1 ✅；B2 派出 |
| 260912 0548 | B2 生产完成（agent-29）：操作手册.md（10 功能章+FAQ 8 条+17 张真实截图）+export_manual_docx.py+操作手册.docx（COM 实测 25 页 ≥15）；徘徊告警截图用阈值 1.5s 演示运行真实产出（图注已注明）；派验证 |
| 260912 0603 | B2 验证 FAIL（agent-30，报告 reports/B2-功能合规.md）：4/5 项通过，1 项严重——手册 5.6 节 CSV 示例表前两行位置/置信度数字与真实 archive/alert_log.csv 不符，全项目无出处（编造）；第 1 轮打回 agent-29 |
| 260912 0615 | B2 修正完成（agent-29）：5.6 表改真实行、5.1 改纯格式描述、5.4 改佐证真实字段，docx 重导；复验 PASS（agent-30 第 2 次测试）；B2 ✅；B3 派出 |
| 260912 0628 | B3 生产完成（agent-31）：copyright/材料清单.md（四项材料+登记流程，文件元数据当次实测）；派验证 |
| 260912 0640 | B3 验证 FAIL（agent-32，报告 reports/B3-功能合规.md）：仅 1 项——源代码文档.docx 大小写 97,470 但实测 96,851（该 docx 在 16:45 被重新保存，清单沿用旧 stat 值）；其余项全过；第 1 轮打回 agent-31 |
| 260912 0652 | B3 修正完成（agent-31，全部元数据当场重测+漂移提示行）；复验 PASS（agent-32 第 2 次测试）；B3 ✅。**plan.md 全部 13 个子任务完成：A1–A10、A11–A13、B1–B3；验证全过（B2、B3 各经 1 轮打回修正后通过，无 ⚠️ 项）** |
| 260923 1823 | 阶段 C 开工：主智能体初始化完成——实验计划 V2 与 lessons.md 已读；`.gitignore` 就位（符合 §12 排除清单）；`docs/实验登记.csv` 模板与 `tools/run_exp.sh` 缺失，纳入 R3 补齐；代码侦察确认后处理段在 detector/tracker/slicer 三处重复（R1 对象）；R1–R3 范围与验收标准、实例状态、BATCH_SIZE 待用户确认后派出 |
| 260923 1831 | 用户拍板：R1–R3 按拟范围派出；实例正开机可 SSH；BATCH_SIZE=2（E1a–E1d 两两并行）；git 走 GitHub 私有仓（yolov8-drone-campus-patrol，AGPL-3.0）。R1 生产完成（agent-0）：postprocess.py/class_map.py 新增、三路径改调公共实现、新增 2 测试文件，改动前基线与改动后回归 8 项测试均 exit=0 且关键产出逐项一致；派验证 |
| 260923 1905 | R1 验证 PASS（agent-1，报告 reports/R1-功能合规.md）：6 条验收标准独立实测全过（8 项既有测试重跑与基线逐值一致、后处理唯一归属 postprocess.py、双条件映射三组合实测、合规无编造）；R1 ✅。遗留观察：同进程先 track 后 slice 检出 ±0.001 浮点抖动（FP16 调用历史相关，非缺陷，可稳定复现）。R2 派出 |
| 260923 1920 | R2 验证 PASS（agent-3，报告 reports/R2-功能合规.md）：10 项测试重跑与 R1 基线逐值一致、引擎唯一归属 grep 核验过、装配单例 is 判定同源且加载日志恰 1 条、E6 换权重用例实测过、互斥/降级路径未受损；R2 ✅。R3（Config 快照+run_exp.sh+实验登记.csv 模板）派出 |
| 260923 2015 | R3 生产完成（agent-4）：run_exp.sh（E0–E6 每实验一函数+register+自动快照+日志）与登记 CSV 模板交付；E1b 口径变更（cutout 非 8.4.21 合法参数，改 Albumentations CoarseDropout(p=0.3) 等价并注释）。一验 FAIL（agent-5）：E4 onnx+int8 在 8.4.21 必败；打回复修（agent-4）：E4 INT8 改 TensorRT engine PTQ（校准集=验证集 548 张，实例需装 tensorrt）、E3/E4 val 补显式 name；复验 PASS（agent-5 第 2 次）。**R1–R3 全部 ✅，重构阶段完成**；建仓前清查：tests/ 验收留痕约 201M、copyright/ 37M 现状会进 git，待用户定夺排除项与远端方案 |
| 260923 2040 | git 建仓完成（用户批准）：`.gitignore` 追加 copyright/ 与 tests/ 验收留痕排除；本地 init + 基线 commit `ef27d2a`（87 文件，署名 wuzhigouno-collab+noreply）；远端 https://github.com/wuzhigouno-collab/yolov8-drone-campus-patrol ——**用户决策：仓库保持公开**（偏离实验计划 §7"私有仓"，已周知）；远端自带 README 经 rebase 保留，push 成功且 ls-remote 校验一致。待办：AGPL-3.0 LICENSE 与中英双语 README 随论文阶段补齐 |
| 260923 2055 | P1 首次派出阻塞（agent-6）：SSH 免密被拒（本机 id_ed25519 未授权），按纪律未试密码即返；用户提供实例密码，经 paramiko 一次性写入公钥至实例 authorized_keys（密码不进任何子代理上下文），免密复测通过，容器名 autodl-container-d3044fa0f1-123a0439 与文档一致、4090 在线；P1 重新派出（resume agent-6） |
| 260923 2140 | P1 验证 PASS（agent-7，报告 reports/P1-功能合规.md）：train/val=6471/548 labels 双向零缺失、test=1610 仅 test-dev（challenge 1580 无标注另存未混入，8.4.21 官方脚本口径亲读属实）、ultralytics 8.4.21/torch 2.8.0+cu128 未动、tensorrt 11.3.0.99 import 过、分支 exp/visdrone-v2 顶部 ef27d2a、下载转换全程 tmux、无残留进程。参照数：COCO 权重 yolov8n 跨域 val mAP50=0.0404（仅 sanity，不进论文）。**数据与环境就绪，E0 派出** |
| 260923 2215 | E0 起跑失败（agent-8，按红线未改代码）：run_exp.sh 两 bug——①DATA 默认 visdrone.yaml 大小写错（8.4.21 内置为 VisDrone.yaml，Windows 冒烟测不出的跨平台坑）②project=runs 相对路径被 8.4.21 get_save_dir 前置 runs/detect/，快照目录必然失配（阻断）。实测 batch=32 真 OOM→自动降 16 后显存贴顶 22.8G，OOM 重试路径后 multi_scale 插值 ZeroDivisionError 崩溃（干净起跑待查）。实例零残留进程、日志两份齐备。打回 R3 生产方修脚本（resume agent-4） |
| 260923 2240 | run_exp.sh 修复+复验 PASS（agent-4/agent-5 第 3 次）：共修 4 缺陷（+EPOCHS/IMGSZ 覆盖、+CRLF→LF——Git Bash 透明吞 \r 掩盖）；commit 9caa89a 推 exp/visdrone-v2（用户授予阶段 C 实验分支常驻 git 授权）。E0 重跑（agent-8）：脚本修复实测生效（save_dir 正确、快照落盘 batch=16 口径一致），但 **ZeroDivisionError 干净起跑确定性复现**（iter 49/405，调用链 preprocess_batch→F.interpolate→_compute_scale→in_size/out_size）——判定为 multi_scale 在 8.4.21+torch 2.8.0+cu128 的环境级 bug，与 OOM 无关；另实测 batch=16 显存 22.7G 贴顶、OOM→CPU 回退 5 次/49it 偏密集。实验矩阵全线暂停，派根因调查（只调查不改动）；教训：实例验证改动须归位或显式交接（agent-4 实例侧改动致 agent-8 merge 受阻，字节一致零损失，已裁决 stash 归位） |
| 260923 2310 | 根因查明（agent-9）：8.4.21 起 multi_scale 由 bool 改 float 语义（PR #23284），True 被按 1.0 解释→缩放下界 0→每 batch 1.64% 概率 size=0→interpolate([0,0]) 崩（405 batch 下不崩率 0.13%，故"epoch1 必崩"）；det 模式 decomp 路径只是异常形态，非开关可解。**裁决：multi_scale=0.5**——官方维护者 issue #23480 确认等价旧版 True 的 ±50%，恢复计划 §4 原意非口径变更；agent-9 实例同口径实测 1 epoch+val 零崩溃，显存峰值下降。打回 agent-4 修脚本该行→复验→推送→E0 重跑 |
| 260924 0800 | 会话续接：工作区已含 multi_scale=0.5 修正（agent-4 交付：run_exp.sh 三处调用点+smoke_train 改 CLI 真实路径冒烟、train.py 同步、lessons 新增条目，未提交未推送）；按 2310 既定路线派 agent-5 第 4 次测试，PASS 后提交推送 exp/visdrone-v2，再 resume agent-8 重跑 E0 |
| 260924 0810 | R3 第 4 次测试 PASS（agent-0，新会话 id 重映射：旧 agent-5 角色）：6 项验收全过——diff 无声明外改动、8.4.21 源码亲读 float 语义与崩溃链属实（default.yaml:39、detect/train.py:120-135）、本机冒烟 exit=0 无崩溃快照 11 项一致、train.py 冒烟+R1 基线 11 测试全过、lessons 四段式合规、§4 口径未松动。观察项（不阻塞，待轻量批处理）：README_TRAIN.md:74 旧值、脚本注释行号区间偏窄。提交推送后 E0 重跑 |
