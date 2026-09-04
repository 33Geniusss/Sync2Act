# Sync2Act 项目实施任务书（直接交给 Codex）

## 你的任务

请在当前工作区从零设计并实现一个可公开发布到 GitHub 的完整项目：**Sync2Act**。

不要只输出方案、伪代码、Notebook 或零散脚本。请实际创建项目文件、实现代码、运行测试、修复问题，并交付一个可启动的 PyTorch 应用。除非遇到必须由用户决定的重大范围变化，否则请自行采用合理默认值持续推进。

允许你在项目目录内创建和修改文件、安装必要的项目依赖、运行非破坏性命令和测试。不要上传代码、创建远程仓库、发布 Release、购买服务或执行其他外部写操作，除非用户随后明确授权。

## 一、项目定位

Sync2Act 是一个研究和演示“机器人数据质量如何影响模仿学习策略”的 PyTorch 工具包与桌面应用。

核心问题：

1. 图像、机器人状态和动作发生时间错位后，策略性能下降多少？
2. 丢帧、动作噪声和状态异常分别会造成什么影响？
3. 一个显式利用数据质量信息的策略，能否比普通策略更加鲁棒？

项目应当形成下面这条完整工作流：

```text
机器人示范数据
    -> 可控的数据损坏
    -> PyTorch 策略训练
    -> 离线指标与闭环 rollout 评估
    -> 曲线、表格、轨迹和视频对比
    -> 可导出的实验报告
```

本项目的目标不是训练通用 VLA 大模型，也不是简单包装现有训练命令。代码必须清楚展示 PyTorch 数据管线、模型定义、损失函数、训练循环、checkpoint、评估和性能测量。

## 二、强制交付结果

完成后，仓库至少应包含：

1. 可安装的 Python 包 `sync2act`。
2. 可启动的 **PySide6 原生可视化桌面窗口**。
3. BC-MLP 基线策略。
4. ACT-Lite 动作块策略。
5. 在 ACT-Lite 上扩展的 Quality-Aware ACT。
6. 可复现的数据损坏模块。
7. 训练、评估、checkpoint 与报告生成流程。
8. 无 GPU、无外部数据时也能运行的合成演示数据和 smoke test。
9. 单元测试、端到端测试和 GitHub Actions 配置。
10. 面向 GitHub 访客的高质量 README，包括架构图、界面截图位置、运行命令、实验设计和结果表。

只有命令行、Notebook或网页而没有桌面窗口，不满足本任务要求。

## 三、推荐技术栈

- Python 3.11
- PyTorch
- torchvision
- PySide6
- PyQtGraph：桌面窗口中的实时训练曲线和交互图表
- NumPy、Pandas
- Pillow 或 OpenCV：图像与视频帧处理
- PyYAML：实验配置
- pytest、pytest-qt
- ruff
- 可选：TensorBoard、LeRobot、Gymnasium/PushT、Hugging Face Hub
- 可选发布阶段：PyInstaller Windows 打包

实现前检查当前稳定兼容版本并在 `pyproject.toml` 中给出合理版本范围。核心测试不得依赖网络，也不得要求下载大型数据集。

## 四、模型定义

### 4.1 BC-MLP

用途：提供最简单、训练快速、便于排错的行为克隆基线。

最低支持两种输入模式：

1. `state_only`：机器人状态经过 MLP，预测下一步动作。
2. `image_state`：RGB 图像经过轻量 CNN，状态经过 MLP encoder，融合后预测下一步动作。

形式：

```text
image[t] + state[t] -> action[t]
```

实现要求：

- 明确的输入输出 shape 检查。
- 状态和动作归一化。
- MSE 或 Smooth L1 action loss。
- 能在合成数据上快速过拟合一个小 batch，用于验证训练管线。

### 4.2 ACT-Lite

用途：作为项目的主要时序模仿学习模型，一次预测未来一段动作，而不是只预测下一步。

形式：

```text
image[t] + state[t] -> action[t : t + H]
```

最低结构：

- 轻量 CNN 图像编码器。
- 机器人状态 embedding。
- 可学习 action queries。
- Transformer encoder 或小型 encoder-decoder。
- 位置编码。
- padding mask。
- 长度可配置的 action chunk，例如默认 `H=16`。
- 推理时支持 receding-horizon execution。
- 支持 temporal ensembling，对多个时刻重复预测的同一动作进行加权融合。

第一版可不实现完整 ACT 的 CVAE 潜变量结构，但必须在 README 中明确说明与完整 ACT 的区别，不能将 ACT-Lite 宣称为原论文的完整复现。

### 4.3 Quality-Aware ACT

用途：这是本项目的主要改进方法，用于验证质量信息能否缓解坏数据造成的性能下降。

它应当复用 ACT-Lite 主体，不要复制一套难以维护的独立代码。

MVP 至少实现：

1. **质量加权损失**：每个样本或时间步有 `quality_score`，低质量样本对总 loss 的贡献更小。
2. **质量特征输入**：将质量分数、缺失 mask 或时间偏差 embedding 作为额外 token/feature 输入模型。

推荐损失结构：

```text
total_loss = weighted_action_loss
             + lambda_smooth * trajectory_smoothness_loss
             + optional_regularization
```

所有开关和权重必须可在 YAML 配置中设置，便于消融实验。

## 五、数据格式与损坏模块

定义一个统一的 episode 接口，至少包含：

```python
{
    "observation.image": Tensor[T, C, H, W],
    "observation.state": Tensor[T, state_dim],
    "action": Tensor[T, action_dim],
    "timestamp": Tensor[T],
    "quality_score": Tensor[T],
    "missing_mask": Tensor[T],
}
```

先提供小型合成数据集，使项目在没有 LeRobot 和仿真器时仍能演示完整流程。随后提供可选的 LeRobot dataset adapter；避免把特定 LeRobot 版本的内部实现散落在核心模块中。

至少实现以下损坏类型：

### 5.1 Temporal Shift

- 可选择移动 image、state 或 action。
- 支持正负帧偏移。
- 边界处理方式可配置：drop、repeat、zero 或 mask。

### 5.2 Frame Drop

- 按给定概率随机丢帧。
- 支持 previous-frame、zero-frame 和 interpolation 替代方式。
- 同时更新 `missing_mask` 和 `quality_score`。

### 5.3 Action Noise

- 高斯噪声。
- 稀疏 spike。
- 固定或随机 action delay。

### 5.4 State Anomaly

- 状态尖峰。
- 短时缺失。
- 某个维度 stuck-at-constant。
- 时间戳 jitter。

损坏过程必须：

- 接受明确随机种子。
- 不修改原始数据。
- 输出完整 provenance 元数据。
- 记录损坏类型、参数、随机种子和受影响索引。
- 相同输入与随机种子产生逐元素一致的结果。

## 六、实验设计

必须明确区分两类实验：

### 6.1 训练数据损坏

```text
损坏数据训练 -> 干净环境测试
```

用于研究坏示范会如何污染策略学习。

### 6.2 运行时观测损坏

```text
干净数据训练 -> 带传感器故障的环境测试
```

用于研究相机延迟、丢帧等部署问题。

MVP 实验矩阵：

- 模型：BC-MLP、ACT-Lite、Quality-Aware ACT。
- 数据：clean、image shift 1/2/4 frames、frame drop 5%/10%/20%、action noise 三档。
- 至少支持多个 seed；快速演示可以运行 1：1个 seed，正式 benchmark 默认至少3个 seed。

不得编造 benchmark 数值。如果当前环境无法完成正式训练，结果表中应明确标注 `not run`，同时提供能让用户运行正式实验的命令。

## 七、评估指标

至少实现并保存：

- Action MSE 和 MAE。
- 动作/轨迹平滑度。
- jerk 或离散二阶差分统计。
- 每个 episode 的结果。
- 推理延迟 P50、P95。
- 模型参数量。
- 训练时间。
- 若仿真环境可用：rollout success rate、return、完成步数。

离线 action loss 不能代替闭环成功率。正式结果应优先展示 rollout success rate；若仿真尚不可用，界面和报告必须明确标记当前结果为 offline-only。

## 八、可视化桌面窗口（强制）

创建一个 Windows 友好的 PySide6 应用，建议入口：

```bash
python -m sync2act.gui
```

或安装后：

```bash
sync2act-gui
```

界面至少包含以下页面或标签：

### 8.1 Overview

- 当前数据集、episode 数、帧数、输入 shape。
- 当前设备：CPU/CUDA。
- 最近训练状态。
- 当前最佳模型和主要指标。
- 一键加载 bundled demo dataset。

### 8.2 Dataset Inspector

- episode 和时间步选择器。
- RGB 帧预览。
- state/action 时间序列曲线。
- 时间戳与采样间隔显示。
- missing mask 和 quality score 可视化。
- 原始数据与损坏数据并排对比。

### 8.3 Corruption Studio

- 选择损坏类型。
- 使用 slider/spin box 配置 shift、drop probability、noise sigma 等参数。
- 设置随机种子。
- 实时预览损坏前后的一小段数据。
- Apply、Reset、Save Config 按钮。
- 不得覆盖原始数据。

### 8.4 Training Monitor

- 选择 BC-MLP、ACT-Lite 或 Quality-Aware ACT。
- 加载/保存 YAML 配置。
- Start、Pause/Stop、Resume 控制。
- 实时显示 train loss、validation loss、learning rate、step、epoch 和 ETA。
- 显示设备、显存（可用时）和 checkpoint 路径。
- 训练必须在后台 worker 中运行，不能冻结 UI 主线程。
- 用户停止训练时要安全保存可恢复 checkpoint。

### 8.5 Evaluation & Compare

- 选择多个 checkpoint。
- 并排显示模型指标。
- 展示 corruption level vs metric 曲线。
- 显示动作真值与预测轨迹。
- 如果有 rollout 视频，支持播放或逐帧查看。
- 清楚区分 offline metric 和 rollout metric。

### 8.6 Report

- 生成汇总表格和图表。
- 导出 HTML；可选导出 PDF/CSV/JSON。
- 报告包含实验配置、Git commit（若仓库可用）、依赖版本、随机种子和运行时间。
- 未运行的数据不可显示为0或伪造值。

### 8.7 界面质量要求

- 使用清晰的侧栏或 tab 导航。
- 合理的留白、字号、颜色层级和状态反馈。
- 长任务显示进度和可取消状态。
- 错误通过可理解的对话框/日志面板呈现，不得让应用直接崩溃。
- 控件在常见 Windows 显示缩放下不得重叠或截断。
- 界面启动后应有可演示的数据，而不是一片空白。
- 最终必须实际启动并检查窗口；如果环境支持截图，保存一张 README 使用的界面截图。

## 九、CLI

桌面界面之外还应提供 CLI，确保实验可以自动化：

```bash
sync2act demo
sync2act corrupt --config configs/corruption/shift_2.yaml
sync2act train --config configs/train/act_lite.yaml
sync2act evaluate --config configs/eval/default.yaml
sync2act benchmark --config configs/benchmark/mvp.yaml
sync2act report --runs runs/ --output reports/benchmark.html
```

GUI 与 CLI 必须调用同一套业务逻辑，不能分别维护两套训练或数据代码。

## 十、推荐仓库结构

```text
sync2act/
├── pyproject.toml
├── README.md
├── LICENSE
├── configs/
│   ├── corruption/
│   ├── train/
│   ├── eval/
│   └── benchmark/
├── src/sync2act/
│   ├── data/
│   ├── corruptions/
│   ├── policies/
│   ├── training/
│   ├── evaluation/
│   ├── reporting/
│   ├── gui/
│   └── cli.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── gui/
├── examples/
├── assets/
├── reports/
└── .github/workflows/
```

运行产物放入被 Git 忽略的 `runs/`、`checkpoints/` 和本地数据目录。不要向 Git 提交大型数据集、缓存、模型权重、密钥或用户机器绝对路径。

## 十一、测试与验收

至少完成以下自动化检查：

### 数据测试

- 各种损坏操作输出 shape 正确。
- 相同 seed 完全可复现。
- 原始输入不被修改。
- quality score 和 missing mask 与实际损坏一致。
- episode 边界不会被跨越。

### 模型测试

- 三个模型的 forward shape 正确。
- padding mask 正确生效。
- weighted loss 在全1权重时与普通 loss 一致。
- 全0有效权重等边界情况不会产生 NaN。
- checkpoint save/load 后输出一致。
- temporal ensemble 行为可测试。

### 训练测试

- BC-MLP 可以在极小合成 batch 上过拟合。
- ACT-Lite 可以完成若干训练 step 并降低 loss。
- 能从 checkpoint 恢复 optimizer、scheduler 和 step。
- CPU smoke test 在合理时间内完成。

### GUI 测试

- 主窗口可以创建和关闭。
- bundled demo dataset 可以加载。
- 切换损坏参数会更新预览。
- 后台训练不会阻塞界面事件循环。
- 无数据、无 checkpoint、无 CUDA 等状态有明确提示。

### 最终人工验收

1. 全新环境按 README 能安装项目。
2. `sync2act demo` 能完成一次微型端到端流程。
3. GUI 能启动并显示 demo dataset。
4. 用户能在 GUI 中制造一种数据损坏。
5. 用户能启动一次短训练并看到曲线更新。
6. 用户能加载 checkpoint 并看到预测/真值对比。
7. 用户能导出一份不含伪造数据的 HTML 报告。
8. 所有自动化测试通过。

## 十二、分阶段实施

请按下面顺序实现，每一阶段结束都运行相关测试；不要先搭大量空壳文件。

### Phase 1：可运行骨架

- Python 包和配置系统。
- 合成 episode 数据。
- BC-MLP。
- 最小训练与评估循环。
- 最小 PySide6 窗口，可加载 demo、启动短训练并画 loss。
- 单元测试和 README 快速开始。

### Phase 2：数据质量实验

- 四类 corruption。
- Dataset Inspector 和 Corruption Studio。
- provenance 和可复现性测试。
- CLI 完整贯通。

### Phase 3：动作块模型

- ACT-Lite。
- temporal ensembling。
- checkpoint 与恢复。
- Training Monitor 完善。

### Phase 4：质量感知方法

- Quality-Aware ACT。
- weighted loss 和 quality feature。
- 消融配置。
- Compare 页面和报告生成。

### Phase 5：真实基准与发布质量

- 可选 LeRobot/PushT adapter。
- 闭环 rollout。
- 多 seed benchmark。
- 界面视觉完善、截图和演示 GIF。
- 可选 PyInstaller Windows 构建。

如果当前环境缺少 GPU、仿真器或网络，仍应完成不依赖它们的阶段，并将外部集成做成可选依赖。不要因为无法跑大型正式实验而停止核心工程实现。

## 十三、README 必须回答的问题

GitHub访客应在一分钟内知道：

1. Sync2Act 解决什么问题？
2. 为什么机器人数据质量会影响模仿学习？
3. BC-MLP、ACT-Lite 和 Quality-Aware ACT 分别是什么？
4. 如何在3条命令内启动 demo 和 GUI？
5. 当前哪些结果来自真实运行，哪些尚未运行？
6. 如何复现实验？
7. 界面长什么样？
8. 如何接入自己的 episode 或 LeRobot 数据集？

README 首页优先放置：项目一句话介绍、GUI 截图、短演示 GIF、快速开始、核心结果表和架构图。不要把大段安装说明放在最吸引注意力的位置。

## 十四、工程原则

- 不伪造实验数据、成功率、性能数字或截图。
- 不把训练 loss 当作闭环任务成功率。
- 不把 ACT-Lite 描述成完整 ACT 复现。
- 不覆盖用户原始数据。
- 不让 GUI 线程执行长时间训练。
- 不把绝对路径硬编码进源代码。
- 不要求测试访问网络。
- 不以“代码能导入”代替实际端到端运行验证。
- 优先实现纵向贯通的小功能，再逐步扩展。
- 遇到依赖不可用时提供清晰的可选降级路径。

## 十五、Codex 工作方式和最终汇报

开始时先检查当前目录、已有文件和可用运行环境，然后直接实施。保持改动集中、模块边界清楚，并在每个阶段运行相关测试。安全的本地读取、代码修改和测试无需逐项请求用户确认。

最终回复必须包含：

- 已实现功能摘要。
- GUI 启动命令。
- CLI 快速演示命令。
- 测试结果。
- 实际运行过的实验及其真实结果。
- 尚未运行或受环境限制的部分。
- 主要文件入口。
- 下一步最有价值的扩展。

不要在只完成计划、空壳界面或未验证代码时声称项目完成。
