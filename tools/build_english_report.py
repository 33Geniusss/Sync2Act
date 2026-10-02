"""Build the English LaTeX counterpart, preserving Chinese-source result cells.

Run from the repository root, then compile the generated source with XeLaTeX.
The translation catalog is deliberately explicit: changed Chinese prose must be
reviewed instead of silently disappearing from the English report.
"""

from pathlib import Path

from sync2act.reporting.localization import english_text

REPORT_TEXT = {
    "第一步预测的总体结果": "Overall results with first-step prediction",
    "本节及第 8 节的模态分析与模型配对采用第一步预测指标；时间集成的总体结果与两种评估方式的 MSE 对照见第 6 节。": "This section and Section 8 analyze modalities and paired models using first-step metrics. Section 6 reports ensemble results and MSE comparisons between evaluation methods.",
    r"\textbf{固定 decay=0.7 的时间集成未降低总体实际 MSE。}270 组逐运行 MSE 变化百分比的均值为 $+21.20\%$；退化比值下降不能代替实际误差改善，也不构成闭环控制收益的证据。": r"\textbf{Ensembling at fixed decay=0.7 does not reduce overall actual MSE.} The mean per-run MSE percentage change across 270 runs is $+21.20\%$. Lower degradation ratios do not establish lower actual error or closed-loop benefits.",
    "混合模态数据损坏研究": "Mixed-Modality Data Corruption Study",
    "混合模态损坏研究": "Mixed-Modality Study",
    "完整真实数据集、六组消融、三个随机种子": "Full datasets, six ablations, three random seeds",
    "实验规模": "Experiment size",
    r"3 数据集 $\times$ 6 消融 $\times$ 5 条件 $\times$ 3 seeds": r"3 datasets $\times$ 6 ablations $\times$ 5 conditions $\times$ 3 seeds",
    "完成状态": "Completion",
    "270 / 270 组成功完成并通过完整性审计": "270 / 270 runs completed and passed integrity audits",
    "训练范围": "Training scope",
    "完整 episode、完整 frame、全部相机；仅训练集损坏": "Full episodes, all frames and cameras; training-only corruption",
    "核心指标": "Main metrics",
    "干净测试集 action MSE，以及相对同模型同 seed 干净基线的 MSE 比值": "Clean-test action MSE and ratios to the same model/seed's clean-training baseline",
    "评估方式 & 同一模型、同一测试集上的第一步预测与因果时间集成配对评估": "Evaluation & Paired first-step and causal-ensemble evaluation on identical models and test data",
    "报告日期": "Report date",
    "2026-10-02": "2026-10-02",
    "结论预览": "Findings at a glance",
    r"在第一步预测评估中，当 action 标签发生错位或缺失时，正确对齐的 Oracle action-label quality 几乎消除了平均退化：Quality-Full 的相对 MSE 为 $0.977\times$，而 ACT-Lite 为 $2.432\times$。在综合损坏下，Quality-Full 将平均相对 MSE 从 $1.535\times$ 降至 $1.144\times$。state quality 输入也有帮助，但 image quality 在当前离线指标上没有形成清晰收益。": r"Under first-step evaluation, for shifted or missing action labels, correctly aligned Oracle label quality nearly removes average degradation: Quality-Full has relative MSE $0.977\times$, versus ACT-Lite's $2.432\times$. Under mixed damage, the respective ratios are $1.144\times$ and $1.535\times$. State-quality input also helps, while image quality has no clear benefit under the current offline metric.",
    "时间集成评估：": "Temporal-ensemble evaluation: ",
    "在同一模型与测试集上比较两种动作预测方式，分别报告相对干净训练的 MSE 比值和集成前后的实际 MSE 变化。": "Both action-selection methods use the same models and test data. The report distinguishes ratios to clean training from actual MSE changes between methods.",
    "实验输出：": "Experiment output: ",
    "研究问题与结论边界": "Research question and evidence boundaries",
    "本轮实验研究四种训练集混合质量情形：仅图像损坏、仅 state 损坏、仅 action 标签损坏，以及三种模态同时混合损坏。核心问题是：在训练数据中同时存在干净和损坏片段时，按模态拆分的质量信息应当作为模型输入、作为 loss 权重，还是二者同时使用。": "This study examines four mixed-quality training conditions: image damage, state damage, action-label damage, and a mixture across modalities. When clean and damaged segments coexist, should modality-specific quality enter the model input, weight the loss, or do both?",
    r"实验中的 quality 是由已知合成损坏过程直接给出的 \textbf{Oracle quality}。它表示已知故障来源的元数据，分数是启发式映射，不是经过校准的正确概率或严格证明的性能上界，也不是部署时自动获得的传感器信号。所有结果均为干净测试集上的\textbf{离线动作预测}；MSE 与 MAE 不能替代真实机器人或仿真器中的闭环成功率、回报和安全失败率。": r"Quality is \textbf{Oracle quality} supplied by the known synthetic corruption process. It encodes known fault provenance using heuristic scores, not calibrated correctness probabilities, a proven performance upper bound, or automatically available deployment signals. All results measure \textbf{offline action prediction} on clean test data. MSE and MAE do not substitute for closed-loop success, return, or safety-failure rates in simulation or hardware.",
    "实验采用两种离线动作评估方式：第一步预测直接取每个动作块的首步，因果时间集成融合不同起点对同一时刻的预测。两种方式使用相同 checkpoint、测试 episodes 和归一化统计量。第 6 节分别给出两种方式的总体结果曲线，并列出实际 MSE 的配对变化。每条退化曲线以对应评估方式的干净训练结果为分母；集成前后变化以第一步预测结果为参照。": "Two offline evaluation methods are compared: first-step prediction selects each chunk's first action; causal temporal ensembling fuses overlapping predictions for the same target time. Both use identical checkpoints, test episodes, and normalization. Section 6 presents overall curves and paired actual MSE changes. Each degradation curve uses the clean-training baseline evaluated by the same method; before/after changes use first-step MSE as their reference.",
    "数据集与训练协议": "Datasets and training protocol",
    "三个公开机器人示范数据集均通过单任务检查。图像训练使用每个数据集的全部同步相机，而不是只选一个视角。数据先按 episode 固定划分，再只对训练 episodes 构造损坏，避免同一 episode 的相邻帧跨训练和测试泄漏。": "The three robot datasets passed the single-task check. Training uses all recorded camera views in each dataset. A fixed episode-level split precedes training-only corruption, avoiding adjacent-frame leakage between training and testing.",
    "相机": "Cameras",
    "完整数据集统计。训练分别使用 1、2、4 路同步相机。": "Full-dataset statistics. Training uses one, two, and four recorded camera views, respectively.",
    "统一配置为 10 epochs、CUDA、batch size 256、segment length 16、随机种子 7/17/27、划分种子 7。图像最长边缩放到 64 像素。Normalization statistics 只由干净 Training episodes 计算，并在同一数据集的 90 个组合间固定。验证集和测试集始终保持干净。": "Shared settings: 10 epochs, CUDA, batch size 256, segment length 16, training seeds 7/17/27, and split seed 7. The image longest edge is resized to 64 pixels. Normalization statistics are computed only from clean training episodes and frozen across all 90 combinations within a dataset. Validation and test remain clean.",
    "数据隔离与审计": "Data separation and audits",
    "270 个 run 均保存独立 checkpoint、预测 CSV、per-episode 指标和 run manifest。审计确认 checkpoint schema、quality schema、experiment signature 与 run metadata 一致；每个数据集只有一套 normalization statistics；没有缺失或重复组合。": "All 270 runs retain individual checkpoints, prediction CSVs, per-episode metrics, and manifests. Audits confirm matching checkpoint/quality schemas, experiment signatures, and run metadata; one normalization definition per dataset; and no missing or duplicate combinations.",
    "混合损坏条件": "Mixed-corruption conditions",
    "每个 episode 被划分为连续 segment，再按权重互斥地分配到一个 component。同一 segment 不会同时被计入多个百分比桶。Temporal shift 使用 2 frames，边界位置标记为 missing；模态缺失将对应输入或标签置零，同时把该模态 quality 置为 0 并记录 missing mask 与 provenance。": "Each episode is divided into contiguous segments, each assigned exclusively to one component according to mixture weights. A segment is not counted in multiple percentage buckets. Shifts use two frames with invalid boundaries marked missing. Modality missingness sets the corresponding input/label and quality to zero, and records missing masks and provenance.",
    "Segment 构成": "Segment composition",
    "image/state/action missing 各": "image/state/action missing, each",
    "本轮正式实验的五个训练条件。": "The five formal training conditions.",
    "在所有数据集和 seeds 中，实际 segment 比例与目标比例的最大偏差约为 0.03 个百分点。由于 episode 尾部 segment 长度可能不足 16，按 frame 统计的比例存在小幅偏差：单模态条件中各 component 的 frame 比例偏差不超过约 0.35 个百分点；综合条件不超过约 0.83 个百分点。": "Across datasets and seeds, segment proportions differ from targets by at most approximately 0.03 percentage points. Final episode segments can have fewer than 16 frames, so frame proportions differ slightly: by at most about 0.35 percentage points for single-modality conditions and 0.83 for the mixed condition.",
    "六组 ACT / Quality 消融": "Six ACT / quality ablations",
    "Quality 输入": "Quality input",
    "Action-label 加权 loss": "Label-weighted loss",
    "目的": "Purpose",
    "ACT-Lite & 否 & 否 & 无质量信息的基线": "ACT-Lite & No & No & Baseline without quality",
    "Quality-Input & 是 & 否 & 仅测试质量条件输入": "Quality-Input & Yes & No & Input conditioning only",
    "Quality-Weighted-Loss & 否 & 是 & 仅测试坏 action 标签降权": "Quality-Weighted-Loss & No & Yes & Label downweighting only",
    "Quality-Full & 是 & 是 & 完整方法": "Quality-Full & Yes & Yes & Full method",
    "Quality-Shuffled & 打乱 & 是 & 破坏质量与样本的对应关系": "Quality-Shuffled & Shuffled & Yes & Break sample correspondence",
    "Quality-Constant & 常数 & 是 & 所有样本伪装为干净": "Quality-Constant & Constant & Yes & Mark every sample as clean",
    "Image/state missing 不会自动删除整帧：样本仍参加训练，模型通过 observation quality 与 missing mask 得知输入不可靠。Action missing 则是标签可靠性问题；只有使用 action-label quality 加权 loss 的模型才会降低或屏蔽该标签的监督贡献。": "Image/state missingness does not remove whole samples. Samples remain in training, and observation quality/missing masks describe unreliable inputs. Missing actions concern label reliability: only label-quality-weighted loss reduces or removes their supervision contribution.",
    "指标与汇总方法": "Metrics and aggregation",
    "主要指标是干净测试集上的 action mean squared error：": "The primary metric is action mean squared error on the clean test set:",
    "三个数据集的 action 单位和尺度不同，因此跨数据集使用同一模型、同一 seed 的相对比值：": "Action units and scales differ across datasets. Cross-dataset aggregation therefore uses ratios within the same model and seed:",
    r"$R=1$ 表示与干净训练持平，$R>1$ 表示退化。下文的总体数字先在每个 dataset-model-seed 内计算 $R$，再对 3 个数据集与 3 个 seeds 的 9 个比值取均值。": r"$R=1$ matches clean training; $R>1$ indicates degradation. Overall results first compute $R$ within each dataset/model/seed, then average nine ratios across three datasets and three seeds.",
    "总体结果": "Overall results",
    "三个数据集与三个 seeds 的平均相对 MSE。越低越好；虚线为干净训练基准。": "Mean relative MSE across three datasets and three seeds. Lower is better; the dashed line denotes clean training.",
    "平均 MSE 比值。绿色仅标示列内最小值，不代表统计显著性。": "Mean MSE ratios. Green marks column minima, not statistical significance.",
    "按故障模态解读": "Interpretation by fault modality",
    "Action 标签损坏：加权 loss 是关键": "Action-label damage: weighted loss is central",
    r"ACT-Lite 在 action-damaged 条件下平均退化到 $2.432\times$，Quality-Input 仍为 $2.394\times$。Quality-Input 只接收观测质量，并不知道动作标签质量；只要错误标签继续以普通权重进入 loss，监督信号仍会把模型拉向错误目标。": r"Under action-label damage, ACT-Lite degrades to $2.432\times$ and Quality-Input remains at $2.394\times$. Quality-Input conditions on observation quality but does not receive action-label quality. Bad labels still enter its ordinary loss at full weight.",
    r"Quality-Weighted-Loss 将比值降至 $0.992\times$，Quality-Full 为 $0.977\times$。Shuffled 与 Constant 分别退化到 $2.463\times$ 和 $2.401\times$，这些对照支持\textbf{正确对齐的逐样本 action-label quality}的重要性，不能仅将收益归因于额外输入维度、随机降权或模型容量。": r"Quality-Weighted-Loss reaches $0.992\times$ and Quality-Full $0.977\times$, while Shuffled and Constant reach $2.463\times$ and $2.401\times$. These controls support the importance of \textbf{correctly aligned per-sample label quality}, rather than attributing gains solely to extra inputs, random downweighting, or capacity.",
    "Mixed action damaged 的三-seed 平均绝对 MSE。不同数据集之间不可横向比较数值大小。": "Three-seed mean absolute MSE under action-label damage. Magnitudes must not be compared across datasets.",
    "State 观测损坏：quality 输入有帮助但不完全": "State damage: quality input helps partially",
    r"State damaged 使 ACT-Lite 平均退化到 $1.388\times$。Quality-Input 和 Quality-Full 分别为 $1.234\times$ 与 $1.238\times$，优于不使用 observation quality 的 Weighted-Loss（$1.374\times$）。然而两者仍明显高于 1，说明 quality 元数据只能帮助模型识别不可靠 state，并不能恢复被错位或置零的信息。": r"State damage raises ACT-Lite to $1.388\times$. Quality-Input and Full reach $1.234\times$ and $1.238\times$, versus Weighted-Loss at $1.374\times$ without observation quality. Both remain above one: quality metadata identifies unreliable state but does not restore shifted or zeroed information.",
    "Image 观测损坏：当前离线指标不敏感": "Image damage: limited sensitivity of the offline metric",
    r"所有模型在 image-damaged 条件下都约为 $1.0\times$。正确 image quality、打乱 quality 和恒定 quality 之间没有稳定差距。这个结果只能说明在当前 64 像素输入、所选任务与离线 action MSE 下，40\% 两帧图像错位加 10\% 全相机缺失没有形成可辨别的平均退化；不能据此断言真实闭环控制对视觉故障不敏感。": r"All models remain near $1.0\times$ under image damage, without consistent separation between correct, shuffled, and constant quality. At the current 64-pixel setting, tasks, and offline MSE, 40\% two-frame image shifts plus 10\% all-camera missingness do not produce clear average degradation. This does not establish that closed-loop control is insensitive to visual faults.",
    "综合损坏：主要收益来自处理坏 action 标签": "Mixed damage: gains mainly from handling bad labels",
    r"综合条件下，ACT-Lite 为 $1.535\times$，Quality-Input 反而达到 $2.007\times$；Weighted-Loss 与 Quality-Full 分别降至 $1.163\times$ 和 $1.144\times$。Full 仅比 Weighted-Loss 略好，说明主要收益仍来自 action-label weighting，observation quality 的额外贡献较小且随 seed 波动。": r"Under mixed damage, ACT-Lite reaches $1.535\times$, Input $2.007\times$, Weighted-Loss $1.163\times$, and Full $1.144\times$. Full is only slightly better than Weighted-Loss; the main benefit is label weighting, with smaller, seed-dependent additional contributions from observation quality.",
    "数据集差异与配对比较": "Dataset differences and paired comparisons",
    "按数据集汇总的三-seed 平均 MSE 比值。": "Three-seed mean MSE ratios by dataset.",
    "XArm 对本轮损坏整体较不敏感；Koch 与 ALOHA 的 state/action 损坏更明显。因此不能只用一个数据集判断质量机制。Quality-Full 相对 ACT-Lite 在 36 个损坏配对中有 31 个获得更低绝对 MSE，按各自 clean 基线归一化后也是 31/36。相对 Shuffled 为 32/36 个更低绝对 MSE，相对 Constant 为 29/36。": "XArm is less sensitive overall; Koch and ALOHA show larger state/action effects, motivating multi-dataset evaluation. Full has lower absolute MSE than ACT-Lite in 31/36 damaged pairs and lower own-baseline-normalized ratios in 31/36. Absolute MSE is lower in 32/36 comparisons with Shuffled and 29/36 with Constant.",
    "Full 更低绝对 MSE": "Full: lower MSE",
    "Full 更低相对比值": "Full: lower ratio",
    "总配对数": "Pairs",
    "比较": "Comparison",
    "结论与下一步": "Conclusions and next steps",
    "主要结论": "Main findings",
    r"\textbf{Oracle action-label quality 有效。}它在 Koch 与 ALOHA 上大幅消除 action shift/missing 带来的退化，并通过 shuffled/constant 控制验证了对齐信息的重要性。": r"\textbf{Oracle label quality is effective.} It substantially mitigates action shift/missingness on Koch and ALOHA; shuffled/constant controls support the importance of alignment.",
    r"\textbf{State quality 输入有中等收益。}它降低了 state-damaged 的平均误差，但无法恢复缺失信息，结果仍差于 clean。": r"\textbf{State-quality input provides moderate gains.} Average state-damage error falls, but missing information is not recovered and results remain worse than clean.",
    r"\textbf{Image quality 暂无清晰收益。}当前损坏强度、图像分辨率和离线 action MSE 可能不足以显现闭环视觉风险。": r"\textbf{Image quality has no clear benefit here.} Current severity, resolution, and offline MSE may not reveal closed-loop visual risk.",
    r"\textbf{Quality-Full 对综合损坏最稳健。}但它相对仅 Weighted-Loss 的优势较小，主要提升仍来自 action-label weighting。": r"\textbf{Full has the lowest aggregate mixed-damage ratio.} Its margin over Weighted-Loss is small; label weighting remains the primary contribution.",
    "建议下一阶段按以下顺序推进：": "Recommended next-stage sequence:",
    "用已知 corruption provenance 训练自动质量估计器，分别预测 image/state/action-label quality，并报告校准误差；": "Train automatic image/state/action-label quality estimators from known corruption provenance and report calibration error;",
    "用预测 quality 替换 Oracle quality，量化质量估计误差对性能的影响；": "Replace Oracle quality with predictions and quantify sensitivity to estimation errors;",
    "提高 image shift/missing 的严重度和持续时间，并保留高分辨率视觉特征，判断当前“近似无影响”是否由实验灵敏度不足造成；": "Increase image-shift/missingness severity and duration, retain higher-resolution features, and test whether current near-neutral results reflect limited sensitivity;",
    "在仿真或真实机器人上做闭环 rollout，报告成功率、安全失败类型和恢复行为；": "Run closed-loop simulation or robot rollouts, reporting success, safety-failure types, and recovery;",
    "若闭环趋势仍成立，再扩展更多 seeds、任务与损坏强度，并使用配对置信区间或显著性检验。": "If closed-loop trends persist, expand seeds, tasks, and severities, using paired confidence intervals or significance tests.",
    "可复现性与产物索引": "Reproducibility and artifact index",
    "本轮 GPU 路径的训练函数墙钟时间累计为 5.40 小时，包含训练循环内验证与 checkpoint 开销，不含外部数据下载、四相机视频解码、最终评估和报告生成。完整性检查结果为 270/270；270 个 checkpoint 的 metadata 与 run manifest 均匹配。旧实验目录没有参与恢复或结果汇总。": "Recorded training-function wall time on the GPU path totals 5.40 hours, excluding external download, four-camera video decoding, final evaluation, and reporting; in-loop validation/checkpoint overhead remains included. Integrity checks passed for 270/270 runs, with checkpoint metadata matching manifests. Old experiment directories were not used for resume or aggregation.",
    "内容 & 路径": "Content & Path",
    "实验定义与总览": "Study definition",
    "结构化汇总": "Structured summary",
    "完整 JSON 证据": "Full JSON evidence",
    "交互式 HTML 报告": "Interactive HTML report",
    "单次运行记录": "Per-run records",
    "本报告由 Sync2Act 实验 schema v3、quality schema v2 的实测输出生成。": "Generated from measured outputs using experiment schema v3 and quality schema v2.",
    "训练版本：Sync2Act v1.2.0；Git commit": "Training version: Sync2Act v1.2.0; Git commit",
    "时间集成评估的代码指纹、环境与协议见": "Temporal-evaluation source fingerprints, environment, and protocol:",
}


def build_english_report(root: Path) -> Path:
    source = root / "output/pdf/sync2act_mixed_modality_damage_report_v1_3_0.tex"
    text = english_text(source.read_text(encoding="utf-8"), REPORT_TEXT)
    text = text.replace(
        r"\documentclass[11pt,a4paper]{ctexart}",
        r"\documentclass[11pt,a4paper]{article}"
        + "\n"
        + r"\usepackage{fontspec}"
        + "\n"
        + r"\setlength{\emergencystretch}{1em}",
    )
    text = text.replace(r"\fontsize{28}{34}", r"\fontsize{22}{28}")
    text = text.replace(r"\section{Overall results}", r"\clearpage\section{Overall results}")
    text = text.replace(r"\section{Mixed-corruption conditions}", r"\clearpage\section{Mixed-corruption conditions}")
    text = text.replace(r"\section{Dataset differences and paired comparisons}", r"\clearpage\section{Dataset differences and paired comparisons}")
    text = text.replace("Temporal ensemble &", "Ensembling &")
    text = text.replace("/report.html", "/report_en.html")
    # English headings need more room than their Chinese counterparts.
    text = text.replace(
        "First-step MSE & Ensemble MSE & MSE difference", "First-step & Ensemble & Difference"
    )
    target = source.with_stem(source.stem + "_en")
    text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    target.write_text(text, encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build_english_report(Path(__file__).resolve().parents[1]))
