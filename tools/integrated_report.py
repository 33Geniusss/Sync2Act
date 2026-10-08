"""One research narrative over the preserved training and test-corruption evidence."""
from __future__ import annotations

import re
from collections import Counter


def section_bodies(source):
    document = source.split(r"\begin{document}", 1)[1].rsplit(r"\end{document}", 1)[0]
    headings = list(re.finditer(r"\\section\{[^{}]+\}", document))
    return [re.sub(r"(?:\\(?:clearpage|appendix)\s*)+$", "",
                   document[h.end():headings[i + 1].start() if i + 1 < len(headings) else len(document)].strip()).strip() + "\n"
            for i, h in enumerate(headings)]


def compose_integrated_report(current, historical, frame, paired, zh, render_table, format_mse):
    def tr(chinese, english):
        return chinese if zh else english

    def paragraph(chinese, english):
        return tr(chinese, english) + "\n\n"

    def heading(chinese, english, level="section", page=True):
        return (r"\clearpage" if page else "") + "\\" + level + "{" + tr(chinese, english) + "}\n"

    def plain_tables(body):
        return re.findall(r"\\begin\{center\}.*?\\end\{center\}", body, re.S)

    old, new = section_bodies(historical), section_bodies(current)
    assert len(old) == 10 and len(new) == 12
    new[1] = new[1].replace("下表仅描述历史训练数据，不是本轮受损测试的配方。", "训练片段的故障构成如下，测试观测使用下一节的独立配方。")
    new[1] = new[1].replace("The following table describes historical training data, not the new test-corruption recipes.", "Training segments use the mixtures below; test observations use the separate recipes in the next section.")
    new[4] = new[4].replace("本页仅比较", "首先比较").replace("This page fixes", "This comparison fixes")
    new[5] = new[5].replace("本页固定为", "进一步固定为").replace("与上一页的绝对 MSE 对照", "与干净训练模型的绝对 MSE 对照").replace("各页 R 的分母", "各组 R 的分母")
    new[5] = new[5].replace("This page fixes", "The next comparison fixes").replace("the previous page", "the clean-trained models")
    # Limits belong in the joint discussion; artifact paths belong in the index.
    new[8] = re.sub(r"(?:限制：|Limitations:)[^\n]+\n", "", new[8])
    new[8] = re.sub(r"(?:主数据目录为 |The main data directory is )[^\n]+\n", "", new[8])
    overall = re.split(r"\\subsection\{[^{}]+\}", old[5])[1:]
    assert len(overall) == 3
    preamble = current.split(r"\begin{document}", 1)[0]
    preamble = preamble.replace("Sync2Act / Test-Time Corruption", "Sync2Act / Data Quality Study")
    s = preamble + r"""
\usepackage{multirow,tocloft}
\setlength{\cftsecnumwidth}{2.8em}
\setlength{\cftbeforesecskip}{5pt}
\definecolor{teal}{HTML}{00A6A6}
\definecolor{purple}{HTML}{7456B8}
\definecolor{rose}{HTML}{D45576}
\definecolor{green}{HTML}{3B9A66}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\newcommand{\good}[1]{\textcolor{green}{\bfseries #1}}
\newcommand{\warn}[1]{\textcolor{orange}{\bfseries #1}}
\hypersetup{pdftitle={Sync2Act v1\_4\_0: Data Quality and Policy Robustness}}
\setcounter{tocdepth}{1}
\begin{document}
\thispagestyle{empty}
\vspace*{12mm}
{\color{blue}\rule{25mm}{2.5pt}}\\[7mm]
{\fontsize{30}{36}\selectfont\bfseries\color{navy}Sync2Act\par}
\vspace{5mm}
"""
    s += r"{\LARGE\bfseries\color{navy}" + tr("数据质量与策略鲁棒性", "Data Quality and Policy Robustness") + "\\par}\n"
    s += r"\vspace{4mm}{\large\color{muted}" + tr("从受损示范学习，到受损观测下的动作预测", "Learning from damaged demonstrations and predicting from damaged observations") + "\\par}\n"
    s += r"\vspace{5mm}{\large v1\_4\_0 / 2026-10-07}\par\vspace{6mm}" + "\n"
    s += r"\note{" + tr(
        "研究问题：已知数据质量信息，应如何进入模仿学习系统？本研究在同一组机器人任务和固定模型上，连接训练数据损坏、测试观测损坏和动作块时间集成三类实验，区分学习阶段与预测阶段的收益。",
        "How should known data-quality information enter an imitation-learning system? Using the same robot tasks and fixed models, this study connects damaged training demonstrations, corrupted test observations, and temporal ensembling of action chunks, separating learning-stage benefits from prediction-stage robustness.") + "}\n"
    s += render_table([tr("实验维度", "Evidence"), tr("规模", "Scope")], [
        [tr("训练与消融", "Training and ablations"), tr("3 数据集 × 6 模型 × 5 条件 × 3 seeds = 270", "3 datasets x 6 models x 5 conditions x 3 seeds = 270")],
        [tr("测试鲁棒性", "Test robustness"), tr("4,158 次评估，包含 270 个干净基线核对", "4,158 evaluations, including 270 clean-baseline audits")],
        [tr("时间集成", "Temporal ensembling"), tr("270 组配对；固定 decay=0.7", "270 paired runs; fixed decay=0.7")],
    ], "lp{121mm}")
    s += paragraph(
        r"\textbf{主要发现。}正确动作标签质量加权能显著缓解训练示范损坏：动作标签损坏时，Q-Full 与 ACT-Lite 的平均训练退化比分别为 0.977 和 2.432；综合损坏时为 1.144 和 1.535。但在同样采用综合损坏训练后，Q-Full 在 ALOHA 中等状态损坏测试中的配对 MSE 比 ACT-Lite 高约 36.1\%，Koch 和 xArm 则分别低约 3.0\% 和 2.1\%。训练阶段的收益不会自动转化为受损输入下的优势。",
        r"\textbf{Principal finding.} Correct action-label weighting mitigates damaged demonstrations: mean training-degradation ratios for Q-Full and ACT-Lite are 0.977 versus 2.432 under label damage, and 1.144 versus 1.535 under mixed damage. Yet with mixed-damage training, Q-Full has about 36.1\% higher paired MSE than ACT-Lite on ALOHA with medium state-corrupted tests, versus 3.0\% and 2.1\% lower on Koch and xArm. Learning-stage gains do not automatically confer robustness to damaged inputs.")
    s += paragraph(
        r"固定 decay=0.7 的时间集成未降低总体实际 MSE：270 组逐运行变化百分比的均值为 $+21.20\%$。本文同时报告绝对误差与四类配对比值，避免将分母变化误解为性能改善。",
        r"Temporal ensembling at fixed decay=0.7 did not lower overall actual MSE: the mean per-run percentage change across 270 pairs was $+21.20\%$. Absolute errors and distinct paired ratios are reported together to avoid mistaking denominator changes for improved accuracy.")
    s += r"\vfill\note{" + tr("本文整合既有训练与评估结果，未重新训练。所有结果是离线动作预测，不代表真实机器人闭环成功率。", "This report integrates existing training and evaluation evidence without retraining. All results concern offline action prediction, not real-robot closed-loop success.") + "}\n"
    s += r"\clearpage\begingroup\setlength{\parskip}{0pt}\tableofcontents\endgroup" + "\n"
    s += r"\vspace{8mm}\note{" + tr(
        "阅读顺序：第 1--4 节统一定义实验与指标；第 5 节研究训练示范质量；第 6--7 节检验测试输入及质量信息；第 8 节评估预测方式；第 9--10 节给出联合结论、局限和复现信息。附录列出完整受损测试矩阵。",
        "Sections 1--4 define the shared experiment and metrics. Section 5 studies demonstration quality; Sections 6--7 examine test inputs and quality metadata. Section 8 evaluates action selection; Sections 9--10 give joint conclusions, limitations, and reproducibility. Appendices contain the complete corrupted-test matrices.") + "}\n"

    # Shared scope: no version-by-version introductions or duplicate dataset tables.
    s += heading("研究问题与统一实验设计", "Research questions and shared experimental design")
    s += paragraph(
        "数据损坏可能发生在学习阶段，也可能发生在模型使用阶段。前者影响监督信号与学到的策略，后者直接改变预测时可用的观测。研究分别检验：质量输入和标签加权是否改善受损示范学习；固定模型面对图像/状态故障时是否稳健；时间集成是否降低同一轨迹上的预测误差。",
        "Data corruption can occur during learning or at model-use time. The former changes supervision and the learned policy; the latter changes the observations available for prediction. We test whether quality inputs and label weighting improve learning from damaged demonstrations, whether fixed models tolerate image/state faults, and whether temporal ensembling lowers error on the same trajectories.")
    s += plain_tables(new[0])[0] + plain_tables(new[0])[1]
    s += paragraph(
        "ALOHA 为静态电池操作，Koch 为单块 Lego 拾放，xArm 为抬升任务；三个数据集均通过单任务检查。使用完整 episodes、完整 frames 和全部同步相机，先按 episode 固定划分，再分别构造训练损坏与测试观测副本，避免相邻帧跨分区泄漏。“干净”表示未追加合成故障，并不保证原始记录完全没有噪声。",
        "ALOHA is static-battery manipulation, Koch is single-Lego pick-and-place, and xArm is lifting; all passed the single-task check. Full episodes, frames, and synchronized cameras are used. Fixed episode partitions precede training corruption and separate corrupted test copies, preventing adjacent-frame leakage. Clean means no added synthetic faults, not proof that recordings are noise-free.")
    s += paragraph(
        "固定划分种子为 7，训练种子为 7、17、27，图像最长边 64 像素，10 epochs、CUDA、batch size 256、动作 horizon 8。每个数据集仅从干净训练 episodes 计算一次归一化，并在其 90 组模型/条件/种子组合中固定；测试从 checkpoint 恢复这些统计量。验证集保持干净，测试故障不改变划分或训练统计量。",
        "The split seed is 7; training seeds are 7, 17, and 27. Images have a 64-pixel maximum side; training uses 10 epochs, CUDA, batch size 256, and action horizon 8. Normalization is fitted once to clean training episodes per dataset and fixed across its 90 model/condition/seed combinations; tests restore it from checkpoints. Validation stays clean, and test faults never alter partitions or training statistics.")

    s += heading("模型消融与受损训练条件", "Model ablations and training corruption")
    # This single model/training table pair combines both earlier definitions.
    s += new[1]
    s += paragraph(
        "表中的 Q- 是 Quality- 的缩写，Q-Weighted 对应 Quality-Weighted-Loss。Q-Shuffled 打破逐样本质量对应关系，Q-Constant 将质量标为正常。缺失图像/状态不删除整个训练样本；缺失动作涉及标签可靠性，只有启用标签质量加权的损失会降低或移除其监督贡献。",
        "Q-Input, Q-Weighted, and Q-Full abbreviate Quality-Input, Quality-Weighted-Loss, and Quality-Full. Q-Shuffled breaks sample-quality correspondence; Q-Constant marks quality as normal. Missing images/state do not remove whole training samples. Missing actions concern label reliability; only label-quality-weighted loss reduces or removes their supervision contribution.")

    s += heading("故障注入与评估流程", "Fault injection and evaluation workflow")
    s += paragraph(
        r"\textbf{训练故障。}每条 episode 按连续 16 帧分段，按权重互斥分配一个故障类别，片段不重复计入多个比例。训练移位为 2 帧，越界标记缺失；对应输入/标签与质量置零，并记录 mask 和 provenance。实际片段比例与目标最大偏差约 0.03 个百分点；按帧计，单模态条件最大约 0.35 个百分点，综合条件约 0.83，源于 episode 尾部不足 16 帧的片段。",
        r"\textbf{Training faults.} Contiguous 16-frame segments receive one exclusive mixture component, without double-counting. Training shifts use two frames with invalid boundaries marked missing; the affected input/label and quality are zeroed, with masks and provenance retained. Maximum segment-fraction deviation is about 0.03 percentage points; frame deviations reach about 0.35 for single-modality and 0.83 for mixed conditions because final segments can be shorter than 16 frames.")
    s += new[2]

    s += heading("指标、配对关系与不确定性", "Metrics, paired comparisons, and uncertainty")
    s += paragraph(
        "默认评估取动作块首步，反归一化到原动作单位；MSE 对全部测试帧和动作维度等权平均，MAE 同样计算。测试指标不按质量降权，动作标签始终固定。不同机器人的动作尺度不同，不跨数据集合并原始 MSE。",
        "Default evaluation takes the first action in each chunk and denormalizes it to original action units. MSE weights every test frame and action dimension equally; MAE is also computed. Quality never downweights test metrics, and action targets stay fixed. Raw MSE is not pooled across robots with different action scales.")
    s += r"\[E=\frac{1}{ND}\sum_{t=1}^{N}\sum_{j=1}^{D}(\hat a_{t,j}-a_{t,j})^2.\]" + "\n"
    s += r"""\begin{align*}
R_{\mathrm{train}}&=\frac{E_{\mathrm{damaged\ train,clean\ test}}}{E_{\mathrm{clean\ train,clean\ test}}},\\
R_{\mathrm{test}}&=\frac{E_{\mathrm{same\ checkpoint,damaged\ test}}}{E_{\mathrm{same\ checkpoint,clean\ test}}},\\
P&=\frac{E_{\mathrm{Q\text{-}Full}}}{E_{\mathrm{ACT\text{-}Lite}}},\qquad
U=\frac{E_{\mathrm{unknown\ metadata}}}{E_{\mathrm{Oracle\ metadata}}}.
\end{align*}
"""
    s += paragraph(
        r"$R_{\mathrm{train}}$ 固定数据集、模型与训练种子，衡量损坏训练的影响；$R_{\mathrm{test}}$ 固定 checkpoint，衡量损坏测试的影响。$P$ 固定训练/测试条件及种子，小于 1 才表示 Q-Full 绝对误差更低；$U$ 仅改变推理质量元数据，大于 1 表示隐去质量信息使误差更高。各比值先逐配对计算再平均，分母为零时未定义。",
        r"$R_{\mathrm{train}}$ fixes dataset, model, and training seed to measure training damage. $R_{\mathrm{test}}$ fixes the checkpoint to measure test damage. $P$ matches train/test conditions and seeds; values below 1 mean lower absolute error for Q-Full. $U$ changes only inference metadata; values above 1 mean hiding quality raises error. Ratios are computed per pair before averaging and are undefined for a zero denominator.")
    s += paragraph(
        "训练损坏总体曲线平均 3 数据集 × 3 训练 seeds 的 9 个比值。受损测试则在每个训练种子内先平均 3 个独立损坏 seeds（107、117、127），再平均 3 个训练 seeds；报告训练种子间样本标准差，并在 CSV 中保留训练种子内损坏标准差的均值。3×3 配对不是 9 次独立训练，不给出显著性或置信区间结论。",
        "Training-damage overview curves average nine ratios from three datasets and three training seeds. Corrupted tests first average three independent damage seeds (107, 117, 127) within each training seed, then average the three training seeds. Between-training-seed sample SD is reported; mean within-training-seed damage SD remains in CSV. These 3x3 pairs are not nine independent training runs; no significance or confidence-interval claim is made.")
    s += r"\[\bar R_s=\tfrac13\sum_r R_{s,r},\quad \bar R=\tfrac13\sum_s\bar R_s,\quad s_{\mathrm{train}}=\sqrt{\frac{\sum_s(\bar R_s-\bar R)^2}{2}}.\]" + "\n"
    s += r"\note{" + tr(
        "退化倍率更低不一定意味着绝对误差更低，尤其不能跨训练条件或跨评估方式忽略分母变化。训练结果章节的 R 指训练退化比，受损测试章节的 R 指同 checkpoint 测试退化比。",
        "A lower degradation ratio need not mean lower absolute error, especially across training conditions or evaluation methods with changing denominators. R in training-result plots denotes training degradation; R in corrupted-test tables and curves denotes same-checkpoint test degradation.") + "}\n"

    s += heading("从受损示范中学习：干净测试结果", "Learning from damaged demonstrations: clean-test results")
    s += paragraph(
        "首先保持测试观测干净，隔离训练示范质量的影响。六组模型采用相同划分、预处理和训练预算，因此可通过质量输入与标签加权的消融判断收益来源。下图与表使用训练退化比；其中最低值只表示观察到的均值最小。",
        "We first keep test observations clean to isolate demonstration quality. All six variants use matched partitions, preprocessing, and training budgets, allowing ablations of quality input and label weighting to locate the gains. The figure and table use training-degradation ratios; highlighted minima are observed means only.")
    # Measured figures/tables are preserved verbatim; narrative is reorganized.
    s += overall[0].replace(r"\clearpage", "")
    s += heading("故障模态揭示的收益来源", "What fault modalities reveal about the gains", "subsection")
    interpretation = old[6]
    interpretation = interpretation[interpretation.index(r"\subsection{"):]
    interpretation = interpretation.replace(r"\subsection{", r"\subsubsection{")
    s += interpretation
    s += heading("任务差异与模型配对", "Task differences and paired models", "subsection")
    s += paragraph(
        "总体均值不能替代逐任务比较。下面同时保留各模型自身基线归一化后的结果，以及同条件绝对误差的胜出次数；两者回答的问题不同。",
        "An overall mean cannot replace task-level comparisons. The following results retain both each model's own-baseline-normalized errors and counts of wins in matched absolute error; these answer different questions.")
    s += old[7]

    s += heading("固定模型面对受损测试观测", "Fixed policies under corrupted test observations")
    s += paragraph(
        "训练损坏实验说明模型能否从不可靠示范中学习，但不能回答部署输入受损时是否可靠。下面保持同一 checkpoint、动作标签与归一化，只改变留出轨迹的观测；先检查干净训练模型，再检查见过综合损坏的模型。",
        "The training experiment shows whether a model can learn from unreliable demonstrations, but does not establish reliability under damaged deployment inputs. We now fix checkpoints, action targets, and normalization, changing only held-out observations. Clean-trained models are examined first, followed by models exposed to mixed training damage.")
    s += heading("干净训练模型的敏感性", "Sensitivity of clean-trained models", "subsection", page=False) + new[4]
    s += heading("综合损坏训练后的测试鲁棒性", "Test robustness after mixed-damage training", "subsection") + new[5]
    s += heading("故障强度与任务敏感性", "Fault severity and task sensitivity", "subsection") + new[7]

    s += heading("质量信息的作用与跨条件比较", "Quality information and comparisons across test conditions")
    s += heading("同一受损输入：Oracle 与 Unknown", "Identical damaged inputs: Oracle versus Unknown", "subsection", page=False) + new[6]
    s += heading("为何干净测试的优势未必延续", "Why a clean-test advantage may not persist", "subsection")
    s += paragraph(
        "为直接连接上述两组结果，下面固定综合损坏训练（M）、首步预测和 Oracle 测试质量信息，比较同一批模型在干净与中等受损观测上的表现。表中为逐种子配对的 Q-Full / ACT-Lite MSE 百分比变化，负数代表 Q-Full 更好。",
        "To connect the two experiments directly, the comparison below fixes mixed-damage training (M), first-step prediction, and Oracle test metadata. The same models are tested on clean and medium-corrupted observations. Cells are mean per-seed paired percentage changes in Q-Full / ACT-Lite MSE; negative values favor Q-Full.")
    datasets = ["aloha_static_battery", "koch_pick_place_1_lego", "xarm_lift_medium"]
    names = dict(zip(datasets, ["ALOHA", "Koch", "xArm"], strict=True))
    baseline = frame[(frame.group == "baseline") & (frame.train_condition == "mixed_three_corruptions")]
    matched = baseline[baseline.model == "quality_full"].merge(baseline[baseline.model == "act_lite"], on=["dataset", "train_seed"], suffixes=("_q", "_a"), validate="one_to_one")
    clean_ratio = (matched.action_mse_q / matched.action_mse_a).groupby(matched.dataset).mean()
    dirty_ratio = paired[(paired.train_condition == "mixed_three_corruptions") & (paired.severity == "medium")].groupby(["dataset", "test_condition"]).quality_to_act_mse.mean()
    rows = [[names[d], *[f"${100*(v-1):+.1f}\\%$" for v in [clean_ratio[d], *[dirty_ratio[d, k] for k in ("image", "state", "mixed")]]]] for d in datasets]
    headers = [tr("数据集", "Dataset"), tr("干净", "Clean"), tr("图像损坏", "Image"), tr("状态损坏", "State"), tr("混合损坏", "Mixed")]
    s += render_table(headers, rows, "lrrrr")
    s += paragraph(
        "ALOHA 的优势在状态/混合损坏下反转；Koch 的优势大幅缩小，xArm 仍为小幅差异。正确质量信息能帮助 Q-Full，并不保证它超越 ACT-Lite；这些均值也未经过显著性检验。下表进一步确认，ACT-Lite 自身同样会随状态损坏退化。",
        "ALOHA reverses under state/mixed damage; Koch's advantage shrinks substantially, while xArm retains small differences. Correct metadata can help Q-Full without making it better than ACT-Lite, and these means have not established significance. ACT-Lite itself also degrades under state damage, as its absolute MSE below shows.")
    act = frame[(frame.model == "act_lite") & (frame.train_condition == "mixed_three_corruptions") & frame.group.isin(["baseline", "main"])]
    means = act.groupby(["dataset", "test_condition"]).action_mse.mean()
    s += render_table(["ACT-Lite MSE", *headers[1:]], [[names[d], *[format_mse(means[d, k]) for k in ("clean", "image", "state", "mixed")]] for d in datasets], "lrrrr")
    s += paragraph(
        "状态损坏后的同 checkpoint 配对退化倍率依次为 7.94、11.55、1.03。损坏训练不是对测试故障的免疫。以上结果支持区分训练时处理坏标签与预测时处理坏观测；对于 ALOHA 反转的具体机制，仍需单独消融验证。",
        "Same-checkpoint paired state-damage ratios are 7.94, 11.55, and 1.03. Damage exposure in training does not confer immunity to test faults. The results motivate separating bad-label handling during learning from bad-observation handling during prediction; the specific mechanism behind ALOHA's reversal still requires targeted ablations.")

    s += heading("预测方式：因果时间集成", "Action selection: causal temporal ensembling")
    s += paragraph(
        "除了数据质量，动作块如何转化为逐帧预测也可能影响指标。本节在干净测试观测上比较首步预测与时间集成，模型权重完全相同；因此这里测量的是预测方式的影响，不能据此推断集成在受损观测上的表现。",
        "Beyond data quality, converting action chunks into per-frame predictions can change error. This section compares first-step prediction with ensembling on clean test observations using identical weights. It isolates action selection and does not establish ensemble behavior under corrupted observations.")
    s += overall[1].replace(r"\clearpage", "")
    s += heading("退化比下降与实际误差的区别", "Degradation ratios versus actual errors", "subsection") + overall[2].replace(r"\clearpage", "")

    s += heading("联合结论、局限与后续验证", "Joint conclusions, limitations, and next validation")
    s += paragraph(
        r"\textbf{标签质量加权是受损示范学习的主要收益来源。}动作标签损坏时 Q-Full 的训练退化比为 0.977，而 ACT-Lite 为 2.432；综合损坏时为 1.144 与 1.535。Shuffled/Constant 对照支持逐样本质量对齐的重要性。Q-Full 的综合损坏平均退化比最低，但相对 Q-Weighted 的额外优势较小。",
        r"\textbf{Label weighting is the main gain in learning from damaged demonstrations.} Q-Full versus ACT-Lite training-degradation ratios are 0.977 versus 2.432 under label damage and 1.144 versus 1.535 under mixed damage. Shuffled/Constant controls support per-sample quality alignment. Q-Full has the lowest aggregate mixed-damage ratio, with only a small margin over Q-Weighted.")
    s += paragraph(
        r"\textbf{状态质量与测试鲁棒性必须分别检验。}质量输入能部分缓解训练状态损坏，却不能恢复缺失信息；测试状态损坏仍可明显放大误差。综合损坏训练下，Q-Full 的 ALOHA 状态测试配对 MSE 高约 36.1\%，而 Koch/xArm 仅有小幅优势。隐去正确质量信息会使该 Q-Full 的状态测试误差分别增加约 50.7\%、38.9\%、9.9\%，说明信息有用不等于模型必然胜出。",
        r"\textbf{State quality and test robustness need separate checks.} Quality input partly mitigates training-state damage but cannot recover missing information; test-state damage can still amplify error. With mixed training damage, Q-Full has about 36.1\% higher paired state-test MSE on ALOHA and only small advantages on Koch/xArm. Hiding correct metadata raises its state-test errors by about 50.7\%, 38.9\%, and 9.9\%: useful information does not guarantee a winning model.")
    s += paragraph(
        r"\textbf{视觉与时间集成结论有明确边界。}当前 64 像素、所选故障与离线 MSE 对图像损坏的敏感性有限；这不是视觉故障没有闭环风险的证明。固定 decay=0.7 的时间集成使 270 个逐运行 MSE 变化平均为 $+21.20\%$，没有显示总体实际误差收益；尚未在受损测试观测上评估集成。",
        r"\textbf{Visual and ensemble conclusions have clear boundaries.} The current 64-pixel inputs, selected faults, and offline MSE show limited image-damage sensitivity, not proof that visual faults are harmless in closed loop. Fixed decay=0.7 ensembling yields a mean per-run MSE change of $+21.20\%$ across 270 pairs, without overall actual-error benefit; ensembling under damaged test observations remains unmeasured.")
    s += paragraph(
        "只有三个任务与三个训练种子；质量分数是已知合成故障的启发式映射，不是校准概率或理论性能上界。未评估自动质量估计、真实在线延迟、动作执行反馈与闭环安全，也未按新训练协议重训。",
        "There are only three tasks and three training seeds. Quality scores are heuristic mappings from known synthetic faults, not calibrated probabilities or theoretical performance bounds. Automatic quality estimation, real online latency, execution feedback, closed-loop safety, and retraining under the newer training protocol are not established here.")
    s += r"\note{" + tr(
        "后续验证：先通过 ALOHA 机制消融区分质量输入、标签加权与故障训练分布；利用 provenance 学习 image/state/action-label 质量估计器并检查校准，再用预测质量替换 Oracle；提高视觉分辨率、故障强度与持续时间；在仿真或实机检验成功率、安全失败和恢复行为；若闭环趋势成立，再扩大 seeds、任务与强度并给出配对置信区间或显著性检验。",
        "Next validation: isolate ALOHA mechanisms by separating quality input, label weighting, and training-fault distribution; learn image/state/action-label quality from provenance and check calibration, then replace Oracle with predicted quality; increase visual resolution, fault severity, and duration; test success, safety failures, and recovery in simulation or hardware; if closed-loop trends hold, expand seeds/tasks/severities and report paired confidence intervals or significance tests.") + "}\n"

    s += heading("审计与可复现性", "Audits and reproducibility")
    # Keep the runtime, all new workflow details and executable commands together.
    s += new[8]
    s += paragraph(
        "训练侧完整性审计为 270/270：保留各 checkpoint、预测 CSV、逐 episode 指标与 manifest，schema、signature、metadata 均匹配，没有缺失或重复组合，每个数据集只有一套归一化。训练函数墙钟累计 5.40 小时，包含循环内验证和 checkpoint，不含下载、四相机解码、最终评估与报告生成；旧实验目录未用于恢复或汇总。该训练耗时与表中的测试批量耗时不是同一种测量。",
        "Training integrity audits passed 270/270: individual checkpoints, prediction CSVs, per-episode metrics, and manifests are retained, with matching schemas, signatures, and metadata; no missing/duplicate combinations and one normalization per dataset. Training-function wall time totals 5.40 hours, including in-loop validation/checkpointing but excluding downloads, four-camera decoding, final evaluation, and reporting. Older experiment directories were not used for resume or aggregation. This timing differs from the batch-test wall time above.")
    s += heading("产物索引与来源记录", "Artifact index and source records", "subsection")
    s += r"\begingroup\raggedright" + "\n"
    index_start = old[9].index(r"\noindent\begin{tabularx}")
    s += old[9][index_start:].replace(r"\vfill", "")
    s += paragraph(
        r"上述路径对应训练与干净测试。受损测试目录为 \path{runs/test_time_corruption_v1/}，保留 \code{results.csv}、\code{aggregate.csv}、\code{per\_training\_seed.csv}、\code{quality\_vs\_act.csv}、\code{metadata\_comparison.csv}、\code{summary.json} 以及逐运行故障清单、预测与 episode 指标。逐运行指标、两级汇总及模型/元数据配对结果分别保存；受损与未受损片段的误差也在原始输出中。",
        r"The indexed paths above hold training and clean-test evidence. Corrupted tests are in \path{runs/test_time_corruption_v1/}, retaining \code{results.csv}, \code{aggregate.csv}, \code{per\_training\_seed.csv}, \code{quality\_vs\_act.csv}, \code{metadata\_comparison.csv}, \code{summary.json}, per-run fault manifests, predictions, and episode metrics. Per-run errors, hierarchical aggregates, and model/metadata pairs are retained separately, including errors on affected and unaffected segments.")
    s += paragraph(
        r"时间集成目录 \path{runs/mixed_modality_damage_v1_3_0/temporal_ensemble/} 保存配对预测、完整动作块、代码指纹、环境与配置。\code{relative\_mse\_by\_model\_condition.csv} 支持总体曲线，\code{mse\_by\_dataset\_model\_condition.csv} 支持完整条件对照，\code{mse\_per\_run.csv} 保存逐运行误差。",
        r"The directory \path{runs/mixed_modality_damage_v1_3_0/temporal_ensemble/} retains paired predictions, complete chunks, code fingerprints, environment, and settings. \code{relative\_mse\_by\_model\_condition.csv} supports the overview curves, \code{mse\_by\_dataset\_model\_condition.csv} supports full condition comparisons, and \code{mse\_per\_run.csv} retains individual errors.")
    s += paragraph(
        r"本文基于日期为 2026-10-02 的 v1\_3\_0 报告及 2026-10-07 的完整测试评估，统一重写研究叙述。源码包保留原始中英文来源和 SHA-256，可核对所有历史结果；当前图表与表格均内嵌于 LaTeX，无外部图片依赖。报告版本 v1\_4\_0 不代表发布了同版本软件安装包。",
        r"This narrative integrates the v1\_3\_0 report dated 2026-10-02 and the completed test evaluation of 2026-10-07. The source bundle retains both original language sources and SHA-256 hashes for auditing historical results. Tables and figures are embedded in LaTeX with no external images. Report version v1\_4\_0 does not denote a same-version application installer.")
    s += r"\begin{tcolorbox}[colback=panel,colframe=rule]\small\ttfamily" + "\n"
    s += r"python tools/build\_test\_corruption\_latex\_report.py\\" + "\n"
    s += tr("使用 XeLaTeX 分别编译两份源码两次。", "Compile each language source twice with XeLaTeX.") + "\n\\end{tcolorbox}\n"

    s += "\\endgroup\n" + r"\appendix" + "\n"
    for i, dataset in enumerate(["ALOHA", "Koch", "xArm"]):
        s += heading(f"完整中等强度测试矩阵：{dataset}", f"Complete medium-severity test matrix: {dataset}")
        s += new[9 + i]
    s += "\\end{document}\n"
    for name in ("relative_mse_by_model_condition.csv", "mse_by_dataset_model_condition.csv", "mse_per_run.csv",
                 "per_training_seed.csv", "quality_vs_act.csv", "metadata_comparison.csv"):
        escaped = name.replace("_", r"\_")
        s = s.replace(r"\code{" + escaped + "}", r"\path{" + name + "}")
    historical_results = re.findall(r"\\begin\{table\}.*?\\end\{table\}", historical, re.S)[3:]
    if len(historical_results) != 7 or any(block not in s for block in historical_results):
        raise ValueError("The integrated report must retain all seven historical result tables")
    # Only the duplicate cover table is removed from the test-corruption report.
    if any(block not in s for block in plain_tables(current)[1:]):
        raise ValueError("A test-corruption table or figure was lost during integration")
    def coords(source):
        return Counter(re.findall(r"coordinates\s*\{[^}]+\}", source))
    if coords(s) != coords(historical) + coords(current):
        raise ValueError("Measured plot coordinates changed during integration")
    return s
