"""Build cumulative bilingual v1_4_0 reports, retaining the entire v1_3_0 study.

python tools/build_test_corruption_latex_report.py
Compile the two generated .tex files with XeLaTeX (twice each).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from integrated_report import compose_integrated_report

VERSION = "1_4_0"
DATASETS = ["aloha_static_battery", "koch_pick_place_1_lego", "xarm_lift_medium"]
DS = {"aloha_static_battery": "ALOHA", "koch_pick_place_1_lego": "Koch", "xarm_lift_medium": "xArm"}
MODELS = ["act_lite", "quality_input", "quality_weighted_loss", "quality_full", "quality_shuffled", "quality_constant"]
MODEL = dict(zip(MODELS, ["ACT-Lite", "Q-Input", "Q-Weighted", "Q-Full", "Q-Shuffled", "Q-Constant"], strict=True))
CONDITIONS = ["clean", "mixed_image_damaged", "mixed_state_damaged", "mixed_action_damaged", "mixed_three_corruptions"]
TRAIN = dict(zip(CONDITIONS, ["C", "I", "S", "A", "M"], strict=True))


def esc(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def num(value, places=3):
    return f"{value:.{places}f}"


def mse(value):
    return f"{value:.3e}" if abs(value) < 0.01 else f"{value:.4f}"


def table(headers, rows, columns=None, size=r"\small"):
    columns = columns or ("l" + "r" * (len(headers) - 1))
    return ("\\begin{center}\n" + size + "\n\\begin{tabular}{" + columns + "}\n\\toprule\n"
            + " & ".join(headers) + r" \\" + "\n\\midrule\n"
            + "\n".join(" & ".join(row) + r" \\" for row in rows)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{center}\n")


def paired_summary(frame, value, keys):
    per_seed = frame.groupby([*keys, "train_seed"])[value].mean()
    return per_seed.groupby(keys).agg(["mean", "std"])


def build(study: Path, output: Path):
    summary = json.loads((study / "summary.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(study / "results.csv")
    aggregate = pd.read_csv(study / "aggregate.csv")
    paired = pd.read_csv(study / "quality_vs_act.csv")
    metadata = pd.read_csv(study / "metadata_comparison.csv")
    expected = {"baseline": 270, "main": 2430, "metadata": 810, "severity": 648}
    if summary["pilot"] or summary["completed_evaluations"] != 4158 or frame.groupby("group").size().to_dict() != expected:
        raise ValueError("The report requires the complete prespecified 4,158-evaluation study")
    if len(set(summary["record_paths"])) != 4158:
        raise ValueError("Evaluation records are missing or duplicated")
    original = Path(summary["source_study"])
    manifests = {d: json.loads((original / d / "dataset_manifest.json").read_text(encoding="utf-8")) for d in DATASETS}
    baseline_audits, test_frames = [], {}
    for relative in summary["record_paths"]:
        if Path(relative).parent.name != "clean":
            continue
        record = json.loads((study / relative).read_text(encoding="utf-8"))
        baseline_audits.append(record["baseline_audit"])
        if record["dataset"] not in test_frames:
            manifest = json.loads((study / relative).with_name("corruption.json").read_text(encoding="utf-8"))
            test_frames[record["dataset"]] = manifest["frames"]
    if len(baseline_audits) != 270 or not all(a["passed"] for a in baseline_audits):
        raise ValueError("Historical baseline audits did not all pass")
    max_difference = max(a["prediction_max_abs_difference"] for a in baseline_audits)
    main = aggregate[(aggregate.group == "main") & (aggregate.quality_mode == "oracle")]
    pair = paired_summary(paired[paired.severity == "medium"], "quality_to_act_mse", ["dataset", "train_condition", "test_condition"])
    meta = paired_summary(metadata, "unknown_to_oracle_mse", ["dataset", "model", "train_condition", "test_condition"])
    file_names = ["summary.json", "results.csv", "aggregate.csv", "quality_vs_act.csv", "metadata_comparison.csv"]
    evidence = {
        "report_version": VERSION, "source_study": str(study.resolve()),
        "evaluation_counts": expected, "baseline_audits_passed": len(baseline_audits),
        "baseline_max_prediction_difference": max_difference,
        "source_sha256": {name: hashlib.sha256((study / name).read_bytes()).hexdigest() for name in file_names},
        "test_frames": test_frames,
    }
    historical_paths = {lang: Path(__file__).resolve().parents[1] / "output/pdf" /
                        ("sync2act_mixed_modality_damage_report_v1_3_0" + ("_en" if lang == "en" else "") + ".tex")
                        for lang in ("zh", "en")}
    evidence["historical_report_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in historical_paths.values()}
    evidence["scope"] = "Integrated study: training quality, test robustness, metadata and temporal ensembling"
    output.mkdir(parents=True, exist_ok=True)
    for lang in ("zh", "en"):
        zh = lang == "zh"

        def L(chinese, english, zh=zh):
            return chinese if zh else english

        def section(chinese, english):
            return "\\clearpage\n\\section{" + L(chinese, english) + "}\n"

        def para(chinese, english):
            return L(chinese, english) + "\n\n"

        test_label = {"image": L("图像", "Image"), "state": L("状态", "State"), "mixed": L("混合", "Mixed")}
        document_class = r"\documentclass[11pt,a4paper,fontset=fandol]{ctexart}" if zh else r"\documentclass[11pt,a4paper]{article}"
        title = L("测试观测损坏与策略鲁棒性", "Test-Time Observation Corruption")
        s = document_class + r"""
\usepackage[a4paper,top=20mm,bottom=19mm,left=19mm,right=19mm]{geometry}
\usepackage{fontspec}
\setmainfont{TeX Gyre Pagella}
\setsansfont{TeX Gyre Heros}
\setmonofont{Latin Modern Mono}
\usepackage{booktabs,tabularx,array,amsmath,amssymb}
\usepackage{xcolor,tcolorbox,graphicx,float,fancyhdr,hyperref,enumitem,microtype}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\definecolor{navy}{HTML}{17324D}
\definecolor{blue}{HTML}{3157D5}
\definecolor{orange}{HTML}{D56B2D}
\definecolor{ink}{HTML}{263442}
\definecolor{muted}{HTML}{66788A}
\definecolor{panel}{HTML}{F3F6FA}
\definecolor{rule}{HTML}{D8E0E8}
\hypersetup{colorlinks=true,linkcolor=navy,urlcolor=blue}
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{2em}
\setlength{\parskip}{0.5em}
\setlength{\headheight}{17pt}
\setlist{nosep,leftmargin=1.5em}
\renewcommand{\arraystretch}{1.18}
\pagestyle{fancy}
\fancyhf{}
\lhead{\small\textcolor{muted}{Sync2Act / Test-Time Corruption}}
\rhead{\small\textcolor{muted}{v1\_4\_0}}
\cfoot{\textcolor{muted}{\thepage}}
\renewcommand{\headrulewidth}{0.4pt}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcommand{\code}[1]{\texttt{\small #1}}
\newcommand{\note}[1]{\begin{tcolorbox}[colback=panel,colframe=rule,boxrule=0.5pt,arc=1mm]#1\end{tcolorbox}}
\newcommand{\metric}[1]{\textcolor{blue}{\bfseries #1}}
\begin{document}
\begin{titlepage}
\vspace*{5mm}
{\color{blue}\rule{25mm}{2.5pt}}\\[7mm]
{\fontsize{30}{36}\selectfont\bfseries\color{navy}Sync2Act\par}
\vspace{3mm}
"""
        s += "{\\LARGE\\bfseries\\color{navy}" + title + "\\par}\n"
        s += "\\vspace{4mm}{\\large\\color{muted}" + L("三类机器人数据集 / 六组消融 / 干净与受损测试配对", "Three robot datasets / six ablations / paired clean and damaged tests") + "\\par}\n"
        s += r"\vspace{7mm}\note{" + table(
            [L("报告版本", "Report version"), r"\texttt{v1\_4\_0}"],
            [[L("评估完成", "Evaluations"), r"\metric{4,158 / 4,158}"],
             [L("固定模型", "Fixed checkpoints"), "270"],
             [L("干净基线核对", "Historical baseline audits"), "270 / 270"],
             [L("最大逐帧预测差异", "Maximum prediction difference"), f"{max_difference:g}"],
             [L("报告日期", "Report date"), "2026-10-07"]], "lr") + "}\n"
        s += r"\vspace{5mm}{\large\bfseries\color{navy}" + L("主要发现", "Principal findings") + "}\\par\n"
        s += para(r"\textbf{状态损坏是主要压力源。}对于干净训练的 ACT-Lite，中等状态损坏使 ALOHA、Koch 和 xArm 的平均 MSE 分别达到其干净测试基线的 $26.40\times$、$28.15\times$ 和 $1.21\times$。", r"\textbf{State damage is the stronger stressor.} For clean-trained ACT-Lite, medium state corruption raises mean MSE ratios to $26.40\times$, $28.15\times$, and $1.21\times$ on ALOHA, Koch, and xArm, respectively.")
        s += para(r"\textbf{质量感知并非普遍胜出。}在混合损坏训练、中等状态损坏测试下，Q-Full / ACT-Lite 的配对 MSE 比值分别为 $1.361$、$0.970$ 和 $0.979$。小于 1 才表示 Q-Full 的绝对误差更低。", r"\textbf{Quality awareness is not universally better.} With mixed-damage training and medium state damage at test time, paired Q-Full / ACT-Lite MSE ratios are $1.361$, $0.970$, and $0.979$. Only ratios below 1 indicate lower absolute error for Q-Full.")
        s += para(r"\textbf{已知质量信息也不是性能上界。}在 ALOHA 与 Koch 的干净训练模型上，未知质量信息条件有时比 Oracle 条件误差更低。", r"\textbf{Known quality information is not a performance upper bound.} For clean-trained models on ALOHA and Koch, unknown-quality tests sometimes have lower error than Oracle tests.")
        s += r"\vfill\note{" + L("本报告复用历史权重，未重新训练。动作标签和训练归一化统计量保持不变。全部结果为离线首步动作预测，不能解释为真实机器人任务成功率。", "Historical weights are reused without retraining. Action targets and training normalization remain fixed. All results concern offline first-step action prediction, not real-robot task success.") + "}\n\\end{titlepage}\n\\setcounter{page}{2}\n"

        # Page 2: scope and datasets.
        s += "\\section{" + L("研究范围与数据划分", "Scope and dataset partitions") + "}\n"
        s += para(r"v1\_3\_0 研究的是训练示范损坏后，在干净测试集上的表现。本报告补充模型使用阶段的图像和状态损坏：同一 checkpoint 先在原始测试轨迹上评估，再在这些轨迹的观测副本上注入损坏。历史的训练损坏条件仍作为独立维度保留。", r"The v1\_3\_0 study measured learning from damaged demonstrations using clean test episodes. This report adds observation damage at model-use time: each checkpoint is evaluated first on original held-out trajectories and then on corrupted observation copies. Historical training corruption remains a separate experimental dimension.")
        s += table([L("训练", "Training"), L("测试", "Test"), L("本报告中的位置", "Role in this report")],
                   [[L("干净", "Clean"), L("干净", "Clean"), L("历史基线的重新核对", "Recomputed historical baseline")],
                    [L("受损", "Damaged"), L("干净", "Clean"), L("历史基线的重新核对", "Recomputed historical baseline")],
                    [L("干净", "Clean"), L("受损", "Damaged"), L("新增测试鲁棒性结果", "New test-robustness results")],
                    [L("受损", "Damaged"), L("受损", "Damaged"), L("新增跨条件结果", "New cross-condition results")]], "llp{94mm}")
        rows = []
        for d in DATASETS:
            m = manifests[d]
            split = m["split"]
            loaded = m["loaded_metadata"]
            rows.append([DS[d], str(m["source_episodes"]), f"{m['source_frames']:,}", str(m["camera_count"]),
                         f"{loaded['state_dim']} / {loaded['action_dim']}",
                         " / ".join(str(len(split[k])) for k in ("train", "validation", "test")), f"{test_frames[d]:,}"])
        s += table([L("数据集", "Dataset"), "Ep.", "Frames", "Cam.", "S / A", "Train / Val / Test", L("测试帧", "Test frames")], rows, "lrrrrrr", r"\footnotesize")
        s += para("ALOHA 对应静态电池操作，Koch 对应单块 Lego 拾放，xArm 对应抬升任务。全部同步相机参与输入。数据先按 episode 划分；测试样本不会因损坏而移入训练或验证分区。", "ALOHA is the static-battery task, Koch is single-Lego pick-and-place, and xArm is the lift task. All synchronized cameras are used. Partitions are episode-based; corruption never moves a test sample into training or validation.")
        s += para("沿用历史划分种子 7；图像最长边为 64 像素。历史模型训练使用 10 epochs、batch size 256、动作 horizon 8，训练种子为 7、17、27。每个数据集的归一化统计量仅来自干净训练 episodes，并由 checkpoint 恢复。", "The historical split seed is 7; images have a maximum side length of 64 pixels. Historical training used 10 epochs, batch size 256, action horizon 8, and training seeds 7, 17, and 27. Normalization is fitted only on clean training episodes and restored from each checkpoint.")
        s += r"\note{" + L("本报告的“干净”指原始、未追加本轮合成故障的记录，并不证明真实示范本身完全没有传感器噪声或标签误差。", "Here, clean means the original recordings without the added synthetic faults; it does not establish that real demonstrations are free of sensor noise or label error.") + "}\n"

        # Page 3: models and historical training intervention.
        s += section("固定模型与训练条件", "Fixed models and training conditions")
        s += para("ACT-Lite 是轻量动作块预测模型，省略原始 ACT 的 CVAE 路径。Q-Full 等质量模型复用其骨干，按相机和状态分别处理观测质量。BC-MLP 虽由软件支持，但没有进入这 270 个历史 checkpoint 的实验矩阵。", "ACT-Lite is a lightweight action-chunk predictor that omits the original ACT CVAE path. Quality variants reuse its backbone and condition camera/state observations separately. BC-MLP is supported by the application but is not part of these 270 historical checkpoints.")
        s += table([L("模型", "Model"), L("训练时质量输入", "Training quality input"), L("标签加权", "Label weighting")],
                   [["ACT-Lite", L("无", "None"), L("否", "No")], ["Q-Input", L("真实合成元数据", "True synthetic metadata"), L("否", "No")],
                    ["Q-Weighted", L("无", "None"), L("是", "Yes")], ["Q-Full", L("真实合成元数据", "True synthetic metadata"), L("是", "Yes")],
                    ["Q-Shuffled", L("打乱的质量元数据", "Shuffled quality metadata"), L("是，使用打乱值", "Yes, shuffled values")],
                    ["Q-Constant", L("恒定为正常", "Constant normal metadata"), L("是，使用常数值", "Yes, constant values")]], "lp{68mm}l")
        s += para("下表仅描述历史训练数据，不是本轮受损测试的配方。C/I/S/A/M 是后续表格使用的训练条件缩写。", "The following table describes historical training data, not the new test-corruption recipes. C/I/S/A/M are training-condition abbreviations used in subsequent tables.")
        s += table([L("条件", "Condition"), L("训练片段构成", "Training segment composition")],
                   [["C", L(r"100\% 干净", r"100\% clean")],
                    ["I", L(r"50\% 干净；40\% 图像延迟 2 帧；10\% 图像缺失", r"50\% clean; 40\% image delay (2 frames); 10\% image missing")],
                    ["S", L(r"50\% 干净；40\% 状态延迟 2 帧；10\% 状态缺失", r"50\% clean; 40\% state delay (2 frames); 10\% state missing")],
                    ["A", L(r"50\% 干净；40\% 动作延迟 2 帧；10\% 动作标签缺失", r"50\% clean; 40\% action delay (2 frames); 10\% action-label missing")],
                    ["M", L(r"20\% 干净；图像/状态延迟各 20\%；动作延迟 25\%；图像/状态/动作缺失各 5\%", r"20\% clean; 20\% each image/state delay; 25\% action delay; 5\% each image/state/action missing")]], "lp{133mm}", r"\footnotesize")
        s += r"\note{" + L("本轮 Oracle 测试向六种训练变体提供相同的真实观测质量元数据，包括训练时使用 shuffled/constant 的模型。训练消融与推理时的元数据干预是两种不同的对照。动作标签质量不参与测试指标加权。", "All six variants receive the same true observation metadata in Oracle tests, including models trained with shuffled/constant metadata. Training ablations and inference-time metadata interventions are distinct controls. Action-label quality never weights test metrics.") + "}\n"

        # Page 4: corruption and size.
        s += section("受损测试协议与实验数量", "Test corruption protocol and evaluation counts")
        s += para("每条测试轨迹划分为连续的 16 帧片段，按确定性随机种子互斥分配故障类型。延迟读取同一 episode 的过去观测，缺失输入置零。边界不足以读取过去帧时，使用缺失标记；不删帧、不跨 episode，也不读取未来帧。", "Test trajectories are divided into disjoint 16-frame segments with deterministic fault assignments. Delays read past observations within the same episode; missing inputs are zeroed. Unavailable boundary observations are marked missing. Frames are not removed, episodes are not crossed, and future frames are never read.")
        s += r"\[\widetilde{o}_t=o_{t-\Delta},\quad \Delta>0;\qquad \widetilde{a}^{\,\mathrm{target}}_t=a_t.\]" + "\n"
        s += table([L("强度", "Severity"), L("延迟帧数", "Delay frames"), L("单模态：延迟 / 缺失", "Single modality: delayed / missing"), L("混合：延迟 / 缺失", "Mixed: delayed / missing")],
                   [[L("轻", "Light"), "1", "0.20 / 0.05", "0.30 / 0.10"],
                    [L("中", "Medium"), "2", "0.40 / 0.10", "0.60 / 0.20"],
                    [L("重", "Heavy"), "4", "0.60 / 0.20", "0.70 / 0.30"]], "lrrr", r"\footnotesize")
        s += para(r"单模态条件分别损坏图像或状态。混合条件把延迟份额、缺失份额各自平分给图像与状态；例如中等混合测试为 20\% 干净、30\% 图像延迟、30\% 状态延迟、10\% 图像缺失、10\% 状态缺失。余下份额为干净片段。", r"Single-modality conditions damage either images or state. Mixed conditions split the delayed and missing fractions equally between image and state. Medium mixed tests therefore use 20\% clean, 30\% image delay, 30\% state delay, 10\% image missing, and 10\% state missing. Remaining segments are clean.")
        s += para("这些比例按片段分配，并非逐帧独立丢失概率。episode 尾部短片段会使实际受损帧比例与目标略有不同；逐评估清单保存实际比例和毫秒偏移。三个测试损坏种子为 107、117、127，与训练种子独立。相同数据集、条件和种子下，所有模型使用同一组受损输入。", "Fractions allocate segments rather than independent frame-drop probabilities. Short final segments can shift actual affected-frame proportions; per-evaluation manifests record realized fractions and millisecond offsets. Test-damage seeds 107, 117, and 127 are separate from training seeds. Every model sees the same damaged inputs for a given dataset, condition, and damage seed.")
        s += table([L("实验组", "Evaluation group"), L("构成", "Construction"), L("数量", "Count")],
                   [[L("干净基线", "Clean baseline"), r"$270$", "270"],
                    [L("中等 Oracle", "Medium Oracle"), r"$270\times3\times3$", "2,430"],
                    [L("未知元数据", "Unknown metadata"), r"$90\times3\times3$", "810"],
                    [L("轻 / 重强度", "Light / heavy"), r"$36\times2\times3\times3$", "648"],
                    [L("总计", "Total"), "", r"\textbf{4,158}"]], "lcr")
        s += para("未知元数据组覆盖 Q-Input、Q-Full 的全部训练条件。强度扩展覆盖 ACT-Lite、Q-Full 的 C 和 M 训练条件。轻/中/重同时改变延迟和受损比例，因此不应解释为单变量敏感性实验。", "Unknown-metadata tests cover every training condition of Q-Input and Q-Full. Severity extensions cover ACT-Lite and Q-Full with C and M training. Light/medium/heavy jointly change delay and damaged fraction, so they are not single-factor sensitivity experiments.")

        # Page 5: metrics and uncertainty.
        s += section("指标、分母与随机种子统计", "Metrics, denominators, and seed variation")
        s += para("模型输出动作块后，仅取首步预测，并反归一化到原数据集动作单位。以下 MSE 对所有测试帧与动作维度等权平均，不按质量分数降权；MAE 同样计算并保存在结果表中。", "Only the first action in each predicted chunk is evaluated, after denormalization to original dataset action units. MSE weights every test frame and action dimension equally, without quality weighting. MAE is computed analogously and retained in the result files.")
        s += r"\[ E_{m,c,s,k,r}=\frac{1}{ND}\sum_{t=1}^{N}\sum_{j=1}^{D}(\hat a_{t,j}-a_{t,j})^2.\]" + "\n"
        s += para(r"其中 $m$ 为模型，$c$ 为训练条件，$s$ 为训练种子，$k$ 为测试损坏条件，$r$ 为测试损坏种子。记同一 checkpoint 的干净测试误差为 $E^0_{m,c,s}$。", r"Here $m$ indexes the model, $c$ the training condition, $s$ the training seed, $k$ the test corruption, and $r$ the damage seed. Let $E^0_{m,c,s}$ denote clean-test error for the same checkpoint.")
        s += r"\begin{align*}R_{m,c,s,k,r}&=E_{m,c,s,k,r}/E^0_{m,c,s},\\ P_{c,s,k,r}&=E_{\mathrm{QF},c,s,k,r}/E_{\mathrm{ACT},c,s,k,r},\\ U_{m,c,s,k,r}&=E^{\mathrm{unknown}}_{m,c,s,k,r}/E^{\mathrm{Oracle}}_{m,c,s,k,r}.\end{align*}" + "\n"
        s += para(r"$R>1$ 表示相对于该模型自身的干净测试发生退化。$P<1$ 表示同训练/测试条件下 Q-Full 的绝对 MSE 更低。$U>1$ 表示未知质量信息的误差更高。三种比值的分母不同，不可互换。尤其是不同训练条件下的 $R$ 更小，并不自动证明绝对 MSE 更低。", r"$R>1$ indicates degradation relative to that checkpoint's clean test. $P<1$ means Q-Full has lower absolute MSE under matched train/test conditions. $U>1$ means unknown metadata produces higher error. These denominators are different and cannot be interchanged. A lower $R$ across training conditions does not by itself establish lower absolute MSE.")
        s += r"\[\bar R_s=\frac{1}{3}\sum_{r}R_{s,r},\quad \bar R=\frac{1}{3}\sum_s\bar R_s,\quad s_{\mathrm{train}}=\sqrt{\frac{\sum_s(\bar R_s-\bar R)^2}{3-1}}.\]" + "\n"
        s += para(r"报告中的 $\bar R\pm s_{\mathrm{train}}$ 先在每个训练种子内平均三个损坏种子，再统计三个训练种子的均值和样本标准差。另一个统计量是各训练种子内损坏种子标准差的平均值，保存在 \code{aggregate.csv}。这不是 9 次独立训练，不给出显著性检验或置信区间结论。", r"Reported $\bar R\pm s_{\mathrm{train}}$ first averages the three damage seeds within each training seed, then computes the mean and sample SD across the three training seeds. The mean within-training-seed damage SD is also retained in \code{aggregate.csv}. These are not nine independent training runs; no significance or confidence-interval claim is made.")
        s += r"\note{" + L("不同数据集的动作尺度不同，绝对 MSE 不跨数据集合并。比值按每次配对计算后求平均，而不是“平均分子 / 平均分母”。分母为零时，比值记为未定义。", "Action scales differ across datasets, so raw MSE is not pooled across datasets. Ratios are formed for each matched pair before averaging, rather than dividing aggregate means. A zero denominator yields an undefined ratio.") + "}\n"

        def main_table(training, test_label=test_label, L=L):
            rows = []
            for d in DATASETS:
                for model in ("act_lite", "quality_full"):
                    for condition in ("image", "state", "mixed"):
                        row = main[(main.dataset == d) & (main.model == model) & (main.train_condition == training) & (main.test_condition == condition)].iloc[0]
                        rows.append([DS[d], MODEL[model], test_label[condition], mse(row.action_mse),
                                     rf"${row.mse_ratio:.3f}\pm{row.training_seed_sd:.3f}$"])
            return table([L("数据集", "Dataset"), L("模型", "Model"), L("测试", "Test"), L("平均 MSE", "Mean MSE"), r"$\bar R\pm s_{\mathrm{train}}$"], rows, "lllrr", r"\footnotesize")

        # Pages 6 and 7: main results.
        s += section("主要结果：干净训练模型", "Main results: clean-trained models")
        s += para("本页仅比较 C 训练条件、中等损坏、Oracle 元数据。每一行对应 3 个训练种子与 3 个损坏种子的配对评估。MSE 是原动作单位下的平均误差，R 的分母是同一 checkpoint 的干净测试误差。", "This page fixes training to C, damage severity to medium, and metadata to Oracle. Each row covers three training seeds and three damage seeds. MSE is the mean error in original action units; R uses the same checkpoint's clean-test denominator.")
        s += main_table("clean")
        s += para("图像受损对三个数据集的影响显著小于状态受损，但并非完全无影响。例如 ALOHA 的 ACT-Lite 图像条件平均 R 为 1.158；状态条件达到 26.397。xArm 的相应比值为 1.000 和 1.209，说明不同任务对同一故障配方的敏感性差异很大。", "Image damage has a much smaller effect than state damage on these datasets, but its effect is not zero. For ALOHA ACT-Lite, mean R is 1.158 for image damage and 26.397 for state damage. The corresponding xArm values are 1.000 and 1.209, showing substantial task dependence under the same fault recipe.")
        s += r"\note{" + L("干净训练的 Q-Full 在 ALOHA 与 Koch 的状态/混合损坏下没有显示优势；xArm 的趋势不同。不能用单一数据集或单个 seed 推断质量机制普遍有效。", "Clean-trained Q-Full does not show an advantage under state/mixed damage on ALOHA and Koch; xArm follows a different pattern. A single dataset or seed cannot establish universal effectiveness of the quality mechanism.") + "}\n"

        s += section("主要结果：混合损坏训练模型", "Main results: models trained with mixed damage")
        s += para("本页固定为 M 训练条件、中等测试损坏和 Oracle 元数据。与上一页的绝对 MSE 对照，可观察历史损坏训练模型在同一测试故障下的表现；各页 R 的分母仍是各自 checkpoint 的干净测试误差。", "This page fixes training to M, test damage to medium, and metadata to Oracle. Compare absolute MSE with the previous page to examine historical damage-trained models under the same test faults; each R still uses its own checkpoint's clean-test denominator.")
        s += main_table("mixed_three_corruptions")
        pair_rows = []
        for d in DATASETS:
            pair_rows.append([DS[d], *[num(pair.loc[(d, "mixed_three_corruptions", k), "mean"]) for k in ("image", "state", "mixed")]])
        s += table([r"$P=$ Q-Full / ACT-Lite", test_label["image"], test_label["state"], test_label["mixed"]], pair_rows, "lrrr", r"\footnotesize")
        s += para(r"Q-Full 在 ALOHA 的状态和混合测试中分别比 ACT-Lite 高约 36.1\% 和 36.2\% 的配对 MSE；Koch 与 xArm 的比值略低于 1。这里报告的是观察到的均值，不意味着小幅差异具有统计显著性。", r"Q-Full has about 36.1\% and 36.2\% higher paired MSE than ACT-Lite for ALOHA state and mixed tests. Koch and xArm ratios are slightly below 1. These are observed means, not claims that small differences are statistically significant.")

        # Page 8: quality availability.
        s += section("已知与未知质量信息的配对对照", "Paired Oracle versus unknown quality information")
        s += para("Oracle 使用合成故障产生的质量分数、缺失标记和时间偏移；Unknown 将观测质量设为 1、缺失标记和时间偏移设为 0。两者使用完全相同的受损图像/状态与同一 checkpoint。Unknown 不会修复观测。下表展示 C 与 M 训练条件，其他训练条件见机器可读结果。", "Oracle uses quality scores, missing flags, and time offsets from the synthetic fault process. Unknown sets observation quality to 1 and missing flags/time offsets to 0. Both use exactly the same damaged image/state inputs and checkpoint. Unknown does not repair observations. The table shows C and M training; other training conditions are retained in machine-readable results.")
        rows = []
        for d in DATASETS:
            for model in ("quality_input", "quality_full"):
                for train in ("clean", "mixed_three_corruptions"):
                    rows.append([DS[d], MODEL[model], TRAIN[train], *[num(meta.loc[(d, model, train, k), "mean"]) for k in ("image", "state", "mixed")]])
        s += table([L("数据集", "Dataset"), L("模型", "Model"), L("训练", "Train"), test_label["image"], test_label["state"], test_label["mixed"]], rows, "lllrrr")
        s += para(r"表内为 $U=\mathrm{MSE}_{\mathrm{unknown}}/\mathrm{MSE}_{\mathrm{Oracle}}$ 的配对均值。对于 M 训练的 Q-Full，状态损坏下的 U 在 ALOHA、Koch 和 xArm 分别约为 1.507、1.389、1.099，说明隐去正确质量信息会提高这些条件下的平均误差。", r"Cells contain paired mean $U=\mathrm{MSE}_{\mathrm{unknown}}/\mathrm{MSE}_{\mathrm{Oracle}}$. For M-trained Q-Full with state damage, U is approximately 1.507, 1.389, and 1.099 on ALOHA/Koch/xArm: hiding correct quality information raises mean error in these conditions.")
        s += para("但 C 训练的 Q-Full 在 ALOHA/Koch 状态损坏下的 U 约为 0.507/0.580，即 Unknown 反而更好。一个可能解释是，干净训练没有让模型学会正确处理低质量输入及其元数据；这只是待验证假设，不能由本实验单独证明。", "However, C-trained Q-Full has state-damage U of about 0.507/0.580 on ALOHA/Koch, so Unknown is better. One possible explanation is that clean training does not teach the model to use low-quality observations and their metadata appropriately. This is a hypothesis, not a mechanism established by this experiment.")
        s += r"\note{" + L("Oracle 在这里表示已知合成故障来源，并不表示质量分数经过概率校准，也不是模型性能的理论上界。部署时如何自动检测质量，未在本轮实验中解决。", "Oracle means that the synthetic fault source is known. Scores are neither calibrated probabilities nor a theoretical upper bound on model performance. Automatic quality detection at deployment is outside this study.") + "}\n"

        # Page 9: embedded editable vector figures.
        s += section("损坏强度曲线", "Corruption severity curves")
        s += para("仅展示 ACT-Lite 与 Q-Full、C 与 M 训练条件、Oracle 元数据。纵轴为同一 checkpoint 的 MSE 退化比 R，使用对数刻度。每个点先平均损坏种子，再平均训练种子。接近基线的面板至少覆盖 0.9--1.1；各面板纵轴范围不同。", "Curves cover ACT-Lite and Q-Full, C and M training, and Oracle metadata. The logarithmic y axis shows same-checkpoint degradation R. Points average damage seeds, then training seeds. Near-baseline panels span at least 0.9--1.1; y-axis ranges differ across panels.")
        for d in DATASETS:
            s += r"\begin{center}\begin{tikzpicture}\begin{groupplot}[group style={group size=3 by 1,horizontal sep=9mm},width=53mm,height=47mm,scale only axis=false,ymode=log,log ticks with fixed point,grid=major,grid style={gray!15},tick label style={font=\scriptsize},title style={font=\small},xlabel style={font=\scriptsize},xtick={1,2,3},xticklabels={" + L("轻,中,重", "Light,Medium,Heavy") + r"},xmin=0.8,xmax=3.2,/tikz/mark size=1.7pt]" + "\n"
            for condition in ("image", "state", "mixed"):
                local = aggregate[(aggregate.dataset == d) & aggregate.model.isin(["act_lite", "quality_full"]) & aggregate.train_condition.isin(["clean", "mixed_three_corruptions"]) & (aggregate.quality_mode == "oracle") & (aggregate.test_condition == condition)]
                lo, hi = min(0.9, local.mse_ratio.min() * .9), max(1.1, local.mse_ratio.max() * 1.1)
                tick_candidates = (0.9, 1, 1.1, 1.25, 1.5, 2) if hi < 3 else (1, 3, 10, 30, 100, 300)
                ticks = ",".join(f"{tick:g}" for tick in tick_candidates if lo <= tick <= hi)
                s += rf"\nextgroupplot[title={{{DS[d]} / {test_label[condition]}}},ymin={lo:.8g},ymax={hi:.8g},ytick={{{ticks}}},yticklabels={{{ticks}}},minor y tick num=0]" + "\n"
                s += r"\addplot[gray,thin,forget plot] coordinates {(1,1)(3,1)};" + "\n"
                for model, color in (("act_lite", "blue"), ("quality_full", "orange")):
                    for train, style in (("clean", "dashed"), ("mixed_three_corruptions", "solid")):
                        coords = []
                        for index, severity in enumerate(("light", "medium", "heavy"), 1):
                            value = local[(local.model == model) & (local.train_condition == train) & (local.severity == severity)].iloc[0].mse_ratio
                            coords.append(f"({index},{value:.10g})")
                        s += rf"\addplot[{color},{style},thick,mark=*] coordinates {{{' '.join(coords)}}};" + "\n"
            s += "\\end{groupplot}\\end{tikzpicture}\\end{center}\n"
        s += para(r"\textcolor{blue}{蓝色：ACT-Lite}；\textcolor{orange}{橙色：Q-Full}。虚线：C 训练；实线：M 训练。图中展示均值，未绘制置信区间。", r"\textcolor{blue}{Blue: ACT-Lite}; \textcolor{orange}{orange: Q-Full}. Dashed: C training; solid: M training. Curves show means, not confidence intervals.")
        s += para("状态/混合故障随强度上升通常呈更大退化，ALOHA 与 Koch 最为明显。R 的大小不能代替跨模型绝对误差比较；对模型优劣应同时查看第 6 节的配对 P。", "State/mixed faults generally cause larger degradation with increasing severity, especially on ALOHA and Koch. R is not a substitute for cross-model absolute-error comparisons; consult the paired P values in Section 6 when comparing models.")

        # Page 10: audit, software, limitations, reproducibility.
        s += section("审计、软件入口与结论边界", "Audit, software workflow, and limitations")
        s += para(f"270 个干净基线全部通过历史目标值、预测值和 MSE 核对，最大逐帧预测绝对差异为 {max_difference:g}。所有测试动作标签、时间戳、episode 顺序和训练归一化均固定。运行清单记录 checkpoint 哈希、代码哈希、测试张量哈希、测试划分及损坏配置，输出文件使用 SHA-256 校验。", f"All 270 clean baselines passed historical target, prediction, and MSE checks; the maximum absolute prediction difference was {max_difference:g}. Test action targets, timestamps, episode order, and training normalization are fixed. Manifests record checkpoint/code/test-tensor hashes, test partitions, and corruption settings; artifacts are checked with SHA-256.")
        s += table([L("执行环境", "Runtime"), L("记录值", "Recorded value")],
                   [["Python", esc(summary["runtime"]["python"])], ["PyTorch", esc(summary["runtime"]["torch"])],
                    ["NumPy", esc(summary["runtime"]["numpy"])], ["GPU", esc(summary["runtime"]["gpu"])],
                    [L("批量评估耗时", "Batch evaluation wall time"), f"{summary['wall_seconds']/60:.2f} min"]], "lp{102mm}")
        s += para("耗时仅表示本机离线批量流程，不是实时控制延迟。新增软件界面支持干净/受损配对评估、摄像头选择、随机种子、Oracle/Unknown 切换、后台执行与取消，以及带配置的 HTML/JSON 导出。实现验证记录为 120 项测试通过。", "Wall time describes this machine's offline batch workflow, not real-time control latency. The desktop interface now supports paired clean/damaged evaluation, camera selection, seeds, Oracle/Unknown switching, background execution/cancellation, and HTML/JSON export with configuration. Implementation validation recorded 120 passing tests.")
        s += r"\begin{tcolorbox}[colback=panel,colframe=rule]\small\ttfamily" + "\n"
        s += r"python tools/evaluate\_test\_corruption\_study.py --device cuda\\" + "\n"
        s += r"python tools/evaluate\_test\_corruption\_study.py --report-only\\" + "\n"
        s += r"python tools/build\_test\_corruption\_latex\_report.py" + "\n\\end{tcolorbox}\n"
        s += para(r"主数据目录为 \code{runs/test\_time\_corruption\_v1/}。\code{results.csv} 保存每次评估；\code{aggregate.csv} 保存两级统计；\code{quality\_vs\_act.csv} 和 \code{metadata\_comparison.csv} 保存配对结果。受损/未受损片段及每 episode 的误差均保留在原始输出中。", r"The main data directory is \code{runs/test\_time\_corruption\_v1/}. \code{results.csv} contains per-evaluation metrics; \code{aggregate.csv} contains hierarchical summaries; \code{quality\_vs\_act.csv} and \code{metadata\_comparison.csv} contain paired comparisons. Affected/unaffected-segment and per-episode errors remain in the original artifacts.")
        s += para(r"限制：只有三个任务和三个训练种子；采用人工故障与启发式质量映射；未评估自动质量估计、真实在线延迟、动作执行反馈或闭环安全；没有将新训练协议用于重训。当前版本只新增报告和测试鲁棒性证据，不意味着已发布 v1\_4\_0 软件安装包。", r"Limitations: three tasks and three training seeds; synthetic faults and heuristic quality mappings; no automatic quality estimation, real online latency, execution feedback, or closed-loop safety evaluation; no retraining under the newer training protocol. This version adds a report and test-robustness evidence, not a released v1\_4\_0 application installer.")

        # Three full-matrix appendix pages, not a cherry-picked subset.
        s += "\\appendix\n"
        for d in DATASETS:
            s += section(f"完整中等强度矩阵：{DS[d]}", f"Complete medium-severity matrix: {DS[d]}")
            s += para(r"六组模型、五种训练条件的 Oracle 测试。各单元格为 $\bar R\pm s_{\mathrm{train}}$，每格由 3 个训练种子和 3 个损坏种子组成。C/I/S/A/M 的含义见第 2 节。", r"Oracle tests for six models and five training conditions. Each cell is $\bar R\pm s_{\mathrm{train}}$, using three training seeds and three damage seeds. Training codes C/I/S/A/M are defined in Section 2.")
            rows = []
            for model in MODELS:
                for train in CONDITIONS:
                    cells = [MODEL[model], TRAIN[train]]
                    for condition in ("image", "state", "mixed"):
                        r = main[(main.dataset == d) & (main.model == model) & (main.train_condition == train) & (main.test_condition == condition)].iloc[0]
                        cells.append(rf"${r.mse_ratio:.3f}\pm{r.training_seed_sd:.3f}$")
                    rows.append(cells)
            s += table([L("模型", "Model"), L("训练", "Train"), test_label["image"], test_label["state"], test_label["mixed"]], rows, "llrrr", r"\footnotesize")
            s += r"\note{" + L("接近 1 的比值不等于绝对 MSE 很小，也不证明闭环任务能够成功。所有比较均保留各自的干净测试分母；原始动作尺度和绝对误差请参照 CSV 数据。", "A ratio near 1 does not imply small absolute MSE or successful closed-loop execution. Each comparison retains its own clean-test denominator; consult the CSV data for original-scale absolute errors.") + "}\n"
        s += "\\end{document}\n"
        name = f"sync2act_test_time_corruption_report_v{VERSION}" + ("_en" if not zh else "")
        historical = historical_paths[lang].read_text(encoding="utf-8")
        if len(re.findall(r"\\section\{", historical)) != 10:
            raise ValueError("Expected all 10 sections of the historical report")
        s = compose_integrated_report(s, historical, frame, paired, zh, table, mse)
        (output / f"{name}.tex").write_text(s, encoding="utf-8")
    (output / f"sync2act_test_time_corruption_report_v{VERSION}_provenance.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    stem = f"sync2act_test_time_corruption_report_v{VERSION}"
    with ZipFile(output / f"{stem}_source.zip", "w", compression=ZIP_DEFLATED) as bundle:
        for name in (f"{stem}.tex", f"{stem}_en.tex", f"{stem}_provenance.json"):
            bundle.write(output / name, name)
        for name in file_names:
            bundle.write(study / name, f"evidence/{name}")
        for historical_path in historical_paths.values():
            bundle.write(historical_path, f"historical/{historical_path.name}")
        bundle.writestr("README.txt", (
            f"Sync2Act cumulative training and test corruption report v{VERSION}\n\n"
            "Integrates all v1_3_0 evidence with test-corruption results into one narrative,\n"
            "with shared methods and conclusions. Original sources are in historical/.\n"
            "Compile with XeLaTeX (not pdfLaTeX). In Overleaf select XeLaTeX and\n"
            "choose the Chinese source or its _en.tex English counterpart as main.\n"
            "Use TeX Live with CTeX, PGFPlots, Fandol, and TeX Gyre fonts.\n"
            "All chart coordinates and tables are embedded. No external assets,\n"
            "checkpoints, datasets, or Python runtime are needed to compile.\n\n"
            f"xelatex -interaction=nonstopmode -halt-on-error {stem}.tex\n"
            f"xelatex -interaction=nonstopmode -halt-on-error {stem}_en.tex\n"
            "Run each command twice to stabilize bookmarks.\n\n"
            "evidence/ contains source summaries and metrics, not the original\n"
            "robot datasets or model weights. The provenance JSON records SHA-256\n"
            "hashes for these exact input files. Full precision is retained there;\n"
            "printed values are rounded, so a displayed 0.000 may be nonzero.\n"
        ))
    print(f"Generated bilingual v{VERSION} LaTeX sources in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("runs/test_time_corruption_v1"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf"))
    args = parser.parse_args()
    build(args.study, args.output)


if __name__ == "__main__":
    main()
