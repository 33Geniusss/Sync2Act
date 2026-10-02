"""MSE-only charts and comparisons for the saved temporal-ensemble study."""

from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sync2act.reporting.localization import english_text

MODEL_NAMES = {
    "act_lite": "ACT-Lite",
    "quality_input": "Quality-Input",
    "quality_weighted_loss": "Weighted-Loss",
    "quality_full": "Quality-Full",
    "quality_shuffled": "Shuffled",
    "quality_constant": "Constant",
}
DATASET_NAMES = {
    "xarm_lift_medium": "XArm",
    "koch_pick_place_1_lego": "Koch",
    "aloha_static_battery": "ALOHA",
}
CONDITION_NAMES = {
    "clean": "Clean",
    "mixed_image_damaged": "Image",
    "mixed_state_damaged": "State",
    "mixed_action_damaged": "Action",
    "mixed_three_corruptions": "Mixed",
}
COLORS = ["#17324D", "#2F6BFF", "#E9852D", "#00A6A6", "#D45576", "#7456B8"]
TEX_COLORS = ["navy", "blue", "orange", "teal", "rose", "purple"]
MARKS = ["*", "square*", "triangle*", "diamond*", "pentagon*", "x"]
KEYS = ["dataset", "model", "condition", "seed"]


def prepare_mse_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Normalize within dataset/model/seed and *within each evaluation method*."""
    columns = KEYS + ["first_step_action_mse", "ensemble_action_mse"]
    data = frame[columns].copy()
    if data.duplicated(KEYS).any():
        raise ValueError("duplicate experiment key")
    mse_columns = columns[-2:]
    if not np.isfinite(data[mse_columns]).all().all() or (data[mse_columns] <= 0).any().any():
        raise ValueError("MSE values must be finite and positive")
    clean = data[data.condition == "clean"].drop(columns="condition")
    clean = clean.rename(columns={key: f"clean_{key}" for key in mse_columns})
    data = data.merge(clean, on=["dataset", "model", "seed"], how="left", validate="many_to_one")
    if data[[f"clean_{key}" for key in mse_columns]].isna().any().any():
        raise ValueError("missing matched clean baseline")
    for mode in ["first_step", "ensemble"]:
        data[f"{mode}_relative_mse"] = data[f"{mode}_action_mse"] / data[f"clean_{mode}_action_mse"]
    data["mse_change_percent"] = 100 * (data.ensemble_action_mse / data.first_step_action_mse - 1)
    impact = (
        data[data.condition != "clean"]
        .groupby(["model", "condition"], sort=False)
        .agg(
            first_step_relative_mse=("first_step_relative_mse", "mean"),
            ensemble_relative_mse=("ensemble_relative_mse", "mean"),
            runs=("seed", "size"),
        )
        .reset_index()
    )
    changes = (
        data.groupby(["model", "condition"], sort=False)
        .agg(
            mean_paired_mse_change_percent=("mse_change_percent", "mean"),
            runs=("seed", "size"),
        )
        .reset_index()
    )

    def aggregate(keys):
        result = (
            data.groupby(keys, sort=False)
            .agg(
                first_step_mse=("first_step_action_mse", "mean"),
                ensemble_mse=("ensemble_action_mse", "mean"),
                runs=("seed", "size"),
            )
            .reset_index()
        )
        result["mse_difference"] = result.ensemble_mse - result.first_step_mse
        result["mse_change_percent"] = 100 * (result.ensemble_mse / result.first_step_mse - 1)
        return result

    return {
        "mse_per_run": data,
        "relative_mse_by_model_condition": impact,
        "mse_change_by_model_condition": changes,
        "by_dataset_model": aggregate(["dataset", "model"]),
        "mse_by_dataset_model_condition": aggregate(["dataset", "model", "condition"]),
        "by_model": data.groupby("model")
        .agg(mean_paired_mse_change_percent=("mse_change_percent", "mean"), runs=("seed", "size"))
        .reset_index(),
        "by_condition": data.groupby("condition")
        .agg(mean_paired_mse_change_percent=("mse_change_percent", "mean"), runs=("seed", "size"))
        .reset_index(),
    }


def _impact_svg(impact: pd.DataFrame) -> str:
    # Match the original overall plot's categories, colors and fixed y scale.
    left, top, width, height = 95, 30, 740, 300
    ymin, ymax = 0.85, 2.65
    xs = [left + i * width / 3 for i in range(4)]

    def y(value):
        return top + height * (ymax - value) / (ymax - ymin)

    parts = [
        '<svg viewBox="0 0 900 470" role="img" aria-label="时间集成后的平均相对 MSE" style="width:100%;height:auto;background:white">'
    ]
    for value in [1, 1.5, 2, 2.5]:
        parts.append(
            f'<line x1="{left}" x2="{left + width}" y1="{y(value):.2f}" y2="{y(value):.2f}" stroke="#D8E0E8"/>'
        )
        parts.append(
            f'<text x="80" y="{y(value) + 5:.2f}" text-anchor="end" font-size="14">{value:g}</text>'
        )
    conditions = list(CONDITION_NAMES)[1:]
    for x, condition in zip(xs, conditions, strict=True):
        parts.append(f'<line x1="{x}" x2="{x}" y1="{top}" y2="{top + height}" stroke="#edf0f4"/>')
        parts.append(
            f'<text x="{x}" y="358" text-anchor="middle" font-size="15">{CONDITION_NAMES[condition]}</text>'
        )
    parts.append(
        f'<line x1="{left}" x2="{left + width}" y1="{y(1):.2f}" y2="{y(1):.2f}" stroke="#66788A" stroke-dasharray="7 5"/>'
    )
    parts.append(
        '<text x="22" y="180" transform="rotate(-90 22 180)" text-anchor="middle" font-size="16">平均相对 MSE R（时间集成）</text>'
    )
    indexed = impact.set_index(["model", "condition"])
    for index, (model, label) in enumerate(MODEL_NAMES.items()):
        values = [
            indexed.loc[(model, condition), "ensemble_relative_mse"] for condition in conditions
        ]
        points = " ".join(f"{x:.2f},{y(value):.2f}" for x, value in zip(xs, values, strict=True))
        color = COLORS[index]
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for x, value in zip(xs, values, strict=True):
            parts.append(f'<circle cx="{x}" cy="{y(value):.2f}" r="4" fill="{color}"/>')
        lx, ly = 130 + (index % 3) * 250, 402 + (index // 3) * 32
        parts.append(
            f'<line x1="{lx}" x2="{lx + 30}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="3"/><text x="{lx + 40}" y="{ly + 5}" font-size="14">{label}</text>'
        )
    return "".join(parts) + "</svg>"


def temporal_report_section(study_root: Path) -> str:
    path = study_root / "temporal_ensemble/section.html"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def write_temporal_report(study_root: Path, frame: pd.DataFrame, protocol: dict) -> None:
    output = study_root / "temporal_ensemble"
    output.mkdir(parents=True, exist_ok=True)
    tables = prepare_mse_tables(frame)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    n, decay = len(frame), protocol["decay"]
    overall_change = tables["mse_per_run"].mse_change_percent.mean()
    impact = tables["relative_mse_by_model_condition"].set_index(["model", "condition"])
    changes = tables["mse_change_by_model_condition"].set_index(["model", "condition"])
    impact_rows, change_rows = [], []
    for model, label in MODEL_NAMES.items():
        impact_rows.append(
            {
                "模型": label,
                **{
                    CONDITION_NAMES[c]: impact.loc[(model, c), "ensemble_relative_mse"]
                    for c in list(CONDITION_NAMES)[1:]
                },
            }
        )
        change_rows.append(
            {
                "模型": label,
                **{
                    label: changes.loc[(model, c), "mean_paired_mse_change_percent"]
                    for c, label in CONDITION_NAMES.items()
                },
            }
        )
    impact_table, change_table = pd.DataFrame(impact_rows), pd.DataFrame(change_rows)

    def display_comparison(source):
        result = source.copy()
        result["dataset"] = result.dataset.map(DATASET_NAMES)
        result["model"] = result.model.map(MODEL_NAMES)
        if "condition" in result:
            result["condition"] = result.condition.map(CONDITION_NAMES)
        result = result.drop(columns="runs")
        return result.rename(
            columns={
                "dataset": "数据集",
                "model": "模型",
                "condition": "训练条件",
                "first_step_mse": "未集成 MSE",
                "ensemble_mse": "集成后 MSE",
                "mse_difference": "MSE 差值",
                "mse_change_percent": "变化 (%)",
            }
        )

    # Keep the reader-facing tables in the same dataset/model order as the figure.
    absolute = (
        tables["by_dataset_model"]
        .set_index(["dataset", "model"])
        .loc[[(d, m) for d in DATASET_NAMES for m in MODEL_NAMES]]
        .reset_index()
    )
    absolute_table = display_comparison(absolute)
    full_table = display_comparison(tables["mse_by_dataset_model_condition"])
    method = (
        f"两种评估方式使用同一组 {n} 个 checkpoint、测试 episodes 和归一化统计量；decay={decay:g} 在评估前固定。"
        "只融合起点不晚于当前时刻的预测，并在每条 episode 边界重置历史。"
        f"第一步预测一致性核对为 {protocol['baseline_audits_passed']}/{n} 通过。"
    )
    normalization = (
        "两种评估方式采用相同的归一化规则：先在每个数据集、模型、seed 内，"
        "用该条件的测试 MSE 除以同一模型同一 seed 的干净训练测试 MSE，再平均 9 个比值。"
        "时间集成曲线的分子和分母均使用集成后的 MSE；两种方式的横轴、六模型配色和纵轴范围一致。"
    )
    change_note = (
        "下表逐运行计算 100×(集成后 MSE / 未集成 MSE − 1)，再对 3 数据集 × 3 seeds 取平均。"
        "负数表示 MSE 降低，正数表示 MSE 增加；不跨机器人直接平均原始 MSE。"
    )
    absolute_note = (
        "下表在同一数据集、同一模型内对五个训练条件和三个 seeds 的 15 次运行取平均。"
        "MSE 差值 = 集成后均值 − 未集成均值；变化 (%) = 100×(集成后均值 / 未集成均值 − 1)。"
        "此处的均值之比不同于上表逐运行百分比的均值。"
    )
    conclusion = (
        f"270 组逐运行 MSE 相对变化的平均值为 {overall_change:+.2f}%。"
        "相对退化曲线下降不等于实际 MSE 下降：时间集成也会改变作为分母的干净训练基线。"
        "判断是否更准确，应同时查看未集成与集成后的实际 MSE。结果仅限离线动作预测。"
    )
    svg = _impact_svg(tables["relative_mse_by_model_condition"])
    (output / "ensemble_relative_mse.svg").write_text(svg, encoding="utf-8")
    section = "<!-- temporal-ensemble:start -->\n<section id='temporal-ensemble'>"
    section += "<h2>时间集成后的总体结果与 MSE 变化</h2>"
    section += f"<p>{html.escape(method)}</p><h3>时间集成后的总体结果</h3><p>{html.escape(normalization)}</p>"
    section += svg + impact_table.to_html(index=False, float_format=lambda v: f"{v:.4f}", border=0)
    section += f"<h3>各训练条件下，MSE 相对未集成的变化（%）</h3><p>{html.escape(change_note)}</p>"
    section += change_table.to_html(index=False, float_format=lambda v: f"{v:+.2f}%", border=0)
    section += f"<h3>未集成与集成后的实际 MSE</h3><p>{html.escape(absolute_note)}</p>"
    section += absolute_table.to_html(index=False, float_format=lambda v: f"{v:.6g}", border=0)
    section += (
        "<details><summary>展开全部数据集 × 模型 × 训练条件的 MSE 对照（每行三个 seeds）</summary>"
    )
    section += (
        full_table.to_html(index=False, float_format=lambda v: f"{v:.6g}", border=0) + "</details>"
    )
    section += f"<p>{html.escape(conclusion)}</p><p>图表数据：relative_mse_by_model_condition.csv；"
    section += "逐运行 MSE：mse_per_run.csv；完整条件对照：mse_by_dataset_model_condition.csv。</p></section>\n<!-- temporal-ensemble:end -->"
    (output / "section.html").write_text(section, encoding="utf-8")
    (output / "section_en.html").write_text(english_text(section), encoding="utf-8")
    (output / "ensemble_relative_mse_en.svg").write_text(english_text(svg), encoding="utf-8")
    css = "body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#263442}h2,h3{color:#17324d}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{padding:7px;border-bottom:1px solid #d8e0e8;text-align:left}th{background:#eef2ff}summary{cursor:pointer;font-weight:bold}"
    (output / "report.html").write_text(
        f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Sync2Act 时间集成 MSE</title><style>{css}</style><body>{section}</body></html>",
        encoding="utf-8",
    )
    (output / "report_en.html").write_text(
        english_text((output / "report.html").read_text(encoding="utf-8")), encoding="utf-8"
    )
    main_report = study_root / "report.html"
    if main_report.exists():
        existing = main_report.read_text(encoding="utf-8")
        existing = re.sub(
            r"<!-- temporal-ensemble:start -->.*?<!-- temporal-ensemble:end -->",
            "",
            existing,
            flags=re.S,
        )
        anchor = "<h2>自动汇总结论</h2>"
        existing = (
            existing.replace(anchor, section + "\n" + anchor)
            if anchor in existing
            else existing.replace("</body>", section + "</body>")
        )
        main_report.write_text(existing, encoding="utf-8")
        (study_root / "report_en.html").write_text(english_text(existing), encoding="utf-8")

    tex = [
        r"\clearpage\subsection{时间集成后的总体结果}",
        method,
        normalization,
        r"\[R_{\mathrm{ens}}=\frac{\mathrm{MSE}_{\mathrm{damaged\ train,\ ens}}}{\mathrm{MSE}_{\mathrm{clean\ train,\ ens}}}.\]",
        r"\begin{figure}[H]\centering\begin{tikzpicture}\begin{axis}[",
        r"width=0.99\linewidth,height=77mm,ymin=0.85,ymax=2.65,ylabel={平均 MSE 比值 $R_{\mathrm{ens}}$},",
        r"symbolic x coords={Image,State,Action,Mixed},xtick=data,",
        r"legend style={at={(0.5,-0.24)},anchor=north,legend columns=3,font=\scriptsize,draw=none},",
        r"grid=major,grid style={rule!60},axis line style={rule},tick style={rule},every axis plot/.append style={thick,mark size=2pt}]",
    ]
    for i, (model, label) in enumerate(MODEL_NAMES.items()):
        coordinates = "".join(
            f"({CONDITION_NAMES[c]},{impact.loc[(model, c), 'ensemble_relative_mse']:.8f})"
            for c in list(CONDITION_NAMES)[1:]
        )
        tex += [
            f"\\addplot[color={TEX_COLORS[i]},mark={MARKS[i]}] coordinates {{{coordinates}}};",
            f"\\addlegendentry{{{label}}}",
        ]
    tex += [
        r"\addplot[color=muted,dashed,no marks] coordinates {(Image,1)(Mixed,1)};",
        r"\end{axis}\end{tikzpicture}\caption{时间集成后的平均相对 MSE。每点平均 3 数据集与 3 seeds 的 9 个比值；虚线为时间集成后的干净训练基准。}\end{figure}",
        r"\begin{table}[H]\centering\small\begin{tabular}{lrrrr}\toprule",
        r"模型 & Image & State & Action & Mixed\\\midrule",
    ]
    for row in impact_rows:
        tex.append(
            " & ".join(
                [row["模型"]] + [f"{row[c]:.4f}" for c in ["Image", "State", "Action", "Mixed"]]
            )
            + r"\\"
        )
    tex += [
        r"\bottomrule\end{tabular}\caption{时间集成的平均 MSE 比值，衡量相对干净训练的退化程度。集成前后的实际误差变化见配对比较。}\end{table}",
        r"\clearpage\subsection{集成前后的 MSE 变化}",
        r"\begingroup\small\renewcommand{\arraystretch}{1.08}\setlength{\intextsep}{8pt}",
        change_note.replace("%", r"\%"),
        r"\begin{table}[H]\centering\small\begin{tabularx}{\linewidth}{Yrrrrr}\toprule",
        r"模型 & Clean & Image & State & Action & Mixed\\\midrule",
    ]
    for row in change_rows:
        tex.append(
            " & ".join([row["模型"]] + [f"{row[c]:+.2f}\\%" for c in CONDITION_NAMES.values()])
            + r"\\"
        )
    tex += [
        r"\bottomrule\end{tabularx}\caption{实际 MSE 的平均配对变化百分比。每格 9 次运行；正数表示误差增加。}\end{table}",
        absolute_note.replace("%", r"\%"),
        r"\begin{table}[H]\centering\small\begin{tabularx}{\linewidth}{lYrrrr}\toprule",
        r"数据集 & 模型 & 未集成 MSE & 集成后 MSE & MSE 差值 & 变化\\\midrule",
    ]
    for row in absolute.to_dict("records"):
        tex.append(
            f"{DATASET_NAMES[row['dataset']]} & {MODEL_NAMES[row['model']]} & {row['first_step_mse']:.5g} & {row['ensemble_mse']:.5g} & {row['mse_difference']:+.4g} & {row['mse_change_percent']:+.2f}\\%"
            + r"\\"
        )
    tex += [
        r"\bottomrule\end{tabularx}\caption{同一数据集内的实际 MSE 对照。每行平均五条件、三 seeds，共 15 次运行。}\end{table}",
        conclusion.replace("%", r"\%"),
        r"\noindent 完整条件对照见 HTML 可展开表格及 \path{temporal_ensemble/mse_by_dataset_model_condition.csv}。",
        f"\\par\\noindent 复现：\\code{{python tools/evaluate\\_temporal\\_study.py --device cuda --decay {decay:g} --report-only}}。",
        r"\par\endgroup",
    ]
    tex_text = "\n".join(tex) + "\n"
    target = Path("output/pdf/sync2act_mixed_modality_damage_report_v1_3_0_temporal_section.tex")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(tex_text, encoding="utf-8")
    master = target.with_name("sync2act_mixed_modality_damage_report_v1_3_0.tex")
    if master.is_file():
        content = master.read_text(encoding="utf-8")
        content = re.sub(
            r"% temporal-ensemble:start.*?% temporal-ensemble:end\s*", "", content, flags=re.S
        )
        block = "% temporal-ensemble:start\n" + tex_text + "% temporal-ensemble:end\n"
        anchor = r"\section{按故障模态解读}"
        if anchor not in content:
            raise ValueError("Cannot locate overall-results section in report source")
        content = re.sub(r"(?:\\clearpage\s*)+(?=\\section\{按故障模态解读\})", "", content)
        content = content.replace(anchor, block + r"\clearpage" + "\n" + anchor)
        content = re.sub(
            r"\{\\small\\color\{ink\}\\textbf\{时间集成评估：\}.*?\\par\}",
            lambda _: (
                r"{\small\color{ink}\textbf{时间集成评估：}在同一模型与测试集上比较两种动作预测方式，分别报告相对干净训练的 MSE 比值和集成前后的实际 MSE 变化。\par}"
            ),
            content,
            flags=re.S,
        )
        master.write_text(content, encoding="utf-8")
