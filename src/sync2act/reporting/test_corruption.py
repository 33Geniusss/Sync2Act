"""Bilingual reports for paired test-time observation corruption."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def _severity_chart(aggregate, output):
    """Standalone research figure; plotting is an optional study dependency."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    data = aggregate[
        aggregate.model.isin(["act_lite", "quality_full"])
        & aggregate.train_condition.isin(["clean", "mixed_three_corruptions"])
        & (aggregate.quality_mode == "oracle")
    ]
    if data.empty:
        return
    datasets = sorted(data.dataset.unique())
    levels = ["light", "medium", "heavy"]
    fig, axes = plt.subplots(len(datasets), 3, figsize=(12, 3.1 * len(datasets)), squeeze=False)
    for row, dataset in enumerate(datasets):
        for col, condition in enumerate(("image", "state", "mixed")):
            ax = axes[row, col]
            local = data[(data.dataset == dataset) & (data.test_condition == condition)]
            for model, color in (("act_lite", "#3157d5"), ("quality_full", "#d56b2d")):
                for training, style in (("clean", "--"), ("mixed_three_corruptions", "-")):
                    curve = local[(local.model == model) & (local.train_condition == training)].set_index("severity").reindex(levels)
                    ax.plot(range(3), curve.mse_ratio, marker="o", linestyle=style, color=color,
                            label=f"{model} / {'clean' if training == 'clean' else 'mixed'} train")
            ax.axhline(1, color="#68758c", linewidth=0.8, alpha=0.7)
            ax.set_yscale("log")
            low, high = ax.get_ylim()
            if high / low < 2:
                low, high = min(low, 0.9), max(high, 1.1)
                ax.set_ylim(low, high)
                ticks = [v for v in (0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 2, 3) if low <= v <= high]
                ax.set_yticks(ticks, [f"{v:g}" for v in ticks])
                ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks(range(3), levels)
            ax.set_title(f"{dataset} / {condition}", fontsize=9)
            ax.grid(True, alpha=0.2)
            if col == 0:
                ax.set_ylabel("MSE / clean-test MSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Test corruption severity: Oracle quality metadata\nLogarithmic y axes; near-baseline panels include 0.9–1.1", fontsize=12)
    fig.tight_layout(rect=(0, 0.18 if len(datasets) == 1 else 0.08, 1, 0.86 if len(datasets) == 1 else 0.92))
    fig.savefig(output / "severity_curves.svg", bbox_inches="tight")
    fig.savefig(output / "severity_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_test_corruption_report(records: list[dict], summary: dict, output: Path):
    rows = []
    for record in records:
        row = {key: record[key] for key in ("dataset", "model", "train_condition", "train_seed",
                                           "group", "severity", "test_condition", "quality_mode", "damage_seed",
                                           "actual_affected_fraction")}
        metrics = record["metrics"]
        row.update({key: metrics.get(key) for key in ("action_mse", "action_mae", "mse_ratio_to_clean_test", "mse_delta_from_clean_test")})
        for group in ("affected", "unaffected"):
            for key in ("frames", "action_mse", "action_mae"):
                row[f"{group}_{key}"] = metrics["frame_groups"][group][key]
        rows.append(row)
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "results.csv", index=False)
    (output / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    dirty = frame[frame.group != "baseline"]
    keys = ["dataset", "model", "train_condition", "group", "severity", "test_condition", "quality_mode"]
    per_seed = dirty.groupby([*keys, "train_seed"], dropna=False).agg(
        action_mse=("action_mse", "mean"), action_mae=("action_mae", "mean"),
        mse_ratio=("mse_ratio_to_clean_test", "mean"), damage_seed_sd=("mse_ratio_to_clean_test", "std"),
        damage_seed_count=("damage_seed", "nunique"),
    ).reset_index()
    aggregate = per_seed.groupby(keys, dropna=False).agg(
        action_mse=("action_mse", "mean"), action_mae=("action_mae", "mean"),
        mse_ratio=("mse_ratio", "mean"), training_seed_sd=("mse_ratio", "std"),
        mean_within_training_seed_damage_sd=("damage_seed_sd", "mean"),
        training_seed_count=("train_seed", "nunique"),
    ).reset_index()
    per_seed.to_csv(output / "per_training_seed.csv", index=False)
    aggregate.to_csv(output / "aggregate.csv", index=False)
    _severity_chart(aggregate, output)
    # Paired model comparison under exactly the same train/test condition and seeds.
    pair_keys = ["dataset", "train_condition", "train_seed", "group", "severity", "test_condition", "quality_mode", "damage_seed"]
    act = dirty[dirty.model == "act_lite"][[*pair_keys, "action_mse"]].rename(columns={"action_mse": "act_mse"})
    quality = dirty[dirty.model == "quality_full"][[*pair_keys, "action_mse"]].rename(columns={"action_mse": "quality_mse"})
    paired = act.merge(quality, on=pair_keys, validate="one_to_one")
    paired["quality_to_act_mse"] = paired.quality_mse / paired.act_mse.where(paired.act_mse > 0)
    paired.to_csv(output / "quality_vs_act.csv", index=False)
    model_comparison = paired.groupby(["dataset", "train_condition", "severity", "test_condition", "quality_mode"])["quality_to_act_mse"].mean().reset_index()
    # Same checkpoint: only inference-time metadata changes.
    metadata_keys = ["dataset", "model", "train_condition", "train_seed", "severity", "test_condition", "damage_seed"]
    oracle = dirty[dirty.quality_mode == "oracle"][[*metadata_keys, "action_mse"]].rename(columns={"action_mse": "oracle_mse"})
    unknown = dirty[dirty.quality_mode == "unknown"][[*metadata_keys, "action_mse"]].rename(columns={"action_mse": "unknown_mse"})
    metadata = oracle.merge(unknown, on=metadata_keys, validate="one_to_one")
    metadata["unknown_to_oracle_mse"] = metadata.unknown_mse / metadata.oracle_mse.where(metadata.oracle_mse > 0)
    metadata.to_csv(output / "metadata_comparison.csv", index=False)
    metadata_summary = metadata.groupby(["dataset", "model", "train_condition", "test_condition"])["unknown_to_oracle_mse"].mean().reset_index()
    chart = aggregate[(aggregate.group == "main") & (aggregate.model.isin(["act_lite", "quality_full"]))]
    bars = []
    for _, row in chart.iterrows():
        value = row.mse_ratio
        if pd.notna(value):
            label = f"{row['dataset']} / {row['model']} / {row['train_condition']} / {row['test_condition']}"
            bars.append(f"<div class='bar'><span>{html.escape(label)}</span><meter min='0' max='{max(2, chart.mse_ratio.max()):.5g}' value='{value:.5g}'></meter><b>{value:.3f}×</b></div>")
    tables = [aggregate[aggregate.group == "main"], model_comparison, metadata_summary, aggregate[aggregate.group == "severity"]]
    for language in ("zh", "en"):
        if language == "zh":
            title = "Sync2Act：受损测试观测评估"
            intro = "固定历史模型权重，对同一批留出测试轨迹注入图像或状态损坏，保持原始动作标签和训练归一化统计量不变。延迟只读取过去帧；缺失观测置零。故障分配到互不重叠的 16 帧片段，实际帧比例记录在各评估清单中。"
            boundary = "仅离线预测评估，不代表机器人任务成功率。Oracle 使用已知合成损坏的质量信息；unknown 将观测质量元数据设为正常，不修复输入。历史模型没有按新训练协议重训。"
            protocol = "中等强度：图像/状态条件为 50% 干净、40% 延迟 2 帧、10% 缺失；混合条件为 20% 干净、30% 图像延迟、30% 状态延迟、10% 图像缺失、10% 状态缺失。轻/重强度同时改变延迟和受损比例，不是单变量敏感性实验。"
            aggregation = "先在每个训练种子内平均测试损坏种子，再平均训练种子；分别报告训练种子标准差和训练种子内损坏种子标准差的均值，单个种子无法估计标准差。MSE 比率以同一 checkpoint 的干净测试为分母。不同数据集的绝对 MSE 不直接合并。所有训练消融组在 Oracle 测试中均获得真实观测质量信息。"
            sections = ["主要实验", "完整质量模型 / ACT-Lite（同条件配对 MSE 比率）", "未知 / 已知质量信息（同 checkpoint 配对 MSE 比率）", "损坏强度曲线数据"]
            headings = ["相对干净测试的误差倍率", "可复现记录"]
            count = f"完成 {summary['completed_evaluations']} / {summary['expected_evaluations']} 次评估；{summary['baseline_audits_passed']} 个干净基线通过历史结果核对。"
            pilot = "这是小规模流程验证，不能作为完整研究结论。" if summary["pilot"] else "完整预定实验矩阵已完成。"
        else:
            title = "Sync2Act: corrupted test observation evaluation"
            intro = "Fixed historical model weights are evaluated on the same held-out trajectories with image/state corruption. Original action labels and frozen training normalization remain unchanged. Delays read past frames only; missing observations are zeroed. Faults occupy disjoint 16-frame segments; manifests record actual frame proportions."
            boundary = "Offline prediction only, not robot task success. Oracle uses known synthetic-corruption metadata; unknown sets observation metadata to normal without repairing inputs. Historical models were not retrained under the new training protocol."
            protocol = "Medium severity: image/state conditions contain 50% clean, 40% two-frame delay, and 10% missing segments. Mixed: 20% clean, 30% image delay, 30% state delay, 10% missing image, 10% missing state. Light/heavy jointly vary delay and damaged fraction; this is not a single-factor sensitivity study."
            aggregation = "Average damage seeds within each training seed, then average training seeds. Report between-training-seed SD and mean within-training-seed damage SD separately; SD is undefined for a single seed. MSE ratios use the same checkpoint's clean-test denominator. Raw MSE is never pooled across datasets. Every training ablation receives true observation metadata in Oracle tests."
            sections = ["Main study", "Full quality model / ACT-Lite (paired same-condition MSE ratio)", "Unknown / Oracle metadata (paired same-checkpoint MSE ratio)", "Severity curve data"]
            headings = ["MSE relative to clean test", "Reproducibility"]
            count = f"Completed {summary['completed_evaluations']} / {summary['expected_evaluations']} evaluations; {summary['baseline_audits_passed']} clean baselines matched historical results."
            pilot = "Small pilot only; not a complete study conclusion." if summary["pilot"] else "The complete prespecified experiment matrix has finished."
        sections_html = "".join(f"<details><summary>{heading}</summary><div class='table'>{table.to_html(index=False, float_format=lambda x: f'{x:.6g}', na_rep='—')}</div></details>" for heading, table in zip(sections, tables, strict=True))
        reduced = {k: v for k, v in summary.items() if k != "record_paths"}
        document = f"""<!doctype html><html lang='{language}'><meta charset='utf-8'><title>{title}</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;color:#17233f}}h1,h2{{color:#3157d5}}.notice{{background:#fff4cf;padding:16px;border-radius:8px}}.table{{overflow-x:auto}}table{{border-collapse:collapse;font-size:12px;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dde3ef;white-space:nowrap;text-align:left}}th{{background:#eef2ff}}.bar{{display:grid;grid-template-columns:minmax(250px,3fr) 2fr 70px;gap:12px;margin:6px 0;font-size:12px}}meter{{width:100%;height:22px}}pre{{overflow:auto;background:#f3f6fb;padding:18px}}summary{{cursor:pointer;font-size:20px;color:#3157d5;padding:16px 0}}img{{width:100%;height:auto}}</style>
<h1>{title}</h1><p>{count} {pilot}</p><p>{intro}</p><p class='notice'>{boundary}</p><p>{protocol}</p><p>{aggregation}</p>
<p><a href='results.csv'>results.csv</a> · <a href='aggregate.csv'>aggregate.csv</a> · <a href='per_training_seed.csv'>per_training_seed.csv</a> · <a href='quality_vs_act.csv'>quality_vs_act.csv</a> · <a href='metadata_comparison.csv'>metadata_comparison.csv</a> · <a href='summary.json'>summary.json</a></p>
<img src='severity_curves.svg' alt='MSE ratio across light, medium and heavy test corruption'>
<details><summary>{headings[0]}</summary>{''.join(bars)}</details>{sections_html}<details><summary>{headings[1]}</summary><pre>{html.escape(json.dumps(reduced, indent=2))}</pre></details></html>"""
        (output / ("report.html" if language == "zh" else "report_en.html")).write_text(document, encoding="utf-8")
