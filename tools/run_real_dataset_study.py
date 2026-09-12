"""Run a resumable real-robot dataset-corruption study with Sync2Act."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch

from sync2act.corruptions import build_mixed_quality_dataset, transform_quality_annotations
from sync2act.data.dataset import NormalizationStats, compute_stats
from sync2act.data.huggingface import download_dataset_snapshot
from sync2act.data.lerobot import load_local_lerobot_dataset
from sync2act.data.split import split_episode_indices
from sync2act.evaluation import evaluate_policy
from sync2act.paths import datasets_root
from sync2act.policies import build_policy
from sync2act.training import QUALITY_SCHEMA_VERSION, inspect_checkpoint, train_policy

DATASETS = [
    {
        "repo_id": "lerobot/xarm_lift_medium",
        "revision": "79efb0e3cef0e530ddec4b8569b190966ab45808",
        "name": "xarm_lift_medium",
        "robot": "xArm",
        "task": "Lift",
        "camera_count": 1,
        "source_episodes": 800,
        "source_frames": 20_000,
    },
    {
        "repo_id": "lerobot/koch_pick_place_1_lego_raph",
        "revision": "1e7afb6601bb4d2405962d27d58109dcfaddc096",
        "name": "koch_pick_place_1_lego",
        "robot": "Koch",
        "task": "Pick and place one Lego piece",
        "camera_count": 2,
        "source_episodes": 100,
        "source_frames": 32_951,
    },
    {
        "repo_id": "lerobot/aloha_static_battery",
        "revision": "06dc3da83c4fd3d1889b00f1dfd3780da8421f64",
        "name": "aloha_static_battery",
        "robot": "ALOHA",
        "task": "Static battery manipulation",
        "camera_count": 4,
        "source_episodes": 49,
        "source_frames": 29_400,
    },
]

ABLATIONS = {
    "act_lite": {
        "policy": "act_lite",
        "quality_features": False,
        "weighted_loss": False,
        "quality_transform": "identity",
    },
    "quality_input": {
        "policy": "quality_input",
        "quality_features": True,
        "weighted_loss": False,
        "quality_transform": "identity",
    },
    "quality_weighted_loss": {
        "policy": "quality_weighted_loss",
        "quality_features": False,
        "weighted_loss": True,
        "quality_transform": "identity",
    },
    "quality_full": {
        "policy": "quality_full",
        "quality_features": True,
        "weighted_loss": True,
        "quality_transform": "identity",
    },
    "quality_shuffled": {
        "policy": "quality_shuffled",
        "quality_features": True,
        "weighted_loss": True,
        "quality_transform": "shuffled",
    },
    "quality_constant": {
        "policy": "quality_constant",
        "quality_features": True,
        "weighted_loss": True,
        "quality_transform": "constant",
    },
}
MODELS = list(ABLATIONS)
COLORS = {
    "act_lite": "#3157d5",
    "quality_input": "#8f5bd7",
    "quality_weighted_loss": "#e66b45",
    "quality_full": "#159570",
    "quality_shuffled": "#c28a13",
    "quality_constant": "#667085",
}
CONDITION_ORDER = [
    "clean",
    "mixed_temporal_state_shift2",
    "mixed_temporal_action_shift2",
    "mixed_action_delay2",
    "mixed_three_corruptions",
]

DEFAULT_EPISODES = 12
DEFAULT_FRAMES = 128
DEFAULT_IMAGE_SIZE = 64
DEFAULT_EPOCHS = 5
DEFAULT_SEEDS = [7]
DEFAULT_SPLIT_SEED = 7
DEFAULT_BC_BATCH_SIZE = 64
DEFAULT_ACT_BATCH_SIZE = 32
DEFAULT_SEGMENT_LENGTH = 16


def _read_saved_study(output_root: Path) -> dict:
    path = output_root / "study.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolved_option(value, saved: dict, key: str, default):
    if value is not None:
        return value
    if key in saved:
        return saved[key]
    return default


def _data_scales(episodes: list[dict]) -> tuple[float, float, float]:
    actions = torch.cat([episode["action"] for episode in episodes])
    states = torch.cat([episode["observation.state"] for episode in episodes])
    action_scale = float(actions.std(dim=0, unbiased=False).mean().clamp_min(1e-4))
    state_scale = float(states.std(dim=0, unbiased=False).mean().clamp_min(1e-4))
    return action_scale, state_scale, float(states[:, 0].mean())


def corruption_conditions(episodes: list[dict], fps: float) -> list[dict]:
    """Focused mixed-quality conditions for the six-way quality ablation."""
    del episodes, fps
    clean = {"name": "clean", "weight": 0.5, "config": None}
    state_shift = {
        "name": "temporal_state_shift2",
        "weight": 0.5,
        "config": {
            "type": "temporal_shift",
            "target": "state",
            "shift": 2,
            "boundary": "mask",
        },
    }
    action_shift = {
        "name": "temporal_action_shift2",
        "weight": 0.5,
        "config": {
            "type": "temporal_shift",
            "target": "action",
            "shift": 2,
            "boundary": "mask",
        },
    }
    action_delay = {
        "name": "action_delay2",
        "weight": 0.5,
        "config": {
            "type": "action_noise",
            "mode": "delay",
            "delay": 2,
            "random_delay": False,
        },
    }
    return [
        {"name": "clean", "family": "clean", "components": None},
        {
            "name": "mixed_temporal_state_shift2",
            "family": "temporal_shift",
            "components": [clean, state_shift],
        },
        {
            "name": "mixed_temporal_action_shift2",
            "family": "temporal_shift",
            "components": [clean, action_shift],
        },
        {
            "name": "mixed_action_delay2",
            "family": "action_noise",
            "components": [clean, action_delay],
        },
        {
            "name": "mixed_three_corruptions",
            "family": "mixed",
            "components": [
                {**clean, "weight": 0.25},
                {**state_shift, "weight": 0.25},
                {**action_shift, "weight": 0.25},
                {**action_delay, "weight": 0.25},
            ],
        },
    ]


def model_config(name: str, state_dim: int, action_dim: int) -> dict:
    ablation = ABLATIONS[name]
    common = {
        "name": ablation["policy"],
        "state_dim": state_dim,
        "action_dim": action_dim,
    }
    return {
        **common,
        "horizon": 8,
        "hidden_dim": 64,
        "num_layers": 1,
        "num_heads": 4,
        "dropout": 0.0,
        "use_quality_features": ablation["quality_features"],
    }


def training_config(
    name: str,
    seed: int,
    epochs: int,
    device: str,
    model: dict,
    bc_batch_size: int = 64,
    act_batch_size: int = 32,
) -> dict:
    return {
        "epochs": epochs,
        "batch_size": act_batch_size,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "loss": "smooth_l1",
        "seed": seed,
        "grad_clip": 1.0,
        "device": device,
        "horizon": 8,
        "quality_weighted_loss": ABLATIONS[name]["weighted_loss"],
        "quality_transform": ABLATIONS[name]["quality_transform"],
        "lambda_smooth": 0.01,
        "model": model,
    }


def _flatten_runs(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        metrics = run["metrics"]
        rows.append(
            {
                "dataset": run["dataset"],
                "model": run["model"],
                "condition": run["condition"],
                "family": run["family"],
                "seed": run["seed"],
                "action_mse": metrics["action_mse"],
                "action_mae": metrics["action_mae"],
                "trajectory_smoothness": metrics["trajectory_smoothness"],
                "jerk": metrics["jerk"],
                "latency_p50_ms": metrics["latency_p50_ms"],
                "latency_p95_ms": metrics["latency_p95_ms"],
                "parameter_count": metrics["parameter_count"],
                "training_seconds": metrics["training_seconds"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    clean = frame[frame["condition"] == "clean"].set_index(["dataset", "model", "seed"])
    frame["mse_ratio_to_clean"] = [
        row.action_mse / clean.loc[(row.dataset, row.model, row.seed), "action_mse"]
        if (row.dataset, row.model, row.seed) in clean.index
        else math.nan
        for row in frame.itertuples()
    ]
    frame["mae_ratio_to_clean"] = [
        row.action_mae / clean.loc[(row.dataset, row.model, row.seed), "action_mae"]
        if (row.dataset, row.model, row.seed) in clean.index
        else math.nan
        for row in frame.itertuples()
    ]
    return frame


def _impact_svg(frame: pd.DataFrame, condition_order: list[str], dataset: str | None = None) -> str:
    width, height = 1280, 480
    left, top, bottom, right = 80, 35, 160, 160
    chart_w, chart_h = width - left - right, height - top - bottom
    plot_frame = frame if dataset is None else frame[frame["dataset"] == dataset]
    aggregate = (
        plot_frame.groupby(["condition", "model"], as_index=False)["mse_ratio_to_clean"].mean()
        if not plot_frame.empty and "mse_ratio_to_clean" in plot_frame
        else pd.DataFrame()
    )
    finite = aggregate["mse_ratio_to_clean"].dropna().tolist() if not aggregate.empty else []
    maximum = max([1.0, *finite]) * 1.08
    x_step = chart_w / max(1, len(condition_order) - 1)
    pieces = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='MSE ratio to clean'>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + chart_h}' stroke='#667085'/>",
        f"<line x1='{left}' y1='{top + chart_h}' x2='{left + chart_w}' y2='{top + chart_h}' stroke='#667085'/>",
    ]
    for tick in range(5):
        value = maximum * tick / 4
        y = top + chart_h - chart_h * value / maximum
        pieces.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{left + chart_w}' y2='{y:.1f}' stroke='#e4e7ec'/>"
        )
        pieces.append(
            f"<text x='{left - 8}' y='{y + 4:.1f}' text-anchor='end' font-size='11'>{value:.2f}×</text>"
        )
    for index, condition in enumerate(condition_order):
        x = left + index * x_step
        pieces.append(
            f"<text transform='translate({x:.1f},{top + chart_h + 12}) rotate(55)' "
            f"font-size='10'>{html.escape(condition)}</text>"
        )
    for legend_index, model in enumerate(MODELS):
        points = []
        subset = aggregate[aggregate["model"] == model] if not aggregate.empty else aggregate
        values = dict(
            zip(subset.get("condition", []), subset.get("mse_ratio_to_clean", []), strict=False)
        )
        for index, condition in enumerate(condition_order):
            value = values.get(condition)
            if value is None or pd.isna(value):
                continue
            x = left + index * x_step
            y = top + chart_h - chart_h * float(value) / maximum
            points.append((x, y))
        if points:
            joined = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            pieces.append(
                f"<polyline points='{joined}' fill='none' stroke='{COLORS[model]}' stroke-width='3'/>"
            )
            pieces.extend(
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{COLORS[model]}'/>"
                for x, y in points
            )
        legend_x = left + 170 * legend_index
        pieces.append(f"<rect x='{legend_x}' y='6' width='16' height='4' fill='{COLORS[model]}'/>")
        pieces.append(f"<text x='{legend_x + 22}' y='14' font-size='12'>{model}</text>")
    pieces.append("</svg>")
    return "".join(pieces)


def write_report(runs: list[dict], study: dict, output_root: Path) -> Path:
    frame = _flatten_runs(runs)
    frame.to_csv(output_root / "results.csv", index=False)
    (output_root / "results.json").write_text(
        json.dumps({"study": study, "runs": runs}, indent=2), encoding="utf-8"
    )
    present_conditions = {run["condition"] for run in runs}
    condition_order = [name for name in CONDITION_ORDER if name in present_conditions]
    seeds = study.get("seeds", [study.get("seed", 7)])
    dataset_specs = study.get("datasets", DATASETS)
    model_names = study.get("models", MODELS)
    configured_conditions = study.get("conditions", CONDITION_ORDER)
    expected = len(dataset_specs) * len(model_names) * len(configured_conditions) * len(seeds)
    status = (
        f"已完成 {len(runs)} / {expected} 个训练组合"
        if len(runs) < expected
        else f"全部完成：{expected} / {expected} 个训练组合"
    )
    dataset_rows_parts = []
    for item in dataset_specs:
        manifest_path = output_root / item["name"] / "dataset_manifest.json"
        loaded = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            loaded = manifest.get("loaded_metadata", {})
        selected_episodes = loaded.get("episodes", study.get("episode_limit", "—"))
        selected_frames = loaded.get("frames", "—")
        selected_frames_text = (
            f"{selected_frames:,}"
            if isinstance(selected_frames, (int, float))
            else str(selected_frames)
        )
        camera_count = loaded.get("camera_count", item["camera_count"])
        dataset_rows_parts.append(
            "<tr>"
            f"<td><a href='https://huggingface.co/datasets/{html.escape(item['repo_id'])}'>"
            f"{html.escape(item['repo_id'])}</a></td><td>{html.escape(item['robot'])}</td>"
            f"<td>{html.escape(item['task'])}</td><td>{camera_count}</td>"
            f"<td>{item['source_episodes']:,}</td><td>{item['source_frames']:,}</td>"
            f"<td>{selected_episodes}</td><td>{selected_frames_text}</td></tr>"
        )
    dataset_rows = "".join(dataset_rows_parts)
    result_rows = ""
    summary_rows = ""
    if not frame.empty:
        for row in frame.sort_values(["dataset", "model", "condition", "seed"]).itertuples():
            ratio = (
                "not available"
                if pd.isna(row.mse_ratio_to_clean)
                else f"{row.mse_ratio_to_clean:.3f}×"
            )
            result_rows += (
                "<tr>"
                f"<td>{html.escape(row.dataset)}</td><td>{html.escape(row.model)}</td>"
                f"<td>{html.escape(row.condition)}</td><td>{row.seed}</td><td>{row.action_mse:.6g}</td>"
                f"<td>{row.action_mae:.6g}</td><td>{ratio}</td>"
                f"<td>{row.training_seconds:.2f}</td></tr>"
            )
        summary = frame.groupby(["dataset", "model", "condition"], as_index=False).agg(
            seeds=("seed", "nunique"),
            action_mse_mean=("action_mse", "mean"),
            action_mse_std=("action_mse", "std"),
            ratio_mean=("mse_ratio_to_clean", "mean"),
            ratio_std=("mse_ratio_to_clean", "std"),
        )
        for row in summary.sort_values(["dataset", "model", "condition"]).itertuples():
            mse_std = 0.0 if pd.isna(row.action_mse_std) else row.action_mse_std
            ratio_std = 0.0 if pd.isna(row.ratio_std) else row.ratio_std
            summary_rows += (
                "<tr>"
                f"<td>{html.escape(row.dataset)}</td><td>{html.escape(row.model)}</td>"
                f"<td>{html.escape(row.condition)}</td><td>{row.seeds}</td>"
                f"<td>{row.action_mse_mean:.6g} ± {mse_std:.3g}</td>"
                f"<td>{row.ratio_mean:.3f} ± {ratio_std:.3f}×</td></tr>"
            )
    findings = []
    clean_expected = len(dataset_specs) * len(model_names) * len(seeds)
    if not frame.empty and len(frame[frame["condition"] == "clean"]) == clean_expected:
        corrupted = frame[frame["condition"] != "clean"]
        for model in model_names:
            local = corrupted[corrupted["model"] == model]
            if not local.empty:
                means = (
                    local.groupby("condition")["mse_ratio_to_clean"]
                    .mean()
                    .sort_values(ascending=False)
                )
                overall = local["mse_ratio_to_clean"].mean()
                findings.append(
                    f"{model}：跨三个数据集平均后，相对 MSE 最高的损坏为 "
                    f"{means.index[0]}（干净基线的 {means.iloc[0]:.3f}×）；"
                    f"所有损坏条件的总体均值为 {overall:.3f}×。"
                )
        act = corrupted[corrupted["model"] == "act_lite"]
        quality = corrupted[corrupted["model"] == "quality_full"]
        paired = act.merge(
            quality, on=["dataset", "condition", "seed"], suffixes=("_act", "_quality")
        )
        if not paired.empty:
            wins = int((paired["action_mse_quality"] < paired["action_mse_act"]).sum())
            ratio_wins = int(
                (paired["mse_ratio_to_clean_quality"] < paired["mse_ratio_to_clean_act"]).sum()
            )
            findings.append(
                f"完整 Quality-Aware ACT 在 {wins}/{len(paired)} 个与 ACT-Lite 成对的损坏实验中"
                f"取得更低的绝对 MSE，但按各自同 seed 干净基线归一化后仅在 "
                f"{ratio_wins}/{len(paired)} 个实验中更稳健，因此不能宣称它对所有损坏都更好。"
            )
        full = corrupted[corrupted["model"] == "quality_full"]
        for control_name, control_label in (
            ("quality_shuffled", "打乱 quality"),
            ("quality_constant", "恒定 quality"),
        ):
            control = corrupted[corrupted["model"] == control_name]
            comparison = full.merge(
                control,
                on=["dataset", "condition", "seed"],
                suffixes=("_full", "_control"),
            )
            if not comparison.empty:
                wins = int((comparison["action_mse_full"] < comparison["action_mse_control"]).sum())
                findings.append(
                    f"完整 Quality-Aware ACT 相对{control_label}对照在 "
                    f"{wins}/{len(comparison)} 个成对损坏实验中取得更低 MSE。"
                )
        consistent = (
            corrupted.assign(degraded=corrupted["mse_ratio_to_clean"] > 1.0)
            .groupby(["model", "condition"])["degraded"]
            .agg(["sum", "count"])
        )
        for model in model_names:
            names = [
                condition
                for model_name, condition in consistent.index
                if model_name == model
                and consistent.loc[(model_name, condition), "sum"]
                == consistent.loc[(model_name, condition), "count"]
            ]
            findings.append(
                f"{model}：在全部 3 数据集 × 3 seeds 上都恶化的条件为 "
                f"{', '.join(names) if names else '无'}。"
            )
        training_hours = frame["training_seconds"].sum() / 3600
        findings.append(f"累计 GPU 模型训练时间为 {training_hours:.2f} 小时。")
    findings_html = (
        "".join(f"<li>{html.escape(item)}</li>" for item in findings) or "<li>结果尚未完整。</li>"
    )
    dataset_charts = "".join(
        f"<h3>{html.escape(item['name'])}</h3>{_impact_svg(frame, condition_order, item['name'])}"
        for item in dataset_specs
    )
    full_dataset = (
        study.get("episode_limit") is None and study.get("frames_per_episode_limit") is None
    )
    scope_label = "完整数据集" if full_dataset else "受控子集"
    seed_text = ", ".join(str(seed) for seed in seeds)
    validation_percent = 100 * float(study.get("validation_split", 1 / 6))
    test_percent = 100 * float(study.get("test_split", 1 / 6))
    train_percent = 100 - validation_percent - test_percent
    protocol_text = (
        f"每个数据集使用{scope_label}，按 Episode 固定划分为 {train_percent:.0f}% 训练、"
        f"{validation_percent:.0f}% 验证和 {test_percent:.0f}% 测试。仅训练 Episodes "
        f"按实验条件构造混合质量片段；验证集和测试集始终保持干净。六组消融均训练 "
        f"{study.get('epochs')} epochs，图像长边缩放至 {study.get('max_image_size')} px，"
        f"在 {str(study.get('device')).upper()} 上运行。独立随机种子为 {seed_text}。"
    )
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Sync2Act Quality-Aware ACT 六组消融研究</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:1280px;margin:36px auto;color:#182230;line-height:1.45}}
h1,h2{{color:#2346a0}}.notice{{background:#fff6d6;border-left:4px solid #d69e00;padding:14px}}
.ok{{background:#eaf8f2;border-left:4px solid #159570;padding:14px}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:8px;border-bottom:1px solid #d9dee8;text-align:left}}th{{background:#eef2ff;position:sticky;top:0}}
.scroll{{max-height:640px;overflow:auto;border:1px solid #d9dee8}}svg{{width:100%;height:auto;background:#fff}}
code,pre{{background:#f5f7fb;padding:2px 5px}}pre{{padding:12px;overflow:auto}}
</style></head><body><h1>Sync2Act Quality-Aware ACT 六组消融研究</h1>
<p class='ok'>{html.escape(status)} · 生成时间 {html.escape(datetime.now(UTC).isoformat())}</p>
<p class='notice'><strong>研究范围：</strong>这是{scope_label}上的离线模仿学习评估，不是机器人闭环 rollout 成功率。本轮使用 {len(seeds)} 个随机种子。不同数据集的 action 单位不同，因此跨数据集主要比较同一数据集、同一模型相对同 seed 干净训练基线的 MSE 比值。</p>
<h2>数据集</h2><table><tr><th>Hugging Face 仓库</th><th>机器人</th><th>单一任务</th><th>相机数</th><th>源 Episodes</th><th>源 Frames</th><th>选用 Episodes</th><th>选用 Frames</th></tr>{dataset_rows}</table>
<h2>实验协议</h2><p>{html.escape(protocol_text)}</p><details><summary>展开查看完整可复现配置</summary><pre>{html.escape(json.dumps(study, indent=2))}</pre></details>
<h2>消融与损坏设计</h2><p>六组消融分别为 ACT-Lite、仅 observation quality 输入、仅 action-label quality 加权、完整 Quality-Aware ACT、打乱 quality 对照和恒定 quality 对照。重点条件为 temporal state shift、temporal action shift、action delay，以及三者混合；每种单项损坏训练集按片段混合 50% 干净与 50% 损坏数据，综合条件各占 25%。归一化统计量只由干净 Training Episodes 计算并在所有组合间固定。</p>
<h2>损坏影响</h2><p>图中是归一化测试 MSE；1.0× 表示同一数据集和模型的干净训练基线。下图先对三个数据集取平均，后面可展开查看每个数据集。</p>{_impact_svg(frame, condition_order)}<details><summary>展开查看三个数据集的独立曲线</summary>{dataset_charts}</details>
<h2>自动汇总结论</h2><ul>{findings_html}</ul>
<p class='notice'><strong>解读提醒：</strong>某些损坏条件的 MSE 可能低于 1.0×，这可能来自随机波动、优化路径差异或类似正则化的效果；应结合三个 seed 的均值和标准差解读，不能据此宣称损坏数据会提高真实机器人性能。</p>
<h2>跨 Seed 统计</h2><div class='scroll'><table><tr><th>数据集</th><th>模型</th><th>训练条件</th><th>Seeds</th><th>测试 MSE（均值 ± 标准差）</th><th>MSE / 同 seed 干净基线</th></tr>{summary_rows}</table></div>
<h2>全部实测结果</h2><div class='scroll'><table><tr><th>数据集</th><th>模型</th><th>训练条件</th><th>Seed</th><th>测试 MSE</th><th>测试 MAE</th><th>MSE / 同 seed 干净基线</th><th>训练秒数</th></tr>{result_rows}</table></div>
<h2>解读限制</h2><ul><li>验证和测试 Episodes 保持干净，仅训练 Episodes 含混合质量片段。</li><li>结果衡量 action 的离线预测误差，不是闭环任务成功率。</li><li>三个 seed 用于估计随机波动；更强统计结论仍需要更多 seeds 与置信区间。</li><li>图像会统一缩放，以控制显存和训练时间。</li><li>本轮使用由人工损坏过程给出的 Oracle quality；若 Oracle 对照有效，才适合继续开发自动质量估计器。</li></ul>
</body></html>"""
    target = output_root / "report.html"
    target.write_text(document, encoding="utf-8")
    return target


def _download_progress(dataset_name: str):
    last = {"percent": -10}

    def callback(event: dict) -> None:
        percent = int(event.get("percent", 0))
        if percent >= last["percent"] + 10 or percent == 100:
            print(
                f"[{dataset_name}] download {percent}%: {event.get('description', '')}", flush=True
            )
            last["percent"] = percent

    return callback


def _load_progress(dataset_name: str):
    last = {"percent": -10}

    def callback(event: dict) -> None:
        percent = int(event.get("percent", 0))
        if percent >= last["percent"] + 10 or percent == 100:
            print(f"[{dataset_name}] load {percent}%: {event.get('description', '')}", flush=True)
            last["percent"] = percent

    return callback


def validate_single_task_dataset(root: Path) -> dict:
    """Reject a dataset unless both metadata sources describe exactly one task."""
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.parquet"
    if not info_path.is_file() or not tasks_path.is_file():
        raise ValueError(f"Missing LeRobot task metadata under {root / 'meta'}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    declared = int(info.get("total_tasks", -1))
    tasks = pd.read_parquet(tasks_path)
    if "task_index" not in tasks:
        raise ValueError(f"{tasks_path} does not contain task_index")
    indices = sorted(int(value) for value in tasks["task_index"].unique())
    if declared != 1 or len(indices) != 1:
        raise ValueError(
            f"Dataset must be single-task, but total_tasks={declared} and task_index={indices}"
        )
    labels = tasks["task"].dropna().astype(str).unique().tolist() if "task" in tasks else []
    return {"total_tasks": declared, "task_indices": indices, "task_labels": labels}


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src" / "sync2act").rglob("*.py")) + [Path(__file__).resolve()]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _experiment_signature(
    dataset_spec: dict,
    model_name: str,
    model: dict,
    training: dict,
    condition: dict,
    seed: int,
    split: dict,
    source_fingerprint: str,
    clean_stats: NormalizationStats,
) -> str:
    return _canonical_hash(
        {
            "experiment_schema_version": 2,
            "quality_schema_version": QUALITY_SCHEMA_VERSION,
            "source_fingerprint": source_fingerprint,
            "dataset_repo": dataset_spec["repo_id"],
            "dataset_revision": dataset_spec["revision"],
            "model_name": model_name,
            "model_config": model,
            "training_config": training,
            "condition": condition,
            "seed": seed,
            "split": split,
            "normalization_stats": clean_stats.to_dict(),
        }
    )


def _completed_run(
    run_path: Path,
    model: torch.nn.Module,
    experiment_signature: str,
) -> tuple[dict | None, str]:
    if not run_path.is_file():
        return None, "run metadata is missing"
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return None, f"run metadata cannot be read: {error}"
    if run.get("status") != "complete":
        return None, "run status is not complete"
    if run.get("experiment_signature") != experiment_signature:
        return None, "experiment signature changed"
    checkpoint_value = run.get("checkpoint")
    checkpoint = run_path.parent / str(checkpoint_value or "checkpoint.pt")
    compatible, reason, _ = inspect_checkpoint(
        checkpoint,
        expected_model=model,
        experiment_signature=experiment_signature,
    )
    if not compatible:
        return None, reason
    for artifact in ("predictions.csv", "per_episode.csv"):
        if not (run_path.parent / artifact).is_file():
            return None, f"{artifact} is missing"
    return run, "complete and compatible"


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _load_compatible_runs(
    output_root: Path, expected_source: str | None = None
) -> tuple[list[dict], list[str]]:
    runs: list[dict] = []
    rejected: list[str] = []
    expected_source = expected_source or _source_fingerprint()
    for run_path in sorted(output_root.rglob("run.json")):
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
            if run.get("source_fingerprint") != expected_source:
                rejected.append(f"{run_path}: source fingerprint changed")
                continue
            model = build_policy(run["model_config"])
            completed, reason = _completed_run(
                run_path, model, str(run.get("experiment_signature", ""))
            )
            if completed is None:
                rejected.append(f"{run_path}: {reason}")
            else:
                runs.append(completed)
        except (KeyError, OSError, ValueError) as error:
            rejected.append(f"{run_path}: {error}")
    return runs, rejected


def run_study(args: argparse.Namespace) -> Path:
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    saved_study = _read_saved_study(output_root) if args.report_only else {}
    saved_dataset_names = [
        item.get("name")
        for item in saved_study.get("datasets", [])
        if isinstance(item, dict) and item.get("name")
    ]
    selected_dataset_names = args.datasets or saved_dataset_names or [
        item["name"] for item in DATASETS
    ]
    selected_model_names = args.models or saved_study.get("models") or MODELS
    selected_condition_names = (
        args.conditions or saved_study.get("conditions") or CONDITION_ORDER
    )
    seeds = list(
        dict.fromkeys(
            _resolved_option(args.seeds, saved_study, "seeds", DEFAULT_SEEDS)
        )
    )
    if len(seeds) < 1:
        raise ValueError("At least one seed is required")
    device = _resolved_option(
        args.device,
        saved_study,
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    saved_episode_limit = saved_study.get("episode_limit", DEFAULT_EPISODES)
    saved_frame_limit = saved_study.get("frames_per_episode_limit", DEFAULT_FRAMES)
    episode_limit = (
        saved_episode_limit
        if args.episodes is None
        else (None if args.episodes <= 0 else args.episodes)
    )
    frame_limit = (
        saved_frame_limit
        if args.frames is None
        else (None if args.frames <= 0 else args.frames)
    )
    image_size = int(
        _resolved_option(
            args.image_size, saved_study, "max_image_size", DEFAULT_IMAGE_SIZE
        )
    )
    epochs = int(_resolved_option(args.epochs, saved_study, "epochs", DEFAULT_EPOCHS))
    split_seed = int(
        _resolved_option(
            args.split_seed, saved_study, "split_seed", DEFAULT_SPLIT_SEED
        )
    )
    bc_batch_size = int(
        _resolved_option(
            args.bc_batch_size,
            saved_study,
            "bc_batch_size",
            DEFAULT_BC_BATCH_SIZE,
        )
    )
    act_batch_size = int(
        _resolved_option(
            args.act_batch_size,
            saved_study,
            "act_batch_size",
            DEFAULT_ACT_BATCH_SIZE,
        )
    )
    segment_length = int(
        _resolved_option(
            args.segment_length,
            saved_study,
            "mixture_segment_length",
            DEFAULT_SEGMENT_LENGTH,
        )
    )
    current_source_fingerprint = _source_fingerprint()
    source_fingerprint = (
        saved_study.get("source_fingerprint", current_source_fingerprint)
        if args.report_only
        else current_source_fingerprint
    )
    selected_datasets = [
        item for item in DATASETS if item["name"] in selected_dataset_names
    ]
    selected_models = [name for name in MODELS if name in selected_model_names]
    selected_conditions = [
        name for name in CONDITION_ORDER if name in selected_condition_names
    ]
    study = {
        "name": "quality_aware_act_mixed_quality_ablation",
        "experiment_schema_version": 2,
        "quality_schema_version": QUALITY_SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint,
        "git_commit": (
            saved_study.get("git_commit", _git_commit())
            if args.report_only
            else _git_commit()
        ),
        "started_at": saved_study.get("started_at", datetime.now(UTC).isoformat()),
        "datasets": selected_datasets,
        "models": selected_models,
        "conditions": selected_conditions,
        "seeds": seeds,
        "split_seed": split_seed,
        "episode_limit": episode_limit,
        "frames_per_episode_limit": frame_limit,
        "max_image_size": image_size,
        "epochs": epochs,
        "validation_split": 0.1,
        "test_split": 0.1,
        "bc_batch_size": bc_batch_size,
        "act_batch_size": act_batch_size,
        "device": device,
        "test_condition": "clean held-out episodes",
        "normalization": "frozen clean-training-episode statistics",
        "mixture_segment_length": segment_length,
        "condition_count": len(selected_conditions),
    }
    runs: list[dict] = []
    if args.report_only:
        runs, rejected = _load_compatible_runs(output_root, source_fingerprint)
        selected_dataset_names = {item["name"] for item in selected_datasets}
        runs = [
            run
            for run in runs
            if run["dataset"] in selected_dataset_names
            and run["model"] in selected_models
            and run["condition"] in selected_conditions
            and run["seed"] in seeds
        ]
        if rejected:
            preview = "\n".join(rejected[:10])
            raise RuntimeError(
                f"Cannot generate a complete report: {len(rejected)} invalid runs\n{preview}"
            )
        expected = (
            len(selected_datasets) * len(selected_models) * len(selected_conditions) * len(seeds)
        )
        if len(runs) != expected:
            raise RuntimeError(f"Expected {expected} compatible runs, found {len(runs)}")
        study["completed_at"] = datetime.now(UTC).isoformat()
        _write_json_atomic(output_root / "study.json", study)
        return write_report(runs, study, output_root)
    if not args.defer_report:
        _write_json_atomic(output_root / "study.json", study)

    for dataset_spec in selected_datasets:
        target = args.datasets_root / dataset_spec["name"]
        if not args.skip_download:
            download_dataset_snapshot(
                dataset_spec["repo_id"],
                target,
                revision=dataset_spec["revision"],
                progress=_download_progress(dataset_spec["name"]),
            )
        task_metadata = validate_single_task_dataset(target)
        print(
            f"[{dataset_spec['name']}] verified single task: {task_metadata['task_labels']}",
            flush=True,
        )
        episodes, metadata = load_local_lerobot_dataset(
            target,
            progress=_load_progress(dataset_spec["name"]),
            max_image_size=image_size,
            episode_limit=episode_limit,
            frames_per_episode_limit=frame_limit,
        )
        if episode_limit is None and metadata["episodes"] != metadata["source_episodes"]:
            raise RuntimeError("Full-dataset run did not load every source episode")
        if frame_limit is None and metadata["frames"] != metadata["source_frames"]:
            raise RuntimeError("Full-dataset run did not load every source frame")
        split = split_episode_indices(len(episodes), 0.1, 0.1, split_seed)
        clean_train = [episodes[index] for index in split["train"]]
        clean_validation = [episodes[index] for index in split["validation"]]
        clean_test = [episodes[index] for index in split["test"]]
        clean_stats = compute_stats(clean_train)
        conditions = corruption_conditions(clean_train, float(metadata.get("fps") or 30))
        conditions = [item for item in conditions if item["name"] in selected_conditions]
        dataset_manifest = {
            **dataset_spec,
            "single_task_validation": task_metadata,
            "loaded_metadata": metadata,
            "split": split,
            "condition_configs": conditions,
            "normalization_stats": clean_stats.to_dict(),
        }
        dataset_dir = output_root / dataset_spec["name"]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset_manifest.json").write_text(
            json.dumps(dataset_manifest, indent=2), encoding="utf-8"
        )

        for name in selected_models:
            config = model_config(
                name,
                int(clean_train[0]["observation.state"].shape[1]),
                int(clean_train[0]["action"].shape[1]),
            )
            for condition in conditions:
                for seed in seeds:
                    run_dir = dataset_dir / name / condition["name"] / f"seed-{seed}"
                    run_path = run_dir / "run.json"
                    train_config = training_config(
                        name,
                        seed,
                        epochs,
                        device,
                        config,
                        bc_batch_size,
                        act_batch_size,
                    )
                    train_config["source_fingerprint"] = source_fingerprint
                    signature = _experiment_signature(
                        dataset_spec,
                        name,
                        config,
                        train_config,
                        condition,
                        seed,
                        split,
                        source_fingerprint,
                        clean_stats,
                    )
                    train_config["experiment_signature"] = signature
                    torch.manual_seed(seed)
                    model = build_policy(config)
                    completed, reason = _completed_run(run_path, model, signature)
                    if completed is not None:
                        runs.append(completed)
                        print(
                            f"skip completed {dataset_spec['name']}/{name}/"
                            f"{condition['name']}/seed-{seed}",
                            flush=True,
                        )
                        continue
                    if run_path.exists():
                        print(f"rerun invalid {run_path}: {reason}", flush=True)
                    checkpoint_path = run_dir / "checkpoint.pt"
                    resumable, resume_reason, _ = inspect_checkpoint(
                        checkpoint_path,
                        expected_model=model,
                        experiment_signature=signature,
                    )
                    resume_from = checkpoint_path if resumable else None
                    if resume_from is not None:
                        print(f"resume compatible checkpoint {checkpoint_path}", flush=True)
                    elif checkpoint_path.exists():
                        print(
                            f"ignore incompatible checkpoint {checkpoint_path}: {resume_reason}",
                            flush=True,
                        )
                    components = condition["components"]
                    if components is None:
                        mixed_episodes = transform_quality_annotations(
                            clean_train, "identity", seed=seed
                        )
                        mixture_manifest = {
                            "schema_version": 1,
                            "type": "clean",
                            "frame_count": sum(len(item["timestamp"]) for item in clean_train),
                        }
                    else:
                        mixed_episodes, mixture_manifest = build_mixed_quality_dataset(
                            clean_train,
                            components,
                            segment_length=segment_length,
                            seed=seed,
                        )
                    training_episodes = transform_quality_annotations(
                        mixed_episodes,
                        ABLATIONS[name]["quality_transform"],
                        seed=seed + 7919,
                    )
                    started = time.perf_counter()
                    result = train_policy(
                        model,
                        training_episodes,
                        train_config,
                        run_dir,
                        resume_from=resume_from,
                        validation_episodes=clean_validation,
                        normalization_stats=clean_stats,
                    )
                    payload = torch.load(result.checkpoint, map_location="cpu", weights_only=False)
                    stats = NormalizationStats.from_dict(payload["stats"])
                    metrics = evaluate_policy(
                        model,
                        clean_test,
                        device=device,
                        batch_size=train_config["batch_size"],
                        stats=stats,
                        output_dir=run_dir,
                    )
                    metrics["training_seconds"] = result.training_seconds
                    metrics["wall_seconds"] = time.perf_counter() - started
                    run = {
                        "name": f"{dataset_spec['name']}-{name}-{condition['name']}-s{seed}",
                        "dataset": dataset_spec["name"],
                        "dataset_repo": dataset_spec["repo_id"],
                        "dataset_revision": dataset_spec["revision"],
                        "model": name,
                        "condition": condition["name"],
                        "family": condition["family"],
                        "corruption": components,
                        "mixture_manifest": mixture_manifest,
                        "seed": seed,
                        "model_config": config,
                        "training_config": train_config,
                        "split": split,
                        "metrics": metrics,
                        "experiment_schema_version": 2,
                        "quality_schema_version": QUALITY_SCHEMA_VERSION,
                        "source_fingerprint": source_fingerprint,
                        "experiment_signature": signature,
                        "status": "complete",
                        "checkpoint": result.checkpoint.name,
                        "checkpoint_metadata": payload.get("metadata", {}),
                    }
                    _write_json_atomic(run_path, run)
                    runs.append(run)
                    if not args.defer_report:
                        write_report(runs, study, output_root)
                    print(
                        f"done {run['name']}: mse={metrics['action_mse']:.6g} "
                        f"time={result.training_seconds:.1f}s",
                        flush=True,
                    )
                    del model, mixed_episodes, training_episodes
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        del episodes, clean_train, clean_validation, clean_test
    study["completed_at"] = datetime.now(UTC).isoformat()
    if args.defer_report:
        return output_root / "report.html"
    _write_json_atomic(output_root / "study.json", study)
    return write_report(runs, study, output_root)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=Path("runs/quality_ablation_v1_2_0"))
    result.add_argument("--datasets-root", type=Path, default=datasets_root())
    result.add_argument("--datasets", nargs="+", choices=[item["name"] for item in DATASETS])
    result.add_argument("--models", nargs="+", choices=MODELS)
    result.add_argument("--conditions", nargs="+", choices=CONDITION_ORDER)
    result.add_argument("--device")
    result.add_argument("--episodes", type=int, help="0 means all episodes")
    result.add_argument("--frames", type=int, help="0 means all frames per episode")
    result.add_argument("--image-size", type=int)
    result.add_argument("--epochs", type=int)
    result.add_argument("--seeds", type=int, nargs="+")
    result.add_argument("--split-seed", type=int)
    result.add_argument("--bc-batch-size", type=int)
    result.add_argument("--act-batch-size", type=int)
    result.add_argument("--segment-length", type=int)
    result.add_argument("--skip-download", action="store_true")
    result.add_argument("--defer-report", action="store_true")
    result.add_argument("--report-only", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    report = run_study(args)
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
