from __future__ import annotations

import html
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def generate_report(
    runs: list[dict[str, Any]], output: str | Path, title: str = "Sync2Act experiment report"
) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    columns = [
        "name",
        "model",
        "corruption",
        "seed",
        "scope",
        "action_mse",
        "action_mae",
    ]
    rows = []
    for run in runs:
        metrics = run.get("metrics", {})
        values = {
            "name": run.get("name", "unnamed"),
            "model": run.get("model", "not run"),
            "corruption": run.get("corruption", "clean"),
            "seed": run.get("seed", "not run"),
            "scope": metrics.get("evaluation_scope", "not run"),
            "action_mse": metrics.get("action_mse", "not run"),
            "action_mae": metrics.get("action_mae", "not run"),
        }
        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(values[column]))}</td>" for column in columns)
            + "</tr>"
        )
    measured = [
        (str(run.get("name", "unnamed")), run.get("metrics", {}).get("action_mse"))
        for run in runs
        if isinstance(run.get("metrics", {}).get("action_mse"), (int, float))
    ]
    maximum = max((value for _, value in measured), default=1.0) or 1.0
    bars = (
        "".join(
            f"<div class='bar-row'><span>{html.escape(name)}</span>"
            f"<i style='width:{max(1, value / maximum * 100):.1f}%'></i>"
            f"<b>{value:.6f}</b></div>"
            for name, value in measured
        )
        or "<p>not run</p>"
    )
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>
<style>body{{font-family:Segoe UI,sans-serif;max-width:1100px;margin:40px auto;color:#172033}}h1{{color:#3157d5}}.notice{{background:#fff4cf;padding:14px;border-left:4px solid #e6a700}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #dce2ee;text-align:left}}th{{background:#eef2ff}}pre{{background:#f6f7fb;padding:14px;overflow:auto}}.bar-row{{display:grid;grid-template-columns:180px 1fr 100px;gap:12px;align-items:center;margin:8px 0}}.bar-row i{{height:16px;background:#3157d5;border-radius:3px}}</style></head><body>
<h1>{html.escape(title)}</h1><p class='notice'>Evaluation scope is explicit. “not run” means no measurement exists; offline action error is not rollout success.</p>
<h2>Results</h2><table><thead><tr>{"".join(f"<th>{column}</th>" for column in columns)}</tr></thead><tbody>{"".join(rows)}</tbody></table>
<h2>Action MSE comparison</h2>{bars}
<h2>Reproducibility metadata</h2><pre>{html.escape(json.dumps(metadata, indent=2))}</pre>
<h2>Run configurations</h2><pre>{html.escape(json.dumps(runs, indent=2))}</pre></body></html>"""
    target.write_text(document, encoding="utf-8")
    target.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata, "runs": runs}, indent=2), encoding="utf-8"
    )
    return target
