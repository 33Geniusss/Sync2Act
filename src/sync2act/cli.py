from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sync2act.config import load_config
from sync2act.corruptions import apply_corruption
from sync2act.pipeline import (
    make_demo_dataset,
    run_benchmark,
    run_demo,
    run_evaluation,
    run_training,
)
from sync2act.reporting import generate_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sync2act", description="Robot data-quality imitation-learning toolkit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Run the bundled offline end-to-end demo")
    demo.add_argument("--output", default="runs/demo")
    corrupt = subparsers.add_parser("corrupt", help="Apply a configured corruption to bundled data")
    corrupt.add_argument("--config", required=True)
    corrupt.add_argument("--output", default="runs/corrupted_demo.pt")
    train = subparsers.add_parser("train", help="Train a policy")
    train.add_argument("--config", required=True)
    train.add_argument("--output", default="runs/train")
    train.add_argument("--resume")
    evaluate = subparsers.add_parser("evaluate", help="Evaluate a checkpoint offline")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", default="runs/evaluation")
    benchmark = subparsers.add_parser("benchmark", help="Run a model/corruption/seed matrix")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--output", default="runs/benchmark")
    report = subparsers.add_parser("report", help="Build an HTML report from run.json files")
    report.add_argument("--runs", required=True)
    report.add_argument("--output", default="reports/benchmark.html")
    subparsers.add_parser("gui", help="Launch the desktop application")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "demo":
        result = run_demo(
            args.output,
            lambda event: (
                print(f"step={event.get('step')} loss={event.get('train_loss'):.6f}")
                if not event.get("epoch_complete")
                else None
            ),
        )
        print(json.dumps(result, indent=2))
    elif args.command == "corrupt":
        config = load_config(args.config)
        episodes = [
            apply_corruption(episode, {**config, "seed": int(config.get("seed", 7)) + index})
            for index, episode in enumerate(make_demo_dataset())
        ]
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(episodes, target)
        print(f"Saved {len(episodes)} corrupted episodes to {target}")
    elif args.command == "train":
        config = load_config(args.config)
        _, _, result = run_training(
            config, args.output, lambda event: print(json.dumps(event)), resume_from=args.resume
        )
        print(f"Checkpoint: {result.checkpoint}")
    elif args.command == "evaluate":
        metrics = run_evaluation(load_config(args.config), args.checkpoint, args.output)
        print(json.dumps(metrics, indent=2))
    elif args.command == "benchmark":
        results = run_benchmark(load_config(args.config), args.output)
        print(f"Completed {len(results)} runs; report: {Path(args.output) / 'benchmark.html'}")
    elif args.command == "report":
        files = sorted(Path(args.runs).rglob("run.json"))
        runs = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        print(generate_report(runs, args.output))
    elif args.command == "gui":
        from sync2act.gui import main as gui_main

        return gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
