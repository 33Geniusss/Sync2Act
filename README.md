# Sync2Act

[![CI](https://github.com/33Geniusss/Sync2Act/actions/workflows/ci.yml/badge.svg)](https://github.com/33Geniusss/Sync2Act/actions/workflows/ci.yml)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A PyTorch + native desktop lab for measuring how robot demonstration quality changes imitation-learning behavior.**

Sync2Act turns an episode-based robot dataset into controlled, reproducible experiments: import data, damage selected training episodes, train one of three imitation-learning policies, compare offline predictions, and export an evidence-bearing report. The desktop application starts empty and does not bundle a dataset, trained checkpoint, or benchmark result.

![Sync2Act Overview](assets/gui-overview.png)

## Research workflow

```text
Original episodes
├─ Training episodes   → clean or deliberately corrupted
├─ Validation episodes → always clean
└─ Test episodes       → always clean
```

The split is episode-level, preventing frames from one trajectory from leaking across partitions. Corruptions are applied only to the training partition. Normalization statistics come from that partition, while model comparison uses the held-out clean test episodes.

## Quick start from source

```bash
python -m pip install -e .[dev]
sync2act-gui
```

Python 3.11–3.13 is supported. Import or download a dataset from **Overview** before opening the training workflow. A deterministic synthetic-data command remains available for development and automated tests through `sync2act demo`, but it is not loaded by the GUI.

## What is implemented

| Policy | Prediction | Purpose |
|---|---|---|
| BC-MLP | `state[t]` or `image[t] + state[t] → action[t]` | Fast baseline and pipeline debugging |
| ACT-Lite | `image[t] + state[t] → action[t:t+H]` | Learnable action queries, positional encoding, Transformer, padding-aware chunk loss, receding-horizon temporal ensemble |
| Quality-Aware ACT | ACT-Lite plus quality token and weighted loss | Tests whether quality metadata reduces sensitivity to damaged demonstrations |

ACT-Lite is **not** a full reproduction of ACT: it intentionally omits the original CVAE latent-variable path. Quality-Aware ACT subclasses and reuses ACT-Lite rather than maintaining a copied model.

All three policies are implemented in this repository and trained from scratch on the selected dataset; they are not fine-tuned third-party checkpoints.

Data quality matters because behavior cloning assumes observations and labels describe the same moment and valid sensor state. A shifted image, repeated frame, delayed action, or stuck joint can turn a valid demonstration into a contradictory training target. Sync2Act records exactly what changed and makes that assumption testable.

## Desktop application

Launch with `sync2act-gui` or `python -m sync2act.gui`. The PySide6 window starts with no dataset loaded, preventing accidental training on demonstration data, and provides:

- **Overview:** empty-until-imported episode/frame/shape/device summary, background Hugging Face downloads, and local LeRobot v3 loading with progress. Downloads default to `<user-home>/Sync2Act/datasets/<repository-name>` while still allowing a custom target. Hub transfer metadata is kept separately under `<user-home>/Sync2Act/.cache/huggingface`, so dataset folders contain only repository files and deeply nested camera paths do not inherit long temporary filenames.
- **Dataset Inspector:** RGB preview, episode/step totals, one shared state/action dimension selector, a separate quality plot, time-labelled axes, selected-step cursors, and original-versus-working-copy comparison.
- **Corruption Studio:** type-aware target controls, deterministic-seed disabling, a live three-stream alignment diagram for temporal shifts, a dropped-frame timeline, and before/after action or state curves with exact frame-level original/new/delta values.
- **Training Monitor:** BC-MLP, ACT-Lite, or Quality-Aware ACT in a background worker; editable device, epochs, batch size, learning rate, weight decay, validation split, loss, seed, and gradient clipping; recommended-value hints; estimated optimizer steps; live loss/LR/ETA; safe stop, checkpoint, and resume. New checkpoints are automatically named `<model>_<data-condition>_<timestamp>.pt` under `<application-folder>/models`.
- **Evaluation & Compare:** checkpoint loading, offline metrics, per-episode results, and target/prediction trajectories.
- **Report:** HTML and JSON export containing configuration, seeds, dependency versions, timestamp, and Git commit when available.

The Training Monitor estimates work before launch as:

```text
steps_per_epoch = ceil(training_frames / batch_size)
total_steps     = epochs × steps_per_epoch
```

Validation batches are not included in this optimizer-step estimate.

The optional local Windows bundle can be rebuilt with:

```bash
python -m pip install pyinstaller
python -m PyInstaller --noconfirm Sync2Act.spec
dist\Sync2Act\Sync2Act.exe
```

At runtime, the packaged CUDA build uses an available NVIDIA GPU when compatible CUDA drivers are present; otherwise the application reports CPU. Prebuilt archives, when published, belong on the repository's [Releases page](https://github.com/33Geniusss/Sync2Act/releases), not in Git history.

## Architecture

```mermaid
flowchart LR
    A[Episode data\nimage · state · action · timestamp] --> B[Seeded corruptions]
    B --> C[Window dataset\nnormalization · masks · chunks]
    C --> D{Shared policy API}
    D --> E[BC-MLP]
    D --> F[ACT-Lite]
    D --> G[Quality-Aware ACT]
    E --> H[Trainer + checkpoint]
    F --> H
    G --> H
    H --> I[Offline evaluator]
    I --> J[CSV · JSON · HTML report]
    K[PySide6 GUI] --> B
    K --> H
    K --> I
    L[CLI] --> B
    L --> H
    L --> I
```

The GUI and CLI call the same functions in `pipeline.py`, `training/`, `evaluation/`, and `reporting/`; there is no second training implementation hidden in the interface.

## Reproducible experiments

```bash
# Save a deterministically corrupted demo dataset
sync2act corrupt --config configs/corruption/shift_2.yaml

# Train and evaluate
sync2act train --config configs/train/act_lite.yaml --output runs/act-lite
sync2act evaluate --config configs/eval/default.yaml \
  --checkpoint runs/act-lite/checkpoint.pt --output runs/act-lite/eval

# Full 3-model × 10-condition × 3-seed matrix (CPU-intensive)
sync2act benchmark --config configs/benchmark/mvp.yaml

# Rebuild a report from completed run manifests only
sync2act report --runs runs/ --output reports/benchmark.html
```

Training-data corruption is represented by a `corruption` block in a training config: damaged demonstrations are used for learning and clean episodes for evaluation. Runtime observation corruption belongs in an environment/rollout adapter and is reported separately; it is not silently approximated by offline loss.

The desktop workflow performs a deterministic episode-level train/validation/test split before applying damage. Corruption Studio changes training episodes only; validation and test episodes remain clean. Validation uses normalization statistics computed from the training partition, and Evaluation Compare runs only on the held-out clean test episodes. The split indices and ratios are saved with checkpoints and reports.

Every corruption clones its input, accepts a seed, updates `quality_score` and/or `missing_mask`, and attaches provenance with type, parameters, seed, and affected indices. Supported variants are:

- temporal shift of image/state/action with `drop`, `repeat`, `zero`, or `mask` boundaries;
- frame drop with previous-frame, zero-frame, or interpolation replacement;
- Gaussian/spike action noise and fixed/random action delay;
- state spike, short missing span, stuck dimension, and timestamp jitter.

## Offline evaluation

Evaluation is performed on held-out samples with ground-truth actions. Sync2Act reports `action_mse`, `action_mae`, trajectory smoothness, jerk, P50/P95 inference latency, parameter count, and per-episode metrics. Smoothness and jerk are calculated inside each episode and then aggregated, so the final frame of one episode is never connected to the first frame of another.

These are offline imitation metrics, not closed-loop robot success rates. The repository intentionally contains no precomputed benchmark results or trained models; reported numbers should come from the user's own dataset, split, seed, and hardware.

## Dataset download and storage

The Overview tab can download a public or locally authenticated private Hugging Face dataset repository. The target is selectable, download progress is shown, and the default paths are:

```text
C:\Users\<username>\Sync2Act\datasets\<repository-name>
C:\Users\<username>\Sync2Act\.cache\huggingface
```

The second directory contains resumable Hugging Face transfer metadata. It is deliberately separated from the dataset directory to avoid Windows path-length failures. Dataset folders, caches, checkpoints, generated runs, and release archives are excluded from Git.

## Local LeRobot datasets

For a LeRobot v3 repository, select its root folder and click **Load local dataset**. A typical supported layout contains:

```text
meta/info.json
data/chunk-*/file-*.parquet
videos/<image-feature>/chunk-*/*.mp4
```

The loader reads parquet rows, decodes every RGB video feature, aligns all camera views by frame, splits frames by `episode_index`, initializes clean quality metadata, and derives state/action dimensions automatically. To keep large multi-camera datasets practical in memory, imported views preserve their aspect ratio and are capped at 128 pixels on the longest edge; cameras with different resolutions are then resized to the same training shape. Original MP4 files are never modified. The dataset must expose `observation.state`, `action`, `episode_index`, `timestamp`, and at least one video feature.

All synchronized cameras are used during training and inference. ACT-Lite and Quality-Aware ACT retain one observation token per camera; image-enabled BC-MLP encodes every camera with shared weights and mean-fuses the camera embeddings. The Dataset Inspector displays the synchronized views side by side. Task text is not currently used, so task-conditioned VLA policies remain a future extension.

For a custom adapter, convert each episode to the public dictionary contract below and pass a list of episodes to `EpisodeWindowDataset` or the training functions:

```python
{
    "observation.image": Tensor[T, cameras, 3, H, W],
    "observation.state": Tensor[T, state_dim],
    "action": Tensor[T, action_dim],
    "timestamp": Tensor[T],
    "quality_score": Tensor[T],
    "missing_mask": Tensor[T],
}
```

`sync2act.data.lerobot.LeRobotEpisodeAdapter` provides the local LeRobot integration boundary. Install optional LeRobot dependencies with `pip install -e .[lerobot]`; version-specific code remains isolated from the core data and model modules.

## Development and verification

```bash
python -m ruff check src tests tools
pytest
```

Tests cover corruption shape/reproducibility/non-mutation/provenance, model shapes, weighted-loss edge cases, temporal ensembling, checkpoint equivalence and resume, BC/ACT training, end-to-end artifacts, and GUI event-loop behavior. GUI tests use `QT_QPA_PLATFORM=offscreen` and `SYNC2ACT_RUN_GUI_TESTS=1` in GitHub Actions; locally they skip when no Qt-capable session is exposed.

The formal benchmark is intentionally not run during installation. It is a 90-run CPU workload by default and must never be replaced with invented values. Optional next work is a concrete LeRobot field converter followed by a Gymnasium/PushT runtime-corruption wrapper and closed-loop success evaluation.

## Repository map

```text
src/sync2act/data/          episode contract, synthetic data, windows, LeRobot boundary
src/sync2act/corruptions/   deterministic damage operators and provenance
src/sync2act/policies/      BC-MLP, ACT-Lite, Quality-Aware ACT
src/sync2act/training/      loss, loop, safe checkpoint and resume
src/sync2act/evaluation/    offline metrics and temporal ensembling
src/sync2act/reporting/     HTML/JSON evidence report
src/sync2act/gui/           native PySide6 application and worker
src/sync2act/cli.py         automation entrypoint
configs/                    corruption, train, eval, benchmark YAML
tests/                      unit, integration, GUI
```

MIT licensed. Large datasets, checkpoints, generated runs, and reports are ignored by Git.
