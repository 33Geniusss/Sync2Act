# Sync2Act implementation brief (for Codex)

[Chinese version](SYNC2ACT_CODEX_PROJECT_BRIEF.md) | English

This is the original implementation brief, not a statement that every requested capability has been implemented. See [README](README.md) for current behavior and evidence.

## Your task

Design and implement **Sync2Act** from scratch in the current workspace as a complete project suitable for publication on GitHub.

Do not deliver only a proposal, pseudocode, a notebook, or disconnected scripts. Create the project files, implement the code, run tests, fix problems, and deliver a runnable PyTorch application. Continue with reasonable defaults unless a major scope change requires a user decision.

You may create and edit files within the project, install necessary dependencies, and run non-destructive commands and tests. Do not upload code, create remote repositories, publish releases, purchase services, or perform other external writes unless the user subsequently authorizes them.

## 1. Project purpose

Sync2Act is a PyTorch toolkit and desktop application for studying and demonstrating how robot data quality affects imitation-learning policies.

Core questions:

1. How much does policy performance degrade when images, robot states, and actions become temporally misaligned?
2. What effects do dropped frames, action noise, and state anomalies have?
3. Can a policy that explicitly uses data-quality information be more robust than an ordinary policy?

The project should provide this complete workflow:

```text
Robot demonstrations
    -> controlled data corruption
    -> PyTorch policy training
    -> offline metrics and closed-loop rollout evaluation
    -> comparisons of curves, tables, trajectories, and videos
    -> exportable experiment reports
```

The goal is neither a general-purpose large VLA model nor a thin wrapper around existing training commands. The code must clearly expose the PyTorch data pipeline, model definitions, losses, training loop, checkpoints, evaluation, and performance measurement.

## 2. Required deliverables

The repository must contain at least:

1. An installable Python package, `sync2act`.
2. A runnable **native PySide6 desktop window**.
3. A BC-MLP baseline policy.
4. An ACT-Lite action-chunk policy.
5. Quality-Aware ACT extending ACT-Lite.
6. Reproducible data-corruption modules.
7. Training, evaluation, checkpoint, and report-generation workflows.
8. Synthetic demonstration data and smoke tests that work without a GPU or external datasets.
9. Unit tests, end-to-end tests, and GitHub Actions configuration.
10. A high-quality README for GitHub visitors, including an architecture diagram, a place for UI screenshots, commands, experiment design, and result tables.

A CLI, notebook, or web page without a desktop window does not satisfy this brief.

## 3. Recommended stack

- Python 3.11
- PyTorch and torchvision
- PySide6
- PyQtGraph for live training curves and interactive desktop charts
- NumPy and Pandas
- Pillow or OpenCV for image/video processing
- PyYAML for experiment configuration
- pytest and pytest-qt
- ruff
- Optional: TensorBoard, LeRobot, Gymnasium/PushT, Hugging Face Hub
- Optional release tooling: PyInstaller for Windows

Check stable, compatible versions before implementation and specify reasonable ranges in `pyproject.toml`. Core tests must not require network access or large dataset downloads.

## 4. Model definitions

### 4.1 BC-MLP

Purpose: a simple, fast behavioral-cloning baseline for debugging.

Support at least two input modes:

1. `state_only`: encode robot state with an MLP and predict the next action.
2. `image_state`: encode RGB images with a lightweight CNN and state with an MLP, fuse their representations, and predict the next action.

```text
image[t] + state[t] -> action[t]
```

Requirements:

- Explicit input/output shape checks.
- State and action normalization.
- MSE or Smooth L1 action loss.
- Rapid overfitting of a small synthetic batch to validate the training pipeline.

### 4.2 ACT-Lite

Purpose: the project's main temporal imitation-learning model, predicting a future action sequence in one forward pass.

```text
image[t] + state[t] -> action[t : t + H]
```

Minimum architecture:

- Lightweight CNN image encoder.
- Robot-state embedding.
- Learnable action queries.
- Transformer encoder or small encoder-decoder.
- Positional encoding.
- Padding mask.
- Configurable action-chunk length, for example `H=16` by default.
- Receding-horizon execution at inference.
- Temporal ensembling: weighted fusion of predictions for the same action time from overlapping chunks.

The first version may omit the full ACT CVAE latent-variable architecture. The README must explain this difference and must not describe ACT-Lite as a complete reproduction of the original paper.

### 4.3 Quality-Aware ACT

Purpose: the main proposed extension, testing whether quality information mitigates degradation from bad data.

Reuse the ACT-Lite core rather than maintaining a separate copy.

The MVP must implement:

1. **Quality-weighted loss:** assign `quality_score` per sample or time step so that low-quality samples contribute less to the loss.
2. **Quality-feature input:** supply quality scores, missing masks, or temporal-offset embeddings as additional tokens/features.

Recommended loss:

```text
total_loss = weighted_action_loss
             + lambda_smooth * trajectory_smoothness_loss
             + optional_regularization
```

All switches and weights must be configurable in YAML for ablation studies.

## 5. Data format and corruption modules

Define a unified episode interface containing at least:

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

First provide a small synthetic dataset that demonstrates the full pipeline without LeRobot or a simulator. Then add an optional LeRobot dataset adapter; keep version-specific LeRobot internals out of core modules.

Implement at least the following corruption types.

### 5.1 Temporal shift

- Select image, state, or action.
- Support positive and negative frame offsets.
- Configurable boundary handling: drop, repeat, zero, or mask.

### 5.2 Frame drop

- Randomly drop frames with a specified probability.
- Support previous-frame, zero-frame, and interpolation replacements.
- Update `missing_mask` and `quality_score`.

### 5.3 Action noise

- Gaussian noise.
- Sparse spikes.
- Fixed or random action delay.

### 5.4 State anomaly

- State spikes.
- Brief missing intervals.
- A dimension stuck at a constant value.
- Timestamp jitter.

Every corruption must:

- Accept an explicit random seed.
- Preserve the original data.
- Output complete provenance metadata.
- Record corruption type, parameters, seed, and affected indices.
- Produce elementwise-identical results for identical inputs and seeds.

## 6. Experiment design

Clearly distinguish two experiment types.

### 6.1 Corrupted training data

```text
Train on damaged demonstrations -> test in a clean environment
```

This studies how bad demonstrations affect learning.

### 6.2 Corrupted observations at runtime

```text
Train on clean demonstrations -> test with sensor faults
```

This studies deployment problems such as camera latency and dropped frames.

MVP matrix:

- Models: BC-MLP, ACT-Lite, Quality-Aware ACT.
- Data: clean; image shifts of 1/2/4 frames; frame drops of 5%/10%/20%; three action-noise levels.
- Support multiple seeds. A quick demonstration may use one seed; formal benchmarks should default to at least three.

Never fabricate benchmark values. If formal training cannot run in the current environment, mark results `not run` and provide commands for running the experiments.

## 7. Evaluation metrics

Implement and save at least:

- Action MSE and MAE.
- Action/trajectory smoothness.
- Jerk or discrete second-difference statistics.
- Per-episode results.
- Inference latency P50 and P95.
- Parameter count.
- Training time.
- If simulation is available: rollout success rate, return, and completion steps.

Offline action loss cannot substitute for closed-loop success rate. Formal results should prioritize rollout success rate. If simulation is unavailable, the UI and report must explicitly label results **offline-only**.

## 8. Native visual desktop application (mandatory)

Create a Windows-friendly PySide6 application with a suggested entry point:

```bash
python -m sync2act.gui
```

Or, after installation:

```bash
sync2act-gui
```

Include at least these pages or tabs.

### 8.1 Overview

- Current dataset, episode count, frame count, input shapes.
- Current device: CPU/CUDA.
- Most recent training status.
- Current best model and main metrics.
- One-click loading of a bundled demo dataset.

### 8.2 Dataset Inspector

- Episode and time-step selectors.
- RGB frame preview.
- State/action time-series plots.
- Timestamps and sampling intervals.
- Missing-mask and quality-score visualizations.
- Side-by-side original/corrupted data.

### 8.3 Corruption Studio

- Corruption-type selection.
- Sliders/spin boxes for shift, drop probability, noise sigma, etc.
- Random-seed setting.
- Live preview of a short original/corrupted segment.
- Apply, Reset, and Save Config buttons.
- No overwriting original data.

### 8.4 Training Monitor

- Choose BC-MLP, ACT-Lite, or Quality-Aware ACT.
- Load/save YAML configuration.
- Start, Pause/Stop, and Resume controls.
- Live train loss, validation loss, learning rate, step, epoch, and ETA.
- Device, GPU memory when available, and checkpoint path.
- Background training workers that do not freeze the UI thread.
- Safely save a resumable checkpoint when the user stops training.

### 8.5 Evaluation & Compare

- Select multiple checkpoints.
- Side-by-side model metrics.
- Corruption-level versus metric curves.
- Ground-truth and predicted action trajectories.
- Playback or frame inspection if rollout video is available.
- Clearly distinguish offline metrics from rollout metrics.

### 8.6 Report

- Generate summary tables and charts.
- Export HTML; optionally PDF/CSV/JSON.
- Include configuration, Git commit when available, dependencies, seeds, and runtime.
- Never display unrun results as zero or invented numbers.

### 8.7 UI quality

- Clear sidebar or tab navigation.
- Appropriate spacing, typography, color hierarchy, and status feedback.
- Progress and cancellation for long tasks.
- Understandable error dialogs/logs rather than application crashes.
- No overlapping or clipped controls at common Windows display scales.
- Demonstrable data on startup rather than an empty window.
- Actually launch and inspect the final window; save a README screenshot if the environment supports capture.

## 9. CLI

Provide a CLI alongside the desktop application for automation:

```bash
sync2act demo
sync2act corrupt --config configs/corruption/shift_2.yaml
sync2act train --config configs/train/act_lite.yaml
sync2act evaluate --config configs/eval/default.yaml
sync2act benchmark --config configs/benchmark/mvp.yaml
sync2act report --runs runs/ --output reports/benchmark.html
```

GUI and CLI must use the same core logic, not separate training or data implementations.

## 10. Recommended repository structure

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

Keep runtime artifacts in ignored `runs/`, `checkpoints/`, and local data directories. Do not commit large datasets, caches, model weights, secrets, or user-machine absolute paths.

## 11. Tests and acceptance

Complete at least these automated checks.

### Data tests

- Correct corruption-output shapes.
- Exact reproducibility with the same seed.
- Original inputs remain unchanged.
- Quality and missing masks match the applied corruption.
- No crossing episode boundaries.

### Model tests

- Correct forward shapes for all three models.
- Correct padding-mask behavior.
- All-one weights make weighted loss equal ordinary loss.
- No NaNs for edge cases such as all-zero valid weights.
- Identical outputs after checkpoint save/load.
- Testable temporal-ensemble behavior.

### Training tests

- BC-MLP can overfit a tiny synthetic batch.
- ACT-Lite can run several training steps and reduce loss.
- Resume optimizer, scheduler, and step from a checkpoint.
- CPU smoke test completes in reasonable time.

### GUI tests

- Main window opens and closes.
- Bundled demo loads.
- Corruption-parameter changes update the preview.
- Background training does not block the UI event loop.
- Clear messages for no data, no checkpoint, and no CUDA.

### Final manual acceptance

1. Install in a fresh environment using the README.
2. `sync2act demo` completes a miniature end-to-end workflow.
3. GUI launches and displays the demo dataset.
4. A user can create a corruption in the GUI.
5. A user can start short training and see curves update.
6. A user can load a checkpoint and compare predictions with ground truth.
7. A user can export an HTML report without fabricated data.
8. All automated tests pass.

## 12. Implementation phases

Implement in this order and test after every phase; do not start with many empty shells.

### Phase 1: Runnable foundation

- Python package and configuration system.
- Synthetic episodes.
- BC-MLP.
- Minimal training/evaluation loops.
- Minimal PySide6 window with demo loading, short training, and a loss plot.
- Unit tests and README quick start.

### Phase 2: Data-quality experiments

- Four corruption families.
- Dataset Inspector and Corruption Studio.
- Provenance and reproducibility tests.
- Working CLI workflow.

### Phase 3: Action-chunk model

- ACT-Lite.
- Temporal ensembling.
- Checkpoints and resume.
- Complete Training Monitor.

### Phase 4: Quality-aware method

- Quality-Aware ACT.
- Weighted loss and quality features.
- Ablation configurations.
- Compare page and report generation.

### Phase 5: Real benchmarks and release quality

- Optional LeRobot/PushT adapter.
- Closed-loop rollouts.
- Multi-seed benchmarks.
- UI polish, screenshots, and demo GIF.
- Optional PyInstaller Windows build.

If GPU, simulator, or network access is missing, finish independent phases and make external integrations optional. Do not stop core implementation because a large formal experiment cannot run.

## 13. Questions the README must answer

Within a minute, a GitHub visitor should understand:

1. What problem does Sync2Act solve?
2. Why does robot data quality affect imitation learning?
3. What are BC-MLP, ACT-Lite, and Quality-Aware ACT?
4. How can the demo and GUI start in three commands?
5. Which results were actually measured, and which were not run?
6. How can experiments be reproduced?
7. What does the UI look like?
8. How can users import their own episodes or LeRobot datasets?

Prioritize a one-sentence description, GUI screenshot, short demo GIF, quick start, main results table, and architecture diagram. Do not lead with lengthy installation details.

## 14. Engineering principles

- Never fabricate data, success rates, performance numbers, or screenshots.
- Never present training loss as closed-loop success rate.
- Never describe ACT-Lite as a full ACT reproduction.
- Preserve original user data.
- Keep long training off the GUI thread.
- Do not hardcode absolute paths in source.
- Do not require network access in tests.
- Import success does not replace end-to-end validation.
- Build a working vertical slice before expanding.
- Provide clear optional fallbacks when dependencies are unavailable.

## 15. Codex workflow and final report

Inspect the current directory, existing files, and runtime first, then implement. Keep changes focused and module boundaries clear. Run relevant tests after each phase. Safe local reads, code edits, and tests do not require individual confirmations.

The final response must include:

- Implemented features.
- GUI launch command.
- CLI quick-demo command.
- Test results.
- Experiments actually run and their measured results.
- Unrun or environment-limited work.
- Main file entry points.
- The most valuable next extension.

Do not claim completion after delivering only a plan, an empty UI, or unverified code.
