# Sync2Act interview Q&A: NVIDIA Robotics / AI Developer Technology

[Chinese version](SYNC2ACT_INTERVIEW_QA_ZH.md) | English

Original review date: 2026-09-25, against source commit `e6077849f3e8181e31d5aa8318da343d4c6830ab` and study `runs/mixed_modality_damage_v1_3_0`. These 84 practice questions are based on the role and project implementation; they are not an actual NVIDIA question bank.

2026-10-01 update: causal temporal-ensemble evaluation now reuses existing checkpoints and the original test split; see Q59. The source hash above identifies the original review, not this update.

This document uses the newer 270-run study. Descriptions of action delay and the absence of image experiments in `quality_ablation_v1_2_0` do not apply here. The artifact directory's `v1_3_0` is not necessarily the Python package version.

## How to use this document

- Understand each answer and explain it in your own words. Personal contributions, development stories, and claims of firsthand validation must match what you actually did.
- Answers provide a spoken-response starting point; code, formulas, and follow-up notes support deeper discussion.
- Prioritize these 20 questions: Q01, Q03, Q06, Q09, Q12, Q19, Q23, Q24, Q28, Q33, Q35, Q36, Q43, Q45, Q53, Q54, Q58, Q63, Q65, Q66.
- Categories: Q01-08 purpose; Q09-18 data; Q19-32 models; Q33-42 quality mechanisms; Q43-52 training/experiments; Q53-62 results/evaluation; Q63-76 GPU/CUDA; Q77-84 engineering/limitations.

## Essential project facts

| Item | Current fact |
|---|---|
| Goal | Study how damaged demonstrations affect offline imitation learning and how to use quality information |
| Formal experiment | 3 datasets × 6 variants × 5 conditions × 3 seeds = 270 runs |
| Training | Single-GPU CUDA path; 10 epochs; batch 256; image longest edge 64; action horizon 8 |
| Core architecture | Small CNN + linear state encoder + action queries + Transformer Encoder |
| Full quality method | Quality-conditioned observation tokens + action-label-quality-weighted loss |
| Quality source | Metadata and heuristic scores from known synthetic faults, called Oracle quality |
| Test | Clean held-out episodes with a fixed split; offline action error; no closed-loop success rate |
| Strongest result | Correct action-label weights mitigate label-damage degradation; image quality has no clear benefit here |
| Limitations | Offline paired temporal-ensemble evaluation only; no language input, custom CUDA kernels, multi-GPU training, or closed-loop evidence |

## 1. Purpose and personal contribution (Q01-Q08)

**Q01 [Priority] Introduce the project in one minute.**

Sync2Act is a platform for experiments on robot-demonstration quality. It organizes camera images, robot states, and demonstrated actions into episodes, injects reproducible misalignment and missing data only into training episodes, and compares an ordinary action-chunk Transformer with quality-aware variants. The latest study contains 270 offline runs: three datasets, six variants, five conditions, and three seeds. State-quality input partly mitigates state damage, whereas damaged action labels benefit more from loss weighting. Image quality has no clear benefit under this setup. There is no real-robot closed-loop conclusion.

**Q02 Why focus on data quality instead of using a larger model?**

Behavioral cloning depends on correctly matched observations and action labels. More capacity does not guarantee recovery of correct supervision from misaligned or missing labels. I wanted controlled variables for identifying what is damaged, how much it matters, and how quality information helps. Small models enable repeated, interpretable experiments, but I have not shown that the conclusions transfer unchanged to large robot foundation models.

**Q03 [Priority] What is the main research question?**

Should unreliable observations and unreliable supervision be treated identically? The experiment uses image/state quality as input conditioning and action-label quality as loss weights, then ablates the two mechanisms. Results support separating knowledge about unreliable inputs from decisions about which target answers deserve supervision.

**Q04 What is the contribution or innovation?**

The defensible contribution is a reproducible multimodal demonstration-quality workflow and a controlled comparison of observation quality versus label quality. The mechanism combines explicit token scaling, metadata encoding, and weighted loss. I do not claim to have invented ACT or quality weighting, or to establish novelty or SOTA without a comprehensive literature review. Personal design and implementation contributions must be described from actual experience.

**Q05 How does this relate to the NVIDIA role?**

The project covers robot data, PyTorch training, Transformers, evaluation, and developer-facing tools. It demonstrates understanding of training workflows and evidence. Profiling would extend it toward the role's training/inference optimization requirements. Distributed training, custom CUDA kernels, and Isaac/GR00T deployment are not completed project experience. CUDA familiarity being preferred does not establish which CUDA questions will be asked. [Role description](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/China-Shanghai/AI-Developer-Technology-Intern--Robotics---2027_JR2024054-1?source=greatrobots)

**Q06 [Priority] If AI helped substantially, what did you personally do?**

Implementation used substantial AI assistance, so I would not claim to have independently handwritten everything. My contribution should identify requirements I chose, configurations I changed, experiments I ran, results I checked, and code I learned. For each item, explain the decision, validation, and finding, backed by files or records. Distinguish modules I can explain from modules I have independently implemented.

Follow-up: do not memorize an invented development story. Before interviewing, personally run a tensor demonstration, understand a corruption operator, and perform a small controlled comparison.

**Q07 Is this reinforcement learning, imitation learning, or VLA?**

It is supervised behavioral cloning, a form of imitation learning. It fits demonstrated actions from current observations without learning from environment rewards. There is no language input, language encoder, or cross-task instruction conditioning, so it is not a VLA. Models are trained separately for the three datasets, not as one universal cross-robot foundation model.

**Q08 How is the system layered, and why a GUI?**

The bottom layer contains episode schemas, loading, and corruption operators. The middle contains model interfaces, Dataset, training, and checkpoints. Evaluation and reporting sit above them. The GUI visualizes data, configures faults, and monitors training. It makes misalignment and missing data concrete; the research value comes from reproducible protocols and results. Shared GUI/CLI modules do not mean every entry point uses identical splitting and preprocessing.

Code: [study runner](tools/run_real_dataset_study.py), [trainer](src/sync2act/training/trainer.py), [GUI worker](src/sync2act/gui/worker.py).

## 2. Data, preprocessing, and splitting (Q09-Q18)

**Q09 [Priority] What data did you use? Did you collect it?**

The formal study uses three public LeRobot datasets. There is no evidence that I collected them by operating robots. Saved loader records show the complete datasets were partitioned:

| Dataset | Episodes / frames | Cameras | State/action dimensions | Train/val/test episodes |
|---|---:|---:|---:|---|
| xarm_lift_medium | 800 / 20,000 | 1 | 4 / 4 | 640 / 80 / 80 |
| koch_pick_place_1_lego | 100 / 32,951 | 2 | 6 / 6 | 80 / 10 / 10 |
| aloha_static_battery | 49 / 29,400 | 4 | 14 / 14 | 41 / 4 / 4 |

Each dataset's `dataset_manifest.json` records source, revision, loading, and splits. Calling them real datasets does not replace checking each subset's collection environment. This project provides no evidence of new data collection or closed-loop execution.

**Q10 What is an episode, and what is one training sample?**

An episode is a complete trajectory. A sample at time t contains current multi-camera images, state, and observation-quality metadata; the target is an action chunk starting at t. Typical episode shapes are image `[T,C,3,Himg,Wimg]`, state `[T,S]`, and action `[T,A]`. Formal model output is `[B,8,A]`; B is batch size, not time length.

**Q11 What does each state/action dimension mean? Are 14 dimensions 14 joint angles?**

Check the dataset's feature definitions, units, and control interface. The code reads vectors and builds networks from their dimensions; it does not unify joint angles, position increments, gripper commands, or torques. Equal 14-dimensional shapes do not prove equal physical meanings. Use the dataset's exact definitions rather than guessing from tensor shape.

**Q12 [Priority] Why split episodes instead of random frames?**

Adjacent frames are highly correlated, and action chunks overlap. Frame-level splitting can place almost identical observations or targets in training and testing, producing optimistic estimates. The formal study first partitions complete episodes into disjoint sets and corrupts only training episodes. The target ratio is 80/10/10; rounding yields 41/4/4 for ALOHA.

**Q13 Which partitions are corrupted, and why keep validation/test clean?**

Only training episodes are corrupted. This asks how problematic demonstrations affect predictions under clean observations. It does not test sudden deployment-time camera failure. That requires a separate test-time fault matrix and ultimately closed-loop evaluation.

**Q14 Why compute and freeze normalization from clean training data?**

State/action standardization uses training means and standard deviations to control scale. The formal study computes these from clean training episodes and reuses them across conditions. Recomputing statistics on damaged data would change both values and their scaling, introducing a confound. Validation and test use training statistics, never statistics estimated from the test set.

**Q15 Do all entry points follow this exact protocol?**

No. The formal study explicitly provides clean validation episodes and frozen statistics. Without explicit validation data, the generic trainer randomly splits windows; `pipeline.run_training()` mainly serves the synthetic demo. The GUI splits episodes, but new training does not explicitly receive the same clean-training statistics as the formal study. Report conclusions belong to the formal path.

**Q16 How are video and numerical data loaded and aligned?**

The adapter reads LeRobot v3 metadata, parquet, and video, assembling cameras, states, and actions under common episode indices using the dataset's frame layout. It follows existing records rather than estimating physical sensor latency from image content. Index alignment is not proof of perfect synchronization at collection time. Format support must be checked against loader tests. [Loader](src/sync2act/data/lerobot.py)

**Q17 Why resize images to 64? Are all images 64×64?**

The study limits the longest edge to 64 to reduce loading, memory, and computation while retaining aspect ratio. Saved shapes are 64×64 for xArm and 48×64 for Koch/ALOHA. This is a budget choice, not a resolution optimum established by ablation; small-object detail may be lost. The general loader defaults to a maximum edge of 128.

**Q18 What happens near the end of a trajectory?**

Windows never cross episodes. If horizon=8 but only two targets remain, six positions receive placeholders and are marked invalid by a padding mask. Action loss ignores padding. Normalization can turn a zero placeholder into a nonzero value, so validity must come from the mask. [Dataset](src/sync2act/data/dataset.py)

## 3. BC-MLP, CNN, and ACT-Lite (Q19-Q32)

**Q19 [Priority] What is BC-MLP, and why include it?**

BC is behavioral cloning; MLP is a multilayer fully connected network. It predicts one action from current state and optional images, useful for debugging data, gradients, and checkpoints and as a simple baseline. However, the 270-run study excludes BC-MLP; its algorithmic baseline is ACT-Lite. The study does not demonstrate that ACT-Lite outperforms BC-MLP.

**Q20 Does BC-MLP contain a Transformer? How does it process images?**

No. A shared CNN produces a 64-dimensional vector per camera, then averages over cameras. A state MLP produces another 64-dimensional vector. Concatenated features feed the action head. State-only mode skips images. Output is consistently `[B,1,A]` to reuse the training interface.

**Q21 What is the CNN architecture? What are the two averages?**

The encoder is Conv(3→16, 5×5, stride 2), ReLU, Conv(16→32, 3×3, stride 2), ReLU, global average pooling, flatten, and Linear. Spatial averaging reduces each feature map to one scalar. BC's camera averaging combines different camera vectors. ACT-Lite only averages spatially and retains a separate token per camera. Cameras share convolution parameters. [Encoder](src/sync2act/policies/common.py), [BC-MLP](src/sync2act/policies/bc_mlp.py)

**Q22 Why a CNN rather than patch embedding or ViT?**

The small CNN extracts local features and compresses each camera into one token, shortening the Transformer sequence at the cost of spatial detail. Patch projection is another option, but only creates an initial representation; later layers must model visual relations. There is no controlled CNN/ViT comparison, so this is a lightweight design choice, not proven superiority. [ViT paper](https://arxiv.org/html/2010.11929v2)

**Q23 [Priority] What are the ACT-Lite architecture and tensor shapes?**

For ALOHA with B=2:

```text
Images [2,4,3,48,64] -> shared CNN -> [2,4,64]
State [2,14] -> Linear -> [2,1,64]
8 action queries -> expand across batch -> [2,8,64]
Concatenate -> [2,13,64] -> sinusoidal positions -> Transformer Encoder
Last 8 positions -> LayerNorm + Linear(64,14) -> [2,8,14]
```

The formal experiment uses hidden=64, one layer, four attention heads, and horizon=8. Class defaults/example YAML can use hidden=96, two layers, and horizon=16; use saved experiment settings when discussing results.

**Q24 [Priority] What is an action query? How can an input be a parameter?**

A token is a vector in a sequence, regardless of its source. Action queries are `nn.Parameter[8,64]`, concatenated with observations every forward pass. Their initial values do not depend on the sample, but Transformer outputs incorporate current observations. Loss depends on queries, so backpropagation computes their gradients and the optimizer updates them. Backpropagation alone does not update parameters or rewrite input images.

**Q25 Is an action query the Q in attention?**

No. An action query describes an input vector's role. Attention Q is obtained by applying W_Q to a layer representation. Image, state, and action-query positions all produce Q, K, and V inside self-attention; Q is not limited to the eight action-query positions.

**Q26 What does attention do here?**

Each position aggregates information via `softmax(QKᵀ/sqrt(d_head))V`. Action queries can read images, state, and other queries; observation tokens can interact too. Hidden size 64 with four heads gives 16 dimensions per head. Layers also contain feed-forward blocks, residuals, and normalization. `norm_first=True` means pre-norm. Attention weights alone are not a rigorous causal explanation.

**Q27 Why positional encoding? Can camera order change?**

Positions distinguish cameras and action queries in the concatenated sequence. There is no separate camera-identity embedding; the implementation relies on fixed camera order and positions. Changing order requires validation. Changing camera count also shifts state/query positions, so a trained model cannot be assumed to support arbitrary camera combinations.

**Q28 [Priority] Why not predict all actions from the state-token output?**

That would work: apply Linear(64,8×A) to the fused state representation and reshape. Eight queries instead provide separate integration positions for future time steps and let those positions communicate. This is an architectural choice, not a mathematical necessity. A matched-budget ablation is needed to establish which is better.

**Q29 Is it autoregressive? Does it see future information?**

It predicts eight actions in parallel without feeding earlier predictions back recursively. Forward inputs are current observations and learned queries, not future ground-truth actions. Future targets appear only in the loss. Although there is no causal mask, bidirectional attention between queries does not leak labels that were never supplied. Horizon=8 means eight predicted steps, not eight historical images.

**Q30 How does ACT-Lite differ from original ACT? What is a CVAE?**

Original ACT includes a conditional variational autoencoder path to represent demonstration variation, learns a latent distribution during training, and applies KL regularization. Its vision path uses ResNet and spatial visual features. ACT-Lite has no CVAE, latent variable, or KL loss. It uses a small CNN, one token per camera, and a Transformer Encoder with concatenated queries, without a separate Transformer Decoder. It borrows action chunking rather than fully reproducing ACT. [ACT paper](https://arxiv.org/html/2304.13705v1)

**Q31 Why predict action chunks, and why horizon 8?**

Chunks jointly supervise consecutive actions and support receding-horizon execution or temporal ensembling. Longer horizons increase output/query computation and future uncertainty; shorter horizons supply less sequence supervision. Eight is the current setting, not a proven optimum. Original evaluation used only the first step. Causal full-chunk temporal evaluation was added on 2026-10-01, but neither establishes the closed-loop benefit of executing eight steps consecutively.

**Q32 How large is the model? Is it pretrained?**

Formal ACT variants train from scratch. For ALOHA, ACT-Lite has 60,462 parameters and Quality-Full 69,294; dimensions vary across datasets. There is no large-VLA fine-tuning or pretrained visual backbone. Small models support controlled studies, but their conclusions do not directly establish large-model behavior. [ACT-Lite](src/sync2act/policies/act_lite.py), [factory](src/sync2act/policies/factory.py)

## 4. Quality-Aware ACT and losses (Q33-Q42)

**Q33 [Priority] What does Quality-Aware ACT add?**

Two separable mechanisms. At input, each camera/state token becomes `q×token + MLP([q,missing,time_offset])`, with separate image/state quality encoders. During training, action-label quality weights action loss. It reuses ACT-Lite's Transformer and action head, without extra quality tokens or an automatic quality detector.

**Q34 Why both multiply by q and add an MLP output? Does q=0 remove a token?**

Multiplication changes observation-content amplitude; the MLP learns a representation of reliability, missingness, and offset. This is not direct attention weighting, especially with subsequent LayerNorm. At q=0, the content term disappears, but the MLP can remain nonzero. The token is not removed or attention-masked, and the network is not guaranteed to ignore it.

**Q35 [Priority] Where do quality scores come from? What does Oracle quality mean?**

The program knows which faults it injected into which segments and generates metadata accordingly. Temporal quality is `exp(-abs(offset)/(2×median_frame_interval))`; missing observations and masked boundaries without a valid source get zero. Oracle means fault provenance is known. The heuristic score is neither a calibrated correctness probability nor a mathematically proven performance upper bound. Deployment requires a separate estimator and evaluation of its errors.

**Q36 [Priority] Why not use image quality directly as action-loss weights?**

Bad input and bad labels differ. With a dropped camera frame, the action label may remain correct, and state or other cameras may still teach the model. Discarding all supervision according to image quality wastes usable information. Conversely, a bad action label harms learning even when inputs are correct. Model input therefore uses only image/state quality; label quality is used only for training loss.

**Q37 What is the weighted-loss formula, and why divide by total weight?**

Average per-step errors over action dimensions, then multiply by label quality and validity:

```text
e[b,h] = mean_over_action_dim(loss(pred[b,h], target[b,h]))
w[b,h] = action_label_quality[b,h] × not_padding[b,h]
L_action = sum(w × e) / sum(w)
```

Dividing by total weight prevents the loss scale shrinking merely because valid weight decreases. With errors 1 and 9 and weights 1 and 0.1, loss is `(1+0.1×9)/1.1≈1.727`. This changes the training objective; a smaller weighted loss is not direct evidence of better prediction.

**Q38 What if all label weights in a batch are zero?**

The function returns zero action loss connected to the prediction graph, with denominator protection against NaNs. Optional smoothness loss and AdamW weight decay can still change parameters. Recording valid-weight fractions and deciding whether to skip entirely invalid batches would improve the implementation; that complete policy is not present.

**Q39 Are missing masks, padding masks, and quality the same?**

No. Padding means a chunk extends beyond the episode and has no target. Missingness means a modality is unavailable within the episode. Quality represents reliability. All action losses exclude padding, but ordinary ACT-Lite does not use action-label quality to mask zero placeholders for missing labels. Weighted variants reduce their contribution through q_action=0. Missing observations do not delete whole frames.

**Q40 Why Smooth L1, and why also smoothness loss?**

Smooth L1 is approximately quadratic for small residuals and linear for large residuals, reducing large-residual dominance relative to MSE. It does not recognize wrong labels and can still learn from bad supervision. Smoothness loss is the mean squared difference of neighboring predicted actions, encouraging continuous chunks. The formal objective is `L_total=L_action+0.01×L_smooth`. It does not prove physical stability and can suppress necessary rapid actions.

Follow-up: action loss masks padding, but smoothness loss does not mask neighboring pairs involving padded steps. It can supply gradients even with zero label weights; effects near episode ends deserve further checks.

**Q41 Is weighting just discarding difficult samples? Why not filter?**

q=0 removes action supervision; 0<q<1 softly downweights it. Filtering known missing labels is an important additional baseline: mask missing labels without temporal-quality weighting. Current conditions combine misalignment and missingness, so gains from ignoring zero placeholders versus softly weighting misalignment remain unresolved. Quality estimates biased toward easy samples could introduce selection bias.

**Q42 Does the method automatically correct misalignment? Is quality needed at inference?**

It uses quality to guide learning, without realigning data or estimating true offsets. Observation-quality models need current observation quality at inference; defaults such as all-one quality do not establish robustness to unknown faults. Label quality is training-only. Automatic correction would require separate offset estimation, alignment, and error evaluation. [Quality mechanism/loss](src/sync2act/policies/quality_act.py), [training call](src/sync2act/training/trainer.py)

## 5. Training protocol and ablations (Q43-Q52)

**Q43 [Priority] What are the six ablations? Why not compare only two models?**

Comparing ACT-Lite only with Full cannot separate input conditioning, label weighting, and extra capacity:

| Variant | Observation-quality input | Action-label weighting | Purpose |
|---|---|---|---|
| ACT-Lite | None | None | Algorithmic baseline |
| Quality-Input | Correct metadata | None | Input mechanism alone |
| Quality-Weighted-Loss | None | Correct label weights | Supervision mechanism alone |
| Quality-Full | Correct metadata | Correct label weights | Full method |
| Quality-Shuffled | Shuffled metadata | Shuffled label weights | Break quality/sample correspondence |
| Quality-Constant | Quality 1, missing/offset 0 | Equal weights | Control for extra architecture |

Shuffled applies a shared global frame permutation to metadata, not to images, states, or actions. Constant retains quality networks; its parameterization is not identical to ACT-Lite.

**Q44 How many faults does the new experiment test? Are image shifts included?**

Three modalities each have temporal misalignment and explicit missingness: six components combined into four damaged conditions plus clean. They are not six independently evaluated conditions.

| Formal condition | Target segment proportions |
|---|---|
| clean | 100% clean |
| mixed_image_damaged | 50% clean + 40% two-frame image shift + 10% image missing |
| mixed_state_damaged | 50% clean + 40% two-frame state shift + 10% state missing |
| mixed_action_damaged | 50% clean + 40% two-frame action-label shift + 10% label missing |
| mixed_three_corruptions | 20% clean + 20% image shift + 20% state shift + 25% action shift + 5% missing for each modality |

Each contiguous segment receives one component, not all faults simultaneously. Proportions are approximate at segment level; final segments shorter than 16 frames slightly change frame proportions. Image faults omit camera_index and affect all cameras in assigned segments, not an independently ablated single camera.

**Q45 [Priority] What are the baselines?**

There are two. Algorithmic comparison uses ACT-Lite's absolute test MSE under the same damaged condition. Normalized degradation uses each model's own clean-training result for the same dataset and seed: `R=MSE_damaged_train/MSE_clean_train`. Therefore 0.977× does not mean 2.3% better than ACT-Lite. The 270 runs contain 54 clean and 216 damaged runs.

**Q46 Why three seeds, and what do they control?**

Training seeds 7, 17, and 27 control initialization, data order, and mixed-segment assignment. Split seed stays 7, so these measure repeatability within one split, not three independent splits. Seeding happens before model creation, but does not guarantee bitwise equality across hardware, libraries, or resumed runs. Three seeds provide limited variability evidence.

**Q47 What optimizer, learning rate, and budget are used? Why these values?**

AdamW, learning rate 0.001, weight decay 0.0001, batch 256, 10 epochs, gradient clip 1.0, CosineAnnealingLR, Smooth L1, and lambda_smooth=0.01. This is a shared reproducible budget, not a proven optimum. Fixed settings control variables, but models can have different optimal hyperparameters. A future comparison should use equal validation-search budgets.

Follow-up: each epoch overwrites the same checkpoint path; evaluation uses the final epoch, not the best validation checkpoint. Running validation each epoch is not the same as selecting the best model.

**Q48 Walk through a batch. How do backward and step differ?**

Load a batch, move tensors to GPU, convert uint8 images to float/255, clear old gradients, predict chunks, compute action/smoothness losses, call `loss.backward()`, clip gradients, and call `optimizer.step()`. Backward computes gradients; step updates parameters. Gradients accumulate by default, hence zero_grad each batch. Clipping limits the current gradient norm, not parameter values.

**Q49 How are faults reproducible without altering originals?**

Operators copy inputs, use explicit seeds, update modality values/quality/offset/missingness, and record provenance. The mixer chooses components for 16-frame segments. It first constructs a corrupted episode, then copies assigned intervals; shift boundaries are episode boundaries, not a fresh zero-padding boundary at every segment. Models share the same seed-specific mixed data to reduce random comparison differences.

**Q50 Input and Full have more parameters. Is comparison fair?**

This needs control. Full has quality MLPs, so superiority over ACT alone cannot establish the value of quality information. Constant and Shuffled share Full's quality architecture while controlling architecture and correspondence respectively. Weighted-Loss shares ACT-Lite's base architecture, isolating weighting more directly. These reduce confounds but do not eliminate initialization, tuning-budget, or training-variation concerns.

**Q51 What should Weighted-Loss do for image-only or state-only damage?**

With all action-label qualities equal to one, weighted action loss reduces mathematically to ordinary averaging over valid positions. It has no dedicated mechanism for image/state faults, so results should resemble ACT-Lite. Floating-point summation order can cause small differences that accumulate during training; tiny gaps do not show that label weighting repaired inputs.

**Q52 What changed from the old study? Did the model improve?**

The main change was the experiment matrix: add image shift/missingness and explicit state/action missingness, and remove a separate action-delay condition largely duplicating action shift. Model core, split, and major training settings stayed unchanged. Old shift/delay both used action[t-2] away from boundaries, differing mainly in zero versus repeated-first-action padding. New and old mixed conditions differ, so this is not an architectural improvement on an identical benchmark.

Evidence: [study runner](tools/run_real_dataset_study.py), [local study configuration](runs/mixed_modality_damage_v1_3_0/study.json), [operators](src/sync2act/corruptions/ops.py), [mixer](src/sync2act/corruptions/mixed.py).

## 6. Results, statistics, and critical questions (Q53-Q62)

**Q53 [Priority] What is the strongest finding?**

For these three datasets, three seeds, and fixed split, correct action-label weighting is the clearest effective mechanism. State-quality input partly mitigates state damage; image quality has no clear benefit. Full has lower absolute MSE than ACT-Lite in 31 of 36 damaged pairs, but is not best everywhere. There is no statistical-significance or closed-loop-success proof.

**Q54 [Priority] Give concrete numbers.**

Each cell averages nine dataset-seed ratios, computed against that model's own clean-training baseline:

| Model | Image damage | State damage | Action-label damage | Mixed damage |
|---|---:|---:|---:|---:|
| ACT-Lite | 0.999 | 1.388 | 2.432 | 1.535 |
| Quality-Input | 0.995 | 1.234 | 2.394 | 2.007 |
| Quality-Weighted-Loss | 1.000 | 1.374 | 0.992 | 1.163 |
| Quality-Full | 0.996 | 1.238 | 0.977 | 1.144 |
| Quality-Shuffled | 1.004 | 1.449 | 2.463 | 1.538 |
| Quality-Constant | 0.994 | 1.303 | 2.401 | 1.500 |

Under action-label damage, ACT-Lite reaches 2.432 times its own baseline versus 0.992 for Weighted-Loss, supporting mitigation of aggregate degradation. Near/below one does not establish that faults help. Full remains above clean at 1.144 under mixed damage, so it does not eliminate all effects.

**Q55 Why does image damage barely matter? Is the model ignoring images?**

That is possible but unproven. State reliance, resolution, tasks, fault severity, or metric sensitivity may explain it. Add state-only/image-only models, shuffled images, critical-region occlusion, larger shifts, and closed-loop tests to distinguish unnecessary vision from unused vision or insensitive evaluation. These results do not justify removing cameras.

**Q56 Why are Shuffled and Constant important? What do they prove?**

Shuffled preserves the metadata collection but breaks sample correspondence. Constant retains extra quality networks while describing all data as clean. Under action damage, both degrade around 2.4× while correct weighting stays near one, supporting the importance of aligned label reliability. This is conditional evidence, not a universal proof excluding all optimization/capacity effects.

**Q57 Is Full always better than one mechanism? Why can Input be worse?**

No. Under state damage, Input's ratio 1.234 is slightly below Full's 1.238. Under mixed damage, Full 1.144 and Weighted-Loss 1.163 are close. Input reaches 2.007 without action-label quality, so bad supervision remains active. Why it is worse than ACT may also involve architecture/optimization interactions; a table alone cannot establish the cause.

**Q58 [Priority] Why does low offline MSE not mean high robot success?**

Offline evaluation always receives fixed dataset observations; wrong predictions do not change the next input. In closed loop, executed predictions alter future state and images, potentially accumulating error outside demonstrations. Success also depends on contact, constraints, recovery, and critical actions. Only environment/controller rollouts can measure that; MSE improvement is not success-rate improvement.

**Q59 What does the evaluator compute? Is temporal ensembling used?**

The original 270 runs compared denormalized `prediction[:,0]` with current demonstrated actions for MSE/MAE. The evaluator now accepts `temporal_decay`: at 0.7 it fuses overlapping eight-step predictions for each time, returns ensemble metrics, and retains the same inference pass's first-step baseline in `first_step`. Omitting the option preserves legacy behavior. Trajectory smoothness is the within-episode mean squared first difference. The field jerk is actually the mean absolute second discrete difference, without time-interval normalization, not automatically physical jerk.

For time t, fusion includes only chunks starting at s with `s≤t` and `t−s<8`, using normalized weights `0.7^(t−s)`. It never uses future observations or crosses episode boundaries. Decay 0.7 was fixed before evaluation from the module's default, not tuned on test results. Existing checkpoints, original test episodes, and frozen normalization are reused; recomputed first-step predictions are audited against historical CSVs first.

Reproduce with `python tools/evaluate_temporal_study.py --device cuda --decay 0.7`. Local evidence in `runs/mixed_modality_damage_v1_3_0/temporal_ensemble/` includes chunks, paired frame predictions, per-episode metrics, summary CSVs, and baseline audits. The report adds the same six-model overall plot: ensembled damaged-training MSE divided by ensembled clean-training MSE for the same dataset/model/seed. Separate tables show first-step MSE, ensembled MSE, differences, and percentage changes. These comparisons have different denominators.

Measured on 2026-10-01: all 270 checkpoints completed paired evaluation, and recomputed first-step predictions exactly matched historical CSVs. With decay=0.7, Quality-Full's Image/State/Action/Mixed ratios against its ensembled clean baselines are **1.0022 / 1.1360 / 1.0002 / 1.0970**. Meanwhile, the mean of 270 per-run ensemble-versus-first-step MSE changes is **+21.20%**. Lower normalized degradation does not imply lower actual MSE because ensembling also changes the clean baseline. Default evaluation remains first-step; this is not closed-loop evidence.

**Q60 Are three seeds enough? Why not treat test frames as independent samples?**

Three seeds reveal some variability but do not robustly characterize splits, tasks, or robots. Correlated trajectory frames are not thousands of independent trials. Add seeds/splits, preserve pairing, use episode-clustered or hierarchical bootstrap, and report intervals. Conditions share data/models, so even the 36 pairs are not fully independent.

**Q61 How do you avoid cross-dataset MSE scale errors? What is 31/36?**

Normalize within dataset/model/seed against the model's own clean baseline before aggregation. Do not sum raw MSE in different robots' units. 31/36 counts pairs where Full's absolute MSE is lower than ACT-Lite for the same dataset/condition/seed. It is descriptive, not a guaranteed win rate. Percentage improvements require explicit matched numerators/denominators; comparing different models' normalized ratios is not absolute-error improvement.

**Q62 What are the strongest criticisms?**

First, Oracle quality is unavailable in deployment. Second, unweighted baselines fit zero placeholders for missing action labels, making an explicit missing-mask baseline necessary. Third, mixed shift/missingness does not isolate mechanisms. Fourth, there are only three seeds and one split. Fifth, original evaluation used only first-step offline error; the added temporal evaluation is still offline. These limits bound, rather than invalidate, the controlled findings.

Evidence: [local CSV](runs/mixed_modality_damage_v1_3_0/results.csv), [local English HTML](runs/mixed_modality_damage_v1_3_0/report_en.html), [English PDF](output/pdf/sync2act_mixed_modality_damage_report_v1_3_0_en.pdf), [evaluator](src/sync2act/evaluation/evaluator.py), [temporal module](src/sync2act/evaluation/temporal.py).

## 7. GPU, CUDA, and performance (Q63-Q76)

These principles support follow-up preparation. Describe unimplemented optimizations as what you would do, not completed achievements.

**Q63 [Priority] Did you use a GPU? Which one? How long did training take?**

Formal configuration uses `device=cuda`; model and batches follow a single-GPU path. Saved `training_seconds` sum to about 5.40 hours, measured as wall-clock time around training functions, including in-loop validation/checkpoint overhead, not pure CUDA-event kernel time. Historical study configuration does not explicitly record the GPU model; today's hardware cannot establish past hardware. Check the current device with `torch.cuda.get_device_name(0)` and record it for future runs.

**Q64 Did you write CUDA? How are PyTorch and CUDA related?**

There are no custom CUDA kernels here. Python calls PyTorch convolution, attention, losses, and optimizers; the framework dispatches GPU implementations, potentially using CUDA libraries or framework kernels. Moving models/tensors to CUDA uses a GPU backend, not a handwritten GPU algorithm. Custom kernels require thread mapping, memory-access design, and operator integration.

**Q65 [Priority] How should GPU latency be measured? What limits the current latency metric?**

CUDA usually submits asynchronously relative to CPU execution; a Python timer may measure submission only. Use CUDA events for GPU elapsed time; define end-to-end boundaries and wait for required outputs. Warm up, then report batch-one P50/P95 and batched throughput. [CUDA execution](https://docs.pytorch.org/docs/2.14/notes/cuda.html#asynchronous-execution)

The evaluator synchronizes, but divides batch duration by sample count and repeats that value per sample for quantiles, without dedicated warmup. Image upload/conversion precedes timing; state/metadata upload, forward, and output transfer to CPU are inside. This is amortized batch cost under those boundaries, not pure kernel time or a robot control cycle's end-to-end latency.

**Q66 [Priority] What would you optimize first?**

Measure before rewriting operators. Separate video decoding, batch loading, CPU-to-GPU transfer, forward, loss, backward, optimizer, logging, and checkpoints. Profile time/memory and verify end-to-end behavior without profiler overhead. Candidates include num_workers=0 input loading, repeated GPU-loss-to-Python-float synchronization, and launch overhead for a small model. None has yet been established as the bottleneck. [Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

**Q67 Is a larger batch always faster? Throughput versus latency?**

Larger batches can improve GPU utilization and amortize launches, increasing samples/second, but consume memory, lengthen individual batches, and may add batching wait time in serving. Batch-one robot control prioritizes individual and tail latency. Batch size also changes optimizer-step counts and gradient statistics, so compare final error alongside throughput.

**Q68 What is mixed precision? Is it used here?**

The trainer currently does not enable AMP. Mixed precision uses FP16/BF16 for suitable operations while retaining appropriate precision elsewhere, potentially improving throughput and memory use. Autocast selects operator precision; FP16 training often uses GradScaler against gradient underflow, while BF16 usually does not need that scaling. Validate loss, gradients, and final metrics; converting every tensor to half is not sufficient. [AMP](https://docs.pytorch.org/docs/2.14/amp.html)

**Q69 What are pin_memory and non_blocking? Is transfer already overlapped with compute?**

Pinned host memory can improve host-to-device transfer, and `non_blocking=True` permits asynchronous submission under appropriate conditions. The CUDA DataLoader enables pin_memory and batch transfer uses non_blocking, but there is no explicit independent transfer stream/pipeline. These flags do not prove transfer time is hidden; inspect dependencies and timelines. [CUDA notes](https://docs.pytorch.org/docs/2.14/notes/cuda.html)

**Q70 When is a custom CUDA kernel worthwhile? Where might it help here?**

Only after profiling identifies significant GPU work that existing operators/compilers handle poorly. Candidates include fused elementwise operations, special layouts, or missing operators. Quality-weighted loss might be fusible, but may not matter for this small model. PyTorch integration must handle autograd, device, dtype, shapes, and tests. [Custom operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)

**Q71 Can you explain a simple CUDA kernel and thread indexing?**

Vector addition assigns one element per thread. Global index is block index times block size plus thread index; out-of-bounds threads do nothing.

```cpp
// Teaching snippet: pointers refer to contiguous float arrays on the GPU.
__global__ void add_vectors(
    const float* a, const float* b, float* out, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

// Host launch snippet: assumes n > 0 and data is already prepared.
int threads = 256;
int blocks = (n + threads - 1) / threads;
add_vectors<<<blocks, threads>>>(a, b, out, n);
```

`__global__` marks a GPU function launched as a kernel. A full program also needs allocation, transfers, error checks, and synchronization. This illustrative snippet was not compiled when the document was written. Understanding it is not evidence of high-performance CUDA experience. [CUDA C++ introduction](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)

**Q72 What are thread, block, grid, and warp? Why shared memory?**

A thread is a logical execution thread; threads form blocks; launched blocks form a grid. A warp is an NVIDIA execution grouping, usually 32 threads, not one CUDA core. Threads within a block can exchange/reuse shared-memory data and synchronize where valid. Shared memory introduces synchronization, resource, and access-conflict considerations; it is not automatically faster. [Kernel programming](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)

**Q73 Compute-bound versus bandwidth-bound? Why might fusion help?**

Compute-bound work is limited mainly by arithmetic throughput; bandwidth-bound work by data movement. Operation count and memory traffic help analyze this. Fusing subtraction, squaring, and weighting can reduce intermediate reads/writes and launches. Adjacent threads accessing adjacent addresses usually improves coalescing. Register pressure, occupancy, and synchronization can offset gains; measure rather than maximize fusion indiscriminately. [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

**Q74 What if GPU memory runs out? Why does training need more than inference?**

Training retains backward activations, gradients, and optimizer states. Inspect peak allocation, then consider smaller batches, resolution/token counts, mixed precision, gradient accumulation, or activation checkpointing. Disable gradients for inference. Accumulation requires correct scaling: averaging microbatch losses with different quality-weight totals is not necessarily equivalent to one large batch. Emptying a cache does not free live tensors.

**Q75 Will torch.compile or FlashAttention automatically accelerate this project?**

Possibly, not necessarily. Compilation can reduce Python overhead or fuse work but adds compilation, graph-change, and compatibility costs; distinguish cold start from steady state. FlashAttention-style implementations optimize attention traffic/execution; framework selection depends on path, hardware, and shapes. ALOHA has only 13 tokens, so attention may not dominate. No controlled comparison was run, and absence of an explicit FlashAttention call does not prove fused attention is never selected underneath.

**Q76 Why not multiple GPUs? DDP versus FSDP?**

The small model fits one GPU; communication/scheduling may outweigh benefits. DDP typically replicates a model per process, computes on different data, and synchronizes gradients. FSDP shards parameters, gradients, and optimizer states to reduce per-device state memory, with communication costs. Distributing independent dataset/model/seed experiments across GPUs may be simpler than distributing one small model. Neither optimization has been validated here. [DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)

## 8. Engineering, next steps, and closing (Q77-Q84)

**Q77 How do you verify correctness? Do tests prove the research?**

Verify layers: corruption shapes, non-mutation, seeds/metadata; model outputs and weighted-loss boundaries; checkpoint round trips; tiny-batch overfitting; splitting/normalization; and episode-local metrics. The repository has unit, integration, and GUI tests. Passing tests validates covered assertions, not research conclusions or robot safety. The original Q&A review checked source/results without retraining 270 runs or rerunning the full suite; historical test counts are not automatically current results. [Tests](tests), [models](tests/unit/test_models.py), [corruptions](tests/unit/test_data_corruptions.py), [training integration](tests/integration/test_training_pipeline.py)

**Q78 What is in a checkpoint? Is resume exactly reproducible?**

Model, optimizer, scheduler, epoch/step, history, configuration, normalization, and architecture/configuration/experiment signatures. Temporary-file writing followed by `os.replace` reduces partial-file risk. Resume restores these but not complete RNG, DataLoader-generator, or within-batch cursor state; stopping mid-epoch resumes at the next epoch. Bitwise equivalence to uninterrupted training is not guaranteed. Git commits also omit uncommitted changes, motivating source fingerprints. [Checkpoint code](src/sync2act/training/checkpoint.py)

**Q79 Why should training not freeze the GUI? How are background jobs handled?**

Training, download, and loading workers use QObject/QThread and signals for progress/completion/errors. A threading.Event requests stop, checked at batch boundaries. Workers must not arbitrarily modify main-thread UI. Some GUI evaluation still calls the evaluator directly, so not all slow work is asynchronous. Python threads do not guarantee parallel speedup for pure-Python CPU computation.

**Q80 Is a fixed seed enough? What records matter most?**

No. Record data revisions, loaded counts, episode splits, corruption assignments, normalization, model/training settings, source version, dependencies, drivers/hardware, and determinism settings. Seeds constrain randomness, not all platform differences. Several manifests/signatures exist, but historical study hardware records are incomplete. Successful execution, artifact integrity, and reproducible training results are distinct.

**Q81 If you had one week, which experiments would you add?**

Prioritize misalignment-only, missingness-only, missing-label-mask-only, and full soft-weighting controls with matched splits/budgets. This tests whether the main gain is simply not learning zero placeholders. Then perturb quality scores and inject false positives/negatives to measure sensitivity. More models alone do not resolve unanswered mechanisms.

**Q82 How would you connect this model to a real robot?**

Define camera/state interfaces, timestamps, physical action semantics, control rate, normalization, device, and latency budget. Route outputs through the correct control interface with action limits, invalid-input handling, fallback, and stop mechanisms. Define task success/failure in simulation or controlled hardware, execute closed-loop rollouts, and record recovery. The current offline code does not complete this deployment chain; outputting an action vector is not robot control.

**Q83 How could it extend to VLA or foundation models?**

Consider pretrained vision/vision-language encoders, language conditioning, more spatial tokens, and unified multi-robot actions. These add data, compute, and alignment requirements and need new baselines. The ablation idea still applies to quality in visual representations, action heads, or supervision. Current small-model results cannot simply be extrapolated, and this project provides no implemented GR00T, Isaac, or vLLM/SGLang experience.

**Q84 How would you summarize the project? What if you do not know an answer?**

One closing answer: "This project separates observation quality from supervision quality in controlled experiments. Reliable action-label weights help most; state quality partly mitigates degradation; image-quality benefits remain unresolved. Evidence is limited to Oracle metadata and offline prediction. I would next add missing-label baselines and quality-estimation-error experiments, then validate closed-loop effects."

For unfamiliar technology: "I have not personally implemented that. My understanding is that I should first do ..., and validate the effect through ...." State knowledge boundaries rather than presenting plans as completed experience. Be truthful about contributions and tool use as in Q06.

## Final check: statements to avoid

| Misleading statement | More accurate statement |
|---|---|
| I fully reproduced ACT | This is ACT-Lite without CVAE and other original paths |
| The new study has no image faults | It includes image misalignment and all-camera missingness |
| Three faults mean state shift, action shift, action delay | That is the old study; the new one organizes shifts/missingness by image/state/action |
| Quality-Input knows label quality | Model input uses observation quality; label quality enters loss |
| 0.977× means 2.3% better than ACT-Lite | It is Full's mean ratio against its own clean baseline |
| GPU training means I implemented CUDA | The project uses PyTorch's CUDA backend, without custom kernels |
| 5.40 training hours is pure kernel time | It sums recorded training-function wall-clock durations |
| Full is best under every fault | It improves most pairs, not every condition |
| Oracle is automatically detected, strictly optimal quality | Fault provenance is known; scores are heuristic, not a deployment estimator or proven bound |
| Temporal ensembling necessarily improves performance or proves stability | Offline paired evaluation separates measured error from smoothness; no closed-loop benefit follows |
| Lower MSE means higher robot success | It measures offline error; closed-loop validation is separate |

## Code reading and hands-on practice

Read Dataset → CNN → ACT-Lite → Quality-Aware → loss → experiment matrix → evaluator before the large GUI module.

If the local teaching artifact is available, run from the repository root:

```powershell
python output/interview-prep/trace_forward.py
```

It demonstrates synthetic tensor shapes, weighting, and temporal ensembling, not benchmark performance. Change cameras, horizon, and label weights; predict shapes/results before execution. Do not memorize old study results as current findings. Use this document, source, and the latest CSV.

Minimal formal-result check from the project Python environment:

```python
import pandas as pd

df = pd.read_csv('runs/mixed_modality_damage_v1_3_0/results.csv')
keys = ['dataset', 'model', 'condition', 'seed']
assert len(df) == 270 and not df.duplicated(keys).any()
table = df.groupby(['model', 'condition'])['mse_ratio_to_clean'].mean().unstack()
print(table.round(3))
```

Some links point to local artifacts. `runs/` and the optional teaching script under `output/` are ignored and are not supplied by a source-only clone. Answers derive from source review, saved artifacts, and official references; they do not replace firsthand practice.
