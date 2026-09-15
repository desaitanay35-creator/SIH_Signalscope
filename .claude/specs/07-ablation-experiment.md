# Spec: Three-Way Ablation Experiment (RGB-only vs. Frequency-only vs. RGB+Frequency Fusion)

## 1. Objective

Run the controlled, three-way ablation the Step 6 spec designed but did not execute: train and evaluate `rgb_only`, `frequency_only`, and `rgb_frequency_fusion` under identical conditions, using the existing Step 4 training pipeline and Step 5 evaluation pipeline exactly as implemented, and produce a single comparison artifact answering one question - **does the frequency-domain branch provide complementary signal for AI-generated image detection, as measured by unseen-generator ROC-AUC on the Tiny-GenImage development shard?**

This is an execution/orchestration step, not a code-architecture step: it defines *how* to run the three experiments so the comparison is scientifically meaningful, and *how* to read the result honestly. It does not modify `model/training/`, `model/evaluation/`, `model/architectures/`, or `data/` - every piece of code this step runs already exists and was verified working end-to-end in Steps 4-6 (129/129 tests passing, real CPU smoke runs against `data/manifests/genimage_dev.csv` for all three architectures, Step 5 evaluation loading all three checkpoint types).

## 2. Hypothesis

**H1 (the hypothesis under test):** Fusing frequency-domain features (log-magnitude FFT spectrum, `model/architectures/frequency_branch.py`) with the existing RGB EfficientNet-B4 baseline improves unseen-generator ROC-AUC relative to the RGB-only baseline, because frequency-domain artifacts (periodic upsampling/deconvolution signatures, radial energy falloff) carry forensic information that is at least partially independent of what an RGB-only classifier learns.

**H0 (null hypothesis):** The frequency branch provides no measurable complementary signal on this dataset - fusion performance is statistically indistinguishable from, or worse than, the RGB-only baseline.

This spec does not presuppose which is true. Per the user's explicit instruction and Step 6's own "How to interpret the experiment" section, **a single experiment on one small development shard cannot establish causality either way** - see Section 13.

## 3. Experimental variables

**Independent variable (the only thing that differs between the three runs): `model_type`.**

| Experiment | `model_type` | Model class | Input representation |
|---|---|---|---|
| A. RGB-only | `"rgb_only"` | `model.architectures.efficientnet_b4.EfficientNetB4Baseline` | `(N, 3, 224, 224)` ImageNet-normalized RGB tensor |
| B. Frequency-only | `"frequency_only"` | `model.architectures.frequency_branch.FrequencyOnlyModel` | `(N, 1, 224, 224)` log-magnitude FFT spectrum |
| C. RGB+Frequency fusion | `"rgb_frequency_fusion"` | `model.architectures.fusion_model.RGBFrequencyFusionModel` | `{"rgb": (N,3,224,224), "frequency": (N,1,224,224)}` dict |

**Dependent variables (what is measured, not controlled):** validation ROC-AUC, validation loss, unseen-generator ROC-AUC (primary; reported everywhere as **"Unseen-generator ROC-AUC (development evaluation)"** - see the naming rule in Section 9), accuracy, macro-F1, FPR at threshold, confusion matrix, per-generator ROC-AUC/accuracy/macro-F1, wall-clock training/evaluation time (optional - see Section 8).

## 4. Controlled variables

Every item the task requires, mapped to the exact mechanism that holds it fixed:

| Controlled variable | Mechanism |
|---|---|
| Same manifest | One shared `data.manifest_path` value (`data/manifests/genimage_dev.csv`) in one shared `model_config.yaml` used by all three training AND all three evaluation invocations. |
| Same resolved generator-disjoint split | Same manifest + same `SplitConfig` (see Section 5) fed into the one, unmodified `data.splitting.split_manifest()` (verified deterministic and cross-process-reproducible - see Section 11). |
| Same `SplitConfig` | `data.split.seed`, `val_fraction`, `unseen_generators`, `held_out_generator_fraction` all come from the one shared `model_config.yaml` - never duplicated or re-typed per experiment. |
| Same training seed | One shared `training_config.yaml`'s `seed` field, consumed by `set_seed()` (`model/training/seed.py`) at the top of every `run_training()` call. **Correction:** this guarantees reproducible stochastic behavior *within* each architecture (re-running the same `model_type` with the same seed reproduces identical results) and identical dataset shuffling/augmentation order across all three runs (Section 4's "Same augmentations" row) - it does **not** mean all three architectures receive numerically identical initial weights. See Section 11, item 6 for the precise, corrected statement of what is and is not shared. |
| Same `TrainingConfig`/hyperparameters | One shared `training_config.yaml`, byte-identical across all three invocations - `optimizer`, `learning_rate`, `weight_decay`, `scheduler`, `epochs`, `batch_size`, `num_workers`, `mixed_precision`, `freeze_backbone_epochs` (0), `pos_weight` (null). |
| Same train/validation sample budget | Same `max_train_samples`/`max_val_samples` value (both `null` for the real run - full split used; identical explicit values for a reduced/CPU-feasibility run - see Section 6) in the one shared `training_config.yaml`. |
| Same augmentations | `model.training.augmentation.build_train_transform()` is architecture-agnostic and untouched by Step 6 - `rgb_only` uses it via `model.training.dataset.build_train_dataset`; `frequency_only`/`rgb_frequency_fusion` use the identical transform via `model.training.dual_branch_dataset.AugmentedDualBranchPreprocessor` (which calls the same `build_train_transform` function - not a re-derived one). Both branches of a dual-branch sample derive from the SAME single augmented image (verified by `tests/test_frequency_fusion.py::test_deterministic_dual_branch_preprocessor_derives_both_tensors_from_same_image`). |
| Same number of epochs | Same `training_config.yaml`'s `epochs` field. |
| Same checkpoint selection rule | `model/training/train.py::run_training()`'s best-checkpoint rule (strictly higher validation ROC-AUC, ties keep the earlier checkpoint) is architecture-agnostic and unchanged by Step 6 - applies identically to all three `model_type` values. |
| Same threshold | `ModelConfig.threshold` (0.5) from the one shared `model_config.yaml`, resolved once per evaluation run, never tuned. |
| Same evaluation pipeline | `model.evaluation.evaluate.run_evaluation()`, unmodified by Step 6 except for one additive architecture-dispatch branch (dataset construction only) - metric computation (`model/evaluation/metrics.py`) is identical code for all three. |
| No test-set tuning | No hyperparameter, threshold, or checkpoint decision in this protocol ever reads `metrics.json["test"]` before it is written - test-split evaluation happens once, after training and checkpoint selection are already complete and frozen (Step 5's existing guarantee, unchanged). |
| No hyperparameter changes between architectures | Explicitly forbidden by this protocol (Section 6) - the one shared `training_config.yaml` is never edited between the three runs. |
| No hidden-generator information leaking into training | `data.splitting.check_generator_leakage()` (invoked inside `split_manifest()`) and `model.evaluation.evaluate._assert_unseen_generator_split_is_valid()` both run, unmodified, for every training and evaluation invocation in this protocol. |
| No modification to dataset/splitting implementation | `data/dataset_loader.py`, `data/preprocessor.py`, `data/splitting.py` are not touched by this step, at all. |

## 5. Dataset and split protocol

- **Manifest:** `data/manifests/genimage_dev.csv` (Tiny-GenImage development shard - 1,000 real / 1,000 AI-generated across ADM, BigGAN, GLIDE, Midjourney, SD15, VQDM, Wukong; SD14 = 0 images present). **This is development data, not the final SIH benchmark - see Section 14.**
- **Explicit warning, preserved from Step 5/6 and repeated here because it directly shapes every metric this experiment reports:** with the current manifest and `data.splitting.split_manifest()` (unmodified), **the `test` split contains zero real images** - `choose_unseen_generators()` only ever selects from non-real generator ids, so every real image is routed into `train`/`val`. The "unseen-generator" evaluation set this experiment's primary metric is computed over is therefore **`test`'s held-out-generator fakes paired with `val`'s real images**, not `test` alone - exactly as `model/evaluation/evaluate.py` already implements (`REAL_PAIRING_NOTE`, appended to `test.overall.notes` and every `test.per_generator[...].notes` entry on every evaluation run). This is not a defect introduced by this step; it is `data/splitting.py`'s existing, unmodified behavior, and this experiment must not obscure it. See Section 9 for how the comparison table preserves this note.
- **`SplitConfig`** (all four fields live in **exactly one** new, shared `config/ablation/model_config.yaml` - see Section 8 for why a new file rather than editing the committed `config/model_config.yaml`; there is no per-architecture copy of this file, and none may be created - see Section 6):
  - `seed: 42`
  - `val_fraction: 0.15`
  - `unseen_generators: ["BigGAN", "Midjourney"]` - **pinned explicitly**, not left to `held_out_generator_fraction`-derived selection. This is a deliberate change from the currently-committed `config/model_config.yaml` (which uses `unseen_generators: []` + `held_out_generator_fraction: 0.3`). Verified empirically (three independent process runs, see Section 11) that the current fraction-based config resolves to exactly this list against the current manifest - pinning it explicitly makes the split auditable by reading the config file alone, rather than requiring the reader to re-run `choose_unseen_generators()` to know what it resolves to, and removes any risk of the resolved set silently changing if the manifest's generator set is ever edited (a risk Step 6's spec already flagged). This does not modify `data/splitting.py` - `choose_unseen_generators()` already uses an explicit list verbatim when one is given (`data/splitting.py:45`).
  - `held_out_generator_fraction: 0.3` - kept in the file for documentation continuity even though it has no effect once `unseen_generators` is non-empty.
- **Resulting split** (verified against the current manifest with the config above): `train` = 1,457 records, `val` = 257 records, `test` = 286 records (143 BigGAN + 143 Midjourney, zero real images - `test` never contains real images by construction of `split_manifest()`; the unseen-generator evaluation set is `test` + `val`'s real images, exactly as `model/evaluation/evaluate.py` already implements and `.claude/specs/05-evaluation-pipeline.md` documents under "Implementation note: real-image pairing for the unseen-generator result"). This split is reused, unrecomputed and unmodified, by feeding the same manifest + same `SplitConfig` to all three training runs and all three evaluation runs - see Section 11 for how this is *verified*, not merely assumed.

## 6. Training protocol

**Exactly one** shared `config/ablation/model_config.yaml` and **exactly one** shared `config/ablation/training_config.yaml` (both new files, see Section 8) are used **byte-identically** across all three `python -m model.training.train` invocations - the only command-line difference between the three reportable runs is `--frequency-config` (one of three minimal, one-field files, each containing only `model_type: "..."`). **No per-architecture copy of `model_config.yaml` or `training_config.yaml` is created at any point** - not for this design, and not at implementation time. If a future change is ever needed for one architecture's run, it must be expressed as a new, explicitly-justified control variable in this spec first, never as a silently-diverged config copy.

- **Optimizer:** AdamW (`model/training/train.py::_build_optimizer`)
- **Learning rate:** `0.0001`
- **Weight decay:** `0.0001`
- **Scheduler:** CosineAnnealingLR over `epochs`
- **Batch size:** `16`
- **Epochs:** `10` (the repo default). **CPU feasibility caveat:** this machine has no CUDA (Intel i5-1334U, ~7.6 GB RAM). EfficientNet-B4 (used in both `rgb_only` and `rgb_frequency_fusion`) is the dominant per-batch cost; the frequency branch alone is comparatively fast. Before committing to a full 10-epoch run for all three experiments, time one epoch of `rgb_only` first and extrapolate. If 10 epochs proves impractical on CPU, an identical **reduced** epoch count (e.g. 3-5) may be substituted - but it must be the same value in the one shared `training_config.yaml` used by all three runs, and the resulting comparison must be labeled as a reduced-budget run in the comparison artifact (Section 8), not silently presented as equivalent to the full-budget protocol.
- **`max_train_samples` / `max_val_samples`:** `null` (full split used) for the reportable ablation run. A separate, explicitly-labeled CPU smoke-test pass (both values small, e.g. 32/16, `pretrained: false`) should be run first purely as an engineering correctness check - per Section 13, smoke-test numbers must never be reported as the ablation result.
- **`pretrained` (RGB branch, `rgb_only` and `rgb_frequency_fusion` only):** `true` for the reportable run (matches Step 4's own baseline default and Step 6's "Fair comparison" requirement that the fusion model's RGB half start from the same initialization as the RGB-only baseline); `false` for the smoke-test pass only (no network access, per Step 4/6's existing smoke-test convention). `frequency_only` has no `pretrained` option at all - `FrequencyOnlyModel`/`build_frequency_only_model()` take no such argument, since no ImageNet-equivalent pretraining source exists for FFT spectra (`model/architectures/frequency_branch.py`).
- **`freeze_backbone_epochs`:** `0` for all three (no staged unfreezing in this baseline ablation, matching Step 6's design).
- **`pos_weight`:** `null` for all three (Tiny-GenImage's real/AI-generated balance is exact; see `model/training/train.py`'s `loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)`).
- **Loss:** `nn.BCEWithLogitsLoss()` - identical code path for all three `model_type` values (architecture-agnostic; the model always returns one raw logit of shape `(N, 1)` regardless of `model_type`).
- **Joint training (fusion only):** both branches and the fusion head train jointly, end-to-end, from epoch 1, sharing one optimizer - no staged/frozen pretraining of either branch (per `.claude/specs/06-frequency-fusion.md`, "Training strategy").
- **Checkpoint selection rule:** best validation ROC-AUC (`model/training/train.py::run_training()`'s existing `if val_metrics.roc_auc is not None and (best_val_roc_auc is None or val_metrics.roc_auc > best_val_roc_auc)` rule) - identical code, applied identically regardless of `model_type`. `checkpoints/best.pt` is what Section 7's evaluation consumes; `checkpoints/last.pt` exists for resume/debugging only and is not evaluated as part of this protocol.

## 7. Evaluation protocol

One shared `config/evaluation_config.yaml` (the existing repo default - no ablation-specific override needed, since nothing in it varies per architecture) and the same `config/ablation/model_config.yaml` from Section 5, used for all three `python -m model.evaluation.evaluate` invocations.

- **Checkpoint:** each experiment's `experiments/runs/<run_id>/checkpoints/best.pt`.
- **Manifest:** `data/manifests/genimage_dev.csv` (same file, same path, all three).
- **Threshold:** `0.5`, resolved from `config/ablation/model_config.yaml`'s `model.threshold` (no `--threshold` override) - identical across all three, never tuned on any split.
- **Metrics computed** (unchanged code, `model/evaluation/metrics.py`):
  - Primary: unseen-generator ROC-AUC (`metrics.json["test"]["overall"]["roc_auc"]`)
  - Secondary: accuracy, macro-F1, FPR at threshold, confusion matrix (all in the same `["test"]["overall"]` block), validation ROC-AUC and validation loss (`["val"]["overall"]["roc_auc"]`, and the last `val_loss` entry in the training run's `metrics.jsonl`), per-generator breakdown (`["test"]["per_generator"]`, `["val"]["per_generator"]`), test sample counts (`config.json`'s `num_val_samples`/`num_test_samples`/`num_unseen_generator_eval_samples`), and wall-clock time (see Section 8 - not currently a dedicated logged field; derived externally).
- **No architecture-specific evaluation code**: `model.evaluation.evaluate.run_evaluation()` dispatches dataset construction (`build_val_dataset` vs. `build_dual_branch_val_dataset`) based on the checkpoint's own recorded `model_config["architecture"]` field - the metric computation, artifact-writing, and reporting code downstream of dataset construction is byte-identical for all three (`model/evaluation/metrics.py`, `model/evaluation/report.py` are untouched by Step 6).

## 8. Artifact/output structure

No new logging system. Each training run produces the existing Step 4 structure; each evaluation run produces the existing Step 5 structure, nested under its training run (per `.claude/specs/05-evaluation-pipeline.md`, "Output artifacts"):

```text
experiments/runs/<rgb_only_run_id>/
├── config.json                 # includes "model_type": "rgb_only", model.get_spec() -> architecture: "efficientnet_b4"
├── metrics.jsonl                # one line per epoch: train_loss, val_loss, val_accuracy, val_roc_auc, lr
├── checkpoints/{last.pt,best.pt}
└── evaluation/<eval_run_id>/
    ├── config.json               # split_config, threshold, preprocessing, training_provenance
    ├── metrics.json               # val.overall, val.per_generator, test.overall, test.per_generator
    ├── predictions.csv
    └── evaluation_report.md

experiments/runs/<frequency_only_run_id>/...        (architecture: "frequency_branch")
experiments/runs/<rgb_frequency_fusion_run_id>/...  (architecture: "rgb_frequency_fusion")
```

Plus, new for this step only, a **comparison artifact** (not a code change - a generated report, produced by hand or a short ad-hoc script at execution time, not part of `model/`):

```text
experiments/ablation/<ablation_run_id>/
├── comparison.md      # human-readable: the table from Section 9, plus per-generator tables (Section 10)
├── comparison.json    # machine-readable: the same data, one record per experiment
└── run_manifest.json  # which three run_ids/eval_run_ids this comparison was built from, and the
                        # verification described in Section 11 (proof the three splits matched)
```

This directory is new (`experiments/ablation/`) but sits under the same git-ignored `experiments/` tree the existing `.gitignore` rule (`experiments/runs/`) was written for - if implemented, the `.gitignore` pattern should be widened to `experiments/` (or a second `experiments/ablation/` line added) so comparison artifacts are never accidentally committed either; this is a one-line `.gitignore` change to flag at implementation time, not something this design-only spec makes now.

**Wall-clock time is optional, not a required measurement or acceptance criterion** (see Section 16 - it does not appear there). Neither `model/training/logging_utils.py` nor `model/evaluation/evaluate.py`'s `config.json` currently records a dedicated training/evaluation duration field. `metrics.jsonl`'s per-epoch `timestamp` field allows an approximate training duration to be derived (last epoch timestamp minus first) with no extra tooling. If a precise figure is wanted, this is a Windows/PowerShell development environment (per `.claude/specs/06-frequency-fusion.md`, "Computational constraints") - Unix `time` is not available; use PowerShell's `Measure-Command`, e.g.:

```powershell
Measure-Command {
    python -m model.training.train `
        --model-config config/ablation/model_config.yaml `
        --config config/ablation/training_config.yaml `
        --frequency-config config/ablation/frequency_config_rgb_only.yaml
}
```

or, for a quick before/after timestamp without capturing output structure, `Get-Date` before and after the command. Whichever method is used (or none), the "Training Time" column in Section 9 is filled in only when captured - its absence never invalidates the comparison.

## 9. Comparison table schema

**Naming rule, mandatory in every generated artifact (`comparison.md`, `comparison.json`, and any other report this step produces):** the primary metric column/field is always labeled **"Unseen-generator ROC-AUC (development evaluation)"** in full - never abbreviated to "Test ROC-AUC" or "ROC-AUC" alone. This makes two things explicit at the point of reading, not just in prose elsewhere: (1) it is evaluated against Tiny-GenImage development data, not a final benchmark (Section 14), and (2) - via the mandatory footnote described below - that the underlying evaluated set pairs `test`'s fakes with `val`'s real images, because `test` itself contains none (Section 5's explicit warning).

`comparison.md`'s primary table, one row per experiment, minimum columns exactly as specified, with the corrected naming:

| Experiment | Val ROC-AUC | Unseen-generator ROC-AUC (development evaluation) | Accuracy | Macro-F1 | FPR |
|---|---|---|---|---|---|
| rgb_only | `val.overall.roc_auc` | `test.overall.roc_auc` | `test.overall.accuracy` | `test.overall.macro_f1` | `test.overall.fpr` |
| frequency_only | ″ | ″ | ″ | ″ | ″ |
| rgb_frequency_fusion | ″ | ″ | ″ | ″ | ″ |

Immediately below this table, `comparison.md` must reproduce the real-image-pairing limitation as a footnote, verbatim in substance (the exact wording of `model/evaluation/evaluate.py::REAL_PAIRING_NOTE`, already present in every row's underlying `test.overall.notes`, must not be dropped when tabulating):

> *"Unseen-generator ROC-AUC (development evaluation)" pairs the `test` split's held-out-generator (BigGAN, Midjourney) images with the `val` split's real images - `data.splitting.split_manifest()` routes every real image into `train`/`val`, so `test` alone contains zero real images. See `model/evaluation/evaluate.py`'s `REAL_PAIRING_NOTE` and `.claude/specs/05-evaluation-pipeline.md`.*

("Accuracy"/"Macro-F1"/"FPR" refer to the same unseen-generator development-evaluation result, consistent with ROC-AUC being reported for the same evaluated set - the val-split equivalents are additional columns, not substitutes.)

Extended columns (all sourced from existing, unmodified evaluation artifacts - no new metric code):

| Column | Source |
|---|---|
| Val Loss | last `val_loss` entry in that run's `metrics.jsonl` |
| Confusion Matrix (TP/TN/FP/FN) | `test.overall.confusion_matrix` |
| Num Val / Num Test / Num Unseen-Eval Samples | evaluation `config.json`'s `num_val_samples` / `num_test_samples` / `num_unseen_generator_eval_samples` |
| Training Time (optional) | derived per Section 8, only if captured - never required |
| Notes | `test.overall.notes`, including the real-image-pairing note verbatim - must never be summarized away |

`comparison.json` contains the same fields as a list of per-experiment records, using the key `"unseen_generator_roc_auc_dev_eval"` (not a bare `"roc_auc"` or `"test_roc_auc"`, so the development-evaluation caveat survives into the machine-readable artifact too), each record also carrying `run_id`, `eval_run_id`, `model_type`, `checkpoint_path`, and the real-image-pairing note string, so the comparison can be regenerated or audited without re-running anything and without needing to re-read prose elsewhere to recover the caveat.

## 10. Per-generator analysis

One table per split direction, columns = the three experiments, rows = generators - sourced from each experiment's `metrics.json["test"]["per_generator"]` (unseen: BigGAN, Midjourney) and `metrics.json["val"]["per_generator"]` (seen: ADM, GLIDE, SD15, VQDM, Wukong). The unseen-generator table's title in `comparison.md` must read **"Per-generator Unseen-generator ROC-AUC (development evaluation)"**, for the same naming-consistency reason as Section 9, and each such per-generator value is itself computed as that generator's `test` rows paired with `val`'s real images (the real-image-pairing note applies identically at the per-generator level, per `model/evaluation/evaluate.py`'s existing `for metrics in test_per_generator.values(): metrics.notes.append(REAL_PAIRING_NOTE)`):

| Generator (unseen) | rgb_only ROC-AUC | frequency_only ROC-AUC | rgb_frequency_fusion ROC-AUC |
|---|---|---|---|
| BigGAN | ... | ... | ... |
| Midjourney | ... | ... | ... |

| Generator (seen, validation) | rgb_only ROC-AUC | frequency_only ROC-AUC | rgb_frequency_fusion ROC-AUC |
|---|---|---|---|
| ADM | ... | ... | ... |
| GLIDE | ... | ... | ... |
| SD15 | ... | ... | ... |
| VQDM | ... | ... | ... |
| Wukong | ... | ... | ... |

Per `model/evaluation/metrics.py::per_generator_breakdown()`'s existing, unmodified design (`.claude/specs/05-evaluation-pipeline.md`), each per-generator entry already pairs that generator's images with the real images in the same evaluated group - a naive single-generator slice (which would always be single-class and produce an undefined ROC-AUC) is never used. No new per-generator logic is written for this step; the table above is purely an extraction/tabulation of existing `metrics.json` fields across three runs.

## 11. Reproducibility requirements

**Every one of these was directly inspected and, where stated, empirically verified as part of preparing this spec (see the prior scientific-verification exchange in this project's history for the full inspection):**

1. `split_manifest()` (`data/splitting.py`) is a pure, deterministic function of `(records, SplitConfig)` - it uses only local `random.Random(config.seed)` instances, never global RNG state, and its candidate-generator ordering is `sorted()` (hash-randomization-independent). Verified: three separate, freshly-launched Python processes against the real manifest with the pinned `SplitConfig` from Section 5 produced byte-identical `unseen_generators`, split sizes, and first-record-per-split.
2. `model_type` has no code path reaching `split_manifest()` or its inputs (`model/training/train.py`: `data_config = load_data_config(model_config_path)` at line ~202 is independent of `frequency_config_path`, and `split_manifest()` at line ~208 runs before `model_type` is used at all) - the split is provably identical across all three `model_type` values given the same `--model-config`.
3. **Known gap, not fixed by this step:** `model/training/train.py`'s `config.json` does **not** log the resolved `SplitConfig` (seed/`val_fraction`/`unseen_generators`/`held_out_generator_fraction`) - this was already flagged in `.claude/specs/05-evaluation-pipeline.md` and `.claude/specs/06-frequency-fusion.md`'s Risks sections. **Mitigation for this step:** verify the three runs' splits actually matched using each run's **evaluation** `config.json` instead, which *does* log `split_config` in full (`model/evaluation/evaluate.py`, `write_config(...)`'s `"split_config": {...}` block) - this is exactly what `run_manifest.json` (Section 8) must record: a diff of the three evaluation runs' `split_config` blocks, confirmed identical, plus confirmation that `num_val_samples`/`num_test_samples` match across all three (further evidence of an identical split, since a different split would almost certainly produce different counts).
4. Smoke-test truncation/shuffling (`model/training/train.py`'s `if smoke_test:` block) only affects the truncated CPU-correctness pass, never the reportable full-budget run (`max_train_samples`/`max_val_samples` are `null` there); it uses `TrainingConfig.seed` (not `SplitConfig.seed`) for its own shuffle, so as long as the smoke pass also shares one `training_config.yaml` across all three architectures, it too is reproducible and comparable, though its numbers must never be reported as the ablation result (Section 13).
5. Given the same manifest, `model_config.yaml`, `training_config.yaml`, and only `--frequency-config` varying, `set_seed(training_config.seed)` seeds Python/NumPy/Torch identically at the start of each `run_training()` call, before any dataset/model construction - dataset shuffling order and augmentation randomness are controlled for identically across all three runs (both depend only on the seed and the deterministic sequence of shuffle/augmentation calls, which is architecture-independent).
6. **Corrected statement on weight initialization (do not overstate what "same seed" buys here):** the same seed guarantees reproducible stochastic behavior *within* each architecture (re-running `rgb_only` twice with the same seed reproduces the same run; likewise for `frequency_only` and `rgb_frequency_fusion` independently) - it does **not** mean all three architectures start from numerically identical initial weights, and it would be wrong to claim otherwise:
   - `rgb_only`'s `EfficientNetB4Baseline` and `rgb_frequency_fusion`'s RGB half **do** share the same initialization when `pretrained: true` - both load the identical ImageNet-pretrained torchvision weights (`EfficientNet_B4_Weights.IMAGENET1K_V1`), a deterministic weight file, not a seeded random draw. This equivalence holds regardless of `training_config.yaml`'s seed value.
   - `frequency_only`'s `FrequencyBranch` and `rgb_frequency_fusion`'s `FrequencyBranch` do **not** receive identical initial weights even under the same seed, because each model's constructor consumes the global PyTorch RNG in a different order: `frequency_only`'s `FrequencyOnlyModel()` draws its conv/batchnorm init immediately after `set_seed()`, while `rgb_frequency_fusion`'s `RGBFrequencyFusionModel()` first constructs `EfficientNetB4Baseline(pretrained=True)` (which itself consumes RNG draws for its initial random init, before that init is overwritten by the loaded pretrained weights) and only then constructs its own `FrequencyBranch()` - by which point the RNG stream has already advanced differently than in the `frequency_only` run. This is expected and does not violate any control in Section 4: nothing in this protocol claims or requires the frequency branch's initial weights to match between those two experiments, only that each experiment is internally reproducible and that all three share the same *training procedure* (seed, hyperparameters, data order).

## 12. Failure conditions

The comparison must be treated as **invalid or inconclusive**, and not reported as a result, if any of the following occurs:

- The three evaluation runs' `config.json["split_config"]["unseen_generators"]` are not identical (proves the runs did not share a split - almost certainly caused by an inconsistent `--model-config` between runs).
- The three evaluation runs' `num_val_samples` or `num_test_samples` differ (same implication as above).
- Any run raises `GeneratorLeakageError` (`data/splitting.py`) or `EmptyUnseenGeneratorSplitError` (`model/evaluation/evaluate.py`) - both indicate the split itself is broken, not just mismatched between runs.
- Any evaluation run's loaded checkpoint reports an `architecture` field (`model_config.json`/`checkpoint["model_config"]["architecture"]`) that does not match the `model_type` that run was supposed to train (e.g. a stale `frequency_config.yaml` left over from a previous run) - `model/training/checkpoint.py`'s `UnknownArchitectureError` will fire for a truly unrecognized id, but a *mismatched-but-valid* id (e.g. accidentally evaluating an `rgb_only` checkpoint under the `frequency_only` label) will not raise anything and must be caught by manually cross-checking `run_manifest.json` against each run's own `config.json["model_type"]`.
- Any training run does not complete all configured epochs (crashed, killed, or manually interrupted) - a comparison built from a run that stopped early at a different epoch than the other two silently confounds "epochs completed" with "model_type," violating the "same number of epochs" control.
- `training_config.yaml` or `model_config.yaml` differs in any field other than what Section 4 explicitly allows to vary (nothing should vary besides `frequency_config.yaml`'s `model_type`) between the three runs - verify by diffing the three runs' resolved config files/`config.json` contents before trusting the comparison.
- Fewer than all three experiments complete successfully - a two-way comparison does not answer this step's three-way objective.

## 13. Scientific interpretation

Applied to whatever `Unseen Test ROC-AUC` values Section 9's table actually contains - **no outcome is assumed by this spec**:

- **Fusion > RGB-only:** frequency features **may** provide complementary information for this task on this dataset. This supports, but does not prove, H1.
- **Fusion ≈ RGB-only:** the frequency branch, in its current minimal design, may be redundant given what EfficientNet-B4 already learns from RGB alone on this dataset - or the frequency representation/fusion mechanism is not yet expressive enough to surface a real signal that exists. Does not by itself refute H1 (see "Known limitations" - a single run per architecture cannot distinguish this from noise).
- **Fusion < RGB-only:** the frequency representation or the fusion mechanism may be introducing noise, or the added parameters (however few) may be creating an optimization difficulty (e.g. a harder loss landscape, gradient interference between the two branches) rather than adding usable information. Inspect the per-epoch `val_roc_auc` curves (`metrics.jsonl`) for both `rgb_only` and `rgb_frequency_fusion` before concluding either way.
- **Frequency-only unexpectedly strong** (competitive with or close to RGB-only, despite discarding all color/texture information): **this result must be treated as a possible dataset/source/compression shortcut until robustness checks support a stronger interpretation - not as evidence of genuine forensic signal.** Tiny-GenImage (and GenImage-family datasets generally) are known to carry source/compression/resolution biases that a frequency branch is exactly the kind of feature likely to exploit as such a shortcut (`.claude/specs/06-frequency-fusion.md`, "How to interpret the experiment", Case D - unchanged, reused verbatim here). No report produced from this experiment may describe a strong frequency-only result as "real"/genuine forensic detection without a dedicated follow-up robustness investigation (e.g. per-generator pattern matching known dataset artifacts, testing against resized/re-compressed variants of the same images) - that investigation is out of scope for this step, and its absence is exactly why the stronger interpretation is withheld here.

**Explicit, mandatory distinctions for any report produced from this experiment:**
- **Engineering validation** (what Steps 4-6's tests and CPU smoke runs already established): all three architectures train, checkpoint, and evaluate correctly; the pipeline is mechanically sound. This is already proven and this step does not need to re-establish it.
- **Scientific evidence** (what this step's real, full-budget run produces): one ROC-AUC number per architecture, on one small development shard, from one training seed. This is a **data point**, not a validated scientific conclusion.
- **Do not claim causality from this single experiment**, in either direction. "Fusion scored higher" is an observation; "frequency features improve generalization" is a claim this one experiment cannot support on its own (see Section 14).
- **Do not describe any number produced against `data/manifests/genimage_dev.csv` as final SIH benchmark performance.** Every artifact this step produces (`comparison.md` included) must carry the same development-shard disclaimer Step 5's `evaluation_report.md` already carries automatically (`model/evaluation/report.py::is_dev_shard()`/`DEV_SHARD_BANNER`).

## 14. Known limitations

- **Single seed, single run per architecture.** This protocol runs each architecture exactly once. It cannot distinguish "architecture A is genuinely better" from "this particular random initialization/shuffle happened to land better." A properly powered comparison would run each architecture across multiple seeds and report a variance estimate (e.g. mean ± std over 3-5 seeds) before treating any ranking as reliable - explicitly out of scope for this step, and a natural next refinement if the single-seed result looks interesting either way.
- **Small dataset.** ~2,000 images, 7 non-real generators (2 held out as unseen). Results here may not transfer to a larger, more diverse dataset - this is exactly why Tiny-GenImage is explicitly never treated as the final SIH benchmark (Section 13).
- **Fraction-derived-then-pinned unseen generators.** The pinned `unseen_generators: ["BigGAN", "Midjourney"]` (Section 5) was chosen because it's what the existing config already resolves to, not because these two generators are a principled "hardest" or "most representative" holdout - a different, equally valid holdout choice might change the result.
- **Per-image FFT normalization** (`model/training/frequency_features.py::compute_log_magnitude_spectrum`) may discard absolute spectral energy information that could itself be forensically relevant (`.claude/specs/06-frequency-fusion.md`, "Risks" - unchanged, inherited here).
- **Shared geometric augmentation** (rotation, crop-resize) feeds resampling-introduced artifacts into the frequency branch's input, for both `frequency_only` and `rgb_frequency_fusion` - an inherent tension with the requirement that both branches see the same augmented image (`.claude/specs/06-frequency-fusion.md`, "Risks" - unchanged, inherited here).
- **No persisted training-time split config** (Section 11, item 3) means this protocol's leakage-safety proof rests on the *evaluation* runs' logged split config, not the training runs' - a real but currently unavoidable asymmetry in what Step 4/5's logging captures.
- **CPU-only training** may force a reduced epoch budget for practical turnaround (Section 6) - a reduced-budget comparison is weaker evidence than the full-budget one and must be labeled as such.

## 15. Exact execution commands

Assumes three new, shared config files exist (created at implementation time - **not created by this spec**, listed under "Files to create" below) plus one distinct `--frequency-config` per experiment. This is a Windows/PowerShell development environment (per `.claude/specs/06-frequency-fusion.md`, "Computational constraints") - commands below use PowerShell syntax (backtick line continuation); Unix `time` is not used (see Section 8 - wall-clock timing is optional, and `Measure-Command` is the PowerShell equivalent if wanted):

```powershell
# One-time engineering correctness check per architecture (not a reportable result -
# see Section 13). Uses --model-config pointing at a *smoke* variant with
# pretrained: false and the manifest path set, plus --smoke-test.
python -m model.training.train --smoke-test `
    --model-config config/ablation/model_config_smoke.yaml `
    --frequency-config config/ablation/frequency_config_rgb_only.yaml

python -m model.training.train --smoke-test `
    --model-config config/ablation/model_config_smoke.yaml `
    --frequency-config config/ablation/frequency_config_frequency_only.yaml

python -m model.training.train --smoke-test `
    --model-config config/ablation/model_config_smoke.yaml `
    --frequency-config config/ablation/frequency_config_fusion.yaml

# --- The three reportable experiments (Section 6 protocol) ---
# Same two shared files (config/ablation/model_config.yaml,
# config/ablation/training_config.yaml) in all three - only
# --frequency-config differs.

# A. RGB-only
python -m model.training.train `
    --model-config config/ablation/model_config.yaml `
    --config config/ablation/training_config.yaml `
    --frequency-config config/ablation/frequency_config_rgb_only.yaml

# B. Frequency-only
python -m model.training.train `
    --model-config config/ablation/model_config.yaml `
    --config config/ablation/training_config.yaml `
    --frequency-config config/ablation/frequency_config_frequency_only.yaml

# C. RGB + Frequency fusion
python -m model.training.train `
    --model-config config/ablation/model_config.yaml `
    --config config/ablation/training_config.yaml `
    --frequency-config config/ablation/frequency_config_fusion.yaml

# --- Evaluation (repeat once per resulting <run_id>, printed by each run above) ---

python -m model.evaluation.evaluate `
    --checkpoint "experiments/runs/<rgb_only_run_id>/checkpoints/best.pt" `
    --manifest data/manifests/genimage_dev.csv `
    --model-config config/ablation/model_config.yaml

python -m model.evaluation.evaluate `
    --checkpoint "experiments/runs/<frequency_only_run_id>/checkpoints/best.pt" `
    --manifest data/manifests/genimage_dev.csv `
    --model-config config/ablation/model_config.yaml

python -m model.evaluation.evaluate `
    --checkpoint "experiments/runs/<rgb_frequency_fusion_run_id>/checkpoints/best.pt" `
    --manifest data/manifests/genimage_dev.csv `
    --model-config config/ablation/model_config.yaml
```

`--eval-config` and `--threshold` are deliberately omitted from every evaluation command - both resolve to their shared repo defaults, satisfying "same evaluation pipeline" and "same threshold" by construction rather than by needing to be repeated identically three times. To capture wall-clock time for any command above (optional - see Section 8), wrap it in `Measure-Command { ... }` instead of prefixing with `time`.

**Files to create** (implementation time - not created by this spec, which creates only itself):
- `config/ablation/model_config.yaml` - the **one, authoritative** shared model/split config for all three reportable experiments: Section 5's `data.split`/`data.manifest_path`, `model.pretrained: true`. No per-architecture variant of this file exists.
- `config/ablation/model_config_smoke.yaml` - a second, still-shared-across-all-three-architectures file, identical to the one above except `model.pretrained: false` (network-free correctness checks only, per Section 6's smoke-test note) - this is not a per-architecture copy; it is one file used by all three smoke-test invocations, differing from the reportable-run config only in `pretrained` for the reason stated.
- `config/ablation/training_config.yaml` - the **one, authoritative** shared training config for all three reportable experiments: Section 6's hyperparameters, `max_train_samples`/`max_val_samples: null`. No per-architecture variant of this file exists; `--smoke-test` overrides its epoch/batch/sample fields at runtime for the smoke pass without needing a second copy.
- `config/ablation/frequency_config_rgb_only.yaml`, `frequency_config_frequency_only.yaml`, `frequency_config_fusion.yaml` - each one line, `model_type: "..."` only. These three are the **sole** point of difference between the three experiments' configuration.
- A short comparison-assembly script (reads the three evaluation runs' `metrics.json`/`config.json` and writes `comparison.md`/`comparison.json`/`run_manifest.json` per Sections 8-10) - not part of `model/`, since it is a one-off experiment-orchestration artifact, not reusable pipeline code; a reasonable location is `experiments/ablation/` itself or a `scripts/` directory, to be decided at implementation time.

**Files to modify:** `.gitignore` (widen the `experiments/runs/` rule to also cover `experiments/ablation/`, or add a second line - one-line change, at implementation time).

**Files NOT to modify:** everything under `model/`, `data/`, `config/model_config.yaml`, `config/training_config.yaml`, `config/frequency_config.yaml` (the committed repo defaults, left untouched - the ablation uses its own `config/ablation/` copies instead), `backend/`, `app/`, `ui/`, `explainability/`.

## 16. Acceptance criteria

These are **process/engineering** acceptance criteria for running the ablation correctly - not a claim about which architecture wins:

- All three training runs complete every configured epoch without error.
- All three runs' evaluation `config.json["split_config"]` blocks are identical (Section 11/12).
- All three runs' evaluation `config.json["num_val_samples"]`/`["num_test_samples"]` are identical.
- Each run's checkpoint (`checkpoints/best.pt`) reports the `architecture` id matching its intended `model_type` (`efficientnet_b4` / `frequency_branch` / `rgb_frequency_fusion` respectively).
- `model.evaluation.evaluate` successfully loads and evaluates all three checkpoints without modification to any evaluation code.
- `comparison.md`/`comparison.json` are produced, containing at minimum the Section 9 table (all three rows populated, no missing cells) and the Section 10 per-generator tables.
- Every generated report/artifact that touches `genimage_dev` data carries the development-shard disclaimer.
- No test-split data was read before every training run's checkpoint was already selected and frozen (true by construction of the unmodified Step 4/5 code - verify no ad-hoc code written for this step violates it, e.g. the comparison-assembly script must only read already-written `metrics.json` files, never re-run evaluation with different parameters after inspecting a result).
- The final report explicitly states the Section 13 interpretation for whatever outcome actually occurred, using the "may"/"flag for investigation" language specified there - not a stronger causal claim.
- No files under `model/`, `data/`, `backend/`, `app/`, `ui/`, `explainability/`, or the committed `config/*.yaml` defaults were modified to run this experiment.
