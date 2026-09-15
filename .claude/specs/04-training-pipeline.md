# Spec: Training Pipeline

## Overview

This is Step 4 of SignalScope's ML foundation: the training pipeline that turns the existing EfficientNet-B4 architecture (Step 2) and generator-disjoint manifest/dataset infrastructure (Steps 1 and 3) into a trained binary real-vs-AI-generated classifier. It introduces a `model/training/` package (manifest-driven `Dataset`/`DataLoader` construction, augmentation, loss, optimizer, LR scheduler, train/validation loops, checkpointing, and logging) plus a CLI entrypoint runnable both as a tiny CPU smoke test and, unchanged, as a full GPU training run later. This spec defines the design only; no training code is implemented here, and no run of it — smoke test or otherwise — occurs against the 2,000-image Tiny-GenImage development shard, which is a development/debugging shard only and must never be reported as the final SIH benchmark.

## Depends on

- Step 1: `data/dataset_loader.py` (`DatasetRecord`, `SignalScopeDataset`, manifest load/save), `data/splitting.py` (generator-disjoint `split_manifest`, `check_generator_leakage`), `data/preprocessor.py` (`ImagePreprocessor`).
- Step 2: `model/architectures/efficientnet_b4.py` (`build_model`, `EfficientNetB4Baseline`, `get_model_spec`), `config/model_config.yaml` + `config/settings.py` (`ModelConfig`, `DataConfig`, `SplitConfig`).
- Step 3: `data/genimage_ingest.py` and the resulting `data/manifests/genimage_dev.csv` (1,000 real / 1,000 AI-generated across ADM, BigGAN, GLIDE, Midjourney, SD15, VQDM, Wukong; SD14 = 0 images) — used only as the CPU smoke-test fixture, never as the reported benchmark.
- All 47 currently-passing ML tests must remain green; this step must not modify `data/`, `model/architectures/`, or `config/` in ways that break them.

## Model / Architecture Changes

No model architecture changes. The training pipeline consumes `build_model()` / `EfficientNetB4Baseline` exactly as defined in `model/architectures/efficientnet_b4.py` (single raw logit, `sigmoid(logit) = P(ai_generated)`, no thresholding inside the model). Training behavior (loss, optimizer, scheduler, freezing) is described below under Hyperparameters; inference behavior is unchanged — `forward()` still returns a raw logit and thresholding stays in the evaluation/inference layer.

## Data Changes

No changes to manifest format, preprocessing, or splitting logic. The training pipeline is a *consumer* of `data.splitting.split_manifest`, not a new splitting method:

- Every training run loads one manifest (`config.data.manifest_path`, no hardcoded paths), calls `split_manifest(records, split_config)`, and trains only on the `train` split, validates only on the `val` split. The unseen-generator test split may be produced by split_manifest(), but Step 4 must not construct a Dataset/DataLoader from it or consume its image paths, labels, or generator IDs. The test split is reserved exclusively for Step 5+ evaluation. It must never be touched by the training loop, LR scheduler, early stopping, or checkpoint selection.
- `check_generator_leakage` (already enforced inside `split_manifest`) is treated as a hard precondition: if it raises, training must abort before any epoch runs — no catch-and-continue.
- Duplicate-image protection: reuse `data/genimage_ingest.py`'s existing dedupe/verify pattern as precedent, but add a lightweight manifest-level check at training-pipeline load time — assert no `image_path` value is duplicated within the loaded manifest (a train/val leak-by-duplicate-row bug is otherwise invisible to `split_manifest`, which operates on whatever rows it's given). This is a new, small validation function in `model/training/`, not a change to `data/`.
- The `feature/ml-foundation` `data/manifests/genimage_dev.csv` (2,000 images) is explicitly a **development shard** for pipeline correctness (smoke tests, debugging convergence), not a benchmark dataset. All training configs and log output must labeled `dataset: genimage_dev (development shard, not SIH benchmark)` so results are never later mistaken for a reportable number.

## API / Inference Changes

No API changes. `app/` is out of scope for the ML foundation phase per `CLAUDE.md`.

## Explainability Changes

No explainability changes. `explainability/` is untouched by this step.

## Evaluation Plan

This step defines the validation-loop metric *interface* only (enough to drive checkpoint selection and early stopping) — it does not implement the full SIH evaluation suite (that is a later step, `model/training/evaluate.py` invoked via `python -m model.training.evaluate`, already named in `CLAUDE.md`'s Commands section).

- Validation loop (every epoch) computes: validation loss (BCEWithLogits), validation accuracy at threshold 0.5, and validation ROC-AUC using sklearn.metrics.roc_auc_score from raw sigmoid probabilities vs. labels.
- **Best-checkpoint criterion is validation ROC-AUC** (ties broken by lower validation loss), computed only on the `val` split of *seen* generators — never on the unseen-generator `test` split and never on any hidden SIH test set.
- Full reporting (macro-F1, confusion matrix, FPR at threshold, unseen-generator ROC-AUC, calibration) belongs to the separate evaluation step and is out of scope here; this spec only guarantees the training loop exposes per-epoch (probabilities, labels, generator ids) so that evaluation step can compute them without retraining.
- Primary success criterion for the *overall* SignalScope effort remains unseen-generator ROC-AUC (computed later, on `test`) — this step's job is only to produce a checkpoint worth evaluating that way.

## Files to change

- `.gitignore` — ignore generated ML checkpoints and run artifacts while keeping experiment configurations and selected small results trackable.

Add:

```gitignore
# ML experiment artifacts
experiments/runs/
model/weights/
*.pth
*.pt
*.ckpt

## Files to create

- `model/training/__init__.py`
- `model/training/config.py` — `TrainingConfig` dataclass (epochs, batch_size, lr, weight_decay, optimizer, scheduler, seed, num_workers, checkpoint_dir, log_dir, etc.) with a `load_training_config(path)` YAML loader, following the same pattern as `config/settings.py`. Kept separate from `config/model_config.yaml` (which stays architecture/data-focused) to avoid conflicting duplicate model settings.
- `config/training_config.yaml` — default training hyperparameters (not `config/model_config.yaml`, per the "do not create conflicting duplicate model settings" instruction — this is a new, distinct config file for run-specific hyperparameters only).
- `model/training/augmentation.py` — `build_train_transform()` / `build_val_transform()`, conservative forensics-safe augmentation (see Augmentation Strategy).
- `model/training/dataset.py` — thin wrapper around `data.dataset_loader.SignalScopeDataset` that injects the augmentation transform (train) or the existing deterministic `ImagePreprocessor` only (val), plus the duplicate-`image_path` manifest check described above.
- `model/training/engine.py` — `train_one_epoch(...)`, `validate(...)` functions (forward/backward, loss, optimizer step, metric accumulation). Pure functions, no CLI/argparse.
- `model/training/checkpoint.py` — `save_checkpoint(...)` / `load_checkpoint(...)` implementing the checkpoint format defined below.
- `model/training/seed.py` — `set_seed(seed: int)` helper (Python `random`, `numpy`, `torch`, `torch.backends.cudnn` determinism flags).
- `model/training/train.py` — CLI entrypoint (`python -m model.training.train --config ... --smoke-test`), wires manifest → split → datasets → dataloaders → model → loss → optimizer → scheduler → loop → checkpoint → logs. Matches the `python -m model.training.train` command already documented in `CLAUDE.md`.
- `model/training/logging_utils.py` — writes per-epoch JSON-lines training logs (loss, val metrics, LR, wall-clock time, git-independent run id) under `experiments/runs/<run_id>/`.
-- `experiments/` — new top-level directory for experiment metadata and generated run artifacts. Tracked files should contain documentation, configurations, and selected small results; generated checkpoints and run artifacts live under `experiments/runs/`.
- `tests/test_training.py` — CPU-fast unit tests: seed determinism, duplicate-manifest-row rejection, one forward/backward step on a tiny synthetic manifest, checkpoint save/load round-trip, best-checkpoint selection logic on synthetic metrics. Must not require network access (`pretrained=False`) or the real GenImage shard.

**## New dependencies**

- `scikit-learn>=1.4.0` — for `sklearn.metrics.roc_auc_score` in the validation loop. This avoids hand-rolling ROC-AUC.

- No other new dependencies are introduced by Step 4. `torch`/`torchvision` are already installed in the ML environment, while `pyyaml`, `numpy`, and `pyarrow` are already used by existing modules.

- The existing dependency file is `backend/requirements.txt`. Step 4 must add `scikit-learn>=1.4.0` there. Do not create a separate root `requirements.txt` or arbitrarily add PyTorch/Torchvision versions in this step.

## Rules for implementation

- Do not use the hidden SIH test set for training or tuning.
- Do not introduce data leakage between train and validation/test splits — reuse `split_manifest`/`check_generator_leakage` as-is; do not reimplement splitting logic inside `model/training/`.
- Reject any manifest containing duplicate `image_path` rows before splitting/training begins.
- Prioritize unseen-generator ROC-AUC over raw training accuracy; select checkpoints by validation ROC-AUC, never by training-set metrics and never by unseen-generator/test metrics.
- Use a single deterministic seed (`TrainingConfig.seed`, default matching `config/model_config.yaml`'s `data.split.seed: 42` for consistency) applied via `model/training/seed.py` before dataset shuffling, model init (when `pretrained=False`), and DataLoader worker seeding.
- Keep model architecture configuration in `config/model_config.yaml` and training/run hyperparameters in the new `config/training_config.yaml` — do not duplicate `pretrained`, `label_mapping`, `image_size`, or normalization values into the new file; `model/training/train.py` reads both.
- Never hardcode dataset paths, checkpoint paths, or log paths — all come from `TrainingConfig`/`DataConfig`.
- Record every run's full resolved configuration and per-epoch metrics under `experiments/runs/<run_id>/`.
- Do not claim an improvement without a measured comparison (this step does not report results — it only makes results reproducible and comparable for the evaluation step).
- Do not make absolute claims such as "this image is definitely AI-generated" anywhere in logs, docstrings, or run summaries — use "likely AI-generated" / probability language, consistent with `PROBABILITY_MEANING` in `efficientnet_b4.py`.
- Do not analyze or identify real people; do not build political or event-claim detection features (not applicable to this step's scope, stated for completeness).
- Do not modify application/backend source code, `ui/`, `explainability/`, `data/`, `model/architectures/`, or `config/model_config.yaml`.

- The only permitted change under `backend/` in Step 4 is adding `scikit-learn>=1.4.0` to `backend/requirements.txt`. No backend Python source files may be modified.

## Hyperparameters (baseline, `config/training_config.yaml` defaults)

- **Optimizer:** AdamW, `lr=3e-4` for a freshly-initialized head-only warmup phase is unnecessary in the simplest baseline — instead use a single discriminative-free AdamW at `lr=1e-4`, `weight_decay=1e-4`. Rationale: EfficientNet-B4 is ImageNet-pretrained; a moderate flat LR fine-tunes the whole network without the added complexity of parameter groups in the first experiment.
- **Backbone freezing:** none in the baseline (full fine-tuning). Staged unfreezing (freeze backbone N epochs, then unfreeze) is a documented future option in `TrainingConfig` (`freeze_backbone_epochs: int = 0`) but not exercised by the default config — keeping the first experiment simple and reproducible per the task's explicit instruction.
- **Scheduler:** `torch.optim.lr_scheduler.CosineAnnealingLR` over `epochs`, no warmup in the baseline (`ReduceLROnPlateau` on validation loss is noted as a documented alternative in `TrainingConfig` but cosine is the default for reproducibility — no dependency on when a plateau is detected).
- **Loss:** `nn.BCEWithLogitsLoss()`, no `pos_weight` in the baseline — the 2,000-image dev shard is exactly balanced (1,000/1,000). `TrainingConfig.pos_weight: Optional[float] = None` is the extension point: when a larger, imbalanced dataset arrives, `pos_weight = n_negative / n_positive` (computed from the train split's own label counts, never from val/test) is passed straight into `BCEWithLogitsLoss(pos_weight=...)` with no other code change.
- **Epochs:** default 10 for a "real" CPU-plausible run on the dev shard; smoke test uses 1–2 epochs on a tiny subset (see Smoke-test plan).
- **Mixed precision:** `torch.cuda.amp.autocast` + `GradScaler`, gated behind `torch.cuda.is_available()` — a no-op context manager on CPU (this machine), automatically active on a future GPU without code changes.

## DataLoader configuration

- **Batch size:** default 16 on CPU (8 GB RAM, i5-1334U — EfficientNet-B4 at 224×224 is memory-heavy per sample; 16 keeps a forward+backward pass well within budget). `TrainingConfig.batch_size` is overridable; a GPU run can raise it (e.g. 32–64) via config, no code change.
- **Shuffle:** `True` for train (fresh shuffle per epoch, seeded via the DataLoader `generator=torch.Generator().manual_seed(seed)` for reproducibility), `False` for val (deterministic order).
- **num_workers:** default `0` on Windows/CPU. Windows multiprocessing DataLoader workers use `spawn` and re-import the training script per worker, which is slow to start up and easy to misconfigure (must guard with `if __name__ == "__main__":`) for marginal benefit on an already CPU-bound single-process training loop on this hardware; `TrainingConfig.num_workers` is overridable (e.g. 4–8 on a Linux GPU box).
- **pin_memory:** `False` by default (irrelevant / not beneficial without CUDA); set `True` automatically when `torch.cuda.is_available()`.
- **drop_last:** `True` for train (keeps batch-norm statistics and batch shapes stable across the epoch, avoids a possibly-tiny final batch); `False` for val (every validation example must be scored).
- **persistent_workers:** `False` when `num_workers == 0` (the flag is invalid otherwise); `True` when `num_workers > 0` to amortize worker startup across epochs on GPU hosts.
- **Memory constraint note:** on this 8 GB machine, `TrainingConfig` should support an optional `max_train_samples` / `max_val_samples` override (used only by the smoke test) so a full-shard forward pass is never accidentally attempted during CI/dev iteration.

## Augmentation strategy

Training and validation use strictly different transform pipelines. Training augmentation is applied to the RGB PIL image first, followed by the existing deterministic `ImagePreprocessor` for resize to 224×224 and ImageNet normalization. Validation applies the existing `ImagePreprocessor` directly with no randomized augmentation. Augmentation must never be applied after normalization.

Training flow:

PIL RGB image
→ forensics-safe randomized augmentation
→ existing ImagePreprocessor
→ resize to 224×224
→ ImageNet normalization
→ tensor

Validation flow:

PIL RGB image
→ existing ImagePreprocessor
→ resize to 224×224
→ ImageNet normalization
→ tensor

- Validation: ImagePreprocessor.preprocess() exactly as-is today. The unseen-generator test split is not constructed or consumed by Step 4.
- **Training augmentation (conservative, forensics-safe):**
  - Random horizontal flip (p=0.5) — safe, does not alter frequency-domain or compression artifacts.
  - Small random rotation (±10°) — mild geometric variety without destroying local artifact structure.
  - Random crop-and-resize with a high minimum scale (e.g. `scale=(0.9, 1.0)`) rather than aggressive `RandomResizedCrop` defaults — avoids over-cropping the kind of global frequency artifacts that distinguish generators.
  - Explicitly **excluded**: heavy color jitter, JPEG re-compression augmentation, Gaussian blur/noise, cutout/random-erasing, and mixup/cutmix. These either destroy or synthetically fabricate the compression/frequency artifacts that are the actual detection signal, which would teach the model to ignore the evidence it needs (and could hide behind explanations that are not grounded in real model evidence, per the XAI/ethics rules).
  - Augmentation strength itself is a first candidate for the `experiment` skill/workflow once a baseline number exists — not tuned in this spec.

## Checkpointing

Checkpoint format (a single `torch.save` dict, not just `state_dict()` — unlike `EfficientNetB4Baseline.save_checkpoint`, which intentionally saves architecture-only weights for inference; this is the separate *training* checkpoint used to resume/reproduce a run):

```text
{
  "model_state_dict": ...,
  "optimizer_state_dict": ...,
  "scheduler_state_dict": ...,
  "epoch": int,
  "best_val_roc_auc": float,
  "val_metrics_history": [...],
  "model_config": get_model_spec(...) dict,
  "training_config": TrainingConfig as dict,
  "seed": int,
  "torch_rng_state": ...,          # for exact resume
  "numpy_rng_state": ...,
  "python_rng_state": ...,
}
```

- **Best-checkpoint rule:** after each epoch's validation pass, if `val_roc_auc > best_val_roc_auc` (strict improvement; ties keep the earlier, lower-val-loss checkpoint), save/overwrite `experiments/runs/<run_id>/checkpoints/best.pt`. Always additionally save `last.pt` every epoch (for resume-on-crash), separate from `best.pt`.
- Test-set metrics are never computed during training and therefore cannot influence checkpoint selection.
- `model/training/checkpoint.py` exposes two responsibilities:

  - `save_checkpoint(...)` / `load_training_checkpoint(...)` — save and restore the complete training state required for resume/reproducibility: model, optimizer, scheduler, epoch, best validation metric, training history, and RNG states.

  - `load_model_from_checkpoint(...)` — convenience helper that constructs the EfficientNet-B4 architecture and loads only `model_state_dict` for downstream evaluation/inference.

## Reproducibility

- One `TrainingConfig.seed` seeds Python `random`, `numpy`, and `torch` (CPU and, when present, CUDA) at the start of `train.py`'s `main()`, before manifest split, dataset construction, model init, and DataLoader creation.
- `torch.backends.cudnn.deterministic = True` / `benchmark = False` set when CUDA is present (no-op on this CPU machine, forward-compatible for GPU training).
- The DataLoader's shuffling uses an explicit `torch.Generator` seeded from `TrainingConfig.seed`, not global RNG state, so train-order is reproducible independent of what else touches global RNG.
- The full resolved config (model + training + data split config, git-independent) is dumped to `experiments/runs/<run_id>/config.json` at run start, so any past run's exact settings can be inspected without re-deriving them from code.

## CPU/GPU behavior

- All device-specific code paths key off `torch.cuda.is_available()` — no hardcoded `"cpu"`/`"cuda"` strings inside the training loop beyond that single check, resolved once into a `device` variable passed everywhere.
- Mixed precision, `pin_memory`, and `persistent_workers` all degrade to safe CPU defaults automatically (see above) rather than requiring a separate CPU-only code path.
- Batch size, `num_workers`, and epoch count are the only values expected to change between this laptop and a future GPU box, and all three are `TrainingConfig` fields, not constants.

## Smoke-test plan

`python -m model.training.train --config config/training_config.yaml --smoke-test` (a boolean flag, not a separate config file) does the following, without requiring any change to the main code path:

1. Loads `data/manifests/genimage_dev.csv`, splits via `split_manifest`.
2. Truncates train/val to `TrainingConfig.max_train_samples` / `max_val_samples` (small defaults, e.g. 32/16) when `--smoke-test` is passed.
3. Forces `epochs=1` (or 2), `batch_size=4`, `num_workers=0` for the smoke run.
4. Runs one full train+validate epoch, asserting:
   - training loss is finite;
   - validation loss is finite;
   - validation ROC-AUC is finite when both classes are present;
   - `last.pt` is written and reloadable;
   - `best.pt` is written when validation ROC-AUC is successfully computed.
5. This spec does not execute the smoke test — it only defines it as the acceptance mechanism the eventual implementation must satisfy, per the task instructions.

## Future full-training plan

Once the pipeline is implemented and smoke-tested, and once a larger/full GenImage (or equivalent) dataset and its own generator-disjoint manifest are available (a separate, later ingestion step — not this one), the same `model/training/train.py` runs unchanged against a GPU-backed `config/training_config.yaml` override (larger batch size, `num_workers>0`, more epochs, possibly `pos_weight` for class imbalance, possibly staged backbone unfreezing if full fine-tuning underperforms). The 2,000-image dev shard result from this step is never substituted for that later, larger run when reporting SIH numbers.

## Acceptance criteria

- `model/training/` package is importable with no import-time side effects (no network access and no file I/O). Model weight loading, including any pretrained-weight download, may occur only when `build_model(pretrained=True)` is explicitly called during model construction.
- `python -m model.training.train --config config/training_config.yaml --smoke-test` completes on this CPU machine in well under a few minutes, produces `experiments/runs/<run_id>/checkpoints/{last.pt,best.pt}` and `experiments/runs/<run_id>/config.json` plus a per-epoch metrics log.
- `tests/test_training.py` passes together with the existing 47 ML tests (total suite green, no regressions) using `pretrained=False` and a small synthetic manifest fixture — no network access, no dependency on the real GenImage shard.
- A manifest containing a duplicate `image_path` value is rejected before training starts, with a clear error.
- Checkpoint save/load round-trip restores model, optimizer, scheduler, epoch, best validation metric, and training history correctly. A resumed run must be reproducible within the project's documented floating-point tolerance under the same seed and configuration.
- No code path in model/training/ constructs a Dataset/DataLoader from or consumes records belonging to the unseen-generator test split. It must be possible to statically verify that test records never reach train_one_epoch, validate, the scheduler, or checkpoint-selection logic.
- No files under `app/`, `ui/`, `backend/`, `explainability/`, `data/`, or `model/architectures/` are modified.

## Risks / deferred decisions

- `backend/requirements.txt` is the existing project dependency file. Step 4 adds `scikit-learn>=1.4.0` there. PyTorch/Torchvision versions are intentionally not changed in this step because the verified ML environment already provides the required CPU builds.
- **AdamW LR of `1e-4` for full fine-tuning is a reasonable default, not an empirically-tuned value** — the first real (non-smoke) run on the dev shard should be treated as a sanity/convergence check, not a benchmark result, and LR/schedule choices are natural candidates for the `experiment` skill afterward.
- **`CosineAnnealingLR` vs `ReduceLROnPlateau`**: cosine was chosen for determinism/reproducibility (no dependency on noisy per-epoch plateau detection on a small dev shard); this can be revisited once a larger dataset makes plateau-based scheduling more meaningful.
- **Class-weighting formula** (`pos_weight = n_negative / n_positive`) is specified but not yet needed/tested against a genuinely imbalanced manifest — worth a dedicated unit test once such a manifest exists.
