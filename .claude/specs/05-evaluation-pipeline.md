# Spec: Evaluation Pipeline

## Overview

Step 5 adds the SignalScope evaluation pipeline: given an already-trained, already-selected checkpoint (produced by Step 4's `model/training/train.py`), it loads the frozen model, runs batched inference over a generator-disjoint test split, and reports the SIH-required metrics (ROC-AUC, accuracy, macro-F1, confusion matrix, FPR at threshold) both overall and per generator — with unseen-generator generalization as the headline number. It also independently reports validation-split ("seen generator") metrics from the same frozen checkpoint, so the two questions central to SignalScope — "how well does it classify seen generators?" vs. "how well does it generalize to unseen ones?" — are answered side by side, from the same run, without ever letting the test split influence any decision that was or could be made about the model.

This is a pure *reporting* step: no training, no checkpoint selection, no threshold tuning, no calibration. Those all happened (or will happen) elsewhere; Step 5 measures the result.

## Goals

- Load a fixed Step 4 checkpoint and reproduce, byte-for-byte, the SIH-required metric set against a generator-disjoint test split.
- Make unseen-generator evaluation the primary, unambiguous headline result, with per-generator breakdowns that are actually meaningful (see the per-generator design note below).
- Guarantee, by construction, that the test split is never used for any decision (checkpoint, threshold, hyperparameters, augmentation, architecture, calibration).
- Produce reproducible, self-describing run artifacts that can never be mistaken for a final SIH benchmark number when run against `data/manifests/genimage_dev.csv`.
- Reuse every existing reusable piece (splitting, preprocessing, dataset construction, checkpoint loading, ROC-AUC) rather than re-deriving it.

## Non-goals

- No threshold optimization or calibration (Section: Threshold policy — frozen at 0.50, sourced from existing config).
- No macro-averaged or per-class calibration, temperature scaling, or Platt scaling.
- No confusion-matrix plotting/visualization dependency by default (optional, off by default — see Dependencies).
- No frequency-domain features, ensembling, robustness testing, Grad-CAM, or generator attribution — Step 5 only defines the metric contract those steps will later be measured against.
- No changes to `backend/`, `app/`, `ui/`, `explainability/`, `data/`, `model/architectures/`, `config/model_config.yaml`, or `backend/requirements.txt`.
- No re-derivation of the Tiny-GenImage dev shard as a benchmark number — its use is explicitly a correctness check, not a result.

## Existing components to reuse (no duplication)

- `data.dataset_loader.load_manifest` / `DatasetRecord` — manifest loading, already validates label/generator consistency.
- `data.splitting.split_manifest` / `check_generator_leakage` / `choose_unseen_generators` — the *only* source of train/val/test assignment. Step 5 must call `split_manifest` itself (same as `model/training/train.py` does) and consume only the `test` split (plus, separately, `val` for the seen-generator report) — never write a second splitting implementation.
- `data.preprocessor.ImagePreprocessor` — resize/normalize, single source of truth for input size and normalization, already shared by training.
- `model.training.dataset.build_val_dataset` — deterministic, no-augmentation dataset construction. This is *exactly* what both the val-split ("seen") and test-split ("unseen") evaluation need — no new dataset-wrapping code should be written; both eval datasets are built by calling this same function with different record lists.
- `model.architectures.efficientnet_b4.EfficientNetB4Baseline`, `get_model_spec` — reused indirectly through `model.training.checkpoint.load_model_from_checkpoint`, never reconstructed directly.
- `model.training.checkpoint.load_model_from_checkpoint(path, map_location)` — already builds the architecture (`pretrained=False`) and loads `model_state_dict` from a Step 4 training checkpoint. This is the *only* way Step 5 loads a model — it must not duplicate `EfficientNetB4Baseline(...)` construction.
- `model.training.engine`'s ROC-AUC implementation — self-contained (numpy, no scikit-learn), already unit-tested (`tests/test_training.py`). Reused via one small, additive change described under Files to change.
- `config.settings.load_model_config` / `load_data_config` — architecture, preprocessing, and split configuration; Step 5 reads `ModelConfig.threshold` as the default classification threshold rather than inventing a second one.
- `.gitignore`'s existing `experiments/runs/`, `*.pt`, `*.pth`, `*.ckpt` rules — evaluation artifacts land under the same ignored tree; no `.gitignore` change needed.

## Evaluation data contract

Evaluation operates on the same manifest schema as training (`image_path, label, generator, split` — `data/dataset_loader.py`'s `DatasetRecord`). It does **not** invent a separate evaluation manifest format. A run:

1. Loads one manifest via `load_manifest(manifest_path)` (CLI `--manifest`, required — there is no usable checked-in default; `config/model_config.yaml`'s `data.manifest_path` default, `data/manifest.csv`, does not exist).
2. Rejects duplicate `image_path` rows using the existing `model.training.dataset.assert_no_duplicate_images` (reused, not reimplemented) before splitting.
3. Derives `train`/`val`/`test` via `data.splitting.split_manifest(records, split_config)`, where `split_config` comes from `config/model_config.yaml`'s `data.split` section by default, or from explicit CLI/`EvaluationConfig` overrides (see Reproducibility — Step 4's `config.json` does not persist its resolved `SplitConfig`, so Step 5 must be told, or must independently trust, the same split parameters that produced the checkpoint being evaluated).
4. Evaluates the `test` split (unseen-generator, the primary result) and, separately, the `val` split (seen-generator report) from the same split call. `train` records are loaded (as a byproduct of calling `split_manifest`) but never touched beyond that.

## Implementation note: real-image pairing for the unseen-generator result

Discovered during implementation, not anticipated when this spec was approved: `data.splitting.split_manifest` routes **every** real image into the seen pool (`train`/`val`) and never into `test` — `choose_unseen_generators` only ever selects from non-real generator ids, and `test_records = [r for r in records if r.generator in unseen_generators]` therefore can never include a `generator == "real"` row. Verified directly against `data/manifests/genimage_dev.csv`: with the checked-in split config, `splits["test"]` contains 286 records (143 BigGAN + 143 Midjourney), zero of them real.

This is a `data/splitting.py` (Step 1) behavior, and `data/` is off-limits for Step 5. The fix does not touch splitting logic at all - it applies the same "real + generator G" pairing already used for `per_generator_breakdown` one level up, to the aggregate unseen-generator result itself:

- The **unseen-generator evaluation set** = `test` split records (held-out-generator fakes) **union** the real-labeled records from the `val` split (the only pool of real images not used for gradient updates - `train`'s real images were).
- The **seen-generator (val) result** is unchanged: computed over the `val` split exactly as split by `split_manifest`.
- `predictions.csv` still stamps each row with its literal `split` origin (`"val"` or `"test"`, exactly as `split_manifest` assigned it) - this pairing is a metric-aggregation choice made in `evaluate.py`, not a relabeling of any prediction row.
- Both `metrics.json`'s `test.overall` block and every `test.per_generator` entry carry an explicit `notes` entry stating that the real images used for this pairing come from `val`, so a reader of the report is never left to assume `test` itself contained real images.
- Generator-disjointness is unaffected: no non-real generator's rows move across the train/val vs. test boundary; only already-real (generator-agnostic) rows are additionally paired into the unseen-generator report.

## Generator-disjoint evaluation design

- Splitting logic is never re-implemented. Step 5's only splitting-related code is: call `split_manifest`, then perform one **additional, evaluation-specific hard check** beyond what `split_manifest`/`check_generator_leakage` already guarantee:
  - `check_generator_leakage` (invoked internally by `split_manifest`) already raises `GeneratorLeakageError` if any non-real generator appears in both train/val and test. Step 5 treats this as a hard precondition — if manifest/config combination raises, evaluation aborts before any inference runs.
  - `split_manifest` does **not**, on its own, guarantee the `test` split is non-empty of unseen generators (an empty test split is technically valid output of that function). Step 5 adds an explicit assertion: `test` must contain at least one AI-generated image and at least one generator not present in `train ∪ val`; `val` must contain at least one real image (the pool `test`'s fakes are paired against - see "Implementation note" above, since `test` itself is never given real images by `split_manifest`). If any of these fail, evaluation aborts with a clear, named error (`EmptyUnseenGeneratorSplitError`) rather than silently reporting on whatever it got.
- The chosen unseen generators are always logged explicitly (see Reproducibility) — never left implicit in a fraction.
- `--unseen-generators` (CLI, optional) lets an operator pin the exact held-out generator list for one evaluation run, overriding `held_out_generator_fraction`-based derivation, for cases where the checkpoint's original training run used an explicit list (or where the operator wants to reproduce a specific historical selection). When omitted, the config's `data.split` section (matching what training used, per the operator's responsibility) is used verbatim.
- Under no circumstance does Step 5 fall back to an ordinary random image-level split — there is exactly one split code path (`split_manifest`), and it is always generator-aware.

## Checkpoint loading

- `model.evaluation.evaluate` accepts `--checkpoint` (required). It is loaded via `model.training.checkpoint.load_model_from_checkpoint(path, map_location=device)`, which already:
  - raises `FileNotFoundError`/`torch`'s own load error clearly if the file is missing or corrupt;
  - constructs `EfficientNetB4Baseline(pretrained=False)` and applies `model_state_dict` — if the state dict is architecturally incompatible (e.g. from a different model), `load_state_dict` raises a clear `RuntimeError` describing the mismatched keys/shapes. Step 5 does not swallow or reinterpret this — it propagates with the checkpoint path added to the message.
- Because `load_model_from_checkpoint` only extracts `model_state_dict`, it does not expose the rest of the training checkpoint (`training_config`, `epoch`, `best_val_roc_auc`, etc.). Step 5 additionally calls `model.training.checkpoint.load_training_checkpoint(path)` once to read `model_config` (the `get_model_spec(...)` dict saved at training time) and `training_config` for **reporting/reproducibility only** (recorded into the evaluation run's `config.json`) — never to alter evaluation behavior. If this metadata is absent from an older/malformed checkpoint, evaluation logs a warning and proceeds using the live `config/model_config.yaml` architecture spec instead of aborting (the model still loaded successfully; only the *reported provenance* is degraded).
- Device: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`, matching Step 4's pattern exactly — no separate device-selection logic.
- `model.eval()` is called immediately after loading and never toggled back to `.train()` anywhere in the evaluation code path (enforced by a unit test — see Testing strategy).

## Prediction pipeline

`model/evaluation/predict.py` implements exactly one function:

```python
def run_inference(model, dataloader, device, threshold) -> List[PredictionRow]
```

`run_inference` is a **pure batched inference function**. It must not load checkpoints and must not construct datasets or dataloaders — those responsibilities belong to `evaluate.py` (checkpoint loading via `model.training.checkpoint.load_model_from_checkpoint`, dataset/dataloader construction via `model.training.dataset.build_val_dataset` + a plain `torch.utils.data.DataLoader`). `run_inference` receives an already-loaded, already-`.eval()`'d model and an already-built dataloader, and is responsible only for the forward pass, thresholding, and row assembly below. This separation keeps `predict.py` trivially unit-testable with a tiny in-memory model/dataloader and no filesystem/config dependency.

Executed under `torch.inference_mode()` (stricter than `no_grad()`, appropriate for pure inference with no autograd needed anywhere downstream). For every image:

| field | source |
|---|---|
| `image_path` | `DatasetRecord.image_path`, via the dataset's metadata dict (already returned by `SignalScopeDataset.__getitem__`) |
| `true_label` | `DatasetRecord.label` (0=real, 1=ai_generated) |
| `predicted_probability` | `torch.sigmoid(model(images))` — raw logit → sigmoid, exactly as `model/training/engine.py::validate` already does; no calibration applied |
| `predicted_label` | `1 if predicted_probability >= threshold else 0` |
| `generator` | `DatasetRecord.generator` |
| `split` | `"val"` or `"test"`, stamped by the caller (`evaluate.py`) — the dataset itself doesn't know which logical split it represents once built, so `run_inference` accepts the split label as a parameter rather than inferring it |

Batched via a plain `torch.utils.data.DataLoader`, constructed in `evaluate.py` (batch_size from `EvaluationConfig`, `shuffle=False`, `num_workers` from config — default 0, matching Step 4's Windows/CPU rationale, `drop_last=False` always, since every evaluation example must be scored). No manual Python-level image loop; DataLoader batching is what keeps this within the 8 GB RAM budget on the dev machine.

## Metric definitions

All implemented in `model/evaluation/metrics.py`, pure functions operating on parallel `labels: Sequence[int]`, `probabilities: Sequence[float]`, `predictions: Sequence[int]` — no framework/torch dependency, easily unit-testable with plain lists.

- **ROC-AUC**: reuses `model.training.engine.roc_auc_score` (public alias of the existing `_roc_auc_score` — see Files to change). Returns `None` when only one class is present in the evaluated set; callers must render this as `null`/`"undefined (single class present)"` in output, never as `0.0` or `1.0`.
- **Accuracy**: `correct / total`. Raises `EmptyEvaluationSetError` if `total == 0` — never silently returns `0.0` or `NaN` for an empty input.
- **Macro-F1**: computed manually (no scikit-learn) as the unweighted mean of per-class F1:
  - For each class `c ∈ {0, 1}`: `precision_c = TP_c / (TP_c + FP_c)`, `recall_c = TP_c / (TP_c + FN_c)`, `F1_c = 2 * precision_c * recall_c / (precision_c + recall_c)`.
  - If a class has zero predicted-positive or zero true-positive support, its precision/recall/F1 is defined as `0.0` with a `notes` entry recorded (not raised as an error), so a degenerate single-class evaluation still produces a well-defined (if uninformative) number rather than crashing the report.
  - `macro_f1 = (F1_0 + F1_1) / 2`.
- **Confusion matrix**: `{"tp": int, "tn": int, "fp": int, "fn": int}` where positive = `ai_generated` (label 1) — matches the FPR definition below and the model's own label convention.
- **FPR at threshold**: `FP / (FP + TN)`. Returns `None` (not `0.0`) when `FP + TN == 0` (no real/negative examples in the evaluated subset), with a `notes` field explaining why.
- Every metrics function accepts already-thresholded `predictions` rather than re-deriving them from `probabilities` internally, so the same probabilities can be re-thresholded for a per-generator breakdown without re-running inference.

## Per-generator analysis

**Design note:** a naive per-generator slice (all rows where `generator == G`) is single-class by construction (every "real" row has `generator="real"`/label 0; every non-real generator's rows are all label 1 — see `data/dataset_loader.py`'s `DatasetRecord.__post_init__`), so it can never yield a meaningful ROC-AUC. Per-generator metrics are instead computed on the pairing **{all real rows in the evaluated split} ∪ {rows where `generator == G`}**, for each non-real generator `G` present in that split. This answers the actually-meaningful question: "how well does the model distinguish real images from generator G's images, specifically?"

For each split (`val`, `test`) and each non-real generator `G` present in it:

- `num_samples` = `n_real_in_split + n_G`
- `num_real`, `num_fake` (= `n_G`)
- `roc_auc` (defined, since real+G is always two-class by construction, unless `n_real_in_split == 0`, in which case `None` with a note)
- `accuracy`, `macro_f1` at the frozen threshold
- No confusion matrix/FPR is required at the per-generator level (the overall split-level confusion matrix already covers this); per-generator adds ROC-AUC/accuracy/F1 only, keeping the report focused.

Additionally reported:

- **Overall test-split ("aggregate unseen-generator") result** — every unseen generator combined with all real test images, computed once as the split-level metrics block. This is the headline SIH number.
- **Overall val-split ("seen-generator") result** — same computation over `val`, reported side by side, explicitly labeled as *not* the unseen-generator result.
- A generator-level table is reported per split it appears in — a generator can only appear in one of val/test by construction (generator-disjoint), so there's no cross-split double-counting, but the report labels which split each generator table row came from for clarity.

## Threshold policy

- Default threshold: `ModelConfig.threshold` from `config/model_config.yaml` (currently `0.5`) — reused, not duplicated into a second config. `--threshold` (CLI, optional) allows an explicit override for one run, recorded in the run's `config.json` either way.
- The threshold is fixed **before** any test-split inference happens; it is never fit, searched, or adjusted using test predictions. No threshold-sweep/optimization logic is implemented in Step 5 at all.
- If a future step wants validation-based threshold tuning, it must select the threshold using only `val`-split predictions from a run that itself never inspected `test`, and freeze it before any `test` evaluation — Step 5's `predict`/`metrics` functions already take an explicit `threshold` argument, so that future workflow slots in without changing this step's code, only its caller.

## Reproducibility

Recorded in every run's `config.json` (via `model.training.logging_utils.RunLogger`, reused rather than a second logger class):

- `eval_run_id`
- `checkpoint_path` (as given) and, when present in the loaded training checkpoint, `training_run_id` (inferred from the checkpoint's parent directory structure — see Output artifacts), `training_config` and `model_config` snapshots read from the checkpoint itself
- `manifest_path`, dataset label (`"genimage_dev (development shard, not SIH benchmark)"` when the manifest path contains `genimage_dev`, exactly matching Step 4's existing `DEV_SHARD_MARKER`/`DEV_SHARD_LABEL` convention in `model/training/train.py` — reused, not reinvented)
- The resolved `SplitConfig` actually used for this evaluation run (seed, `val_fraction`, `unseen_generators` — explicit list actually chosen, whether derived or overridden — `held_out_generator_fraction`)
- `threshold` (resolved value and its source: config default vs. CLI override)
- Preprocessing: image size, normalization mean/std (from `data_config.preprocessing`, logged verbatim)
- `device` (`cpu`/`cuda`), `batch_size`, `num_workers`
- `timestamp`

Given the same checkpoint file, the same manifest, and the same `SplitConfig`, evaluation is deterministic: no shuffling occurs anywhere in the eval path (`shuffle=False` always), preprocessing is already deterministic (`ImagePreprocessor`/`build_val_dataset`), and inference has no dropout/batchnorm-training-mode randomness because `model.eval()` is enforced. No seeding call is needed in the eval path itself beyond what determines the split (already seeded via `SplitConfig.seed`).

## Output artifacts

Nested under the training run's own directory when the checkpoint path matches the expected `experiments/runs/<training_run_id>/checkpoints/*.pt` shape (inferred by taking the checkpoint's grandparent directory name) — otherwise a standalone `experiments/runs/eval-<timestamp>/` is created. This keeps a checkpoint and its evaluation(s) co-located without inventing a second top-level artifact tree:

```text
experiments/runs/<training_run_id>/
└── evaluation/
    └── <eval_run_id>/
        ├── config.json              # full reproducibility record (see above)
        ├── metrics.json             # machine-readable: overall val + overall test + per-generator tables
        ├── predictions.csv          # one row per evaluated image (both val and test, "split" column distinguishes)
        └── evaluation_report.md     # human-readable summary (headline unseen-generator ROC-AUC first, dev-shard disclaimer banner, per-generator table)
```

- `confusion_matrix.png` (or any plot) is **not** produced by default — the confusion matrix is always present, machine-readable, in `metrics.json`. An optional `--plot` flag may render `confusion_matrix.png` via `matplotlib` **only if installed**; its absence must not break evaluation (import attempted lazily, inside the flag's code path only).
- `evaluation_report.md`'s first line is always a disclaimer banner when the dataset label indicates the dev shard, e.g. `> Tiny-GenImage development shard — NOT the final SIH benchmark.` — this is not optional and is asserted by a test.
- Nothing under `experiments/` is committed; the existing `.gitignore` rule (`experiments/runs/`) already covers this new nested path.

## CLI

```bash
python -m model.evaluation.evaluate \
    --checkpoint experiments/runs/<run_id>/checkpoints/best.pt \
    --manifest data/manifests/genimage_dev.csv \
    [--model-config config/model_config.yaml] \
    [--eval-config config/evaluation_config.yaml] \
    [--output-dir <override>] \
    [--unseen-generators ADM,Midjourney] \
    [--threshold 0.5]
```

- `--checkpoint` (required), `--manifest` (required — no usable default exists).
- `--model-config` / `--eval-config` split mirrors Step 4's own `--config`/`--model-config` split in `model/training/train.py` (architecture/data/split config vs. run-specific config) rather than a single `--config`, kept consistent with the existing precedent rather than introducing a second CLI convention.
- `--output-dir` overrides the inferred nested path (Output artifacts) for cases where the checkpoint isn't under `experiments/runs/`.
- `--unseen-generators` (comma-separated) overrides config-derived unseen-generator selection for one run (Generator-disjoint evaluation design).
- `--threshold` overrides `ModelConfig.threshold` for one run.
- No `--smoke-test` flag: the dev shard is already tiny (2,000 images), and evaluation (unlike training) has no epochs/iterations to truncate — a full pass over `test` + `val` is already fast on CPU. No batch-size-hiding flags beyond what `EvaluationConfig`/`--eval-config` already exposes.

## Testing strategy

New `tests/test_evaluation.py`, following `tests/test_training.py`'s conventions exactly (`pytest.importorskip("torch")`, tiny synthetic images written via a local `_write_image` helper, `pretrained=False` throughout, no real dataset or network access):

- checkpoint loading: valid checkpoint loads and returns a usable model; missing file raises clearly; a checkpoint with an incompatible state dict raises clearly.
- prediction pipeline: output row count matches input dataset size; `predicted_probability` is in `[0, 1]`; `predicted_label` matches `probability >= threshold` at both `0.5` and a non-default threshold; label mapping (0=real, 1=ai_generated) is respected in output rows; `run_inference` works given only a plain model + dataloader + device + threshold (no checkpoint/manifest arguments), verifying the pure-function boundary from `evaluate.py`.
- metrics: accuracy on a hand-constructed known confusion; macro-F1 against a hand-computed expected value including a zero-support-class case; confusion matrix counts against a hand-constructed set; FPR at threshold including the `FP+TN==0` case (`None`, not `0.0`); ROC-AUC via the reused function including the single-class-returns-`None` case (already covered in `tests/test_training.py`, re-asserted here at the integration level).
- empty evaluation input: an empty split raises `EmptyEvaluationSetError`, never returns degenerate zero metrics.
- generator leakage / disjoint evaluation: reusing `data.splitting` fixtures (mirroring `tests/test_training.py`'s `split_config` fixture) to assert (a) `check_generator_leakage` still fires correctly through this pipeline's own call path, (b) the additional "test split must contain ≥1 unseen generator" assertion fires on a manifest engineered to produce an empty/all-real test split.
- per-generator analysis: a synthetic manifest with two non-real generators produces two well-defined per-generator ROC-AUCs (not `None`), verifying the real+G pairing design rather than the naive single-generator-slice trap.
- determinism: two evaluation runs over the same checkpoint/manifest/config produce identical `predictions.csv` content and identical `metrics.json` values.
- artifact schema: `predictions.csv` has exactly the required columns; `metrics.json` contains both a `val` and `test` block plus a `per_generator` block; `evaluation_report.md` contains the dev-shard disclaimer banner when the manifest path contains `genimage_dev`.
- `model.eval()` is never left/returned in training mode after an evaluation call (assert `model.training is False` post-call).

All 66 currently-passing tests remain unaffected; the only shared-file change (`model/training/engine.py`, see below) is purely additive.

## Files to create

- `model/evaluation/__init__.py`
- `model/evaluation/config.py` — `EvaluationConfig` dataclass (batch_size, num_workers, pin_memory, output_dir override, generate_plots: bool = False) + `load_evaluation_config(path)`, mirroring `model/training/config.py`'s pattern exactly.
- `config/evaluation_config.yaml` — defaults for the above, parallel to `config/training_config.yaml`.
- `model/evaluation/predict.py` — `run_inference(...)` (see Prediction pipeline). Pure batched inference only — no checkpoint loading, no dataset/dataloader construction.
- `model/evaluation/metrics.py` — `accuracy`, `macro_f1`, `confusion_matrix`, `fpr_at_threshold`, `per_generator_breakdown`, re-exporting `roc_auc_score` from `model.training.engine`.
- `model/evaluation/report.py` — assembles `metrics.json`, `predictions.csv`, `evaluation_report.md` (and optionally `confusion_matrix.png` behind `--plot`) from `predict`/`metrics` outputs; reuses `model.training.logging_utils.RunLogger` for `config.json`.
- `model/evaluation/evaluate.py` — CLI entrypoint wiring manifest → split → checkpoint load (via `model.training.checkpoint.load_model_from_checkpoint`) → dataset/dataloader construction (via `model.training.dataset.build_val_dataset` + `torch.utils.data.DataLoader`) → `predict.run_inference` → metrics → report.
- `tests/test_evaluation.py`.

## Files to change

- `model/training/engine.py` — add one line exposing the existing private `_roc_auc_score` under a public name (`roc_auc_score = _roc_auc_score`, keeping the old name working as an alias so `tests/test_training.py`'s existing import is unaffected). No behavioral change; purely makes an already-correct, already-tested implementation importable from `model/evaluation/` without duplicating it. (`model/training/` is not in Step 5's forbidden-boundary list.)

No other existing file is modified. `config/model_config.yaml`, `data/`, `model/architectures/`, `backend/`, `app/`, `ui/`, `explainability/` are untouched.

## Dependencies

- None required for the core metrics/CLI path — everything is `torch`/`numpy`/`pyyaml`, already in `requirements.txt`.
- `matplotlib` is an **optional** dependency, imported lazily only inside the `--plot` code path in `model/evaluation/report.py`; its absence must not raise at import time or break the default (no-plot) evaluation run. Not added to `requirements.txt` under Step 5.
- scikit-learn is deliberately not reintroduced — the existing self-contained ROC-AUC is reused as-is.

## Rules / architectural boundaries

- Do not modify `backend/`, `app/`, `ui/`, `explainability/`, `data/`, `model/architectures/`, `config/model_config.yaml`, or `backend/requirements.txt`.
- Do not use `test`-split predictions for any decision — checkpoint, threshold, hyperparameters, augmentation, architecture, or calibration. The checkpoint is loaded read-only; the threshold is fixed before any test inference runs.
- Do not implement calibration in this step.
- Do not reintroduce scikit-learn for ROC-AUC.
- Do not present Tiny-GenImage dev-shard results as final SIH/hidden-test/full-GenImage performance — every report generated against a manifest containing `genimage_dev` must carry the disclaimer banner.
- Do not duplicate `split_manifest`, `ImagePreprocessor`, `build_val_dataset`, or model-construction logic.
- Never leave the model in `.train()` mode after an evaluation call.
- `predict.run_inference` must not load checkpoints or construct datasets/dataloaders — those responsibilities remain in `evaluate.py`.

## Acceptance criteria

- `python -m model.evaluation.evaluate --checkpoint <Step-4 best.pt> --manifest data/manifests/genimage_dev.csv` runs to completion on this CPU machine, producing the full artifact set under `experiments/runs/<training_run_id>/evaluation/<eval_run_id>/`.
- `metrics.json` contains distinct, correctly-labeled `val` (seen-generator) and `test` (unseen-generator) blocks, each with ROC-AUC (or an explicit `null`+reason), accuracy, macro-F1, confusion matrix, and FPR at the frozen threshold.
- Per-generator entries use the real+G pairing design and produce defined ROC-AUC values (not `null`) whenever both real images and that generator's images are present in the split.
- Re-running evaluation with the same checkpoint/manifest/config produces byte-identical `predictions.csv` and identical `metrics.json` values.
- A manifest/config combination that would produce an empty or all-real unseen-generator test split is rejected with a clear, named error before any inference runs.
- `tests/test_evaluation.py` passes together with all pre-existing tests (66 + new, all green).
- No files under `backend/`, `app/`, `ui/`, `explainability/`, `data/`, `model/architectures/` are modified; `config/model_config.yaml` is read-only.
- Any report generated against a `genimage_dev` manifest visibly states it is a development shard, not a benchmark.

## Risks and mitigations

- **Step 4 doesn't persist its `SplitConfig`** — mitigated by evaluation recording *its own* resolved `SplitConfig` in every run's `config.json` going forward, and by the checkpoint's own recorded `training_config`/`model_config` giving partial provenance; full closure would need a small, separate Step 4 patch (not part of Step 5) to also log `data_config.split`.
- **Fraction-derived unseen-generator selection is manifest-shape-sensitive** — mitigated by `--unseen-generators` allowing an explicit, reproducible override per evaluation run; long-term mitigation (pinning `unseen_generators` explicitly in config) is out of scope here.
- **Checkpoint/manifest mismatch** (evaluating a checkpoint against a manifest whose generator set differs from what it was trained on) can silently produce a misleading "unseen-generator" claim if the operator supplies inconsistent inputs — mitigated by the hard leakage/non-empty-unseen-split checks, but ultimately relies on the operator supplying the correct paired manifest/config; not fully preventable in software without persisting more training-time provenance than Step 4 currently does.
- **8 GB RAM / CPU-only constraint** — mitigated by batched `DataLoader` inference (never loading the full dataset into memory) and by the dev shard being small enough (2,000 images) that even a full pass is fast; batch size remains configurable for a future larger dataset.

## Implementation sequence

1. `model/training/engine.py`: add the public `roc_auc_score` alias (smallest, most isolated change; verify all 66 existing tests still pass unchanged).
2. `model/evaluation/config.py` + `config/evaluation_config.yaml`.
3. `model/evaluation/metrics.py` (pure functions, unit-testable in isolation first, before any model/dataset wiring).
4. `model/evaluation/predict.py`
   Implement `run_inference(model, dataloader, device, threshold)` as a pure
   batched inference function. It must not load checkpoints or construct
   datasets; those responsibilities remain in `evaluate.py`.
5. `model/evaluation/report.py` (artifact writers, reusing `RunLogger`).
6. `model/evaluation/evaluate.py` (CLI, wiring everything above, including the generator-disjoint hard checks).
7. `tests/test_evaluation.py`, written alongside each module above rather than all at the end.
8. Manual verification: run the CLI against the real `genimage_dev.csv` manifest and a Step 4 `best.pt`/`last.pt`, inspect the produced `evaluation_report.md` for correctness and the disclaimer banner, confirm `experiments/runs/` artifacts are git-ignored.
