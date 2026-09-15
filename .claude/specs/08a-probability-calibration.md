# Spec: Probability Calibration

## Overview

Steps 1-7 produced a trained detector (best development result: `rgb_frequency_fusion`, unseen-generator ROC-AUC = 0.9396 on the Tiny-GenImage development shard, per `.claude/specs/07-ablation-experiment.md`) whose raw `sigmoid(logit)` output is explicitly documented as "a raw model probability, NOT a calibrated one" (`model/architectures/fusion_model.py::PROBABILITY_MEANING`). Before any confidence number is shown to a user, CLAUDE.md requires it be calibrated ("Calibrate confidence before user-facing predictions"). This step adds **temperature scaling** as a strictly post-training, checkpoint-preserving operation: fit one scalar temperature `T` on the validation split's raw logits, apply `sigmoid(logit / T)` at inference, and report ranking metrics (unchanged by calibration) alongside calibration metrics (Brier score, ECE) before vs. after, using the existing generator-disjoint evaluation protocol unmodified. No detector weights are retrained or altered.

## Depends on

- Step 4 training pipeline (`model/training/`) - checkpoint format, `model_config.get_spec()` registry.
- Step 5 evaluation pipeline (`model/evaluation/`) - `data.splitting.split_manifest`, generator-disjoint split protocol, `SplitMetrics`/`compute_split_metrics`, the real-image-pairing behavior (`REAL_PAIRING_NOTE`).
- Step 6 frequency fusion (`model/architectures/fusion_model.py`, `model/training/dual_branch_dataset.py`) - the fusion checkpoint's dict-input `{"rgb": ..., "frequency": ...}` shape and its architecture id `rgb_frequency_fusion` in `model/training/checkpoint.py`'s architecture registry.
- Step 7 ablation (`.claude/specs/07-ablation-experiment.md`) - the trained fusion checkpoint (`experiments/runs/<rgb_frequency_fusion_run_id>/checkpoints/best.pt`) this step's primary experiment calibrates, and `config/ablation/model_config.yaml` (the split config that checkpoint's evaluation must reuse to stay generator-disjoint and leakage-free).
- `data/manifests/genimage_dev.csv` (Tiny-GenImage development shard - development data only, not the final SIH benchmark).

## Model / Architecture Changes

No model architecture changes. No detector weights are created, modified, or retrained. Temperature scaling adds exactly one learned scalar parameter `T > 0`, stored and versioned entirely separately from the frozen detector checkpoint:

- **Fitting:** collect raw logits (pre-sigmoid, shape `(N, 1)` -> `(N,)`) for every validation-split example by running the frozen model in `eval()`/`no_grad` mode. Fit a strictly positive temperature parameter by minimizing validation negative log-likelihood of `sigmoid(logit / T)` against true binary labels using LBFGS, parameterized as `T = exp(log_temperature)` so `T > 0` holds by construction without a manual clamp - the standard temperature-scaling procedure (Guo et al., 2017), applied to a single binary logit rather than a multi-class softmax.
- **Inference (calibrated):** `calibrated_probability = sigmoid(raw_logit / T)`. Since `T > 0`, `x -> x / T` is strictly monotonic increasing, and `sigmoid` is itself strictly monotonic increasing - so the composition `logit -> sigmoid(logit / T)` is strictly monotonic in `logit` for every example. This means calibration is **provably rank-invariant**: it cannot change the relative ordering of any two examples' scores, and therefore cannot change ROC-AUC (which depends only on ranking), on **either** the validation split or the unseen-generator split.
- **Threshold policy invariance at threshold 0.5:** for the fixed threshold used throughout this repo (`config/model_config.yaml`'s `model.threshold = 0.5`), `sigmoid(z) >= 0.5 <=> z >= 0`. Substituting `z = logit` (raw) and `z = logit / T` (calibrated) with `T > 0` gives the identical condition `logit >= 0` in both cases. **Therefore, at threshold 0.5, temperature scaling cannot change any example's binary classification decision** - accuracy, macro-F1, FPR, and the confusion matrix at threshold 0.5 must be numerically identical before and after calibration. This is a correctness invariant of the implementation, not merely an expected outcome, and must be directly verified (Rules for implementation, Acceptance criteria) rather than assumed from the algebra alone - because floating-point evaluation of `sigmoid` and comparison against `0.5` can in principle disagree with the exact-algebra argument at representable-precision boundaries, and because this guarantee is specific to a threshold of exactly `0.5` (it does not generalize to an arbitrary threshold `t != 0.5`, where `sigmoid(logit/T) >= t` is **not** equivalent to `sigmoid(logit) >= t` for `T != 1`). If threshold-policy metrics were ever to differ beyond floating-point tolerance at threshold 0.5, that must be treated as an implementation bug, never reported as a calibration effect.

## Data Changes

No data changes. Reuses the existing manifest (`data/manifests/genimage_dev.csv`), the existing `data.splitting.split_manifest()` (unmodified, imported not copied), and the existing generator-disjoint `SplitConfig` recorded in the checkpoint's paired training run (`config/ablation/model_config.yaml` for the Step 7 fusion checkpoint). Temperature is fit **only** on the `val` split's logits. The `test` split (unseen-generator holdout) is read only for the before/after evaluation pass, strictly after `T` is already fixed - never to fit or select `T`. This preserves train/validation/test separation exactly as Steps 4-7 established it; no new split logic is introduced.

## API / Inference Changes

No API changes. This step does not touch `app/` and does not create any HTTP route (per CLAUDE.md's "Development Boundaries" - backend/app integration happens after the ML inference pipeline is stable). It adds a library module (`model/calibration/temperature_scaling.py`) that a later, separate integration step can import from `model/inference/` or `app/` - that wiring is explicitly out of scope here.

## Explainability Changes

No explainability changes. Grad-CAM/evidence extraction (`model/explainability/`) is untouched. The only user-facing-confidence-relevant change is that a *later* consumer of this module could report `calibrated_probability` instead of the raw one - not implemented or wired up by this step.

## Evaluation Plan

Every metric is computed **twice** on the identical generator-disjoint evaluation protocol Step 5/7 already use (`test`'s held-out-generator fakes paired with `val`'s real images, per `model/evaluation/evaluate.py`'s existing `REAL_PAIRING_NOTE` behavior, reused verbatim) - once from raw `sigmoid(logit)`, once from calibrated `sigmoid(logit / T)`:

**Ranking metrics (provably invariant under calibration - see "Model / Architecture Changes"; verified directly, not just asserted):**
- Validation ROC-AUC: before vs. after must match within `1e-6`.
- Unseen-generator ROC-AUC: before vs. after must match within `1e-6`.

Both are required because `T > 0` makes `logit -> sigmoid(logit / T)` strictly monotonic on any input distribution - the invariance is not specific to the unseen-generator split, and both must be checked so a bug affecting only one split's logit collection (e.g. an accidental re-sort, a mismatched row ordering between labels and probabilities) cannot slip through.

**Threshold-policy metrics (evaluated at the existing frozen threshold = 0.5, applied identically to both raw and calibrated probabilities; provably identical before vs. after at this specific threshold - see "Model / Architecture Changes"):**
- Accuracy at threshold 0.5: must remain unchanged (identical before vs. after).
- Macro-F1 at threshold 0.5: must remain unchanged.
- FPR at threshold 0.5: must remain unchanged.
- Confusion matrix (TP/TN/FP/FN) at threshold 0.5: must remain unchanged.

Calibration changes probability/confidence *values*; with a 0.5 threshold it does not change the underlying binary decisions - `calibration_report.md` must state this distinction explicitly rather than implying calibration could shift examples across the decision boundary.

**Calibration metrics (the reason this step exists - expected to improve, not guaranteed to):**
- Brier score (mean squared error between probability and true label)
- Expected Calibration Error (ECE), fixed-width-bin definition, reported with the chosen bin count

All four groups are computed for both the `val` split (the split `T` was fit on - reported for transparency, never used to claim generalization) and the unseen-generator evaluation set (the primary, held-out result). The **primary success criterion is whether ECE and Brier score improve on the unseen-generator set without moving unseen-generator ROC-AUC** - calibration is a probability-quality fix, not a ranking-performance objective, and must not be reported as improving detection accuracy.

## Files to change

None. (This step is designed as fully additive to avoid touching the already-verified Step 4-7 training/evaluation code paths - see "Files that must remain untouched" and Risk R1.)

## Files to create

- `model/calibration/__init__.py` - package marker.
- `model/calibration/temperature_scaling.py` - `TemperatureScaler` (a 1-parameter `nn.Module` wrapping `log_temperature`), `fit_temperature(logits, labels) -> TemperatureScaler`, `apply_temperature(logits, temperature) -> probabilities`. Operates purely on `torch.Tensor` logits/labels - no dataset/checkpoint/config dependency, matching the existing pure-function style of `model/evaluation/metrics.py`.
- `model/calibration/calibration_metrics.py` - `expected_calibration_error(probabilities, labels, n_bins=15) -> float`, `brier_score(probabilities, labels) -> float`. Pure functions over plain sequences, mirroring `model/evaluation/metrics.py`'s existing style (no torch dependency required at this layer).
- `model/calibration/logit_inference.py` - `run_logit_inference(model, dataloader, device, split) -> List[LogitPredictionRow]`, a `predicted_probability`-free sibling of `model.evaluation.predict.run_inference` that returns the raw pre-sigmoid logit instead of (in addition to) the probability. Deliberately duplicated rather than importing/modifying `model/evaluation/predict.py`, following the repo's own established precedent (`predict.py`'s own docstring: "Duplicated (not imported) from model/training/engine.py's identical helper - a two-line utility does not warrant a cross-package dependency") - keeps `model/evaluation/` completely untouched (Risk R1).
- `model/calibration/calibrate.py` - the orchestration entrypoint (CLI + `run_calibration()`), described under CLI/API design below.
- `config/calibration_config.yaml` - `calibration.n_bins` (ECE bin count, default 15), `calibration.lbfgs_max_iter` (default 50), `calibration.output_dir` (default `experiments/calibration`). Kept separate from `config/model_config.yaml`/`config/evaluation_config.yaml`, matching the existing one-concern-per-file convention (`model/evaluation/config.py`'s own docstring rationale).
- `tests/test_calibration.py` - unit tests per "Rules for implementation" / Definition of done (temperature scaling module, fitting, probability bounds, ECE, Brier score, edge cases).
- `tests/test_calibrate_pipeline.py` - integration-level tests for `run_calibration()` against a tiny synthetic/fixture checkpoint and manifest (checkpoint/config reproducibility, artifact contents), following the existing pattern in `tests/test_evaluation.py`/`tests/test_frequency_fusion.py` (CPU-only, no real dataset or network access).

## New dependencies

No new dependencies. `torch.optim.LBFGS` is already available via the existing `torch` dependency; no new pip package is required for temperature scaling, ECE, or Brier score (both are straightforward to implement with `torch`/pure Python, matching `model/evaluation/metrics.py`'s existing no-sklearn convention for `roc_auc_score`).

## Rules for implementation

- Do not use the hidden SIH test set for training or tuning.
- Do not introduce data leakage between train and validation/test splits.
- Fit temperature **only** on the `val` split's logits - never on `test` (the unseen-generator split), and never combine `val` and `test` logits for fitting.
- Prioritize unseen-generator ROC-AUC over raw training accuracy - and explicitly verify, for **both** the validation split and the unseen-generator split (via a test asserting rank-correlation = 1.0 between raw and calibrated scores, in addition to the `1e-6`-tolerance ROC-AUC comparison), that calibration does not change either. Temperature scaling with `T > 0` is strictly monotonic, so any measured AUC change on either split indicates an implementation bug, never a modeling result.
- Threshold-policy invariance at threshold 0.5 is a correctness invariant, not an assumption: directly compare the raw-probability thresholded predictions and the calibrated-probability thresholded predictions element-by-element and require exact equality (not merely re-derive accuracy/macro-F1/FPR/confusion-matrix separately and observe they match) - see "Acceptance criteria".
- Use reproducible random seeds - the LBFGS fit is deterministic given fixed logits/labels (no stochastic minibatching), but record the seed used to reproduce the evaluation-time split resolution (already the checkpoint's own training seed / the split's `seed` field) in the output artifact regardless.
- Keep model configuration separate from source code - `config/calibration_config.yaml`, not hardcoded constants in `calibrate.py`.
- Do not hardcode dataset paths - `--manifest` is a required CLI argument, exactly as `model/evaluation/evaluate.py` already requires.
- Record experiment configuration and metrics - every calibration run writes a full artifact bundle (see CLI/API design) under `experiments/calibration/<calib_run_id>/`, mirroring the existing `experiments/runs/<run_id>/evaluation/<eval_run_id>/` artifact discipline.
- Do not claim an improvement without measured comparison - `calibrate.py` always computes and writes both the before-calibration and after-calibration metric blocks in the same run; never only the after-calibration numbers.
- Explanations must be grounded in model evidence - not applicable to this step (no explainability changes), noted for completeness.
- Do not make absolute claims such as "this image is definitely AI-generated"; use responsible wording such as "likely AI-generated" - not applicable to this step's artifacts (metrics/config only, no user-facing text), noted for completeness and to bind any later `model/inference/` consumer of this module.
- Do not analyze or identify real people; do not build political or event-claim detection features - not applicable, noted for completeness.
- Calibration must operate on logits, never on already-sigmoided probabilities - enforced structurally: `TemperatureScaler`/`fit_temperature`/`apply_temperature` take raw logits as input and internally apply `sigmoid(logit / T)`; there is no code path that takes a probability and attempts to invert it.
- Never call `model.evaluation.evaluate.run_evaluation()` or `model.evaluation.predict.run_inference()` from `model/calibration/` in a way that requires editing either file - build only additive, parallel functions in `model/calibration/` (Risk R1 / "Files to modify: None").

## Experiment protocol

**Primary experiment (the one this step reports):**

1. **Checkpoint:** the Step 7 `rgb_frequency_fusion` run's `experiments/runs/<rgb_frequency_fusion_run_id>/checkpoints/best.pt` (unseen-generator ROC-AUC = 0.9396 per `.claude/specs/07-ablation-experiment.md`), loaded read-only via the existing `model.training.checkpoint.load_model_from_checkpoint` (architecture dispatch already handles `rgb_frequency_fusion` - no new loader code needed).
2. **Split:** resolve `data/manifests/genimage_dev.csv` against `config/ablation/model_config.yaml`'s `SplitConfig` (the same config the Step 7 fusion run used, so the split is provably identical to the one that checkpoint was trained/evaluated against) via the existing, unmodified `data.splitting.split_manifest()`.
3. **Fit:** run `logit_inference.run_logit_inference()` over the `val` split's dual-branch dataset (`model.training.dual_branch_dataset.build_dual_branch_val_dataset`, dispatched exactly as `model/evaluation/evaluate.py` already does by checkpoint architecture id) to collect `(logit, label)` pairs; fit `T` with `fit_temperature()`.
4. **Evaluate (before):** compute the full metric set (ranking + threshold-policy + calibration) on raw `sigmoid(logit)` for both `val` and the unseen-generator set (`test` fakes + `val` real images, reusing the exact real-image-pairing logic `model/evaluation/evaluate.py::run_evaluation()` already implements, re-derived here rather than imported to keep `model/evaluation/` untouched).
5. **Evaluate (after):** recompute the identical metric set using `sigmoid(logit / T)` in place of `sigmoid(logit)`, same threshold, same grouping.
6. **Persist:** write the full artifact bundle (below) under `experiments/calibration/<calib_run_id>/`.

**Secondary check (non-blocking compatibility note, not the headline result):** if an existing checkpoint for at least one other registered architecture (e.g. the Step 4 `rgb_only` checkpoint, or a `frequency_only` checkpoint from Step 6/7) is available, run steps 1-6 against it to demonstrate `model/calibration/` works across architectures without modification (in the spirit of requirement 12). This is reported as a secondary compatibility result, not benchmarked against the fusion result, and its success or failure does not gate this step's Definition of Done - the primary and only blocking deliverable is successful calibration of the Step 7 `rgb_frequency_fusion` checkpoint. If no second checkpoint is available at implementation time, this check is skipped and noted as skipped, not treated as a failure.

**Artifact bundle**, per `calib_run_id` (mirrors `experiments/runs/.../evaluation/<eval_run_id>/`'s existing shape):

```text
experiments/calibration/<calib_run_id>/
├── config.json          # checkpoint_path, checkpoint architecture id, manifest_path, split_config
│                         # (seed/val_fraction/unseen_generators/held_out_generator_fraction),
│                         # threshold, threshold_source, calibration method ("temperature_scaling"),
│                         # n_bins, lbfgs config, device, timestamp
├── temperature.json      # fitted T, final NLL loss, num_val_samples used to fit, optimizer iterations
├── metrics_before.json   # val + unseen-generator: overall ranking/threshold-policy/calibration metrics
│                         # (SAME schema as model/evaluation/metrics.py's SplitMetrics.to_dict(), plus
│                         # brier_score/ece fields), per-generator breakdown, REAL_PAIRING_NOTE
├── metrics_after.json    # identical schema, computed on calibrated probabilities
└── calibration_report.md # human-readable before/after comparison table + dev-shard disclaimer,
                           # mirroring model/evaluation/report.py::write_evaluation_report_md's style
```

## CLI/API design

```bash
python -m model.calibration.calibrate \
    --checkpoint experiments/runs/<rgb_frequency_fusion_run_id>/checkpoints/best.pt \
    --manifest data/manifests/genimage_dev.csv \
    --model-config config/ablation/model_config.yaml \
    --calibration-config config/calibration_config.yaml \
    [--output-dir experiments/calibration] \
    [--unseen-generators BigGAN,Midjourney]
```

Mirrors `model/evaluation/evaluate.py`'s existing argument conventions exactly (`--checkpoint`, `--manifest`, `--model-config`, `--output-dir`, `--unseen-generators` all reused verbatim in name and meaning) so the two tools compose predictably; `--calibration-config` is the one new flag, defaulting to the repo-default `config/calibration_config.yaml` path (same `None`-falls-back-to-default pattern as `model/evaluation/config.py::load_evaluation_config`). No `--threshold` override flag is exposed - threshold is always read from `model_config.threshold`, since this step evaluates the existing frozen threshold policy rather than letting it be silently changed alongside calibration.

`run_calibration(checkpoint_path, manifest_path, model_config_path, calibration_config_path=None, output_dir_override=None, unseen_generators_override=None) -> Dict[str, Any]` is the importable function `calibrate.main()` wraps, returning `{"calib_run_id", "run_dir", "temperature", "val_roc_auc", "unseen_generator_roc_auc", "brier_before", "brier_after", "ece_before", "ece_after"}` - mirrors `run_evaluation()`'s return-dict convention.

No HTTP API. No change to `app/`.

## Acceptance criteria

- `python -m model.calibration.calibrate` run against the Step 7 `rgb_frequency_fusion` checkpoint completes without modifying `experiments/runs/<...>/checkpoints/best.pt` (verified by checksum before/after).
- The fitted temperature `T` is a strictly positive finite float, fit using only `val`-split logits (verified in `config.json`'s recorded sample count against the known Step 7 `val` split size).
- `metrics_before.json` and `metrics_after.json`'s **validation** ROC-AUC values are numerically identical within `1e-6`.
- `metrics_before.json` and `metrics_after.json`'s **unseen-generator** ROC-AUC values are numerically identical within `1e-6`.
- A direct, per-example comparison of raw-probability thresholded predictions (`predicted_label` at threshold 0.5) vs. calibrated-probability thresholded predictions is run and asserted **exactly equal**, for both `val` and the unseen-generator set - not inferred from the aggregate metrics matching. Accuracy, macro-F1, FPR, and the confusion matrix at threshold 0.5 are then unchanged before vs. after as a direct consequence, and are also compared exactly (no tolerance needed, since both are derived from the same equal label sets).
- `metrics_before.json` and `metrics_after.json` both report Brier score and ECE on both `val` and the unseen-generator set, using the fixed frozen threshold for accuracy/macro-F1/FPR/confusion matrix.
- `calibration_report.md` carries the Tiny-GenImage development-shard disclaimer (reusing `model/evaluation/report.py::DEV_SHARD_BANNER`'s wording convention) and the real-image-pairing note (`REAL_PAIRING_NOTE`'s wording, reproduced verbatim) wherever the unseen-generator result is presented.
- All new unit/integration tests pass (`pytest tests/test_calibration.py tests/test_calibrate_pipeline.py`), alongside the full existing suite (no regression in `tests/test_evaluation.py`, `tests/test_frequency_fusion.py`, `tests/test_training.py`, `tests/test_model.py`, `tests/test_data.py`, `tests/test_genimage_ingest.py`).
- No file under `model/training/`, `model/evaluation/`, `model/architectures/`, `data/`, `backend/`, `app/`, `ui/`, `explainability/`, or the committed `config/*.yaml` defaults (excluding the new `config/calibration_config.yaml`) is modified.
- No file outside `experiments/calibration/` (which is `.gitignore`d, matching `experiments/runs/`'s existing rule) is written by a calibration run.

## Definition of done

- [ ] `model/calibration/temperature_scaling.py`, `calibration_metrics.py`, `logit_inference.py`, `calibrate.py`, `config/calibration_config.yaml` exist and import cleanly.
- [ ] `python -m model.calibration.calibrate --checkpoint <Step 7 fusion best.pt> --manifest data/manifests/genimage_dev.csv --model-config config/ablation/model_config.yaml` runs to completion on CPU and writes the full artifact bundle under `experiments/calibration/<calib_run_id>/`.
- [ ] `temperature.json` records a fitted `T` with its fitting sample count matching the `val` split size logged in `config.json`.
- [ ] `metrics_before.json`/`metrics_after.json` **validation** ROC-AUC matches to within `1e-6`, and **unseen-generator** ROC-AUC matches to within `1e-6` (ranking preserved on both splits); Brier score and ECE are present in both, for both `val` and the unseen-generator set.
- [ ] Thresholded predictions at threshold 0.5 (raw vs. calibrated) are directly compared and are exactly equal, for both `val` and the unseen-generator set; accuracy, macro-F1, FPR, and confusion matrix at threshold 0.5 are unchanged before vs. after - verified, not merely assumed from the monotonicity argument.
- [ ] `calibration_report.md` states, in its own text: the "development shard, not final SIH benchmark" disclaimer, the real-image-pairing limitation (Requirement 18), and the "calibration changes confidence values but not threshold-0.5 decisions" distinction - none merely inherited via a linked doc.
- [ ] **Primary deliverable (blocking):** calibration succeeds end-to-end against the Step 7 `rgb_frequency_fusion` checkpoint, satisfying every item above.
- [ ] **Secondary check (non-blocking):** if a checkpoint for at least one other registered architecture (`efficientnet_b4` `rgb_only`, and/or `frequency_branch`) is available, `calibrate.py` runs without modification against it and produces a valid artifact bundle, reported as a compatibility note. If no second checkpoint is available, this item is marked skipped, not failed, and does not block sign-off.
- [ ] `tests/test_calibration.py` covers: temperature scaling forward pass (logit -> calibrated probability, bounds `(0, 1)`), `fit_temperature` convergence on a synthetic separable-logit fixture, `expected_calibration_error` on a hand-computed small example, `brier_score` on a hand-computed small example, edge cases (`T` fit on all-one-class labels; empty input raises rather than silently returning 0.0/NaN, matching `model/evaluation/metrics.py::EmptyEvaluationSetError`'s existing convention; single-example input), and reproducibility (same checkpoint + same manifest + same model-config -> identical `T` across two separate runs).
- [ ] `pytest` passes for the full repo test suite, including the new calibration tests, with zero modifications to any existing test file.
- [ ] `git diff --stat` for this step's final commit touches only files under `model/calibration/`, `tests/test_calibration.py`, `tests/test_calibrate_pipeline.py`, `config/calibration_config.yaml`, and (if needed) one `.gitignore` line adding `experiments/calibration/`.

## Risks and limitations

- **R1 - Deliberate code duplication vs. reuse.** `model/calibration/logit_inference.py` duplicates most of `model/evaluation/predict.py::run_inference()`'s batch-loop structure (device transfer, dual-branch dict handling) rather than adding a `return_logits: bool` parameter to the existing function. This trades a small amount of duplication for a hard guarantee that `model/evaluation/` (already verified across Steps 4-7) is never touched by this step - consistent with the repository's own precedent of duplicating small helpers across `predict.py`/`engine.py` rather than adding cross-module coupling. If a future step wants a single shared implementation, that refactor should be proposed and reviewed on its own, not folded into this one.
- **R2 - Inherited real-image-pairing limitation.** Per requirement 18: the unseen-generator evaluation set this step's primary result is computed over pairs `test`'s held-out-generator (BigGAN, Midjourney) fakes with `val`'s real images, because `data.splitting.split_manifest()` never routes real images into `test` (`REAL_PAIRING_NOTE`, `.claude/specs/05-evaluation-pipeline.md`). This is not introduced or worsened by calibration, but it means the calibration metrics (ECE/Brier) on the "unseen-generator" set are computed over the same non-purely-unseen population as every other unseen-generator metric in this repo - both `metrics_before.json` and `metrics_after.json`, and `calibration_report.md`, must carry this note verbatim rather than only linking to it.
- **R3 - Small development shard.** Tiny-GenImage's `val` split (257 records per the Step 7 split) is a small calibration-fitting set; a temperature fit on ~250 examples carries meaningful variance, and ECE itself is a biased, bin-count-sensitive estimator on small samples. `n_bins` is therefore configurable (`config/calibration_config.yaml`) rather than hardcoded, and every artifact states the sample count fit/evaluated over so a reader can judge reliability - this step does not claim the fitted `T` would generalize to the final SIH benchmark's scale or generator distribution.
- **R4 - Temperature scaling's known limitation.** A single scalar `T` assumes uniform miscalibration across the whole confidence range and across generators; it cannot correct calibration that varies by generator or by confidence band (a limitation of temperature scaling generally, not specific to this implementation). If per-generator or per-confidence-band miscalibration is later found to be large (visible in the per-generator breakdown this step already reports), a follow-up step (e.g. per-generator temperature, isotonic regression, or Platt scaling) would be needed - explicitly out of scope here.
- **R5 - Calibration is not a detection-quality improvement.** Because `T` is fit only on `val`, any change in Brier/ECE reflects how well `val`'s calibration transfers to the unseen-generator set, not the detector's own accuracy. This step's report must not describe an ECE/Brier improvement as "the model got better at detecting AI-generated images" - only that its confidence values are better calibrated, consistent with CLAUDE.md's "avoid over-claiming."
- **R6 - Not the final SIH benchmark.** Every number this step produces is against `data/manifests/genimage_dev.csv`, per requirement 17 and consistent with every prior step's disclaimer - never presented as final SIH performance.
