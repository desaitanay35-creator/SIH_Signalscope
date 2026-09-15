# Spec: Backend ML Integration

## Overview

Steps 1-8A produced a trained, calibrated detector (`rgb_frequency_fusion`, checkpoint `experiments/runs/20260915-054703/checkpoints/best.pt`, development unseen-generator ROC-AUC = 0.9396367521, validation ROC-AUC = 0.9069567355, temperature `T = 1.1838611364364624` fit on validation logits per `.claude/specs/08a-probability-calibration.md`) that has so far only run inside `model/`'s own training/evaluation scripts. The FastAPI backend (`backend/app/`) already defines a complete, loosely-coupled detector contract (`ImageDetector` in `backend/app/ml/detector.py`) and preprocessing/calibration pipeline (`backend/app/services/inference_service.py`, `backend/app/ml/calibration.py`), but `ModelLoader.load_detector()` still always returns `StubImageDetector` even when weights exist (`backend/app/ml/model_loader.py:47-52`, explicitly marked `# Integration point for ML model loading`). This step replaces that placeholder with a real detector adapter that wraps the trained `RGBFrequencyFusionModel`, wires it into the existing `ModelLoader`, and corrects a calibration-input gap in the existing backend (see "Model / Architecture Changes") — all without changing the public `ImageDetector` contract, the HTTP API, or any route.

This step is the intended, planned transition out of CLAUDE.md's ML-foundation-phase "Development Boundaries" ("Do not modify `backend/`... Backend and frontend integration will happen after the ML inference pipeline is stable"): the inference pipeline is now stable (trained, evaluated, calibrated), so this spec is the backend-integration step that boundary was deferring. No `frontend/` files are touched, and no new API routes are created.

## Depends on

- Step 4 training pipeline (`model/training/checkpoint.py::load_model_from_checkpoint`, architecture registry) and Step 6 frequency fusion (`model/architectures/fusion_model.py::RGBFrequencyFusionModel`, `model/training/frequency_features.py::compute_log_magnitude_spectrum`) — the trained architecture and checkpoint format this step loads.
- Step 7 ablation (`.claude/specs/07-ablation-experiment.md`) — the production checkpoint `experiments/runs/20260915-054703/checkpoints/best.pt` and its recorded architecture id `rgb_frequency_fusion`.
- Step 8A calibration (`.claude/specs/08a-probability-calibration.md`) — the fixed temperature `T = 1.1838611364364624`, fit only on the validation split, and the invariant that calibration must operate on `raw_logit`, never on an already-sigmoided probability.
- The existing backend scaffold: `backend/app/ml/detector.py` (`ImageDetector`, `StubImageDetector`), `backend/app/ml/model_loader.py` (`ModelLoader`), `backend/app/ml/calibration.py` (`Calibrator`), `backend/app/services/inference_service.py` (`InferenceService`), `backend/app/core/config.py` (`Settings`), `backend/tests/conftest.py` (test fixtures/mock detector).

## Model / Architecture Changes

No changes to `model/architectures/`, `model/training/`, or the checkpoint. This step adds one new **backend adapter class** that composes existing, unmodified model code:

- **New class:** `RGBFrequencyFusionDetector(ImageDetector)` in `backend/app/ml/rgb_frequency_detector.py`.
  - **Construction / loading:** on first use, loads the checkpoint exactly once via `model.training.checkpoint.load_model_from_checkpoint` (imported, not duplicated — satisfies requirement 4), on a device selected by `torch.cuda.is_available()` (CUDA if available, else CPU). Calls `.eval()` immediately after loading and never calls `.train()`. Validates that the checkpoint's recorded `model_config["architecture"]` equals the expected `rgb_frequency_fusion` id (`model.architectures.fusion_model.ARCHITECTURE_ID`) — a mismatch raises a clear, typed error at load time rather than failing obscurely inside `forward()`.
  - **Inputs:** the *same* NCHW RGB tensor `backend/app/services/inference_service.py::InferenceService.preprocess_image` already produces (resized to `MODEL_INPUT_WIDTH`×`MODEL_INPUT_HEIGHT`, scaled to `[0,1]`, then channel-wise normalized with `MODEL_NORM_MEAN`/`MODEL_NORM_STD`). The public backend contract is unchanged: callers still pass one RGB tensor.
  - **Frequency branch construction (new logic, the one non-trivial piece of this step):** `RGBFrequencyFusionModel.forward()` expects a dict `{"rgb": ..., "frequency": ...}`, and the frequency input must be the log-magnitude FFT spectrum of the **un-normalized** `[0,1]` RGB image (`model/training/frequency_features.py::compute_log_magnitude_spectrum`'s documented precondition: "must be called... BEFORE any ImageNet mean/std normalization is applied"). Since the backend tensor arriving at `predict()` is *already* ImageNet-normalized, the detector must first **invert** that normalization (`x * std + mean`, using the same `MODEL_NORM_MEAN`/`MODEL_NORM_STD` values the backend used to normalize, converted to tensors) to reconstruct the `[0,1]` RGB array, then call the existing `compute_log_magnitude_spectrum` (imported from `model/training/frequency_features.py`, not duplicated) per-image in the batch to build the `(N, 1, H, W)` frequency tensor. This denormalize-then-reconstruct step is documented inline with a comment pointing at this spec, since it is the one place a subtle mismatch with the training-time frequency input could silently degrade accuracy without raising an error.
  - **Forward pass:** `with torch.no_grad(): logit = model({"rgb": rgb_tensor, "frequency": frequency_tensor})` → squeeze to a Python float per image (batch size is always 1 from `InferenceService`, but the adapter does not assume that structurally beyond what `ImageDetector.predict()` already implies).
  - **Output (`predict()`):** returns
    ```python
    {
        "raw_ai_probability": float(sigmoid(raw_logit)),
        "raw_real_probability": float(1.0 - sigmoid(raw_logit)),
        "logits": float(raw_logit),
        "model_version": <checkpoint-derived version string>,
    }
    ```
    satisfying the minimum contract (`raw_ai_probability`, `model_version`) and populating both documented optional fields (`raw_real_probability`, `logits`) so calibration can operate on the true raw logit rather than reconstructing it (see the calibration-path fix below).
  - **`is_loaded()`:** `True` only after the checkpoint has successfully loaded and `.eval()` has been called; `False` before that / if loading failed (the adapter is never partially constructed — see "Rollback/failure behavior").
  - **`get_info()`:** returns `{"name": "SignalScope Detector", "version": <model_version>, "task": "real-vs-ai-generated", "loaded": True, "device": "cpu"|"cuda", "architecture": "rgb_frequency_fusion", "checkpoint_path": <str>}` — additive fields only, so `ModelInfoResponse` (`backend/app/schemas/analysis.py`) is unaffected as long as it already tolerates/ignores unknown extra keys (verified in "Files to change" below; if it uses a strict Pydantic model, `ModelInfoResponse` gains the new optional fields rather than breaking).
  - **`explain()`:** returns `None` for this step (per requirement 19 — Grad-CAM is explicitly out of scope and handled separately), preserving the existing contract exactly as `StubImageDetector.explain()` already does.

- **Calibration-path correctness fix (existing code, not new architecture, but required to satisfy requirements 8-10):** `backend/app/ml/calibration.py::Calibrator.apply_calibration` currently takes only a probability and reconstructs a logit via `log(p / (1-p))` before dividing by temperature — mathematically equivalent to `sigmoid(raw_logit / T)` **only if** the incoming probability was produced by an *exact*, undamaged `sigmoid(raw_logit)` with no intervening rounding, and it silently breaks the explicit repo-wide rule ("Calibration must operate on logits, never on already-sigmoided probabilities... there is no code path that takes a probability and attempts to invert it" — `.claude/specs/08a-probability-calibration.md`, "Rules for implementation"). This step adds a logit-based calibration path and prefers it whenever a detector supplies one, removing the round-trip:
  - `Calibrator.apply_calibration_from_logit(raw_logit: float) -> Tuple[float, bool, str]`: when calibration is enabled with `CALIBRATION_METHOD == "temperature"`, computes `calibrated = sigmoid(raw_logit / T)` directly — no `log`/inverse-sigmoid round trip.
  - `InferenceService.run_inference` is updated to call `apply_calibration_from_logit(raw_output["logits"])` when the detector output includes a finite `"logits"` value, and to fall back to the existing `apply_calibration(raw_ai_probability)` path only when a detector does not supply logits (e.g. `StubImageDetector`, which never reaches this code, or any future/mock detector that only reports a probability) — preserving full backward compatibility with `MockLoadedDetector` in `backend/tests/conftest.py`, which does not set `"logits"`.
  - The existing probability-based `apply_calibration` is kept, unchanged in its own behavior, purely as the compatibility fallback — it is not removed, since `StubImageDetector`/mock-detector-based tests depend on it and this step must not require changing those tests (requirement 14).

## Data Changes

No data changes.

## API / Inference Changes

No API changes. No new routes are added and no existing route's request/response shape changes:

- `GET /api/v1/model` (`backend/app/api/routes/model.py`) — unchanged route; `get_info()`'s additive fields (`architecture`, `checkpoint_path`) surface automatically once `ModelInfoResponse` is extended to declare them as optional.
- The analysis endpoint(s) that call `InferenceService.run_inference` (`backend/app/services/analysis_service.py`) are unchanged — they already consume `verdict`, `confidence`, `ai_probability`, `raw_ai_probability`, `threshold`, `model_version`, `is_calibrated`, `calibration_method` from `run_inference`'s return dict, and that dict's shape does not change (only the calibration *path* it takes internally changes, per the fix above).

## Explainability Changes

No explainability changes. `RGBFrequencyFusionDetector.explain()` returns `None`, exactly preserving `StubImageDetector`'s existing behavior — Grad-CAM/evidence extraction is explicitly deferred to a separate step (requirement 19).

## Evaluation Plan

This step does not retrain or re-evaluate the model — Step 7/8A's measured metrics (validation ROC-AUC = 0.9069567355, unseen-generator ROC-AUC = 0.9396367521, temperature `T = 1.1838611364364624`) are carried forward unchanged and must not be re-derived or re-claimed here. The evaluation for *this* step is an **integration-correctness** check, not a model-quality check:

- **Parity check (required, blocking):** for a small fixed set of sample images, the raw logit produced by `RGBFrequencyFusionDetector.predict()` through the full backend preprocessing path (`InferenceService.preprocess_image` → detector) must match, within floating-point tolerance (`atol=1e-4`), the raw logit produced by feeding the same images through `model/evaluation`'s existing dual-branch preprocessing path (`model.training.dual_branch_dataset`) directly. This is the only way to catch a subtle mismatch in the frequency-input reconstruction (denormalize → FFT) described above; it is treated as a correctness test, not an accuracy benchmark, and is part of the test plan (integration test), not a separate "evaluation" script.
- **No new ROC-AUC/accuracy/F1/FPR/confusion-matrix/calibration numbers are computed** in this step, since no data or model changes occur — CLAUDE.md's "Report AUC, macro-F1, confusion matrix, accuracy and FPR" applies to model-development steps, and this step explicitly is not one (requirement 20: no architecture change or retraining).
- **Regression check (required, blocking):** the full existing backend test suite (`pytest backend/tests/`) must continue to pass unmodified, confirming the stub-detector-based contract and API behavior are unaffected.

## Files to change

- `backend/app/ml/model_loader.py` — replace the placeholder branch (lines 47-52) that returns `StubImageDetector` when weights exist with logic that constructs `RGBFrequencyFusionDetector`, catches load failures explicitly, and falls back to `StubImageDetector` **only** when weights are absent (never as a silent fallback for a corrupt/incompatible checkpoint — see "Rollback/failure behavior").
- `backend/app/ml/calibration.py` — add `Calibrator.apply_calibration_from_logit()` (new method; existing `apply_calibration()` and `compute_verdict()` remain unchanged in behavior).
- `backend/app/services/inference_service.py` — update `run_inference()` to prefer the logit-based calibration path when `raw_output["logits"]` is present and finite, otherwise use the existing probability-based path.
- `backend/app/core/config.py` — add settings needed to locate/validate the real checkpoint without hardcoding paths in source: `MODEL_ARCHITECTURE: str = "rgb_frequency_fusion"` (expected architecture id, used for the compatibility check) and reuse the existing `MODEL_WEIGHTS_PATH`/`MODEL_VERSION`/`CALIBRATION_*` settings as-is (no renaming, to avoid breaking `.env`/deployment configs already keyed on them).
- `backend/app/schemas/analysis.py` — extend `ModelInfoResponse` with optional `architecture: Optional[str]` and `checkpoint_path: Optional[str]` fields (defaulted to `None`) so the additive `get_info()` fields serialize without breaking existing consumers or the frontend contract test (`backend/tests/test_frontend_contract.py`).
- `.gitignore` — confirm/add an entry excluding `model/weights/` (or wherever `MODEL_WEIGHTS_PATH` points) and any test-fixture checkpoint output directory, if not already excluded (requirement 17 — never commit checkpoints or generated inference artifacts).

## Files to create

- `backend/app/ml/rgb_frequency_detector.py` — the `RGBFrequencyFusionDetector(ImageDetector)` adapter described above, plus its checkpoint-compatibility validation and frequency-input reconstruction helper (`_denormalize_to_unit_range`, `_build_frequency_input`, both private module-level functions since they are adapter-internal, not general-purpose model code).
- `backend/app/core/model_errors.py` (or inline in `backend/app/core/errors.py` if the team prefers a single errors module — **decision left to implementation, default: extend `errors.py`**) — a `ModelIncompatibleError(AppException)` (or reuse `ModelUnavailableError` with a distinguishing `details` payload — **default: new distinct exception** so "missing weights" (503, expected/transient) and "corrupt/incompatible checkpoint" (500, a real bug) are distinguishable in logs/monitoring) raised when a checkpoint fails to load or declares an unexpected architecture id.
- `backend/tests/fixtures/tiny_fusion_checkpoint.py` — a helper that builds and saves a tiny, randomly-initialized `RGBFrequencyFusionModel` checkpoint (`pretrained=False`, a handful of forward/backward-compatible state, correct `model_config["architecture"] = "rgb_frequency_fusion"` metadata) to a temp path at test time — this is the "test fixture strategy that does not require committing the real production checkpoint" requirement 16 calls for. Never checked into git as a binary; generated on the fly inside `conftest.py`/tests via `tmp_path`.
- `backend/tests/test_rgb_frequency_detector.py` — the unit tests enumerated in requirement 15 (see "Test plan").
- `backend/tests/test_backend_ml_integration.py` — the integration test enumerated in requirement 16 (see "Test plan"), including the parity check against `model/evaluation`'s preprocessing path.

## New dependencies

No new dependencies. `backend/` already depends on `torch`, `numpy`, and `Pillow` (implied by the existing preprocessing/detector code); `model.training.checkpoint`, `model.architectures.fusion_model`, and `model.training.frequency_features` are imported as first-party modules from the existing `model/` package (the backend's Python environment must be able to import `model.*` — confirm/record the path/package configuration that already makes this possible, e.g. a shared virtualenv and repo-root-relative imports; no `pip install` addition is needed either way).

## Rules for implementation

- Do not use the hidden SIH test set for training or tuning — not applicable (no training in this step), noted for completeness.
- Do not introduce data leakage between train and validation/test splits — not applicable (no data/splits touched).
- Prioritize unseen-generator ROC-AUC over raw training accuracy — not applicable; this step must not change any metric, only wire in the already-measured model.
- Use reproducible random seeds — not applicable to inference; the only place randomness could matter is the test fixture checkpoint (`tiny_fusion_checkpoint.py`), which must set a fixed seed so tests are deterministic across runs.
- Keep model configuration separate from source code — checkpoint path, architecture id, and calibration temperature stay in `backend/app/core/config.py` (`Settings`, env-overridable), never hardcoded inside `rgb_frequency_detector.py` or `calibration.py`.
- Do not hardcode dataset paths — not applicable (no dataset access from the backend); the checkpoint path is a configured setting, not a literal in code.
- Record experiment configuration and metrics — not applicable; this step does not run an experiment. `get_info()` does record checkpoint path/architecture for observability.
- Do not claim an improvement without measured comparison — this step's PR/report must state it performs integration only and carries forward Step 7/8A's numbers unchanged, not a new result.
- Explanations must be grounded in model evidence — not applicable (`explain()` returns `None` this step).
- Do not make absolute claims such as "this image is definitely AI-generated"; use responsible wording such as "likely AI-generated" — enforced already by `Calibrator.compute_verdict()` (`likely_ai_generated`/`likely_real`), unchanged by this step; the new adapter must not introduce any user-facing string of its own.
- Do not analyze or identify real people; do not build political or event-claim detection features — not applicable, noted for completeness.
- Do not use unseen/test data to determine integration parameters (requirement 12) — the checkpoint path, architecture id, and temperature `T = 1.1838611364364624` are fixed values already determined in prior steps; this step must not re-fit, re-tune, or re-derive any of them from any dataset.
- Do not tune threshold or temperature during inference (requirement 11) — `DECISION_THRESHOLD` and `CALIBRATION_TEMPERATURE` remain configuration values read once at inference time, never computed or adjusted per-request.
- Never silently fall back to fake/stub predictions when weights are present but broken (requirement 5/13) — a present-but-corrupt or architecture-mismatched checkpoint must raise a clear, typed, logged error from `ModelLoader.load_detector()` at startup, never silently substitute `StubImageDetector` (that substitution is reserved strictly for the "weights file does not exist" case, matching current behavior).
- Preserve the existing `ImageDetector`, `StubImageDetector`, `ModelLoader.get_detector()`/`set_detector()` public interfaces exactly — no signature changes.
- Do not duplicate the FFT/frequency computation or the fusion architecture inside `backend/` — import `model.training.frequency_features.compute_log_magnitude_spectrum` and `model.architectures.fusion_model.RGBFrequencyFusionModel` (via `model.training.checkpoint.load_model_from_checkpoint`) rather than reimplementing them (requirement 4).
- Do not modify `frontend/` in this step (requirement 18).
- Do not add Grad-CAM or any explainability computation in this step beyond preserving `explain() -> None` (requirement 19).
- Do not change `model/architectures/`, `model/training/`, or retrain/re-save the checkpoint (requirement 20).

## Test plan

**Unit tests (`backend/tests/test_rgb_frequency_detector.py`), all against the fixture checkpoint from `tiny_fusion_checkpoint.py`, CPU-only:**

1. Checkpoint loading succeeds and `is_loaded()` becomes `True` after construction.
2. Missing checkpoint path (`MODEL_WEIGHTS_PATH` pointing nowhere) → `ModelLoader.load_detector()` returns `StubImageDetector` with `is_loaded() == False` (existing behavior, verified unchanged).
3. Corrupt checkpoint file (e.g. truncated/garbage bytes at the configured path) → constructing `RGBFrequencyFusionDetector` (or `ModelLoader.load_detector()`) raises the new typed model-incompatibility error, not a raw `torch` exception and not a silent stub fallback.
4. Incompatible checkpoint (a valid `torch.save`d file whose `model_config["architecture"]` is some other id, e.g. `"efficientnet_b4"`) → the same typed error, with a message naming the expected vs. found architecture id.
5. `get_info()` reports `loaded=True`, the correct `model_version`, `architecture="rgb_frequency_fusion"`, and a `device` of `"cpu"` or `"cuda"` matching `torch.cuda.is_available()`.
6. `predict()` on a well-formed NCHW tensor returns `raw_ai_probability` strictly within `[0.0, 1.0]`, and it is a finite Python `float` (not a tensor/ndarray).
7. `predict()`'s output includes a finite `"logits"` field, and `sigmoid(logits) == raw_ai_probability` within `atol=1e-6`.
8. Determinism: two calls to `predict()` on the identical input tensor return bit-for-bit (or `atol=1e-7`) identical `raw_ai_probability`/`logits` — verifies `eval()` mode (no BatchNorm/Dropout stochasticity) and `torch.no_grad()` usage.
9. Inference runs successfully with `torch.cuda.is_available()` mocked/forced `False` (CPU-only path), independent of the machine running CI.
10. Frequency-branch construction: given a known synthetic RGB tensor (already normalized, as the backend would produce it), the internal frequency-input helper produces a `(N, 1, H, W)` tensor whose values lie in `[0, 1]` (matching `compute_log_magnitude_spectrum`'s documented output range) and whose spectrum, after manually inverting the normalization and calling `compute_log_magnitude_spectrum` directly on the recovered `[0,1]` array, matches within `atol=1e-5`.
11. `Calibrator.apply_calibration_from_logit()`: for a known logit and `T = 1.1838611364364624`, the returned calibrated probability equals `sigmoid(logit / T)` computed independently (hand-checked or `math`-computed in the test), and differs from naively reconstructing a logit from an already-rounded probability (regression guard against reintroducing the round-trip).
12. Threshold decision consistency: `Calibrator.compute_verdict(calibrated_probability, threshold)` produces the same verdict/confidence whether the calibrated probability was produced via the logit path or (for a probability that exactly equals `sigmoid(some_logit)`) via the legacy probability path — confirms the two paths agree on well-behaved inputs.
13. No fake/hardcoded prediction behavior: assert that `predict()` output values change when the input tensor changes (using two different synthetic fixture tensors) and that `RGBFrequencyFusionDetector` never returns a fixed constant regardless of input (guards against an accidental stub-like shortcut).

**Integration test (`backend/tests/test_backend_ml_integration.py`):**

- End-to-end: a synthetic decoded PIL image → `InferenceService.preprocess_image` → `RGBFrequencyFusionDetector.predict()` (loaded from the fixture checkpoint, injected via `ModelLoader.set_detector()`) → `InferenceService.run_inference()` → assert the returned dict has `is_calibrated=True`, `calibration_method="temperature_scaling"` (or the existing method string `apply_calibration_from_logit` reports), and `0.0 <= ai_probability <= 1.0`.
- Parity check (see "Evaluation Plan"): the raw logit from the above path matches the raw logit obtained by running the same source image through `model/training/dual_branch_dataset`'s existing preprocessing + the same fixture checkpoint's model, within `atol=1e-4`.
- No test in this file loads or requires `experiments/runs/20260915-054703/checkpoints/best.pt` or any other committed production checkpoint — only the on-the-fly fixture from `tiny_fusion_checkpoint.py`.

**Regression:**

- `pytest backend/tests/` (full existing suite: `test_analysis.py`, `test_e2e_integration.py`, `test_frontend_contract.py`, `test_health.py`, `test_image_validation.py`, `test_security_reliability.py`) passes unmodified.
- `pytest` (full repo suite, including `model/`'s existing tests) passes unmodified — this step touches no file under `model/`.

## Rollback/failure behavior

- **Weights file absent at `MODEL_WEIGHTS_PATH`:** unchanged from current behavior — `ModelLoader.load_detector()` returns `StubImageDetector`, `is_loaded() == False`, and any `predict()` call raises `ModelUnavailableError` (HTTP 503) exactly as today.
- **Weights file present but corrupt (fails `torch.load`) or declares an incompatible architecture:** `ModelLoader.load_detector()` must **not** return `StubImageDetector` and must **not** return a partially-constructed detector. It raises the new typed error (`ModelIncompatibleError` or equivalent) with a message identifying the checkpoint path and the specific failure (deserialization error vs. architecture mismatch vs. `state_dict` shape mismatch), which propagates and fails application startup loudly — consistent with requirement 5 ("fail clearly... never silently fall back to fake predictions"). This is a deliberate behavior change from today's placeholder (which returns `StubImageDetector` unconditionally once weights exist) and must be called out in the PR description as an intentional, spec-required change.
- **A previously-loaded detector begins raising unexpected errors during `predict()` at runtime (e.g. an out-of-memory error on CUDA):** the adapter does not catch and mask this — it propagates up through `InferenceService.run_inference()` to the existing `InferenceError` handling already in place in `backend/app/services/analysis_service.py`/route-level exception handlers (unchanged in this step), resulting in an HTTP 500 with the existing `INFERENCE_ERROR` code. No new exception-swallowing is introduced.
- **Rolling back this step:** since the singleton `ModelLoader._instance` is created lazily and `set_detector()` already exists for overriding it, an operator can force `StubImageDetector` behavior without a code revert by pointing `MODEL_WEIGHTS_PATH` at a nonexistent path (env var change only) — a documented mitigation if `RGBFrequencyFusionDetector` needs to be disabled in production without a redeploy.

## Definition of done

- [ ] `backend/app/ml/rgb_frequency_detector.py` exists, implements `ImageDetector` fully, and imports (not duplicates) `model.training.checkpoint.load_model_from_checkpoint`, `model.architectures.fusion_model` (for the architecture id constant), and `model.training.frequency_features.compute_log_magnitude_spectrum`.
- [ ] `ModelLoader.load_detector()` returns a loaded, working `RGBFrequencyFusionDetector` when `MODEL_WEIGHTS_PATH` points at a valid `rgb_frequency_fusion` checkpoint (verified against the production checkpoint `experiments/runs/20260915-054703/checkpoints/best.pt` in a manual/local smoke check — not committed to the test suite, since the checkpoint itself is not committed).
- [ ] `ModelLoader.load_detector()` still returns `StubImageDetector` when weights are absent, and raises a typed, clearly-logged error (never a silent stub) when weights are present but corrupt or architecture-mismatched.
- [ ] `predict()` returns `raw_ai_probability` in `[0.0, 1.0]`, a `model_version` string, and a finite `logits` field satisfying `sigmoid(logits) == raw_ai_probability`.
- [ ] `Calibrator.apply_calibration_from_logit()` exists and computes `sigmoid(raw_logit / 1.1838611364364624)` exactly, and `InferenceService.run_inference()` uses it whenever a detector supplies `logits`, falling back to the existing probability-based path otherwise.
- [ ] Threshold decisions (`compute_verdict`) are computed from the calibrated probability in both paths, unchanged in logic from today.
- [ ] All 13 unit tests and the integration test (including the preprocessing-parity check) pass on CPU, using only the on-the-fly fixture checkpoint — no production checkpoint is committed or required by `pytest`.
- [ ] `pytest backend/tests/` and the full repo `pytest` both pass with zero modifications to any existing test file.
- [ ] `git diff --stat` for this step's final commit touches only: `backend/app/ml/model_loader.py`, `backend/app/ml/calibration.py`, `backend/app/ml/rgb_frequency_detector.py` (new), `backend/app/services/inference_service.py`, `backend/app/core/config.py`, `backend/app/core/errors.py` (or a new `model_errors.py`), `backend/app/schemas/analysis.py`, `backend/tests/fixtures/tiny_fusion_checkpoint.py` (new), `backend/tests/test_rgb_frequency_detector.py` (new), `backend/tests/test_backend_ml_integration.py` (new), and `.gitignore` (if a new exclusion is needed) — no file under `model/`, `frontend/`, or `experiments/` is modified, and no checkpoint or generated inference artifact is committed.
