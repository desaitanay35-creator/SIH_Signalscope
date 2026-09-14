# Spec: Frequency-Domain Forensic Features + RGB/Frequency Fusion

## Overview

Step 6 designs a lightweight frequency-domain branch and an RGB+frequency fusion model, to be evaluated against the existing EfficientNet-B4 RGB baseline (Step 4/5) via a controlled three-way ablation. The research question is not "does adding FFT help because the problem statement mentions it" - it is: **does frequency-domain forensic information provide complementary evidence that improves unseen-generator generalization compared with the RGB baseline?** The spec is designed so that question can be answered with a fair, leakage-safe, reproducible comparison using the existing Step 4 training engine and Step 5 evaluation pipeline, with only the minimal, clearly-justified extensions needed to support more than one architecture.

This is a design-only document. No source code, tests, or configuration files are created by this step.

## Goals

- Define one concrete, defensible frequency representation (not several, for complexity's own sake) and its exact, deterministic construction from the existing RGB preprocessing pipeline.
- Define a lightweight frequency CNN branch and one RGB+frequency fusion architecture, using EfficientNet-B4's actual (inspected, not guessed) feature dimension.
- Extend the model-loading/checkpoint interface to support multiple architectures **without breaking** any Step 4 or Step 5 behavior for the existing RGB-only checkpoints.
- Design three comparable experiments (RGB-only, frequency-only, RGB+frequency fusion) that reuse the Step 4 training engine and Step 5 evaluation pipeline unchanged in their metric logic, with unseen-generator ROC-AUC as the primary comparison metric.
- Keep every generator-disjoint, test-set-isolation, and reproducibility guarantee already established in Steps 1, 4, and 5 fully intact.

## Non-goals

- Not implementing Grad-CAM or any explainability output in this step (Step 7's concern - see "Explainability forward-look" for how this step should set that up without doing it).
- Not adding calibration, threshold tuning, or test-time augmentation.
- Not adding DCT, wavelet, or multi-scale frequency representations - one representation only, with the alternatives documented and explicitly deferred.
- Not modifying `backend/`, `app/`, `ui/`, or `explainability/`.
- Not claiming Tiny-GenImage ablation results represent final SIH benchmark performance, regardless of outcome.
- Not drawing a causal conclusion ("frequency information generally helps/hurts forensic detection") from one dataset's ablation result - see "How to interpret the experiment."

## Existing components to reuse (inspected, not guessed)

- `model/architectures/efficientnet_b4.py::EfficientNetB4Baseline` - reused as-is. Inspected directly: `backbone.classifier` is `Sequential(Dropout(p=0.4), Linear(in_features=1792, out_features=1))`; `backbone.features(x)` followed by `backbone.avgpool(...)` and flatten produces the **1792-dimensional** pooled embedding actually consumed by that final `Linear`. This number (1792) is used verbatim below - it is EfficientNet-B4's real feature width, not an invented one.
- `data.preprocessor.ImagePreprocessor` - `load_image()`, `target_size`, `mean`/`std` are reused directly; the module itself is not modified.
- `model.training.augmentation.build_train_transform(preprocessor)` and `tensorize(image, preprocessor)` - reused verbatim so the RGB branch's augmentation and normalization are byte-identical to Step 4's baseline, guaranteeing a fair comparison.
- `data.dataset_loader.SignalScopeDataset` - reused unchanged. Its `preprocessor` argument only needs a `.preprocess(image_path) -> Any` method (already how Step 4's `AugmentedPreprocessor` duck-types augmentation in); Step 6 uses the same technique so **no change to `data/dataset_loader.py` is needed at all**, even though a dual-branch sample must carry two tensors (see "Input construction").
- `data.splitting.split_manifest` / `check_generator_leakage` - reused unchanged and called exactly once per experiment, with the same manifest, seed, and split config across all three experiments (see "Fair comparison").
- `model.training.engine.train_one_epoch` / `validate` - reused with one small, additive, backward-compatible change (see "Training strategy" / "Files to modify").
- `model.training.checkpoint.save_checkpoint` / `load_model_from_checkpoint` / `load_training_checkpoint` - reused with one small, additive, backward-compatible extension to support more than one architecture (see "Model-loading interface").
- `model.evaluation.evaluate.run_evaluation`, `model.evaluation.metrics`, `model.evaluation.report` - reused **unchanged in their metric logic**. `evaluate.py` needs one small, additive extension to pick the right dataset-construction path per architecture (see "Evaluation integration"); `metrics.py`/`report.py` need zero changes, since they already operate on plain `(label, probability)` sequences regardless of what produced them.
- `config/model_config.yaml` / `config/training_config.yaml` / `model/training/config.py` - the split, preprocessing, and default hyperparameter conventions are reused; Step 6 adds one new, separate config file rather than editing these (see "Configuration").

## 1. Frequency representation

**Chosen representation: grayscale, log-magnitude, fftshift-centered 2D FFT spectrum. Nothing else.**

Rejected alternatives and why:
- **Per-channel FFT (3 spectra)** - triples frequency-branch input size and parameter count for a baseline ablation whose job is to test *whether frequency information helps at all*, not to find the best possible frequency encoding. Luminance already carries the dominant up-sampling/periodic-artifact signal that generator fingerprints exploit; per-channel FFT is a reasonable *future* refinement once the baseline ablation result is known, not a day-one requirement.
- **Raw (non-log) magnitude** - FFT magnitude has an extreme dynamic range dominated by the DC/low-frequency component; a CNN trained directly on raw magnitude would need to learn to compress that range itself, wasting capacity and hurting numerical stability in a small, lightweight branch.
- **DCT or wavelet transforms** - both are legitimate forensic tools, but adding a second and third representation before knowing whether the *simplest* one (FFT log-magnitude) already carries signal is exactly the "add complexity for its own sake" failure mode this step is explicitly designed to avoid. Deferred to a later, explicitly-scoped step if FFT alone proves inconclusive.
- **Radial-profile reduction (1D radial energy vector)** - discards spatial layout of spectral peaks entirely; a small 2D CNN over the 2D spectrum can learn radial and angular structure if present, whereas a hand-reduced 1D radial vector cannot recover angular information a CNN might use. Not chosen for the baseline, but noted as a much cheaper future ablation of the frequency branch itself.

This is the smallest representation that still preserves the two things generator-fingerprint literature repeatedly points to: **radial energy falloff** (natural images vs. upsampling-checkerboard artifacts) and **discrete spectral peaks** (periodic upsampling/deconvolution artifacts) - both visible in a single grayscale log-magnitude spectrum, both destroyed by reducing to a 1D profile, and both requiring no extra channels or transforms to be observable in principle.

## 2. Input construction

Given an already-resized, already-augmented (train) or already-resized-only (val/test) RGB image as a `(H, W, 3)` float array in `[0, 1]` (the same array `data/preprocessor.py::ImagePreprocessor.preprocess()` produces just before its `transpose`+normalize step - see "Augmentation interaction" for exactly which array this is, shared with the RGB branch):

1. **Grayscale conversion**: `gray = 0.299*R + 0.587*G + 0.114*B` (standard ITU-R BT.601 luminance weights) → `(H, W)` float32 array in `[0, 1]`. Deterministic, no learned parameters.
2. **2D FFT**: `F = numpy.fft.fft2(gray)` → complex `(H, W)` array. Using `numpy.fft` (not `torch.fft`) keeps the frequency transform in the same numpy-first preprocessing layer as `data/preprocessor.py`, consistent with the existing repo convention of doing all deterministic image math in numpy and only producing a `torch.Tensor` as the final step.
3. **Center the spectrum**: `F_shifted = numpy.fft.fftshift(F)` - zero-frequency (DC) component moved to the center, matching how every FFT-based forensic visualization in the literature is presented, and making the "radial distance from center = frequency magnitude" interpretation direct.
4. **Magnitude**: `magnitude = numpy.abs(F_shifted)` → real, non-negative `(H, W)` float64 array.
5. **Log transform for numerical stability and dynamic-range compression**: `log_magnitude = numpy.log1p(magnitude)` (i.e. `log(1 + magnitude)`). `log1p` is used specifically because `magnitude` is always `>= 0` and can legitimately be exactly `0` for a uniform (constant-color) region - `log1p(0) = 0`, so no epsilon hack or `log(magnitude + eps)` is needed and no `-inf`/`NaN` can occur.
6. **Per-image min-max normalization**: `normalized = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min() + 1e-8)` → `[0, 1]` float32. Per-image (not dataset-level running statistics) is chosen deliberately: it requires no separate "fit on train split" step (which would itself be a subtle leakage surface to get right - computing normalization statistics from the full manifest instead of train-only would leak test-set distribution information into every sample's normalization), it is trivially deterministic and reproducible given only the one image, and it is the simplest design that still keeps values in a stable, boundedrange for a CNN with batchnorm. A future refinement could use train-split-only running statistics if per-image normalization proves to wash out consistent, comparable-across-images spectral energy levels - flagged as a deferred decision, not a defect, in "Risks."
7. **Output tensor**: `torch.from_numpy(normalized).unsqueeze(0)` → shape **`(1, H, W)`**, `float32`. `H, W` are exactly the RGB pipeline's post-resize dimensions (`data_config.preprocessing.image_size`, currently `[224, 224]`) - no separate frequency-specific resize step is introduced, so the frequency and RGB tensors are always spatially co-registered representations of literally the same augmented image, satisfying the "same underlying image" requirement by construction rather than by convention.

Determinism: every step above is a pure function of the input pixel array - no randomness, no learned parameters, no data-dependent (e.g. dataset-mean) normalization. Given the same (already-augmented) image, the frequency tensor is bit-for-bit reproducible.

Numerical stability: `float64` is used internally for the FFT/magnitude/log steps (numpy's default for these operations) and only cast to `float32` at the final tensor-construction step, avoiding intermediate precision loss during the log/min-max computation on a wide-dynamic-range signal.

## 3. Frequency branch architecture

A small, purpose-built CNN - not a repurposed classification backbone:

```text
Input:  (N, 1, 224, 224)  - log-magnitude FFT spectrum
Conv2d(1, 16, kernel=7, stride=2, padding=3) -> BatchNorm2d(16) -> ReLU   -> (N, 16, 112, 112)
MaxPool2d(2)                                                              -> (N, 16, 56, 56)
Conv2d(16, 32, kernel=3, stride=1, padding=1) -> BatchNorm2d(32) -> ReLU  -> (N, 32, 56, 56)
MaxPool2d(2)                                                              -> (N, 32, 28, 28)
Conv2d(32, 64, kernel=3, stride=1, padding=1) -> BatchNorm2d(64) -> ReLU  -> (N, 64, 28, 28)
AdaptiveAvgPool2d(1) -> flatten                                           -> (N, 64)          <- D2 = 64
```

- **Input channels**: 1 (single-channel log-magnitude spectrum - see "Frequency representation").
- **Depth/width**: 3 conv blocks, channel progression 1→16→32→64. Deliberately shallow and narrow: spectral forensic signal (radial energy falloff, discrete periodic peaks) is a comparatively low-complexity, largely global pattern, not the deep hierarchical texture/shape structure natural-image CNNs are built for - a shallow network with global average pooling is standard practice in the frequency-forensics literature for exactly this reason, and avoids over-parameterizing a single-channel input relative to the amount of information it actually carries.
- **Normalization**: BatchNorm2d after every conv, before activation - stabilizes training given the branch has no pretrained initialization (frequency spectra have no ImageNet analogue to pretrain against).
- **Activation**: ReLU throughout - no need for anything more elaborate in a 3-layer branch.
- **Pooling**: strided conv + one explicit MaxPool2d for early downsampling (spectra are smooth/global enough that aggressive early downsampling loses little), `AdaptiveAvgPool2d(1)` at the end (matches EfficientNet-B4's own global-average-pooling convention, keeping both branches' embeddings on comparable footing - a spatial average, not a flattened spatial grid).
- **Output embedding dimension**: **D2 = 64**.
- **Parameter count** (approximate, for the CPU-feasibility argument): on the order of 30,000-40,000 parameters - roughly three orders of magnitude smaller than EfficientNet-B4's ~19M, so this branch's forward/backward cost is negligible next to the RGB branch's; it does not meaningfully change the frequency-only or fusion model's CPU smoke-test time relative to Step 4's existing RGB-only smoke test.

Why appropriate for the development hardware (Intel i5-1334U, ~7.6 GB RAM, no CUDA): three small conv layers over a single-channel `224x224` input, with two `stride=2`-equivalent downsamplings before the deepest (64-channel) layer, keeps both the activation memory footprint and FLOP count trivial relative to EfficientNet-B4 - this branch will never be the CPU/RAM bottleneck in any of the three experiments; the RGB branch (unchanged from Step 4) remains the bottleneck, exactly as it already is today.

## 4. RGB branch

Reused exactly as Step 4/5 built it, with **no changes to `model/architectures/efficientnet_b4.py`**:

- For **Experiment 1 (RGB-only)**: `EfficientNetB4Baseline` is used precisely as-is - this is not a new model, it is literally re-running Step 4's training pipeline (see "Ablation design").
- For **Experiment 3 (fusion)**: the fusion model holds an `EfficientNetB4Baseline` instance as `self.rgb_branch` and calls `self.rgb_branch.backbone.features(x)` then `self.rgb_branch.backbone.avgpool(...)` then flattens, **bypassing `self.rgb_branch.backbone.classifier` entirely** (that Sequential's `Dropout(0.4)` + final `Linear(1792, 1)` are never invoked in the fusion model - the fusion head, described below, replaces that classifier's role at the fusion embedding's width, not at 1792 alone).
- **Pretrained weights**: used by default (`pretrained=True`), matching Step 4's own default in `config/model_config.yaml`, so the fusion model's RGB half starts from the exact same initialization as the RGB-only baseline - a deliberate fairness choice (see "Fair comparison").
- **Trainable layers**: all RGB backbone layers trainable by default (`freeze_backbone_epochs=0`, matching Step 4's baseline default) - the fusion model's config still exposes the same `freeze_backbone_epochs` knob Step 4 defined, for later experimentation, but the baseline ablation run does not exercise it, again to keep Experiment 1 vs. Experiment 3's RGB-side training dynamics as comparable as possible.
- The existing classifier head is **not reused** for the fusion model (a 1792→1 linear layer has no meaningful role once its input is concatenated with a second embedding) - it is replaced by the fusion head described in "Feature fusion." This does not touch `efficientnet_b4.py`; it simply means the fusion model's own module does not include that `Sequential` in its computation graph.

## 5. Feature fusion

Conceptual comparison (for the record - only one is implemented):

- **Concatenation**: simplest, zero extra trainable fusion-specific parameters beyond the head itself, most directly interpretable (the head's first-layer weights can later be inspected per-source-embedding), and the standard first move in multi-modal/multi-branch forensic literature before anything more elaborate is justified.
- **Weighted sum**: requires the two embeddings to share the same dimensionality (they do not here - 1792 vs. 64) or a projection step to force them to match, which adds complexity and an extra design decision (which projection?) without a clear benefit at this stage.
- **Gated fusion**: strictly more expressive than concatenation, but introduces a learned gate that could itself absorb dataset-specific bias (exactly the failure mode "How to interpret the experiment," Case D, warns about) - premature before knowing if unweighted concatenation already shows a signal.
- **Attention fusion** (e.g. cross-attention between spatial RGB features and spectral features): the most expressive and the most likely to overfit a 2,000-image development shard, and the hardest to interpret when asking "did frequency information help, or did the fusion mechanism just learn to memorize".

**Chosen: concatenation.** Simplest method that is experimentally defensible - any observed improvement (or lack thereof) is attributable to the frequency embedding's information content, not to an expressive fusion mechanism's own capacity.

Exact dimensions (using the real, inspected EfficientNet-B4 width, not an invented one):

```text
RGB embedding      D1 = 1792   (EfficientNet-B4 backbone.features -> backbone.avgpool -> flatten)
Frequency embedding D2 = 64    (frequency branch, "Frequency branch architecture" above)
Fusion vector       D1 + D2 = 1856
Fusion head:  Dropout(p=0.4) -> Linear(1856, 1)   -> one raw logit, sigmoid(logit) = P(ai_generated)
```

The fusion head deliberately mirrors `EfficientNetB4Baseline`'s own head shape (`Dropout(0.4)` then a single `Linear` to one logit) rather than adding hidden layers - so that the fusion model's only *architectural* difference from the RGB-only baseline is the presence of the frequency embedding, not a larger or deeper classifier head. This keeps the ablation clean: any accuracy/ROC-AUC difference between Experiment 1 and Experiment 3 is attributable to the added frequency information, not to added head capacity.

## Model-loading interface for multiple architectures

This is the one place a *training-infrastructure* file genuinely needs to change, and it is scoped narrowly and kept fully backward compatible.

**Problem**: `model.training.checkpoint.save_checkpoint()` currently calls the free function `model.architectures.efficientnet_b4.get_model_spec(...)` unconditionally to build the checkpoint's `model_config` field, and `load_model_from_checkpoint()` unconditionally constructs `EfficientNetB4Baseline(pretrained=False)` regardless of what a checkpoint actually contains. Neither of these works once a checkpoint might hold a frequency-only or fusion model.

**Design**:
- Every architecture class (the existing `EfficientNetB4Baseline`, and the two new classes this step introduces) exposes a `.get_spec() -> dict` instance method that includes at minimum `{"architecture": <string id>, ...}`. `EfficientNetB4Baseline.get_spec()` already exists and already does this (`architecture: "efficientnet_b4"`, from `config/model_config.yaml`) - no change needed there.
- `save_checkpoint()` changes its one line from `get_model_spec(pretrained=...)` to `model.get_spec()` (duck-typed - it does not need to import or know about any specific architecture class). This is the only change to that function's body.
- `load_model_from_checkpoint()` gains a small, internal architecture registry keyed by the `architecture` string recorded in the checkpoint's `model_config`, defaulting to `"efficientnet_b4"` when that key is absent (true of every checkpoint saved before this step, since `get_model_spec()` has always set it) - so every existing Step 4/5 checkpoint continues to load exactly as it does today, byte-for-byte, with zero behavior change. The two new architecture ids (`"frequency_branch"`, `"rgb_frequency_fusion"`) are imported **lazily inside the function body**, not at module load time, so `model/training/checkpoint.py` does not gain a hard import-time dependency on the new architecture files for code paths that never use them (e.g. Step 4's existing RGB-only training run).
- `model/evaluation/evaluate.py::run_evaluation` needs one small, additive change: after loading the checkpoint, branch on the *same* `architecture` string (already available from `load_training_checkpoint(...)["model_config"]["architecture"]`, already read into `training_provenance` today) to decide whether to build datasets via the existing `model.training.dataset.build_val_dataset` (RGB-only) or the new dual-branch equivalent (frequency-only/fusion - see "Input construction for training/evaluation" below). This is a dispatch on an already-available string, not a redesign of the evaluation flow.
- `model.training.engine.train_one_epoch` / `validate` and `model.evaluation.predict.run_inference` each currently do `images = images.to(device)`, assuming `images` is a single tensor. A dual-branch batch is a `{"rgb": Tensor, "frequency": Tensor}` dict (torch's default `collate_fn` already batches a dict of per-sample tensors correctly with no custom collate function needed - verified against the existing `SignalScopeDataset.__getitem__`'s `(tensor, label, meta)` contract, where "tensor" can be *any* object the injected preprocessor's `.preprocess()` returns, dict included). Each of these three call sites needs its one `.to(device)` line replaced with a tiny helper that moves either a plain tensor or every value in a dict of tensors - a five-line, purely additive utility, not a redesign of any of the three functions.

This is deliberately the **entire** scope of change to already-shipped training/evaluation code: `model/training/checkpoint.py` (small, additive, backward-compatible), `model/training/engine.py` (one device-move helper, used in two functions), `model/evaluation/predict.py` (the same helper, used once), `model/evaluation/evaluate.py` (one dispatch on an already-available string). No change to `model/training/dataset.py`'s existing functions, `data/`, `config/model_config.yaml`, `model/evaluation/metrics.py`, or `model/evaluation/report.py`.

## Input construction for training/evaluation (dataset wiring)

New, additive-only file `model/training/dual_branch_dataset.py` (does not modify `model/training/dataset.py`):

- `AugmentedDualBranchPreprocessor` - mirrors Step 4's `AugmentedPreprocessor` exactly in spirit: `.preprocess(image_path)` loads the image via the shared `ImagePreprocessor.load_image()`, applies the **same** `build_train_transform(preprocessor)` augmentation Step 4 already uses (byte-identical augmentation policy, not a re-derived one), and from that **one** augmented image produces both the RGB tensor (via the existing `tensorize()`, i.e. ImageNet-normalized) and the frequency tensor (via the new frequency-transform function, computed on the pre-normalization `[0,1]` array - see "Input construction," step 1, which explicitly operates on the un-normalized pixel array, never on ImageNet-normalized values, since channel-wise mean/std normalization would distort the spectrum in a way that has no physical/forensic meaning). Returns `{"rgb": rgb_tensor, "frequency": freq_tensor}`.
- `DeterministicDualBranchPreprocessor` - the val/test equivalent: same two-derivation logic, but the shared image is only resized (via `image.resize(preprocessor.target_size, ...)`, matching `ImagePreprocessor.preprocess()`'s own deterministic behavior), never augmented.
- `build_dual_branch_train_dataset(records, preprocessor, root_dir=None)` / `build_dual_branch_val_dataset(records, preprocessor, root_dir=None)` - thin wrappers around the existing, unmodified `data.dataset_loader.SignalScopeDataset`, exactly matching the existing `build_train_dataset`/`build_val_dataset` signatures in `model/training/dataset.py`, just injecting the dual-branch preprocessor instead. `SignalScopeDataset` itself needs no awareness that its "tensor" is now a dict.
- New pure function `model/training/frequency_features.py::compute_log_magnitude_spectrum(rgb_array_0_1) -> torch.Tensor` implements exactly the seven steps in "Input construction" above, and is the single place both the training dual-branch preprocessors and any future standalone tooling call into - not duplicated.

## 6. Training strategy

- **Loss**: `nn.BCEWithLogitsLoss()` - unchanged from Step 4, identical across all three experiments (no `pos_weight`, matching the balanced dev shard, exactly as Step 4's baseline).
- **Optimizer**: `torch.optim.AdamW` - unchanged from Step 4.
- **Learning rate / weight decay**: identical to Step 4's baseline defaults (`lr=1e-4`, `weight_decay=1e-4`) for Experiments 1 and 3's RGB portion and the fusion head; the frequency branch (Experiments 2 and 3) is trained with the **same** optimizer instance and **same** learning rate as the rest of the model - no separate per-branch learning rate in the baseline design, to keep the comparison simple and avoid a confound where a tuned frequency-branch LR (rather than the frequency information itself) explains any observed difference. A per-branch LR is a legitimate future refinement, explicitly deferred.
- **Scheduler**: `CosineAnnealingLR` over `epochs` - unchanged from Step 4, identical across all three experiments.
- **Epochs**: identical epoch budget across all three experiments (see "Fair comparison") - not decided per-experiment based on convergence, specifically to avoid implicitly giving one architecture a larger training budget.
- **Joint vs. staged training (fusion model)**: **jointly trained end-to-end from epoch 1** - both branches and the fusion head share one optimizer and one backward pass per batch. Staged training (e.g. pretrain the frequency branch alone, then freeze it while training the fusion head) is not used in the baseline design: it would introduce an extra hyperparameter (how long to pretrain which part) that has nothing to do with the actual research question, and joint training is the simpler, more standard starting point for a first fusion ablation.
- **Initialization**: RGB branch from ImageNet pretrained weights (as today); frequency branch from PyTorch's default `Conv2d`/`BatchNorm2d` initialization (no pretrained frequency-domain weights exist to use - there is no equivalent of ImageNet for FFT spectra); fusion head (`Linear(1856, 1)`) from PyTorch's default `Linear` initialization, exactly as `EfficientNetB4Baseline`'s own final layer already is (that layer is "freshly initialized" per its own docstring today) - so the fusion head is not disadvantaged relative to the existing baseline's head in terms of initialization novelty.
- **Freezing/unfreezing**: none by default in the baseline ablation (see "RGB branch" above) - `freeze_backbone_epochs` remains available as a config knob for later experiments, not exercised in Experiments 1-3 as specified here.
- **Not invalidating comparison with Step 4**: Experiment 1 is not a "new" training run with different code - it is Step 4's existing `model/training/train.py` invoked with the exact same config it already uses today. The only new training-time code path is the one exercised by Experiments 2 and 3 (dual-branch dataset + non-EfficientNet-B4 model construction), gated behind an explicit, additive `model_type` config field (see "Files to modify," `model/training/train.py`) that defaults to the value reproducing today's exact behavior - so re-running Step 4's own tests and Step 4's own smoke test after this step's implementation must still produce identical results.

## 7. Augmentation interaction

Inspected: Step 4's augmentation (`model/training/augmentation.py`) is: `RandomHorizontalFlip(p=0.5)` → `RandomRotation(degrees=10)` → `RandomResizedCrop(size=target_size, scale=(0.9, 1.0))`, applied to the **PIL image**, before `tensorize()` (which does resize-to-array, `/255`, transpose, ImageNet-normalize).

For the dual-branch case:
- The **same** three augmentation transforms, with the **same** parameters, are applied **once** to the loaded PIL image - producing one augmented PIL image that both branches derive from. This directly satisfies "RGB and frequency branches see the SAME underlying augmented image": it is not two independently-augmented copies that happen to use the same random parameters, it is literally one augmented image object, read twice.
- **RGB branch**: `tensorize(augmented_image, preprocessor)` - ImageNet-normalized, exactly as today.
- **Frequency branch**: the augmented image converted to a `[0,1]` numpy array (grayscale conversion happens on this array, per "Input construction") - explicitly **before** ImageNet mean/std normalization, since that normalization is channel-wise and would distort the frequency content in a way with no physical meaning (a spectrum computed from a per-channel-shifted-and-scaled image is not a meaningful "spectrum of the image" in the forensic sense).
- **No new augmentations are introduced.** JPEG re-compression, blur, and noise augmentations remain explicitly out of scope, exactly as Step 4's spec already reasoned: these either destroy or fabricate the very compression/frequency artifacts the frequency branch exists to detect, which would be actively counterproductive here specifically.
- **Validation/test**: no augmentation for either branch, exactly as Step 4 - the deterministic dual-branch preprocessor only resizes.

## 8. Data leakage

No new leakage surface is introduced, and all Step 1/4/5 safeguards are inherited unchanged:

- `data.splitting.split_manifest` and `check_generator_leakage` are called exactly once per experiment run, on the same manifest, with the same `SplitConfig`, before any model (RGB, frequency, or fusion) is constructed - splitting logic is never touched or reimplemented by this step.
- The frequency transform is a deterministic, per-image function computed **after** the train/val/test assignment already exists (it operates on one already-split record's image at dataset-`__getitem__` time) - it has no access to, and no dependency on, any other image's statistics (per-image min-max normalization, not dataset-level - see "Input construction," step 6, which explicitly names this as the reason for that choice).
- No frequency-branch or fusion-model hyperparameter, checkpoint-selection rule, or threshold is ever chosen using `test`-split predictions, exactly matching Step 5's existing rules - this applies identically to Experiment 2 and Experiment 3's checkpoint selection (validation ROC-AUC only, per Step 4's existing best-checkpoint rule, unchanged).
- Step 5's evaluation-time hard checks (`GeneratorLeakageError`, `EmptyUnseenGeneratorSplitError`, the real-image-from-val pairing documented in `.claude/specs/05-evaluation-pipeline.md`) apply identically regardless of which architecture produced the checkpoint being evaluated - none of that logic is architecture-specific.

## 9. Ablation design

All three experiments share: the same manifest (`data/manifests/genimage_dev.csv` for development; explicitly not claimed as a final benchmark - see "Non-goals"), the same `SplitConfig` (same seed, same `val_fraction`, same `unseen_generators` - either the same explicit list or the same `held_out_generator_fraction`-derived selection, verified identical by logging the actual resolved generator list, exactly as Step 5 already does), the same random seed (`TrainingConfig.seed`), the same epoch budget, the same checkpoint-selection rule (best validation ROC-AUC, Step 4's existing rule, unchanged), and the same evaluation protocol and threshold (Step 5's `run_evaluation`, threshold fixed at `ModelConfig.threshold`, never tuned on test).

| | Experiment 1: RGB-only | Experiment 2: Frequency-only | Experiment 3: RGB+Frequency fusion |
|---|---|---|---|
| Model | `EfficientNetB4Baseline` (Step 4, unchanged) | new `FrequencyOnlyModel` (frequency branch + `Dropout(0.4)`+`Linear(64,1)` head) | new `RGBFrequencyFusionModel` (both branches, concatenation, fusion head) |
| Dataset construction | existing `build_train_dataset`/`build_val_dataset` | new `build_dual_branch_*_dataset` (frequency tensor only consumed; RGB tensor present but unused by this model, or a frequency-only preprocessor variant that skips the RGB derivation entirely for efficiency - implementation detail, not a design fork) | new `build_dual_branch_*_dataset` |
| `model_type` config | `"rgb_only"` (default, reproduces Step 4 exactly) | `"frequency_only"` | `"rgb_frequency_fusion"` |
| Checkpoint `architecture` id | `"efficientnet_b4"` | `"frequency_branch"` | `"rgb_frequency_fusion"` |
| Checkpoint selection | best validation ROC-AUC (Step 4 rule, unchanged) | best validation ROC-AUC (same rule) | best validation ROC-AUC (same rule) |
| Evaluation | Step 5 `run_evaluation`, unchanged | Step 5 `run_evaluation`, same manifest/split/threshold | Step 5 `run_evaluation`, same manifest/split/threshold |
| **Primary metric** | unseen-generator ROC-AUC | unseen-generator ROC-AUC | unseen-generator ROC-AUC |
| Secondary metrics | macro-F1, accuracy, FPR, confusion matrix (all at the frozen threshold, Step 5's existing metric set) | same | same |

The threshold is never optimized per experiment or on the test split, for any of the three - Step 5's existing "Threshold policy" (frozen at `ModelConfig.threshold`, resolved before any inference) applies identically here.

## 10. Fair comparison

Concretely, to keep Experiment 1 a valid baseline against Experiments 2/3:

- **Same split**: one `split_manifest` call's output (train/val/test record lists) is the input to all three experiments - not three independent calls that happen to use the same seed (which would be fragile against any future change to `choose_unseen_generators`'s internals); the recommended experiment-runner script computes the split once and passes the same three record lists into each experiment's training entrypoint.
- **Same unseen generators**: the *actual resolved* unseen-generator list (not just "the same seed") is logged and asserted identical across all three runs' `config.json` - Step 5 already logs this (`split_config.unseen_generators`, the resolved list); the experiment tracker (below) diffs this field across the three runs as a sanity check before trusting any comparison.
- **Same preprocessing policy where applicable**: the RGB branch's preprocessing (resize, ImageNet normalization) is byte-identical across Experiments 1 and 3 (same `ImagePreprocessor`, same config); "where applicable" acknowledges Experiment 2 has no RGB branch to compare that preprocessing against, but its frequency preprocessing is the same function used in Experiment 3.
- **Same seed**: one `TrainingConfig.seed` value used for all three (`set_seed()`, unchanged from Step 4), so weight initialization randomness (where applicable - the frequency branch and fusion head have no pretrained initialization to control for otherwise), augmentation order, and DataLoader shuffling are controlled for as much as an ablation across genuinely different architectures can control for.
- **Same number of epochs where practical**: identical epoch budget across all three, not tuned per-architecture for "best" convergence - a shorter/longer schedule for one architecture would confound "does frequency help" with "was one model just trained longer."
- **Same checkpoint selection**: best validation ROC-AUC, Step 4's existing rule, unchanged and applied identically to all three model types.
- **Same evaluation threshold**: `ModelConfig.threshold` (0.5 by default), resolved once, never tuned per experiment or on test - Step 5's existing rule.

## 11. Experiment tracking

No new logging system - Step 4's `RunLogger`/`config.json` convention (already reused unchanged by Step 5) is extended with one new, additive field, `model_type`, and Step 6's dual-branch models' `training_config`/`model_config` snapshots naturally include whatever their own `TrainingConfig.to_dict()` and `.get_spec()` report - no schema redesign needed, since `config.json` was always a plain dict.

Per-run `config.json` records (existing fields unchanged, new ones marked):

- `run_id`, `dataset` (dev-shard-labeled exactly as today), `manifest_path`
- **`model_type`** (new: `"rgb_only"` | `"frequency_only"` | `"rgb_frequency_fusion"`)
- `model_config` (now includes `.get_spec()`'s `architecture` field for whichever model was actually built - `"efficientnet_b4"`, `"frequency_branch"`, or `"rgb_frequency_fusion"`)
- seed, `training_config` (all hyperparameters, unchanged shape)
- `num_train_samples`, `num_val_samples`
- resolved split info: seed, `val_fraction`, resolved `unseen_generators` list, `held_out_generator_fraction` (Step 5 already logs an equivalent block for evaluation; the training-side `config.json` should log the same fields it already omits today - see "Risks," this is the same Step 4 gap Step 5's spec already flagged, now doubly relevant since three experiments must be verified to share one split)
- preprocessing: image size, normalization mean/std (RGB branch); frequency representation id (`"log_magnitude_fft"`, a fixed string for now, not yet a variable, since only one representation exists) for the frequency/fusion runs
- validation metrics history (per-epoch, Step 4's existing `val_metrics_history`)
- checkpoint paths (`last.pt`, `best.pt`)

Then, separately, each experiment's Step 5 evaluation run (`experiments/runs/<run_id>/evaluation/<eval_run_id>/`) records its own `config.json`/`metrics.json`/`predictions.csv`/`evaluation_report.md` exactly as Step 5 already defines - unchanged. A three-way comparison is simply: run all three experiments, run Step 5's `evaluate` against each resulting `best.pt`, and compare the three `metrics.json["test"]["overall"]["roc_auc"]` values (plus the secondary metrics) - no new comparison tooling is required for the baseline ablation; a small, optional comparison-table script is a reasonable convenience but not part of this step's required deliverable.

## 12. Evaluation integration

Reused, not redesigned, per the "Model-loading interface" section above:

- `model.evaluation.metrics` - **zero changes**. Every function already operates on plain `labels`/`probabilities`/`predictions`/`generators` sequences; it has no idea what architecture produced them and does not need to.
- `model.evaluation.report` - **zero changes**. Same reasoning.
- `model.evaluation.predict.run_inference` - one small change (the device-move helper described above), otherwise unchanged; still returns the same `PredictionRow` shape regardless of architecture.
- `model.evaluation.evaluate.run_evaluation` - one small, additive dispatch (dataset-construction path chosen by the checkpoint's own recorded architecture id) - everything after inference (metrics computation, artifact writing) is identical regardless of which of the three models produced the predictions.
- `model.training.checkpoint` - the architecture-registry extension described above is what actually *enables* this reuse; without it, `load_model_from_checkpoint` would hard-fail (or silently mis-load) on anything but an EfficientNet-B4 checkpoint.

## 13. Explainability forward-look (not implemented here)

Grad-CAM is explicitly out of scope for this step. However, the frequency branch is deliberately designed so Step 7 has something concrete to build on later: because the frequency branch ends in a small, spatially-organized feature map (`(N, 64, 28, 28)` before the final global pool - see "Frequency branch architecture"), a future Grad-CAM-style analysis could, in principle, localize *which region of the spectrum* (not which region of the image) drove the frequency branch's contribution to a prediction - conceptually different from RGB Grad-CAM, which localizes image regions. This step does not build that; it only avoids architectural choices (e.g. collapsing to a 1D vector before any spatial feature map exists) that would foreclose it later. To be explicit: **the FFT branch itself is not an explanation** - a high frequency-branch activation is evidence of a spectral pattern correlating with the training label, not a human-interpretable "this is fake because of X" statement, and nothing in this step should be read or reported as already providing that.

## 14. Computational constraints

Development machine: Intel Core i5-1334U, ~7.6 GB usable RAM, no CUDA, Python 3.12 ML venv (matching Step 4/5's documented environment exactly).

- The frequency branch's parameter count and per-sample FLOPs are negligible next to EfficientNet-B4 (see "Frequency branch architecture") - Experiments 2 and 3's CPU smoke tests are not expected to be meaningfully slower than Step 4's existing RGB-only smoke test; Experiment 3's forward pass runs both branches, so its wall-clock cost is dominated by the RGB branch exactly as it already is for Experiment 1.
- **CPU smoke testing** (this machine): tiny truncated train/val subsets (`max_train_samples`/`max_val_samples`, reusing `TrainingConfig`'s existing fields - Step 4 already defined and Step 6 does not need to add new ones), 1-2 epochs, small batch size, `pretrained=False` for the RGB branch specifically to avoid any network access during CPU smoke tests (matching Step 4's own smoke-test convention exactly) - this is a correctness check (forward/backward/checkpointing work, tensors are finite, shapes are right), never a reported result.
- **Full training** (deferred to cloud GPU, not part of this step's deliverable): larger batch size, `num_workers > 0`, full epoch budget, `pretrained=True`, run against a real (non-dev-shard) dataset once available - explicitly separated from the CPU smoke test in the same way Step 4's spec already separates them.
- No part of this design requires a GPU to be *correct* - only to be *fast enough for a real, reportable result*. The CPU path exists solely to catch bugs before spending cloud-GPU time.

## 15. Testing (design only - no test files created by this step)

When implemented, `tests/test_frequency_fusion.py` (new file, following `tests/test_training.py`/`tests/test_evaluation.py`'s existing conventions - `pytest.importorskip("torch")`, tiny synthetic images, `pretrained=False`, no network access, no real dataset) should cover:

- **Deterministic FFT transform**: calling `compute_log_magnitude_spectrum` twice on the same input array produces bit-identical output.
- **Output shape**: `(1, H, W)` for a range of `H, W` (not hardcoded to 224 only, to catch accidental shape assumptions).
- **Finite values**: no `NaN`/`inf` in the output, including for a constant-color (all-zero-variance) input image - the exact edge case `log1p(0) = 0` is designed to handle (see "Input construction," step 5).
- **Normalization**: output values lie in `[0, 1]`; a constant-color input's normalized output is exactly `0` everywhere (min == max, so the `+1e-8` denominator guard prevents a divide-by-zero rather than producing `NaN`).
- **Frequency branch forward pass**: `(N, 1, H, W)` in → `(N, 64)` embedding out, for at least two batch sizes including `N=1`.
- **RGB branch forward pass**: unchanged - `tests/test_model.py`'s existing coverage already covers this; Step 6 adds no new RGB-branch-only tests, since nothing about it changed.
- **Fusion forward pass**: `(N, 1)` output shape, finite values, for the combined `{"rgb": ..., "frequency": ...}` batch input.
- **Gradient flow**: after one backward pass on a fusion-model forward output, every trainable parameter in both branches and the fusion head has a non-`None`, finite `.grad`.
- **Batch-size handling**: forward pass works for `N=1` and `N>1` for all three models (frequency-only, fusion, and a regression check that RGB-only still does, matching `tests/test_model.py`'s existing `test_forward_pass_output_shape`).
- **Checkpoint save/load roundtrip**: for each of the three architectures, `save_checkpoint` → `load_model_from_checkpoint` → identical forward-pass output on the same input (mirroring `tests/test_training.py::test_save_and_load_training_checkpoint_roundtrip`'s existing pattern) - this is also where the architecture-registry dispatch itself gets exercised and verified.
- **Configuration validation**: an unrecognized `model_type` value raises a clear, named error rather than silently defaulting to RGB-only or crashing with an unrelated `AttributeError`.
- **Regression**: re-running `tests/test_training.py` and `tests/test_evaluation.py` unmodified must still pass in full (91/91 today) - the device-move helper and checkpoint-registry changes must not alter their behavior for the plain-tensor, `"efficientnet_b4"` code paths those tests exercise.

None of these tests require downloading pretrained weights - every model in every test is constructed with `pretrained=False`.

## 16. Backward compatibility

- Step 4's `model/training/train.py` behavior is unchanged when `model_type` is absent or `"rgb_only"` - byte-for-byte the same training run as today, including its existing CPU smoke test.
- Step 5's `model/evaluation/evaluate.py` behavior is unchanged for any checkpoint whose `model_config.architecture` is `"efficientnet_b4"` (every checkpoint that exists today) - byte-for-byte the same evaluation run as today.
- No changes to `backend/`, `app/`, `ui/`, or `explainability/` - none of Step 6's design requires touching any of them; there is no "genuinely unavoidable integration boundary" here, since the frequency/fusion work is entirely internal to the ML training/evaluation pipeline.
- `data/dataset_loader.py`, `data/preprocessor.py`, `data/splitting.py`, and `model/architectures/efficientnet_b4.py` are **not modified** - every new capability is additive, built by injecting new preprocessor objects (duck-typed to the existing `.preprocess()` interface) and new sibling architecture files, not by changing what those existing files do.

## 17. Configuration

New file: `config/frequency_config.yaml`, holding facts specific to the frequency representation and the new architectures - not duplicating anything already in `config/model_config.yaml` (RGB architecture, preprocessing, split) or `config/training_config.yaml` (run hyperparameters):

```yaml
# SignalScope Frequency-Domain Branch & Fusion Configuration
frequency:
  representation: "log_magnitude_fft"   # fixed for now - the only representation this step defines
  grayscale_weights: [0.299, 0.587, 0.114]
  use_fftshift: true
  normalization: "per_image_minmax"

frequency_branch:
  input_channels: 1
  conv_channels: [16, 32, 64]
  embedding_dim: 64

fusion:
  rgb_embedding_dim: 1792     # EfficientNet-B4's actual pooled feature width - see "Existing components to reuse"
  frequency_embedding_dim: 64
  fusion_dim: 1856
  dropout: 0.4

# Which model to build - read by model/training/train.py's model factory.
# "rgb_only" reproduces Step 4 exactly; this key is additive to
# config/training_config.yaml's existing schema, not a duplicate of it.
model_type: "rgb_only"   # "rgb_only" | "frequency_only" | "rgb_frequency_fusion"
```

`model_type` could instead live inside `config/training_config.yaml` (it is arguably a run-selection choice, not a frequency-representation fact) - this is flagged as an open, low-stakes decision for implementation time (see "Risks"); either placement is compatible with everything else in this spec, since both files are already loaded together by `model/training/train.py`.

## 18. Git scope

**Files to create:**
- `.claude/specs/06-frequency-fusion.md` (this document)
- `config/frequency_config.yaml`
- `model/architectures/frequency_branch.py` (the `FrequencyBranch` CNN + `FrequencyOnlyModel` wrapper with its own head + `.get_spec()`)
- `model/architectures/fusion_model.py` (`RGBFrequencyFusionModel`, holding an `EfficientNetB4Baseline` and a `FrequencyBranch`, concatenation fusion head, `.get_spec()`)
- `model/training/frequency_features.py` (`compute_log_magnitude_spectrum`, pure function)
- `model/training/dual_branch_dataset.py` (`AugmentedDualBranchPreprocessor`, `DeterministicDualBranchPreprocessor`, `build_dual_branch_train_dataset`, `build_dual_branch_val_dataset`)
- `tests/test_frequency_fusion.py` (at implementation time - not created by this design-only step)

**Files to modify (small, additive, backward-compatible - see "Model-loading interface" for exact scope of each):**
- `model/training/checkpoint.py` - `save_checkpoint` calls `model.get_spec()` instead of the EfficientNet-specific free function; `load_model_from_checkpoint` gains an architecture-id dispatch with `"efficientnet_b4"` as the default/fallback.
- `model/training/engine.py` - `train_one_epoch`/`validate` use a small dict-or-tensor device-move helper instead of a bare `.to(device)`.
- `model/evaluation/predict.py` - `run_inference` uses the same device-move helper.
- `model/evaluation/evaluate.py` - `run_evaluation` dispatches dataset construction (existing `build_val_dataset` vs. new `build_dual_branch_val_dataset`) based on the checkpoint's own recorded architecture id.
- `model/training/train.py` - gains a small `model_type`-driven model-and-dataset factory; default value reproduces today's exact RGB-only behavior.

**Files to leave untouched:**
- `model/architectures/efficientnet_b4.py`
- `data/dataset_loader.py`, `data/preprocessor.py`, `data/splitting.py`
- `config/model_config.yaml`
- `model/evaluation/metrics.py`, `model/evaluation/report.py`
- `backend/`, `app/`, `ui/`, `explainability/`, `backend/requirements.txt`
- `tests/test_training.py`, `tests/test_evaluation.py`, `tests/test_model.py`, `tests/test_data.py` (must keep passing unmodified, per "Acceptance criteria" - not edited to make them pass)

No implementation, tests, or configuration files are created by this step - the above describes what a future implementation step would create/modify, per the deliverable instructions.

## 19. Acceptance criteria

- All 91 currently-passing tests remain passing, unmodified.
- New Step 6 tests (`tests/test_frequency_fusion.py`, at implementation time) pass.
- `EfficientNetB4Baseline`/Experiment 1 continues to train, checkpoint, and evaluate exactly as before (regression-tested against Step 4/5's own existing tests, not re-derived).
- The frequency-only model (Experiment 2) trains, checkpoints, and evaluates end-to-end on a CPU smoke test.
- The fusion model (Experiment 3) trains, checkpoints, and evaluates end-to-end on a CPU smoke test.
- CPU smoke training completes in well under a few minutes for all three model types on the development machine.
- The frequency transform is deterministic (bit-identical output for the same input across repeated calls).
- No generator leakage: `check_generator_leakage` passes for all three experiments' splits (all three use the same, single `split_manifest` call's output).
- Checkpoint save/load roundtrip works for all three architectures, verified by identical forward-pass output before/after reload.
- Step 5's `run_evaluation` can load and evaluate a checkpoint from any of the three architectures without modification to its metric logic.
- All three ablation experiments can be run against the same manifest, seed, and split configuration, and their `metrics.json["test"]["overall"]["roc_auc"]` values are directly comparable.
- The unseen-generator `test` split is never used for training, hyperparameter selection, threshold selection, or checkpoint selection, for any of the three experiments.
- No changes to `backend/`, `app/`, `ui/`, or `explainability/`.
- No generated checkpoints, run artifacts, or experiment outputs are committed (existing `experiments/runs/`/`*.pt` `.gitignore` rules apply unchanged).
- No report or spec text claims Tiny-GenImage ablation results represent final SIH benchmark performance, regardless of which experiment "wins."

## Risks and mitigations

- **Per-image min-max normalization may wash out genuinely comparable spectral energy levels across images** (each image's spectrum is independently rescaled to [0,1], discarding absolute magnitude information that could itself be a forensic cue, e.g. overall spectral energy differing systematically between real photos and a specific generator). Mitigation: documented as a deliberate, revisitable simplification (see "Input construction," step 6); a train-split-only running-statistics normalization is a natural first refinement if the baseline ablation is inconclusive, not a defect to fix before running the baseline.
- **Shared geometric augmentation (rotation, crop-resize) may introduce interpolation artifacts into the frequency branch's input that have nothing to do with the generator's own fingerprint** - rotation and resizing both involve resampling, which itself perturbs high-frequency content. Mitigation: not avoidable while also satisfying the explicit "same augmented image for both branches" requirement; flagged here so a future ablation-of-the-ablation (frequency branch trained on *unaugmented* crops only) can isolate this if results are surprising - see "How to interpret the experiment," Case D.
- **Step 4's `config.json` still does not log the resolved `SplitConfig`** (the same gap Step 5's spec already flagged) - now more consequential, since verifying all three experiments actually share one split depends on comparing that information across three runs. Mitigation: this step's "Experiment tracking" section explicitly asks for the same split-config fields Step 5 already logs on the evaluation side to also be logged on the training side; closing this gap is a small, clearly-scoped addition an implementation step should make (whether as part of Step 6 or as a fast-follow) rather than something this design silently assumes away.
- **GenImage-family datasets are known to carry source/compression/resolution biases** (different generators' outputs may differ systematically in JPEG quality, resolution, or capture pipeline, independent of any genuine generative artifact). A frequency branch is, by construction, exactly the kind of feature most likely to pick up a compression-related shortcut rather than a genuine generative fingerprint. Mitigation: not solvable by architecture alone - this is precisely why "How to interpret the experiment" treats a surprisingly strong frequency-only result as a flag for further investigation, not a validated finding, and why this spec repeatedly refuses to let Tiny-GenImage results stand in for a benchmark claim.
- **`model_type` placement (frequency_config.yaml vs. training_config.yaml) is an unresolved, low-stakes design choice** - left open for implementation time; does not affect any other part of this design.
- **Dict-shaped batches through the DataLoader's default collate function have not been empirically verified end-to-end in this codebase** (only reasoned about from PyTorch's documented `default_collate` behavior) - an implementation step should verify this early (it is exactly the kind of assumption a five-minute smoke test resolves conclusively before more code is built on top of it).

## How to interpret the experiment

**Case A: Fusion > RGB on unseen-generator ROC-AUC.** Evidence that frequency-domain information provides complementary forensic signal beyond what the RGB branch alone captures, on this dataset. Justifies continued investment in the frequency branch (e.g. per-channel FFT, train-split-normalized statistics, a more expressive fusion mechanism) as a Step 7+ direction.

**Case B: Fusion ≈ RGB.** The frequency branch, in its current minimal design, may be redundant given what EfficientNet-B4 already learns from RGB alone - or the frequency representation/fusion mechanism is not yet expressive enough to surface a real signal that exists. Does not by itself justify abandoning frequency features; does justify checking Experiment 2 (frequency-only) in isolation before concluding either way (see below).

**Case C: Fusion < RGB.** The frequency representation or the fusion mechanism may be introducing noise, or the added parameters (however few) are overfitting the small development shard rather than learning generalizable signal. Should prompt inspection of Experiment 2 (frequency-only) and Experiment 3's training curves (does the fusion model's validation ROC-AUC ever exceed the RGB-only model's, even mid-training) before concluding frequency information is unhelpful in general.

**Case D: Frequency-only performs surprisingly well** (competitive with or close to the RGB baseline, despite the frequency branch's tiny capacity and the loss of all color/texture information). This specifically warrants investigating whether the frequency branch is exploiting a **compression, resolution, or source bias** rather than a genuine generative-forensic signal - GenImage-family datasets are known to carry exactly these kinds of confounds (different generators' sample pipelines can differ systematically in JPEG quality or output resolution, both of which leave detectable, non-forensic frequency-domain signatures). A strong frequency-only result should be treated as a prompt for further investigation (e.g. checking per-generator ROC-AUC for a pattern that tracks known dataset artifacts rather than image content, or testing against resized/re-compressed variants of the same images), not as confirmation that the frequency branch has learned "real" forensic evidence.

**In every case**: this is one ablation, on one small development shard (Tiny-GenImage, ~2,000 images, a handful of generators), with one specific frequency representation and one specific fusion mechanism. No result from this experiment should be reported or interpreted as a general claim about whether frequency-domain features help AI-image forensics, or as the final SIH benchmark result - both because the dataset is explicitly a development shard (see "Non-goals") and because a single ablation cannot establish causality about *why* an observed difference occurred (Case D above is exactly an example of a plausible alternative explanation for a positive-looking result).
