# SignalScope Dataset Pipeline

This document describes the dataset manifest format, label/generator
semantics, and the generator-disjoint split methodology implemented in
`data/dataset_loader.py`, `data/splitting.py`, and `data/preprocessor.py`.

This pipeline does **not** ship with, fabricate, or reproduce any dataset -
including the hidden SIH test set. It only defines how a real, user-supplied
dataset should be described and split.

## Manifest format

A manifest is a CSV or JSON file with one row per image:

```csv
image_path,label,generator,split
real/001.jpg,0,real,
synthetic/sd_001.jpg,1,stable_diffusion,
synthetic/mj_001.jpg,1,midjourney,
synthetic/dalle_001.jpg,1,dalle,
```

Columns:

- **image_path** (required) - path to the image, relative to a `root_dir`
  passed to `SignalScopeDataset`, or absolute.
- **label** (required) - `0` = real, `1` = AI-generated. No other values are
  accepted.
- **generator** (required) - the source identifier:
  - Real images must use `generator = "real"`.
  - AI-generated images must use the specific generator/model that produced
    them (e.g. `stable_diffusion`, `midjourney`, `dalle`, `sdxl`, ...). This
    is the field the generator-disjoint split partitions on - it is not
    optional metadata, it is load-bearing for the SIH unseen-generator
    requirement.
- **split** (optional) - `train` / `val` / `test`, if the manifest already
  encodes a trusted split. Leave blank to have `data.splitting.split_manifest`
  derive it.

JSON manifests use the same fields, either as a top-level list of objects or
a dict with a `"records"` key holding that list.

## Supplying a real dataset

Never commit real dataset images (or the hidden SIH test set) to this
repository. Instead:

1. Store images anywhere on disk (or point at an existing dataset directory).
2. Write/generate a manifest CSV or JSON listing those images with their
   real label and true generator identity.
3. Point `config/model_config.yaml`'s `data.manifest_path` (or a
   `manifest_path` argument) at that file.

`load_manifest()` and `SignalScopeDataset` never fabricate data - they only
read what you point them at, and raise `ManifestError`/`FileNotFoundError`
loudly if the manifest is missing or malformed.

## Generator-disjoint splitting

`data/splitting.py::split_manifest` produces three splits:

- **train** - real images + all "seen" generators.
- **val** - real images + the same "seen" generators as train (for threshold
  and calibration development later), held out at the image level via a
  seeded shuffle.
- **test** - real images + generators that are **completely absent** from
  train and val. This is the unseen-generator evaluation set.

Which generators are "unseen" is controlled by
`config/model_config.yaml`'s `data.split`:

- `unseen_generators`: an explicit list (e.g. `["dalle", "midjourney"]`).
  Recommended once you know which generators you want to reserve for final
  evaluation.
- `held_out_generator_fraction`: used only when `unseen_generators` is empty
  - a seeded shuffle of all distinct generator ids picks this fraction to
    hold out automatically.

Both paths are deterministic: the same manifest + the same `data.split`
config (in particular the same `seed`) always produce the same split. This
is what "reproducible" means here - re-running the split does not silently
reshuffle which generators are unseen.

**This is not a random image-level split.** An entire generator is assigned
to one side of the train/val vs. test boundary - never partially. That is
what makes the `test` split usable as a genuine unseen-generator benchmark
rather than a same-distribution held-out set.

## Leakage protection

`data/splitting.py::check_generator_leakage` is called automatically at the
end of `split_manifest` and raises `GeneratorLeakageError` if any non-"real"
generator appears in both `{train, val}` and `test`. Call it directly on any
manually assembled split mapping (e.g. one loaded via
`group_by_existing_split` from a manifest that already has a `split`
column) to validate it before use.

## Preprocessing

`data/preprocessor.py::ImagePreprocessor`:

1. Loads the image from disk and converts it to RGB (handles grayscale,
   palette, RGBA, etc. - always 3 channels out).
2. Resizes to a configurable `image_size` (default `224x224`, from
   `config.settings.PreprocessingConfig`).
3. Scales to `[0, 1]`, transposes to channel-first `(3, H, W)`, and
   normalizes with configurable per-channel `mean`/`std`.
4. Returns a `torch.Tensor` if PyTorch is installed, otherwise a
   `numpy.ndarray` of the same shape/dtype - so this module and its tests
   don't hard-require torch, while producing a torch-ready tensor once the
   full training stack (a later step) is installed.
5. Raises `ImageLoadError` - never a bare PIL/OS exception - for a missing
   file or a corrupt/unreadable image, so callers have one well-defined
   "invalid image" failure mode.

## Running the dataset tests

```bash
pytest tests/test_data.py -v
```

All fixtures are tiny synthetic images generated on the fly inside the test
file (via PIL, in `tmp_path`) - no real dataset or SIH hidden test data is
read or required.

## EfficientNet-B4 baseline

`model/architectures/efficientnet_b4.py` defines SignalScope's first ML
baseline: `build_model(pretrained=True) -> nn.Module` and
`get_model_spec() -> dict`.

**Why this is the baseline.** EfficientNet-B4 is a well-understood,
ImageNet-pretrained CNN with a good accuracy/compute trade-off on CPU and
GPU alike, making it a reasonable starting point for transfer learning
before investing in anything more specialized (frequency-domain fusion,
ensembles, ViTs, etc.).

**What the model predicts.** A torchvision EfficientNet-B4 backbone with its
1000-class ImageNet head replaced by a single binary logit. Given a
preprocessed `(N, 3, H, W)` RGB tensor (see `data/preprocessor.py` - this
module does not reimplement preprocessing), `forward()` returns the raw
logit of shape `(N, 1)`.

**Label mapping:** `0 = real`, `1 = AI-generated` (matches
`data/dataset_loader.py`'s manifest label convention).

**Logit vs. probability.** The model returns a raw logit, not a probability
and not a decision. `sigmoid(logit)` is the estimated probability that the
image is AI-generated. No decision threshold is applied inside the model -
thresholding is an evaluation/inference-layer concern, applied later to
`sigmoid(logit)`.

**Why this is only a baseline.**
- The classification head is freshly initialized and has not been trained
  on any real-vs-AI-generated data - its raw output is not yet meaningful.
- No training loop exists yet (deliberately out of scope for this step).
- `sigmoid(logit)` is **not calibrated**. Calibration (e.g. temperature
  scaling) will be added later, once real training data is available, and
  must not be assumed or claimed before then.
- This baseline has not been evaluated - it has **not** achieved any
  accuracy, ROC-AUC, or other metric. No such claim should be made until a
  real training + evaluation run is performed.
- The generator-disjoint `test` split (see above) - not a random image-level
  holdout - will be the important benchmark once training happens: unseen-
  generator ROC-AUC, not in-distribution accuracy, is the metric that
  matters for SIH.

**What this step does NOT include:** model training, FFT/frequency
features, Error Level Analysis, Grad-CAM, calibration, or any backend/API
integration. Those are later steps.

Run the model tests with:

```bash
pytest tests/test_model.py -v
```

These tests always construct the model with `pretrained=False`, so they
require no network access and never download ImageNet weights.
