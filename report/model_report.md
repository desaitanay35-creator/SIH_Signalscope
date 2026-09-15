# SignalScope — Model Report

One-page model report per the SIH 2026 SignalScope problem statement, Section 7.3.

| Field | What we report |
|---|---|
| **Task** | Binary real-vs-AI-generated image classification (core task). Bonus modules attempted: **A — Faithful Explanation** (Grad-CAM + grounded text), **D — Provenance & Metadata** (EXIF/C2PA reporting). Modules B, C, E, G not attempted in this submission. |
| **Data & split** | GenImage development shard, `data/manifests/genimage_dev.csv`, ~2,000 images (real + ADM, BigGAN, GLIDE, Midjourney, SD1.4/1.5, VQDM, Wukong). **Generator-disjoint split** (not random image-level): train=1,457, val=257, test=286, seed=42, unseen generators held out entirely from train/val = **BigGAN, Midjourney** (`data/splitting.py`). No additional public dataset was added to training. **Caveat:** real images are never routed into `test` (only fake generators can be held out), so the reported "unseen-generator" set pairs `test`'s held-out-generator fakes with `val`'s real images — documented in `data/README.md`. This is a development shard, not the organizers' final held-out benchmark. |
| **Model / approach** | RGB branch: EfficientNet-B4, ImageNet-pretrained, full fine-tune. Frequency branch: log-magnitude FFT spectrum (grayscale, ITU-R BT.601 weights, per-image min-max normalized) → 3-block CNN → 64-dim embedding. Fusion: concatenation of the RGB branch's 1792-dim pooled features with the 64-dim frequency embedding (1856-dim) → dropout(0.4) → linear → 1 logit. Optimizer: AdamW, lr=1e-4, weight decay=1e-4, cosine LR schedule, 10 epochs, batch size 16, `BCEWithLogitsLoss`. Augmentation: horizontal flip, ±10° rotation, random-resized crop — deliberately **no** compression/blur/noise augmentation, to avoid destroying the forensic signal the model relies on. Calibration: temperature scaling implemented (`model/calibration/`, fitted T≈1.184) but **disabled** in the deployed backend; reported confidences are raw `sigmoid(logit)`. |
| **Metric & result** | At threshold 0.50, on the development shard's generator-disjoint split: **Unseen-generator ROC-AUC = 0.9396** (primary), overall val ROC-AUC = 0.9070, macro-F1 = 0.8599, accuracy = 0.8767, FPR = 0.2083, confusion matrix (tp/tn/fp/fn) = 263/114/30/23. Per unseen generator: BigGAN AUC = 0.9390, Midjourney AUC = 0.9403. Source: `experiments/ablation/step7/comparison.json`. |
| **Baseline** | Three-way controlled ablation under an identical protocol (same manifest, split, seed, hyperparameters — only the model type differs): **RGB-only** (unseen-AUC 0.9209, val-AUC 0.9027) and **Frequency-only** (unseen-AUC 0.8438, val-AUC 0.6535) were each trained as baselines for the production **RGB+Frequency Fusion** model (unseen-AUC 0.9396, val-AUC 0.9070). Fusion improves over both single-branch baselines on the unseen-generator split, especially over the weak frequency-only baseline; frequency-only alone is markedly worse and, on this shard, largely trails RGB-only except on BigGAN specifically (frequency-only AUC 0.9893 vs. fusion 0.9390 on BigGAN — noted in `.claude/specs/06-frequency-fusion.md` as a possible dataset-artifact confound, not confirmed forensic signal). |
| **Limitations** | (1) The unseen-generator result above is a small (2,000-image) development shard with the real-image-pairing caveat described above — not the organizers' held-out benchmark, and not yet validated at that scale. (2) Manual real-world testing (phone photos, screenshots, downloaded images — outside the GenImage distribution) has surfaced false positives on genuinely authentic images; a rigorous, source-disjoint authentic-image robustness benchmark and root-cause attribution is in progress but not part of this submission's shipped evaluation, so we do not claim the model is robust to real-world image sources beyond GenImage's distribution. (3) Frequency-only degrades sharply relative to RGB-only on most generators; whether the frequency branch helps or harms robustness to real-world processing (JPEG recompression, screenshotting, resizing) has not been isolated in a controlled experiment for this submission. (4) No degradation-robustness (Module C), generator-attribution (Module B), or multimodal (Module E) evaluation is included. |

## Explanation samples (Module A)

See `POST /api/v1/analyze` → `explanation.summary` (grounded natural-language cue text) and `GET /api/v1/analyses/{id}/heatmap` (Grad-CAM overlay, 224×224 JPEG, computed on the RGB branch's final EfficientNet-B4 feature block, targeting the pre-sigmoid, pre-calibration AI-generated logit). Example response fields from a live local run:

```json
{
  "verdict": "likely_ai_generated",
  "confidence": 0.8731,
  "is_calibrated": false,
  "explanation": {
    "summary": "Highlighted regions show where the model's RGB (visual) analysis concentrated when evaluating this image's likelihood of being likely AI-generated. The frequency-domain branch also contributes to the prediction but is not directly visualized by this heatmap. This shows where the model focused, not proof of any specific forensic artifact.",
    "heatmap_available": true
  }
}
```

Per the problem statement's Section 4.3 scoring rubric, this explanation deliberately hedges ("shows where the model focused, not proof of any specific forensic artifact") rather than asserting unverified cue-level claims (e.g. "warped text", "inconsistent lighting") the model has not been separately validated to detect — avoiding over-claiming at the cost of specificity.
