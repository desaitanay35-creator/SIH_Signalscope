# SignalScope

**Telling Real From Synthetic in the Age of Generative Media**

SignalScope classifies an input image as real or AI-generated, reports calibrated confidence, and explains its verdict with a Grad-CAM heatmap and grounded natural-language cues. Built for SIH 2026 Internal Hackathon [C-433], Problem Statement 2.

> Outputs are always framed as likelihood assessments ("likely AI-generated" / "likely real") — never as certainty or accusation. SignalScope does not identify, profile, or make claims about specific real people, and does not adjudicate political claims or real-world events.

---

## 1. Core + bonus modules built

| Module | Status | Notes |
|---|---|---|
| **Core task** — real-vs-AI-generated classification | ✅ Built | EfficientNet-B4 (RGB) + a frequency-domain CNN branch, fused by concatenation. FastAPI backend + React frontend. |
| **A. Faithful Explanation** (headline bonus) | ✅ Built | Grad-CAM heatmap over the RGB branch's final conv block, plus a grounded natural-language summary of what the model attended to. |
| **B. Generator Attribution** | ❌ Not built | Out of scope for this submission. |
| **C. Robustness to Degradation** | ❌ Not built (see Limitations) | Investigated but not part of this submission's shipped system. |
| **D. Provenance & Metadata** | ✅ Built | Reads EXIF and checks for C2PA/Content Credentials markers per upload; reported alongside the verdict as supporting context — not fused into the model's decision. |
| **E. Multimodal (image + text)** | ❌ Not built | Out of scope for this submission. |
| **F. Real-Time / Deployable** | ✅ Built | Working web app: drag-and-drop upload, sub-second local inference latency, responsible "likely AI-generated" presentation. |
| **G. Active Defence Analysis** | ❌ Not built | Out of scope for this submission. |

---

## 2. Setup and run instructions

**Requirements:** Python 3.12, Node.js 18+, ~1.5 GB free disk for ML dependencies.

### Backend
```bash
cd backend
pip install -r requirements.txt
copy .env.example .env          # sets MODEL_WEIGHTS_PATH and other config
python -m uvicorn app.main:app --reload --port 8000
```
Also install the root ML dependencies (`torch`, `torchvision`, etc.) the backend's detector imports:
```bash
pip install -r requirements.txt   # from the repository root
```
On Linux/macOS, install the CPU-only PyTorch build to avoid pulling an unnecessary CUDA toolchain:
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### Frontend
```bash
cd frontend
npm install
copy .env.example .env          # sets VITE_API_BASE_URL=http://127.0.0.1:8000
npm run dev
```
Open `http://localhost:3000`, upload an image, and get a verdict — reproducible well under 10 minutes from a clean clone.

### Verify without the UI
```bash
curl http://127.0.0.1:8000/api/v1/health
curl -F "image=@path/to/sample.jpg" http://127.0.0.1:8000/api/v1/analyze
```

### Running tests
```bash
python -m pytest tests/ -v          # ML pipeline (data/splitting/model/training/evaluation)
cd backend && python -m pytest -v   # backend API and ML-integration tests
```

**If no model checkpoint is present:** `GET /api/v1/health` reports `model_loaded: false` and `POST /api/v1/analyze` returns `503 MODEL_UNAVAILABLE` — the API never fabricates a placeholder prediction.

---

## 3. Datasets used

| Dataset | Role | Source / licence |
|---|---|---|
| **GenImage** (dev shard: real photos + ADM, BigGAN, GLIDE, Midjourney, SD1.4/1.5, VQDM, Wukong) | Training / validation / generator-disjoint unseen-generator evaluation | Public dataset; cited per `data/README.md`. `data/manifests/genimage_dev.csv` is a documented ~2,000-image **development shard**, not the organizers' final held-out benchmark. |
| Organizers' held-out test set | Final core-metric evaluation | Provided at judging time; **never trained on**, per the challenge's data rules. |

No additional public dataset was added to training beyond GenImage for this submission. Model weights are distributed via the training pipeline (`model/training/train.py`), not committed to the repository (see `.gitignore`) — regenerate via the commands in `CLAUDE.md` or restore a checkpoint into `experiments/runs/<run_id>/checkpoints/`.

---

## 4. Reported metrics (development shard — see caveat below)

Ablation results from `experiments/ablation/step7/comparison.json`, at threshold 0.50, generator-disjoint split (unseen generators: **BigGAN, Midjourney**):

| Model | Val ROC-AUC | **Unseen-generator ROC-AUC** | Accuracy | Macro-F1 | FPR | Confusion Matrix (tp/tn/fp/fn) |
|---|---|---|---|---|---|---|
| RGB-only | 0.9027 | 0.9209 | 0.8488 | 0.8306 | 0.2222 | 253 / 112 / 32 / 33 |
| Frequency-only | 0.6535 | 0.8438 | 0.7488 | 0.7284 | 0.2917 | 220 / 102 / 42 / 66 |
| **RGB+Frequency Fusion (production)** | **0.9070** | **0.9396** | **0.8767** | **0.8599** | **0.2083** | 263 / 114 / 30 / 23 |

**Caveat — read before citing these numbers.** This is a **2,000-image GenImage development shard**, not the organizers' held-out test set, and the current split implementation places all real images in train/validation while only held-out **fake** generators go to `test` — so the "unseen-generator" real images are drawn from validation, not a fully independent real-image pool (documented in `data/README.md` and `.claude/specs/05-evaluation-pipeline.md`). These are honest development-time numbers, not the final SIH-reported result, which will be computed by organizers on their own held-out set via this repo's `/api/v1/analyze` interface.

---

## 5. Architecture, robustness/calibration approach, and known limitations

**Architecture.** `Image → shared preprocessing (224×224, BILINEAR resize, ImageNet normalization) → [RGB branch: EfficientNet-B4, ImageNet-pretrained] + [Frequency branch: log-magnitude FFT spectrum → small CNN] → concatenation fusion (1792 + 64 = 1856-dim) → linear classifier → sigmoid → threshold @ 0.50 → Grad-CAM (RGB branch's final conv block) + grounded explanation text`. See `model/architectures/`, `model/training/`, and `.claude/specs/06-frequency-fusion.md` for the full design rationale.

**Calibration.** Temperature scaling is implemented (`model/calibration/`) and a fitted temperature (T≈1.184) exists, but is currently **disabled** in the deployed backend (`CALIBRATION_ENABLED=False`) pending further validation — reported confidences are raw, uncalibrated `sigmoid(logit)` values. This is disclosed rather than presented as calibrated.

**Robustness.** Training augmentation deliberately excludes JPEG recompression, blur, and noise (these can destroy or fabricate the exact forensic signal the model relies on — see `model/training/augmentation.py`). No degradation-robustness benchmark (Bonus Module C) ships with this submission.

**Known limitations (honest disclosure, not resolved in this submission):**
- The unseen-generator benchmark above is a small development shard with the real-image pairing caveat described in Section 4 — treat the 0.94 unseen-AUC as a promising development signal, not a final result.
- Manual testing with real-world photos and screenshots (outside the GenImage distribution) has surfaced false positives — genuine phone photos and screenshots sometimes classified as "likely AI-generated." A rigorous, source-disjoint authentic-image benchmark and root-cause investigation is in progress but not part of this submission's shipped evaluation.
- The frequency branch alone underperforms the RGB branch on this benchmark (0.8438 vs. 0.9209 unseen-AUC) and its contribution to real-world robustness (vs. potential sensitivity to compression/resizing artifacts) has not been independently isolated in a controlled experiment within this submission.
- No degradation-robustness (Module C), generator-attribution (Module B), or multimodal (Module E) evaluation is included.

---

## 6. Demo video and deployed app

- **Demo video:** https://drive.google.com/file/d/1iMBgXU1MxSOKLfKZJlnN_QkJJMKdOiok/view?usp=sharing
- **Deployed app:** _[add a public URL here if one is deployed; otherwise the app is run locally per Section 2 above]_

---

## Repository structure

```
SIH_Signalscope/
├── backend/                 # FastAPI backend (ImageDetector contract, Grad-CAM, calibration, storage)
├── frontend/                # React + Vite web UI
├── model/                   # Architectures, training, evaluation, calibration (ML foundation)
├── data/                    # Manifest schema, generator-disjoint splitting, preprocessing
├── experiments/             # Ablation results, calibration runs, training run artifacts (gitignored)
├── report/                  # One-page model report (Section 7.3 of the problem statement)
├── tests/                   # ML pipeline test suite
├── requirements.txt         # Root ML dependencies (torch, torchvision, numpy, pyyaml, pyarrow, pillow)
└── CLAUDE.md                # Development conventions and ML rules
```

## Originality declaration

This project uses open-source libraries (PyTorch, torchvision, FastAPI, React, Radix UI — see `requirements.txt` / `frontend/package.json`) and an ImageNet-pretrained EfficientNet-B4 backbone (via `torchvision.models`), all cited above. The GenImage dataset is a public dataset, cited in Section 3. No public real-vs-fake notebook was copied; the training/evaluation/fusion pipeline, backend, and frontend were built for this submission. AI coding assistance (Claude Code) was used during development, per the problem statement's allowance — the working system and its evaluation are what is submitted.

## Responsible-use policy

- Outputs are strictly likelihood assessments ("likely AI-generated" / "likely real"), never definitive claims.
- Not designed to identify, profile, or adjudicate claims about specific real people or real-world/political events.
- Metadata (EXIF/C2PA) is reported as supporting context only, and does not by itself determine authenticity.
