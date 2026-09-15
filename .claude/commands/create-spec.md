---
description: Create a technical specification for the next SignalScope AI/ML step

argument-hint: "Step number and feature name e.g. 2 FFT frequency branch"

allowed-tools: Read, Write, Glob, Bash(git:\*)
---

You are a senior AI/ML engineer spinning up a new feature for the SignalScope image authenticity detector. Always follow the rules in CLAUDE.md.

User input: $ARGUMENTS

**## Step 1 — Check working directory is clean**

Run `git status` and check for uncommitted, unstaged, or untracked files. If any exist, stop immediately and tell the user to commit or stash changes before proceeding.

DO NOT CONTINUE until the working directory is clean.

**## Step 2 — Parse the arguments**

From $ARGUMENTS extract:

1. `step_number` — zero-padded to 2 digits: 2 → 02, 11 → 11

2. `feature_title` — human readable title in Title Case

   - Example: "Baseline Detector" or "FFT Frequency Branch"

3. `feature_slug` — git and file safe slug

   - Lowercase, kebab-case
   - Only a-z, 0-9 and -
   - Maximum 40 characters
   - Example: baseline-detector, fft-frequency-branch

4. `branch_name` — format: `feature/<feature_slug>`

   - Example: `feature/fft-frequency-branch`

If you cannot infer these from $ARGUMENTS, ask the user to clarify before proceeding.

**## Step 3 — Check branch name is not taken**

Run `git branch` to list existing branches.

If `branch_name` is already taken, append a number:

`feature/fft-frequency-branch-01`, `feature/fft-frequency-branch-02` etc.

**## Step 4 — Switch to main and pull latest**

Run:

```bash
git checkout main
git pull origin main
```

**## Step 5 — Create and switch to the feature branch**

Run:

```bash
git checkout -b <branch_name>
```

**## Step 6 — Research the codebase**

Read these files before writing the spec:

- `CLAUDE.md` — roadmap, architecture, conventions and ML rules
- Relevant files in `app/`
- Relevant files in `model/`
- Relevant files in `experiments/`
- All files in `.claude/specs/` — avoid duplicating existing specs

Check `CLAUDE.md` to confirm the requested step is not already marked complete. If it is, warn the user and stop.

Check existing model/training/evaluation code before proposing new files or dependencies.

**## Step 7 — Write the spec**

Generate a spec document with this exact structure:

---

**# Spec: <feature_title>**

**## Overview**

One paragraph describing what this feature does and why it exists at this stage of the SignalScope roadmap.

**## Depends on**

Which previous steps, models, datasets, or infrastructure this feature requires to be complete.

**## Model / Architecture Changes**

Describe:

- New model components
- Inputs and outputs
- Feature extraction
- Fusion strategy if applicable
- Training behaviour
- Inference behaviour

If none: state "No model architecture changes".

**## Data Changes**

Describe any dataset, preprocessing, augmentation, split, or metadata changes.

Always preserve train/validation/test separation.

The hidden SIH test set must never be used for training, tuning, threshold selection, or experiment development.

If none: state "No data changes".

**## API / Inference Changes**

Every new or modified endpoint:

- `METHOD /path` — description — access level

If no API changes: state "No API changes".

**## Explainability Changes**

Describe any changes to:

- Grad-CAM or attention maps
- Artifact localization
- Evidence extraction
- Human-readable explanations
- Confidence or uncertainty

If none: state "No explainability changes".

**## Evaluation Plan**

Define the exact metrics and comparisons required.

Always consider:

- Overall ROC-AUC
- Unseen-generator ROC-AUC where available
- Macro-F1
- Accuracy
- FPR at the selected threshold
- Confusion matrix
- Calibration where applicable
- Robustness where applicable

The primary success criterion should prioritize unseen-generator generalization.

**## Files to change**

Every file that will be modified.

**## Files to create**

Every new file that will be created.

**## New dependencies**

Any new pip packages.

If none: state "No new dependencies".

**## Rules for implementation**

Specific constraints Claude must follow. Always include:

- Do not use the hidden SIH test set for training or tuning
- Do not introduce data leakage between train and validation/test splits
- Prioritize unseen-generator ROC-AUC over raw training accuracy
- Use reproducible random seeds
- Keep model configuration separate from source code
- Do not hardcode dataset paths
- Record experiment configuration and metrics
- Do not claim an improvement without measured comparison
- Explanations must be grounded in model evidence
- Do not make absolute claims such as "this image is definitely AI-generated"
- Use responsible wording such as "likely AI-generated"
- Do not analyze or identify real people
- Do not build political or event-claim detection features

**## Definition of done**

A specific testable checklist. Each item must be something that can be verified by running the training, evaluation, inference, or application.

Every ML feature must include measurable evaluation against the appropriate baseline.

---

**## Step 8 — Save the spec**

Save to:

`.claude/specs/<step_number>-<feature_slug>.md`

**## Step 9 — Report to the user**

Print a short summary in this exact format:

```text
Branch:    <branch_name>
Spec file: .claude/specs/<step_number>-<feature_slug>.md
Title:     <feature_title>
```

Then tell the user:

"Review the spec at `.claude/specs/<step_number>-<feature_slug>.md` then enter Plan Mode with Shift+Tab twice to begin implementation."

Do not print the full spec in chat unless explicitly asked.