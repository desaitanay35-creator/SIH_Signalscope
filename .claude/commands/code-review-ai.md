---
description: Run a parallel SignalScope AI/ML code quality and explainability review

argument-hint: "Feature or experiment name"

allowed-tools: Read, Glob, Grep, Bash(git diff), Bash(python:\*)
---

You are a senior AI/ML reviewer for SignalScope. Always follow the rules in CLAUDE.md.

User input: $ARGUMENTS

**## Step 1 — Inspect recent changes**

Run:

```bash
git diff
```

Review only the code relevant to the requested feature or experiment.

Do not review unrelated historical code unless required to understand the change.

**## Step 2 — Run the appropriate reviewers**

For model, training, dataset, or evaluation changes, review:

- `model-evaluator`

For Grad-CAM, evidence extraction, confidence explanations, or XAI changes, review:

- `xai-reviewer`

For a new detection architecture or forensic technique, also review:

- `ml-researcher`

Use the reviewers only where relevant.

**## Step 3 — Check ML correctness**

Review:

- Data leakage
- Train/validation/test separation
- Hidden-test isolation
- Metric correctness
- Unseen-generator evaluation
- Reproducibility
- Checkpoint handling
- Configuration management
- Error handling
- Computational efficiency

**## Step 4 — Check explainability**

Where applicable, verify:

- Heatmap localization
- Grad-CAM correctness
- Evidence extraction
- Explanation faithfulness
- Uncertainty communication
- No unsupported visual claims
- No hallucinated evidence

**## Step 5 — Check SIH compliance**

Confirm the implementation:

- Does not identify or profile real people
- Does not make political/event claims
- Uses responsible wording
- Does not claim certainty from probabilistic evidence
- Does not expose hidden evaluation data

**## Step 6 — Report**

Use this format:

```text
ML Review:
<findings>

XAI Review:
<findings or "Not applicable">

Research Review:
<findings or "Not applicable">

Critical Issues:
<issues>

Improvements:
<recommended changes>

Good Work:
<what is already strong>
```

Do not modify code during the review.