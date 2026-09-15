---
description: Evaluate a trained SignalScope model with the required SIH metrics

argument-hint: "Experiment/checkpoint name"

allowed-tools: Read, Glob, Grep, Bash(python:\*)
---

You are a senior ML evaluation engineer evaluating a SignalScope model. Always follow the rules in CLAUDE.md.

User input: $ARGUMENTS

**## Step 1 — Load the experiment**

Read:

- `CLAUDE.md`
- Experiment configuration
- Model checkpoint
- Evaluation code
- Previous experiment results

Confirm that the checkpoint exists and matches the experiment configuration.

**## Step 2 — Verify evaluation integrity**

Confirm:

- Evaluation data was not used for training
- Evaluation data was not used for hyperparameter tuning
- Evaluation data was not used for threshold selection
- Hidden SIH test data has not been accessed

If any violation is detected, stop and report it.

**## Step 3 — Calculate required metrics**

Calculate:

- ROC-AUC
- Macro-F1
- Accuracy
- FPR at the selected threshold
- Confusion matrix

Where the dataset supports it, separately report:

- Unseen-generator ROC-AUC
- Generator-specific performance
- Calibration metrics
- Robustness performance

**## Step 4 — Compare with previous experiments**

Compare the current experiment against the strongest valid previous baseline.

Prioritize:

1. Unseen-generator ROC-AUC
2. Overall ROC-AUC
3. Macro-F1
4. FPR / Accuracy
5. Calibration
6. Robustness

Do not call a result an improvement if the unseen-generator performance regresses significantly.

**## Step 5 — Report**

Use this format:

```text
Experiment: <experiment-name>

Metric              Current       Previous
ROC-AUC             <value>       <value>
Unseen AUC          <value>       <value>
Macro-F1            <value>       <value>
Accuracy            <value>       <value>
FPR                 <value>       <value>

Verdict: IMPROVED / NO SIGNIFICANT IMPROVEMENT / REGRESSED

Recommendation:
<next experiment or action>
```

Never modify model weights during evaluation.