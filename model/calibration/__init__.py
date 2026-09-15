"""
Probability calibration for SignalScope (Step 8A).

Post-training only: no detector weights are created, modified, or
retrained by this package. Temperature scaling fits one scalar parameter
on top of a frozen checkpoint's raw logits. See
.claude/specs/08a-probability-calibration.md.
"""
