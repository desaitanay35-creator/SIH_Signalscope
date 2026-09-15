# Step 7 Ablation Comparison

> Tiny-GenImage development shard - NOT the final SIH benchmark.

| Experiment | Val ROC-AUC | Unseen-generator ROC-AUC (development evaluation) | Accuracy | Macro-F1 | FPR |
|---|---|---|---|---|---|
| rgb_only | 0.9027 | 0.9209 | 0.8488 | 0.8306 | 0.2222 |
| frequency_only | 0.6535 | 0.8438 | 0.7488 | 0.7284 | 0.2917 |
| rgb_frequency_fusion | 0.9070 | 0.9396 | 0.8767 | 0.8599 | 0.2083 |

> *Real images for this unseen-generator result come from the validation split: data.splitting.split_manifest routes every real image into train/val and never into test (only non-real generators can be held out), so the aggregate unseen-generator set pairs test's held-out-generator fakes with val's real images. See .claude/specs/05-evaluation-pipeline.md, "Implementation note: real-image pairing for the unseen-generator result".*

## Extended metrics

| Experiment | Val Loss | Confusion Matrix | Num Val | Num Test | Num Unseen-Eval |
|---|---|---|---|---|---|
| rgb_only | 0.4121 | TP=253 TN=112 FP=32 FN=33 | 257 | 286 | 430 |
| frequency_only | 0.6321 | TP=220 TN=102 FP=42 FN=66 | 257 | 286 | 430 |
| rgb_frequency_fusion | 0.4089 | TP=263 TN=114 FP=30 FN=23 | 257 | 286 | 430 |

## Per-generator Unseen-generator ROC-AUC (development evaluation)

| Generator (unseen) | rgb_only | frequency_only | rgb_frequency_fusion |
|---|---|---|---|
| BigGAN | 0.8993 | 0.9893 | 0.9390 |
| Midjourney | 0.9425 | 0.6982 | 0.9403 |

## Per-generator seen-generator (validation) ROC-AUC

| Generator (seen) | rgb_only | frequency_only | rgb_frequency_fusion |
|---|---|---|---|
| ADM | 0.8819 | 0.6764 | 0.8944 |
| GLIDE | 0.9444 | 0.9919 | 0.9711 |
| SD15 | 0.8966 | 0.4203 | 0.8827 |
| VQDM | 0.8456 | 0.7451 | 0.8399 |
| Wukong | 0.9354 | 0.4556 | 0.9354 |

