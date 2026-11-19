# Replacement head detector: evaluation

The current bundled checkpoint is the KitchenHygiene v3 head-only replacement, trained for 50 epochs in Colab. The best validation checkpoint is epoch 48. Training took approximately 28 minutes. It replaces the original agts/hairnet baseline; a historical model card and Git history retain that baseline.

## Current held-out results

| Metric | Replacement model |
|---|---:|
| Mean precision | 79.4% |
| Mean recall | 69.2% |
| mAP at IoU 0.50 | 74.7% |
| mAP at IoU 0.50–0.95 | 33.0% |

Test set: 298 images, containing 452 hairnet and 68 no_hairnet objects.

| Class | Precision | Recall | AP50 |
|---|---:|---:|---:|
| hairnet | 85.0% | 78.1% | 82.2% |
| no_hairnet | 73.9% | 60.3% | 67.2% |

These are detector metrics, not an overall safety-accuracy percentage. The app's fixed 0.65 confidence threshold is not necessarily the operating point used by the reported precision/recall summary. A missing detection remains uncertain.

## Comparison on the same test images

The old baseline was evaluated locally on the same 298 prepared test images and image size. It scored **21.7% mAP50**, compared with the replacement's **74.7%**. Precision was 31.2% versus 79.4%; recall was 41.9% versus 69.2%. Old-model evaluation used CPU and replacement results came from Colab; minor hardware numerical differences are possible. No exact image duplicates were found between old training and this test set, but near-duplicate/source overlap is not fully excluded. See reports/same-test-comparison.json.

The original 19.1% baseline score used a different test set and should not be used for this comparison.

## Important observed limitations

- The replacement produced **zero head predictions above confidence 0.65 on the bundled real restaurant photograph**. The photo has no verified safety ground truth, so this is an observed failure case, not a measured accuracy estimate. Kitchen model loading succeeds; the current app correctly leaves uncertain heads for human review.
- No-hairnet recall is only about 60%, meaning many labeled uncovered heads were missed even on the public test set.
- This mixed-domain dataset contains office demonstrations, kitchen CCTV, food-production media and stock photos. Known kitchen CCTV sources are kept together in training. The holdout is not independent college-kitchen footage.
- Source-family and near-duplicate grouping reduce obvious leakage but do not establish full scene independence. The test class distribution is uneven.
- These metrics do not validate person/head association, preparation-area assignment, walkway overlap or temporal event accuracy.

## Readiness

An improved experimental student-project detector, suitable for demonstrating an evaluated prototype with human review. It is not sufficiently validated for unattended safety monitoring or enforcement. No fire, food-temperature, microbial-contamination or overall sanitation capability is claimed.

## Next evaluation

Collect independently labeled real kitchen sessions from different locations and cameras, including small/occluded heads and chef hats. Agree on label meaning before training. Measure head detection, association and false/missed review events separately. Keep future test sessions untouched by training and model selection.
