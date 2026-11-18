# Trained model evaluation

The returned Colab archive contains actual fine-tuned YOLO11n weights. The bundled app automatically finds `models/hairnet.pt`. The general-object detector remains the separate pretrained `yolo11n.pt`.

## Training evidence

- 397 training images, 39 validation images and 420 test images.
- 50 epochs requested; 16 recorded in results.csv. With patience 7, this is consistent with early stopping rather than a failed run.
- Total recorded training time: approximately 141 seconds on the Colab runtime.
- Classes: hairnet, no_hairnet. Sources were grouped conservatively by filename to reduce obvious shared-scene leakage. Session independence is not fully proven.

## Held-out results from the returned model card

| Metric | Value |
|---|---:|
| Mean precision | 34.1% |
| Mean recall | 28.7% |
| mAP at IoU 0.50 | 19.1% |
| mAP averaged over IoU 0.50–0.95 | 6.7% |

These are detector metrics, not an overall “accuracy” percentage. The test plots show missed heads, false detections on background objects and confusion between covered and uncovered heads. The exported confusion matrix uses its evaluation settings; its raw false-positive counts should not be treated as the precision of the app's fixed 0.65 confidence threshold. No test-set threshold tuning has been performed here.

## Readiness

Experimental project model only. It is not sufficiently reliable for unattended safety enforcement. Human review is required, and no detection must remain uncertain. The app's person-association rules and walkway logic are additional heuristics and have not been validated by these detector metrics. The software tests check logic, not real-world safety performance.

## Next improvement cycle

Collect permitted, diverse kitchen images with both covered and uncovered heads in each source environment. Include small, partially hidden heads and true negative background images. Fix labeling inconsistencies, retain uncertain heads for human review, and split by complete capture session before augmentation. The current training set has 169 hairnet versus 948 no_hairnet objects, so add diverse positive examples rather than simply duplicating frames. Keep a new untouched evaluation set for the next model version; the current test results have now been inspected and must not become a tuning target.
