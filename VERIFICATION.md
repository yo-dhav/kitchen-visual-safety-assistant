# Version 2 verification

- 27 unit tests passed, covering spatial scope, floor-contact exclusions, uncertainty, dwell timing, session data auditing and training-label export safeguards.
- JavaScript syntax check passed.
- Live HTTP checks passed for the app page, bundled real kitchen photo, scoped review endpoint and trained-model inference on the kitchen photograph.
- Four-frame repeated-photo video smoke test passed: annotated output was written and no event shorter than the three-second dwell threshold was emitted. This is software verification, not real-video accuracy evaluation.
- Interactive browser visual QA could not be completed because the browser-control tool timed out.
- Existing model weights and held-out metrics are unchanged. No new trained-model accuracy claim is made.

## KitchenHygiene v3 preparation update

- 33 unit tests pass, including source-family grouping, mirror hashing, preservation of existing destinations, class remapping and exact-duplicate handling.
- The real prepared archive contains 2,133 images; component separation and notebook YAML generation were verified.
- The pinned local Ultralytics version accepted the new class-weighting and patience settings, and the required per-class metric fields were checked.
- The self-contained Colab notebook was syntax checked. GPU training has not run, and no replacement checkpoint or accuracy measurements exist yet.

## Replacement checkpoint integration (19 November 2026)

- Returned manifest matched the prepared train/validation/test image assignments.
- Returned hairnet.pt matched training/weights/best.pt exactly; class names and checkpoint loading were verified.
- All 50 epochs were recorded; epoch 48 had the best validation mAP50-95.
- The old model was evaluated on the same new 298-image test set: mAP50 21.7%, versus 74.7% for the replacement's returned Colab evaluation. See reports/same-test-comparison.json for limitations.
- 33 software tests and JavaScript syntax checks passed after integration.
- Live page/photo/scoped API checks passed; the running app reported the exact replacement checkpoint SHA256 after inference.
- The replacement produced zero confident head detections on the bundled real photo at threshold 0.65. This failure is retained in reports/local-inference-check.json and EVALUATION.md.
- The old model remains in Git history and local backups. Real-kitchen deployment accuracy and temporal event performance remain unvalidated.
