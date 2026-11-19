# Training data and model provenance

## Current replacement

KitchenHygiene v3, by Kitchen Hygiene, CC BY 4.0: https://universe.roboflow.com/kitchen-hygiene-efuu5/kitchenhygiene/dataset/3 . The supplied export contained 9,304 images, not the advertised 9,400.

The prepared head-only subset contains 1,603 training, 232 validation and 298 test images. Only hairnet/no_hairnet labels were used. Images with no head labels and unknown-origin augmentations were excluded. Related source families and mirrored difference-hash components were kept in one split. See KITCHENHYGIENE_V3.md and reports/kitchenhygiene-v3-audit.json for the method and limits.

The current model completed all 50 epochs, with best validation checkpoint at epoch 48. It starts from general YOLO11n weights, rather than the old fine-tuned checkpoint. Test metrics: precision 79.4%, recall 69.2%, mAP50 74.7%, mAP50-95 33.0%. Per-class metrics and the checkpoint fingerprint are in models/model_card.json. Read EVALUATION.md before interpreting these numbers.

## Historical baseline

The original model used agts/hairnet version 1, CC BY 4.0: https://universe.roboflow.com/agts/hairnet/dataset/1 . Its prepared splits were 397/39/420 images and it stopped at epoch 16, best epoch 9. Its original 19.1% mAP50 used a different test set. The old model scores 21.7% when evaluated on the replacement's 298-image test set. Its historical metadata is in reports/baseline-agts-model-card.json and the weights remain available in Git history/local backups.

## Reproduction

Use the relevant notebook under notebooks/ or the preparation/training commands in README.md and KITCHENHYGIENE_V3.md. Full dataset images are not included in Git. Retain attribution files when downloading or sharing data. No Roboflow API key is needed once the dataset ZIP is available locally.

No public-data score certifies real-kitchen safety. Source grouping is heuristic and the known kitchen CCTV data are in training. Independently labeled kitchen footage is still needed for a deployment-oriented evaluation.
