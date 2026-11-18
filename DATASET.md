# Dataset and training record

Selected public source: **hairnet**, by **agts**, version 1, Roboflow Universe.
Source: https://universe.roboflow.com/agts/hairnet/dataset/1
Publisher lists CC BY 4.0 and 1,005 images for version 1. Classes are `hairnet` and `no_hairnet`. Preserve the exported attribution/license files. No dataset is bundled in this project.

## Current evidence

The downloaded source was inspected: 734 train, 164 validation, 107 test images, with complete labels and no exact cross-split duplicates. Visual inspection showed shared scenes across the publisher's splits. A conservative filename-source regrouping was prepared: 397 train, 39 validation, 420 test; 149 unknown-origin augmented images were excluded. The original download was preserved.

Training objects: 169 hairnet and 948 no_hairnet. Validation: 47 hairnet and 14 no_hairnet. Test: 431 hairnet and 205 no_hairnet. These are object counts, not image counts. The small validation set and uneven class mix limit conclusions. Filename grouping cannot establish full session independence; see split-manifest.json.

Custom training was completed in Colab and the returned results were inspected. The bundled hairnet checkpoint was selected at epoch 9; early stopping ended training at epoch 16. Held-out precision is 34.1%, recall 28.7%, mAP50 19.1%, and mAP50-95 6.7%. These results are weak and do not support unattended monitoring. See EVALUATION.md and models/model_card.json. General-object inference still uses an upstream pretrained YOLO model, separate from the custom hairnet detector.

## Reproducible procedure

1. Install `requirements-training.txt` in your virtual environment.
2. Download version 1 in YOLOv8 format using the provider website, or set your own `ROBOFLOW_API_KEY` environment variable locally and run `python download_dataset.py`. Do not paste keys into chat or commit them.
3. Inspect example images and label definitions. Ensure `no_hairnet` boxes surround visible uncovered heads rather than full bodies. Remove incorrect examples.
4. Keep all images from the same video/session/person in one split. Existing public splits require review; augmentation variants must not cross splits. For this exact version, `python prepare_dataset.py SOURCE EMPTY_DESTINATION` reproduces the conservative filename grouping. The audit detects exact duplicates but cannot establish session independence.
5. Ensure `data.yaml` points to image folders for `train`, `val`, and `test`. Parallel label folders must exist. Fix exported relative paths if needed.
6. Run `python train.py --data data/hairnet/data.yaml --audit-only`.
7. Run `python train.py --data data/hairnet/data.yaml --device 0 --epochs 50 --batch 8` on a CUDA machine, or use `--device cpu --batch 2`. Runtime depends on hardware. You can run the same commands on a Colab GPU after uploading the project and dataset.
8. The script evaluates the best validation checkpoint on the test split once and writes a model card, dataset audit, test metrics, confusion matrix and training artifacts. Do not repeatedly tune against test results.
9. Set `KITCHEN_HAIRNET_MODEL` to the resulting `best.pt` before starting `server.py`. Keep the general object model separate.
10. Evaluate again on separately labeled college-kitchen images before making deployment claims. Record per-class precision, recall, mAP50 and mAP50-95, including uncertain cases and failures. Walkway rules need their own labels and evaluation; detector mAP does not validate them.

## Scope

This dataset teaches head-covering appearance, not fire detection, food temperatures, sanitation, or general safety certification. Public-data performance is not evidence of performance in your campus kitchen.
