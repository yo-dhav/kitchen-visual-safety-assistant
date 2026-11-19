# KitchenHygiene v3 replacement-model experiment

## Scope

Prepare a drop-in replacement for the app's current **hairnet/no_hairnet** detector. The original archive also contains apron, glove and pest labels, but this experiment does not train those capabilities or imply that the app already supports them.

Source: Kitchen Hygiene, KitchenHygiene version 3, CC BY 4.0.
https://universe.roboflow.com/kitchen-hygiene-efuu5/kitchenhygiene/dataset/3

## What the supplied archive actually contains

- 9,304 images: 6,632 train, 1,614 validation and 1,058 test. The public listing advertises 9,400; these counts reflect the downloaded archive.
- Three exact cross-split image duplicates were found across the full archive. Those examples are not retained in the selected head subset.
- Many source videos appear in multiple original splits, so those splits are not used unchanged.
- 675 non-box annotation lines are valid apron polygons, not corrupt head annotations. Head labels are bounding boxes.
- Images are mixed-domain: kitchen CCTV, office demonstrations, food-production media, stock imagery and pest footage. The dataset title does not make every image a kitchen example.

## Prepared head-only subset

| Split | Images | Hairnet objects | No-hairnet objects |
|---|---:|---:|---:|
| Train | 1,603 | 1,291 | 2,296 |
| Validation | 232 | 415 | 118 |
| Test | 298 | 452 | 68 |

7,065 images without head labels are excluded rather than treated as verified head-free negatives. Another 106 unknown-origin augmented images are excluded. The remaining 2,133 images are remapped to class 0 hairnet and class 1 no_hairnet.

Known kitchen CCTV frames are kept together in training, as are known office aliases. Food-production media are held out. Other source families are assigned deterministically using seed 42. Mirrored difference hashes with Hamming radius 3 connect potential near duplicates before split assignment. This groups related samples; it does not prove complete independence or remove all near-duplicate frames.

The held-out set is **not independent college-kitchen footage**. It is class-imbalanced and includes other domains. Read per-class scores and inspect errors. Use new, independently labeled kitchens for deployment-oriented validation.

## Reproduce preparation

```powershell
python prepare_kitchenhygiene.py KitchenHygiene.v3i.yolov8.zip data/kitchenhygiene-heads
python train.py --data data/kitchenhygiene-heads/data.yaml --audit-only
```

The destination must be new/empty. The original ZIP is never edited. Audit output includes counts and limitations; the manifest records source-family and component assignments. Full images stay outside Git.

## Train on Colab

Use `notebooks/Train_KitchenHygiene_v3.ipynb`. It contains its own preparation/training code and needs **one dataset ZIP**. It accepts either the prepared two-class ZIP or the original nine-class ZIP. Select a GPU runtime, then run the cells in order. No GitHub token or Roboflow API key is needed after the ZIP has been downloaded.

Training starts from general YOLO11n weights, **not the previous fine-tuned head checkpoint**, because the public datasets may overlap. The experiment requests 50 epochs, patience 15 and mild inverse-frequency class weighting (cls_pw 0.25), using the pinned Ultralytics version. Checkpoint selection uses validation data, then test data is evaluated once. The model card includes actual completed epochs and per-class precision, recall, AP50 and AP50-95.

No replacement training has completed yet, and the currently bundled weights remain the original baseline. New scores cannot be compared directly to the old 19.1% mAP50 because the test set changed. Do not claim improved accuracy until the new result has been inspected and compared appropriately.
