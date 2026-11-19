# Kitchen Visual Safety Assistant

An **experimental student computer-vision project** for reviewing kitchen images and recorded video. It combines a fine-tuned YOLO11n hairnet baseline with marked preparation areas, walkway-overlap rules and human review.

**Current model performance is weak.** This is a learning project, not a reliable safety monitoring or enforcement system. The application does not issue an overall safe/unsafe kitchen verdict.

## What works

- Upload a kitchen image or load the credited real restaurant photograph.
- Run general-object and hairnet/no_hairnet detection with bundled weights.
- Limit head-covering review to staff inside a user-marked preparation area.
- Mark a walkway and review possible obstructions; exclude objects confirmed off the floor.
- Confirm or dismiss review candidates and export annotated images, CSV and JSON.
- Process recorded video with tracking and a configurable consecutive dwell time.
- Draw human-verified head labels and export training samples with capture-session IDs.
- Build session-separated datasets, audit labels/duplicates, train and evaluate a detector.

Illustrated kitchen scenarios use supplied labels to test rules. They are explicitly not AI predictions or real validation footage.

## Measured baseline

| Item | Result |
|---|---:|
| Training / validation / test images | 397 / 39 / 420 |
| Training epochs completed | 16 (early stopping; best checkpoint at epoch 9) |
| Held-out mean precision | 34.1% |
| Held-out mean recall | 28.7% |
| mAP@50 | 19.1% |
| mAP@50–95 | 6.7% |

These are detection metrics, not an overall accuracy percentage. Source-family grouping reduced obvious shared-scene leakage but cannot prove session independence. Small validation data, class imbalance and domain shift limit the results. See [evaluation](EVALUATION.md) and [recorded metrics](models/model_card.json). Interface improvements have not changed the trained model's scores.

## Run locally

Requires Python 3.10 or later. On Windows, double-click `START_APP.bat`, or run:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-ml.txt
.venv\Scripts\python server.py
```

Open http://127.0.0.1:8765 . Keep the server running. On macOS/Linux use `.venv/bin/python` for the environment's interpreter. Model files are already included; package installation needs internet. No API key is needed for local inference. Images are processed on the local machine and are not automatically published.

Start with **Load real restaurant kitchen**, or choose an illustrated scenario. Mark preparation and walkway areas, inspect detections, then run area-specific checks. The real photo has no verified safety ground truth. The existing model may confuse chef hats and other head coverings.

## Recorded video

```powershell
python video.py kitchen.mp4 --zone 200,300,600,700 --prep 100,100,500,700 --dwell 3
```

Replace the example rectangles with original-frame pixel coordinates. Outputs go to `video-results/`. Tracking errors, occlusion and 2D floor-contact errors remain possible. The pipeline has a short software smoke test, not a measured real-kitchen video benchmark.

## Training and evaluation

**New experiment:** [KitchenHygiene v3 preparation and training](KITCHENHYGIENE_V3.md) uses 2,133 head-labeled images, including kitchen CCTV, with related sources grouped before splitting. Use `notebooks/Train_KitchenHygiene_v3.ipynb`; it needs just one dataset ZIP. The original trained weights and measured baseline remain unchanged until the replacement run completes.

The original dataset is [agts/hairnet v1](https://universe.roboflow.com/agts/hairnet/dataset/1), CC BY 4.0. Download its YOLOv8 export, then:

```powershell
python -m pip install -r requirements-training.txt
python prepare_dataset.py PATH_TO_ORIGINAL_EXPORT data/hairnet-grouped
python train.py --data data/hairnet-grouped/data.yaml --audit-only
python train.py --data data/hairnet-grouped/data.yaml --device 0 --epochs 50
```

Use a suitable GPU for training. A Colab notebook is included under `notebooks/`; it takes a downloaded source ZIP of this repository and the original dataset export ZIP. See [the kitchen workflow](KITCHEN_WORKFLOW.md) for verified annotation exports and session-separated new data. Do not tune against the already-inspected test set; reserve new independent kitchen footage for the next model version.

## Validation

```powershell
python -m pip install -r requirements.txt PyYAML
python -m unittest discover -s tests -v
```

33 unit tests passed, covering scoping, geometry, uncertainty, dwell timing, annotation safeguards, dataset auditing and KitchenHygiene source grouping/remapping. Live image/API inference and a four-frame repeated-photo video smoke check passed for the existing app. The new training notebook is syntax/configuration checked but has not yet run in Colab. These checks do not establish model accuracy or real-world safety. Browser visual QA was unavailable. See [verification details](VERIFICATION.md).

## Layout

- `server.py`, `core.py`, `hairnet.py`, `temporal.py`: inference and review logic
- `static/`: local web interface and credited kitchen photo
- `models/`: trained head detector and portable model card
- `train.py`, `prepare_dataset.py`, `build_kitchen_dataset.py`: data/training workflow
- `reports/`: training history, test curves and confusion matrix
- `tests/`: software tests

## License and attribution

Code is provided under AGPL-3.0; see LICENSE. Bundled weights derive from Ultralytics YOLO. The real kitchen photograph retains CC BY-SA 2.0 with attribution; dataset attribution is retained in the documentation. See [third-party notices](THIRD_PARTY.md).
