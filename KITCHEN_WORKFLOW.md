# Kitchen Visual Safety Assistant

## What this version actually does

1. The browser can load a real restaurant-kitchen photo, attributed to Lars Plougmann under CC BY-SA 2.0. No safety labels are supplied for real people in the image. It is an interface/inference example, not a validated kitchen benchmark.
2. Mark the **food-preparation work area** around where staff stand. A person's bounding-box center must fall inside it for head-covering review. This is an editable spatial heuristic; it cannot establish that a person is handling food.
3. Mark the **walkway**. The bottom part of non-person boxes is tested for overlap. Mark floor contact “no” to exclude countertop or suspended objects. Unverified floor contact remains a candidate for review. Head boxes and recognized fixed appliances are excluded.
4. Run object/head detection, then area-specific checks. Inspect predictions and choose pending, confirmed or dismissed for each review event. Export evidence images and review records. A “confirmed” decision is the human reviewer's assessment, not model certification.
5. Three clearly illustrated scenarios test an uncovered worker plus crate, a covered worker plus clear floor, and an uncertain head plus visitor outside prep. Their labels are supplied, not predicted. The visitor must not receive prep-area head review.

The existing model specifically learned hairnet/no_hairnet labels; it may confuse chef hats or other head coverings. Neither its output nor a manual appearance label establishes a regulatory violation. Decide the kitchen's head-covering policy with a responsible reviewer.

## Recorded video

With the inference environment installed, run from the project folder:

```powershell
python video.py kitchen.mp4 --zone 480,330,730,565 --prep 210,210,440,565 --dwell 3
```

Replace coordinates with areas measured in the original video frame. They are example values only. The bundled head model is used when a preparation area is provided. Without it, head checks are skipped. The video pipeline tracks object IDs and emits review events only after continuous candidates persist for the dwell time. Missing detections reset the timer. A repeated stationary image is not a real video evaluation. Tracking failures and 2D floor-contact errors remain possible.

Outputs include annotated video, evidence JPEGs and CSV/JSON review records. Video events begin pending. The dwell filter is unit-tested; real-kitchen temporal detection accuracy is not established.

## Collect useful new training data

Upload permitted kitchen images with varied people, camera angles, lighting and head coverings. Draw tight head boxes manually with hairnet/no_hairnet labels. Keep uncertain or occluded heads out of training images rather than guessing labels. Predicted boxes are never automatically exported as human labels. Check every visible head and export a verified sample under a consistent capture-session ID.

Keep complete capture sessions in a single split. Put exported ZIPs in one folder and prepare a new dataset:

```powershell
python build_kitchen_dataset.py verified-exports data/my-kitchen --train-sessions cameraA-day01,cameraB-day01 --val-sessions cameraA-day02 --test-sessions cameraC-day03
python train.py --data data/my-kitchen/data.yaml --device 0 --epochs 50
```

Each split needs both classes, and the builder rejects repeated images and sessions assigned to multiple splits. Filenames/session IDs cannot prove that scenes are independent; review source footage. Before using new imagery, retain any applicable photo credit/license and ensure permitted use. This workflow prepares data; it does not create new accuracy measurements without actual training and held-out evaluation.

## Current evidence and remaining limits

- Original head model: held-out mAP50 19.1%, precision 34.1%, recall 28.7%; weak.
- These app improvements do not change model weights or improve those scores.
- No new kitchen dataset has been labeled and used for retraining yet.
- The real kitchen photo has not been assigned ground-truth safety labels.
- No fire, food-temperature, microbial contamination, sanitation or overall kitchen-safety verdicts.
- A useful next evaluation needs independently labeled kitchen images/video, separately measured head detection, zone assignment, obstruction precision/recall and false events per minute.
