"""Download an attributed public export using the user's provider access."""
import os
from pathlib import Path

if __name__=='__main__':
    key=os.environ.get('ROBOFLOW_API_KEY')
    if not key:
        raise SystemExit('Set ROBOFLOW_API_KEY locally, or download YOLOv8 format from https://universe.roboflow.com/agts/hairnet/dataset/1 and extract into data/hairnet.')
    from roboflow import Roboflow
    dest=Path('data/hairnet').resolve()
    if dest.exists() and any(dest.iterdir()): raise SystemExit('Destination is not empty; preserving existing data.')
    rf=Roboflow(api_key=key)
    rf.workspace('agts').project('hairnet').version(1).download('yolov8',location=str(dest))
    print('Downloaded. Review split quality and run train.py --audit-only before training.')
