"""Run the custom hairnet detector directly on an image or folder."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('source');p.add_argument('--model',default='models/hairnet.pt')
    p.add_argument('--confidence',type=float,default=.65)
    p.add_argument('--output',default='predictions')
    a=p.parse_args()
    if not Path(a.model).is_file():p.error('Trained checkpoint not found. Train first or specify --model.')
    if not 0<a.confidence<1:p.error('Confidence must be between 0 and 1.')
    from ultralytics import YOLO
    model=YOLO(a.model)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    records=[]
    for result in model.predict(a.source,conf=a.confidence,stream=True,verbose=False):
        filename=Path(result.path).name
        result.save(filename=str(out/filename))
        records.append({'source':result.path,'detections':[{'class':result.names[int(b.cls.item())],
            'confidence':float(b.conf.item()),'box':b.xyxy[0].tolist()} for b in result.boxes]})
    (out/'detections.json').write_text(json.dumps(records,indent=2))
    print('Annotated predictions and JSON saved to',out)

if __name__=='__main__':main()
