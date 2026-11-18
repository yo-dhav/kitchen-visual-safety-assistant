"""Offline video inference with tracked-object dwell filtering and evidence output."""
import argparse, csv, json
from pathlib import Path
from core import review
from hairnet import associate
from temporal import DwellFilter

def main():
    p=argparse.ArgumentParser()
    p.add_argument('video'); p.add_argument('--hairnet-model')
    p.add_argument('--zone',required=True,help='x1,y1,x2,y2 in original video pixels')
    p.add_argument('--prep',help='Food-preparation worker area, x1,y1,x2,y2; needed for head-covering checks')
    p.add_argument('--dwell',type=float,default=3)
    p.add_argument('--output',default='video-results')
    args=p.parse_args()
    if args.dwell<0: p.error('Dwell must be nonnegative')
    import cv2
    from ultralytics import YOLO
    root=Path(__file__).parent
    detector=YOLO(str(root/'yolo11n.pt'))
    model_path=args.hairnet_model or str(root/'models/hairnet.pt')
    hairnet=YOLO(model_path) if args.prep and Path(model_path).exists() else None
    zone=list(map(float,args.zone.split(',')))
    prep=list(map(float,args.prep.split(','))) if args.prep else None
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise ValueError('Could not open video')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<=0: raise ValueError('Video must have a valid frame rate')
    width,height=int(cap.get(3)),int(cap.get(4))
    writer=cv2.VideoWriter(str(out/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened(): raise ValueError('Video encoder unavailable')
    dwell=DwellFilter(args.dwell); events=[]; frame_id=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            now=frame_id/fps; frame_id+=1
            pred=detector.track(frame,persist=True,tracker='bytetrack.yaml',conf=.35,verbose=False)[0]
            objects=[]; ids=[]
            for b in pred.boxes:
                objects.append(dict(label=pred.names[int(b.cls.item())],box=b.xyxy[0].tolist(),hairnet='uncertain'))
                ids.append(int(b.id.item()) if b.id is not None else None)
            if hairnet:
                hp=hairnet.predict(frame,conf=.65,verbose=False)[0]
                associate(objects,[dict(label=hp.names[int(b.cls.item())],box=b.xyxy[0].tolist(),confidence=float(b.conf.item())) for b in hp.boxes])
            candidates=review(objects,zone,width,height,prep=prep,scoped=True)
            keyed={}
            for event in candidates:
                idx=event['object']-1; tid=ids[idx]
                if tid is None or (event['type']=='hairnet_review' and event['status']!='absent'): continue
                keyed[(tid,event['type'])]=event
            for key in dwell.update(now,keyed):
                event=keyed[key];tid=key[0]
                events.append(dict(time_seconds=round(now,3),duration_seconds=args.dwell,track_id=tid,review_decision='pending',**event))
                x1,y1,x2,y2=map(int,event['box'])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,80,255),3)
                cv2.imwrite(str(out/f'evidence-{frame_id}-{tid}.jpg'),frame)
            cv2.rectangle(frame,tuple(map(int,zone[:2])),tuple(map(int,zone[2:])),(80,190,50),2)
            if prep:cv2.rectangle(frame,tuple(map(int,prep[:2])),tuple(map(int,prep[2:])),(220,120,40),2)
            for d in objects:
                x1,y1,x2,y2=map(int,d['box']); cv2.rectangle(frame,(x1,y1),(x2,y2),(220,140,30),2)
                cv2.putText(frame,d['label']+' '+d.get('hairnet',''),(x1,max(20,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,40,20),1)
            writer.write(frame)
    finally:
        cap.release();writer.release()
    (out/'events.json').write_text(json.dumps(events,indent=2))
    with (out/'events.csv').open('w',newline='') as f:
        keys=['time_seconds','track_id','type','label','status','detail']
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(events)
    print(f'Processed {frame_id} frames; {len(events)} sustained review events. Output: {out}')

if __name__=='__main__':main()
