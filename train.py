"""Validate a YOLO dataset, fine-tune a detector, and evaluate held-out data."""
import argparse, hashlib, json, math, csv
from pathlib import Path

def audit(config):
    import yaml
    config=Path(config).resolve()
    data=yaml.safe_load(config.read_text())
    root=Path(data.get('path',config.parent))
    if not root.is_absolute(): root=(config.parent/root).resolve()
    names=data['names']
    names=list(names.values()) if isinstance(names,dict) else names
    if sorted(names)!=['hairnet','no_hairnet']:
        raise ValueError('Dataset must have exactly hairnet and no_hairnet classes. Remap labels explicitly first.')
    seen={}; report={'config':str(config),'names':names,'splits':{}}
    for split in ('train','val','test'):
        folder=root/str(data.get(split,''))
        if not data.get(split) or not folder.is_dir():
            raise ValueError(f'{split} must point to an existing image directory.')
        images=sorted(p for p in folder.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
        if not images: raise ValueError(f'{split} is empty')
        counts=[0]*len(names)
        for image in images:
            digest=hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen and seen[digest]!=split:
                raise ValueError(f'Exact duplicate across splits: {image}')
            seen[digest]=split
            parts=list(image.parts)
            if 'images' not in parts: raise ValueError('Expected images/ and labels/ parallel folders.')
            parts[len(parts)-1-parts[::-1].index('images')]='labels'
            label=Path(*parts).with_suffix('.txt')
            if not label.exists(): raise ValueError(f'Missing label: {label}; use empty files for verified negatives.')
            for line in label.read_text().splitlines():
                values=line.split()
                if len(values)!=5: raise ValueError(f'Invalid YOLO label: {label}')
                cls=int(values[0]); x,y,w,h=map(float,values[1:])
                if not 0<=cls<len(names) or not all(math.isfinite(v) for v in (x,y,w,h)) or not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1) or x-w/2 < -0.001 or x+w/2 > 1.001 or y-h/2 < -0.001 or y+h/2 > 1.001:
                    raise ValueError(f'Out-of-range annotation: {label}')
                counts[cls]+=1
        if not all(counts): raise ValueError(f'{split} must contain both classes.')
        report['splits'][split]={'images':len(images),'objects':dict(zip(names,counts))}
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--output',default='runs/hairnet')
    p.add_argument('--freeze',type=int,default=0)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--patience',type=int,default=7)
    p.add_argument('--class-power',type=float,default=0.0,help='Ultralytics cls_pw; supported by the pinned notebook version')
    args=p.parse_args()
    report=audit(args.data)
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=True)
    (output/'dataset_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if args.audit_only: return
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(args.threads)
    model=YOLO('yolo11n.pt')
    weighting={'cls_pw':args.class_power} if args.class_power else {}
    result=model.train(data=str(Path(args.data).resolve()),epochs=args.epochs,device=args.device,
        batch=args.batch,imgsz=args.imgsz,workers=0,seed=42,deterministic=True,
        project=str(output),name='training',exist_ok=False,patience=args.patience,freeze=args.freeze or None,**weighting)
    best=Path(result.save_dir)/'weights/best.pt'
    metrics=YOLO(str(best)).val(data=str(Path(args.data).resolve()),split='test',device=args.device,imgsz=args.imgsz,batch=args.batch,workers=0,
        project=str(output),name='held_out_test',plots=True)
    per_class={}
    for i,cls in enumerate(metrics.box.ap_class_index):
        per_class[report['names'][int(cls)]]={'precision':float(metrics.box.p[i]),'recall':float(metrics.box.r[i]),'AP50':float(metrics.box.ap50[i]),'AP50_95':float(metrics.box.ap[i])}
    history=list(csv.DictReader((Path(result.save_dir)/'results.csv').open()))
    card={'status':'trained','weights':str(best),'dataset':report,'test_metrics':metrics.results_dict,'per_class_test_metrics':per_class,
        'epochs_requested':args.epochs,'epochs_completed':len(history),'base_model':'yolo11n.pt','imgsz':args.imgsz,'freeze':args.freeze,'patience':args.patience,'class_power':args.class_power,
        'limitations':['Public data may not represent your kitchen','No detection means uncertain, not absent','Exact duplicate audit does not detect near duplicates; split by source session before training']}
    (output/'model_card.json').write_text(json.dumps(card,indent=2,default=str))
    print('Training and held-out evaluation complete:',best)

if __name__=='__main__': main()
