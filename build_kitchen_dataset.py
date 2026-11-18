"""Merge human-verified app exports with explicit session-based split assignment."""
from pathlib import Path
from zipfile import ZipFile
import argparse,json,hashlib

def main():
    p=argparse.ArgumentParser()
    p.add_argument('exports',help='Folder containing verified sample ZIP files')
    p.add_argument('destination',help='New, empty output folder')
    p.add_argument('--train-sessions',required=True)
    p.add_argument('--val-sessions',required=True)
    p.add_argument('--test-sessions',required=True)
    a=p.parse_args();dest=Path(a.destination).resolve()
    assignments={}
    for split,text in [('train',a.train_sessions),('val',a.val_sessions),('test',a.test_sessions)]:
        sessions=[s.strip() for s in text.split(',') if s.strip()]
        if not sessions:p.error('Every split needs at least one capture session.')
        for session in sessions:
            if session in assignments:p.error('A session cannot appear in multiple splits.')
            assignments[session]=split
    if dest.exists() and any(dest.iterdir()):p.error('Destination must be empty.')
    records=[];seen={}
    for archive in sorted(Path(a.exports).glob('*.zip')):
        with ZipFile(archive) as z:
            metadata=[n for n in z.namelist() if n.endswith('/metadata.json')]
            for name in metadata:
                m=json.loads(z.read(name));session=m.get('session')
                if not m.get('human_verified') or session not in assignments:raise ValueError(f'Unverified or unassigned session in {archive}')
                folder=name.rsplit('/',1)[0];image=z.read(folder+'/image.jpg');label=z.read(folder+'/label.txt')
                digest=hashlib.sha256(image).hexdigest()
                if digest in seen:raise ValueError('Repeated image detected; resolve duplicate or revised exports before building.')
                seen[digest]=session;split=assignments[session]
                records.append(dict(image=image,label=label,id=digest[:20],split=split,session=session))
    if not records:raise ValueError('No verified sample exports found.')
    for r in records:
        for kind,key,suffix in [('images','image','.jpg'),('labels','label','.txt')]:
            target=dest/r['split']/kind/(r['id']+suffix);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(r[key])
    cfg=dest/'data.yaml'
    cfg.write_text('path: '+dest.as_posix()+'\ntrain: train/images\nval: val/images\ntest: test/images\nnames: [hairnet, no_hairnet]\n')
    from train import audit
    (dest/'audit.json').write_text(json.dumps(audit(cfg),indent=2))
    (dest/'sessions.json').write_text(json.dumps(assignments,indent=2))
    print('Prepared and audited dataset:',cfg)

if __name__=='__main__':main()
