"""Prepare conservative source-family splits for agts/hairnet version 1."""
import argparse, hashlib, json, shutil
from pathlib import Path
from collections import Counter
from train import audit

def family(name):
    if name.startswith('augmented_'): return None
    if name.startswith(('hairnet_','mask_data_')): return 'office_camera'
    if name.startswith('video_BC'): return 'factory_video'
    if name.startswith('vlcsnap'): return 'vlc_capture'
    if name.startswith(('Shoplifting','Normal_shopping')): return 'shopping_video'
    return 'other_images'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('source');p.add_argument('destination')
    a=p.parse_args(); source=Path(a.source).resolve(); dest=Path(a.destination).resolve()
    if source==dest or source in dest.parents or dest in source.parents:
        p.error('Use separate source and destination folders.')
    if dest.exists() and any(dest.iterdir()): p.error('Destination must be empty.')
    mapping={'office_camera':'train','shopping_video':'train','vlc_capture':'val','factory_video':'test','other_images':'test'}
    manifest=[];excluded=0
    for im in sorted(source.glob('*/images/*')):
        if not im.is_file():continue
        group=family(im.name)
        if group is None:excluded+=1;continue
        split=mapping[group]; label=im.parent.parent/'labels'/(im.stem+'.txt')
        if not label.exists():raise ValueError(f'Missing label {label}')
        target=dest/split/'images'/im.name; target_label=dest/split/'labels'/label.name
        target.parent.mkdir(parents=True,exist_ok=True);target_label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(im,target);shutil.copy2(label,target_label)
        manifest.append(dict(file=im.name,original_split=im.parent.parent.name,group=group,split=split,sha256=hashlib.sha256(im.read_bytes()).hexdigest()))
    for name in ('README.dataset.txt','README.roboflow.txt'):
        if (source/name).exists():shutil.copy2(source/name,dest/name)
    config=dest/'data.yaml'
    config.write_text('path: '+dest.as_posix()+'\ntrain: train/images\nval: val/images\ntest: test/images\nnames: [hairnet, no_hairnet]\n')
    record=dict(method='Conservative filename source-family grouping; scene independence not fully verifiable',mapping=mapping,excluded_augmented=excluded,files=manifest)
    (dest/'split-manifest.json').write_text(json.dumps(record,indent=2))
    (dest/'dataset-audit.json').write_text(json.dumps(audit(config),indent=2))
    print(config)

if __name__=='__main__':main()
