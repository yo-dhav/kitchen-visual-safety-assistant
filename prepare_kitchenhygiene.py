"""Prepare head-only KitchenHygiene v3 data with conservative group isolation."""
import argparse,hashlib,json,math,re
from collections import Counter,defaultdict
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from PIL import Image,ImageOps
import yaml

OFFICE=('haiiiir','hair03','hair05','hair1','hairnet_102','hairnet_104','hairnet_ver','hairr','hairrr1','mask_data_102')

def source_family(filename):
    s=Path(filename).name.split('.rf.')[0]
    if s.startswith('augmented_image'):return None
    if s.startswith('Frame_'):return 'kitchen_cctv'
    if s.startswith(OFFICE):return 'office_camera'
    if s.startswith('video_BC'):return 'food_production_media'
    if '_mp4' in s:return 'video:'+s.split('_mp4')[0]
    s=re.sub(r'_(jpg|jpeg|png)$','',s,flags=re.I)
    return 'unverified_source:'+s

def dhash(raw):
    im=ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert('L')
    hashes=[]
    for view in (im,ImageOps.mirror(im)):
        small=view.resize((9,8));pixels=list(small.tobytes());bits=0
        for y in range(8):
            for x in range(8):bits=(bits<<1)|int(pixels[y*9+x]>pixels[y*9+x+1])
        hashes.append(bits)
    return min(hashes)

def prepare(archive,destination):
    dest=Path(destination).resolve()
    if dest.exists() and any(dest.iterdir()):raise ValueError('Destination must be new or empty; existing data was preserved.')
    with ZipFile(archive) as z:
        configs=[n for n in z.namelist() if Path(n).name=='data.yaml']
        if len(configs)!=1:raise ValueError('Expected exactly one dataset data.yaml.')
        original=yaml.safe_load(z.read(configs[0]));names=original['names']
        if isinstance(names,dict):names=[names[i] for i in sorted(names)]
        target={names.index('hairnet'):0,names.index('no_hairnet'):1}
        records=[];total=0;excluded=Counter();polygon_lines=0
        image_files=sorted(n for n in z.namelist() if '/images/' in n and Path(n).suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
        for n in image_files:
            total+=1;label=n.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt'
            try:lines=z.read(label).decode().splitlines()
            except KeyError:raise ValueError(f'Missing annotation: {label}') from None
            selected=[];counts=[0,0]
            for line in lines:
                fields=line.split();cls=int(fields[0])
                if len(fields)!=5:
                    if len(fields)<7 or len(fields)%2!=1:raise ValueError(f'Invalid label structure: {label}')
                    coordinates=list(map(float,fields[1:]))
                    if not all(math.isfinite(v) and 0<=v<=1 for v in coordinates):raise ValueError(f'Invalid polygon: {label}')
                    polygon_lines+=1
                    if cls in target:raise ValueError('A head polygon needs explicit conversion; preparation stopped.')
                    continue
                if cls not in target:continue
                x,y,w,h=map(float,fields[1:])
                if not all(math.isfinite(v) for v in (x,y,w,h)) or not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1) or min(x-w/2,y-h/2)<-.001 or max(x+w/2,y+h/2)>1.001:
                    raise ValueError(f'Invalid head box: {label}')
                new=target[cls];selected.append(f'{new} {x:.8f} {y:.8f} {w:.8f} {h:.8f}');counts[new]+=1
            if not selected:excluded['no_head_labels']+=1;continue
            family=source_family(n)
            if family is None:excluded['unknown_augmented_origin']+=1;continue
            raw=z.read(n)
            records.append(dict(member=n,source_split=n.split('/')[0],family=family,sha256=hashlib.sha256(raw).hexdigest(),dhash=dhash(raw),labels=selected,counts=counts))
        if not records:raise ValueError('No usable head-labeled images found.')
        # Union by source family, exact image identity and a conservative visual near-duplicate heuristic.
        parent=list(range(len(records)))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        def union(a,b):
            a,b=find(a),find(b)
            if a!=b:parent[b]=a
        families={};digests={};visual=[];exact_removed=set();near_pairs=0
        for i,r in enumerate(records):
            if r['family'] in families:union(i,families[r['family']])
            else:families[r['family']]=i
            if r['sha256'] in digests:
                previous=digests[r['sha256']]
                if records[previous]['labels']!=r['labels']:raise ValueError('Identical image has conflicting head labels; manual correction required.')
                union(i,previous);exact_removed.add(i)
            else:digests[r['sha256']]=i
            for j,h in visual:
                if (r['dhash']^h).bit_count()<=3:
                    union(i,j);near_pairs+=1
            visual.append((i,r['dhash']))
        components=defaultdict(list)
        for i,r in enumerate(records):components[find(i)].append(i)
        assignments={};forced={};conflicts=[]
        for key,indices in components.items():
            fs={records[i]['family'] for i in indices}
            if fs & {'kitchen_cctv','office_camera'}:
                split='train';forced[key]=True
                if 'food_production_media' in fs:conflicts.append('Food-production family connected to training sources by visual/hash similarity; retained in train.')
            elif 'food_production_media' in fs:split='test';forced[key]=True
            else:
                signature='|'.join(sorted(records[i]['member'] for i in indices))
                value=int(hashlib.sha256(('42:'+signature).encode()).hexdigest()[:8],16)%100
                split='val' if value<50 else 'test' if value<80 else 'train'
            assignments[key]=split
        # Ensure each split contains both classes using entire unforced components, never frame-level moves.
        for _ in range(8):
            counts={s:[0,0] for s in ('train','val','test')}
            for k,indices in components.items():
                for i in indices:
                    if i not in exact_removed:
                        for c in (0,1):counts[assignments[k]][c]+=records[i]['counts'][c]
            missing=[(s,c) for s in counts for c in (0,1) if not counts[s][c]]
            if not missing:break
            s,c=missing[0]
            candidates=[k for k,indices in components.items() if k not in forced and assignments[k]!=s and any(records[i]['counts'][c] for i in indices)]
            if not candidates:raise ValueError('Insufficient independent groups for both classes in every split. Supply a reviewed source mapping.')
            chosen=min(candidates,key=lambda k:(len(components[k]),min(records[i]['member'] for i in components[k])))
            assignments[chosen]=s
        else:raise ValueError('Could not build class-covered independent splits.')
        dest.mkdir(parents=True,exist_ok=True);manifest=[]
        for k,indices in components.items():
            split=assignments[k]
            for i in indices:
                if i in exact_removed:continue
                r=records[i];stem=r['sha256'][:24]
                image=dest/split/'images'/(stem+Path(r['member']).suffix.lower());label=dest/split/'labels'/(stem+'.txt')
                image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
                image.write_bytes(z.read(r['member']));label.write_text('\n'.join(r['labels'])+'\n')
                manifest.append({key:r[key] for key in ('member','source_split','family','sha256','counts')}|{'split':split,'component':str(k)})
        for name in z.namelist():
            if Path(name).name in ('README.dataset.txt','README.roboflow.txt'):(dest/Path(name).name).write_bytes(z.read(name))
    cfg=dest/'data.yaml';cfg.write_text('path: '+dest.as_posix()+'\ntrain: train/images\nval: val/images\ntest: test/images\nnames: [hairnet, no_hairnet]\n')
    from train import audit
    report=audit(cfg)
    report.update(source_dataset='KitchenHygiene v3; CC BY 4.0',source_images=total,excluded=dict(excluded),valid_non_head_polygon_lines=polygon_lines,
        exact_duplicates_removed=len(exact_removed),near_duplicate_links=near_pairs,source_components=len(components),split_method='Seed-42 source families plus mirrored dHash radius-3 components; known kitchen CCTV and office families in train; food-production media held out unless connected to train',conflicts=conflicts,
        limitations=['Mixed kitchen, office, food-production and stock imagery; not an all-kitchen benchmark','Source identification is heuristic and incomplete','Known kitchen CCTV retained entirely in train; held-out set is not independent college-kitchen footage','Near-duplicate hashing can overmerge sources and cannot guarantee independence','Both public datasets may share images; start from general YOLO weights, not the old fine-tuned checkpoint'])
    (dest/'audit.json').write_text(json.dumps(report,indent=2))
    (dest/'split-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('destination');a=p.parse_args();prepare(a.archive,a.destination)
