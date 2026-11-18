"""Export explicitly human-labeled head boxes, never model guesses, for training."""
from io import BytesIO
from zipfile import ZipFile,ZIP_DEFLATED
from PIL import Image,ImageOps
from core import box
import base64,hashlib,json,re

def export_sample(data):
    if str(data.get('image_source','')).startswith('illustrative'):
        raise ValueError('Illustrated scenarios are not verified real-world training examples.')
    if not data.get('verified'):raise ValueError('Confirm that you checked all visible heads and have permission to use the image.')
    session=str(data.get('session','')).strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{3,60}',session):raise ValueError('Use a session ID of 3–60 letters, numbers, underscores or hyphens.')
    raw=base64.b64decode(data['image'].split(',',1)[1],validate=True)
    image=ImageOps.exif_transpose(Image.open(BytesIO(raw)))
    if max(image.size)>6000:raise ValueError('Image exceeds 6000 pixels per side.')
    w,h=image.size; labels=[]; annotations=[]
    if any(d.get('source')=='manual' and d.get('label')=='uncertain_head' for d in data['detections']):
        raise ValueError('This image has uncertain head labels. Resolve them or exclude this image from training.')
    for item in data['detections']:
        if item.get('source')!='manual' or item['label'] not in ('hairnet','no_hairnet'):continue
        x1,y1,x2,y2=box(item['box'],w,h);cls=0 if item['label']=='hairnet' else 1
        labels.append(f'{cls} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}')
        annotations.append(item)
    if not labels:raise ValueError('Draw verified head boxes labeled hairnet or no_hairnet first. Person boxes and predictions are not training labels.')
    identifier=hashlib.sha256(raw).hexdigest()[:20]
    output=BytesIO(); encoded=BytesIO();image.convert('RGB').save(encoded,format='JPEG',quality=95)
    meta={'session':session,'image_id':identifier,'width':w,'height':h,'annotations':annotations,'human_verified':True,'source':'user supplied image; permission attested by reviewer'}
    credit=None
    if str(data.get('image_source','')).startswith('online-kitchen-Lars'):
        credit='Restaurant kitchen by Lars Plougmann. Source: https://www.flickr.com/photos/75062596@N00/54174155431 . CC BY-SA 2.0: https://creativecommons.org/licenses/by-sa/2.0/ . Image re-encoded as JPEG; human annotations added. Preserve credit and share-alike terms for image derivatives.'
        meta['attribution']=credit
    with ZipFile(output,'w',ZIP_DEFLATED) as z:
        z.writestr(f'{identifier}/image.jpg',encoded.getvalue())
        z.writestr(f'{identifier}/label.txt','\n'.join(labels)+'\n')
        z.writestr(f'{identifier}/metadata.json',json.dumps(meta,indent=2))
        if credit:z.writestr(f'{identifier}/ATTRIBUTION.txt',credit)
    return base64.b64encode(output.getvalue()).decode()
