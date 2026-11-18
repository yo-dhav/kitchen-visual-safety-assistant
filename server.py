"""Local application: python server.py, then http://127.0.0.1:8765."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from io import BytesIO
import base64, json, os, threading, socket
from PIL import Image, ImageOps
from core import review
from hairnet import associate
from annotation import export_sample

ROOT = Path(__file__).parent
LOCK = threading.Lock()
MODEL = None
HAIRNET_MODEL = None

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(ROOT/'static'),**kwargs)

    def do_GET(self):
        if self.path != '/api/status':
            return super().do_GET()
        card=ROOT/'models/model_card.json'
        info={'hairnet_model_available':(ROOT/'models/hairnet.pt').is_file(),'experimental':True}
        if card.exists():
            info['test_metrics']=json.loads(card.read_text()).get('test_metrics',{})
        payload=json.dumps(info).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(payload)

    def do_POST(self):
        if self.path not in ('/api/detect','/api/review','/api/annotation'):
            self.send_error(404); return
        try:
            length = int(self.headers.get('Content-Length',0))
            if not 0 < length <= 16_000_000:
                raise ValueError('Request too large or empty. Use an image below 10 MB.')
            data = json.loads(self.rfile.read(length))
            if self.path == '/api/annotation':
                result={'zip_base64':export_sample(data)}
            elif self.path == '/api/review':
                w,h = int(data['width']),int(data['height'])
                if not (0 < w <= 12000 and 0 < h <= 12000):
                    raise ValueError('Invalid image dimensions.')
                result = {'events':review(data['detections'],data.get('zone'),w,h,prep=data.get('prep'),scoped=True)}
            else:
                global MODEL, HAIRNET_MODEL
                raw = base64.b64decode(data['image'].split(',',1)[1],validate=True)
                im = ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert('RGB')
                if max(im.size)>12000:
                    raise ValueError('Image dimensions exceed 12000 pixels.')
                try:
                    from ultralytics import YOLO
                except ImportError:
                    raise ValueError('Automatic detection is not installed. Run: python -m pip install -r requirements-ml.txt. You can use manual review now.')
                with LOCK:
                    if MODEL is None:
                        MODEL = YOLO(os.environ.get('KITCHEN_MODEL',str(ROOT/'yolo11n.pt')))
                    pred = MODEL.predict(im,conf=.35,verbose=False)[0]
                    detections = [dict(label=pred.names[int(b.cls.item())],box=b.xyxy[0].tolist(),confidence=round(float(b.conf.item()),3),hairnet='uncertain',source='model') for b in pred.boxes]
                    bundled=ROOT/'models/hairnet.pt'
                    hairnet_path=os.environ.get('KITCHEN_HAIRNET_MODEL') or (str(bundled) if bundled.exists() else None)
                    if hairnet_path:
                        if HAIRNET_MODEL is None:
                            HAIRNET_MODEL=YOLO(hairnet_path)
                        hp=HAIRNET_MODEL.predict(im,conf=.65,verbose=False)[0]
                        heads=[dict(label=hp.names[int(b.cls.item())],box=b.xyxy[0].tolist(),confidence=float(b.conf.item())) for b in hp.boxes]
                        detections=associate(detections,heads)
                        for head in heads:
                            head['source']='model';detections.append(head)
                result = {'detections':detections}
            payload = json.dumps(result).encode()
            self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(payload)
        except Exception as exc:
            payload=json.dumps({'error':str(exc)}).encode()
            self.send_response(400); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(payload)

class LocalHTTPServer(ThreadingHTTPServer):
    allow_reuse_address=False
    def server_bind(self):
        if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()

if __name__ == '__main__':
    print('Kitchen Safety Monitor: http://127.0.0.1:8765 (Ctrl+C to stop)')
    LocalHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
