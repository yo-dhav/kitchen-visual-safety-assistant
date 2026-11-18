import base64,json,unittest
from io import BytesIO
from zipfile import ZipFile
from PIL import Image
from annotation import export_sample

class AnnotationExport(unittest.TestCase):
    def data(self):
        b=BytesIO();Image.new('RGB',(100,100),'white').save(b,format='JPEG')
        return dict(image='data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode(),session='cameraA-day01',verified=True,detections=[dict(label='hairnet',box=[10,10,30,30],source='manual')])
    def test_verified_head_export(self):
        result=export_sample(self.data())
        with ZipFile(BytesIO(base64.b64decode(result))) as z:
            label=next(n for n in z.namelist() if n.endswith('label.txt'))
            self.assertTrue(z.read(label).startswith(b'0 '))
    def test_predictions_not_training_labels(self):
        d=self.data();d['detections'][0]['source']='model'
        with self.assertRaises(ValueError):export_sample(d)
    def test_attestation_required(self):
        d=self.data();d['verified']=False
        with self.assertRaises(ValueError):export_sample(d)
    def test_uncertainty_not_silently_negative(self):
        d=self.data();d['detections'].append(dict(label='uncertain_head',box=[40,40,60,60],source='manual'))
        with self.assertRaises(ValueError):export_sample(d)
    def test_illustration_not_real_training_data(self):
        d=self.data();d['image_source']='illustrative-kitchen-blocked'
        with self.assertRaises(ValueError):export_sample(d)
    def test_online_photo_credit_preserved(self):
        d=self.data();d['image_source']='online-kitchen-Lars-Plougmann-CC-BY-SA-2.0'
        with ZipFile(BytesIO(base64.b64decode(export_sample(d)))) as z:
            self.assertTrue(any(n.endswith('ATTRIBUTION.txt') for n in z.namelist()))
