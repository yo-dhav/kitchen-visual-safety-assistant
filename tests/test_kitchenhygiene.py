from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import tempfile,unittest,random,json
from PIL import Image,ImageOps
import yaml
from prepare_kitchenhygiene import source_family,dhash,prepare

class KitchenHygienePreparation(unittest.TestCase):
    def test_office_aliases_share_source(self):
        self.assertEqual(source_family('hairnet_104_mp4-12_jpg.rf.hash.jpg'),source_family('hairnet_ver_10_mp4-9_jpg.rf.other.jpg'))
    def test_kitchen_cctv_is_one_group(self):
        self.assertEqual(source_family('Frame_12_802_jpg.rf.x.jpg'),source_family('Frame_8802_90_jpg.rf.y.jpg'))
    def test_unknown_augmentations_excluded(self):
        self.assertIsNone(source_family('augmented_image_11_jpg.rf.hash.jpg'))
    def test_mirror_has_same_hash(self):
        image=Image.new('L',(40,30));image.putdata([random.Random(i).randrange(256) for i in range(1200)])
        a=BytesIO();b=BytesIO();image.save(a,format='PNG');ImageOps.mirror(image).save(b,format='PNG')
        self.assertEqual(dhash(a.getvalue()),dhash(b.getvalue()))
    def test_existing_destination_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'keep.txt').write_text('retain')
            with self.assertRaises(ValueError):prepare('not-opened.zip',p)
            self.assertEqual((p/'keep.txt').read_text(),'retain')
    def test_real_preparation_remaps_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'input.zip'
            names=['apron','cockroach','gloves','hairnet','lizard','no_apron','no_gloves','no_hairnet','rat']
            with ZipFile(archive,'w') as z:
                z.writestr('data.yaml',yaml.safe_dump({'names':names}))
                first=None
                for i in range(9):
                    rng=random.Random(i);image=Image.frombytes('RGB',(32,32),bytes(rng.randrange(256) for _ in range(3072)))
                    b=BytesIO();image.save(b,format='PNG');raw=b.getvalue()
                    name='Frame_12_10.png' if i==0 else f'stocksource{i}.png'
                    z.writestr('train/images/'+name,raw);z.writestr('train/labels/'+Path(name).stem+'.txt','3 .25 .25 .2 .2\n7 .75 .75 .2 .2\n')
                    if i==0:first=raw
                z.writestr('test/images/duplicate.png',first)
                z.writestr('test/labels/duplicate.txt','3 .25 .25 .2 .2\n7 .75 .75 .2 .2\n')
            report=prepare(archive,root/'prepared')
            self.assertEqual(report['exact_duplicates_removed'],1)
            self.assertEqual(report['names'],['hairnet','no_hairnet'])
            manifest=json.loads((root/'prepared/split-manifest.json').read_text())
            components={}
            for record in manifest:
                components.setdefault(record['component'],set()).add(record['split'])
            self.assertTrue(all(len(s)==1 for s in components.values()))
