import tempfile, unittest
from pathlib import Path
from train import audit

class DatasetAudit(unittest.TestCase):
    def dataset(self,root):
        for split in ('train','val','test'):
            (root/split/'images').mkdir(parents=True)
            (root/split/'labels').mkdir()
            (root/split/'images/a.jpg').write_bytes(split.encode())
            (root/split/'labels/a.txt').write_text('0 .25 .25 .2 .2\n1 .75 .75 .2 .2\n')
        cfg=root/'data.yaml'
        cfg.write_text('train: train/images\nval: val/images\ntest: test/images\nnames: [hairnet, no_hairnet]\n')
        return cfg
    def test_valid_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=audit(self.dataset(Path(tmp)))
            self.assertEqual(r['splits']['train']['images'],1)
    def test_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg=self.dataset(root)
            (root/'test/images/a.jpg').write_bytes(b'train')
            with self.assertRaisesRegex(ValueError,'duplicate'): audit(cfg)
    def test_missing_class_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg=self.dataset(root)
            (root/'val/labels/a.txt').write_text('0 .25 .25 .2 .2')
            with self.assertRaisesRegex(ValueError,'both classes'): audit(cfg)
