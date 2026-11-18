import unittest
from core import review
from temporal import DwellFilter
class KitchenScope(unittest.TestCase):
    def test_no_prep_no_hairnet_checks(self):
        self.assertEqual(review([dict(label='person',box=[10,10,40,90],hairnet='absent')],None,100,100,scoped=True),[])
    def test_staff_outside_prep_ignored(self):
        self.assertEqual(review([dict(label='person',box=[10,10,40,90],hairnet='absent')],None,100,100,prep=[60,0,100,100],scoped=True),[])
    def test_staff_in_prep_reviewed(self):
        e=review([dict(label='person',box=[10,10,40,90],hairnet='absent')],None,100,100,prep=[0,0,50,100],scoped=True)
        self.assertEqual(e[0]['status'],'absent')
    def test_counter_object_excluded(self):
        self.assertEqual(review([dict(label='box',box=[10,10,40,90],floor_contact='no')],[0,0,100,100],100,100),[])
    def test_head_boxes_not_obstacles(self):
        self.assertEqual(review([dict(label='hairnet',box=[10,10,40,90])],[0,0,100,100],100,100),[])
    def test_dwell_and_repeat(self):
        f=DwellFilter(3)
        self.assertEqual(f.update(0,['box1']),[])
        self.assertEqual(f.update(2,['box1']),[])
        self.assertEqual(f.update(3,['box1']),['box1'])
        self.assertEqual(f.update(4,['box1']),[])
    def test_gap_resets_dwell(self):
        f=DwellFilter(3);f.update(0,['box']);f.update(2,[])
        self.assertEqual(f.update(3,['box']),[])
        self.assertEqual(f.update(5,['box']),[])
        self.assertEqual(f.update(6,['box']),['box'])
