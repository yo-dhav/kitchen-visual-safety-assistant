import unittest
from hairnet import associate

class Association(unittest.TestCase):
    def person(self): return {'label':'person','box':[0,0,100,200]}
    def head(self,label,conf=.9): return {'label':label,'confidence':conf,'box':[30,10,60,40]}
    def test_explicit_absence(self):
        self.assertEqual(associate([self.person()],[self.head('no_hairnet')])[0]['hairnet'],'absent')
    def test_no_detection_not_absence(self):
        self.assertEqual(associate([self.person()],[])[0]['hairnet'],'uncertain')
    def test_conflicting_predictions(self):
        self.assertEqual(associate([self.person()],[self.head('hairnet'),self.head('no_hairnet')])[0]['hairnet'],'uncertain')
    def test_ambiguous_person(self):
        self.assertTrue(all(p['hairnet']=='uncertain' for p in associate([self.person(),self.person()],[self.head('hairnet')])))
    def test_low_confidence(self):
        self.assertEqual(associate([self.person()],[self.head('hairnet',.3)])[0]['hairnet'],'uncertain')
