import unittest
from core import review

class Rules(unittest.TestCase):
    def test_floor_overlap(self):
        events=review([{'label':'box','box':[20,20,60,80]}],[0,60,100,100],100,100)
        self.assertEqual(events[0]['type'],'possible_obstruction')
    def test_person_is_not_obstruction(self):
        events=review([{'label':'person','box':[20,20,60,80]}],[0,60,100,100],100,100)
        self.assertEqual([e['type'] for e in events],['hairnet_review'])
        self.assertEqual(events[0]['status'],'uncertain')
    def test_upper_silhouette_does_not_trigger(self):
        self.assertEqual(review([{'label':'box','box':[20,20,60,80]}],[0,0,100,50],100,100),[])
    def test_no_zone(self):
        self.assertEqual(review([{'label':'box','box':[20,20,60,80]}],None,100,100),[])
    def test_invalid_coordinates(self):
        for b in ([1,1,0,2],[0,0,101,100],[0,0,float('nan'),5]):
            with self.assertRaises(ValueError):
                review([{'label':'box','box':b}],None,100,100)
    def test_boundary_contact_is_not_overlap(self):
        self.assertEqual(review([{'label':'box','box':[0,0,20,20]}],[20,0,100,100],100,100),[])

if __name__=='__main__': unittest.main()
