import unittest
import numpy as np
from numpy.random import rand

from eights.decontaminate import label_encode
from eights.decontaminate import replace_missing_vals
import utils_for_tests 

from collections import Counter
class TestDecontaminate(unittest.TestCase):
    
    def test_label_encoding(self):
        M = np.array(
            [('a', 0, 'Martin'),
             ('b', 1, 'Tim'),
             ('b', 2, 'Martin'),
             ('c', 3, 'Martin')],
            dtype=[('letter', 'S1'), ('idx', int), ('name', 'S6')])
        ctrl = np.array(
            [(0, 0, 0),
             (1, 1, 1),
             (1, 2, 0),
             (2, 3, 0)],
            dtype=[('letter', int), ('idx', int), ('name', int)])
        self.assertTrue(np.array_equal(ctrl, label_encode(M)))
        
    def test_replace_missing_vals(self):
        M = np.array([('a', 0, 0.0, 0.1),
                      ('b', 1, 1.0, np.nan),
                      ('', -999, np.nan, 0.0),
                      ('d', 1, np.nan, 0.2),
                      ('', -999, 2.0, np.nan)],
                     dtype=[('str', 'S1'), ('int', int), ('float1', float),
                            ('float2', float)])

        ctrl = M.copy()
        ctrl['float1'] = np.array([0.0, 1.0, -1.0, -1.0, 2.0])
        ctrl['float2'] = np.array([0.1, -1.0, 0.0, 0.2, -1.0])
        res = replace_missing_vals(M, 'constant', constant=-1.0)
        self.assertTrue(np.array_equal(ctrl, res))

        ctrl = M.copy()
        ctrl['int'] = np.array([100, 1, -999, 1, -999])
        ctrl['float1'] = np.array([100, 1.0, np.nan, np.nan, 2.0])
        ctrl['float2'] = np.array([0.1, np.nan, 100, 0.2, np.nan])
        res = replace_missing_vals(M, 'constant', missing_val=0, constant=100)
        self.assertTrue(utils_for_tests.array_equal(ctrl, res))

        ctrl = M.copy()
        ctrl['int'] = np.array([0, 1, 1, 1, 1])
        res = replace_missing_vals(M, 'most_frequent', missing_val=-999)
        self.assertTrue(utils_for_tests.array_equal(ctrl, res))

        ctrl = M.copy()
        ctrl['float1'] = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
        ctrl['float2'] = np.array([0.1, 0.1, 0.0, 0.2, 0.1])
        res = replace_missing_vals(M, 'mean', missing_val=np.nan)
        self.assertTrue(utils_for_tests.array_equal(ctrl, res))
        
if __name__ == '__main__':
    unittest.main()


