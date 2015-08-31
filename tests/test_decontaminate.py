import unittest
import numpy as np
from numpy.random import rand

from eights.decontaminate.decontaminate import  replace_with_n_bins
import utils_for_tests as utils

from collections import Counter
class TestDecontaminate(unittest.TestCase):
    #1
    def test_replace_with_n_bins(self):
        d = [1,1,1,3,3,3,5,5,5,5,2,6]
        correct = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 3]
        self.assertTrue(np.array_equal(correct, replace_with_n_bins(d,3)))
    
    def test_label_encoding(self):
        self.assertTrue(np.array_equal(1,2))
        
    def test_replace_missing_vals(self):
        self.assertTrue(np.array_equal(1,2))
        
        
if __name__ == '__main__':
    unittest.main()


