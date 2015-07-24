import os

import unittest
from collections import Counter

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

import eights.operate as op
import eights.utils as utils

import utils_for_tests

class TestOperate(unittest.TestCase):
    def test_load_and_run_rf_cv(self):
        file_loc = 'test_operate_matrix.csv'
        M, labels = utils_for_tests.generate_test_matrix(100, 20, 
                                                         random_state=0)
        to_write = utils.append_cols(M, labels, 'label')
        np.savetxt(file_loc, to_write, delimiter=',', 
                   header=','.join(to_write.dtype.names)) 
        exp = op.load_and_run_rf_cv(file_loc, -1)
        print exp.average_score()
        os.remove(file_loc)


if __name__ == '__main__':
    unittest.main()
