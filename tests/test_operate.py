import os

import unittest
import cPickle
from collections import Counter

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

import eights.operate as op
import eights.utils as utils
import eights.perambulate as per

import utils_for_tests as uft

class TestOperate(unittest.TestCase):
    def __pkl_store(self, obj, key):
        with open(uft.path_of_data(key + '.pkl'), 'w') as pkl:
            cPickle.dump(obj, pkl)

    def __get_ref_pkl(self, key):
        with open(uft.path_of_data(key + '.pkl')) as pkl:
            return cPickle.load(pkl)

    def __compare_to_ref_pkl(self, result, key):
        ref = self.__get_ref_pkl(key)
        self.assertEqual(ref, result) 

    def test_operate(self):
        M, y = uft.generate_test_matrix(100, 5, 2, random_state=0)
        for label, clfs in zip(('std',), (op.DBG_std_clfs,)):
            exp = per.Experiment(M, y, clfs)
            result = {str(key) : val for key, val in 
                      exp.average_score().iteritems()}
            print label
            print '='*80
            print result
            print
            self.__pkl_store(result, 'test_operate_{}'.format(label))

if __name__ == '__main__':
    unittest.main()
