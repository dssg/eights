import unittest

from sklearn import datasets

import utils_for_tests as utils
utils.add_to_python_path()

from eights.perambulate.perambulate import *

class TestPerambulate(unittest.TestCase):

    def test_perambulate(self):
        iris = datasets.load_iris()
        y = iris.target
        M = iris.data
        clfs = {RF: {}}
        subsets = {SWEEP_TRAINING_SIZE: {'subset_size': [20, 40, 60, 80, 100]}}
        cvs = {K_FOLD: {}}
        exp = Experiment(M, y, clfs, subsets, cvs)
        print exp.average_score()

if __name__ == '__main__':
    unittest.main()
	

