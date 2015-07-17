#l = [2000,2001,2002,2003,2004,2005,2006,2007]
#
#dist = [1,2,5,10,15]
#w = 2
#tst_w = 1
#ret = expanding_window(l,w,tst_w)
#for x in ret: print x 

import unittest

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

import utils_for_tests as utils
utils.add_to_python_path()

from eights.operate.operate_helper import feature_pairs_in_tree

class TestOperate(unittest.TestCase):
    def test_feature_pairs_in_tree(self):
        iris = datasets.load_iris()
        rf = RandomForestClassifier(random_state=0)
        rf.fit(iris.data, iris.target)
        dt = rf.estimators_[0]
        result = feature_pairs_in_tree(dt)
        ctrl = [[(3, 2)], [(2, 3), (2, 0)], [(0, 2), (3, 1)]]
        self.assertEqual(result, ctrl)

if __name__ == '__main__':
    unittest.main()
