import unittest
from eights.communicate.communicate import feature_pairs_in_tree
from eights.communicate.communicate import feature_pairs_in_rf

class TestCommunicate(unittest.testcase):
    def test_feature_pairs_in_tree(self):
        iris = datasets.load_iris()
        rf = RandomForestClassifier(random_state=0)
        rf.fit(iris.data, iris.target)
        dt = rf.estimators_[0]
        result = feature_pairs_in_tree(dt)
        ctrl = [[(2, 3)], [(2, 3), (0, 2)], [(0, 2), (1, 3)]]
        self.assertEqual(result, ctrl)

    def test_feature_pairs_in_rf(self):
        iris = datasets.load_iris()
        rf = RandomForestClassifier(random_state=0)
        rf.fit(iris.data, iris.target)
        result = feature_pairs_in_rf(rf, [1, 0.5])
        # TODO make sure these results are actually correct
#        ctrl = {'Depth 3->4': 
#                Counter({(0, 3): 3, (1, 2): 2, (2, 3): 2, (0, 1): 1, 
#                         (1, 3): 1, (3, 3): 1}), 
#                'Depth 4->5': 
#                Counter({(0, 1): 1, (1, 3): 1, (3, 3): 1, (0, 2): 1}), 
#                'Depth 6->7': 
#                Counter({(0, 3): 1}), 
#                'Depth 2->3': 
#                Counter({(2, 3): 5, (0, 2): 5, (2, 2): 2, (0, 3): 2, 
#                         (1, 2): 1, (0, 1): 1, (1, 3): 1, (3, 3): 1, 
#                         (0, 0): 1}), 
#                'Overall': 
#                Counter({(2, 3): 16, (0, 2): 14, (0, 3): 12, (3, 3): 7, 
#                         (2, 2): 6, (0, 1): 4, (1, 2): 3, (1, 3): 3, 
#                         (0, 0): 2, (1, 1): 1}), 
#                'Depth 0->1': 
#                Counter({(2, 3): 3, (0, 3): 3, (3, 3): 2, (2, 2): 2, 
#                         (0, 1): 1, (0, 0): 1}), 
#                'Depth 1->2': 
#                Counter({(0, 2): 7, (2, 3): 5, (3, 3): 2, (2, 2): 2, 
#                         (0, 3): 2, (1, 1): 1}), 'Depth 5->6': 
#                Counter({(0, 3): 1, (2, 3): 1, (0, 2): 1})}
#        self.assertEqual(result, ctrl)
         # TODO Alter to deal w/ new output format
