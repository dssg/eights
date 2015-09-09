import unittest
import eights.communicate as comm
from eights.communicate.communicate import feature_pairs_in_tree
from eights.communicate.communicate import feature_pairs_in_rf
from eights import utils
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from utils_for_tests import rerout_stdout
import numpy as np

class TestCommunicate(unittest.TestCase):
    def test_print_matrix_row_col(self):
        M = [(1, 2, 3), (4, 5, 6), (7, 8, 'STRING')]
        ctrl = """
                            f0             f1             f2
              0              1              2              3
              1              4              5              6
              2              7              8         STRING
        """.strip()
        with rerout_stdout() as get_stdout:
            comm.print_matrix_row_col(M)
            self.assertEqual(get_stdout().strip(), ctrl)
        M = np.array([(1000, 'Bill'), (2000, 'Sam'), (3000, 'James')],
                     dtype=[('number', float), ('name', 'S5')])
        row_labels = [name[0] for name in M['name']]
        ctrl = """
                        number           name
              B         1000.0           Bill
              S         2000.0            Sam
              J         3000.0          James
        """.strip()
        with rerout_stdout() as get_stdout:
            comm.print_matrix_row_col(M, row_labels=row_labels)
            self.assertEqual(get_stdout().strip(), ctrl)

    def test_plot_correlation_scatter_plot(self):
        col1 = range(10)
        col2 = [cell * 3 + 1 for cell in col1]
        col3 = [1, 5, 8, 4, 1, 8, 5, 9, 0, 1]
        sa = utils.convert_to_sa(
                zip(col1, col2, col3), 
                col_names=['base', 'linear_trans', 'no_correlation'])
        comm.plot_correlation_scatter_plot(sa)

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
        result = feature_pairs_in_rf(rf, [1, 0.5], verbose=False)
        result = feature_pairs_in_rf(rf, verbose=True, n=5)
        
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

if __name__ == '__main__':
    unittest.main()
