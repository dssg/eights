import unittest
import eights.communicate as comm
from eights.communicate.communicate import feature_pairs_in_tree
from eights.communicate.communicate import feature_pairs_in_rf
from eights import utils
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from utils_for_tests import rerout_stdout
from utils_for_tests import path_of_data
from utils_for_tests import generate_correlated_test_matrix
from utils_for_tests import generate_test_matrix
import numpy as np

REPORT_PATH=path_of_data('test_communicate.pdf')
REFERENCE_REPORT_PATH=path_of_data('test_communicate_ref.pdf')

class TestCommunicate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = comm.Report(report_path=REPORT_PATH)

    @classmethod
    def tearDownClass(cls):
        report_path = cls.report.to_pdf()
        print 'Test Communicate visual regression tests:'
        print '-----------------------------------------'
        print 'graphical output available at: {}.'.format(report_path)
        print 'Reference available at: {}.'.format(REFERENCE_REPORT_PATH)

    def add_fig_to_report(self, fig, heading):
        self.report.add_heading(heading)
        self.report.add_fig(fig)

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
        fig = comm.plot_correlation_scatter_plot(sa, verbose=False)
        self.add_fig_to_report(fig, 'plot_correlation_scatter_plot')

    def test_plot_simple_histogram(self):
        np.random.seed(0)
        data = np.random.normal(size=(1000,))
        fig = comm.plot_simple_histogram(data, verbose=False)
        self.add_fig_to_report(fig, 'plot_simple_histogram')

    def test_plot_prec_recall(self):
        M, labels = generate_correlated_test_matrix(1000)
        M_train, M_test, labels_train, labels_test = train_test_split(
                M, 
                labels)
        clf = RandomForestClassifier(random_state=0)
        clf.fit(M_train, labels_train)
        score = clf.predict_proba(M_test)[:,-1]
        fig = comm.plot_prec_recall(labels_test, score, verbose=False)
        self.add_fig_to_report(fig, 'plot_prec_recall')

    def test_plot_roc(self):
        M, labels = generate_correlated_test_matrix(1000)
        M_train, M_test, labels_train, labels_test = train_test_split(
                M, 
                labels)
        clf = RandomForestClassifier(random_state=0)
        clf.fit(M_train, labels_train)
        score = clf.predict_proba(M_test)[:,-1]
        fig = comm.plot_roc(labels_test, score, verbose=False)
        self.add_fig_to_report(fig, 'plot_roc')

    def test_plot_box_plot(self):
        np.random.seed(0)
        data = np.random.normal(size=(1000,))
        fig = comm.plot_box_plot(data, col_name='box', verbose=False)
        self.add_fig_to_report(fig, 'plot_box_plot')

    def test_get_top_features(self):
        M, labels = generate_test_matrix(1000, 10)
        M = utils.cast_np_sa_to_nd(M)
        M_train, M_test, labels_train, labels_test = train_test_split(
                M, 
                labels)
        clf = RandomForestClassifier(random_state=0)
        clf.fit(M_train, labels_train)
        scores = clf.feature_importances_

    # TODO stopped at get_top_features

    def xtest_feature_pairs_in_tree(self):
        iris = datasets.load_iris()
        rf = RandomForestClassifier(random_state=0)
        rf.fit(iris.data, iris.target)
        dt = rf.estimators_[0]
        result = feature_pairs_in_tree(dt)
        ctrl = [[(2, 3)], [(2, 3), (0, 2)], [(0, 2), (1, 3)]]
        self.assertEqual(result, ctrl)

    def xtest_feature_pairs_in_rf(self):
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
