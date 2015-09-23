import unittest
import cPickle
import os

import numpy as np

from sklearn.svm import SVC 
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold

import eights.perambulate as per
import eights.communicate as comm

import utils_for_tests as uft

REPORT_PATH = uft.path_of_data('test_perambulate.pdf')
SUBREPORT_PATH = uft.path_of_data('test_perambulate_sub.pdf')
REFERENCE_REPORT_PATH = uft.path_of_data('test_perambulate_ref.pdf')
REFERENCE_PKL_PATH = uft.path_of_data('test_perambulate')

class TestPerambulate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.report = comm.Report(report_path=REPORT_PATH)

    @classmethod
    def tearDownClass(cls):
        report_path = cls.report.to_pdf(verbose=False)
        uft.print_in_box(
                'Test Perambulate visual regression tests',
                ['graphical output available at:',
                 report_path,
                 'Reference available at:',
                 REFERENCE_REPORT_PATH])

    def __pkl_store(self, obj, key):
        with open(os.path.join(REFERENCE_PKL_PATH, key + '.pkl'), 'w') as pkl:
            cPickle.dump(obj, pkl)

    def __get_ref_pkl(self, key):
        with open(os.path.join(REFERENCE_PKL_PATH, key + '.pkl')) as pkl:
            return cPickle.load(pkl)

    def __compare_to_ref_pkl(self, result, key):
        ref = self.__get_ref_pkl(key)
        self.assertEqual(ref, result) 
        
    def test_run_experiment(self):
        iris = datasets.load_iris()
        y = iris.target
        M = iris.data
        clfs = [{'clf': RandomForestClassifier, 'random_state': [0]}]
        subsets = [{'subset': per.SubsetRandomRowsActualDistribution, 
                    'subset_size': [20, 40, 60, 80, 100],
                    'random_state': [0]}]
        cvs = [{'cv': StratifiedKFold}]
        exp = per.Experiment(M, y, clfs, subsets, cvs)
        result = {str(key) : val for key, val in 
                  exp.average_score().iteritems()}
        self.__compare_to_ref_pkl(result, 'run_experiment')

    def test_slice_on_dimension(self):
        iris = datasets.load_iris()
        y = iris.target
        M = iris.data
        clfs = [{'clf': RandomForestClassifier, 
                 'n_estimators': [10, 100], 
                 'max_depth': [1, 10],
                 'random_state': [0]}, 
                 {'clf': SVC, 'kernel': ['linear', 'rbf']}]        
        subsets = [{'subset': per.SubsetRandomRowsActualDistribution, 
                    'subset_size': [20, 40, 60, 80, 100],
                    'random_state': [0]}]
        cvs = [{'cv': StratifiedKFold}]
        exp = per.Experiment(M, y, clfs, subsets, cvs)
        result = [str(trial) for trial in exp.slice_on_dimension(
                per.CLF, 
                RandomForestClassifier).trials]
        self.__compare_to_ref_pkl(result, 'slice_on_dimension_clf')
        result = [str(trial) for trial  in exp.slice_on_dimension(
                per.SUBSET_PARAMS, 
                {'subset_size': 60}).trials]
        self.__compare_to_ref_pkl(result, 'slice_on_dimension_subset_params')

    def test_slice_by_best_score(self):
        iris = datasets.load_iris()
        y = iris.target
        M = iris.data
        clfs = [{'clf': RandomForestClassifier, 
                 'n_estimators': [10, 100], 
                 'max_depth': [1, 10],
                 'random_state': [0]}, 
                 {'clf': SVC, 'kernel': ['linear', 'rbf']}]        
        subsets = [{'subset': per.SubsetRandomRowsActualDistribution, 
                    'subset_size': [20, 40, 60, 80, 100],
                    'random_state': [0]}]
        cvs = [{'cv': StratifiedKFold}]
        exp = per.Experiment(M, y, clfs, subsets, cvs)
        exp.run()
        result = {str(trial): trial.average_score() for trial in 
                  exp.slice_by_best_score(per.CLF_PARAMS).trials}
        #self.__pkl_store(result, 'slice_by_best_score')
        self.__compare_to_ref_pkl(result, 'slice_by_best_score')

    def test_report_simple(self):
        M, y = uft.generate_test_matrix(100, 5, 2)
        clfs = [{'clf': RandomForestClassifier, 
                 'n_estimators': [10, 100, 1000]}]
        cvs = [{'cv': StratifiedKFold}]
        exp = Experiment(M, y, clfs=clfs, cvs=cvs)
        exp.make_report()

    def test_make_csv(self):
        M, y = uft.generate_test_matrix(1000, 15, 2)
        #M, y = uft.generate_correlated_test_matrix(10000)
        clfs = [{'clf': RandomForestClassifier, 
                 'n_estimators': [10, 100, 1000], 
                 'max_depth': [5, 25]},
                {'clf': SVC, 
                 'kernel': ['linear', 'rbf'], 
                 'probability': [True]}]        
        subsets = [{'subset': SubsetSweepNumRows, 
                    'num_rows': [[100, 200, 300]]}]
        cvs = [{'cv': StratifiedKFold, 
                'n_folds': [2, 3]}]
        exp = Experiment(M, y, clfs=clfs, subsets=subsets, cvs=cvs)
        exp.make_csv()

    def test_subsetting(self):
        M, y = uft.generate_test_matrix(1000, 5, 2)
        subsets = [{'subset': SubsetRandomRowsEvenDistribution, 
                    'subset_size': [20]},
                   {'subset': SubsetRandomRowsActualDistribution, 
                    'subset_size': [20]},
                   {'subset': SubsetSweepNumRows, 
                    'num_rows': [[10, 20, 30]]},
                   {'subset': SubsetSweepVaryStratification, 
                    'proportions_positive': [[.5, .75, .9]],
                    'subset_size': [10]}]
        exp = Experiment(M, y, subsets=subsets)
        exp.run()
        for trial in exp.trials:
            print trial
            for run in trial.runs:
                print run

    def test_sliding_windows(self):
        M = np.array([(0, 2003),
                      (1, 1997),
                      (2, 1998),
                      (3, 2003),
                      (4, 2002),
                      (5, 2000),
                      (6, 2000),
                      (7, 2001),
                      (8, 1997),
                      (9, 2005), 
                      (10, 2005)], dtype=[('id', int), ('year', int)])
        y = np.array([True, False, True, False, True, False, True, False,
                      True, False])
        cvs = [{'cv': SlidingWindowIdx, 
                'train_start': [0], 
                'train_window_size': [2], 
                'test_start': [2], 
                'test_window_size': [2],
                'inc_value': [2]},
                {'cv': SlidingWindowValue, 
                 'train_start': [1997], 
                 'train_window_size': [2],
                 'test_start': [1999], 
                 'test_window_size': [2],
                 'inc_value': [2], 
                 'col_name': ['year']}]
        exp = Experiment(M, y, cvs=cvs)
        exp.make_csv()

    def test_report_complex(self):
        M, y = uft.generate_test_matrix(100, 5, 2)
        clfs = [{'clf': RandomForestClassifier, 
                 'n_estimators': [10, 100], 
                 'max_depth': [1, 10]}, 
                 {'clf': SVC, 
                  'kernel': ['linear', 'rbf'], 
                  'probability': [True]}]        
        subsets = [{'subset': SubsetRandomRowsActualDistribution, 
                    'subset_size': [20, 40, 60, 80, 100]}]
        cvs = [{'cv': StratifiedKFold}]
        exp = Experiment(M, y, clfs, subsets, cvs)
        exp.make_report(dimension=CLF)

if __name__ == '__main__':
    unittest.main()
	

