import unittest
import utils_for_tests
from sklearn.svm import SVC 

from sklearn import datasets

from eights.perambulate.perambulate import *

class TestPerambulate(unittest.TestCase):

    def test_run_experiment(self):
        iris = datasets.load_iris()
        y = iris.target
        M = iris.data
        clfs = {RandomForestClassifier: {}}
        subsets = {SubsetRandomRowsActualDistribution: {'subset_size': 
                                             [20, 40, 60, 80, 100]}}
        cvs = {StratifiedKFold: {}}
        exp = Experiment(M, y, clfs, subsets, cvs)
        for item in exp.average_score():
            print item

    def test_slice_on_dimension(self):
        iris = datasets.load_iris()
        y = iris.target
        M = iris.data
        clfs = {RandomForestClassifier: {'n_estimators': [10, 100], 'max_depth': [1, 10]}, 
                SVC: {'kernel': ['linear', 'rbf']}}        
        subsets = {SubsetRandomRowsActualDistribution: {'subset_size': [20, 40, 60, 80, 100]}}
        cvs = {StratifiedKFold: {}}
        exp = Experiment(M, y, clfs, subsets, cvs)
        for trial in exp.slice_on_dimension(CLF, RandomForestClassifier):
            print trial
        print
        for trial in exp.slice_on_dimension(SUBSET_PARAMS, {'subset_size': 60}):
            print trial

    def test_slice_by_best_score(self):
        iris = datasets.load_iris()
        y = iris.target
        M = iris.data
        clfs = {RandomForestClassifier: {'n_estimators': [10, 100], 'max_depth': [1, 10]}, 
                SVC: {'kernel': ['linear', 'rbf']}}        
        subsets = {SubsetRandomRowsActualDistribution: {'subset_size': [20, 40, 60, 80, 100]}}
        cvs = {StratifiedKFold: {}}
        exp = Experiment(M, y, clfs, subsets, cvs)
        for trial in exp.run():
            print trial, trial.average_score()
        print
        for trial in exp.slice_by_best_score(CLF_PARAMS):
            print trial, trial.average_score()

    def test_report_simple(self):
        M, y = utils_for_tests.generate_test_matrix(100, 5, 2)
        clfs = {RandomForestClassifier: {'n_estimators': [10, 100, 1000]}}
        cvs = {StratifiedKFold: {}}
        exp = Experiment(M, y, clfs=clfs, cvs=cvs)
        exp.make_report()

    def test_make_csv(self):
        #M, y = utils_for_tests.generate_test_matrix(1000, 5, 2)
        M, y = utils_for_tests.generate_correlated_test_matrix(10000)
        clfs = {RandomForestClassifier: {'n_estimators': [10, 100, 1000], 
                                         'max_depth': [5, 25]},
                SVC: {'kernel': ['linear', 'rbf'], 'probability': [True]}}        
        subsets = {SubsetSweepNumRows: {'num_rows': [[100, 200, 300]]}}
        cvs = {StratifiedKFold: {'n_folds': [2, 3]}}
        exp = Experiment(M, y, clfs=clfs, subsets=subsets, cvs=cvs)
        exp.make_csv()

    def test_subsetting(self):
        M, y = utils_for_tests.generate_test_matrix(1000, 5, 2)
        subsets = {SubsetRandomRowsEvenDistribution: {'subset_size': [20]},
                   SubsetRandomRowsActualDistribution: {'subset_size': [20]},
                   SubsetSweepNumRows: {'num_rows': [[10, 20, 30]]},
                   SubsetSweepVaryStratification: {'proportions_positive': [[.5, .75, .9]],
                                                   'subset_size': [10]}}
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
                      (8, 1997)], dtype=[('id', int), ('year', int)])

    def test_report_complex(self):
        M, y = utils_for_tests.generate_test_matrix(100, 5, 2)
        clfs = {RandomForestClassifier: {'n_estimators': [10, 100], 
                                         'max_depth': [1, 10]}, 
                SVC: {'kernel': ['linear', 'rbf'], 'probability': [True]}}        
        subsets = {SubsetRandomRowsActualDistribution: {'subset_size': 
                                             [20, 40, 60, 80, 100]}}
        cvs = {StratifiedKFold: {}}
        exp = Experiment(M, y, clfs, subsets, cvs)
        exp.make_report(dimension=CLF)

if __name__ == '__main__':
    unittest.main()
	

