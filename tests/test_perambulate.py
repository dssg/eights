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
        subsets = {SubsetSweepTrainingSize: {'subset_size': 
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
        subsets = {SubsetSweepTrainingSize: {'subset_size': [20, 40, 60, 80, 100]}}
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
        subsets = {SubsetSweepTrainingSize: {'subset_size': [20, 40, 60, 80, 100]}}
        cvs = {StratifiedKFold: {}}
        exp = Experiment(M, y, clfs, subsets, cvs)
        for trial in exp.run():
            print trial, trial.average_score()
        print
        for trial in exp.slice_by_best_score(CLF_PARAMS):
            print trial, trial.average_score()

    def test_report(self):
#        iris = datasets.load_iris()
#        y = iris.target
#        M = iris.data
        M, y = utils_for_tests.generate_test_matrix(100, 5, 2)
        clfs = {RandomForestClassifier: {'n_estimators': [10, 100], 'max_depth': [1, 10]}, 
                SVC: {'kernel': ['linear', 'rbf'], 'probability': [True]}}        
        subsets = {SubsetSweepTrainingSize: {'subset_size': [20, 40, 60, 80, 100]}}
        cvs = {StratifiedKFold: {}}
        exp = Experiment(M, y, clfs, subsets, cvs)
        for trial, score in exp.roc_auc().iteritems():
            print trial
            print score

if __name__ == '__main__':
    unittest.main()
	

