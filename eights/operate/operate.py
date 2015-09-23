
from sklearn.ensemble import (AdaBoostClassifier, 
                              RandomForestClassifier)
from sklearn.linear_model import (LogisticRegression, 
                                  RidgeClassifier, 
                                  SGDClassifier, 
                                  Perceptron, 
                                  PassiveAggressiveClassifier)
from sklearn.cross_validation import (StratifiedKFold, 
                                      KFold)
from sklearn.naive_bayes import (BernoulliNB, 
                                 MultinomialNB)
from sklearn.neighbors import(KNeighborsClassifier, 
                              NearestCentroid)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

from ..utils import remove_cols
from ..perambulate import Experiment



def run_std_classifiers(M, labels, clfs=None, cvs=None, report_file='report.pdf'):
    """ standard first past sanity check on the classifiers
    Parameters
    ----------
    M : type
    Description 


    Returns
    -------
    ? : type
    Description

    Example
    -------

    """
    if clfs is None:
        clfs = [{'clf': AdaBoostClassifier, 'n_estimators': [20,50,100]}, 
                {'clf': RandomForestClassifier, 
                 'n_estimators': [10,30,50],
                 'max_features': ['sqrt','log2'],
                 'max_depth': [None,4,7,15],
                 'n_jobs':[1]}, 
                {'clf': LogisticRegression, 
                 'C': [1.0,2.0,0.5,0.25],
                 'penalty': ['l1','l2']}, 
                {'clf': DecisionTreeClassifier, 
                 'max_depth': [None,4,7,15,25]},
               # {'clf': SVC, 'kernel': ['linear','rbf'], 
                # 'probability': [True]},
                {'clf': DummyClassifier, 
                 'strategy': ['stratified','most_frequent','uniform']}
             ]
    if cvs == None:
        cvs = [{'cv': StratifiedKFold}]
    exp = Experiment(
        M, 
        labels, 
        clfs=clfs,
        cvs=cvs
        )
    exp.make_report(report_file)
    return exp


def rg_classifiers(M, labels, report_file='report.pdf'):
    clfs= [{'clf': RandomForestClassifier,
          'n_estimators': [1,10,100,1000,10000], 
          'max_depth': [1,5,10,20,50,100], 
          'max_features': ['sqrt','log2'],
          'min_samples_split': [2,5,10],
          'n_jobs': [1]},
        {'clf': LogisticRegression,
         'penalty': ['l1','l2'], 
         'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
        {'clf': SGDClassifier,
         'loss':['hinge','log','perceptron'], 
         'penalty':['l2','l1','elasticnet']},
       {'clf': ExtraTreesClassifier,
        'n_estimators': [1,10,100,1000,10000], 
        'criterion' : ['gini', 'entropy'],
        'max_depth': [1,5,10,20,50,100], 
        'max_features': ['sqrt','log2'],
        'min_samples_split': [2,5,10],
        'n_jobs': [1]},
        {'clf': AdaBoostClassifier,
         'algorithm' :['SAMME', 'SAMME.R'], 
         'n_estimators': [1,10,100,1000,10000],
         'base_estimator': [DecisionTreeClassifier(max_depth=1)]},
        {'clf': GradientBoostingClassifier,
         'n_estimators': [1,10,100,1000,10000], 
         'learning_rate' : [0.001,0.01,0.05,0.1,0.5],
         'subsample' : [0.1,0.5,1.0], 
         'max_depth': [1,3,5,10,20,50,100]},
        {'clf': GaussianNB },
        {'clf': DecisionTreeClassifier,
         'criterion': ['gini', 'entropy'],
         'max_depth': [1,5,10,20,50,100], 
         'max_features': ['sqrt','log2'],
         'min_samples_split': [2,5,10]},
        {'clf':SVC,
         'C': [0.00001,0.0001,0.001,0.01,0.1,1,10],
         'kernel': ['linear'],
         'probability': [True]},
        {'clf': KNeighborsClassifier,
         'n_neighbors':[1,5,10,25,50,100],
         'weights': ['uniform','distance'],
         'algorithm':['auto','ball_tree','kd_tree']}]      
    cvs = [{'cv': StratifiedKFold}]
    exp = Experiment(
        M, 
        labels, 
        clfs=clfs,
        cvs = cvs
        )
    exp.make_report(report_file)
    return exp


    
def run_alt_classifiers(M, labels, clfs=None, cvs=None, report_file='report.pdf'):
    if clfs is None:
        clfs = [{'clf': RidgeClassifier, 'tol':[1e-2], 'solver':['lsqr']},
                {'clf': SGDClassifier, 'alpha':[.0001], 'n_iter':[50],'penalty':['l1', 'l2', 'elasticnet']},
                {'clf': Perceptron, 'n_iter':[50]},
                {'clf': PassiveAggressiveClassifier, 'n_iter':[50]},
                {'clf': BernoulliNB, 'alpha':[.01]},
                {'clf': MultinomialNB, 'alpha':[.01]},
                {'clf': KNeighborsClassifier, 'n_neighbors':[10]},
                {'clf': NearestCentroid}]
    if cvs == None:
        cvs = [{'cv': StratifiedKFold}]
    exp = Experiment(
        M, 
        labels, 
        clfs=clfs,
        cvs = cvs
        )
    exp.make_report(report_file)
    return exp    



def _rg_classifiers(M, labels, report_file='report.pdf'):
    clfs= [{'clf': RandomForestClassifier,
          'n_estimators': [1], 
          'max_depth': [1], 
          'max_features': ['sqrt'],
          'min_samples_split': [2],
          'n_jobs': [1]},
        {'clf': LogisticRegression,
         'penalty': ['l1'], 
         'C': [0.00001]},
        {'clf': SGDClassifier,
         'loss':['log'], # hinge doesn't have predict_proba
         'penalty':['l2']},
        {'clf': ExtraTreesClassifier,
         'n_estimators': [1], 
         'criterion' : ['gini'],
         'max_depth': [1], 
         'max_features': ['sqrt'],
         'min_samples_split': [2],
         'n_jobs': [1]},
        {'clf': AdaBoostClassifier,
         'algorithm' :['SAMME'],
         'n_estimators': [1],
         'base_estimator': [DecisionTreeClassifier(max_depth=1)]},
        {'clf': GradientBoostingClassifier,
         'n_estimators': [1],
         'learning_rate' : [0.001],
         'subsample' : [0.1],
         'max_depth': [1]},
        {'clf': GaussianNB },
        {'clf': DecisionTreeClassifier,
         'criterion': ['gini'],
         'max_depth': [1],
         'max_features': ['sqrt'],
         'min_samples_split': [2]},
        {'clf':SVC,
         'C': [0.00001],
         'kernel': ['linear'],
         'probability': [True]},
        {'clf': KNeighborsClassifier,
         'n_neighbors':[1],
         'weights': ['uniform'],
         'algorithm':['auto']}]     
    cvs = [{'cv': StratifiedKFold}]
    exp = Experiment(
        M, 
        labels, 
        clfs=clfs,
        cvs=cvs
        )
    exp.make_report(report_file)
    return exp       
