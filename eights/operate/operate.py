from operate_helper import *
import eights.investigate as inv
import eights.utils as utils

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import StratifiedKFold



# operate is full of liiitle pipelines

def run_classifier(M, labels, clfs):
    """function for running a single classifier
    Parameters
    ----------
    M : structured array
    Matrix  
    
    labels : np.ndarray
    the correct labels

    Returns
    -------
    exp : Experiment
    Description

    Example
    -------

    """
    exp = Experiment(
        M, 
        labels, 
        clfs=clfs)
    return exp

def run_std_classifiers(M, labels, clfs=None, cvs=None, report_file='report.pdf'):
    if clfs is None:
        clfs = {AdaBoostClassifier: {'n_estimators': [20,50,100]}, 
               RandomForestClassifier: {'n_estimators': [10,30,50],'max_depth': [None,4,7,15],'n_jobs':[1]}, 
               LogisticRegression:{'C': [1.0,2.0,0.5,0.25],'penalty': ['l1','l2']}, 
               DecisionTreeClassifier {'max_depth': [None,4,7,15,25]},
               SVC:{'kernel': ['linear','rbf']}
               DummyClassifier:{'strategy': ['stratified','most_frequent','uniform']}
              }        
    if cv == None:
        cv = {StratifiedKFold:{}}
    
    exp = Experiment(
        M, 
        labels, 
        clfs=clfs,
        cvs = cvs
        )
    
    raise NotImplementedError    

def load_and_run_rf_cv(file_loc, label_col=0):
    M = inv.open_csv(file_loc)
    labels_name = M.dtype.names[label_col]
    labels = M[labels_name]
    M = utils.remove_cols(M, labels_name)
    return inv.simple_CV_clf(M, labels)
    
