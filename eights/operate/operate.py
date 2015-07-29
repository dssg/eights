
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import StratifiedKFold, KFold

from ..utils import remove_cols
from ..investigate.investigate import open_csv
from .operate_helper import *

def simple_clf(M, labels, clfs):
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

#this should be clfs
def simple_clf_cv(M, labels, clfs={RandomForestClassifier:{}},
                  cvs={KFold:{}}):
    """This is simple execution a clf in our module.  
    Parameters
    ----------
    M : Structured array
       The matrix you wish to use for training and testing 
    labels : a one dimenional nd array
       This these are the labels that are assigned to the rows in the matrix M.
    clf : Sklearn Class object
        This is the type of algorithim you would use. 
    clf_params : a dictionary of parameters to assign to your clf
        The appropriate paramterts to asign to the clf, empty dict if none.
    cv : sklearn cv 
        kfold if default
    cv_parms : dict of paramters to apply to the cv
        empty if default
           
    Returns
    -------
    temp : list
       the list of trained models

    Examples
    --------
    ...
    """
    exp = Experiment(
        M, 
        labels, 
        clfs=clfs,
        cvs=cvs)
    return exp

def run_std_classifiers(M, labels, clfs=None, cvs=None, report_file='report.pdf'):
    """ standard first past sanatity check on the classifiers
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
        clfs = {AdaBoostClassifier: {'n_estimators': [20,50,100]}, 
               RandomForestClassifier: {'n_estimators': [10,30,50],'max_depth': [None,4,7,15],'n_jobs':[1]}, 
               LogisticRegression:{'C': [1.0,2.0,0.5,0.25],'penalty': ['l1','l2']}, 
               DecisionTreeClassifier: {'max_depth': [None,4,7,15,25]},
               SVC:{'kernel': ['linear','rbf']},
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
    exp.run_report(report_file)
    return exp

def load_csv_simple_rf_cv(csv_loc, label_col=0):
    """ load from csv, run random forest crossvalidated.
    Parameters
    ----------
    : type
    Description 

    Returns
    -------
    : type
    Description

    Example
    -------
    """
    M = inv.open_csv(csv_loc)
    labels_name = M.dtype.names[label_col]
    labels = M[labels_name]
    M = utils.remove_cols(M, labels_name)
    return simple_clf_cv(M, labels)
    
