from itertools import islice
import numpy as np

def slidding_window(l,w,tst_w):
    ret = []
    for idx,_ in enumerate(l):
        if idx+w+tst_w > len(l): 
            break
        train =[l[idx+x] for x in range(w)]
        test = [l[idx+w+x] for x in range(tst_w)]
        ret.append((train,test))
    return ret
    
def expanding_window(l,w,tst_w):
    ret = []
    
    for idx, i in enumerate(l):
        if idx+w+tst_w > len(l): break
        
        train =[l[x] for x in range(idx+w)]
        test = []
        for x in range(tst_w):
            test.append(l[idx+w+x])
        ret.append((train,test))
    return ret

def sweep_exclude_columns(M, cols_to_exclude=None):
    """
    
    Analyze feature importance when each of a specified set of columns are
    excluded. 
    
    Parameters
    ----------
    M : Numpy structured array
    cols_to_exclude : list of str or None
         List of names of columns to exclude one at a time. If None, tries
         all columns
         
    Returns
    -------
    Numpy Structured array
        First col
            Excluded col name
        Second col
            Accuracy score
        Third col
            Feature importances
    """
    # not providing classifier because it's always rf
    # not providing cv because it's always Kfold
    raise NotImplementedError
    
def sweep_vary_training_set_size(M, size):
    raise NotImplementedError
    
def sweep_vary_stratification(M, positive_percents):
    raise NotImplementedError
    
def run_std_classifiers(M_train, M_test, y_train, y_test, report_file):
    raise NotImplementedError    
    
def stratified_cv(y, positive_percents):
    # train on different, artificial stratifications; test on stratification
    # reflecting perct in actual data set
    raise NotImplementedError
