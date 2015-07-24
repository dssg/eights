from operate_helper import *
import eights.investigate as inv
import eights.utils as utils

# operate is full of liiitle pipelines

def run_classifier(M, y, clf_class, hyperparameters):
    # returns fitted estimator
    raise NotImplementedError

def run_std_classifiers(M_train, M_test, y_train, y_test, report_file):
    raise NotImplementedError    

def load_and_run_rf_cv(file_loc, label_col=0):
    M = inv.open_csv(file_loc)
    labels_name = M.dtype.names[label_col]
    labels = M[labels_name]
    M = utils.remove_cols(M, labels_name)
    return inv.simple_CV_clf(M, labels)
    
