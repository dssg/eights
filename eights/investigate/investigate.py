from investigate_helper import *
import itertools as it
from ..communicate import *
from ..utils import is_sa
import numpy as np

import sklearn

from collections import Counter
import matplotlib.pyplot as plt


from sklearn import cross_validation
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

#open files 
def open_csv(file_loc):
    """single line description
    Parameters
    ----------
    temp : type
       Description 
    
    Attributes
    ----------
    temp : type
       Description 
       
    Returns
    -------
    temp : type
       Description
       
    """
    f = open_csv_as_structured_array(file_loc)
    return set_structured_array_datetime_as_day(f,file_loc)

def open_JSON():
    """single line description
    Parameters
    ----------
    temp : type
       Description 
    
    Attributes
    ----------
    temp : type
       Description 
       
    Returns
    -------
    temp : type
       Description
       
    """
    raise NotImplementedError

def open_SQL():
    """works with an sql database
    Parameters
    ----------
    temp : type
       Description 
    
    Returns
    -------
    temp : type
       Description
       
    """
    raise NotImplementedError

#descriptive statistics
def describe_cols(M):
    """takes a SA or list of Np.rayas and returns the summary statistcs
    Parameters
    ----------
    M : Structured Array or list of Numpy ND arays.
       Description 
       
    Returns
    -------
    temp : type
       Description
       
    """
    #remove the [] if only one?
    if is_sa(M):
        #then its a structured array
        return [describe_column(M[x]) for x in M.dtype.names]
    else:
        #then its a list of np.arrays
        return [describe_column(M[x]) for x in M]

def print_crosstab(L_1, L_2, verbose=True):
    """this prints a crosstab results
    Parameters
    ----------
    temp : type
       Description 
    
    Returns
    -------
    temp : type
       Description
    """
    #assume np.structured arrays?
    crosstab_dict = crosstab(L_1, L_2)
    if verbose:
        print_crosstab_dict(crosstab_dict)
    return crosstab_dict


#Plots of desrcptive statsitics
def plot_box_plot(col, verbose=True):
    """Makes a box plot for a feature
    comment
    
    Parameters
    ----------
    col : np.array
    
    Returns
    -------
    matplotlib.figure.Figure
    
    """
    raise NotImplementedError

from ..communicate.communicate import plot_correlation_matrix
from ..communicate.communicate import plot_correlation_scatter_plot
from ..communicate.communicate import plot_kernel_density
from ..communicate.communicate import plot_on_map
from ..communicate.communicate import plot_on_timeline

#simple non-permabulated rfs
def simple_CV(M, labels, clf, clf_params={},
             cv=cross_validation.KFold, cv_parms={}):
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
        clfs={clf: clf_params},
        cvs={cv: cv_parms})
    runs = exp.run()
    
    scores = [run.clf.score(M[run.test_indices], labels[run.test_indices]) 
                for run in runs]
    return scores

