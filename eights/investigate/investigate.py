import itertools as it
import numpy as np

import sklearn

from collections import Counter
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

from .investigate_helper import *
from ..communicate import *
from ..utils import is_sa

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
    return set_structured_array_datetime_as_day(f, file_loc)

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
    
def open_PostgreSQL():
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
from ..communicate.communicate import plot_correlation_matrix
from ..communicate.communicate import plot_correlation_scatter_plot
from ..communicate.communicate import plot_kernel_density
from ..communicate.communicate import plot_on_map
from ..communicate.communicate import plot_on_timeline
from ..communicate.communicate import plot_box_plot


#simple non-permabulated rfs
from ..operate.operate import simple_clf
from ..operate.operate import simple_clf_cv
