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


def open_csv_list(file_loc):
    # Opens a csv as a list
    return open_simple_csv_as_list(file_loc)

def open_csv(file_loc, delimiter=','):
    # opens csv as a structured array
    return open_csv_as_structured_array(file_loc, delimiter)

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
    elif len(M.shape)==1: #intented to solve np.array([1,2,3]) data
        return [describe_column(M)] 
    else:
        #then its a list of np.arrays
        return [describe_column(M[:,x]) for x in range(M.shape[1])]



def convert_list_to_structured_array_wrap(L, col_names=None, dtype=None):
# What is this supposed to do?
    return convert_list_to_structured_array(L, col_names, dtype)

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

def connect_sql(con_str, allow_caching=False, cache_dir='.'):
    return SQLConnection(con_str, allow_caching, cache_dir)
    


#Plots of desrcptive statsitics
from ..communicate.communicate import plot_correlation_matrix
from ..communicate.communicate import plot_correlation_scatter_plot
from ..communicate.communicate import plot_kernel_density
from ..communicate.communicate import plot_on_map
from ..communicate.communicate import plot_on_timeline
from ..communicate.communicate import plot_box_plot



