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



def cast_list_of_list_to_sa_wrap(L, col_names=None, dtype=None):
# What is this supposed to do?
    return cast_list_of_list_to_sa(L, col_names, dtype)

def crosstab(col1, col2):
    """
    Makes a crosstab of col1 and col2. This is represented as a
    structured array with the following properties:

    1. The first column is the value of col1 being crossed
    2. The name of every column except the first is the value of col2 being
       crossed
    3. To find the number of cooccurences of x from col1 and y in col2,
       find the row that has 'x' in col1 and the column named 'y'. The 
       corresponding cell is the number of cooccurrences of x and y
    """
    col1 = np.array(col1)
    col2 = np.array(col2)
    col1_unique = np.unique(col1)
    col2_unique = np.unique(col2)
    crosstab_rows = []
    for col1_val in col1_unique:
        loc_col1_val = np.where(col1==col1_val)[0]
        col2_vals = col2[loc_col1_val]
        cnt = Counter(col2_vals)
        counts = [cnt[col2_val] if cnt.has_key(col2_val) else 0 for col2_val 
                  in col2_unique]
        crosstab_rows.append(['{}'.format(col1_val)] + counts)
    col_names = ['col1_value'] + ['{}'.format(col2_val) for col2_val in 
                                  col2_unique]
    return convert_to_sa(crosstab_rows, col_names=col_names)

def connect_sql(con_str, allow_caching=False, cache_dir='.'):
    return SQLConnection(con_str, allow_caching, cache_dir)
    


#Plots of desrcptive statsitics
from ..communicate.communicate import plot_correlation_matrix
from ..communicate.communicate import plot_correlation_scatter_plot
from ..communicate.communicate import plot_kernel_density
from ..communicate.communicate import plot_on_map
from ..communicate.communicate import plot_on_timeline
from ..communicate.communicate import plot_box_plot



