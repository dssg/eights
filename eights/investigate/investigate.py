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


__describe_cols_metrics = [('Count', len),
                           ('Mean', np.mean),
                           ('Standard Dev', np.std),
                           ('Minimum', min),
                           ('Maximum', max)]

__describe_cols_fill = [np.nan] * len(__describe_cols_metrics)

def describe_cols(M):
    """takes a SA or list of Np.rayas and returns the summary statistcs
    Parameters
    ----------
    M import numpy as np
    : Structured Array or list of Numpy ND arays.
       Description 
       
    Returns
    -------
    temp : type
       Description
       
    """ 
    M = convert_to_sa(M)           
    descr_rows = []
    for col_name, col_type in M.dtype.descr:
        if 'f' in col_type or 'i' in col_type:
            col = M[col_name]
            row = [col_name] + [func(col) for _, func in 
                                __describe_cols_metrics]
        else:
            row = [col_name] + __describe_cols_fill
        descr_rows.append(row)
    col_names = ['Column Name'] + [col_name for col_name, _ in 
                                   __describe_cols_metrics]
    return convert_to_sa(descr_rows, col_names=col_names)


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
from ..communicate.communicate import plot_on_timeline
from ..communicate.communicate import plot_box_plot



