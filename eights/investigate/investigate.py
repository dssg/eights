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
import sklearn.datasets

from eights.investigate import (cast_np_nd_to_sa, describe_cols,)
from eights.communicate import (plot_correlation_scatter_plot,
                               plot_correlation_matrix, 
                               plot_kernel_density,
                               plot_box_plot)

#import numpy array
M = sklearn.datasets.load_iris().data
labels = sklearn.datasets.load_iris().target

M = cast_np_nd_to_sa(M)


#M is multi class, we want to remove those rows.
keep_index = np.where(labels!=2)

labels = labels[keep_index]
M = M[keep_index]




if False:
    for x in describe_cols(M):
        print x

if False:
   plot_correlation_scatter_plot(M) 
   plot_correlation_matrix(M)
   plot_kernel_density(M['f0']) #no designation of col name
   plot_box_plot(M['f0']) #no designation of col name


if False:
    from eights.generate import val_between, where_all_are_true, append_cols  #val_btwn, where
    #generate a composite rule
    M = where_all_are_true(M, 
                          [{'func': val_between, 
                            'col_name': 'f0', 
                            'vals': (3.5, 5.0)},
                           {'func': val_between, 
                            'col_name': 'f1', 
                            'vals': (2.7, 3.1)}
                           ], 
                           'a new col_name')

    #new eval function
    def rounds_to_val(M, col_name, boundary):
        return (np.round(M[col_name]) == boundary)
    
    M = where_all_are_true(M,
                          [{'func': rounds_to_val, 
                            'col_name': 'f0', 
                            'vals': 5}],
                            'new_col')
    
    from  eights.truncate import (fewer_then_n_nonzero_in_col, 
                                 remove_rows_where,
                                 remove_cols,
                                 val_eq)
    #remove Useless row
    M = fewer_then_n_nonzero_in_col(M,1)
    M = append_cols(M, labels, 'labels')
    M = remove_rows_where(M, val_eq, 'labels', 2)
    labels=M['labels']
    M = remove_cols(M, 'labels')


from eights.operate import run_std_classifiers, run_alt_classifiers #run_alt_classifiers not working yet
exp = run_std_classifiers(M,labels)
exp.make_csv()
import pdb; pdb.set_trace()


####################Communicate#######################



#Pretend .1 is wrong so set all values of .1 in M[3] as .2
# make a new column where its a test if col,val, (3,.2), (2,1.4) is true.


import pdb; pdb.set_trace()

#from decontaminate import remove_null, remove_999, case_fix, truncate
#from generate import donut
#from aggregate import append_on_right, append_on_bottom
#from truncate import remove
#from operate import run_list, fiveFunctions
#from communicate import graph_all, results_invtestiage

#investiage
#M_orginal = csv_open(file_loc, file_descpiption)  # this is our original files
#results = eights.investigate.describe_all(M_orginal)
#results_invtestiage(results)

#decontaminate
#aggregate
#generate
#M = np.array([]) #this is the master Matrix we train on.
#labels = np.array([]) # this is tells us

#truncate
#models = [] #list of functions

#operate

#communicate


#func_list = [sklearn.randomforest,sklearn.gaussian, ]


#If main:
#run on single csv
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
from ..communicate.communicate import plot_on_map
from ..communicate.communicate import plot_on_timeline
from ..communicate.communicate import plot_box_plot



