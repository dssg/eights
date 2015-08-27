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
    return open_simple_csv_as_list(file_loc)

def open_csv(file_loc, delimiter=','):
    f = open_csv_as_structured_array(file_loc, delimiter)
    return set_structured_array_datetime_as_day(f, file_loc, delimiter)
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

def connect_sql(con_str):
    return SQLConnection(con_str)
    


#Plots of desrcptive statsitics
from ..communicate.communicate import plot_correlation_matrix
from ..communicate.communicate import plot_correlation_scatter_plot
from ..communicate.communicate import plot_kernel_density
from ..communicate.communicate import plot_on_map
from ..communicate.communicate import plot_on_timeline
from ..communicate.communicate import plot_box_plot



#this works but need the sa's to be good...
#data = [[(1,2),(1,2),(1,4)],[(1,2),(3,1)],[(1,2)]]
#
#def turn_list_of_list_of_items_to_SA_by_over_Lap(lol_items):
#    adict  = {}
#    feature_index = 0
#    bigram_index = []
#    l = [item for l_items in lol_items for item in l_items]
#    num_unique_items = len(set(l))
#    M = np.zeros(shape=(len(lol_items), num_unique_items))    
#    feature_index = 0
#    for row, l_items in enumerate(lol_items):
#        for item in l_items:
#            try:
#                M[row, adict[item]] += 1
#                adict[item] += 1 #
#            except KeyError:
#                M[row, feature_index] += 1
#                feature_index += 1
#                bigram_index.append(item)
#                adict[item] = 1
#    return M
#
#M = turn_list_of_list_of_items_to_SA_by_over_Lap(data)     
#

#def open_JSON():
#    """single line description
#    Parameters
#    ----------
#    temp : type
#       Description 
#    
#    Attributes
#    ----------
#    temp : type
#       Description 
#       
#    Returns
#    -------
#    temp : type
#       Description
#       
#    """
#    raise NotImplementedError    
#def open_PostgreSQL():
#    """single line description
#    Parameters
#    ----------
#    temp : type
#       Description 
#    
#    Attributes
#    ----------
#    temp : type
#       Description 
#       
#    Returns
#    -------
#    temp : type
#       Description
#       
#    """
#    raise NotImplementedError    
#def open_SQL():
#    """works with an sql database
#    Parameters
#    ----------
#    temp : type
#       Description 
#    
#    Returns
#    -------
#    temp : type
#       Description
#       
#    """
#    raise NotImplementedError
#def query_sql():
#    raise NotImplementedError
#def query_postgreSQL():
#    raise NotImplementedError
#
