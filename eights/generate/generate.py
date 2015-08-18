import numpy as np
from sklearn import cross_validation
import generate_helper as gh
from ..utils import append_cols
from uuid import uuid4


def where_all_are_true(M, arguments, generated_name=None):
    to_select = np.ones(M.size, dtype=bool)
    for arg_set in arguments:
        lambd, col_name, vals = (arg_set['func'], arg_set['col_name'],
                                    arg_set['vals'])
        to_select = np.logical_and(to_select, lambd(M, col_name, vals))
    return append_cols(M, to_select, generated_name)

# where_all_are_true(
#    M,
#    [{'func': val_eq, 'col_name': 'f1', 'vals': 4},
#     {'func': val_between, 'col_name': 'f7', 'vals': (1.2, 2.5)}]      

def is_outlier(M, col_name, boundary):
    std = np.std(M[col_name])
    mean = np.mean(M[col_name])
    return (np.logical_or( (mean-3*std)>M[col_name], (mean+3*std)<M[col_name]) )
    

def val_eq(M, col_name, boundary):
    return M[col_name] == boundary

def val_lt(M, col_name, boundary):
    return M[col_name] < boundary

def val_lt_TIME_EDITION(M, col_name, boundary):
    return M[col_name] < boundary

def val_gt(M, col_name, boundary):
    return M[col_name] > boundary

def val_between(M, col_name, boundary):
    return np.logical_and(boundary[0] <= M[col_name], M[col_name] <= boundary[1])



#def sweep_where_all_are_true(args, M):
#   changing_args = [arg[2] for arg in args]
#   runs = product(*changing_args)
#   for r in runs:
#      fixed_lambdas = [lambda col: lam(col, r[idx], args[idx][3]) for idx, lam in enumerate(lambdas)]
#      yield select_one(fixed_lambdas, args[idx][1], M)
#      
      

#
#M = np.array([ ('home', 40.761036, -73.977374),
#                  ('work', 45.5660930, -73.92599),
#                  ('fun', 40.702646, -74.013799)],
#                  dtype = [('name', 'S4'), ('lng', float), ('lat', float)]
#                )    
#                
#target = GPS(40.748784, -73.985429)
#threshold = .001
#
#lat_col_name = 'lat'
#lng_col_name = 'lng'
#
#M_id =  select_by_dist_from(M, target, threshold, lat_col_name, lng_col_name)
#
#
#import pdb; pdb.set_trace()
#
#
#
#
#def generate():
#    return
#

def generate_bin(col, number_of_bins):
    """Generates a column of categories, where each category is a bin.

    Parameters
    ----------
    col : np.array
    
    Returns
    -------
    np.array
    
    Examples
    --------
    >>> M = np.array([0.1, 3.0, 0.0, 1.2, 2.5, 1.7, 2])
    >>> generate_bin(M, 3)
    [0 2 0 1 2 1 2]

    """

    raise NotImplementedError
    
def normalize(col):
    """
    
    Generate a normalized column.
    
    Normalize both mean and std dev.
    
    Parameters
    ----------
    col : np.array
    
    Returns
    -------
    np.array
    
    """
    raise NotImplementedError    

def distance_from_point(lat_origin, lng_origin, lat_col, lng_col):
    """ Generates a column of how far each record is from the origin"""
    raise NotImplementedError    

@np.vectorize
def combine_sum(*args):
    return sum(args)

@np.vectorize
def combine_mean(*args):
    return np.mean(args)

def combine_cols(M, lambd, col_names, generated_name):
    new_col = lambd(*[M[name] for name in col_names])
    return append_cols(M, new_col, generated_name)

