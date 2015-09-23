import numpy as np
from sklearn import cross_validation
from ..utils import append_cols, distance
from uuid import uuid4


def where_all_are_true(M, arguments, generated_name=None):
    if generated_name is None:
        generated_name = str(uuid4())
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



def generate_bin(col, num_bins):
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
    [0 3 0 1 2 1 2]

    """

    minimum = float(min(col))
    maximum = float(max(col))
    distance = float(maximum - minimum)
    return [int((x - minimum) / distance * num_bins) for x in col]

def normalize(col, mean=None, stddev=None, return_fit=False):
    """
    
    Generate a normalized column.
    
    Normalize both mean and std dev.
    
    Parameters
    ----------
    col : np.array
    mean : float or None
        Mean to use for fit. If none, will use 0
    stddev : float or None
    return_fit : boolean
        If True, returns tuple of fitted col, mean, and standard dev of fit.
        If False, only returns fitted col
    Returns
    -------
    np.array or (np.array, float, float)
    
    """
    # see infonavit for applying to different set than we fit on
    # https://github.com/dssg/infonavit-public/blob/master/pipeline_src/preprocessing.py#L99
    # Logic is from sklearn StandardScaler, but I didn't use sklearn because
    # I want to pass in mean and stddev rather than a fitted StandardScaler
    # https://github.com/scikit-learn/scikit-learn/blob/a95203b/sklearn/preprocessing/data.py#L276
    if mean is None:
        mean = np.mean(col)
    if stddev is None:
        stddev = np.std(col)
    res = (col - mean) / stddev
    if return_fit:
        return (res, mean, stddev)
    else:
        return res

def distance_from_point(lat_origin, lng_origin, lat_col, lng_col):
    """ Generates a column of how far each record is from the origin"""
    return distance(lat_origin, lng_origin, lat_col, lng_col)

@np.vectorize
def combine_sum(*args):
    return sum(args)

@np.vectorize
def combine_mean(*args):
    return np.mean(args)

def combine_cols(M, lambd, col_names, generated_name):
    new_col = lambd(*[M[name] for name in col_names])
    return append_cols(M, new_col, generated_name)

