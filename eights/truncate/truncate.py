import numpy as np
from collections import Counter

from truncate_helper import *

def col_has_all_same_val(col):
    return np.all(col==col[0])

def col_has_one_unique_val(col):
    d = Counter(col)
    if len(d) == 2: #ignores case for -999, null
        return (1 in d.values())
    return False

def col_has_few_unique_values(col):
    d = Counter(col)
    vals = sort(d.values())
    return ( sum(vals[:-1]) < threshold) 

def remove_cols(M):
    raise NotImplementedError

def remove_rows_if_true(M, lamd):
    raise NotImplementedError

def row_val_eq(M, col_name, boundary):
    return M[col_name] == boundary

def row_val_lt(M, col_name, boundary):
    return M[col_name] < boundary

def row_val_gt(M, col_name, boundary):
    return M[col_name] > boundary

def row_val_between(M, col_name, boundary):
    return np.and(boundary[0] <= M[col_name], M[col_name] <= boundary[1])

def row_not_within_region(M, col_name, boundar):
    raise NotImplementedError:

    
    

