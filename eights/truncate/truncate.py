import numpy as np
from collections import Counter
from truncate_helper import *
from ..utils import remove_cols

def fewer_then_n_nonzero_in_col(M, boundary):
    col_names = M.dtype.names
    num_rows =M.shape[0]
    l = [sum(M[n]==0) for n in col_names]
    remove_these_columns = np.where(np.array(l)>=(num_rows-boundary))
    names = [col_names[i] for i in remove_these_columns]
    return remove_cols(M, names)

def all_equal_values_in_col(M, col_name, boundary):
    col_names = M.dtype.names
    remove_these_columns = []
    for n in col_names:
        if sum(M[n]==M[n][0]) == M.shape[0]:
            remove_these_columns.append(n)
    return remove_cols(M, remove_these_columns)
    
def remove_these_cols(M, col_name, boundary):
    return remove_cols(M, col_name)

def remove_rows_where(M, lamd, col_name, vals):
    to_remove = lamd(M, col_name, vals)
    to_keep = np.logical_not(to_remove)
    return M[to_keep]
    
from ..generate.generate import val_eq
from ..generate.generate import val_lt
from ..generate.generate import val_gt
from ..generate.generate import val_between

#def row_not_within_region(M, col_name, boundar):
#    raise NotImplementedError

def remove_col_where(M, lamd, col_name, vals):
    #std to this later
    raise NotImplementedError    
    

