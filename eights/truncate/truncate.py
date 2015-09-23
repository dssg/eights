import numpy as np
from collections import Counter
from truncate_helper import *
from ..utils import remove_cols

def remove_col_where(M, arguments):
    to_remove = np.ones(len(M.dtype), dtype=bool)
    for arg_set in arguments:
        lambd, vals = (arg_set['func'], arg_set['vals'])
        to_remove = np.logical_and(to_remove, lambd(M,  vals))
    remove_col_names = [col_name for col_name,included in zip(M.dtype.names, to_remove) if included] 
    return remove_cols(M, remove_col_names)

def all_equal_to(M, boundary):
    return [np.all(M[col_name] == boundary) for col_name in M.dtype.names]

def all_same_value(M, boundary=None):
    return [np.all(M[col_name]==M[col_name][0]) for col_name in M.dtype.names]

def fewer_then_n_nonzero_in_col(M, boundary):
    return [len(np.where(M[col_name]!=0)[0])<2 for col_name in M.dtype.names]

def remove_rows_where(M, lamd, col_name, vals):
    to_remove = lamd(M, col_name, vals)
    to_keep = np.logical_not(to_remove)
    return M[to_keep]


    
from ..generate.generate import val_eq
from ..generate.generate import val_lt
from ..generate.generate import val_gt
from ..generate.generate import val_between
from ..generate.generate import is_outlier



   

#def fewer_then_n_nonzero_in_col(M, boundary):
#    col_names = M.dtype.names
#    num_rows =M.shape[0]
#    l = [sum(M[n]==0) for n in col_names]
#    remove_these_columns = np.where(np.array(l)>=(num_rows-boundary))[0]
#    names = [col_names[i] for i in remove_these_columns]
#    return remove_cols(M, names)


 
    

