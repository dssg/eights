import numpy as np
from collections import Counter

from truncate_helper import *
from ..utils import remove_cols

def remove_rows_where(M, lamd, col_name, vals):
    raise NotImplementedError
    
def remove_col_where(M, lamd, col_name, vals):
    raise NotImplementedError
  
def col_has_fewer_then_n_nonzero(M, col_name, boundary):
    col_names = M.dtype.names
    l = [sum(M[n]==0) for n in col_names]
    remove_these_columns = np.where(np.array(l)<=boundary)
    raise NotImplementedError

def col_has_all_equal_values(M, col_name, boundary):
    col_names = M.dtype.names
    remove_these_columns = []
    for n in col_names:
        if sum(M[n]==M[n][0])== M.shape[0]:
            remove_these_columns.append(n)
    raise NotImplementedError

    
from ..generate.generate import val_eq
from ..generate.generate import val_lt
from ..generate.generate import val_gt
from ..generate.generate import val_between

#def row_not_within_region(M, col_name, boundar):
#    raise NotImplementedError

    
    

