import numpy as np
from collections import Counter


#checks

def is_within_region(L, point):
    import matplotlib.path as mplPath
    bbPath = mplPath.Path(np.array(L))
    return bbPath.contains_point(point)
    
#remove
def remove_these_columns(M, list_of_col_to_remove):
    return M[[col for col in M.dtype.names if col not in list_of_col_to_remove]]
    
    


def col_has_all_same_val(col):
    return np.all(col==col[0])

def col_has_one_unique_val(col):
    d = Counter(col)
    if len(d) == 2: #ignores case for -999, null
        return (1 in d.values())
    return False

def col_has_lt_threshold_unique_values(col, threshold):
    d = Counter(col)
    vals = sort(d.values())
    return ( sum(vals[:-1]) < threshold) 

