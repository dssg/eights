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
    
    
def remove_these_rows(M, list_of_rows_to_remove):
    #given List, remove rows
    return
