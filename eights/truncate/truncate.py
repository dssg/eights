import numpy as np
from collections import Counter

from truncate_helper import *

def remove_cols_if_all_same_value(M):
    remove_column_list =[name for name in M.dtype.names if are_all_col_equal(M[name])]
    M = remove_these_columns(M, remove_column_list)
    return M
    
def remove_cols_if_one_unique_value(M):
    remove_column_list = [name for name in M.name if is_only_one_unique(M[name])]
    M = remove_cols(M, remove_column_list)
    return M    

def remove_cols_if_few_unique_values(M,threshold):
    remove_column_list =[name for name in M.name if are_few_unique(M[name],threshold)]
    M = remove_cols(M, remove_column_list)
    return M
    
#Negative examples(critera flip floped here
def remove_rows_if_too_far_from(M, col_id, origin, dist):
    remove_row_list = [row_id for row_id, entry in enumerate(M[col_id]) if too_far_from(entry, origin, max_dist)]
    M = remove_rows(M, remove_row_list)
    return M
    
#negative example(ie ousdie region)
def remove_rows_if_outside_region(M, col_id, region):
    remove_row_list = [row_id for row_id, entry in enumerate(M[col_id]) if not is_within_region(entry, region)]
    M = remove_rows(M, remove_row_list)
    return M


    

