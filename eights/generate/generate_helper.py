import numpy as np

import numpy.lib.recfunctions as nprf

from generate_helper import *


def list_all_same_value(L):
    #this takes a list of lists, ie [[1],[1],[1]]
    ret = []
    for row in zip(*L):
        ret.append(len(set(row))<=1)
    return ret        

def time_less_then(time_1, time_2, val):
    #check both in datetime64
    return (time_2-time_1 < val) and  (time_2 - time_1 >= 0 ) 

def select_by_time_from(data, target, threshold, time_col_name):
    ret = []
    for idx, x in enumerate(data):
        if time_less_then(target, x[time_col_name]):
            ret.apped(idx)
    return ret

    
def true_if_rows_fit_values(M, cols, values):
    if len(cols)!= len(values):
        raise ValueError('cols and values misnumbered')
    ret = np.zeros(len(M))
    for x in M:
        for idc, c in enumerate(cols):
            if x[col] == values[idc]:
                ret[idc]=1
    return ret

##This code works but we currently are not useing
#def are_arrays_identical(RA_1, RA_2):
#    if len(RA_1) != len(RA_2):
#        raise ValueError('Lists are mismatched lengths')
#    else:
#        return RA_1==RA_2

def stack_rows(M1, M2):
    raise NotImplementedError

def sa_from_cols(cols):
    raise NotImplementedError

def append_columns(M, cols, names):
    if isinstance(cols, np.ndarray):
        cols = (cols,)
    return nprf.append_fields(M, names, data=cols, usemask=False)

def append_column(M, col):
    raise NotImplementedError
