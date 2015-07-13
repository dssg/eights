import numpy as np
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

def distance(lat_1, lon_1, lat_2, lon_2):
    from math import radians, cos, sin, asin, sqrt
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    from:
    http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # convert decimal degrees to radians 

    lon_1, lat_1, lon_2, lat_2 = map(radians, [lon_1, lat_1, lon_2, lat_2])

    # haversine formula 
    dlon = lon_2 - lon_1 
    dlat = lat_2 - lat_1 
    a = sin(dlat/2)**2 + cos(lat_1) * cos(lat_2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 # 6371 Radius of earth in kilometers. Use 3956 for miles
    return c * r
    
def dist_less_than(lat_1, lon_1, lat_2, lon_2, threshold):
    """single line description
    Parameters
    ----------
    val : float
       miles 
    Returns
    -------
    boolean 
    
    """
    return (distance(lat_1, lon_1, lat_2, lon_2) < threshold)

    
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

