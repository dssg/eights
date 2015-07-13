import numpy as np
from collections import Counter


#checks
def are_all_col_equal(col):
    return np.all(col==col[0])

def is_only_one_unique(col):
    d = Counter(col)
    if len(d) == 2: #ignores case for -999, null
        return (1 in d.values())
    return False


def are_few_unique(col, threshold):
    ##Need to deal with -999 etc. right now a couple -999 fucks shit up.
    d = Counter(col)
    vals = sort(d.values())
    return ( sum(vals[:-1]) < threshold) 

def distance(lat_1, lng_1, lat_2, lng_2):
    from math import radians, cos, sin, asin, sqrt
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    from:
    http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = map(radians, [lat_1, lng_1, lat_2, lng_2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 # 6371 Radius of earth in kilometers. Use 3956 for miles
    return c * r    

def too_far_from(lat_1, lng_1, lat_2, lng_2, max_dist):
    return (distance(lat_1, lng_1, lat_2, lng_2) > max_dist)

def is_within_region(L, point):
    from matplotlib.path as mplPath
    bbPath = mplPath.Path(np.array(L))
    return bbPath.contains_point(point)
    
#remove
def remove_these_columns(M, list_of_col_to_remove):
    return M[col for col in M.dtype.names if col not in list_of_col_to_remove]
    
    
def remove_these_rows(M, list_of_rows_to_remove):
    #given List, remove rows
    return