import numpy as np

@np.vectorize
def validate_time(date_text):
    if not date_text:
        return False
    try:
        np.datetime64(date_text)
        return True
    except ValueError:
        return False
def str_to_time(date_text):
    try:
        return np.datetime64(date_text)
    except ValueError:
        return np.datetime64('NaT')    


#
def cast_list_of_list_to_sa(lol, dtype=None, names=None):
    nd = np.array(lol)
    return cast_np_nd_to_sa(nd, dtype=dtype, names=names)

def convert_to_sa(M, c_name=None):
    """Converts an list of lists or a np ndarray to a Structured Arrray
    Parameters
    ----------
    M  : List of List or np.ndarray
       This is the Matrix M, that it is assumed is the basis for the ML algorithm

    Attributes
    ----------
    temp : type
       Description

    Returns
    -------
    temp : Numpy Structured array
       This is the matrix of an appropriate type that eights expects.

    """
    if is_sa(M):
        return M

    if is_nd(M):
        return cast_np_nd_to_sa(M, names=c_name)

    if isinstance(M, list):
        return cast_list_of_list_to_sa(M, names=c_name)
        # TODO make sure this function ^ ensures list of /lists/

    raise ValueError('Can\'t cast to sa')

def cast_np_nd_to_sa(nd, dtype=None, names=None):
    """
    
    Returns a view of a numpy, single-type, 0, 1 or 2-dimensional array as a
    structured array
    Parameters
    ----------
    nd : numpy.ndarray
        The array to view
    dtype : numpy.dtype or None (optional)
        The type of the structured array. If not provided, or None, nd.dtype is
        used for all columns.
        If the dtype requested is not homogeneous and the datatype of each
        column is not identical nd.dtype, this operation may involve copying
        and conversion. Consequently, this operation should be avoided with
        heterogeneous or different datatypes.
    Returns
    -------
    A structured numpy.ndarray
    """
    if nd.ndim not in (0, 1, 2):
        raise TypeError('np_nd_to_sa only takes 0, 1 or 2-dimensional arrays')
    nd_dtype = nd.dtype
    if nd.ndim <= 1:
        nd = nd.reshape(nd.size, 1)
    if dtype is None:
        n_cols = nd.shape[1]
        if names is None:
            names = map('f{}'.format, xrange(n_cols))
        dtype = np.dtype({'names': names,
                          'formats': [nd_dtype for i in xrange(n_cols)]})
        return nd.reshape(nd.size).view(dtype)
    type_len = nd_dtype.itemsize
    if all(dtype[i] == nd_dtype for i in xrange(len(dtype))):
        return nd.reshape(nd.size).view(dtype)
    # if the user requests an incompatible type, we have to convert
    cols = (nd[:,i].astype(dtype[i]) for i in xrange(len(dtype))) 
    return np.array(it.izip(*cols), dtype=dtype)

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

def is_sa(M):
    return is_nd(M) and M.dtype.names is not None

def is_nd(M):
    return isinstance(M, np.ndarray)

