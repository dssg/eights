import numpy as np
import numpy.lib.recfunctions as nprf
import matplotlib.mlab

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

def invert_dictionary(aDict):
    return {v: k for k, v in aDict.items()}

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

__type_permissiveness_ranks = {'b': 0, 'M': 100, 'm': 100, 'i': 200, 'f': 300, 'S': 400}
def __type_permissiveness(dtype):
    # TODO handle other types
    return __type_permissiveness_ranks[dtype.kind] + dtype.itemsize

def np_dtype_is_homogeneous(A):
    """True iff dtype is nonstructured or every sub dtype is the same"""
    # http://stackoverflow.com/questions/3787908/python-determine-if-all-items-of-a-list-are-the-same-item
    if not is_sa(A):
        return True
    dtype = A.dtype
    first_dtype = dtype[0]
    return all(dtype[i] == first_dtype for i in xrange(len(dtype)))

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

def cast_np_sa_to_nd(sa):
    """
    
    Returns a view of a numpy structured array as a single-type 1 or
    2-dimensional array. If the resulting nd array would be a column vector,
    returns a 1-d array instead. If the resulting array would have a single
    entry, returns a 0-d array instead
    All elements are converted to the most permissive type. permissiveness
    is determined first by finding the most permissive type in the ordering:
    datetime64 < int < float < string
    then by selecting the longest typelength among all columns with with that
    type.
    If the sa does not have a homogeneous datatype already, this may require
    copying and type conversion rather than just casting. Consequently, this
    operation should be avoided for heterogeneous arrays
    Based on http://wiki.scipy.org/Cookbook/Recarray.
    Parameters
    ----------
    sa : numpy.ndarray
        The structured array to view
    Returns
    -------
    np.ndarray
    """
    if not is_sa(sa):
        return sa
    dtype = sa.dtype
    if len(dtype) == 1:
        if sa.size == 1:
            return sa.view(dtype=dtype[0]).reshape(())
        return sa.view(dtype=dtype[0]).reshape(len(sa))
    if np_dtype_is_homogeneous(sa):
        return sa.view(dtype=dtype[0]).reshape(len(sa), -1)
    # If type isn't homogeneous, we have to convert
    dtype_it = (dtype[i] for i in xrange(len(dtype)))
    most_permissive = max(dtype_it, key=__type_permissiveness)
    col_names = dtype.names
    cols = (sa[col_name].astype(most_permissive) for col_name in col_names)
    nd = np.column_stack(cols)
    return nd

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

def get_type(d):
    #assumes everything identical structure
    if type(d) == list:
        return 'list'
    elif type(d)== np.ndarray:
        if is_sa(d):
            return 'sa'
        elif len(d.shape)==1:
            return
    return


def stack_rows(M1, M2):
    raise NotImplementedError

def sa_from_cols(cols):
    return nprf.merge_arrays(cols, usemask=False)    

def append_cols(M, cols, names):
    return nprf.append_fields(M, names, data=cols, usemask=False)

def remove_cols(M, col_names):
    return nprf.drop_fields(M, col_names, usemask=False)

def __fill_by_descr(s):
    if 'b' in s:
        return False
    if 'i' in s:
        return -999
    if 'f' in s:
        return np.nan
    if 'S' in s:
        return ''
    if 'U' in s:
        return u''
    if 'M' in s or 'm' in s:
        return np.datetime64('NaT')
    raise ValueError('Unrecognized description {}'.format(s))

def join(left, right, how, left_on, right_on, suffixes=('_x', '_y')):
    # approximates Pandas DataFrame.merge
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html
    # implements a hash join 
    # http://blogs.msdn.com/b/craigfr/archive/2006/08/10/687630.aspx

    # left_on and right_on can both be strings or lists
    if isinstance(left_on, basestring):
        left_on = [left_on]
    if isinstance(right_on, basestring):
        right_on = [right_on]

    # get arrays without join columns
    left_no_idx = remove_cols(left, left_on)
    right_no_idx = remove_cols(right, right_on)

    # assemble dtype for the merged array
    col_names = ([left_on_col + suffixes[0] for left_on_col in left_on] + 
                 [name + suffixes[0] for name in 
                            left_no_idx.dtype.names] +
                 [name + suffixes[1] for name in right_no_idx.dtype.names])
    col_dtypes = ([left[left_on_col].dtype for left_on_col in left_on] +
                  [left_no_idx[col].dtype for col in 
                   left_no_idx.dtype.names] +
                  [right_no_idx[col].dtype for col in
                   right_no_idx.dtype.names])

    take_all_right_rows = how in ('outer', 'right')
    take_all_left_rows = how in ('outer', 'left')
    # data to fill in if we're doing an outer join and one of the sides is
    # missing
    left_fill = tuple([__fill_by_descr(dtype) for _, dtype in 
                       left_no_idx.dtype.descr])
    right_fill = tuple([__fill_by_descr(dtype) for _, dtype in 
                       right_no_idx.dtype.descr])

    # Make a hash of the first join column in the left table
    left_col = left[left_on[0]]
    hashed_col = {}
    for left_idx, left_cell in enumerate(left_col):
        try:
            rows = hashed_col[left_cell]
        except KeyError:
            rows = []
            hashed_col[left_cell] = rows
        rows.append(left_idx)

    # Pick out columns that we will be joining on beyond the 0th
    extra_left_cols = [left[left_on_name] for left_on_name in left_on[1:]]
    extra_right_cols = [right[right_on_name] for right_on_name in right_on[1:]]
    extra_contraint_cols = zip(extra_left_cols, extra_right_cols)

    rows_new_table = []
    right_col = right[right_on[0]]
    # keep track of used left rows so we can include all the rows if we're
    # doint a left or outer join
    left_rows_used = set()
    # Iterate through every row in the right table
    for right_idx, right_cell in enumerate(right_col):
        has_match = False
        # See if we have matches from the hashed col of the left table
        try:
            left_matches = hashed_col[right_cell]
            
            for left_idx in left_matches:
                # If all the constraints are met, we have a match
                if all([extra_left_col[left_idx] == extra_right_col[right_idx] 
                        for extra_left_col, extra_right_col in 
                        extra_contraint_cols]):
                    has_match = True
                    rows_new_table.append(
                            tuple([left[left_on_col][left_idx] 
                                   for left_on_col in left_on]) +
                            tuple(left_no_idx[left_idx]) + 
                            tuple(right_no_idx[right_idx]))
                    left_rows_used.add(left_idx) 
        # No match found for this right row
        except KeyError:
            pass  
        # If we're doing a right or outer join and we didn't find a match, add
        # this row from the right table, filled with type-appropriate versions
        # of NULL from the left table
        if (not has_match) and take_all_right_rows:
            rows_new_table.append(
                    tuple([right[right_on_col][right_idx] for right_on_col in
                           right_on]) + left_fill + 
                    tuple(right_no_idx[right_idx]))

    # if we're doing a left or outer join, we have to add all rows from the 
    # left table, using type-appropriate versions of NULL for the right table
    if take_all_left_rows:    
        left_rows_unused = [i for i in xrange(len(left)) if i not in 
                            left_rows_used]
        for unused_left_idx in left_rows_unused:
            rows_new_table.append(
                    tuple([left[left_on_col][left_idx] 
                           for left_on_col in left_on]) +
                    tuple(left_no_idx[unused_left_idx]) +
                    right_fill)

    return np.array(rows_new_table, dtype={'names': col_names, 
                                           'formats': col_dtypes})
