import csv
import numpy as np
from collections import Counter

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

def open_simple_csv_as_list(file_loc):
    with open(file_loc, 'rb') as f:
        reader = csv.reader(f)
        data= list(reader)
    return data
    
def open_csv_as_structured_array(file_loc):
    return np.genfromtxt(file_loc, dtype=None, names=True, delimiter=',')

def convert_fixed_width_list_to_CSV_list(data, list_of_widths):
    #assumes you loaded a fixed with thing into a list of list csv.
    #not clear what this does with the 's's...
    s = "s".join([str(s) for s in list_of_widths])
    s = s + 's'
    out = []
    for x in data:
        out.append(struct.unpack(s, x[0]))
    return out

def set_structured_array_datetime_as_day(first_pass,file_loc):
    date_cols = []
    int_cols = []
    new_dtype = []
    for i, (col_name, col_dtype) in enumerate(first_pass.dtype.descr):
        if 'S' in col_dtype:
            col = first_pass[col_name]
            if np.any(validate_time(col)):
                date_cols.append(i)
                col_dtype = 'M8[D]'
        elif 'i' in col_dtype:
            int_cols.append(i)
        new_dtype.append((col_name, col_dtype))
    
    converter = {i: str_to_time for i in date_cols}        
    missing_values = {i: '' for i in int_cols}
    filling_values = {i: -999 for i in int_cols}
    return np.genfromtxt(file_loc, dtype=new_dtype, names=True, delimiter=',',
                         converters=converter, missing_values=missing_values,
                         filling_values=filling_values)
 
def cast_np_nd_to_sa(nd, dtype=None):
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
        dtype = np.dtype({'names': map('f{}'.format, xrange(n_cols)),
                          'formats': [nd_dtype for i in xrange(n_cols)]})
        return nd.reshape(nd.size).view(dtype)
    type_len = nd_dtype.itemsize
    if all(dtype[i] == nd_dtype for i in xrange(len(dtype))):
        return nd.reshape(nd.size).view(dtype)
    # if the user requests an incompatible type, we have to convert
    cols = (nd[:,i].astype(dtype[i]) for i in xrange(len(dtype))) 
    return np.array(it.izip(*cols), dtype=dtype)
    

def convert_list_to_structured_array(L, col_names, type_info):
    type_fixed = []
    for idx, x in enumerate(type_info):

        if x is not 'string':
            type_fixed.append((col_names[idx],x))
        if x =='string':
            max_val = max([len(z[idx]) for z in L])

            import pdb; pdb.set_trace()

            type_fixed.append((col_names[idx],'S'+ str(max_val)))

    dtype=[('f0', int), ('f1', 'S6'), ('f2', float)]
    return np.array(L, dtype=[type_info])

def describe_column(col):
    if col.dtype.kind not in ['f','i']:
        return []
    cnt = len(col)
    mean = np.mean(np.array(col))
    std = np.std(np.array(col))
    mi = min(col)
    mx = max(col)
    return [cnt, mean, std, mi, mx]

def crosstab(L_1, L_2):
    #is this nessasry?
    L_1 = np.array(L_1)
    L_2 = np.array(L_2)
    K_1 = np.unique(L_1)
    K_2 = np.unique(L_2)
    a_dict = {}
    for k in K_1:
        loc_k = np.where(np.array(L_1)==k)[0]
        tmp_list = L_2[loc_k]
        cnt = Counter(tmp_list)
        a_dict[k] = cnt
    return a_dict


