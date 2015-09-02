import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from eights.utils import convert_to_sa

def replace_with_n_bins(col, num_bins):
    #this just drops a list we still have to reattach or overwrite
    minimum = float(min(col))
    maximum = float(max(col))
    distance = float(maximum-minimum)
    l =[]
    for x in col:
        l.append(int(((x-minimum)/distance)*num_bins))
    return l

def label_encode(M):
    """
    Changes string cols to integers so that there is a 1-1 mapping between 
    strings and ints
    """

    M = convert_to_sa(M)
    le = preprocessing.LabelEncoder()
    new_dtype = []
    result_arrays = []
    for (col_name, fmt) in M.dtype.descr:
        if 'S' in fmt:
            result_arrays.append(le.fit_transform(M[col_name]))
            new_dtype.append((col_name, int))
        else:
            result_arrays.append(M[col_name])
            new_dtype.append((col_name, fmt))
    return np.array(zip(*result_arrays), dtype=new_dtype)

def replace_missing_vals(M, strategy, missing_val=np.nan, constant=0):
    # TODO support times, strings
    M = convert_to_sa(M)

    if strategy not in ['mean', 'median', 'most_frequent', 'constant']:
        raise ValueError('Invalid strategy')

    M_cp = M.copy()

    if strategy == 'constant':

        try:
            missing_is_nan = np.isnan(missing_val)
        except TypeError:
            # missing_val is not a float
            missing_is_nan = False

        if missing_is_nan: # we need to be careful about handling nan
            for col_name, col_type in M_cp.dtype.descr:
                if 'f' in col_type:
                    col = M_cp[col_name]
                    col[np.isnan(col)] = constant
            return M_cp        

        for col_name, col_type in M_cp.dtype.descr:
            if 'i' in col_type or 'f' in col_type:
                col = M_cp[col_name]
                col[col == missing_val] = constant
        return M_cp

    # we're doing one of the sklearn imputer strategies
    imp = Imputer(missing_values=missing_val, strategy=strategy, axis=1)
    for col_name, col_type in M_cp.dtype.descr:
        if 'f' in col_type or 'i' in col_type:
            # The Imputer only works on float and int columns
            col = M_cp[col_name]
            col[:] = imp.fit_transform(col)
    return M_cp


