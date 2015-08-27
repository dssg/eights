import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

def replace_with_n_bins(data, num_bins):
    #this just drops a list we still have to reattach or overwrite
    minimum = float(min(data))
    maximum = float(max(data))
    distance = float(maximum-minimum)
    l =[]
    for x in data:
        l.append(int(((x-minimum)/distance)*num_bins))
    return l

def label_encoding(data):
    #requires same datatype
    #what does this do?
    le = preprocessing.LabelEncoder()
    return le.fit_transform(data) 

def replace_missing_vals(missing_val, strategy, data, constant=0):
    #TODO work on structured arrays in general
    if strategy not in ['mean', 'median', 'most_frequent', 'constant']:
        raise ValueError('Invalid strategy')
    if strategy == 'constant':
        data_cp = data.copy()
        # TODO work on things other than float
        for col_name, col_type in data_cp.dtype.descr:
            if 'f' in col_type:
                col = data_cp[col_name]
                if np.isnan(missing_val):
                    col[np.isnan(col)] = constant
                else:
                    col[col == missing_val] = constant
        return data_cp
    imp = Imputer(missing_values=missing_val, strategy=strategy, axis=0)
    return imp.fit_transform(data)


