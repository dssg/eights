import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

def cast_to_n_bins(data, num_bins):
    minimum = min(data)
    maximum = max(data)
    distance = maximum-minimum
    l =[]
    for x in data:
        l.append(int(((x-minimum)/distance)*num_bins))
    return l

def label_encoding(data):
    #requires smae datatype
    le = preprocessing.LabelEncoder()
    return le.fit_transform(data) 

def replace_missing_value(missing_val, strategy, data):
    if strategy not in ['mean','median','most_frequent']:
        raise ValueError('Invalid strategy')
    imp = Imputer(missing_values=missing_val, strategy=strategy, axis=0)
    return imp.fit_transform(data)


