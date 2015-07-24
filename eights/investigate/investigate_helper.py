import csv
import numpy as np
from collections import Counter
from ..utils import *

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
 
def convert_list_to_structured_array(L, col_names, type_info):
    type_fixed = []
    for idx, x in enumerate(type_info):

        if x is not 'string':
            type_fixed.append((col_names[idx],x))
        if x =='string':
            max_val = max([len(z[idx]) for z in L])

            type_fixed.append((col_names[idx],'S'+ str(max_val)))

    return np.array(L, dtype=[type_info])

def describe_column(col):
    if col.dtype.kind not in ['f','i']:
        return {}
    cnt = len(col)
    mean = np.mean(np.array(col))
    std = np.std(np.array(col))
    mi = min(col)
    mx = max(col)
    return {'Count:' : cnt,'Mean:': mean, 'Standard Dev:': std, 'Minimal ': mi,'Maximal:': mx}

def crosstab(L_1, L_2):
    #is this nessasry?
    L_1 = np.array(L_1)
    L_2 = np.array(L_2)
    K_1 = np.unique(L_1)
    a_dict = {}
    for k in K_1:
        loc_k = np.where(np.array(L_1)==k)[0]
        tmp_list = L_2[loc_k]
        cnt = Counter(tmp_list)
        a_dict[k] = cnt
    return a_dict


