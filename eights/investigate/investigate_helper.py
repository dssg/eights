import csv
from collections import Counter
import numpy as np
import sqlalchemy as sqla
from ..utils import *

def open_simple_csv_as_list(file_loc,delimiter=','):
    with open(file_loc, 'rU') as f:
        reader = csv.reader(f,  delimiter=delimiter)
        data= list(reader)
    return data
    
def open_csv_as_structured_array(file_loc, delimiter=','):
    return np.genfromtxt(file_loc, dtype=None, names=True, delimiter=delimiter)

def convert_fixed_width_list_to_CSV_list(data, list_of_widths):
    #assumes you loaded a fixed with thing into a list of list csv.
    #not clear what this does with the 's's...
    s = "s".join([str(s) for s in list_of_widths])
    s = s + 's'
    out = []
    for x in data:
        out.append(struct.unpack(s, x[0]))
    return out

def set_structured_array_datetime_as_day(first_pass,file_loc, delimiter=','):
    date_cols = []
    int_cols = []
    new_dtype = []
    for i, (col_name, col_dtype) in enumerate(first_pass.dtype.descr):
        if 'S' in col_dtype:
            col = first_pass[col_name]
            if np.any(validate_time(col)):
                date_cols.append(i)
		# TODO better inference
                # col_dtype = 'M8[D]'
                col_dtype = np.datetime64(col[0]).dtype
        elif 'i' in col_dtype:
            int_cols.append(i)
        new_dtype.append((col_name, col_dtype))
    
    converter = {i: str_to_time for i in date_cols}        
    missing_values = {i: '' for i in int_cols}
    filling_values = {i: -999 for i in int_cols}
    return np.genfromtxt(file_loc, dtype=new_dtype, names=True, delimiter=delimiter,
                         converters=converter, missing_values=missing_values,
                         filling_values=filling_values)

def _u_to_ascii(s):
    if isinstance(s, unicode):
        return s.encode('utf-8')
    return s
 
def convert_list_to_structured_array(L, col_names=None, dtype=None):
    # TODO deal w/ datetimes, unicode, null etc
    # TODO don't blow up if we're inferring types and types are inhomogeneous
    # TODO utils.cast_list_of_list_to_sa is redundant
    if dtype is None:
        row1 = L[0]
        n_cols = len(row1)
        if col_names is None:
            col_names = ['f{}'.format(i) for i in xrange(n_cols)]
        # can't have unicode col names?
        col_names = [name.encode('utf-8') for name in col_names]
        dtype = []
        for idx, cell in enumerate(row1):
            if isinstance(cell, int):
                dtype.append((col_names[idx], int))
            elif isinstance(cell, float):
                dtype.append((col_names[idx], float))
            else:
                dtype.append((col_names[idx], str))
        dtype = np.dtype(dtype)
        
    dtype_fixed = []

    for idx, (name, type_descr) in enumerate(dtype.descr):
        if 'S' in type_descr:
            max_val = max([len(row[idx]) for row in L])
            dtype_fixed.append((name, 'S' + str(max_val)))
        else:
            dtype_fixed.append((name, type_descr))
    # Can't have unicode anywhere. Also, needs to explicity convert to tuples
    L = [tuple([_u_to_ascii(cell) for cell in row]) for row in L]
    return np.array(L, dtype=dtype_fixed)

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


class SQLConnection(object):
    # Intended to vaguely implement DBAPI 2
    # TODO get this to work w/ time, unicode
    def __init__(self, con_str):
        self.__engine = sqla.create_engine(con_str)

    def execute(self, exec_str):
        result = self.__engine.execute(exec_str)
        return convert_list_to_structured_array(result.fetchall(), result.keys())
