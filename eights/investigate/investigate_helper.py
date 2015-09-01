import csv
import os
import cPickle
from collections import Counter
import numpy as np
import sqlalchemy as sqla
from ..utils import *
import itertools as it
from datetime import datetime


__special_csv_strings = {'': None,
                         'True': True,
                         'False': False} 

def __correct_csv_cell_type(cell):
    # Change strings in CSV to appropriate Python objects
    try:
        return __special_csv_strings[cell]
    except KeyError:
        pass
    try: 
        return int(cell)
    except ValueError:
        pass
    try:
        return float(cell)
    except ValueError:
        pass
    try:
        return parse(cell)
    except (TypeError, ValueError):
        pass
    return cell

def open_simple_csv_as_list(file_loc, delimiter=',', return_col_names=False):
    # infers types
    with open(file_loc, 'rU') as f:
        reader = csv.reader(f,  delimiter=delimiter)
        names = reader.next() # skip header
        data = [[__correct_csv_cell_type(cell) for cell in row] for
                row in reader]
    if return_col_names:
        return data, names
    return data
    
def open_csv_as_structured_array(file_loc, delimiter=','):
    python_list, names = open_simple_csv_as_list(file_loc, delimiter, True)
    return convert_list_to_structured_array(python_list, names)

def convert_fixed_width_list_to_CSV_list(data, list_of_widths):
    #assumes you loaded a fixed with thing into a list of list csv.
    #not clear what this does with the 's's...
    s = "s".join([str(s) for s in list_of_widths])
    s = s + 's'
    out = []
    for x in data:
        out.append(struct.unpack(s, x[0]))
    return out

# let's not use this any more
#def set_structured_array_datetime_as_day(first_pass,file_loc, delimiter=','):
#    date_cols = []
#    int_cols = []
#    new_dtype = []
#    for i, (col_name, col_dtype) in enumerate(first_pass.dtype.descr):
#        if 'S' in col_dtype:
#            col = first_pass[col_name]
#            if np.any(validate_time(col)):
#                date_cols.append(i)
#		# TODO better inference
#                # col_dtype = 'M8[D]'
#                col_dtype = np.datetime64(col[0]).dtype
#        elif 'i' in col_dtype:
#            int_cols.append(i)
#        new_dtype.append((col_name, col_dtype))
#    
#    converter = {i: str_to_time for i in date_cols}        
#    missing_values = {i: '' for i in int_cols}
#    filling_values = {i: -999 for i in int_cols}
#    return np.genfromtxt(file_loc, dtype=new_dtype, names=True, delimiter=delimiter,
#                         converters=converter, missing_values=missing_values,
#                         filling_values=filling_values)



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
    # If allow_caching is True, will pickle results in cache_dir and reuse
    # them if it encounters an identical query twice.
    def __init__(self, con_str, allow_caching=False, cache_dir='.'):
        self.__engine = sqla.create_engine(con_str)
        self.__cache_dir = cache_dir
        if allow_caching:
            self.execute = self.__execute_with_cache

    def __sql_to_sa(self, exec_str):
        raw_python = self.__engine.execute(exec_str)
        return convert_list_to_structured_array(
            raw_python.fetchall(),
            [str(key) for key in raw_python.keys()])

    def __execute_with_cache(self, exec_str, invalidate_cache=False):
        pkl_file_name = os.path.join(
            self.__cache_dir, 
            'eights_cache_{}.pkl'.format(hash(exec_str)))
        if os.path.exists(pkl_file_name) and not invalidate_cache:
            with open(pkl_file_name) as fin:
                return cPickle.load(fin)
        ret = self.__sql_to_sa(exec_str)
        with open(pkl_file_name, 'w') as fout:
            cPickle.dump(ret, fout)
        return ret

    def execute(self, exec_str, invalidate_cache=False):
        return self.__sql_to_sa(exec_str)
