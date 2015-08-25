import csv
from collections import Counter
import numpy as np
import sqlalchemy as sqla
from ..utils import *
import itertools as it

def open_simple_csv_as_list(file_loc):
    with open(file_loc, 'rb') as f:
        reader = csv.reader(f)
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


TYPE_PRECEDENCE = {type(None): 0, 
                   bool: 100,
                   int: 200, 
                   long: 300,
                   float: 400,
                   str: 500,
                   unicode: 600}

def __primitive_clean(cell, expected_type, alt):
    if cell == None:
        return alt
    try:
        return expected_type(cell)
    except TypeError:
        return alt

CLEAN_FUNCTIONS = {type(None): lambda cell: '',
                   bool: lambda cell: __primitive_clean(cell, bool, False),
                   int: lambda cell: __primitive_clean(cell, int, -999),
                   long: lambda cell: __primitive_clean(cell, long, -999L),
                   float: lambda cell: __primitive_clean(cell, float, np.nan),
                   str: lambda cell: __primitive_clean(cell, str, ''),
                   unicode: lambda cell: __primitive_clean(cell, unicode, u'')}

STR_TYPE_LETTERS = {str: 'S',
                    unicode: 'U'}

def convert_list_to_structured_array(L, col_names=None, dtype=None):
    # TODO deal w/ datetimes
    # TODO utils.cast_list_of_list_to_sa is redundant
    n_cols = len(L[0])
    if col_names is None:
        col_names = ['f{}'.format(i) for i in xrange(n_cols)]
    dtypes = []
    cleaned_cols = []
    if dtype is None:
        for idx, col in enumerate(it.izip(*L)):
            dom_type = type(max(
                col, 
                key=lambda cell: TYPE_PRECEDENCE[type(cell)]))
            if dom_type in (bool, int, float, long, float):
                dtypes.append(dom_type)
                cleaned_cols.append(map(CLEAN_FUNCTIONS[dom_type], col))
            elif dom_type in (str, unicode): 
                cleaned_col = map(CLEAN_FUNCTIONS[dom_type], col)
                max_len = max(
                        len(max(cleaned_col, 
                            key=lambda cell: len(dom_type(cell)))),
                        1)
                dtypes.append('|{}{}'.format(
                    STR_TYPE_LETTERS[dom_type],
                    max_len))
                cleaned_cols.append(cleaned_col)
            elif dom_type == type(None):
                # column full of None make it a column of empty strings
                dtypes.append('|S1')
                cleaned_cols.append([''] * len(col))
            else:
                raise ValueError(
                        'Type of col: {} could not be determined'.format(
                            col_names[idx]))

    return np.fromiter(it.izip(*cleaned_cols), 
                       dtype={'names': col_names, 
                              'formats': dtypes})

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
