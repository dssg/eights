
import numpy as np
import eights.investigate as inv
from collections import OrderedDict

class dbarray(np.ndarray):
    """
    Numpy array backed by SQL queries rather than actual data. Populated with
    actual data as soon as a ufunc is performed, but can be sliced on rows
    without being populated.
    """
    def __new__(cls, input_array, con_str):
        # Mostly boilerplate; adds "sigma" attribute
        obj = np.asarray(input_array).view(cls)
        obj._conn = inv.connect_sql(con_str)
        obj._queries = OrderedDict()
        return obj

    def __array_finalize__(self, obj):
        # Mostly boilerplate; adds "sigma" attribute
        if obj is None: return
        self._conn = getattr(obj, '_conn', None)
        self._queries = getattr(obj, '_queries', OrderedDict())

    def __array_prepare__(self, array, context=None):
        for col_name, query in self._queries.iteritems():
            self[:] = self._conn.execute(query)
        return array


    def append_query(self, col_name, query):
        self._queries[col_name] = query
