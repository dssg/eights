import unittest
from datetime import datetime
import numpy as np

import utils_for_tests as uft
from eights import array_emitter

class TestArrayEmitter(unittest.TestCase):

    def test_basic(self):
        db_file = uft.path_of_data('rg_students.db')
        conn_str = 'sqlite:///{}'.format(db_file)
        ae = array_emitter.ArrayEmitter()
        ae.get_rg_from_sql(conn_str, 'rg_students')
        ae.set_aggregation('absences', 'MAX')
        res = ae.emit_M(2005, 2007)
        ctrl = np.array([(0, 2.2, 3.95, 8.0),
                         (1, 3.45, np.nan, 0.0),
                         (2, 3.4, np.nan, 96.0)],
                        dtype=[('id', '<i8'), ('math_gpa', '<f8'), 
                               ('english_gpa', '<f8'), 
                               ('absences', '<f8')])
        self.assertTrue(uft.array_equal(res, ctrl))

    def test_complex_date(self):
        db_file = uft.path_of_data('rg_complex_dates.db')
        conn_str = 'sqlite:///{}'.format(db_file)
        ae = array_emitter.ArrayEmitter(convert_to_unix_time=True)
        ae.set_aggregation('bounded', 'SUM')
        ae.set_aggregation('no_start', 'SUM')
        ae.set_aggregation('no_stop', 'SUM')
        ae.set_aggregation('unbounded', 'SUM')
        ae.get_rg_from_sql(conn_str, 'rg_complex_dates', feature_col='feature')
        res1 = ae.emit_M(datetime(2010, 1, 1), datetime(2010, 6, 30))
        res2 = ae.emit_M(datetime(2010, 7, 1), datetime(2010, 12, 31))
        res3 = ae.emit_M(datetime(2010, 1, 1), datetime(2010, 12, 31))
        ctrl_dtype = [('id', '<i8'), ('bounded', '<f8'), 
                      ('no_start', '<f8'), ('no_stop', '<f8'), 
                      ('unbounded', '<f8')]
        ctrl1_dat = [(0, 1.0, 100.0, 100000.0, 1000000.0),
                     (1, 0.01, 0.001, 1e-06, 1e-07), 
                     (2, np.nan, np.nan, np.nan, 2e-08)]
        ctrl2_dat = [(0, 10.0, 1000.0, 10000.0, 1000000.0),
                     (1, 0.1, 0.0001, 1e-05, 1e-07),
                     (2, np.nan, np.nan, np.nan, 2e-08)]
        ctrl3_dat = [(0, 11.0, 1100.0, 110000.0, 1000000.0),
                     (1, 0.11, 0.0011, 1.1e-05, 1e-07),
                     (2, np.nan, np.nan, np.nan, 2e-08)]
        for res, ctrl_dat in zip((res1, res2, res3), (ctrl1_dat, ctrl2_dat, 
                                                      ctrl3_dat)):
            self.assertTrue(uft.array_equal(
                res, 
                np.array(ctrl_dat, dtype=ctrl_dtype)))  

if __name__ == '__main__':
    unittest.main()


