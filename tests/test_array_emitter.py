import unittest
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
        
if __name__ == '__main__':
    unittest.main()


