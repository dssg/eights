import unittest
import numpy as np

import utils_for_tests as uft
from eights import array_emitter

class TestArrayEmitter(unittest.TestCase):

    def test_array_emitter(self):
        db_file = uft.path_of_data('rg_students.db')
        conn_str = 'sqlite:///{}'.format(db_file)
        ae = array_emitter.ArrayEmitter()
        ae.get_rg_from_sql(conn_str, 'rg_students')
        M = ae.emit_M(2005, 2007)
        print M
        print M.dtype
        
if __name__ == '__main__':
    unittest.main()


