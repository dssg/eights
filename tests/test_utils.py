import unittest
import utils_for_tests
from eights import utils

import numpy as np
import pandas as pd

class TestUtils(unittest.TestCase):
    def test_generate_matrix(self):
        M, y = utils_for_tests.generate_test_matrix(100, 5, 3, [float, str, int])
        print M
        print y

    def test_join(self):
        a1 = np.array([(0, 'Lisa', 2),
                       (1, 'Bill', 1),
                       (2, 'Fred', 2),
                       (3, 'Samantha', 2),
                       (4, 'Augustine', 1),
                       (5, 'William', 0)], dtype=[('id', int),
                                                  ('name', 'S64'),
                                                  ('dept_id', int)])
        a2 = np.array([(0, 'accts receivable'),
                       (1, 'accts payable'),
                       (2, 'shipping')], dtype=[('id', int),
                                                ('name', 'S64')])
        ctrl = pd.DataFrame(a1).merge(
                    pd.DataFrame(a2),
                    left_on='dept_id',
                    right_on='id').to_records(index=False)
        res = utils.join(a1, a2, 'inner', 'dept_id', 'id')
        print res
        print res.dtype
        print
        print ctrl
        print ctrl.dtype


if __name__ == '__main__':
    unittest.main()
