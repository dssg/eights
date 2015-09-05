import unittest
from eights.dbarray import dbarray
from utils_for_tests import path_of_data
import numpy as np

class TestDbarray(unittest.TestCase):
    def test_basic(self):
        db = dbarray([0, 0, 0], 'sqlite:///{}'.format(path_of_data('small.db')))
        db.append_query('id', 'SELECT id FROM employees')
        print db + 1

if __name__ == '__main__':
    unittest.main()
