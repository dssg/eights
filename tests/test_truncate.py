import unittest
import numpy as np
from collections import Counter

#from eights.truncate.truncate import (all_col_equal,
#                                     remove_cols_if_all_same_value)

from eights.truncate.truncate_helper import (are_all_col_equal,
                                            is_only_one_unique
                                            )

class TestInvestigate(unittest.TestCase):
    def test_are_all_col_equal(self):
        correct = np.array([1,1,1,1])
        incorrect = np.array([1,1,1,2])
        self.assertTrue(are_all_col_equal(correct))
        self.assertFalse(are_all_col_equal(incorrect))
    
    def test_is_only_one_unique(self):
        correct = np.array([1,1,1,2])
        self.assertTrue(is_only_one_unique(correct))
        incorrect = np.array([1,1,2,2])
        self.assertFalse(is_only_one_unique(incorrect))
        


if __name__ == '__main__':
    unittest.main()

