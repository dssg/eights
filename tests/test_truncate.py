import unittest
import numpy as np
from collections import Counter

from eights.utils import remove_cols,cast_list_of_list_to_sa

from eights.truncate.truncate_helper import (col_has_all_same_val)

from eights.truncate.truncate import (remove_col_where,
                                      all_equal_to,
                                      all_same_value,
                                      fewer_then_n_nonzero_in_col,
                                      remove_rows_where,
                                      val_eq)


class TestInvestigate(unittest.TestCase):
    def test_are_all_col_equal(self):
        M = cast_list_of_list_to_sa(
            [[1,2,3], [1,3,4], [1,4,5]],
            col_names=['height','weight', 'age'])
        
        arguments = [{'func': all_equal_to,  'vals': 1}]  
        M = remove_col_where(M, arguments)  
        correct = cast_list_of_list_to_sa(
            [[2,3], [3,4], [4,5]],
            col_names=['weight', 'age'])    
        self.assertTrue(np.array_equal(M, correct))
        
    def test_all_same_value(self):
        M = cast_list_of_list_to_sa(
            [[1,2,3], [1,3,4], [1,4,5]],
            col_names=['height','weight', 'age'])
        arguments = [{'func': all_same_value,  'vals': None}]  
        M = remove_col_where(M, arguments)  
        correct = cast_list_of_list_to_sa(
            [[2,3], [3,4], [4,5]],
            col_names=['weight', 'age'])    
        self.assertTrue(np.array_equal(M, correct)) 
        
    def test_fewer_then_n_nonzero_in_col(self):
        M = cast_list_of_list_to_sa(
            [[0,2,3], [0,3,4], [1,4,5]],
            col_names=['height','weight', 'age'])
        arguments = [{'func': fewer_then_n_nonzero_in_col,  'vals': 2}]  
        M = remove_col_where(M, arguments)  
        correct = cast_list_of_list_to_sa(
            [[2,3], [3,4], [4,5]],
            col_names=['weight', 'age'])   
        self.assertTrue(np.array_equal(M, correct))    
                   
    def test_remove_row(self):
        M = cast_list_of_list_to_sa(
            [[0,2,3], [0,3,4], [1,4,5]],
            col_names=['height','weight', 'age'])
        arguments = [{'func': fewer_then_n_nonzero_in_col,  'vals': 2}]  
        M = remove_rows_where(M, val_eq, 'weight', 3)
        correct = cast_list_of_list_to_sa(
             [[0, 2, 3], [1, 4, 5]],
            col_names=['height','weight', 'age'])   
        self.assertTrue(np.array_equal(M, correct))   



if __name__ == '__main__':
    unittest.main()

