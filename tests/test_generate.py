import unittest
import numpy as np
from numpy.random import rand
import eights.investigate
import eights.generate
import eights.generate.generate_helper as gh


class TestGenerate(unittest.TestCase):
    def test_where_all_are_true(self):
        M = [[1,2,3], [2,3,4], [3,4,5]]
        col_names = ['heigh','weight', 'age']
        lables= [0,0,1]
        M = eights.investigate.convert_list_of_list_to_sa(
            np.array(M),
            c_name=col_names)

        def test_equality(M, col_name, boundary):
            return M[col_name] == boundary

        res = eights.generate.where_all_are_true(
            M, 
            [test_equality, test_equality, test_equality], 
            ['heigh','weight', 'age'], 
            [1,2,3],
            ('eq_to_stuff',))
        ctrl = np.array(
            [(1, 2, 3, True), (2, 3, 4, False), (3, 4, 5, False)], 
            dtype=[('heigh', '<i8'), ('weight', '<i8'), ('age', '<i8'),
                   ('eq_to_stuff', '?')])
                   
        self.assertTrue(np.array_equal(res, ctrl))
        
if __name__ == '__main__':
    unittest.main()
