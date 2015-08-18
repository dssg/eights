import unittest
import numpy as np
from numpy.random import rand
import eights.investigate
import eights.utils
import eights.generate
import eights.generate.generate_helper as gh
import eights.generate as gen


class TestGenerate(unittest.TestCase):
    def test_where_all_are_true(self):
        M = [[1,2,3], [2,3,4], [3,4,5]]
        col_names = ['heigh','weight', 'age']
        lables= [0,0,1]
        M = eights.utils.cast_list_of_list_to_sa(
            M,
            names=col_names)

        arguments = [{'func': gen.val_eq, 'col_name': 'heigh', 'vals': 1},
                     {'func': gen.val_lt, 'col_name': 'weight', 'vals': 3},
                     {'func': gen.val_between, 'col_name': 'age', 'vals': 
                      (3, 4)}]

        res = eights.generate.where_all_are_true(
            M, 
            arguments,
            'eq_to_stuff')
        ctrl = np.array(
            [(1, 2, 3, True), (2, 3, 4, False), (3, 4, 5, False)], 
            dtype=[('heigh', '<i8'), ('weight', '<i8'), ('age', '<i8'),
                   ('eq_to_stuff', '?')])
                   
        self.assertTrue(np.array_equal(res, ctrl))

    def test_combine_cols(self):
        M = np.array(
                [(0, 1, 2), (3, 4, 5), (6, 7, 8)], 
                dtype=[('f0', float), ('f1', float), ('f2', float)])
        ctrl = np.array(
                [(0, 1, 2, 1, 1.5), (3, 4, 5, 7, 4.5), (6, 7, 8, 13, 7.5)], 
                dtype=[('f0', float), ('f1', float), ('f2', float), 
                       ('sum', float), ('avg', float)])
        M = gen.combine_cols(M, gen.combine_sum, ('f0', 'f1'), 'sum')
        M = gen.combine_cols(M, gen.combine_mean, ('f1', 'f2'), 'avg')
        self.assertTrue(np.array_equal(M, ctrl))
        
if __name__ == '__main__':
    unittest.main()
