import unittest
import numpy as np
from numpy.random import rand

from eights.investigate.investigate import open_csv, describe_cols, open_csv_list,convert_list_to_structured_array_wrap, print_crosstab

import utils_for_tests as utils

from collections import Counter
class TestInvestigate(unittest.TestCase):
    #1
    def test_open_csv_list(self):
        csv_file = utils.path_of_data("mixed.csv")
        correct = np.array([(0, 'Jim', 5.6), (1, 'Jill', 5.5)],
                            dtype=[('id', '<i8'), ('name', 'S4'), ('height', '<f8')])
        self.assertTrue(np.array_equal(open_csv_list(csv_file),correct))
    #2    
    def test_open_csv(self):
        csv_file = utils.path_of_data("mixed.csv")
        correct = np.array([(0, 'Jim', 5.6), (1, 'Jill', 5.5)],dtype=[('id', '<i8'), ('name', 'S4'), ('height', '<f8')])
        self.assertTrue(np.array_equal(open_csv(csv_file),correct))
    #3    
    def test_describe_cols(self):
        test = np.array([[1],[2],[3],[4],[5],[6]])
        test_list = np.array([1,2,3,4,5,6])
        
        test_sa =np.array([(1,), (2,),(3,), (4,),(5,),(6,)],dtype=[('id', '<i8')])
        correct = [{'Maximal:': 6, 'Standard Dev:': 1.707825127659933, 'Count:': 6, 'Mean:': 3.5, 'Minimal ': 1}]
        self.assertTrue(np.array_equal(describe_cols(test),correct))
        self.assertTrue(np.array_equal(describe_cols(test_list),correct))
        self.assertTrue(np.array_equal(describe_cols(test_sa),correct))           
    #4
    def test_convert_list_to_structured_array_wrap(self):
        test = [[1,2.,'a'],[2,4.,'b'],[4,5.,'g']]
        names = ['ints','floats','strings']
        correct_1 = np.array([(1, 2.0, 'a'), (2, 4.0, 'b'), (4, 5.0, 'g')],dtype=[('f0', '<i8'), ('f1', '<f8'), ('f2', 'S1')])
        correct_2 = np.array([(1, 2.0, 'a'), (2, 4.0, 'b'), (4, 5.0, 'g')], dtype=[('ints', '<i8'), ('floats', '<f8'), ('strings', 'S1')])
        self.assertTrue(np.array_equal(correct_1, convert_list_to_structured_array_wrap(test)))
        self.assertTrue(np.array_equal(correct_2, convert_list_to_structured_array_wrap(test, names)))
    #5
    def test_print_crosstab(self):
        l1= [1,2,3,3,2,1]
        l2= [1,2,3,3,2,1]
        correct = {1: Counter({1: 2}), 2: Counter({2: 2}), 3: Counter({3: 2})}
        self.assertTrue(np.array_equal(correct, print_crosstab(l1,l2,False)))


        
if __name__ == '__main__':
    unittest.main()


