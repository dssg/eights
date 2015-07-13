import unittest
import numpy as np
import utils_for_tests as utils
utils.add_to_python_path()

from eights.investigate.investigate import open_csv

class TestInvestigate(unittest.TestCase):
    def test_open_csv(self):
        csv_file = utils.path_of_data("mixed.csv")
        correct = np.array([(0, 'Jim', 5.6), (1, 'Jill', 5.5)],
                            dtype=[('id', '<i8'), ('name', 'S4'), ('height', '<f8')])

        self.assertTrue(np.array_equal(open_csv(csv_file),correct))

        
#validate_time        


if __name__ == '__main__':
    unittest.main()


