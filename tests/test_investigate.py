import unittest
import numpy as np
import utils_for_tests as utils
from numpy.random import rand
utils.add_to_python_path()

from eights.investigate.investigate import open_csv, describe_col
from eights.investigate.investigate import plot_correlation_matrix
from eights.investigate.investigate import plot_correlation_scatter_plot

class TestInvestigate(unittest.TestCase):
    def test_open_csv(self):
        csv_file = utils.path_of_data("mixed.csv")
        correct = np.array([(0, 'Jim', 5.6), (1, 'Jill', 5.5)],
                            dtype=[('id', '<i8'), ('name', 'S4'), ('height', '<f8')])

        self.assertTrue(np.array_equal(open_csv(csv_file),correct))
    def test_describe_column(self):
        test = np.array([1,2,3,4,5,6])
        correct = {'Maximal:': 6, 'Standard Dev:': 1.707825127659933, 'Count:': 6, 'Mean:': 3.5, 'Minimal ': 1}
        self.assertTrue(np.array_equal(describe_col(test),correct))
        
    def test_plot_correlation_matrix(self):
        data = rand(100, 10)
        fig = plot_correlation_matrix(data)
        
    def test_plot_correlation_scatter_plot(self):
        data = rand(100, 10)
        fig = plot_correlation_scatter_plot(data)
        fig.show()
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    unittest.main()


