# from sklearn tutorial

from eights.perambulate.perambulate import run_experiment, Experiment

class TestSKLearnParity(unittest.TestCase):

    def test_sklearn_parity(self):
        iris = datasets.load_iris()

        y = iris.target
        M = iris.data
        # Converts 2-dimensional homogeneous array to structured array
        M = cast_np_nd_to_sa(M)

	exp = Experiment(...)

	

