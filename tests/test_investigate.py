import unittest
import numpy as np
import utils_for_tests as utils
from numpy.random import rand
utils.add_to_python_path()

from sklearn.neighbors import KernelDensity


from eights.investigate.investigate import open_csv, describe_cols
from eights.investigate.investigate import plot_correlation_matrix
from eights.investigate.investigate import plot_correlation_scatter_plot
from eights.investigate.investigate import convert_to_sa
from eights.investigate.investigate import plot_histogram

class TestInvestigate(unittest.TestCase):
    def test_open_csv(self):
        csv_file = utils.path_of_data("mixed.csv")
        correct = np.array([(0, 'Jim', 5.6), (1, 'Jill', 5.5)],
                            dtype=[('id', '<i8'), ('name', 'S4'), ('height', '<f8')])

        self.assertTrue(np.array_equal(open_csv(csv_file),correct))
    def test_describe_cold(self):
        test = np.array([1,2,3,4,5,6])
        correct = {'Maximal:': 6, 'Standard Dev:': 1.707825127659933, 'Count:': 6, 'Mean:': 3.5, 'Minimal ': 1}
        self.assertTrue(np.array_equal(describe_cols(test),correct))
        
    def test_plot_correlation_matrix(self):
        data = rand(100, 10)
        fig = plot_correlation_matrix(data)
        
    def test_plot_correlation_scatter_plot(self):
        data = rand(100, 10)
        #fig = plot_correlation_scatter_plot(data) ##  
        #fig.show()
    
    def test_convert_list_of_list_to_sa(self):
        test = [[1,2.,'a'],[2,4.,'b'],[4,5.,'g']]
        names = ['ints','floats','strings']
        test_1 = convert_to_sa(test)
        test_2 = convert_to_sa(test, names)
        correct_1 = 0
        correct_2 = 0
        
    def test_plot_histogram(self):
        test = np.array([[ 0.94888426],[ 1.00435848],[ 0.13563403],[-0.72318153],[ 1.3204944 ],[-1.5182872 ],
        [ 5.05387508],[ 5.26197651],[ 5.13953576],[ 5.6487067 ],[ 5.1851657 ],[ 5.74520732],[ 4.47001378],
        [ 5.09516057],[ 5.08668904],[ 5.48892083],[ 4.14439814],[ 4.58574639],[ 4.3827732 ],[ 5.03868177]])
        
        correct = np.array([-14.39823826,  -6.5120383 ,  -3.22551564,  -2.59372907,
                -2.77393719,  -2.94132193,  -1.15432552,  -2.90292248,
                -8.77546669, -19.20631506])
        #self.assertTrue(np.array_equal(plot_histogram(test),correct)) 
        from scipy.stats.distributions import norm
        np.random.seed(0)
        x = np.concatenate([norm(-1, 1.).rvs(400),norm(1, 0.3).rvs(100)])
        x_grid = np.linspace(-4.5, 3.5, 1000)
        from sklearn.grid_search import GridSearchCV
        grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1,1.0,30)}, cv=20) # 20-fold cross-validation
        grid.fit(x[:, None])
        print grid.best_params_


        kde = grid.best_estimator_
        pdf = np.exp(kde.score_samples(x_grid[:, None]))
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
        ax.hist(x, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
        ax.legend(loc='upper left')
        ax.set_xlim(-4.5, 3.5)
        plt.show()
        import pdb; pdb.set_trace()
        
        #
        #fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
        #fig.subplots_adjust(hspace=0.05, wspace=0.05)
        #ax[0, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
        #ax[0, 0].text(-3.5, 0.31, "Gaussian Kernel Density")
        #plt.show()xz
        
if __name__ == '__main__':
    unittest.main()


