import csv
import urllib2

import numpy as np
import sklearn.datasets

from eights.investigate import (cast_np_nd_to_sa, describe_cols,)
from eights.communicate import (plot_correlation_scatter_plot,
                               plot_correlation_matrix, 
                               plot_kernel_density,
                               plot_box_plot)




url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
response = urllib2.urlopen(url)
cr = csv.reader(response, delimiter=';')

data = list(cr)
col_names = data[0]
labels = np.array([x[-1] for x in data[1:]])
M = [x[:-1] for x in data[1:]]
M = cast_np_nd_to_sa(np.array(M,dtype='float'))

if True:
    for x in describe_cols(M):
        print x

if True:
   plot_correlation_scatter_plot(M) 
   plot_correlation_matrix(M)
   plot_kernel_density(M['f0']) #no designation of col name
   plot_box_plot(M['f0']) #no designation of col name


import pdb; pdb.set_trace()