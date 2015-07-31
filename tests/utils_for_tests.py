import os
import sys
import itertools as it
import numpy as np
import string
import eights.utils
from numpy.random import rand

TESTS_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
DATA_PATH = os.path.join(TESTS_PATH, 'data')
EIGHTS_PATH = os.path.join(TESTS_PATH, '..')

def path_of_data(filename):
    return os.path.join(DATA_PATH, filename)

def generate_test_matrix(rows, cols, n_classes=2, types=[], random_state=None):
    full_types = list(it.chain(types, it.repeat(float, cols - len(types))))
    np.random.seed(random_state)
    cols = []
    for col_type in full_types:
        if col_type is int:
            col = np.random.randint(100, size=rows)
        elif issubclass(col_type, basestring):
            col = np.random.choice(list(string.uppercase), size=rows)
        else:
            col = np.random.random(size=rows)
        cols.append(col)
    labels = np.random.randint(n_classes, size=rows)
    M = eights.utils.sa_from_cols(cols)
    return M, labels

def generate_correlated_test_matrix(n_rows):
    M = rand(n_rows, 1)
    y = rand(n_rows) < M[:,0]
    return M, y

