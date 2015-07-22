import numpy as np
import matplotlib.pyplot as plt
from ..perambulate import Experiment


def print_matrix_row_col(M, L_1, L_2,):
    row_format ="{:>15}" * (len(L_2) + 1)
    print row_format.format("", *L_2)
    for team, row in zip(L_1, M):
        print row_format.format(team, *row)
    return None

def print_crosstab_dict(a_dict):
    K_1 = a_dict.keys()
    K_2 = np.unique([item for sublist in [list(x.elements()) for x in a_dict.values()] for item in sublist])
    #Here for simplicity REMOVE
    M = np.zeros(shape=(len(K_1),len(K_2)))
    for idx, x in enumerate(K_1):
        for idy, y in enumerate(K_2):
            M[idx,idy] = a_dict[x][y]
    row_format ="{:>15}" * (len(K_2) + 1)
    print row_format.format("", *K_2)
    for team, row in zip(K_1, M):
        print row_format.format(team, *row)
    return None
    
def print_describe_all(a_dict):
    K_1 = a_dict.keys()
    K_2 = a_dict.values()
    row_format ="{:>15}" * (2)
    print row_format.format("", *a_dict.keys())
    for team, row in zip(K_1, K_2):
        print row_format.format(team, row)
    return None
    

def plot_simple_histogram(x):
    hist, bins = np.histogram(x, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


# all of the below take output from any func in perambulate or operate

def generate_report(info):
    raise NotImplementedError

def plot_roc(info):
    raise NotImplementedError

def plot_prec_recall(info):
    raise NotImplementedError

def get_top_features(info):
    raise NotImplementedError

def get_roc_auc(info):
    raise NotImplementedError

