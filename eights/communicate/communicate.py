import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import boxplot 

from sklearn.metrics import roc_curve
from ..perambulate import Experiment
from communicate_helper import *

def generate_report(info):
    raise NotImplementedError

def print_matrix_row_col(M, row_labels, col_labels):
    row_format ="{:>15}" * (len(col_labels) + 1)
    print row_format.format("", *col_labels)
    for row_name, row in zip(row_labels, M):
        print row_format.format(row_name, *row)

def print_crosstab_dict(a_dict):
    K_1 = a_dict.keys()
    K_2 = np.unique([item for sublist in [list(x.elements()) for 
                                          x in a_dict.values()] 
                     for item in sublist])
    #Here for simplicity REMOVE
    M = np.zeros(shape=(len(K_1),len(K_2)))
    for idx, x in enumerate(K_1):
        for idy, y in enumerate(K_2):
            M[idx,idy] = a_dict[x][y]
    row_format ="{:>15}" * (len(K_2) + 1)
    print row_format.format("", *K_2)
    for team, row in zip(K_1, M):
        print row_format.format(team, *row)
    
def print_describe_all(a_dict):
    row_labels = a_dict.keys()
    rows = a_dict.values()
    row_format ="{:>15}" * (2)
    print row_format.format("", *a_dict.keys())
    for row_label, row in zip(row_labels, rows):
        print row_format.format(row_label, row)
    
def plot_simple_histogram(col, verbose=True):
    hist, bins = np.histogram(col, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    f = figure()
    plt.bar(center, hist, align='center', width=width)
    if verbose:
        plt.show()
    return f

# all of the below take output from any func in perambulate or operate



def plot_roc(labels, score, title='roc', verbose=True):
    fpr, tpr = roc_curve(labels, score)
    n_entries = fpr.shape[0] 
    X = (np.arange(n_entries) + 1) / float(n_entries)
    fig = plt.figure()
    plt.plot(X, fpr, X, tpr)
    plt.legend(['False Positive Rate', 'True Positive Rate'], 'upper left')
    plt.xlabel('% Selected as True')
    plt.ylabel('Rate')
    plt.title(title)
    if verbose:
        fig.show()
    return fig

def plot_box_plot(col, col_name, verbose=True):
    """Makes a box plot for a feature
    comment
    
    Parameters
    ----------
    col : np.array
    
    Returns
    -------
    matplotlib.figure.Figure
    
    """

    fig = boxplot(col)
    #add col_name to graphn
    if verbose:
        show()
    return fig

def plot_prec_recall(labels, score, verbose=True):
    raise NotImplementedError

def get_top_features(clf, n):
    raise NotImplementedError

# TODO features form top % of clfs

def get_roc_auc(labels, score):
    raise NotImplementedError

def plot_correlation_matrix(M, verbose=True):
    """Plot correlation between variables in M
    
    Parameters
    ----------
    M : numpy structured array
    Returns
    -------
    matplotlib.figure.Figure
    
    """
    # http://glowingpython.blogspot.com/2012/10/visualizing-correlation-matrices.html
    # TODO work on structured arrays or not
    # TODO ticks are col names
    if is_sa(M):
        names = m.dtype.names
        M = cast_np_sa_to_nd(M)
    else: 
        if is_nd(M):
            n_cols = M.shape[1]
        else: # list of arrays
            n_cols = len(M)
        names = ['f{}'.format(i) for i in xrange(n_cols)]
    
    #set rowvar =0 for rows are items, cols are features
    cc = np.corrcoef(M, rowvar=0)
    
    fig = plt.figure()
    plt.pcolor(cc)
    plt.colorbar()
    plt.yticks(np.arange(0.5, M.shape[1] + 0.5), range(0, M.shape[1]))
    plt.xticks(np.arange(0.5, M.shape[1] + 0.5), range(0, M.shape[1]))
    if verbose:
        plt.show()
    return fig

def plot_correlation_scatter_plot(M, verbose=True):
    """Makes a grid of scatter plots representing relationship between variables
    
    Each scatter plot is one variable plotted against another variable
    
    Parameters
    ----------
    M : numpy structured array
    
    Returns
    -------
    matplotlib.figure.Figure
    
    """
    # TODO work for all three types that M might be
    # TODO ignore classification variables
    # adapted from the excellent 
    # http://stackoverflow.com/questions/7941207/is-there-a-function-to-make-scatterplot-matrices-in-matplotlib
    
    if is_sa(M):
        names = M.dtype.names
        M = cast_np_sa_to_nd(M)
    else:
        names = ['f{}'.format(i) for i in xrange(M.shape[1])]    

    numdata, numvars = M.shape
    fig, axes = plt.subplots(numvars, numvars)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the M.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]: 
            axes[x,y].plot(M[x], M[y], '.')

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), it.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)
    if verbose:
        plt.show()
    return fig

def plot_kernel_density(col, n=None, missing_val=np.nan, verbose=True): 

    x_grid = np.linspace(min(col), max(col), 1000)

    grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1,1.0,30)}, cv=20) # 20-fold cross-validation
    grid.fit(col[:, None])

    kde = grid.best_estimator_
    pdf = np.exp(kde.score_samples(x_grid[:, None]))

    fig, ax = plt.subplots()
    #fig = plt.figure()
    ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
    ax.hist(col, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
    ax.legend(loc='upper left')
    ax.set_xlim(min(col), max(col))
    if verbose:
        plt.show()
    return fig

def plot_on_map(lat_col, lng_col):
    """Plots points on a map
    
    Parameters
    ----------
    lat_col : np.array
    lng_col : np.array
    
    Returns
    -------
    matplotlib.figure.Figure
    
    """
    raise NotImplementedError

def plot_on_timeline(col):
    """Plots points on a timeline
    
    Parameters
    ----------
    col : np.array
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError

def feature_pairs_in_rf(rf, weight_by_depth=None, verbose=True):
    """Describes the frequency of features appearing subsequently in each tree
    in a random forest"""
    # weight be depth is a vector. The 0th entry is the weight of being at
    # depth 0; the 1st entry is the weight of being at depth 1, etc.
    # If not provided, weights are linear with negative depth. If 
    # the provided vector is not as long as the number of depths, then 
    # remaining depths are weighted with 0


    pairs_by_est = [feature_pairs_in_tree(est) for est in rf.estimators_]
    pairs_by_depth = [list(it.chain(*pair_list)) for pair_list in 
                      list(it.izip_longest(*pairs_by_est, fillvalue=[]))]
    pairs_flat = list(it.chain(*pairs_by_depth))
    depths_by_pair = {}
    for depth, pairs in enumerate(pairs_by_depth):
        for pair in pairs:
            try:
                depths_by_pair[pair] += [depth]
            except KeyError:
                depths_by_pair[pair] = [depth]
    counts_by_pair=Counter(pairs_flat)
    count_pairs_by_depth = [Counter(pairs) for pairs in pairs_by_depth]

    depth_len = len(pairs_by_depth)
    if weight_by_depth is None:
        weight_by_depth = [(depth_len - float(depth)) / max_depth for depth in
                           xrange(depth_len)]
    weight_filler = it.repeat(0.0, depth_len - len(weight_by_depth))
    weights = list(it.chain(weight_by_depth, weight_filler))
    
    average_depth_by_pair = {pair: float(sum(depths)) / len(depths) for 
                             pair, depths in depths_by_pair.iteritems()}

    weighted = {pair: sum([weights[depth] for depth in depths])
                for pair, depths in depths_by_pair.iteritems()}

    if verbose:
        print '=' * 80
        print 'RF Subsequent Pair Analysis'
        print '=' * 80
        print
        _feature_pair_report(
                counts_by_pair.most_common(), 
                'Overall Occurrences', 
                'occurrences')
        _feature_pair_report(
                sorted([item for item in average_depth_by_pair.iteritems()], 
                       key=lambda item: item[1]),
                'Average depth',
                'average depth',
                'Max depth was {}'.format(depth_len - 1))
        _feature_pair_report(
                sorted([item for item in weighted.iteritems()], 
                       key=lambda item: item[1]),
                'Occurrences weighted by depth',
                'sum weight',
                'Weights for depth 0, 1, 2, ... were: {}'.format(weights))

        for depth, pairs in enumerate(count_pairs_by_depth):
            _feature_pair_report(
                    pairs.most_common(), 
                    'Occurrences at depth {}'.format(depth), 
                    'occurrences')


    return (counts_by_pair, count_pairs_by_depth, average_depth_by_pair, 
            weighted)

class Report(object):
    def __init__(self, exp):
        self.__back_indices = {trial, i for i, trial in enumerate(exp.trials)}
        self.__objects = []

    def add_summary_graph(self, measure):
        #TODO get measures, map them to indices, make a figure
        pass

