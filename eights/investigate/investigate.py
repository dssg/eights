from investigate_helper import *
from ..communicate import *
from ..utils import is_sa
import numpy as np

import sklearn

from collections import Counter
import matplotlib.pyplot as plt

from sklearn import cross_validation


def simple_CV(M, labels, clf, clf_params={},
              cv=sklearn.cross_validation.KFold, cv_parms={}):
    
    exp = Experiment(
        M, 
        labels, 
        clfs={clf: clf_params},
        cvs={cv: sv_params})
    runs = exp.run()[0][0]
    
    scores = [(run.clf.score(M[run.test_indices], labels[run.test_indices]) 
                for run in runs)]
    return scores



def convert_list_of_list_to_sa(M, c_name=None):
    return cast_np_nd_to_sa(M, names=c_name)
        
        
def open_csv(file_loc):
    f = open_csv_as_structured_array(file_loc)
    return set_structured_array_datetime_as_day(f,file_loc)
    
def open_JSON():
    return

#summary Statistics
def describe_col(col):
    return describe_column(col)

def describe_all(M):
    if is_sa(M):
        #then its a structured array
        return [describe_column(M[x]) for x in M.dtype.names]
    else:
        #then its a list of np.arrays
        return [describe_column(M[x]) for x in M]
        
def histogram(L, n=None): 
    if n is None:
        n = len(L)
    return Counter(L).most_common(n)
    
def print_crosstab(L_1, L_2):
    #assume np.structured arrays?
    crosstab_dict = crosstab(L_1, L_2)
    print_crosstab_dict(crosstab_dict)
    return crosstab_dict


def plot_box_plot(col):
    """Makes a box plot for a feature
    comment
    
    Parameters
    ----------
    col : np.array
    
    Returns
    -------
    matplotlib.figure.Figure
    
    """
    raise NotImplementedError
    
def plot_correlation_matrix(M):
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
    
    cc = np.corrcoef(M, rowvar=0)
    
    fig = plt.figure()
    plt.pcolor(cc)
    plt.colorbar()
    plt.yticks(np.arange(0.5, M.shape[1] + 0.5), range(0, M.shape[1]))
    plt.xticks(np.arange(0.5, M.shape[1] + 0.5), range(0, M.shape[1]))
    return fig
    #set rowvar =0 for rows are items, cols are features
    
def plot_correlation_scatter_plot(M):
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

    numvars, numdata = M.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars)
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
            axes[x,y].plot(M[x], M[y], **kwargs)

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    return fig
    
def plot_histogram(col, missing_val=np.nan):
    """Plot histogram of variables in col
    
    Includes a bar represented missing entries.
    
    Does a really good job showing the variation in the variable.
    
    Parameters
    ----------
    col : np.array
        Column to plot
    missing_val : ?
        Value representing a missing entry
    
    Returns
    -------
    matplotlib.figure.Figure
    
    """
    raise NotImplementedError
    
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
