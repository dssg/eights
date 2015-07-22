from investigate_helper import *
from ..communicate import *
from ..utils import is_sa
import numpy as np



from collections import Counter
import matplotlib.pyplot as plt

from sklearn import cross_validation


def simple_CV(M, labels, clf, clf_params={},
              cv=cross_validation.KFold, cv_parms={}):
    """This is simple execution a clf in our module.  
    Parameters
    ----------
    M : Structured array
       The matrix you wish to use for training and testing 
    labels : a one dimenional nd array
       This these are the labels that are assigned to the rows in the matrix M.
    clf : Sklearn Class object
        This is the type of algorithim you would use. 
    clf_params : a dictionary of parameters to assign to your clf
        The appropriate paramterts to asign to the clf, empty dict if none.
    cv : sklearn cv 
        kfold if default
    cv_parms : dict of paramters to apply to the cv
        empty if default
           
    Returns
    -------
    temp : list
       the list of trained models
    """
    exp = Experiment(
        M, 
        labels, 
        clfs={clf: clf_params},
        cvs={cv: cv_parms})
    runs = exp.run()
    
    scores = [run.clf.score(M[run.test_indices], labels[run.test_indices]) 
                for run in runs]
    return scores


def convert_to_sa(M, c_name=None):
    """Converts an list of lists or a np ndarray to a Structured Arrray
    Parameters
    ----------
    M  : List of List or np.ndarray
       This is the Matrix M, that it is assumed is the basis for the ML algorithm 
    
    Attributes
    ----------
    temp : type
       Description 
       
    Returns
    -------
    temp : Numpy Structured array
       This is the matrix of an appropriate type that eights expects.
       
    """
    if isinstance(M, type(np.array([1]))):
        return cast_np_nd_to_sa(M, names=c_name)
    
    elif isinstance(M, list) and isinstance(M[0], list): #good idea or bad?
        return cast_list_of_list_to_sa(M, names=c_name)
    
    elif is_sa(M):
        return M
        
    else: 
        raise TypeError # approrpriate?
        
        
def open_csv(file_loc):
    """single line description
    Parameters
    ----------
    temp : type
       Description 
    
    Attributes
    ----------
    temp : type
       Description 
       
    Returns
    -------
    temp : type
       Description
       
    """
    f = open_csv_as_structured_array(file_loc)
    return set_structured_array_datetime_as_day(f,file_loc)
    
def open_JSON():
    """single line description
    Parameters
    ----------
    temp : type
       Description 
    
    Attributes
    ----------
    temp : type
       Description 
       
    Returns
    -------
    temp : type
       Description
       
    """
    raise NotImplementedError

def open_SQL():
    raise NotImplementedError
    
    

#summary Statistics
# Silly to have two.  I should have only made one.
def describe_cols(M):
    """takes a SA or list of Np.rayas and returns the summary statistcs
    Parameters
    ----------
    M : Structured Array or list of Numpy ND arays.
       Description 
       
    Returns
    -------
    temp : type
       Description
       
    """
    #remove the [] if only one?
    if is_sa(M):
        #then its a structured array
        return [describe_column(M[x]) for x in M.dtype.names]
    else:
        #then its a list of np.arrays
        return [describe_column(M[x]) for x in M]
        
def histogram(L, n=None): 
    """ returns a count of elements on the numpy array or a list 
    Parameters
    ----------
    temp : list or np.ndarray
       Description 
    
    Returns
    -------
    temp : list of tuples 
       first element is the value, the second number is the count
       
    """
    if n is None:
        n = len(L)
    return Counter(L).most_common(n)
    
def print_crosstab(L_1, L_2):
    """this prints a crosstab results
    Parameters
    ----------
    temp : type
       Description 
    
    Returns
    -------
    temp : type
       Description
    """
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
