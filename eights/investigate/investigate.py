from investigate_helper import *
from ..communicate import *
from ..utils import is_sa
import numpy as np


from collections import Counter
from pylab import pcolor, show, colorbar, xticks, yticks, figure

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
    
    fig = figure()
    pcolor(cc)
    colorbar()
    yticks(np.arange(0.5, M.shape[1] + 0.5), range(0, M.shape[1]))
    xticks(np.arange(0.5, M.shape[1] + 0.5), range(0, M.shape[1]))
    return fig
    #set rowvar =0 for rows are items, cols are features
    raise NotImplementedError
    
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
    raise NotImplementedError
    
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
