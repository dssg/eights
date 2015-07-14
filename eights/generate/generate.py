import numpy as np
from sklearn import cross_validation




def select_by_dist_from(M, target, threshold, lat_col_name, lng_col_name):
    ret = []
    for idx, x in enumerate(M):        
        if dist_less_than(target, GPS(x[lng_col_name], x[lat_col_name]),threshold):
            ret.append(idx)
    return ret



def where_all_are_true(M, lambdas, col_names, vals ):
    raise NotImplementedError
    to_select = vector_of_trues(M.rows)
    for lambd, col_name in zip(lambdas, col_names):
        to_select = np.logical_and(to_select, lambd(M[col_name]))
    return to_select

#where_all_are_true(
#    M, 
#    [(where_val_eq, 'f1', 4),
#     (where_val_between, 'f7', (1.2, 2.5)]))

def where_val_eq(M, col_name, boundary):
    return M[col_name] == boundary

def where_val_lt(M, col_name, boundary):
    return M[col_name] < boundary

def where_val_gt(M, col_name, boundary):
    return M[col_name] > boundary

def where_val_between(M, col_name, boundary):
    return np.and(boundary[0] <= M[col_name], M[col_name] <= boundary[1])

#def sweep_where_all_are_true(args, M):
#   changing_args = [arg[2] for arg in args]
#   runs = product(*changing_args)
#   for r in runs:
#      fixed_lambdas = [lambda col: lam(col, r[idx], args[idx][3]) for idx, lam in enumerate(lambdas)]
#      yield select_one(fixed_lambdas, args[idx][1], M)
#      
      

#
#M = np.array([ ('home', 40.761036, -73.977374),
#                  ('work', 45.5660930, -73.92599),
#                  ('fun', 40.702646, -74.013799)],
#                  dtype = [('name', 'S4'), ('lng', float), ('lat', float)]
#                )    
#                
#target = GPS(40.748784, -73.985429)
#threshold = .001
#
#lat_col_name = 'lat'
#lng_col_name = 'lng'
#
#M_id =  select_by_dist_from(M, target, threshold, lat_col_name, lng_col_name)
#
#
#import pdb; pdb.set_trace()
#
#
#
#
#def generate():
#    return
#

def generate_bin(col, number_of_bins):
    """Generates a column of categories, where each category is a bin.

    Parameters
    ----------
    col : np.array
    
    Returns
    -------
    np.array
    
    Examples
    --------
    >>> M = np.array([0.1, 3.0, 0.0, 1.2, 2.5, 1.7, 2])
    >>> generate_bin(M, 3)
    [0 2 0 1 2 1 2]

    """

    raise NotImplementedError
    
def normalize(col):
    """
    
    Generate a normalized column.
    
    Normalize both mean and std dev.
    
    Parameters
    ----------
    col : np.array
    
    Returns
    -------
    np.array
    
    """
    raise NotImplementedError    