import numpy as np
from sklearn import cross_validation




def select_by_dist_from(data, target, threshold, lat_col_name, lng_col_name):
    ret = []
    for idx, x in enumerate(data):        
        if dist_less_than(target, GPS(x[lng_col_name], x[lat_col_name]),threshold):
            ret.append(idx)
    return ret



def where_all_are_true(lambdas, col_names, data):
   to_select = vector_of_trues(data.rows)
   for lambd, col_name in zip(lambdas, col_names):
       to_select = np.logical_and(to_select, lambd(data[col_name]))
   return to_select



#def sweep_where_all_are_true(args, data):
#   changing_args = [arg[2] for arg in args]
#   runs = product(*changing_args)
#   for r in runs:
#      fixed_lambdas = [lambda col: lam(col, r[idx], args[idx][3]) for idx, lam in enumerate(lambdas)]
#      yield select_one(fixed_lambdas, args[idx][1], data)
#      
      

#
#data = np.array([ ('home', 40.761036, -73.977374),
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
#data_id =  select_by_dist_from(data, target, threshold, lat_col_name, lng_col_name)
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
    >>> data = np.array([0.1, 3.0, 0.0, 1.2, 2.5, 1.7, 2])
    >>> generate_bin(data, 3)
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