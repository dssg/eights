class Experiment(object):
    def __init__(self, list_of_plots, list_of_clfs_with_parms, list_of_of_subsets_with_parms, list_of_cv_with_parms):
        self.list_of_plots = list_of_cv
        self.list_of_clfs_with_parms = list_of_clfs_with_parms
        self.list_of_of_subsets_with_parms = list_of_of_subsets_with_parms
        self.list_of_cv = list_of_cv
    
runOne = Experiment(['roc','acc'],
                    [('random forest',['RF PARMS']), ('svm',['SVM PARMS'])],
                    [('leave one out col', ['PARMS']), ('sweep training size',['PARMS'])],    #parms for this
                    [('cv', ['parms']), ('stratified cv', ['parms'])]
                    )


def sliding_window(l,w,tst_w):
    ret = []
    for idx,_ in enumerate(l):
        if idx+w+tst_w > len(l): 
            break
        train =[l[idx+x] for x in range(w)]
        test = [l[idx+w+x] for x in range(tst_w)]
        ret.append((train,test))
    return ret
    
def expanding_window(l,w,tst_w):
    ret = []
    
    for idx, i in enumerate(l):
        if idx+w+tst_w > len(l): break
        
        train =[l[x] for x in range(idx+w)]
        test = []
        for x in range(tst_w):
            test.append(l[idx+w+x])
        ret.append((train,test))
    return ret

def sweep_exclude_columns(M, cols_to_exclude=None):
    """
    
    Analyze feature importance when each of a specified set of columns are
    excluded. 
    
    Parameters
    ----------
    M : Numpy structured array
    cols_to_exclude : list of str or None
         List of names of columns to exclude one at a time. If None, tries
         all columns
         
    Returns
    -------
    Numpy Structured array
        First col
            Excluded col name
        Second col
            Accuracy score
        Third col
            Feature importances
    """
    # not providing cv because it's always Kfold
    # returns fitted classifers along a bunch of metadata
    #Why don't we use this as buildinger for slices. AKA the way the cl
    # 
    raise NotImplementedError

#sweep calls random


    
def sweep_leave_one_col_out_(list_of_values):
    #returns list of list eachone missing a value in order.  
    #needs to be tested
    d={}
    for x in list_of_values:
       d[x+' is left out']=list_of_values.remove(x) 
    raise NotImplementedError


def run_slice_with_CV(M,Y,clf,  slice_dict, CV_TYPE = 'stratified'):
    for x in slick_dict:
        train, test = run_cv()
        clf.train
    raise NotImplementedError
    

def sweep_vary_training_set_size(Y, size, distribution = 'even'):
    # returns dictionary of indices.
    #runs 
    #random_subset_of_rows_even_distribution,
    #random_subset_of_rows_actual_distribution
    raise NotImplementedError
    
def sweep_vary_stratification(M, Y, positive_percents):
    # returns dictionary of indices.
    raise NotImplementedError
        
def stratified_cv(M, y, positive_percents):
    # returns dictionary of indices.
    # train on different, artificial stratifications; test on stratification
    # reflecting perct in actual data set
    raise NotImplementedError



def random_subset_of_columns(list_of_values, number_to_select):
    #np.rand(x, y)
    #handle id's as well as names
    raise NotImplementedError
    
def random_subset_of_rows_even_distribution(Y, list_of_values, number_to_select):
    #np.rand(x, y)
    #handle id's as well as names
    raise NotImplementedError

def random_subset_of_rows_actual_distribution(list_of_values, number_to_select):
    #np.rand(x, y)
    #handle id's as well as names
    raise NotImplementedError


def run_std_classifiers(M_train, M_test, y_train, y_test, report_file):
    raise NotImplementedError    

    

