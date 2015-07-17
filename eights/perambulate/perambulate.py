from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SKSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import KFold

from random import sample

(RF, SVC, DESC_TREE, ADA_BOOST,
NO_SUBSET, LEAVE_ONE_COL_OUT, SWEEP_TRAINING_SIZE,
NO_CV, K_FOLD, STRAT_ACTUAL_K_FOLD, STRAT_EVEN_K_FOLD) = range(12)

clf_ids =  (RF, SVC, DESC_TREE, ADA_BOOST)
subset_ids = (NO_SUBSET, LEAVE_ONE_COL_OUT, SWEEP_TRAINING_SIZE)
cv_ids = (NO_CV, K_FOLD, STRAT_ACTUAL_K_FOLD, STRAT_EVEN_K_FOLD)
    
sk_learn_clfs = {RF: RandomForestClassifier,
                 SVC: SKSVC,
                 DESC_TREE: DecisionTreeClassifier,
                 ADA_BOOST: AdaBoostClassifier}   
    
subset_iters = {SWEEP_TRAINING_SIZE: subset_sweep_training_size}
#TODO others    

sk_learn_cvs = {K_FOLD: KFOLD}
#TODO others
    
class Experiment(object):
    def __init__(self, clfs={}, subsets={}, cvs={}):
        self.clfs = clfs
        self.subsets = subsets
        self.cvs = cvs


class Trial(object):
    def __init__(
        self, 
        M,
        y,
        clf_id=RF,
        clf_params={},
        subset_id=NO_SUBSET,
        subset_params={},
        cv_id=NO_CV,
        cv_params={}):
        self.runs = None
        self.clf_id = clf_id
        self.clf_params = clf_params
        self.subset_id = subset_id
        self.subset_params = subset_params
        self.cv_id = cv_id
        self.cv_params = cv_params

def subset_sweep_training_size(y, subset_size, n_subsets):
    
    count = Counter(y)
    size_space = float(sum(count.values()))
    proportions = {key: count[key] / size_space for key in count}
    n_choices = {key: int(proportions[key] * subset_size) for 
                 key in proportions}
    indices = {key: np.where(y == key)[0] for key in count}
    for _ in xrange(n_subsets):
        samples = {key: sample(indices[key], n_choices[key]) for key in count}
        all_indices = np.sort(np.concatenate(samples.values()))
        yield all_indices


class Run(object):
    def __init__(
        self,
        clf,
        test_indices):
        self.clf = clf
        self.test_indices = test_indices

def run_trial(trail):
    
    runs = []
    for subset in subset_iters[trail.subset_id](**trial.subst_params):
        y_sub = trial.y[subset]
        M_sub = trial.M[subset]
        cv_cls = sk_learn_cvs[trial.cv_id]
        cv_kwargs = **trial.cv_param
        expected_cv_kwargs = inspect.getargspec(cv_cls.__init__).args
        if 'n' in expected_cv_kwargs:
            cv_kwargs['n'] = y_sub.shape[0]
        if 'y' in expected_cv_kwargs:
            cv_kwargs['y'] = y_sub
        cv_inst = cv_cls(**cv_kwargs)
        for train, test in cv_inst:
            clf_inst = sk_learn_clfs[trial.clf_id](**trial.clf_params)
            clf_inst.fit(M_sub[train], y_sub[train])
            test_indices = subset[test]
            runs.append(Run(clf_inst, test_indices))
            
    trial.runs = runs
    
def run_all_trials(trials):
    # TODO some multiprocess thing
    
    for trial in trials:
        run_trial(trial)
        
def trainspose_dict_of_lists(dol):
    # http://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    return (dict(izip(dol, x)) for 
            x in product(*dol.itervalues()))

def run_experiment(M, y, experiment):
    trial = []
    for clf_id in exeriment.clfs:
        all_clf_ps = experiment.clfs[clf_id]
        for clf_params in transpose_dict_of_lists(all_clf_ps):
            for subset_id in experiment.subsets:
                all_sub_ps = experiment.subsets[subset_id]
                for subset_params in transpose_dict_of_lists(all_sub_ps):
                    for cv_id in experiment.cvs:
                        all_cv_ps = experiment.cvs[cv_id]
                        for cv_params in transpose_dict_of_lists(all_cv_ps):
                            trial = Trial(
                                M-M,
                                y=y,
                                clf_id=clf_id,
                                clf_params=clf_params,
                                subset_id=subset_id,
                                subset_params=subset_params,
                                cv_id=cv_id
                                cv_params=cv_params)
                            trials.append(trial)
    run_all_trials(trials)
    return trials

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

    

