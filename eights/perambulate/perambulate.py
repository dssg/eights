import inspect
import itertools as it
import numpy as np

from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SKSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import KFold, StratifiedKFold

from random import sample

(RF, SVC, DESC_TREE, ADA_BOOST,
 NO_SUBSET, LEAVE_ONE_COL_OUT, SWEEP_TRAINING_SIZE,
 NO_CV, K_FOLD, STRAT_ACTUAL_K_FOLD, STRAT_EVEN_K_FOLD) = range(11)

clf_ids =  (RF, SVC, DESC_TREE, ADA_BOOST)
subset_ids = (NO_SUBSET, LEAVE_ONE_COL_OUT, SWEEP_TRAINING_SIZE)
cv_ids = (NO_CV, K_FOLD, STRAT_ACTUAL_K_FOLD, STRAT_EVEN_K_FOLD)
    
def subset_sweep_training_size(y, subset_size, n_subsets=3):
    
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


    
subset_iters = {SWEEP_TRAINING_SIZE: subset_sweep_training_size}
#TODO others    

sk_learn_cvs = {K_FOLD: KFold,
                STRAT_ACTUAL_K_FOLD: StratifiedKFold,
                STRAT_EVEN_K_FOLD: StratifiedKFold}
#TODO others

sk_learn_clfs = {RF: RandomForestClassifier,
                 SVC: SKSVC,
                 DESC_TREE: DecisionTreeClassifier,
                 ADA_BOOST: AdaBoostClassifier}   

CLF_ID, CLF_PARAMS, SUBSET_ID, SUBSET_PARAMS, CV_ID, CV_PARAMS = range(6)
dimensions = (CLF_ID, CLF_PARAMS, SUBSET_ID, SUBSET_PARAMS, CV_ID, CV_PARAMS)

class Run(object):
    def __init__(
        self,
        clf,
        test_indices):
        self.clf = clf
        self.test_indices = test_indices

    def __repr__(self):
        return 'Run(clf={})'.format(
                self.clf)

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
        self.M = M
        self.y = y
        self.runs = None
        self.clf_id = clf_id
        self.clf_params = clf_params
        self.subset_id = subset_id
        self.subset_params = subset_params
        self.cv_id = cv_id
        self.cv_params = cv_params
        self.__by_dimension = {CLF_ID: self.clf_id,
                               CLF_PARAMS: self.clf_params,
                               SUBSET_ID: self.subset_id,
                               SUBSET_PARAMS: self.subset_params,
                               CV_ID: self.cv_id,
                               CV_PARAMS: self.cv_params}

    def __repr__(self):
        return ('Trial(clf_id={}, clf_params={}, subset_id={}, '
                'subset_params={}, cv_id={}, cv_params={})').format(
                        self.clf_id,
                        self.clf_params,
                        self.subset_id,
                        self.subset_params,
                        self.clf_id,
                        self.cv_params)

    def __getitem__(self, arg):
        return self.__by_dimension[arg]

    def has_run(self):
        return self.runs is not None

    def run(self):
        if self.has_run():
            return self.runs
        runs = []
        for subset in subset_iters[self.subset_id](self.y, **self.subset_params):
            runs_this_subset = []
            y_sub = self.y[subset]
            M_sub = self.M[subset]
            cv_cls = sk_learn_cvs[self.cv_id]
            cv_kwargs = self.cv_params
            expected_cv_kwargs = inspect.getargspec(cv_cls.__init__).args
            if 'n' in expected_cv_kwargs:
                cv_kwargs['n'] = y_sub.shape[0]
            if 'y' in expected_cv_kwargs:
                cv_kwargs['y'] = y_sub
            cv_inst = cv_cls(**cv_kwargs)
            for train, test in cv_inst:
                clf_inst = sk_learn_clfs[self.clf_id](**self.clf_params)
                clf_inst.fit(M_sub[train], y_sub[train])
                test_indices = subset[test]
                runs_this_subset.append(Run(clf_inst, test_indices))
            runs.append(runs_this_subset)    
        self.runs = runs
        return runs

    def average_score(self):
        self.run()
        M = self.M
        y = self.y
        return np.mean(
                [np.mean(
                    [run.clf.score(
                        M[run.test_indices], y[run.test_indices])
                     for run in subset])
                 for subset in self.runs])
    
class Experiment(object):
    def __init__(self, M, y, clfs={}, subsets={}, cvs={}):
        self.M = M
        self.y = y
        self.clfs = clfs
        self.subsets = subsets
        self.cvs = cvs
        self.trials = None

    def __repr__(self):
        print 'Experiment(clfs={}, subsets={}, cvs={})'.format(
                self.clfs, 
                self.subsets, 
                self.cvs)
        
    def __run_all_trials(self, trials):
        # TODO some multiprocess thing
        
        for trial in trials:
            trial.run()
            
    def __transpose_dict_of_lists(self, dol):
        # http://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
        return (dict(it.izip(dol, x)) for 
                x in it.product(*dol.itervalues()))

    def slice_on_dimension(self, dimension, value, trials=None):
        if trials is None:
            trials = self.run()
        return [trial for trial in trials if trial[dimension] == value]  

    def slice_by_best_score(self, dimension, trials=None):
        if trials is None:
            trials = self.run()
        categories = {}
        other_dims = list(dimensions)
        other_dims.remove(dimension)
        for trial in trials:
            key = [trial[dim] for dim in other_dims]
            try:
                categories[key].append(trial)
            except KeyError:
                categories[key] = [trial]
        result = []
        for key in categories:
            result.append(max(
                categories[key], 
                key=lambda trial: trial.average_score()))
        return result

    def has_run(self):
        return self.trials is not None

    def run(self):
        if self.has_run():
            return self.trials
        trials = []
        for clf_id in self.clfs:
            all_clf_ps = self.clfs[clf_id]
            for clf_params in self.__transpose_dict_of_lists(all_clf_ps):
                for subset_id in self.subsets:
                    all_sub_ps = self.subsets[subset_id]
                    for subset_params in self.__transpose_dict_of_lists(all_sub_ps):
                        for cv_id in self.cvs:
                            all_cv_ps = self.cvs[cv_id]
                            for cv_params in self.__transpose_dict_of_lists(all_cv_ps):
                                trial = Trial(
                                    M=self.M,
                                    y=self.y,
                                    clf_id=clf_id,
                                    clf_params=clf_params,
                                    subset_id=subset_id,
                                    subset_params=subset_params,
                                    cv_id=cv_id,
                                    cv_params=cv_params)
                                trials.append(trial)
        self.__run_all_trials(trials)
        self.trials = trials
        return trials

    def average_score(self):
        self.run()
        return [(trial, trial.average_score()) for trial in self.trials]

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

    

