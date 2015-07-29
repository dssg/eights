import abc
import copy
import inspect
import numpy as np
import itertools as it
from collections import Counter
from random import sample
from sklearn.cross_validation import _PartitionIterator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

class _BaseSubsetIter(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, y):
        self._y = y
    
    @abc.abstractmethod
    def __iter__(self):
        yield np.array([], dtype=int)

    @abc.abstractmethod
    def __repr__(self):
        return ''

class SubsetNoSubset(_BaseSubsetIter):
    def __iter__(self):
        yield np.arange(self._y.shape[0])

    def __repr__(self):
        return 'SubsetNoSubset()'

class SubsetSweepTrainingSize(_BaseSubsetIter):
        
    def __init__(self, y, subset_size, n_subsets=3):
        super(SubsetSweepTrainingSize, self).__init__(y)
        self.__subset_size = subset_size
        self.__n_subsets = n_subsets

    def __iter__(self):
        y = self._y
        subset_size = self.__subset_size
        n_subsets = self.__n_subsets
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

    def __repr__(self):
        return 'SubsetSweepTrainingSize(subset_size={}, n_subsets={})'.format(
                self.__subset_size,
                self.__n_subsets)

class SubsetSweepExcludeColumns(_BaseSubsetIter):
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
    def __init__(self, M, cols_to_exclude=None):
        raise NotImplementedError

class SubsetSweepLeaveOneColOut(_BaseSubsetIter):
    # TODO
    #returns list of list eachone missing a value in order.  
    #needs to be tested
    pass

class SubsetSweepVaryStratification(_BaseSubsetIter):
    # TODO
    #returns list of list eachone missing a value in order.  
    #needs to be tested
    pass


# Left here so we know which sweeps to do
#NO_SUBSET, LEAVE_ONE_COL_OUT, SWEEP_TRAINING_SIZE = range(3)

class NoCV(_PartitionIterator):
    """Cross validator that just returns the entire set as the training set
    to begin with

    Parameters
    ----------
    n : int
        The number of rows in the data
    """
    def _iter_test_indices(self):
        yield np.array([], dtype=int)

class FlexibleStatifiedCV(_PartitionIterator):
    pass
    # TODO
    # To allow specification of the distribution of both the training and
    # test sets (e.g. train on 50/50, test on 90/10

CLF, CLF_PARAMS, SUBSET, SUBSET_PARAMS, CV, CV_PARAMS = range(6)
dimensions = (CLF, CLF_PARAMS, SUBSET, SUBSET_PARAMS, CV, CV_PARAMS)
dimension_descr = {CLF: 'classifier',
                   CLF_PARAMS: 'classifier parameters',
                   SUBSET: 'subset type',
                   SUBSET_PARAMS: 'subset parameters',
                   CV: 'cross-validation method',
                   CV_PARAMS: 'cross-validation parameters'}
    
class Run(object):
    def __init__(
        self,
        M,
        y,
        clf,
        test_indices):
        self.M = M
        self.y = y
        self.clf = clf
        self.test_indices = test_indices

    def __repr__(self):
        return 'Run(clf={})'.format(
                self.clf)

    def __test_M(self):
        return self.M[self.test_indices]

    def __test_y(self):
        return self.y[self.test_indices]

    def __pred_proba(self):
        return self.clf.predict_proba(self.__test_M())[:,0]

    def score(self):
        return self.clf.score(self.__test_M(), self.__test_y())

    def roc_curve(self):
        from ..communicate import plot_roc
        return plot_roc(self.__test_y(), self.__pred_proba(), verbose=False) 

    def roc_auc(self):
        return roc_auc_score(self.__test_y(), self.__pred_proba())

class Trial(object):
    def __init__(
        self, 
        M,
        y,
        clf=RandomForestClassifier,
        clf_params={},
        subset=SubsetNoSubset,
        subset_params={},
        cv=NoCV,
        cv_params={}):
        self.M = M
        self.y = y
        self.runs = None
        self.clf = clf
        self.clf_params = clf_params
        self.subset = subset
        self.subset_params = subset_params
        self.cv = cv
        self.cv_params = cv_params
        self.__by_dimension = {CLF: self.clf,
                               CLF_PARAMS: self.clf_params,
                               SUBSET: self.subset,
                               SUBSET_PARAMS: self.subset_params,
                               CV: self.cv,
                               CV_PARAMS: self.cv_params}
        self.__cached_ave_score = None
        self.repr = ('Trial(clf={}, clf_params={}, subset={}, '
                     'subset_params={}, cv={}, cv_params={})').format(
                        self.clf,
                        self.clf_params,
                        self.subset,
                        self.subset_params,
                        self.cv,
                        self.cv_params)
        self.hash = hash(self.repr)


    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self.repr
    def __getitem__(self, arg):
        return self.__by_dimension[arg]

    def has_run(self):
        return self.runs is not None

    def run(self):
        if self.has_run():
            return self.runs
        runs = []
        for subset in self.subset(self.y, **self.subset_params):
            runs_this_subset = []
            y_sub = self.y[subset]
            M_sub = self.M[subset]
            cv_cls = self.cv
            cv_kwargs = copy.deepcopy(self.cv_params)
            expected_cv_kwargs = inspect.getargspec(cv_cls.__init__).args
            if 'n' in expected_cv_kwargs:
                cv_kwargs['n'] = y_sub.shape[0]
            if 'y' in expected_cv_kwargs:
                cv_kwargs['y'] = y_sub
            cv_inst = cv_cls(**cv_kwargs)
            for train, test in cv_inst:
                clf_inst = self.clf(**self.clf_params)
                clf_inst.fit(M_sub[train], y_sub[train])
                test_indices = subset[test]
                runs_this_subset.append(Run(self.M, self.y, clf_inst, 
                                            test_indices))
            runs.append(runs_this_subset)    
        self.runs = runs
        return runs

    def average_score(self):
        if self.__cached_ave_score is not None:
            return self.__cached_ave_score
        self.run()
        M = self.M
        y = self.y
        ave_score = np.mean(
                [np.mean(
                    [run.score() for run in subset])
                 for subset in self.runs])
        self.__cached_ave_score = ave_score
        return ave_score
    
    def median_run(self):
        # Give or take
        #runs_with_score = [(run.score(), run) for run in self.runs]
        runs_with_score = [(run.score(), run) for run in it.chain(*self.runs)]
        runs_with_score.sort(key=lambda t: t[0])
        return runs_with_score[len(runs_with_score) / 2][1]

    def roc_curve(self):
        return self.median_run().roc_curve()

    def roc_auc(self):
        return self.median_run().roc_auc()
