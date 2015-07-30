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
        yield (np.array([], dtype=int), '')

    @abc.abstractmethod
    def __repr__(self):
        return ''

class SubsetNoSubset(_BaseSubsetIter):
    def __iter__(self):
        yield (np.arange(self._y.shape[0]), '')

    def __repr__(self):
        return 'SubsetNoSubset()'

class SubsetRandomRowsActualDistribution(_BaseSubsetIter):
        
    def __init__(self, y, subset_size, n_subsets=3):
        super(SubsetRandomRowsActualDistribution, self).__init__(y)
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
        for i in xrange(n_subsets):
            samples = {key: sample(indices[key], n_choices[key]) for key in count}
            all_indices = np.sort(np.concatenate(samples.values()))
            yield (all_indices, 'sample_num={}'.format(i))

    def __repr__(self):
        return 'SubsetRandomRowsActualDistribution(subset_size={}, n_subsets={})'.format(
                self.__subset_size,
                self.__n_subsets)

class SubsetRandomRowsEvenDistribution(_BaseSubsetIter):
        
    def __init__(self, y, subset_size, n_subsets=3):
        super(SubsetRandomRowsEvenDistribution, self).__init__(y)
        self.__subset_size = subset_size
        self.__n_subsets = n_subsets

    def __iter__(self):
        y = self._y
        subset_size = self.__subset_size
        n_subsets = self.__n_subsets
        count = Counter(y)
        n_categories = len(count)
        proportions = {key: 1.0 / n_categories for key in count}
        n_choices = {key: int(proportions[key] * subset_size) for 
                     key in proportions}
        indices = {key: np.where(y == key)[0] for key in count}
        for i in xrange(n_subsets):
            samples = {key: sample(indices[key], n_choices[key]) for key in count}
            all_indices = np.sort(np.concatenate(samples.values()))
            yield (all_indices, 'sample_num={}'.format(i))

    def __repr__(self):
        return 'SubsetRandomRowsEvenDistribution(subset_size={}, n_subsets={})'.format(
                self.__subset_size,
                self.__n_subsets)

class SubsetSweepNumRows(_BaseSubsetIter):
        
    def __init__(self, y, num_rows):
        super(SubsetSweepNumRows, self).__init__(y)
        self.__num_rows = num_rows

    def __iter__(self):
        y = self._y
        num_rows = self.__num_rows
        for rows in num_rows:
            all_indices = np.sort(sample(np.arange(0, y.shape[0]), rows))
            yield (all_indices, 'rows={}'.format(rows))

    def __repr__(self):
        return 'SubsetSweepNumRows(num_rows={})'.format(
                self.__num_rows)

class SubsetSweepVaryStratification(_BaseSubsetIter):
        
    def __init__(self, y, proportions_positive, subset_size):
        super(SubsetSweepVaryStratification, self).__init__(y)
        self.__proportions_positive = proportions_positive
        self.__subset_size = subset_size

    def __iter__(self):
        y = self._y
        subset_size = self.__subset_size
        positive_cases = np.where(y)[0]
        negative_cases = np.where(np.logical_not(y))[0]
        for prop_pos in self.__proportions_positive:
            positive_sample = sample(positive_cases, int(subset_size * prop_pos))
            negative_sample = sample(negative_cases, int(subset_size * (1 - prop_pos)))
            # If one of these sets is empty, concatentating them casts to float, so we have
            # to cast it back (hence the astype)
            all_indices = np.sort(np.concatenate([positive_sample, negative_sample])).astype(int)
            yield (all_indices, 'prop_positive={}'.format(prop_pos))

    def __repr__(self):
        return 'SubsetSweepVaryStratification(proportions_positive={}, subset_size={})'.format(
                self.__proportions_positive,
                self.__subset_size)

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
        train_indices,
        test_indices,
        subset_note,
        cv_note):
        self.M = M
        self.y = y
        self.clf = clf
        self.test_indices = test_indices
        self.train_indices = train_indices
        self.subset_note = subset_note
        self.cv_note = cv_note

    def __repr__(self):
        return 'Run(clf={}, subset_note={}, cv_note={})'.format(
                self.clf, self.subset_note, self.cv_note)

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

    def prec_recall_curve(self):
        from ..communicate import plot_prec_recall
        return plot_prec_recall(self.__test_y(), self.__pred_proba(), 
                                verbose=False)


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
        for subset, subset_note in self.subset(self.y, **self.subset_params):
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
                if hasattr(cv_inst, 'cv_note'):
                    cv_note = cv_inst.cv_note()
                else:
                    cv_note = ''
                clf_inst = self.clf(**self.clf_params)
                clf_inst.fit(M_sub[train], y_sub[train])
                test_indices = subset[test]
                train_indices = subset[train]
                runs_this_subset.append(Run(self.M, self.y, clf_inst, 
                                            train_indices, test_indices,
                                            subset_note, cv_note))
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

    # TODO These should all be average across runs rather than picking the 
    # median

    def roc_curve(self):
        return self.median_run().roc_curve()

    def roc_auc(self):
        return self.median_run().roc_auc()

    def prec_recall_curve(self):
        return self.median_run().prec_recall_curve()
