import inspect 
import json
import copy
import abc
import datetime
import itertools as it
import numpy as np

from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.cross_validation import _PartitionIterator


from .perambulate_helper import *
import eights.utils as utils

class Experiment(object):
    def __init__(
            self, 
            M, 
            y, 
            clfs={RandomForestClassifier: {}}, 
            subsets={SubsetNoSubset: {}}, 
            cvs={NoCV: {}},
            trials=None):
        self.M = utils.cast_np_sa_to_nd(M)
        self.y = y
        self.clfs = clfs
        self.subsets = subsets
        self.cvs = cvs
        self.trials = trials

    def __repr__(self):
        print 'Experiment(clfs={}, subsets={}, cvs={})'.format(
                self.clfs, 
                self.subsets, 
                self.cvs)
        
    def __run_all_trials(self, trials):
        # TODO some multiprocess thing
        
        for trial in trials:
            trial.run()

    def __copy(self, trials):
        return Experiment(
                self.M, 
                self.y,
                self.clfs,
                self.subsets,
                self.cvs,
                trials)

    def __transpose_dict_of_lists(self, dol):
        # http://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
        return (dict(it.izip(dol, x)) for 
                x in it.product(*dol.itervalues()))

    def slice_on_dimension(self, dimension, value, trials=None):
        self.run()
        return self.__copy([trial for trial in self.trials if 
                            trial[dimension] == value])

    def slice_by_best_score(self, dimension):
        self.run()
        categories = {}
        other_dims = list(dimensions)
        other_dims.remove(dimension)
        for trial in self.trials:
            # http://stackoverflow.com/questions/5884066/hashing-a-python-dictionary
            key = repr([trial[dim] for dim in other_dims])
            try:
                categories[key].append(trial)
            except KeyError:
                categories[key] = [trial]
        result = []
        for key in categories:
            result.append(max(
                categories[key], 
                key=lambda trial: trial.average_score()))
        return self.__copy(result)

    def has_run(self):
        return self.trials is not None

    def run(self):
        if self.has_run():
            return self.trials
        trials = []
        for clf in self.clfs:
            all_clf_ps = self.clfs[clf]
            for clf_params in self.__transpose_dict_of_lists(all_clf_ps):
                for subset in self.subsets:
                    all_sub_ps = self.subsets[subset]
                    for subset_params in self.__transpose_dict_of_lists(all_sub_ps):
                        for cv in self.cvs:
                            all_cv_ps = self.cvs[cv]
                            for cv_params in self.__transpose_dict_of_lists(all_cv_ps):
                                trial = Trial(
                                    M=self.M,
                                    y=self.y,
                                    clf=clf,
                                    clf_params=clf_params,
                                    subset=subset,
                                    subset_params=subset_params,
                                    cv=cv,
                                    cv_params=cv_params)
                                trials.append(trial)
        self.__run_all_trials(trials)
        self.trials = trials
        return trials

    def average_score(self):
        self.run()
        return {trial: trial.average_score() for trial in self.trials}

    def roc_auc(self):
        self.run()
        return {trial: trial.roc_auc() for trial in self.trials}

    def make_report(self, report_file_name='report.pdf'):
        from ..communicate import Report
        self.run()
        rep = Report(self, report_file_name)
        rep.add_heading('Eights Report {}'.format(datetime.datetime.now()), 1)
        rep.add_heading('Roc AUCs', 3)
        rep.add_summary_graph_roc_auc()
        rep.add_heading('Average Scores', 3)
        rep.add_summary_graph_average_score()
        rep.add_heading('ROC for best trial', 3)
        rep.add_graph_for_best_roc()
        rep.add_heading('Legend', 3)
        rep.add_legend()
        return rep.to_pdf()
        # TODO make this more flexible


def simple_sliding_window_index(n, training_window_size, testing_window_size):
    for train, test in sliding_window_index(
            n, 
            0, 
            train_window_size - 1, 
            train_window_size, 
            train_window_size + testing_window_size - 1, 
            1):
        yield train, test
            

def sliding_window_index(n, 
        init_train_window_start, 
        init_train_window_end, 
        init_test_window_start,
        init_test_window_end,
        increment):
    """

    Parameters
    ----------
    n : int
        number of rows in the matrix
    init_train_window_start : int
    init_train_window_end : int
    init_test_window_start : int
    init_test_window_end : int
    increment : int
        distance training and testing window are moved per iteration

    """
    raise NotImplementedError

def simple_expanding_window_index(n, training_window_size, testing_window_size):
    for train, test in expanding_window_index(
            n, 
            0, 
            train_window_size - 1, 
            train_window_size, 
            train_window_size + testing_window_size - 1, 
            1):
        yield train, test

def expanding_window_index(n, 
        init_train_window_start, 
        init_train_window_end, 
        init_test_window_start,
        init_test_window_end,
        increment):
    """

    Parameters
    ----------
    n : int
        number of rows in the matrix
    init_train_window_start : int
    init_train_window_end : int
    init_test_window_start : int
    init_test_window_end : int
    increment : int
        distance training and testing window are moved per iteration

    """
    raise NotImplementedError

def simple_sliding_window_time(n, training_window_size, testing_window_size):
    for train, test in sliding_window_index(
            n, 
            0, 
            train_window_size - 1, 
            train_window_size, 
            train_window_size + testing_window_size - 1, 
            1):
        yield train, test
            

def sliding_window_time(n, 
        init_train_window_start, 
        init_train_window_end, 
        init_test_window_start,
        init_test_window_end,
        increment):
    """

    Parameters
    ----------
    n : int
        number of rows in the matrix
    init_train_window_start : int
    init_train_window_end : int
    init_test_window_start : int
    init_test_window_end : int
    increment : int
        distance training and testing window are moved per iteration

    """
    raise NotImplementedError

def simple_expanding_window_time(n, training_window_size, testing_window_size):
    for train, test in expanding_window_time(
            n, 
            0, 
            train_window_size - 1, 
            train_window_size, 
            train_window_size + testing_window_size - 1, 
            1):
        yield train, test

def expanding_window_time(n, 
        init_train_window_start, 
        init_train_window_end, 
        init_test_window_start,
        init_test_window_end,
        increment):
    """
    Parameters
    ----------
    n : int
        number of rows in the matrix
    init_train_window_start : int
    init_train_window_end : int
    init_test_window_start : int
    init_test_window_end : int
    increment : int
        distance training and testing window are moved per iteration

    """
    raise NotImplementedError

#def sliding_window(l, w, tst_w):
#    """
#    Parameters
#    ----------
#    l : list
#        the data
#    w : int
#        window size
#    tst_w : int
#        size of test windows
#
#    """
#    ret = []
#    for idx, _ in enumerate(l):
#        if idx + w + tst_w > len(l): 
#            break
#        train = [l[idx + x] for x in range(w)]
#        test = [l[idx + w + x] for x in range(tst_w)]
#        ret.append((train, test))
#    return ret
#    
#def expanding_window(l,w,tst_w):
#    ret = []
#    
#    for idx, i in enumerate(l):
#        if idx + w + tst_w > len(l): break
#        
#        train = [l[x] for x in range(idx+w)]
#        test = []
#        for x in range(tst_w):
#            test.append(l[idx + w + x])
#        ret.append((train, test))
#    return ret


#sweep calls random


    


def random_subset_of_columns(M, number_to_select):
    #np.rand(x, y)
    #handle id's as well as names
    # returns an M with fewer cols
    raise NotImplementedError
    
def random_subset_of_rows_even_distribution(M, y, number_to_select):
    #np.rand(x, y)
    #handle id's as well as names
    raise NotImplementedError

def random_subset_of_rows_actual_distribution(M, y, number_to_select):
    #np.rand(x, y)
    #handle id's as well as names
    raise NotImplementedError



    

