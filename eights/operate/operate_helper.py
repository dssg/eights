from sklearn.tree._tree import TREE_LEAF
from collections import Counter
import itertools as it

def _feature_pair_report(pairs, description='pairs', verbose=False):
    count = Counter(pairs)
    if verbose:
        print description
        print '=' * 80
        print 'feature pair : occurences'
        for key, freq in count.most_common():
            print '{} : {}'.format(key, freq)
        print '=' * 80
        print
    return count

def feature_pairs_in_rf(rf, verbose=False):
    """Describes the frequency of features appearing subsequently in each tree
    in a random forest"""

    feature_pairs = [feature_pairs_in_tree(est) for est in rf.estimators_]
    aggr_by_depth = {}
    aggr_all = []
    for tree_pairs in feature_pairs:
        for depth, pairs in enumerate(tree_pairs):
            try:
                aggr_by_depth[depth] += pairs
            except KeyError:
                aggr_by_depth[depth] = pairs
            aggr_all += pairs

    result = {}
    for depth in sorted(aggr_by_depth.keys()):
        descr = 'Depth {}->{}'.format(depth, depth+1)
        result[descr] = _feature_pair_report(aggr_by_depth[depth], descr, 
                                             verbose)
    descr = 'Overall'
    result[descr] = _feature_pair_report(aggr_all, descr, verbose)
    return result
    
def feature_pairs_in_tree(dt):
    """Lists subsequent features sorted by importance

    Parameters
    ----------
    dt : sklearn.tree.DecisionTreeClassifer
    
    Returns
    -------
    list of list of tuple of int :
        Going from inside to out:

        1. Each int is a feature that a node split on
        
        2. If two ints appear in the same tuple, then there was a node
           that split on the second feature immediately below a node
           that split on the first feature

        3. Tuples appearing in the same inner list appear at the same
           depth in the tree

        4. The outer list describes the entire tree

    """
    t = dt.tree_
    feature = t.feature
    children_left = t.children_left
    children_right = t.children_right
    result = []
    if t.children_left[0] == TREE_LEAF:
        return result
    next_queue = [0]
    while next_queue:
        this_queue = next_queue
        next_queue = []
        results_this_depth = []
        while this_queue:
            node = this_queue.pop()
            left_child = children_left[node]
            right_child = children_right[node]
            if children_left[left_child] != TREE_LEAF:
                results_this_depth.append(tuple(sorted(
                    (feature[node], 
                     feature[left_child]))))
                next_queue.append(left_child)
            if children_left[right_child] != TREE_LEAF:
                results_this_depth.append(tuple(sorted(
                    (feature[node], 
                     feature[right_child]))))
                next_queue.append(right_child)
        result.append(results_this_depth)
    result.pop() # The last results are always empty
    return result



