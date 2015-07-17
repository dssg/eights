from sklearn.tree._tree import TREE_LEAF


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
                results_this_depth.append(
                    (feature[node], 
                     feature[left_child]))
                next_queue.append(left_child)
            if children_left[right_child] != TREE_LEAF:
                results_this_depth.append(
                    (feature[node], 
                     feature[right_child]))
                next_queue.append(right_child)
        result.append(results_this_depth)
    result.pop() # The last results are always empty
    return result



