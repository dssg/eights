
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

def open_simple_csv_as_list(file_loc):
    with open(file_loc, 'rb') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data 

def make_grams(tweet, gram):
    t =[]     
    for idx in range(len(tweet)-gram):
        t_2=[]
        for x in range(gram):
            t_2.append(tweet[idx+x])
        t.append(tuple(t_2))
    return t

def turn_list_of_list_of_items_to_SA_by_overlap(lol_items):
    adict  = {}
    feature_index = 0
    bigram_index = []
    l = [item for l_items in lol_items for item in l_items]
    num_unique_items = len(set(l))
    M = np.zeros(shape=(len(lol_items), num_unique_items))    
    feature_index = 0
    for row, l_items in enumerate(lol_items):
        for item in l_items:
            try:
                M[row, adict[item]] += 1
                adict[item] += 1 #
            except KeyError:
                M[row, feature_index] += 1
                feature_index += 1
                bigram_index.append(item)
                adict[item] = 1
    return M

def strip_out_labels(data,key):
    return [x[key] for x in data]

def reverse_dict(adict, value):
    for x in adict.items():
        if x[1] == value:
            return x[0]
    return None
    
data = open_simple_csv_as_list('./Tweets-DataFixed.csv')
labels = strip_out_labels(data, 3)    
tweets  = [x[1].split() for x in data]
bigrams = [make_grams(tweet, 2) for tweet in tweets] 
M = turn_list_of_list_of_items_to_SA_by_over_Lap(bigrams)     

skf = cross_validation.StratifiedKFold(labels, n_folds=5)
clf = RandomForestClassifier()

for train_index, test_index in skf:
    clf.fit(M[train_index],np.array(labels)[train_index])
    print clf.score(M[test_index],np.array(labels)[test_index])


import pdb; pdb.set_trace()


