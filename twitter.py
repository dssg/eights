#are we classifying tweets or users?
#(Pdb) from collections import Counter
#(Pdb) Counter(labels)
#[('IBeenTrippin', '2'), ('oficiallyyyjess', '2'), ('Coleifa', '2'), ('_JMilly', '2'), ('WORLDMAFIAMUSIC', '2'), ('Chicago_708', '2'), ('RekoJ_DtM', '3'), ('ArrogantLo', '2'), ('NikeAirAaron', '2'), ('Los_104SC', '2'), ('IDoIt4Rocc', '3'), ('ItsGunnaHoe', '2'), ('Dasani__', '2'), ('Hamma2xs', '3'), ('NoGood_Aaron87', '2'), ('RapCatchup', '2'), ('tosama4', '2'), ('jolting2', '2'), ('jolting2', '2'), ('dreadheadusher', '2'), ('Shvnv', '2'), ('alexandracee_', '2'), ('Kece24_Sauce', '2'), ('picassopaige', '2'), ('BlackBoardG', '2'), ('Chi_Aficionado', '2'), ('Husselius', '2'), ('BHaynes_70', '3'), ('ItsTotallyNoah', '2'), ('lishamisha22', '2'), ('BadGirl_Asia', '2'), ('JIMBO_GO_DUMB', '2'), ('ezeek___', '2'), ('donkaenutts', '2'), ('NoGood_Aaron87', '2'), ('GirlDats_Marcus', '2'), ('KJFRMABM', '3'), ('KJFRMABM', '2'), ('KJFRMABM', '2'), ('D_Marie132', '2'), ('RekoJ_DtM', '3'), ('AYE_FLINO', '2'), ('Brannntastic', '2'), ('DeeDeeGoins', '2'), ('EddietheHUNK', '2'), ('ChicagoMadeKeem', '2'), ('Darealreon', '2')]
#(Pdb) Counter([x for x in user_score if x[1] != '1'])

#Counter({('RekoJ_DtM', '3'): 2, ('jolting2', '2'): 2, ('KJFRMABM', '2'): 2, ('NoGood_Aaron87', '2'): 2, ('donkaenutts', '2'): 1, ('ItsTotallyNoah', '2'): 1, ('Chi_Aficionado', '2'): 1, ('WORLDMAFIAMUSIC', '2'): 1, ('Chicago_708', '2'): 1, ('ChicagoMadeKeem', '2'): 1, ('ezeek___', '2'): 1, ('GirlDats_Marcus', '2'): 1, ('Coleifa', '2'): 1, ('JIMBO_GO_DUMB', '2'): 1, ('ArrogantLo', '2'): 1, ('Kece24_Sauce', '2'): 1, ('BHaynes_70', '3'): 1, ('Darealreon', '2'): 1, ('Husselius', '2'): 1, ('Brannntastic', '2'): 1, ('Hamma2xs', '3'): 1, ('NikeAirAaron', '2'): 1, ('BlackBoardG', '2'): 1, ('KJFRMABM', '3'): 1, ('RapCatchup', '2'): 1, ('IBeenTrippin', '2'): 1, ('oficiallyyyjess', '2'): 1, ('Shvnv', '2'): 1, ('BadGirl_Asia', '2'): 1, ('D_Marie132', '2'): 1, ('Los_104SC', '2'): 1, ('_JMilly', '2'): 1, ('EddietheHUNK', '2'): 1, ('alexandracee_', '2'): 1, ('Dasani__', '2'): 1, ('picassopaige', '2'): 1, ('DeeDeeGoins', '2'): 1, ('tosama4', '2'): 1, ('ItsGunnaHoe', '2'): 1, ('IDoIt4Rocc', '3'): 1, ('AYE_FLINO', '2'): 1, ('dreadheadusher', '2'): 1, ('lishamisha22', '2'): 1})

import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation


def open_simple_csv_as_list(file_loc):
    with open(file_loc, 'rb') as f:
        reader = csv.reader(f)
        data= list(reader)
    return data
   

def make_grams(tweets, gram):
    r = []
    for txt in tweets:   
        t =[]     
        for idx in range(len(txt)-gram):
            t_2=[]
            for x in range(gram):
                t_2.append(txt[idx+x])
            t.append(tuple(t_2))
        r.append(t)
    return r

def turn_list_of_list_of_items_to_SA_by_over_Lap(lol_items):
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
    
data = open_simple_csv_as_list('./Tweets-DataFixed.csv')
labels = [x[3] for x in data]    
tweets  = [x[1].split() for x in data]
bigrams = make_grams(tweets,2)  
data = [[(1,2),(1,2),(1,4)],[(1,2),(3,1)],[(1,2)]]    
M = turn_list_of_list_of_items_to_SA_by_over_Lap(bigrams)     

import pdb; pdb.set_trace()         

def reverse_dict(adict, value):
    for x in adict.items():
        if x[1] == value:
            return x[0]
    return None

print reverse_dict(adict,15)

import pdb; pdb.set_trace()

skf = cross_validation.StratifiedKFold(labels, n_folds=5)
clf = RandomForestClassifier()

for train_index, test_index in skf:
    clf.fit(M[train_index],np.array(labels)[train_index])
    print clf.score(M[test_index],np.array(labels)[test_index])


import pdb; pdb.set_trace()


