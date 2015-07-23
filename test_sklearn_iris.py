#This is the running file.
#This should obscure as much as possible.
from eights.investigate import *
from eights.generate import *

#load iris
import sklearn.datasets
iris = sklearn.datasets.load_iris()
M = iris.data

#cast from ND to SA
M = cast_np_nd_to_sa(M)

#Pretend .1 is wrong so set all values of .1 in M[3] as .2
# make a new column where its a test if col,val, (3,.2), (2,1.4) is true.
print 'that'


import pdb; pdb.set_trace()

#from decontaminate import remove_null, remove_999, case_fix, truncate
#from generate import donut
#from aggregate import append_on_right, append_on_bottom
#from truncate import remove
#from operate import run_list, fiveFunctions
#from communicate import graph_all, results_invtestiage


#investiage
#M_orginal = csv_open(file_loc, file_descpiption)  # this is our original files
#results = eights.investigate.describe_all(M_orginal)
#results_invtestiage(results)

#decontaminate

#aggregate

#generate
#M = np.array([]) #this is the master Matrix we train on.
#labels = np.array([]) # this is tells us

#truncate
#models = [] #list of functions

#operate

#communicate


#func_list = [sklearn.randomforest,sklearn.gaussian, ]


#If main:
#run on single csv
