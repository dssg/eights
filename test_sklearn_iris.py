#This is the running file.
#This should obscure as much as possible.

import numpy as np
#load iris
import sklearn.datasets
iris = sklearn.datasets.load_iris()
M = iris.data
labels = iris.target





####################investigate#######################
import eights.investigate as  inv

#cast from ND to SA
M = inv.cast_np_nd_to_sa(M)

for x in inv.describe_cols(M): print x

plot = False
if plot:
    inv.plot_correlation_scatter_plot(M) #this is stange
    inv.plot_correlation_matrix(M)
    inv.plot_kernel_density(M['f0'])
    inv.plot_box_plot(M['f0'])    

####################Decontaminate#######################





####################generate#######################
import eights.generate  as gen

#lets generate row of our data
M = gen.where_all_are_true(M, [gen.val_between,gen.val_between], ['f0','f1'],[(3.5,5.0),(2.7, 3.1)], '4 and(2.7-3.1)')
#
#M = gen.where(M, 
#        [[gen.val_between,['f0'],(3.5,2.7)],
#        [gen.val_between,['f1'],(5.,3.1)]],
#        '4 and(2.7-3.1)'


M = gen.where_all_are_true(M, [gen.val_between,gen.val_between], ['f0','f1'],[(3.5,5.0),(2.7, 3.1)], 'bad_rules')



#new eval function
def rounds_to_val(M, col_name, boundary):
    return (np.round(M[col_name]) == boundary)
    
M = gen.where_all_are_true(M, [rounds_to_val], ['f0'],[5], 'rounds to 5')

#making a useless row
M = gen.where_all_are_true(M, [gen.val_between,gen.val_between], ['f0','f1'],[(35,50),(27, 31)], 'Useless Cols')

import pdb; pdb.set_trace()

####################Truncate#######################
import eights.truncate  as tr
#remove Useless row
M = tr.fewer_then_n_nonzero_in_col(M,1)

#remove class 2
M = gen.append_cols(M, labels, 'labels')
M = tr.remove_rows_where(M, tr.val_eq, 'labels', 2)
labels=M['labels']
M = tr.remove_cols(M,'labels')

####################Operate/Permabulate#######################

import eights.operate as op
exp = op.run_std_classifiers(M,labels)
exp.make_csv()



####################Communicate#######################



#Pretend .1 is wrong so set all values of .1 in M[3] as .2
# make a new column where its a test if col,val, (3,.2), (2,1.4) is true.


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
