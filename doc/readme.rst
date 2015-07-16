
By default we use pass structured arrays from within functions. However to accommodate both list of list and list of nd.arrays, we cast structured arrays as: 
name, M = sa_to_nd(M)

For our the rapid analysis of different CLF's in SKLEARN we use dictionaries of dictionaries of Lists. Where lists are the slice indices, the outermost dictionary is the test, the inner dictionary is the run.  For instance, Test["sweepParamtersize"]['one'] == nd.array([1,2,3])

M is a structured array
Y is the target in SKLEARN parlance. It is the known labels.



supported plots:      (ROC, PER_RECALL, ACC, N_TOP_FEAT, AUC)
supported clfs:       (RF, SVM, DESC_TREE, ADA_BOOST)
supported subsetting: (LEAVE_ONE_COL_OUT, SWEEP_TRAINING_SIZE, )
supported cv:         (K_FOLD, STRAT_ACTUAL_K_FOLD, STRAT_EVEN_K_FOLD)         


plots = ['roc', 'acc']
clfs =  [('random forest', ['RF PARMS'] ),
         ('svm',           ['SVM PARMS'] )]
         
subsets = [('leave one out col',   ['PARMS'] ), 
           ('sweep training size', ['PARMS'] )]

cv =     [('cv', ['parms']),
          ('stratified cv', ['parms'])]
          
runOne = Experiment(plots, clfs, subsets, cv)

exp = Experiment(
      [RF: {'depth': [10, 100],
            'n_trees': [40, 50]},
       SVM: {'param_1': [1, 2],
             'param_2': ['a', 'b']}],
      [LEAVE_ONE_COL_OUT: {'col_names': ['f0', 'f1', 'f2', 'f3']},
       SWEEP_TRAINING_SIZE: {'sizes': (10, 20, 40)}
      ],
      [STRAT_ACTUAL_K_FOLD : {'y': y}])

