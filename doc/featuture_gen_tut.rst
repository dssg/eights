****************************************
Using Eights for Feature Generation
****************************************
Quick test::

   import numpy as np
   
   import eights.investigate
   import eights.generate
   
   M = [[1,2,3], [2,3,4], [3,4,5]]
   col_names = ['heigh','weight', 'age']
   lables= [0,0,1]
   
   # Eights uses Structured arrays, which allow for different data types in different columns
   M = eights.investigate.convert_list_of_list_to_sa(np.array(M), c_name=col_names)
   #By convention M is the our matrix on which our ML algo will run
   
   #This is a sample lambada statment, to show how easy it is to craft your own.  
   #the signitutre(M, col_name, boundary) is standardized.  
   def test_equality(M, col_name, boundary):
       return M[col_name] == boundary

   #This generates a new frow where the values are all true
   M_new = eights.generate.where_all_are_true(
                  M,
                  [test_equality, test_equality, test_equality], 
                  ['height','weight', 'age'], 
                  [1,2,3], 
                  ('new_column_name',)
                  )
   # Read top to bottom:
   # If test_equality in column 'height' == 1 AND
   # If test_equality in column 'weight' == 2 AND
   # If test_equality in column 'age' == 3 
   # return true
   