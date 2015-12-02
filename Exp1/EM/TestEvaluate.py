import numpy
from sklearn import linear_model
import MultiLayerPerceptron

def test_evaluate(regr,regr_map,x_list,test_scores):

   N = len(x_list)

   input_dim = 9 
   test_X = numpy.empty([0,input_dim])
   for i in range(N):
      cur_X = numpy.concatenate((numpy.mean(x_list[i].T,axis=0),numpy.std(x_list[i].T,axis=0),numpy.ptp(x_list[i].T,axis=0)),axis=1) 
      test_X = numpy.concatenate((test_X,cur_X),axis=0)



   x_nn = regr_map.predict(test_X)
   output = regr.predict(x_nn)

   print("Residual sum of squares: %.5f"
      % numpy.mean(numpy.square((output - test_scores))))
   
   return output 
