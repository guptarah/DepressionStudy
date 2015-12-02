import numpy
from sklearn import linear_model
import MultiLayerPerceptron 

def m_step(X_est,Y,x_list):
   """Variables:
   X_est: estimate of X from E step NxD
   Y: target lables Nx1
   x_list: list of N+ matrices. Matrices are of size 3xL, where L is variable per file and equals number of frames per file
   train_indices: from x_list only get matrices with these indices 
   """

   N = X_est.shape[0]

   input_dim = 9
   output_dim = 3

   # Estimating x_nn
   train_X = numpy.empty([0,input_dim])
   train_Y = numpy.empty([0,output_dim])
   for i in range(N):
      cur_Y = X_est[i,:]
      cur_X = numpy.concatenate((numpy.mean(x_list[i].T,axis=0),numpy.std(x_list[i].T,axis=0),numpy.ptp(x_list[i].T,axis=0)),axis=1) 
      train_X = numpy.concatenate((train_X,cur_X),axis=0)
      train_Y = numpy.concatenate((train_Y,cur_Y),axis=0)

   regr = linear_model.LinearRegression()
   regr.fit(train_X, train_Y)
   
  
   # getting output for each train instance
   x_nn = regr.predict(train_X) 

   # Estimating W
   regr_w_est = linear_model.LinearRegression()
   regr_w_est.fit(x_nn, Y)

   print("Residual sum of squares: %.5f"
      % numpy.mean(numpy.square((regr_w_est.predict(x_nn) - Y))))

   return regr_w_est,x_nn,regr
