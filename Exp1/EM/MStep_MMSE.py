import numpy
import neurolab as nl
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
      cur_X = numpy.concatenate((x_list[i].T, numpy.square(x_list[i].T), numpy.exp(x_list[i].T)),axis=1)  # this is made of size Lx9
      cur_L = cur_X.shape[0]
      train_X = numpy.concatenate((train_X,cur_X),axis=0)
      train_Y = numpy.concatenate((train_Y,numpy.tile(cur_Y,[cur_L,1])),axis=0)

   regr = linear_model.LinearRegression()
   regr.fit(train_X, train_Y)
 
#   train_X_min = -0.5 #numpy.amin(train_X,axis=0)  
#   train_X_max = 0.5 #numpy.amax(train_X,axis=0) 
#   min_max_list = [] 
#   for i in range(input_dim):
#      min_max_list.append([train_X_min, train_X_max]) 
#   net = nl.net.newff(min_max_list, [5, 3]) 
#   err = net.train(train_X, train_Y, show=15) 
 
   # getting output for each train instance
   x_nn = numpy.zeros((N,output_dim))  
   for i in range(N):
      cur_X = numpy.concatenate((x_list[i].T, numpy.square(x_list[i].T), numpy.exp(x_list[i].T)),axis=1)  # this is made of size Lx9
      x_nn[i,:] = numpy.mean(regr.predict(cur_X),axis=0)

   # Estimating W
   regr_w_est = linear_model.LinearRegression()
   regr_w_est.fit(x_nn, Y)

   print("Residual sum of squares: %.2f"
      % numpy.mean(numpy.square((regr_w_est.predict(x_nn) - Y))))

   return regr_w_est,x_nn
