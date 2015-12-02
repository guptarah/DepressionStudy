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

   # Estimating x_nn
   out = []
   for i in range(N):
      cur_Y = X_est[i,:]
      cur_X = numpy.concatenate((x_list[i].T, numpy.square(x_list[i].T), numpy.exp(x_list[i].T)),axis=1)  # this is made of size Lx9

      for j in range(cur_X.shape[0]):
         tupledata = list((cur_X[j,:].tolist()[0], cur_Y.tolist()[0])) # don't mind this variable name
         out.append(tupledata)


   input_dim = 9
   hidden_dim = 6
   output_dim = 3
   NN = MultiLayerPerceptron.MLP_Classifier(input_dim,hidden_dim,output_dim, iterations = 20, learning_rate = 0.01,\
                        momentum = 0.5, rate_decay = 0.0001,\
                        output_layer = 'softmax')
   NN.fit(out)
  
   # getting output for each train instance
   x_nn = numpy.zeros((N,output_dim))  
   for i in range(N):
      test_out = []
      cur_Y = X_est[i,:]
      cur_X = numpy.concatenate((x_list[i].T, numpy.square(x_list[i].T), numpy.exp(x_list[i].T)),axis=1)  # this is made of size Lx9
      for j in range(cur_X.shape[0]):
         tupledata = list((cur_X[j,:].tolist()[0], cur_Y.tolist()[0])) # don't mind this variable name
         test_out.append(tupledata)
      x_nn[i,:] = numpy.mean(numpy.matrix(NN.test(out)),axis=0)

   # Estimating W
   w_est = numpy.dot(numpy.linalg.pinv(x_nn),Y)

   return w_est,x_nn
