import numpy
from numpy import matlib
import sys
sys.path.append('/home/rcf-proj/pg/guptarah/DepressionStudy/Scripts/theano-rnn/')
from sklearn import linear_model

def PrepareData(train_scores,train_vad_ts,train_features_ts):
   n_steps = 5 # number of past/future frames used for current frame prediction
  
   num_feats = train_features_ts[0].shape[1] 
   width_window_matrix = num_feats*(2*n_steps+1)
   window_matrix = numpy.zeros((0, width_window_matrix)) 
   labels_matrix = numpy.zeros((0, 3)) 
   score_matrix = numpy.zeros((0, 1)) 
   for i in range(len(train_vad_ts)):
      cur_score = train_scores[i]
      cur_feat_matrix = train_features_ts[i]
      cur_label_matrix = train_vad_ts[i].T
      num_frames = cur_feat_matrix.shape[0]

      len_cur_window_matrix = num_frames # need to remove top n bottown n_steps later
      cur_window_matrix = numpy.zeros((len_cur_window_matrix,width_window_matrix)) 
      for j in range(-1*n_steps,n_steps+1):
         cur_window_matrix[:,(j+n_steps)*num_feats:(j+n_steps+1)*num_feats] = numpy.roll(cur_feat_matrix,j,axis=0)
     

      cur_window_matrix = cur_window_matrix[n_steps:-1*n_steps,:] 
      cur_label_matrix = cur_label_matrix[n_steps:-1*n_steps,:] 
      cur_score_vector = numpy.matrix(numpy.ones((num_frames-2*n_steps,1))) * cur_score
 
      window_matrix = numpy.concatenate((window_matrix,cur_window_matrix),axis=0) 
      labels_matrix = numpy.concatenate((labels_matrix,cur_label_matrix),axis=0) 
      score_matrix = numpy.concatenate((score_matrix,cur_score_vector),axis=0) 
   

   return window_matrix, labels_matrix, score_matrix

def MstepSecondLayer(updated_output,labels_matrix,score_matrix):
   num_feats = updated_output.shape[1]
   new_updated_matrix = matlib.zeros((updated_output.shape[0],updated_output.shape[1]+1)) # this will have
   # each feat multiplied by score and score as feature 
   for i in range(num_feats):
      new_updated_matrix[:,i] = numpy.multiply(updated_output[:,i],score_matrix) 
   new_updated_matrix[:,i+1] = score_matrix
   second_layer_regr = linear_model.LinearRegression(fit_intercept=True)
   second_layer_regr.fit(new_updated_matrix,labels_matrix)

   # print training error for this iteration
   print 'cur_error: ', numpy.std(labels_matrix - second_layer_regr.predict(new_updated_matrix))
 
   return second_layer_regr
 
def EStep(updated_output,score_matrix,labels_matrix,second_layer_coeff,second_layer_intercept):
   labels_less_dep = labels_matrix - score_matrix*second_layer_coeff[:,-1].T - numpy.ones(score_matrix.shape)*second_layer_intercept
   frames_x_star = updated_output.shape[0]
   dim_x_star = updated_output.shape[1]
 
   x_star = numpy.zeros(updated_output.shape) 
   w = second_layer_coeff[:,:-1] 
   for i in range(frames_x_star):
      cur_d_n = score_matrix[i][0,0]
      inv_mat = numpy.linalg.inv(numpy.square(cur_d_n)*w.T*w + numpy.eye(dim_x_star))
      cur_x_n = inv_mat * (cur_d_n*w*labels_matrix[i,:].T+updated_output[i,:].T) 
      x_star[i,:] = cur_x_n.T
      
   return x_star 
   

def TrainEM(train_scores,train_vad_ts,train_features_ts):
   # prepare data for training
   [window_matrix,labels_matrix,score_matrix] = PrepareData(train_scores,train_vad_ts,train_features_ts) 

   # Initialization
   x_star = labels_matrix   

   not_converged = True
   count_iterations = 0
   while not_converged:
      # M step for first layer
      first_layer_regr = linear_model.LinearRegression(n_jobs=4) 
      first_layer_regr.fit(window_matrix, x_star)
      updated_output = numpy.matrix(first_layer_regr.predict(window_matrix))
      
      # M step for second layer
      second_layer_regr = MstepSecondLayer(updated_output,labels_matrix,score_matrix)           
      second_layer_coeff, second_layer_intercpet = numpy.matrix(second_layer_regr.coef_), second_layer_regr.intercept_ 

      # E step to find x_star
      x_star = EStep(updated_output,score_matrix,labels_matrix,second_layer_coeff,second_layer_intercept) 
      
      if count_iterations == 10:
         break
      count_iterations += 1

   return first_layer_regr, second_layer_regr

if __name__ == "__main__":
   TrainEM(train_scores,train_vad_ts,train_features_ts)
