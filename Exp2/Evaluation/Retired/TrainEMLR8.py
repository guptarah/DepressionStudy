import numpy
from numpy import matlib
import sys
from sklearn import linear_model
from sklearn import decomposition 
from TestOutput import TestOutput7 
from PrepareData import PrepareData 


def MstepSecondLayer(updated_output,labels_matrix,score_matrix):
   num_feats = updated_output.shape[1]
   new_updated_matrix = matlib.zeros((updated_output.shape[0],updated_output.shape[1])) # this will have
   # each feat multiplied by score and score as feature

   new_updated_matrix[:,0] = numpy.multiply(updated_output[:,0],score_matrix) 
   new_updated_matrix[:,1] = numpy.divide(updated_output[:,1],(score_matrix+.01*numpy.ones(score_matrix.shape))) 
   new_updated_matrix[:,2] = updated_output[:,2]

   second_layer_regr = linear_model.LinearRegression(fit_intercept=True)
   second_layer_regr.fit(new_updated_matrix,labels_matrix)

   # print training error for this iteration
   print 'cur_error: ', numpy.std(labels_matrix - second_layer_regr.predict(new_updated_matrix),axis=0), \
   numpy.mean(numpy.std(labels_matrix - second_layer_regr.predict(new_updated_matrix),axis=0))
   corr_coef = numpy.corrcoef(labels_matrix.T,second_layer_regr.predict(new_updated_matrix).T)
   print 'cur_rho: ', corr_coef[1,0]
   return second_layer_regr
 
def EStep(updated_output,score_matrix,labels_matrix,second_layer_coeff,second_layer_intercept):

   labels_less_dep = labels_matrix - numpy.ones(score_matrix.shape)*second_layer_intercept
   frames_x_star = updated_output.shape[0]
   dim_x_star = updated_output.shape[1]
 
   x_star = numpy.zeros(updated_output.shape) 
   w = second_layer_coeff
   for i in range(frames_x_star):
      d_n = score_matrix[i][0,0]   
      D_n = numpy.eye(3)
      D_n[0,0], D_n[1,1] = d_n,1.0/(d_n + .01)
      w_cur = w*D_n 
      inv_mat = numpy.linalg.inv(w_cur.T*w_cur + numpy.eye(dim_x_star))
      cur_x_n = inv_mat * (w_cur.T*labels_less_dep[i,:].T+updated_output[i,:].T) 
      x_star[i,:] = cur_x_n.T
      
   return x_star 
   

def TrainEM(train_scores,train_vad_ts,train_features_ts,dev_scores,dev_vad_ts,dev_features_ts):
   # prepare data for training
   [train_window_matrix,train_labels_matrix,train_score_matrix] = PrepareData(train_scores,train_vad_ts,train_features_ts) 
   [dev_window_matrix,dev_labels_matrix,dev_score_matrix] = PrepareData(dev_scores,dev_vad_ts,dev_features_ts) 

   train_labels_val, train_labels_aro, train_labels_dom = train_labels_matrix[:,0], train_labels_matrix[:,1], train_labels_matrix[:,2]
   dev_labels_val, dev_labels_aro, dev_labels_dom = dev_labels_matrix[:,0], dev_labels_matrix[:,1], dev_labels_matrix[:,2]

   # First getting baselines
   train_bl_features = numpy.concatenate((train_window_matrix,train_score_matrix),axis=1) 
   dev_bl_features = numpy.concatenate((dev_window_matrix,dev_score_matrix),axis=1) 
   bl1_regr = linear_model.LinearRegression(n_jobs=8)
   bl1_regr.fit(train_bl_features,train_labels_val)

   bl2_regr = linear_model.LinearRegression(n_jobs=8)
   bl2_regr.fit(train_window_matrix,train_labels_val)

   # having depression features added
   train_window_matrix = train_bl_features
   dev_window_matrix = dev_bl_features

   # Initialization
   x_star = numpy.tile(train_labels_val,(1,3)) 

   count_iterations = 0
   while True:
      # M step for first layer
      first_layer_regr = linear_model.LinearRegression(n_jobs=8) 
      first_layer_regr.fit(train_window_matrix, x_star)
      updated_output = numpy.matrix(first_layer_regr.predict(train_window_matrix))
      print 'Intermediate error from updated output: ', numpy.mean(numpy.std(train_labels_val - first_layer_regr.predict(train_window_matrix),axis=0)) 

 
      # M step for second layer
      second_layer_regr = MstepSecondLayer(updated_output,train_labels_val,train_score_matrix)           
      second_layer_coeff, second_layer_intercept = numpy.matrix(second_layer_regr.coef_), second_layer_regr.intercept_ 

      # E step to find x_star
      x_star = EStep(updated_output,train_score_matrix,train_labels_val,second_layer_coeff,second_layer_intercept) 
      
      if count_iterations == 5:
         break
      count_iterations += 1

      # testing current iteration performance on dev set 
      model = [first_layer_regr, second_layer_regr]
      TestOutput7(dev_scores,dev_vad_ts,dev_features_ts,bl1_regr,bl2_regr,model)

   # testing current iteration performance on dev set 
   model = [first_layer_regr, second_layer_regr]
   TestOutput7(dev_scores,dev_vad_ts,dev_features_ts,bl1_regr,bl2_regr,model)
   
   print 'End of iteration evaluation'
   print '-----------------'
   print '-----------------'
   print ''

   return bl1_regr, bl2_regr, model 

if __name__ == "__main__":
   TrainEM(train_scores,train_vad_ts,train_features_ts)
