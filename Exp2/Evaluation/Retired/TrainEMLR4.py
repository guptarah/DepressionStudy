import numpy
from numpy import matlib
import sys
from sklearn import linear_model
from sklearn import decomposition 
from TestOutput import TestOutput 

def PrepareData(train_scores,train_vad_ts,train_features_ts):
   n_steps = 0 # number of past/future frames used for current frame prediction
  
   num_feats = train_features_ts[0].shape[1] 
   width_window_matrix = num_feats*(2*n_steps+1)
   window_matrix = numpy.zeros((0, width_window_matrix)) 
   labels_matrix = numpy.zeros((0, 3)) 
   score_matrix = numpy.zeros((0, 1)) 
   for i in range(len(train_vad_ts)):
      cur_score = train_scores[i]

      cur_feat_matrix = train_features_ts[i]
      # normalizeing cur_feat_matrix
#      mean_cur_feat_matrix = numpy.mean(cur_feat_matrix,axis=0) 
#      std_cur_feat_matrix = numpy.std(cur_feat_matrix,axis=0)
#      std_cur_feat_matrix[std_cur_feat_matrix == 0] = 1 
#      cur_feat_matrix = numpy.divide((cur_feat_matrix - numpy.tile(mean_cur_feat_matrix,(cur_feat_matrix.shape[0],1))), \
#      numpy.tile(std_cur_feat_matrix,(cur_feat_matrix.shape[0],1)))

      cur_label_matrix = train_vad_ts[i].T
      num_frames = cur_feat_matrix.shape[0]

      len_cur_window_matrix = num_frames # need to remove top n bottown n_steps later
      cur_window_matrix = numpy.zeros((len_cur_window_matrix,width_window_matrix))
      if n_steps > 0: 
         for j in range(-1*n_steps,n_steps+1):
            cur_window_matrix[:,(j+n_steps)*num_feats:(j+n_steps+1)*num_feats] = numpy.roll(cur_feat_matrix,j,axis=0)
         cur_window_matrix = cur_window_matrix[n_steps:-1*n_steps,:] 
         cur_label_matrix = cur_label_matrix[n_steps:-1*n_steps,:] 
         cur_score_vector = numpy.matrix(numpy.ones((num_frames-2*n_steps,1))) * cur_score
      else: 
         cur_window_matrix = cur_feat_matrix

      cur_score_vector = numpy.matrix(numpy.ones((num_frames-2*n_steps,1))) * cur_score
 
      window_matrix = numpy.concatenate((window_matrix,cur_window_matrix),axis=0) 
      labels_matrix = numpy.concatenate((labels_matrix,cur_label_matrix),axis=0) 
      score_matrix = numpy.concatenate((score_matrix,cur_score_vector),axis=0) 

   # normalizing score matrix by dividing by 100
   score_matrix = .01*score_matrix   

   return window_matrix, labels_matrix, score_matrix

def MstepSecondLayer(updated_output,labels_matrix,score_matrix):
   num_feats = updated_output.shape[1]
   new_updated_matrix = matlib.zeros((updated_output.shape[0],updated_output.shape[1]+1)) # this will have
   # each feat multiplied by score and score as feature

   new_updated_matrix[:,0] = numpy.multiply(updated_output[:,0],score_matrix) 
   new_updated_matrix[:,1] = numpy.divide(updated_output[:,1],(score_matrix+.01*numpy.ones(score_matrix.shape))) 
   new_updated_matrix[:,3] = score_matrix

   second_layer_regr = linear_model.LinearRegression(fit_intercept=True)
   second_layer_regr.fit(new_updated_matrix,labels_matrix)

   # print training error for this iteration
   print 'cur_error: ', numpy.std(labels_matrix - second_layer_regr.predict(new_updated_matrix),axis=0), \
   numpy.mean(numpy.std(labels_matrix - second_layer_regr.predict(new_updated_matrix),axis=0))
   correlation_matrix = numpy.corrcoef(labels_matrix.T,second_layer_regr.predict(new_updated_matrix).T)
   print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3
   return second_layer_regr
 
def EStep(updated_output,score_matrix,labels_matrix,second_layer_coeff,second_layer_intercept):

   labels_less_dep = labels_matrix - score_matrix*second_layer_coeff[:,-1].T - numpy.ones(score_matrix.shape)*second_layer_intercept
   frames_x_star = updated_output.shape[0]
   dim_x_star = updated_output.shape[1]
 
   x_star = numpy.zeros(updated_output.shape) 
   w = second_layer_coeff[:,:-1] 
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

   # perform PCA on data
   # pca = decomposition.PCA(n_components=200)
   # pca.fit(train_window_matrix)
   # train_window_matrix = pca.transform(train_window_matrix)
   # dev_window_matrix = pca.transform(dev_window_matrix)

   # First getting baselines
   train_bl_features = numpy.concatenate((train_window_matrix,train_score_matrix),axis=1) 
   dev_bl_features = numpy.concatenate((dev_window_matrix,dev_score_matrix),axis=1) 
   bl1_regr = linear_model.LinearRegression(n_jobs=8)
   bl1_regr.fit(train_bl_features,train_labels_matrix)

   print 'Training error on baseline 1: ', numpy.std(train_labels_matrix - bl1_regr.predict(train_bl_features),axis=0), \
   numpy.mean(numpy.std(train_labels_matrix - bl1_regr.predict(train_bl_features),axis=0))
   correlation_matrix = numpy.corrcoef(train_labels_matrix.T,bl1_regr.predict(train_bl_features).T)
   print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3
   print 'Dev error on baseline 1: ', numpy.std(dev_labels_matrix- bl1_regr.predict(dev_bl_features),axis=0), \
   numpy.mean(numpy.std(dev_labels_matrix - bl1_regr.predict(dev_bl_features),axis=0))
   correlation_matrix = numpy.corrcoef(dev_labels_matrix.T,bl1_regr.predict(dev_bl_features).T)
   print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3
   
   bl2_regr = linear_model.LinearRegression(n_jobs=8)
   bl2_regr.fit(train_window_matrix,train_labels_matrix)

   print 'Training error on baseline 2: ', numpy.std(train_labels_matrix - bl2_regr.predict(train_window_matrix),axis=0), \
   numpy.mean(numpy.std(train_labels_matrix - bl2_regr.predict(train_window_matrix),axis=0))
   correlation_matrix = numpy.corrcoef(train_labels_matrix.T,bl2_regr.predict(train_window_matrix).T)
   print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3
   print 'Dev error on baseline 2: ', numpy.std(dev_labels_matrix - bl2_regr.predict(dev_window_matrix),axis=0), \
   numpy.mean(numpy.std(dev_labels_matrix - bl2_regr.predict(dev_window_matrix),axis=0))
   correlation_matrix = numpy.corrcoef(dev_labels_matrix.T,bl2_regr.predict(dev_window_matrix).T)
   print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3
   
   # having depression features added
   train_window_matrix = train_bl_features
   dev_window_matrix = dev_bl_features

   # Initialization
   x_star = train_labels_matrix   

   count_iterations = 0
   while True:
      # M step for first layer
      first_layer_regr = linear_model.LinearRegression(n_jobs=8) 
      first_layer_regr.fit(train_window_matrix, x_star)
      updated_output = numpy.matrix(first_layer_regr.predict(train_window_matrix))
      print 'Intermediate error from updated output: ', numpy.mean(numpy.std(train_labels_matrix - first_layer_regr.predict(train_window_matrix),axis=0)) 

 
      # M step for second layer
      second_layer_regr = MstepSecondLayer(updated_output,train_labels_matrix,train_score_matrix)           
      second_layer_coeff, second_layer_intercept = numpy.matrix(second_layer_regr.coef_), second_layer_regr.intercept_ 

      # E step to find x_star
      x_star = EStep(updated_output,train_score_matrix,train_labels_matrix,second_layer_coeff,second_layer_intercept) 
      
      if count_iterations == 4:
         break
      count_iterations += 1

      # testing current iteration performance on dev set 
      model = [first_layer_regr, second_layer_regr]
      TestOutput(dev_scores,dev_vad_ts,dev_features_ts,bl1_regr,bl2_regr,model)


   model = [first_layer_regr, second_layer_regr]
   return bl1_regr, bl2_regr, model 

if __name__ == "__main__":
   TrainEM(train_scores,train_vad_ts,train_features_ts)
