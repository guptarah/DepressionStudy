import numpy
from numpy import matlib
import sys
from sklearn import linear_model
from sklearn import decomposition 

def PrepareData(train_scores,train_vad_ts,train_features_ts):
   n_steps = 1 # number of past/future frames used for current frame prediction
  
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
