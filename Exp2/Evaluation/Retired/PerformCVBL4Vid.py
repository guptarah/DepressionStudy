import numpy
from sklearn import linear_model
from LoadDataSynced import LoadData
from PrepareData import PrepareData 
from numpy import matlib

def ZNormalize(train_window_matrix,dev_window_matrix,test_window_matrix):
   # removing constant features
   train_std = numpy.std(train_window_matrix,axis=0)
   remove_feats = numpy.where(train_std == 0)[0]   
   train_window_matrix = numpy.delete(train_window_matrix,remove_feats,1)
   dev_window_matrix = numpy.delete(dev_window_matrix,remove_feats,1)
   test_window_matrix = numpy.delete(test_window_matrix,remove_feats,1)

   # znormalizing
   train_mean = numpy.mean(train_window_matrix,axis=0)
   train_std = numpy.std(train_window_matrix,axis=0)

   train_norm = train_window_matrix - numpy.tile(train_mean,(train_window_matrix.shape[0],1))
   train_norm = numpy.divide(train_norm,numpy.tile(train_std,(train_window_matrix.shape[0],1)))

   dev_norm = dev_window_matrix - numpy.tile(train_mean,(dev_window_matrix.shape[0],1))
   dev_norm = numpy.divide(dev_norm,numpy.tile(train_std,(dev_window_matrix.shape[0],1)))

   test_norm = test_window_matrix - numpy.tile(train_mean,(test_window_matrix.shape[0],1))
   test_norm = numpy.divide(test_norm,numpy.tile(train_std,(test_window_matrix.shape[0],1)))

   return train_norm,dev_norm,test_norm

def PrintResults(all_ground_truth,all_b1_output,all_b2_output,all_b3_output):
   print 'Error on baseline 1: ', numpy.std(all_ground_truth - all_b1_output,axis=0), \
   numpy.mean(numpy.std(all_ground_truth - all_b1_output,axis=0))
   correlation_matrix = numpy.corrcoef(all_ground_truth.T,all_b1_output.T)
   print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3

   print 'Error on baseline 2: ', numpy.std(all_ground_truth - all_b2_output,axis=0), \
   numpy.mean(numpy.std(all_ground_truth - all_b2_output,axis=0))
   correlation_matrix = numpy.corrcoef(all_ground_truth.T,all_b2_output.T)
   print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3

   print 'Error on baseline 3: ', numpy.std(all_ground_truth - all_b3_output,axis=0), \
   numpy.mean(numpy.std(all_ground_truth - all_b3_output,axis=0))
   correlation_matrix = numpy.corrcoef(all_ground_truth.T,all_b3_output.T)
   print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3

def PerformCV():
   # Load the data
   [depression_scores,vad_ts,feature_ts]=LoadData('/home/rcf-proj/pg/guptarah/DepressionStudy/DataNW/PreparedDataVideo')

   split_size = 10
   count_data = len(vad_ts) 
   if numpy.mod(count_data,split_size) != 0:
      print 'provide #splits that can divide #datapoits'
      return 0
   count_splits = count_data/split_size 

   all_test_ground_truth, all_test_b1_output, all_test_b2_output, all_test_b3_output, all_test_b4_output, all_test_combined_output= \
   numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3))
   all_dev_ground_truth, all_dev_b1_output, all_dev_b2_output, all_dev_b3_output, all_dev_b4_output, all_dev_combined_output= \
   numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3))

   for i in range(count_splits):
      test_indices = range(split_size*i,(split_size*i)+split_size)
      dev_indices = range(split_size*(numpy.mod(i+1,count_splits)),split_size*(numpy.mod(i+1,count_splits)+1))
      train_indices = list(set(range(0,count_data)).difference(set(test_indices)).difference(set(dev_indices)))
      
      test_scores = depression_scores[test_indices,:] 
      dev_scores = depression_scores[dev_indices,:] 
      train_scores = depression_scores[train_indices,:] 
      test_vad_ts = [vad_ts[j] for j in test_indices]
      dev_vad_ts = [vad_ts[j] for j in dev_indices]
      train_vad_ts = [vad_ts[j] for j in train_indices]
      test_features_ts = [feature_ts[j] for j in test_indices]
      dev_features_ts = [feature_ts[j] for j in dev_indices]
      train_features_ts = [feature_ts[j] for j in train_indices]
    
      # Training
      print ''
      print '======================'
      print 'Training for iteration: ',i 
      [train_window_matrix,train_labels_matrix,train_score_matrix] = PrepareData(train_scores,train_vad_ts,train_features_ts) 
      [dev_window_matrix,dev_labels_matrix,dev_score_matrix] = PrepareData(dev_scores,dev_vad_ts,dev_features_ts) 
      [test_window_matrix,test_labels_matrix,test_score_matrix] = PrepareData(test_scores,test_vad_ts,test_features_ts) 

      [train_window_matrix,dev_window_matrix,test_window_matrix] = ZNormalize(train_window_matrix,dev_window_matrix,test_window_matrix)
 
      num_train_frames = train_window_matrix.shape[0] 
      train_bl1_features = train_window_matrix
      dev_bl1_features = dev_window_matrix
      test_bl1_features = test_window_matrix

      train_scaled_features = numpy.multiply(train_window_matrix,numpy.tile(train_score_matrix,(1,train_window_matrix.shape[1]))) 
      dev_scaled_features = numpy.multiply(dev_window_matrix,numpy.tile(dev_score_matrix,(1,dev_window_matrix.shape[1]))) 
      test_scaled_features = numpy.multiply(test_window_matrix,numpy.tile(test_score_matrix,(1,test_window_matrix.shape[1]))) 

      train_bl2_features = numpy.concatenate((train_window_matrix,train_score_matrix),axis=1) 
      dev_bl2_features = numpy.concatenate((dev_window_matrix,dev_score_matrix),axis=1) 
      test_bl2_features = numpy.concatenate((test_window_matrix,test_score_matrix),axis=1) 

      train_bl3_features = numpy.concatenate((train_window_matrix,train_score_matrix,train_scaled_features),axis=1) 
      dev_bl3_features = numpy.concatenate((dev_window_matrix,dev_score_matrix,dev_scaled_features),axis=1) 
      test_bl3_features = numpy.concatenate((test_window_matrix,test_score_matrix,test_scaled_features),axis=1) 

      bl1_regr = linear_model.LinearRegression(n_jobs=8)
      bl1_regr.fit(train_bl1_features,train_labels_matrix)
      dev_bl1_predict = bl1_regr.predict(dev_bl1_features)
      test_bl1_predict = bl1_regr.predict(test_bl1_features)

      bl2_regr = linear_model.LinearRegression(n_jobs=8)
      bl2_regr.fit(train_bl2_features,train_labels_matrix)
      dev_bl2_predict = bl2_regr.predict(dev_bl2_features)
      test_bl2_predict = bl2_regr.predict(test_bl2_features)
      
      bl3_regr = linear_model.LinearRegression(n_jobs=8)
      bl3_regr.fit(train_bl3_features,train_labels_matrix)
      dev_bl3_predict = bl3_regr.predict(dev_bl3_features)
      test_bl3_predict = bl3_regr.predict(test_bl3_features)

      all_test_ground_truth = numpy.concatenate((all_test_ground_truth,test_labels_matrix),axis=0) 
      all_test_b1_output = numpy.concatenate((all_test_b1_output,test_bl1_predict),axis=0) 
      all_test_b2_output = numpy.concatenate((all_test_b2_output,test_bl2_predict),axis=0) 
      all_test_b3_output = numpy.concatenate((all_test_b3_output,test_bl3_predict),axis=0) 

      all_dev_ground_truth = numpy.concatenate((all_dev_ground_truth,dev_labels_matrix),axis=0) 
      all_dev_b1_output = numpy.concatenate((all_dev_b1_output,dev_bl1_predict),axis=0) 
      all_dev_b2_output = numpy.concatenate((all_dev_b2_output,dev_bl2_predict),axis=0) 
      all_dev_b3_output = numpy.concatenate((all_dev_b3_output,dev_bl3_predict),axis=0) 

      # Printing results
      print 'Dev set results'
      PrintResults(all_dev_ground_truth,all_dev_b1_output,all_dev_b2_output,all_dev_b3_output)
     
      print ' ' 
      print 'Test set results'
      PrintResults(all_test_ground_truth,all_test_b1_output,all_test_b2_output,all_test_b3_output)


if __name__ == "__main__":
   PerformCV()

