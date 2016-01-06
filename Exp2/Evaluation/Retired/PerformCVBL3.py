import numpy
from sklearn import linear_model
from LoadDataSynced import LoadData
from PrepareData import PrepareData 
from numpy import matlib
from TestOutput import TestOutput  

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

def PrintResults(all_ground_truth,all_b1_output,all_b2_output,all_b3_output,all_b4_output,all_combined_output):
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

   print 'Error on baseline 4: ', numpy.std(all_ground_truth - all_b4_output,axis=0), \
   numpy.mean(numpy.std(all_ground_truth - all_b4_output,axis=0))
   correlation_matrix = numpy.corrcoef(all_ground_truth.T,all_b4_output.T)
   print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3

   print 'Error on combined: ', numpy.std(all_ground_truth - all_combined_output,axis=0), \
   numpy.mean(numpy.std(all_ground_truth - all_combined_output,axis=0))
   correlation_matrix = numpy.corrcoef(all_ground_truth.T,all_combined_output.T)
   print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3

def PerformCV():
   # Load the data
   [depression_scores,vad_ts,feature_ts]=LoadData('/home/rcf-proj/pg/guptarah/DepressionStudy/DataNW/PreparedData')

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
 
      num_train_frames = train_window_matrix.shape[0] 
      train_bl1_features_l1 = train_window_matrix[0:int(.8*num_train_frames),:]
      train_bl1_features_l2 = train_window_matrix[int(.8*num_train_frames):-1,:]
      train_bl1_labels_l1 = train_labels_matrix[0:int(.8*num_train_frames),:]
      train_bl1_labels_l2 = train_labels_matrix[int(.8*num_train_frames):-1,:]
      dev_bl1_features = dev_window_matrix
      test_bl1_features = test_window_matrix

      train_bl2_features = numpy.concatenate((train_window_matrix,train_score_matrix),axis=1) 
      train_bl2_features_l1 = train_bl2_features[0:int(.8*num_train_frames),:]
      train_bl2_features_l2 = train_bl2_features[int(.8*num_train_frames):-1,:]
      dev_bl2_features = numpy.concatenate((dev_window_matrix,dev_score_matrix),axis=1) 
      test_bl2_features = numpy.concatenate((test_window_matrix,test_score_matrix),axis=1) 

      train_bl3_features = numpy.multiply(train_window_matrix,numpy.tile(train_score_matrix,(1,train_window_matrix.shape[1]))) 
      train_bl3_features_l1 = train_bl3_features[0:int(.8*num_train_frames),:]
      train_bl3_features_l2 = train_bl3_features[int(.8*num_train_frames):-1,:]
      dev_bl3_features = numpy.multiply(dev_window_matrix,numpy.tile(dev_score_matrix,(1,dev_window_matrix.shape[1]))) 
      test_bl3_features = numpy.multiply(test_window_matrix,numpy.tile(test_score_matrix,(1,test_window_matrix.shape[1]))) 

      # normalize data for log
      train_window_matrix_log, dev_window_matrix_log, test_window_matrix_log = ZNormalize(train_window_matrix,dev_window_matrix,test_window_matrix)
      min_bias = numpy.amin([numpy.amin(train_window_matrix_log),numpy.amin(dev_window_matrix_log),numpy.amin(test_window_matrix_log)]) - 1
      train_bl4_features_log = numpy.log(train_window_matrix_log - min_bias*numpy.ones(train_window_matrix_log.shape))
      train_bl4_features_divisor = numpy.log(10*numpy.tile(numpy.ones(train_score_matrix.shape)+200*train_score_matrix,(1,train_window_matrix_log.shape[1])))
      train_bl4_features = numpy.divide(train_bl4_features_log,train_bl4_features_divisor)
      train_bl4_features_l1 = train_bl4_features[0:int(.8*num_train_frames),:]
      train_bl4_features_l2 = train_bl4_features[int(.8*num_train_frames):-1,:]
      dev_bl4_features_log = numpy.log(dev_window_matrix_log - min_bias*numpy.ones(dev_window_matrix_log.shape))
      dev_bl4_features_divisor = numpy.log(10*numpy.tile(numpy.ones(dev_score_matrix.shape)+200*dev_score_matrix,(1,dev_window_matrix_log.shape[1])))
      dev_bl4_features = numpy.divide(dev_bl4_features_log,dev_bl4_features_divisor)
      test_bl4_features_log = numpy.log(test_window_matrix_log - min_bias*numpy.ones(test_window_matrix_log.shape))
      test_bl4_features_divisor = numpy.log(10*numpy.tile(numpy.ones(test_score_matrix.shape)+200*test_score_matrix,(1,test_window_matrix_log.shape[1])))
      test_bl4_features = numpy.divide(test_bl4_features_log,test_bl4_features_divisor)

      bl1_regr = linear_model.LinearRegression(n_jobs=8)
      bl1_regr.fit(train_bl1_features_l1,train_bl1_labels_l1)
      train_bl1_predict_l2 = bl1_regr.predict(train_bl1_features_l2)
      dev_bl1_predict = bl1_regr.predict(dev_bl1_features)
      test_bl1_predict = bl1_regr.predict(test_bl1_features)

      bl2_regr = linear_model.LinearRegression(n_jobs=8)
      bl2_regr.fit(train_bl2_features_l1,train_bl1_labels_l1)
      train_bl2_predict_l2 = bl2_regr.predict(train_bl2_features_l2)
      dev_bl2_predict = bl2_regr.predict(dev_bl2_features)
      test_bl2_predict = bl2_regr.predict(test_bl2_features)
      
      bl3_regr = linear_model.LinearRegression(n_jobs=8)
      bl3_regr.fit(train_bl3_features_l1,train_bl1_labels_l1)
      train_bl3_predict_l2 = bl3_regr.predict(train_bl3_features_l2)
      dev_bl3_predict = bl3_regr.predict(dev_bl3_features)
      test_bl3_predict = bl3_regr.predict(test_bl3_features)
      
      bl4_regr = linear_model.LinearRegression(n_jobs=8)
      bl4_regr.fit(train_bl4_features_l1,train_bl1_labels_l1)
      train_bl4_predict_l2 = bl4_regr.predict(train_bl4_features_l2)
      dev_bl4_predict = bl4_regr.predict(dev_bl4_features)
      test_bl4_predict = bl4_regr.predict(test_bl4_features)

      # training combiner
      #combiner_features = numpy.concatenate((train_bl2_predict_l2,train_bl3_predict_l2,train_bl4_predict_l2),axis=1)
      combiner_features = numpy.concatenate((train_bl2_predict_l2,train_bl3_predict_l2),axis=1)
      combiner_regr = linear_model.LinearRegression(n_jobs=8)
      combiner_regr.fit(combiner_features,train_bl1_labels_l2)
      #combined_dev_features = numpy.concatenate((dev_bl2_predict,dev_bl3_predict,dev_bl4_predict),axis=1)
      combined_dev_features = numpy.concatenate((dev_bl2_predict,dev_bl3_predict),axis=1)
      combined_dev_predict = combiner_regr.predict(combined_dev_features)
      #combined_test_features = numpy.concatenate((test_bl2_predict,test_bl3_predict,test_bl4_predict),axis=1)
      combined_test_features = numpy.concatenate((test_bl2_predict,test_bl3_predict),axis=1)
      combined_test_predict = combiner_regr.predict(combined_test_features)
      
      all_test_ground_truth = numpy.concatenate((all_test_ground_truth,test_labels_matrix),axis=0) 
      all_test_b1_output = numpy.concatenate((all_test_b1_output,test_bl1_predict),axis=0) 
      all_test_b2_output = numpy.concatenate((all_test_b2_output,test_bl2_predict),axis=0) 
      all_test_b3_output = numpy.concatenate((all_test_b3_output,test_bl3_predict),axis=0) 
      all_test_b4_output = numpy.concatenate((all_test_b4_output,test_bl4_predict),axis=0) 
      all_test_combined_output = numpy.concatenate((all_test_combined_output,combined_test_predict),axis=0) 
      wtd_sum_test_predict = all_test_b2_output + all_test_b3_output 


      all_dev_ground_truth = numpy.concatenate((all_dev_ground_truth,dev_labels_matrix),axis=0) 
      all_dev_b1_output = numpy.concatenate((all_dev_b1_output,dev_bl1_predict),axis=0) 
      all_dev_b2_output = numpy.concatenate((all_dev_b2_output,dev_bl2_predict),axis=0) 
      all_dev_b3_output = numpy.concatenate((all_dev_b3_output,dev_bl3_predict),axis=0) 
      all_dev_b4_output = numpy.concatenate((all_dev_b4_output,dev_bl4_predict),axis=0) 
      all_dev_combined_output = numpy.concatenate((all_dev_combined_output,combined_dev_predict),axis=0) 
      wtd_sum_dev_predict = all_dev_b2_output + all_dev_b3_output 

      # Printing results
      print 'Dev set results'
      PrintResults(all_dev_ground_truth,all_dev_b1_output,all_dev_b2_output,all_dev_b3_output,all_dev_b4_output,all_dev_combined_output)
      print ' ' 
      PrintResults(all_dev_ground_truth,all_dev_b1_output,all_dev_b2_output,all_dev_b3_output,all_dev_b4_output,wtd_sum_dev_predict)
     
      print ' ' 
      print 'Test set results'
      PrintResults(all_test_ground_truth,all_test_b1_output,all_test_b2_output,all_test_b3_output,all_test_b4_output,all_test_combined_output)
      print ' ' 
      PrintResults(all_test_ground_truth,all_test_b1_output,all_test_b2_output,all_test_b3_output,all_test_b4_output,wtd_sum_test_predict)


   print 'Wtd results:'
#   for alpha in range(10):
#      for beta in range(10-alpha):
#         wtd_sum_test_predict = .1*((10-alpha-beta)*all_test_b2_output) + (alpha*all_test_b3_output) + (beta*all_dev_b4_output)
#         wtd_sum_dev_predict = .1*((10-alpha-beta)*all_dev_b2_output) + (alpha*all_dev_b3_output) + (beta*all_dev_b4_output)
#         print 'alpha, beta: ', alpha, beta
#         print 'Dev results: ' 
#         PrintResults(all_dev_ground_truth,all_dev_b1_output,all_dev_b2_output,all_dev_b3_output,all_dev_b4_output,wtd_sum_dev_predict)
#         print 'Test results: ' 
#         PrintResults(all_test_ground_truth,all_test_b1_output,all_test_b2_output,all_test_b3_output,all_test_b4_output,wtd_sum_test_predict)
   for alpha in range(100):
         wtd_sum_test_predict = .01*(((100-alpha)*all_test_b2_output) + (alpha*all_test_b3_output)) 
         wtd_sum_dev_predict = .01*(((100-alpha)*all_dev_b2_output) + (alpha*all_dev_b3_output)) 
         print 'alpha, beta: ', alpha, beta
         print 'Dev results: ' 
         PrintResults(all_dev_ground_truth,all_dev_b1_output,all_dev_b2_output,all_dev_b3_output,all_dev_b4_output,wtd_sum_dev_predict)
         print 'Test results: ' 
         PrintResults(all_test_ground_truth,all_test_b1_output,all_test_b2_output,all_test_b3_output,all_test_b4_output,wtd_sum_test_predict)


if __name__ == "__main__":
   PerformCV()

