import numpy
from sklearn import linear_model
from LoadDataSynced import LoadData
from TrainEMLR8 import PrepareData 
from numpy import matlib
from TestOutput import TestOutput  

def PrintResults(all_ground_truth,all_b1_output,all_b2_output,all_b3_output,all_b4_output):
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

def PerformCV():
   # Load the data
   [depression_scores,vad_ts,feature_ts]=LoadData('/home/rcf-proj/pg/guptarah/DepressionStudy/DataNW/PreparedData')

   split_size = 10
   count_data = len(vad_ts) 
   if numpy.mod(count_data,split_size) != 0:
      print 'provide #splits that can divide #datapoits'
      return 0
   count_splits = count_data/split_size 

   all_test_ground_truth, all_test_b1_output, all_test_b2_output, all_test_b3_output, all_test_b4_output= \
   numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3))
   all_dev_ground_truth, all_dev_b1_output, all_dev_b2_output, all_dev_b3_output, all_dev_b4_output= \
   numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3))

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
   
      train_bl1_features = train_window_matrix
      dev_bl1_features = dev_window_matrix
      test_bl1_features = test_window_matrix

      train_bl2_features = numpy.concatenate((train_window_matrix,train_score_matrix),axis=1) 
      dev_bl2_features = numpy.concatenate((dev_window_matrix,dev_score_matrix),axis=1) 
      test_bl2_features = numpy.concatenate((test_window_matrix,test_score_matrix),axis=1) 

      train_bl3_features = numpy.multiply(train_window_matrix,numpy.tile(train_score_matrix,(1,train_window_matrix.shape[1]))) 
      dev_bl3_features = numpy.multiply(dev_window_matrix,numpy.tile(dev_score_matrix,(1,dev_window_matrix.shape[1]))) 
      test_bl3_features = numpy.multiply(test_window_matrix,numpy.tile(test_score_matrix,(1,test_window_matrix.shape[1]))) 

      train_bl4_features = numpy.divide(train_window_matrix,numpy.tile(.01*numpy.ones(train_score_matrix.shape)+train_score_matrix,(1,train_window_matrix.shape[1]))) 
      dev_bl4_features = numpy.divide(dev_window_matrix,numpy.tile(.01*numpy.ones(dev_score_matrix.shape)+dev_score_matrix,(1,dev_window_matrix.shape[1]))) 
      test_bl4_features = numpy.divide(test_window_matrix,numpy.tile(.01*numpy.ones(test_score_matrix.shape)+test_score_matrix,(1,test_window_matrix.shape[1]))) 

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
      
      bl4_regr = linear_model.LinearRegression(n_jobs=8)
      bl4_regr.fit(train_bl4_features,train_labels_matrix)
      dev_bl4_predict = bl4_regr.predict(dev_bl4_features)
      test_bl4_predict = bl4_regr.predict(test_bl4_features)
      
      all_test_ground_truth = numpy.concatenate((all_test_ground_truth,test_labels_matrix),axis=0) 
      all_test_b1_output = numpy.concatenate((all_test_b1_output,test_bl1_predict),axis=0) 
      all_test_b2_output = numpy.concatenate((all_test_b2_output,test_bl2_predict),axis=0) 
      all_test_b3_output = numpy.concatenate((all_test_b3_output,test_bl3_predict),axis=0) 
      all_test_b4_output = numpy.concatenate((all_test_b4_output,test_bl4_predict),axis=0) 

      all_dev_ground_truth = numpy.concatenate((all_dev_ground_truth,dev_labels_matrix),axis=0) 
      all_dev_b1_output = numpy.concatenate((all_dev_b1_output,dev_bl1_predict),axis=0) 
      all_dev_b2_output = numpy.concatenate((all_dev_b2_output,dev_bl2_predict),axis=0) 
      all_dev_b3_output = numpy.concatenate((all_dev_b3_output,dev_bl3_predict),axis=0) 
      all_dev_b4_output = numpy.concatenate((all_dev_b4_output,dev_bl4_predict),axis=0) 


      combined_dev_results = .8*all_dev_b2_output + .2*all_dev_b3_output + .2*all_dev_b4_output 
      combined_test_results = .8*all_test_b2_output + .8*all_test_b3_output + .8*all_test_b4_output 

      # Printing results
      print 'Dev set results'
      PrintResults(all_dev_ground_truth,all_dev_b1_output,all_dev_b2_output,all_dev_b3_output,all_dev_b4_output)
      print ' ' 
      print 'Dev set results: weighted combination'
      PrintResults(all_dev_ground_truth,combined_dev_results,all_dev_b2_output,all_dev_b3_output,all_dev_b4_output)
     
      print ' ' 
      print 'Test set results'
      PrintResults(all_test_ground_truth,all_test_b1_output,all_test_b2_output,all_test_b3_output,all_test_b4_output)
      print ' ' 
      print 'Test set results: weighted combination'
      PrintResults(all_test_ground_truth,combined_test_results,all_test_b2_output,all_test_b3_output,all_test_b4_output)


if __name__ == "__main__":
   PerformCV()

