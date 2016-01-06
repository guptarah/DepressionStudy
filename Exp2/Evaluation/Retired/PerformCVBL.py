import numpy
from sklearn import linear_model
from LoadDataSynced import LoadData
from TrainEMLR4 import TrainEM
from TrainEMLR4 import PrepareData 
from numpy import matlib
from TestOutput import TestOutput  
from TestOutput import PrintResults 

def PerformCV():
   # Load the data
   [depression_scores,vad_ts,feature_ts]=LoadData('/home/rcf-proj/pg/guptarah/DepressionStudy/DataNW/PreparedData')

   split_size = 10
   count_data = len(vad_ts) 
   if numpy.mod(count_data,split_size) != 0:
      print 'provide #splits that can divide #datapoits'
      return 0
   count_splits = count_data/split_size 

   all_ground_truth, all_b1_output, all_b2_output, all_model_output = \
   numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3))
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
   
      train_bl_features = numpy.concatenate((train_window_matrix,train_score_matrix),axis=1) 
      dev_bl_features = numpy.concatenate((dev_window_matrix,dev_score_matrix),axis=1) 
      test_bl_features = numpy.concatenate((test_window_matrix,test_score_matrix),axis=1) 

      bl1_regr = linear_model.LinearRegression(n_jobs=8)
      bl1_regr.fit(train_bl_features,train_labels_matrix)

      print 'Dev error on baseline 1: ', numpy.std(dev_labels_matrix- bl1_regr.predict(dev_bl_features),axis=0), \
      numpy.mean(numpy.std(dev_labels_matrix - bl1_regr.predict(dev_bl_features),axis=0))
      correlation_matrix = numpy.corrcoef(dev_labels_matrix.T,bl1_regr.predict(dev_bl_features).T)
      print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
      (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3
      print 'Test error on baseline 1: ', numpy.std(test_labels_matrix- bl1_regr.predict(test_bl_features),axis=0), \
      numpy.mean(numpy.std(test_labels_matrix - bl1_regr.predict(test_bl_features),axis=0))
      correlation_matrix = numpy.corrcoef(test_labels_matrix.T,bl1_regr.predict(test_bl_features).T)
      print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
      (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3
      

      bl2_regr = linear_model.LinearRegression(n_jobs=8)
      bl2_regr.fit(train_window_matrix,train_labels_matrix)
      print 'Dev error on baseline 2: ', numpy.std(dev_labels_matrix - bl2_regr.predict(dev_window_matrix),axis=0), \
      numpy.mean(numpy.std(dev_labels_matrix - bl2_regr.predict(dev_window_matrix),axis=0))
      correlation_matrix = numpy.corrcoef(dev_labels_matrix.T,bl2_regr.predict(dev_window_matrix).T)
      print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
      (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3
      print 'Test error on baseline 2: ', numpy.std(test_labels_matrix - bl2_regr.predict(test_window_matrix),axis=0), \
      numpy.mean(numpy.std(test_labels_matrix - bl2_regr.predict(test_window_matrix),axis=0))
      correlation_matrix = numpy.corrcoef(test_labels_matrix.T,bl2_regr.predict(test_window_matrix).T)
      print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
      (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3

      ground_truth = test_labels_matrix
      b1_output = bl1_regr.predict(test_bl_features)
      b2_output = bl2_regr.predict(test_window_matrix)

      all_ground_truth = numpy.concatenate((all_ground_truth,ground_truth),axis=0) 
      all_b1_output = numpy.concatenate((all_b1_output,b1_output),axis=0) 
      all_b2_output = numpy.concatenate((all_b2_output,b2_output),axis=0) 

      print ''
      print 'Running results on test set'
      print 'Test error on baseline 1: ', numpy.std(all_ground_truth- all_b1_output,axis=0), \
      numpy.mean(numpy.std(all_ground_truth - all_b1_output,axis=0))
      correlation_matrix = numpy.corrcoef(all_ground_truth.T,all_b1_output.T)
      print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
      (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3
      print 'Test error on baseline 2: ', numpy.std(all_ground_truth- all_b2_output,axis=0), \
      numpy.mean(numpy.std(all_ground_truth - all_b2_output,axis=0))
      correlation_matrix = numpy.corrcoef(all_ground_truth.T,all_b2_output.T)
      print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
      (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3
      print ''

      print '======================'
      print ''


   print ''
   print 'Final results'
   print 'Test error on baseline 1: ', numpy.std(all_ground_truth- all_b1_output,axis=0), \
   numpy.mean(numpy.std(all_ground_truth - all_b1_output,axis=0))
   correlation_matrix = numpy.corrcoef(all_ground_truth.T,all_b1_output.T)
   print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3
   print 'Test error on baseline 2: ', numpy.std(all_ground_truth- all_b2_output,axis=0), \
   numpy.mean(numpy.std(all_ground_truth - all_b2_output,axis=0))
   correlation_matrix = numpy.corrcoef(all_ground_truth.T,all_b2_output.T)
   print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3
   

if __name__ == "__main__":
   PerformCV()

