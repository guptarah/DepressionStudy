import numpy
from sklearn import linear_model
from LoadData import LoadData
from TrainEMLR6 import TrainEM
from numpy import matlib
from TestOutput import TestOutput6 
from TestOutput import PrintResults 

def PerformCV():
   # Load the data
   [depression_scores,vad_ts,feature_ts]=LoadData('/home/rcf-proj/pg/guptarah/DepressionStudy/DataNW/PreparedData')

   split_size = 5 
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
      [baseline1, baseline2, model] = TrainEM(train_scores,train_vad_ts,train_features_ts,dev_scores,dev_vad_ts,dev_features_ts)  

      # Testing
      print ''
      print ''
      print 'Testing for iteration: ',i 
      [ground_truth,b1_output,b2_output,model_output] = TestOutput6(test_scores,test_vad_ts,test_features_ts,baseline1,baseline2,model)
      
      all_ground_truth = numpy.concatenate((all_ground_truth,ground_truth),axis=0) 
      all_b1_output = numpy.concatenate((all_b1_output,b1_output),axis=0) 
      all_b2_output = numpy.concatenate((all_b2_output,b2_output),axis=0) 
      all_model_output = numpy.concatenate((all_model_output,model_output),axis=0) 

      print ''
      print 'Running results on test set'
      PrintResults(all_ground_truth,all_b1_output,all_b2_output,all_model_output) 
      print ''
      print '======================'
      print ''


   print ''
   print 'Final results'
   PrintResults(all_ground_truth,all_b1_output,all_b2_output,all_model_output) 
   

if __name__ == "__main__":
   PerformCV()
