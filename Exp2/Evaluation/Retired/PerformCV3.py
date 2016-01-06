import numpy
from sklearn import linear_model
from LoadData import LoadData
from TrainEMLR3 import TrainEM
from TrainEMLR3 import PrepareData 
from numpy import matlib

def PrintResults(ground_truth,b1_output,b2_output,model_output): 
   # printing results for supplied test split
   print 'Results for baseline 1'
   print 'Test error: ', numpy.std(ground_truth - b1_output,axis=0), \
   numpy.mean(numpy.std(ground_truth - b1_output,axis=0))
   correlation_matrix = numpy.corrcoef(ground_truth.T,b1_output.T)
   print 'Test rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3 

   print 'Results for baseline 2'
   print 'Test error: ', numpy.std(ground_truth - b2_output,axis=0), \
   numpy.mean(numpy.std(ground_truth - b2_output,axis=0))
   correlation_matrix = numpy.corrcoef(ground_truth.T,b2_output.T)
   print 'Test rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3 

   print 'Results for model'
   print 'Test error: ', numpy.std(ground_truth - model_output,axis=0), \
   numpy.mean(numpy.std(ground_truth - model_output,axis=0))
   correlation_matrix = numpy.corrcoef(ground_truth.T,model_output.T)
   print 'Test rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3 

def TestOutput(test_scores,test_vad_ts,test_features_ts,baseline1,baseline2,model):
   [window_matrix,labels_matrix,score_matrix] = PrepareData(test_scores,test_vad_ts,test_features_ts)
   bl_features = numpy.concatenate((window_matrix,score_matrix),axis=1)

   # Baseline outputs
   b1_output = baseline1.predict(bl_features)
   b2_output = baseline2.predict(window_matrix)

   # model output
   l1_reg = model[0]
   l2_reg = model[1]
   layer1_output = numpy.matrix(l1_reg.predict(bl_features))

   num_feats = layer1_output.shape[1]
   layer2_input = matlib.zeros((layer1_output.shape[0],layer1_output.shape[1]+1)) # this will have
   layer2_input[:,0] = numpy.multiply(layer1_output[:,0],score_matrix) 
   layer2_input[:,1] = numpy.divide(layer1_output[:,1],(score_matrix+.01*numpy.ones(score_matrix.shape))) 
   layer2_input[:,3] = score_matrix
   model_output = l2_reg.predict(layer2_input)

   # printing results for supplied test split
   print 'Results for iteration'
   PrintResults(labels_matrix,b1_output,b2_output,model_output)

   return labels_matrix,b1_output,b2_output,model_output

def PerformCV():
   # Load the data
   [depression_scores,vad_ts,feature_ts]=LoadData('/home/rcf-proj/pg/guptarah/DepressionStudy/Data/PreparedData')

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
      train_indices = list(set(range(0,count_data)).difference(set(test_indices)))
      
      test_scores = depression_scores[test_indices,:] 
      train_scores = depression_scores[train_indices,:] 
      test_vad_ts = [vad_ts[j] for j in test_indices]
      train_vad_ts = [vad_ts[j] for j in train_indices]
      test_features_ts = [feature_ts[j] for j in test_indices]
      train_features_ts = [feature_ts[j] for j in train_indices]
    
      # Training
      print ''
      print '======================'
      print 'Training for iteration: ',i 
      [baseline1, baseline2, model] = TrainEM(train_scores,train_vad_ts,train_features_ts) 

      # Testing
      print ''
      print ''
      print 'Testing for iteration: ',i 
      [ground_truth,b1_output,b2_output,model_output] = TestOutput(test_scores,test_vad_ts,test_features_ts,baseline1,baseline2,model)
      
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
