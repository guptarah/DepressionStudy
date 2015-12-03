import numpy
from sklearn import linear_model
from LoadData import LoadData
from TrainEMLR import TrainEM


def TestOutput(test_scores,test_vad_ts,test_features_ts,baseline1,baseline2,model):
   [window_matrix,labels_matrix,score_matrix] = PrepareData(test_scores,test_vad_ts,test_features_ts)
   bl_features = numpy.concatenate((window_matrix,score_matrix),axis=1)

   # Baseline outputs
   b1_output = baseline1.predict(bl_features)
   b2_output = baseline2.predict(window_matrix)


   # model output
   l1_reg = model[0]
   l2_reg = model[1]
   layer1_output = l1_reg.predict(bl_features) 

   num_feats = layer1_output.shape[1]
   layer2_input = matlib.zeros((updated_output.shape[0],updated_output.shape[1]+1)) # this will have
   for i in range(num_feats):
      layer2_input[:,i] = numpy.multiply(updated_output[:,i],score_matrix)
   layer2_input[:,i+1] = score_matrix
   model_output = l2_reg.predict(layer2_input)

   # printing results for supplied test split
   print 'Results for baseline 1'
   print 'iteration test error: ', numpy.std(labels_matrix - b1_output,axis=0), \
   numpy.mean(numpy.std(labels_matrix - b1_output,axis=0))
   correlation_matrix = numpy.corrcoef(labels_matrix.T,b1_output.T)
   print 'iteration test rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3 

   print 'Results for baseline 2'
   print 'iteration test error: ', numpy.std(labels_matrix - b2_output,axis=0), \
   numpy.mean(numpy.std(labels_matrix - b2_output,axis=0))
   correlation_matrix = numpy.corrcoef(labels_matrix.T,b2_output.T)
   print 'iteration test rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3 

   print 'Results for model'
   print 'iteration test error: ', numpy.std(labels_matrix - model_output,axis=0), \
   numpy.mean(numpy.std(labels_matrix - model_output,axis=0))
   correlation_matrix = numpy.corrcoef(labels_matrix.T,model_output.T)
   print 'iteration test rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3 

   return b1_output,b2_output,model_output

def PerformCV():
   # Load the data
   [depression_scores,vad_ts,feature_ts]=LoadData('/home/rcf-proj/pg/guptarah/DepressionStudy/Data/PreparedData')

   split_size = 10
   count_data = len(vad_ts) 
   if numpy.mod(count_data,split_size) != 0:
      print 'provide #splits that can divide #datapoits'
      return 0
   count_splits = count_data/split_size 

   test_errors = numpy.zeros((0,1))
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
      [baseline1, baseline2, model] = TrainEM(train_scores,train_vad_ts,train_features_ts) 

      # Testing
      [b1_output,b2_output,model_output] = PrepareData(train_scores,train_vad_ts,train_features_ts)
 
   
   print 'RMSE in scores: ', numpy.std(depression_scores)
   print 'Baseline RMSE: ', numpy.sqrt(numpy.dot(test_errors.T,test_errors)/count_data) 

if __name__ == "__main__":
   PerformCV()
