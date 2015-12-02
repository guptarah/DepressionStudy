import numpy
from sklearn import linear_model
from LoadData import LoadData

def PerformCV():
   # Load the data
   [depression_scores,vad_ts,feature_ts]=LoadData('/home/rcf-proj/pg/guptarah/DepressionStudy/Data/PreparedData')

   split_size = 10
   count_data = len(vad_ts) 
   if numpy.mod(count_data,split_size) != 0:
      print 'provide #splits that can divide #datapoits'
      return 0
   count_splits = count_data/split_size 

   # computing statisticals for baseline
  
   baseline_features = numpy.zeros((count_data,12)) # right now I have 9 features 
   for i in range(count_data):
      current_ts = vad_ts[i]
      mean_feature = numpy.mean(current_ts,axis=1).T
      std_feature = numpy.std(current_ts,axis=1).T
      exp_feature = numpy.mean(numpy.exp(current_ts),axis=1).T
      min_feature = numpy.amin(numpy.exp(current_ts),axis=1).T
      max_feature = numpy.amax(numpy.exp(current_ts),axis=1).T
      baseline_features[i] = numpy.concatenate((mean_feature, std_feature, exp_feature, min_feature),axis=1)

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
     
      model = TrainEM(train_scores,train_vad_ts,train_features_ts) 

   print 'RMSE in scores: ', numpy.std(depression_scores)
   print 'Baseline RMSE: ', numpy.sqrt(numpy.dot(test_errors.T,test_errors)/count_data) 

if __name__ == "__main__":
   PerformCV()
