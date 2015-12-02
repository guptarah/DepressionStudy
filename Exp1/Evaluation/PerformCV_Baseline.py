import numpy
from sklearn import linear_model
from LoadData import LoadData

def PerformCV():
   # Load the data
   [depression_scores,vad_ts]=LoadData('/home/rcf-proj/pg/guptarah/DepressionStudy/Data/PreparedData')

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
      test_baseline_features = baseline_features[test_indices,:] 

      train_scores = depression_scores[train_indices,:] 
      train_baseline_features = baseline_features[train_indices,:] 
      
      lr = linear_model.LinearRegression()
      lr.fit(train_baseline_features,train_scores)
      test_pred = lr.predict(test_baseline_features)
      test_errors = numpy.concatenate((test_errors,(test_scores - test_pred)),axis=0)

   print 'RMSE in scores: ', numpy.std(depression_scores)
   print 'Baseline RMSE: ', numpy.sqrt(numpy.dot(test_errors.T,test_errors)/count_data) 

if __name__ == "__main__":
   PerformCV()
