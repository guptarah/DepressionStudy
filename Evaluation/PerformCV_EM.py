import numpy
from sklearn import linear_model
from LoadData import LoadData
import sys
sys.path.append('/home/rcf-proj/pg/guptarah/DepressionStudy/Scripts/EM')
from EStep import e_step
from MStep import m_step

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
  
   init_targets = numpy.zeros((count_data,3)) # right now mapping everything to 9 features 
   for i in range(count_data):
      current_ts = vad_ts[i]
      init_targets[i] = numpy.mean(current_ts,axis=1).T

   test_errors = numpy.zeros((0,1))
   for i in range(count_splits):
      test_indices = range(split_size*i,(split_size*i)+split_size)
      train_indices = list(set(range(0,count_data)).difference(set(test_indices)))
      
      test_scores = depression_scores[test_indices,:] 
      train_scores = depression_scores[train_indices,:] 
      test_vad_ts = [vad_ts[j] for j in test_indices] 
      train_vad_ts = [vad_ts[j] for j in train_indices] 
      x_nn = init_targets[train_indices,:] # initializing x_nn 
      w = numpy.ones((init_targets.shape[1],1)) # initializeing w   
 
      not_converged = True 
      while not_converged:
         # Estep
         x_est = e_step(w,train_scores,x_nn) #x_nn are statisticals 
         
         # Mstep
         [w,x_nn] = m_step(x_est,train_scores,train_vad_ts) 
         

   print 'RMSE in scores: ', numpy.std(depression_scores)
   print 'Baseline RMSE: ', numpy.sqrt(numpy.dot(test_errors.T,test_errors)/count_data) 

if __name__ == "__main__":
   PerformCV()
