import numpy
import sys
sys.path.append('/home/rcf-proj/pg/guptarah/DepressionStudy/Scripts/theano-rnn/')
from rnn import MetaRNN 

def PrepareData(train_vad_ts,train_features_ts):
   n_steps = 100 # matrix of the form n_steps, n_seq, feature_vector
   
   # first getting a count of sequences
   count_seq = 0
   for i in range(len(train_vad_ts)):
      count_seq += int(train_vad_ts[i].shape[1]/n_steps)

   input_dim = train_features_ts[0].shape[1]
   output_dim = train_vad_ts[0].shape[0] 
   input_matrix = numpy.zeros((count_seq,n_steps,input_dim))
   target_matrix = numpy.zeros((count_seq,n_steps,output_dim)) 

   total_count_seq = 0 
   for i in range(len(train_vad_ts)):
      cur_count_seq = int(train_vad_ts[i].shape[1]/n_steps)
      for j in range(cur_count_seq):
         input_sequence = train_features_ts[i][j*n_steps:(j+1)*n_steps,:]
         target_sequence = (train_vad_ts[i].T)[j*n_steps:(j+1)*n_steps,:]
         for t in range(n_steps):
            input_matrix[total_count_seq+j,t,:] = input_sequence[t,:] 
            target_matrix[total_count_seq+j,t,:] = target_sequence[t,:] 
      total_count_seq += cur_count_seq

   return input_matrix, target_matrix

def TrainEM(train_scores,train_vad_ts,train_features_ts):
   # prepare data for training
   [input_matrix,target_matrix] = PrepareData(train_vad_ts,train_features_ts) 

   n_in = input_matrix.shape[2]
   n_out = target_matrix.shape[2]
   n_hidden = 10
   not_converged = True
   while not_converged:
      model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,\
                    learning_rate=0.001, learning_rate_decay=0.999,\
                    n_epochs=400, activation='tanh')

      model.fit(input_matrix, target_matrix, validation_frequency=1000)



if __name__ == "__main__":
   TrainEM(train_scores,train_vad_ts,train_features_ts)
