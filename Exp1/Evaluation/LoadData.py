import numpy
import os 
import sys 

def LoadData(input_dir):
   depression_scores = numpy.zeros((0,1))
   vad_ts = []
   for patient_dir in os.listdir(input_dir):
      print 'In dir: ', patient_dir        
      depression_file = input_dir + '/' + patient_dir + '/depression_score'  
      depression_score = numpy.genfromtxt(depression_file,dtype='int')
      depression_scores = numpy.concatenate((depression_scores,numpy.matrix(depression_score)),axis=0)

      valence_file = input_dir + '/' + patient_dir + '/valence_values' 
      valence_values = numpy.genfromtxt(valence_file,delimiter=',')
      arousal_file = input_dir + '/' + patient_dir + '/arosal_values' 
      arousal_values = numpy.genfromtxt(arousal_file,delimiter=',')
      dominance_file = input_dir + '/' + patient_dir + '/dominance_values' 
      dominance_values = numpy.genfromtxt(dominance_file,delimiter=',')

      if (valence_values.shape[0] != arousal_values.shape[0]) or (valence_values.shape[0] != dominance_values.shape[0]):
         print valence_values.shape, arousal_values.shape, dominance_values.shape

      min_frames = numpy.amin([valence_values.shape[0],arousal_values.shape[0],dominance_values.shape[0]])
      valence_values = valence_values[0:min_frames]
      arousal_values = arousal_values[0:min_frames]
      dominance_values = dominance_values[0:min_frames]


      vad_matrix = numpy.concatenate((numpy.matrix(valence_values),numpy.matrix(arousal_values),numpy.matrix(dominance_values)))
      vad_ts.append(vad_matrix)

   return depression_scores,vad_ts

if __name__ == "__main__":
   input_dir = sys.argv[1]
   LoadData(input_dir)
