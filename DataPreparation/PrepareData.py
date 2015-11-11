import numpy
import fnmatch 
import sys 
import os 

def PrepareData(input_dir,output_dir):
   depression_dir = input_dir + '/Depression/'
   emotions_dir = input_dir + '/Emotion/' 
   for partition_dir in os.listdir(depression_dir):

      print "In paritition: ", partition_dir
      depression_partition = depression_dir + '/' + partition_dir 
      emo_arousal_partition = emotions_dir + '/' + partition_dir + '/Arousal/Freeform/' 
      emo_dominance_partition = emotions_dir + '/' + partition_dir + '/Dominance/Freeform/' 
      emo_valence_partition = emotions_dir + '/' + partition_dir + '/Valence/Freeform/' 
      for file_name in os.listdir(depression_partition):
         # loading files
         depression_file = depression_partition + '/' + file_name
         print depression_file
         depression_value = numpy.genfromtxt(depression_file)      
         print depression_value

         file_id = '_'.join(file_name.split('_')[0:2])
         arousal_file_pattern = file_id + '_Freeform-AROUSAL_A1-A*'
         dominance_file_pattern = file_id + '_Freeform-DOMINANCE_A1-A*'
         valence_file_pattern = file_id + '_Freeform-VALENCE_A1-A*'
         
         for arousal_file in os.listdir(emo_arousal_partition):
            if fnmatch.fnmatch(arousal_file, arousal_file_pattern):
               arousal_data = numpy.genfromtxt(emo_arousal_partition + '/' + arousal_file,delimiter=',')    
               print arousal_file, arousal_data.shape
         for dominance_file in os.listdir(emo_dominance_partition):
            if fnmatch.fnmatch(dominance_file, dominance_file_pattern):
               print dominance_file
               dominance_data = numpy.genfromtxt(emo_dominance_partition + '/' + dominance_file,delimiter=',')    
         for valence_file in os.listdir(emo_valence_partition):
            if fnmatch.fnmatch(valence_file, valence_file_pattern):
               print valence_file
               valence_data = numpy.genfromtxt(emo_valence_partition + '/' + valence_file,delimiter=',')    
         
         # saving values
         save_dir = output_dir + '/' + file_id 
         if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
         depression_save_file = save_dir + '/depression_score'
         arousal_save_file = save_dir + '/arosal_values'
         dominance_save_file = save_dir + '/dominance_values'
         valence_save_file = save_dir + '/valence_values'
         numpy.savetxt(depression_save_file,numpy.matrix(depression_value),fmt='%d')
         numpy.savetxt(arousal_save_file,arousal_data[:,0],fmt='%f')
         numpy.savetxt(dominance_save_file,dominance_data[:,0],fmt='%f')
         numpy.savetxt(valence_save_file,valence_data[:,0],fmt='%f')


if __name__ == "__main__":
   if len(sys.argv) != 3:
      print "Incorrect usage \
            PrepareData.py InputDir OutputDir"
   else: 
      input_dir = sys.argv[1] # the Labels directory supplied by organizers
      output_dir = sys.argv[2]
      if not os.path.exists(output_dir):
         os.makedirs(output_dir) 
      PrepareData(input_dir,output_dir)
