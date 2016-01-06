import numpy
import os 
import sys 

def DSFeats(feature_values,num_frames):
   ds_factor = float(feature_values.shape[0])/num_frames
   win_len = 4
   filtered_matrix = numpy.zeros(feature_values.shape)
   for feat_id in range(0,feature_values.shape[1]):
      cur_feat = feature_values[:,feat_id]
      filtered_matrix[:,feat_id] = numpy.convolve(cur_feat,numpy.ones(win_len)/float(win_len),mode='same')

   sampled_features = numpy.zeros((num_frames,feature_values.shape[1]))
   for i in range(0,num_frames):
      take_audio_frame = int(numpy.around(i*ds_factor,decimals=0))
      sampled_features[i,:] = filtered_matrix[take_audio_frame,:]

   return sampled_features

def LoadData(input_dir_aud,input_dir_vid):
   depression_scores = numpy.zeros((0,1))
   vad_ts = []
   audio_feature_ts = []
   video_feature_ts = []
   for patient_dir in os.listdir(input_dir_aud):
      print 'In dir: ', patient_dir        
      depression_file = input_dir_aud + '/' + patient_dir + '/depression_score'  
      depression_score = numpy.genfromtxt(depression_file,dtype='int')
      depression_scores = numpy.concatenate((depression_scores,numpy.matrix(depression_score)),axis=0)

      valence_file = input_dir_aud + '/' + patient_dir + '/valence_values' 
      valence_values = numpy.genfromtxt(valence_file,delimiter=',')
      arousal_file = input_dir_aud + '/' + patient_dir + '/arousal_values' 
      arousal_values = numpy.genfromtxt(arousal_file,delimiter=',')
      dominance_file = input_dir_aud + '/' + patient_dir + '/dominance_values' 
      dominance_values = numpy.genfromtxt(dominance_file,delimiter=',')

      audio_features_file = input_dir_aud + '/' + patient_dir + '/feature_values' 
      audio_feature_values = numpy.genfromtxt(audio_features_file,delimiter=',')
      video_features_file = input_dir_vid + '/' + patient_dir + '/feature_values' 
      video_feature_values = numpy.genfromtxt(video_features_file,delimiter=',')
      
      #if (valence_values.shape[0] != arousal_values.shape[0]) or (valence_values.shape[0] != dominance_values.shape[0]):
      #   print valence_values.shape, arousal_values.shape, dominance_values.shape

      min_frames = numpy.amin([valence_values.shape[0],arousal_values.shape[0],dominance_values.shape[0],audio_feature_values.shape[0],video_feature_values.shape[0]])
      valence_values = valence_values[0:min_frames]
      arousal_values = arousal_values[0:min_frames]
      dominance_values = dominance_values[0:min_frames]


      vad_matrix = numpy.concatenate((numpy.matrix(valence_values),numpy.matrix(arousal_values),numpy.matrix(dominance_values)))
      vad_ts.append(vad_matrix)

      audio_feature_matrix = audio_feature_values[0:min_frames,:] 
      video_feature_matrix = video_feature_values[0:min_frames,:] 
      audio_feature_ts.append(audio_feature_matrix)
      video_feature_ts.append(video_feature_matrix)

   return depression_scores,vad_ts,audio_feature_ts,video_feature_ts

if __name__ == "__main__":
   input_dir = sys.argv[1]
   LoadData(input_dir_aud,input_dir_vid)
