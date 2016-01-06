import numpy
from sklearn import linear_model
from LoadDataSyncedAudVid import LoadData
from PrepareData import PrepareData 
from numpy import matlib

def ZNormalize(train_window_matrix,dev_window_matrix,test_window_matrix):
   # removing constant features
   train_std = numpy.std(train_window_matrix,axis=0)
   remove_feats = numpy.where(train_std == 0)[0]   
   train_window_matrix = numpy.delete(train_window_matrix,remove_feats,1)
   dev_window_matrix = numpy.delete(dev_window_matrix,remove_feats,1)
   test_window_matrix = numpy.delete(test_window_matrix,remove_feats,1)

   # znormalizing
   train_mean = numpy.mean(train_window_matrix,axis=0)
   train_std = numpy.std(train_window_matrix,axis=0)

   train_norm = train_window_matrix - numpy.tile(train_mean,(train_window_matrix.shape[0],1))
   train_norm = numpy.divide(train_norm,numpy.tile(train_std,(train_window_matrix.shape[0],1)))

   dev_norm = dev_window_matrix - numpy.tile(train_mean,(dev_window_matrix.shape[0],1))
   dev_norm = numpy.divide(dev_norm,numpy.tile(train_std,(dev_window_matrix.shape[0],1)))

   test_norm = test_window_matrix - numpy.tile(train_mean,(test_window_matrix.shape[0],1))
   test_norm = numpy.divide(test_norm,numpy.tile(train_std,(test_window_matrix.shape[0],1)))

   return train_norm,dev_norm,test_norm

def PrintResults(all_ground_truth,all_b1_output):
   print 'Error on baseline 1: ', numpy.std(all_ground_truth - all_b1_output,axis=0), \
   numpy.mean(numpy.std(all_ground_truth - all_b1_output,axis=0))
   correlation_matrix = numpy.corrcoef(all_ground_truth.T,all_b1_output.T)
   print 'cur_rho: ', correlation_matrix[0,3], correlation_matrix[1,4], correlation_matrix[2,5], \
   (correlation_matrix[0,3]+correlation_matrix[1,4]+correlation_matrix[2,5])/3
   return numpy.sum([correlation_matrix[0,3],correlation_matrix[1,4],correlation_matrix[2,5]])
   

def PerformCV():
   # Load the data
   [depression_scores,vad_ts,feature_ts_aud,feature_ts_vid]=LoadData('/home/rcf-proj/pg/guptarah/DepressionStudy/DataNW/PreparedData','/home/rcf-proj/pg/guptarah/DepressionStudy/DataNW/PreparedDataVideo')

   split_size = 10
   count_data = len(vad_ts) 
   if numpy.mod(count_data,split_size) != 0:
      print 'provide #splits that can divide #datapoits'
      return 0
   count_splits = count_data/split_size 

   all_test_ground_truth, all_test_fused_output, all_test_audio_output, all_test_video_output, all_test_b4_output, all_test_combined_output= \
   numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3))
   all_dev_ground_truth, all_dev_fused_output, all_dev_audio_output, all_dev_video_output, all_dev_b4_output, all_dev_combined_output= \
   numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3)), numpy.zeros((0,3))

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
      test_features_ts_aud = [feature_ts_aud[j] for j in test_indices]
      dev_features_ts_aud = [feature_ts_aud[j] for j in dev_indices]
      train_features_ts_aud = [feature_ts_aud[j] for j in train_indices]
      test_features_ts_vid = [feature_ts_vid[j] for j in test_indices]
      dev_features_ts_vid = [feature_ts_vid[j] for j in dev_indices]
      train_features_ts_vid = [feature_ts_vid[j] for j in train_indices]
    
      # Training
      print ''
      print '======================'
      print 'Training for iteration: ',i

      # preparing audio features 
      [train_window_matrix_audio,train_labels_matrix,train_score_matrix] = PrepareData(train_scores,train_vad_ts,train_features_ts_aud) 
      [dev_window_matrix_audio,dev_labels_matrix,dev_score_matrix] = PrepareData(dev_scores,dev_vad_ts,dev_features_ts_aud) 
      [test_window_matrix_audio,test_labels_matrix,test_score_matrix] = PrepareData(test_scores,test_vad_ts,test_features_ts_aud) 
      [train_window_matrix_audio,dev_window_matrix_audio,test_window_matrix_audio] = ZNormalize(train_window_matrix_audio,dev_window_matrix_audio,test_window_matrix_audio)
 
      # preparing audio features 
      [train_window_matrix_video,train_labels_matrix,train_score_matrix] = PrepareData(train_scores,train_vad_ts,train_features_ts_vid) 
      [dev_window_matrix_video,dev_labels_matrix,dev_score_matrix] = PrepareData(dev_scores,dev_vad_ts,dev_features_ts_vid) 
      [test_window_matrix_video,test_labels_matrix,test_score_matrix] = PrepareData(test_scores,test_vad_ts,test_features_ts_vid) 
      [train_window_matrix_video,dev_window_matrix_video,test_window_matrix_video] = ZNormalize(train_window_matrix_video,dev_window_matrix_video,test_window_matrix_video)

      # Preparing features
      num_train_frames = train_window_matrix_audio.shape[0] 
      train_scaled_features_audio = numpy.multiply(train_window_matrix_audio,numpy.tile(train_score_matrix,(1,train_window_matrix_audio.shape[1]))) 
      dev_scaled_features_audio = numpy.multiply(dev_window_matrix_audio,numpy.tile(dev_score_matrix,(1,dev_window_matrix_audio.shape[1]))) 
      test_scaled_features_audio = numpy.multiply(test_window_matrix_audio,numpy.tile(test_score_matrix,(1,test_window_matrix_audio.shape[1]))) 
      train_added_features_audio = numpy.concatenate((train_window_matrix_audio,train_score_matrix),axis=1) 
      dev_added_features_audio = numpy.concatenate((dev_window_matrix_audio,dev_score_matrix),axis=1) 
      test_added_features_audio = numpy.concatenate((test_window_matrix_audio,test_score_matrix),axis=1) 
      train_scaled_features_video = numpy.multiply(train_window_matrix_video,numpy.tile(train_score_matrix,(1,train_window_matrix_video.shape[1]))) 
      dev_scaled_features_video = numpy.multiply(dev_window_matrix_video,numpy.tile(dev_score_matrix,(1,dev_window_matrix_video.shape[1]))) 
      test_scaled_features_video = numpy.multiply(test_window_matrix_video,numpy.tile(test_score_matrix,(1,test_window_matrix_video.shape[1]))) 
      train_added_features_video = numpy.concatenate((train_window_matrix_video,train_score_matrix),axis=1) 
      dev_added_features_video = numpy.concatenate((dev_window_matrix_video,dev_score_matrix),axis=1) 
      test_added_features_video = numpy.concatenate((test_window_matrix_video,test_score_matrix),axis=1) 



      train_audio_features = numpy.concatenate((train_scaled_features_audio,train_added_features_audio),axis=1)
      train_labels_l1 = train_labels_matrix
      train_audio_features_l1 = train_audio_features
      dev_audio_features = numpy.concatenate((dev_scaled_features_audio,dev_added_features_audio),axis=1) 
      test_audio_features = numpy.concatenate((test_scaled_features_audio,test_added_features_audio),axis=1) 
      train_video_features = numpy.concatenate((train_scaled_features_video,train_added_features_video),axis=1)
      train_video_features_l1 = train_video_features
      dev_video_features = numpy.concatenate((dev_scaled_features_video,dev_added_features_video),axis=1) 
      test_video_features = numpy.concatenate((test_scaled_features_video,test_added_features_video),axis=1) 
   

      # first layer
      audio_regr_l1 = linear_model.LinearRegression(n_jobs=8)
      audio_regr_l1.fit(train_audio_features_l1,train_labels_l1)
      dev_audio_predict_l1 = audio_regr_l1.predict(dev_audio_features)
      test_audio_predict_l1 = audio_regr_l1.predict(test_audio_features)
      video_regr_l1 = linear_model.LinearRegression(n_jobs=8)
      video_regr_l1.fit(train_video_features_l1,train_labels_l1)
      dev_video_predict_l1 = video_regr_l1.predict(dev_video_features)
      test_video_predict_l1 = video_regr_l1.predict(test_video_features)

      # modality-wise results 
      all_test_ground_truth = numpy.concatenate((all_test_ground_truth,test_labels_matrix),axis=0) 
      all_test_audio_output = numpy.concatenate((all_test_audio_output,test_audio_predict_l1),axis=0) 
      all_test_video_output = numpy.concatenate((all_test_video_output,test_video_predict_l1),axis=0) 

      all_dev_ground_truth = numpy.concatenate((all_dev_ground_truth,dev_labels_matrix),axis=0) 
      all_dev_audio_output = numpy.concatenate((all_dev_audio_output,dev_audio_predict_l1),axis=0) 
      all_dev_video_output = numpy.concatenate((all_dev_video_output,dev_video_predict_l1),axis=0) 

      print 'Results on audio'
      print ''
      print 'Dev set results'
      PrintResults(all_dev_ground_truth,all_dev_audio_output)
     
      print '' 
      print 'Test set results'
      PrintResults(all_test_ground_truth,all_test_audio_output)

      print 'Results on video'
      print '' 
      print 'Dev set results'
      PrintResults(all_dev_ground_truth,all_dev_video_output)
     
      print '' 
      print 'Test set results'
      PrintResults(all_test_ground_truth,all_test_video_output)

      # tuning weights
      dev_results = numpy.zeros((11,1))
      for alpha in range(11):
         wt = .1*alpha
         dev_combined = (wt*dev_video_predict_l1) + ((1-wt)*dev_audio_predict_l1) 
         dev_results[alpha] = PrintResults(dev_labels_matrix,dev_combined)
      best_alpha = numpy.argmax(dev_results,axis=0) 
      wt = .1*best_alpha
      print 'chosen wts: ', wt
      dev_combined = (wt*dev_video_predict_l1) + ((1-wt)*dev_audio_predict_l1) 
      test_combined = (wt*test_video_predict_l1) + ((1-wt)*test_audio_predict_l1) 

      # tuning window length for lp filtering
      #for alpha in range(11):
      #   win_len = 5*alpha
      #   conv_filt = (1.0/win_len)*numpy.ones((1,win_len))
         

      all_dev_fused_output = numpy.concatenate((all_dev_fused_output,dev_combined),axis=0) 
      all_test_fused_output = numpy.concatenate((all_test_fused_output,test_combined),axis=0) 
      print 'Test set results: combined'
      PrintResults(test_labels_matrix,test_combined)

   print 'Final Test set results: audio'
   PrintResults(all_test_ground_truth,all_test_audio_output)
      
   print 'Final Test set results: video'
   PrintResults(all_test_ground_truth,all_test_video_output)

   print 'Final Test set results: combined'
   PrintResults(all_test_ground_truth,all_test_fused_output)

if __name__ == "__main__":
   PerformCV()

