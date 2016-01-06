import numpy
from numpy import matlib
from PrepareData import PrepareData

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

def PrintResults7(ground_truth,b1_output,b2_output,model_output): 
   # printing results for supplied test split
   print 'Results for baseline 1'
   print 'Test error: ', numpy.std(ground_truth - b1_output,axis=0) 
   correlation_matrix = numpy.corrcoef(ground_truth.T,b1_output.T)
   print 'Test rho: ', correlation_matrix[1,0]

   print 'Results for baseline 2'
   print 'Test error: ', numpy.std(ground_truth - b2_output,axis=0)
   correlation_matrix = numpy.corrcoef(ground_truth.T,b2_output.T)
   print 'Test rho: ', correlation_matrix[1,0]

   print 'Results for model'
   print 'Test error: ', numpy.std(ground_truth - model_output,axis=0)
   correlation_matrix = numpy.corrcoef(ground_truth.T,model_output.T)
   print 'Test rho: ', correlation_matrix[1,0]

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

def TestOutput6(test_scores,test_vad_ts,test_features_ts,baseline1,baseline2,model):
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
   layer2_input = matlib.zeros((layer1_output.shape[0],layer1_output.shape[1])) # this will have
   layer2_input[:,0] = numpy.multiply(layer1_output[:,0],score_matrix) 
   layer2_input[:,1] = numpy.divide(layer1_output[:,1],(score_matrix+.01*numpy.ones(score_matrix.shape))) 
   model_output = l2_reg.predict(layer2_input)

   # printing results for supplied test split
   print 'Results for iteration'
   PrintResults(labels_matrix,b1_output,b2_output,model_output)

   return labels_matrix,b1_output,b2_output,model_output

def TestOutput7(test_scores,test_vad_ts,test_features_ts,baseline1,baseline2,model):
   [window_matrix,labels_matrix,score_matrix] = PrepareData(test_scores,test_vad_ts,test_features_ts)
   labels_matrix_val, labels_matrix_aro, labels_matrix_dom = \
   labels_matrix[:,0], labels_matrix[:,1], labels_matrix[:,2]   

   bl_features = numpy.concatenate((window_matrix,score_matrix),axis=1)
   # Baseline outputs
   b1_output = baseline1.predict(bl_features)
   b2_output = baseline2.predict(window_matrix)

   # model output
   l1_reg = model[0]
   l2_reg = model[1]
   layer1_output = numpy.matrix(l1_reg.predict(bl_features))

   num_feats = layer1_output.shape[1]
   layer2_input = matlib.zeros((layer1_output.shape[0],layer1_output.shape[1])) # this will have
   layer2_input[:,0] = numpy.multiply(layer1_output[:,0],score_matrix) 
   layer2_input[:,1] = numpy.divide(layer1_output[:,1],(score_matrix+.01*numpy.ones(score_matrix.shape))) 
   model_output = l2_reg.predict(layer2_input)

   # printing results for supplied test split
   print 'Results for iteration'
   PrintResults7(labels_matrix_val,b1_output,b2_output,model_output)

   return labels_matrix_val,b1_output,b2_output,model_output

def TestOutputPCA(test_scores,test_vad_ts,test_features_ts,baseline1,baseline2,model,pca):
   [window_matrix,labels_matrix,score_matrix] = PrepareData(test_scores,test_vad_ts,test_features_ts)

   window_matrix = pca.transform(window_matrix)
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
