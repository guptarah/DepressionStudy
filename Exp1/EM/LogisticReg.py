import numpy

def encode_one_in_K(Y,K):
   N = Y.shape[0]
   Y_encoded = numpy.zeros((N,K))
    

def compute_probs(w,X,Y):
'''
variables:
N: number of instances
D: dimensionality in which X_cur, X_nn lie
K: number of classes
w: class weights shape KxD
X: features, NXD
Y: class labels \in [0,1,2,3,..]
'''
  
   N = X.shape[0] 
   D = X.shape[1]
   K = w.shape[0]

   unnorm_class_probs = numpy.exp(numpy.dot(X,w.T))
   class_normalizer = numpy.tile(numpy.sum(unnorm_class_probs,axis=1),(K,1)).T
   class_probs = numpy.divide(unnorm_class_probs,class_normalizer)

  
   Y_encoded = encode_one_in_K(Y,K) 
