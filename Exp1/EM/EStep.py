import numpy

def e_step(w,Y,X_nn):
	"""Variables:
	N: number of instances
	D: dimensionality in which X_cur, X_nn lie
	w: weight vectors of shape Dx1
	X_cur: of shape NxD; the estimate of X from previous E step
	X_nn: of shape NxD; the estimate of X from nn only 
	Y: Regression labels Nx1
	X_n = (w*w.T + I) f(x_n) + (w*w.T + I)*y_n*w
	"""
	D = w.shape[0]
	N = Y.shape[0]
	#a = numpy.matrix(numpy.tile(Y, (1,D)))
	common_term = numpy.linalg.inv(numpy.dot(w,w.T) + numpy.eye(D))
	term1 = numpy.dot(common_term,X_nn.T)
	
	a = numpy.matrix(numpy.tile(Y, (1,D)))
	b = numpy.matrix(numpy.tile(w.T, (N,1)))
	term2 = numpy.multiply(a,b)
	term2 = numpy.dot(common_term,term2.T)
	X_est = (term1+term2).T

   # normalizeing X_est
	X_est -= numpy.min(X_est) 
	X_est /= numpy.max(X_est) 

	return X_est

if __name__ == "__main__":
   e_step(w,Y,X_nn) 
