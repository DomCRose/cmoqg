import numpy as np
import scipy.linalg

def random_real_vector(dimension, number = 1):
	"""Uniformly generates a normalized real vector."""
	random_unnormalized_real_vectors = np.random.randn(number, dimension)
	norms = np.linalg.norm(random_unnormalized_real_vectors, axis = 1)
	return random_unnormalized_real_vectors / norms[:,np.newaxis]

def random_complex_vector(dimension, number = 1):
	"""Uniformly generates a normalized complex vector."""
	random_unnormalized_complex_vectors = (random_real_vector(dimension, number)
										  + 1j*random_real_vector(dimension, number))
	norms = np.linalg.norm(random_unnormalized_complex_vectors, axis = 1)
	return random_unnormalized_complex_vectors / norms[:,np.newaxis]

def random_orthogonal_matrix(dimension, mode = 'uniform', variance = 0.001):
	"""Generates an orthogonal matrix.

	If mode is set to 'uniform', generates an orthogonal matrix from 
	the Haar (uniform) measure for the orthogonal group of the 
	specified dimension, via construction of the QR decomposition of a 
	random general linear matrix generated from a gaussian 
	distribution for the components: the orthogonal matrix Q for such 
	a random operator has the desired distribution.

	See "The Efficient Generation of Random Orthogonal Matrices with 
	an Application to Condition Estimators" by G. W. Stewart for proof
	of this relation, and a more efficient approach if only the action
	of the orthognal matrix is required.

	If mode is set to 'tangent', generates a random skew-symmetric 
	matrix by combining a matrix - with elements from a gaussian with 
	mean 0 and the specified variance - with its own transpose, then 
	exponentiates to get an orthogonal matrix.
	"""
	if mode == 'uniform':
		random_normal_matrix = np.random.randn(dimension,dimension)
		random_orthogonal, random_upper_triangle = np.linalg.qr(random_normal_matrix)
		return random_orthogonal
	elif mode == 'tangent':
		random_normal_matrix = variance * np.random.randn(dimension,dimension)
		random_symmetric_matrix = random_normal_matrix - random_normal_matrix.T
		return scipy.linalg.expm(random_symmetric_matrix)
	else:
		raise ValueError("mode must be either 'uniform' or 'tangent'. "
						 "mode given was {}".format(mode))