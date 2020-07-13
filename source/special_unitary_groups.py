import sys
import os
import scipy.misc
import numpy as np


def pauli(label):
	if label == 'z':
		return np.array([[1.0, 0.0], [0.0, -1.0]])
	elif label == 'x':
		return np.array([[0.0, 1.0], [1.0, 0.0]])
	elif label == '-':
		return np.array([[0.0, 0.0], [1.0, 0.0]])
	elif label == '+':
		return np.array([[0.0, 1.0], [0.0, 0.0]])
	elif label == 'y':
		return np.array([[0.0, -1.0j], [1.0j, 0.0]])
	else:
		raise ValueError("label must be either 'x', 'y', 'z', '+' or '-'. "
						 "label given was {}".format(label))