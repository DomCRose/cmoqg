import sys
import os
import scipy.misc
import scipy.linalg as linalg
import numpy as np


def bosonic_operator(bosonic_type, fock_truncation):
	"""Generates the required operator on a truncated fock space."""
	operator = np.zeros((fock_truncation+1, fock_truncation+1), dtype = complex)
	if bosonic_type == '+':
		for i in range(fock_truncation):
			operator[i+1][i] = np.sqrt(i+1)
		return operator
	elif bosonic_type == '-':
		for i in range(fock_truncation):
			operator[i][i+1] = np.sqrt(i+1)
		return operator
	elif bosonic_type == 'n':
		for i in range(fock_truncation):
			operator[i][i] = i
		return operator
	else:
		raise ValueError("bosonic_type must be either '+', '-' or 'n'. "
						 "bosonic_type given was {}".format(bosonic_type))

def coherent_state(coherent_value, fock_truncation):
	"""Creates a normalized coherent state on a truncated fock space."""
	index = np.arange(fock_truncation+1)
	state = coherent_value**index/np.sqrt(scipy.misc.factorial(index))
	return state/np.linalg.norm(state)

def displacement_operator(displacement, fock_truncation):
	"""Creates a displacement operator on a truncated fock space."""
	annihilation_operator = bosonic_operator('-', fock_truncation)
	creation_operator = bosonic_operator('+', fock_truncation)
	return linalg.expm(displacement*creation_operator
					   - np.conjugate(displacement)*annihilation_operator)

class wigner(object):

	"""Stores operators for the efficient calculation of wigner distributions.

	"""

	def __init__(self, fock_truncation, support_parameters = {}):
		self.annihilation_operator = bosonic_operator('-', fock_truncation)
		self.creation_operator = bosonic_operator('+', fock_truncation)
		if 'limit' in support_parameters:
			self.limit = support_parameters['limit']
		else:
			self.limit = 1
		if 'step_size' in support_parameters:
			self.step_size = support_parameters['step_size']
		else:
			self.step_size = 0.1
		self.steps = int(2*self.limit / self.step_size)
		self.position_displacement = self._displacement_operator(self.step_size)
		self.position_displacement_hc = np.conjugate(self.position_displacement).T
		self.momentum_displacement = self._displacement_operator(self.step_size*1j)
		self.momentum_displacement_hc = np.conjugate(self.momentum_displacement).T
		self.alternating_sign = (-1)**np.arange(fock_truncation+1)

	def _displacement_operator(self, displacement):
		return linalg.expm(displacement*self.creation_operator
						   - np.conjugate(displacement)*self.annihilation_operator)

	def _update_displacements(self):
		self.position_displacement = self._displacement_operator(self.step_size)
		self.position_displacement_hc = np.conjugate(self.position_displacement).T
		self.momentum_displacement = self._displacement_operator(self.step_size*1j)
		self.momentum_displacement_hc = np.conjugate(self.momentum_displacement).T

	def wigner_distribution(self, fock_basis_density, limit = None, step_size = None):
		update_step_numbers = False
		if limit != None:
			self.limit = limit
			update_step_numbers = True
		if step_size != None:
			self.step_size = step_size
			self._update_displacements()
			update_step_numbers = True
		if update_step_numbers:
			self.steps = int(2*self.limit / self.step_size)
		wigner_distribution = np.zeros((self.steps+1, self.steps+1))
		x_index = int(self.steps/2)
		y_index = int(self.steps/2)
		wigner_distribution[x_index][y_index] = np.einsum('ii,i', 
														  fock_basis_density, 
														  self.alternating_sign)
		for i in range(1, self.steps+1, 2):
			for j in range(i):
				fock_basis_density = (self.position_displacement_hc
									  @ fock_basis_density
									  @ self.position_displacement)
				x_index += 1
				wigner_distribution[x_index][y_index] = np.einsum('ii,i', 
																  fock_basis_density, 
																  self.alternating_sign)
			for j in range(i):
				fock_basis_density = (self.momentum_displacement_hc
									  @ fock_basis_density
									  @ self.momentum_displacement)
				y_index += 1
				wigner_distribution[x_index][y_index] = np.einsum('ii,i', 
																  fock_basis_density, 
																  self.alternating_sign)
			for j in range(i+1):
				fock_basis_density = (self.position_displacement
									  @ fock_basis_density
									  @ self.position_displacement_hc)
				x_index -= 1
				wigner_distribution[x_index][y_index] = np.einsum('ii,i', 
																  fock_basis_density, 
																  self.alternating_sign)
			for j in range(i+1):
				fock_basis_density = (self.momentum_displacement
									  @ fock_basis_density
									  @ self.momentum_displacement_hc)
				y_index -= 1
				wigner_distribution[x_index][y_index] = np.einsum('ii,i', 
																  fock_basis_density, 
																  self.alternating_sign)
		for i in range(self.steps):
			fock_basis_density = (self.position_displacement_hc
								  @ fock_basis_density
								  @ self.position_displacement)
			x_index += 1
			wigner_distribution[x_index][y_index] = np.einsum('ii,i', 
															  fock_basis_density, 
															  self.alternating_sign)
		return wigner_distribution * 2/np.pi