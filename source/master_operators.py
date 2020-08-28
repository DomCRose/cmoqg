import sys
import os
import numpy as np
import scipy.linalg

class lindbladian(object):

	"""Stores, applies and constructs Lindblad master operators.

	
	"""

	def __init__(self, parameters):
		self.hamiltonian = parameters['hamiltonian']
		self.jump_operators = parameters['jump_operators']
		self.hilbert_space_dimension = len(self.hamiltonian[0])
		self._generate_matrix_representation()

	def _hamiltonian_matrix(self):
		"""Returns a dimensionless hamiltonian matrix.

		Each component corresponds to a matrix in the choi 
		representation of one of the dimensionless Hamiltonians 
		defining the model. These can then be combined linearly with 
		the desired dimensionful parametrization to get the matrix 
		representation of the models full Lindblad operator.
		"""
		hamiltonian_term = -1j * np.kron(self.hamiltonian,
										 np.eye(self.hilbert_space_dimension))
		hamiltonian_term += 1j * np.kron(np.eye(self.hilbert_space_dimension),
										 self.hamiltonian.T)
		return hamiltonian_term

	def _jump_term(self, index):
		"""Provides the jump term matrix for the indexed operator."""
		return np.kron(self.jump_operators[index], self.jump_operators[index].conjugate())

	def _jump_matrix(self):
		"""Returns a dimensionless jump group matrix.

		Each term corresponds to a matrix in the choi 
		representation of one of the dimensionless jump operator groups
		defining the model. These can then be combined linearly with 
		the desired dimensionful parametrization to get the matrix 
		representation of the models full Lindblad operator.
		"""
		conjugated_jump_operators = np.conjugate(self.jump_operators)
		jump_operator_number = len(conjugated_jump_operators)
		jump_term = self._jump_term(0)
		for i in range(1, jump_operator_number):
			jump_term += self._jump_term(i)
		trace_preservation_term = np.tensordot(conjugated_jump_operators,
											   self.jump_operators,
											   axes = ([0, 1], [0, 1]))
		jump_term -= 0.5*np.kron(trace_preservation_term,
								 np.eye(self.hilbert_space_dimension))
		jump_term -= 0.5*np.kron(np.eye(self.hilbert_space_dimension),
								 trace_preservation_term.T)
		return jump_term

	def action(self, matrix):
		"""Applies the usual action of the Lindblad equation."""
		return matrix

	def adjoint_action(self, matrix):
		"""Applies the adjoint action of the Lindblad equation."""
		return matrix

	def _generate_matrix_representation(self):
		"""Constructs the matrix representation of the Lindblad operator.

		When run for the first time, dimensionless components will be
		saved for future updates if the program has been instructed to.
		"""
		self.matrix_representation = np.zeros((self.hilbert_space_dimension**2,
											   self.hilbert_space_dimension**2), 
											  dtype = complex)
		self.matrix_representation += self._hamiltonian_matrix()
		self.matrix_representation += self._jump_matrix()

	def spectrum(self, return_number = None, extra_eigenvalues = 0, rounding = 10):
		"""Diagonalized the Lindblad matrix and reshapes the eigenvectors."""
		if return_number == None:
			return_number = self.hilbert_space_dimension**2 + 1
		eigenvalues, left_eigenvectors, right_eigenvectors = scipy.linalg.eig(
			self.matrix_representation, left = True)
		sorting_index = eigenvalues.argsort()[::-1]
		eigenvalues = np.around(eigenvalues[sorting_index], rounding)
		left_eigenmatrices = np.reshape(
			left_eigenvectors[:, sorting_index].T, 
			(self.hilbert_space_dimension**2, 
			 self.hilbert_space_dimension, 
			 self.hilbert_space_dimension))
		right_eigenmatrices = np.reshape(
			right_eigenvectors[:, sorting_index].T, 
			(self.hilbert_space_dimension**2, 
			 self.hilbert_space_dimension, 
			 self.hilbert_space_dimension))
		right_eigenmatrices[0] = right_eigenmatrices[0] / np.trace(right_eigenmatrices[0])
		left_eigenmatrices = np.einsum(
			"ijk,i->ijk",
			left_eigenmatrices, 
			1/np.einsum("ikj,ikj->i", 
						np.conjugate(left_eigenmatrices), 
						right_eigenmatrices))
		left_eigenmatrices = np.around(left_eigenmatrices, rounding)
		right_eigenmatrices = np.around(right_eigenmatrices, rounding)
		return (eigenvalues[0 : return_number + extra_eigenvalues], 
				left_eigenmatrices[0 : return_number], 
				right_eigenmatrices[0 : return_number])

class activity_biased_lindbladian(lindbladian):
	"""Constructs matrix representations of activity biased Lindbladians."""

	def __init__(self, parameters):
		super().__init__(parameters)
		self.biases = parameters['biases'] * np.ones(len(self.jump_operators))

	def _jump_term(self, index):
		"""Provides the biased jump term matrix for the indexed operator."""
		return self.biases[index] * super()._jump_term(index)

class weakly_symmetric_lindbladian(object):

	"""Stores, applies and constructs weakly symmetric Lindbladians.

	Only set up for systems with a single weak symmetry. The jump 
	operators are required to be in a form respecting the weak 
	symmetry, see chapter 3 of the thesis at
	http://eprints.nottingham.ac.uk/56892/ or the paper [REF], input
	as sets of blocks. The Hamiltonian must have also been block 
	diagonalized.

	Parameters
	----------
	hamiltonians : list of 2d arrays
		The Hamiltonian components, each consisting of a list of their
		symmetry eigenspace blocks.
	jump_operators : list of lists of 2d arrays
		The jump operators, grouped in lists which share parameters. 
		Each jump operator is represented by a list of their symmetry 
		eigenspace blocks.
	jump_operator_mappings : list of lists of integers
		Indexes which prescribe which eigenspace each block of each
		jump operator maps too. The eigenspace they map from is
		prescribed by their order. Must have same shape as
		jump_operators bar the last two indices.
	eigenspace_number : integer
		The number of unique eigenvalues possessed by the symmetry.
	eigenspace_pairs : list of list of integers
		For each block of the Lindbladian, the index of the eigenspace
		to which each eigenspace corresponding to the list index is 
		paired.
	eigenspace_dimensions : list of integers
		The Hilbert space dimension of each symmetry eigenspace.
	current_parameters : list of list of floats, optional
		The coefficients used with each component when combined to
		construct the full Lindbladian.
	save_component : list of list of lists of booleans
		Determines whether each component is saved for fast iteraction
		over parameters, with a choice of whether to save only 
		particular eigenspaces of interest.
	"""

	def __init__(self, parameters):
		self.hamiltonian = parameters['hamiltonians']
		self.jump_operators = parameters['jump_operators']
		self.eigenspace_pairs = parameters['eigenspace_pairs']
		self.eigenspace_dimensions = parameters['eigenspace_dimensions']
		self.eigenspace_number = len(self.eigenspace_dimensions)
		self._generate_matrix_representation()

	def _hamiltonian_term(self, block_index):
		"""Returns a dimensionless hamiltonian matrix.

		Each component corresponds to a matrix in the choi 
		representation of one of the dimensionless Hamiltonians 
		defining the model. These can then be combined linearly with 
		the desired dimensionful parametrization to get the matrix 
		representation of the models full Lindblad operator.
		"""
		pair_index = self.eigenspace_pairs[block_index][0]
		hamiltonian_term = -1j * np.kron(
			self.hamiltonian[0], np.eye(self.eigenspace_dimensions[pair_index]))
		hamiltonian_term += 1j * np.kron(
			np.eye(self.eigenspace_dimensions[pair_index]), self.hamiltonian[0].T)
		print(hamiltonian_term.shape)
		for symmetry_index in range(1, self.eigenspace_number):
			pair_index = self.eigenspace_pairs[block_index][symmetry_index]
			print(self.hamiltonian[symmetry_index].shape)
			print(np.eye(self.eigenspace_dimensions[pair_index]).shape)
			hamiltonian_term += -1j * np.kron(
				self.hamiltonian[symmetry_index], 
				np.eye(self.eigenspace_dimensions[pair_index]))
			hamiltonian_term += 1j * np.kron(
				np.eye(self.eigenspace_dimensions[pair_index]),
				self.hamiltonian[symmetry_index].T)
		return hamiltonian_term

	def _jump_term(self, block_index):
		"""Returns a dimensionless jump group matrix.

		Each component corresponds to a matrix in the choi 
		representation of one of the dimensionless jump operator groups
		defining the model. These can then be combined linearly with 
		the desired dimensionful parametrization to get the matrix 
		representation of the models full Lindblad operator.
		"""
		conjugated_jump_operators = np.conjugate(self.jump_operators)
		jump_operator_number = len(conjugated_jump_operators)
		pair_index = self.eigenspace_pairs[block_index][0]
		block_dimension = (self.eigenspace_dimensions[0] 
						   * self.eigenspace_dimensions[pair_index])
		jump_component = np.zeros((block_dimension, block_dimension),
								  dtype = complex)
		trace_preservation_term = np.einsum('ijkl,ijkm->jlm',
											conjugated_jump_operators,
											self.jump_operators)
		for symmetry_index in range(self.eigenspace_number):
			pair_index = self.eigenspace_pairs[block_index][symmetry_index]
			for i in range(jump_operator_number):
				jump_component += np.kron(
					self.jump_operators[i][symmetry_index],
					conjugated_jump_operators[i][pair_index])
			jump_component -= 0.5*np.kron(
				trace_preservation_term[symmetry_index],
				np.eye(self.eigenspace_dimensions[pair_index]))
			jump_component -= 0.5*np.kron(
				np.eye(self.eigenspace_dimensions[symmetry_index]),
				trace_preservation_term[pair_index].T)
		return jump_component

	def action(self, matrix):
		"""Applies the usual action of the Lindblad equation."""
		return matrix

	def adjoint_action(self, matrix):
		"""Applies the adjoint action of the Lindblad equation."""
		return matrix

	def _generate_matrix_representation(self):
		"""Constructs the matrix representation of the Lindblad operator.

		When run for the first time, dimensionless components will be
		saved for future updates if the program has been instructed to.
		"""
		self.matrix_representation = []
		for block_index in range(self.eigenspace_number):
			pair_index = self.eigenspace_pairs[block_index][0]
			print(pair_index)
			block_dimension = (self.eigenspace_dimensions[0] 
							   * self.eigenspace_dimensions[pair_index])
			self.matrix_representation.append(np.zeros((block_dimension, block_dimension),
											  dtype = complex))
			self.matrix_representation[-1] += self._hamiltonian_term(block_index)
			self.matrix_representation[-1] += self._jump_term(block_index)