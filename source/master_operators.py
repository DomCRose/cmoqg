import sys
import os
import numpy as np
import scipy.linalg

class lindbladian(object):

	"""Stores and constructs Lindblad master operators."""

	def __init__(self, parameters):
		self.hamiltonian = parameters['hamiltonian']
		self.jump_operators = parameters['jump_operators']
		self.hilbert_space_dimension = len(self.hamiltonian[0])
		self._generate_matrix_representation()

	def _hamiltonian_matrix(self):
		"""Returns the Hamiltonian commutator term of the Lindbladian."""
		hamiltonian_term = -1j * np.kron(self.hamiltonian,
										 np.eye(self.hilbert_space_dimension))
		hamiltonian_term += 1j * np.kron(np.eye(self.hilbert_space_dimension),
										 self.hamiltonian.T)
		return hamiltonian_term

	def _jump_term(self, index):
		"""Provides the jump term matrix for the indexed operator."""
		return np.kron(self.jump_operators[index], self.jump_operators[index].conjugate())

	def _jump_matrix(self):
		"""Returns the jump term in the Lindbladian."""
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

	def _generate_matrix_representation(self):
		"""Constructs the matrix representation of the Lindblad operator."""
		self.matrix_representation = np.zeros((self.hilbert_space_dimension**2,
											   self.hilbert_space_dimension**2), 
											  dtype = complex)
		self.matrix_representation += self._hamiltonian_matrix()
		self.matrix_representation += self._jump_matrix()

	def spectrum(self, return_number = None, extra_eigenvalues = 0, rounding = 10):
		"""Diagonalized the Lindblad matrix and reshapes the eigenvectors."""
		if return_number == None:
			return_number = self.hilbert_space_dimension**2
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
		return np.exp(-self.biases[index]) * super()._jump_term(index)

class weakly_symmetric_lindbladian(object):

	"""Stores and constructs weakly symmetric Lindbladians.

	Only set up for systems with a single weak symmetry. The jump 
	operators are required to be in a form respecting the weak 
	symmetry, see chapter 3 of the thesis at
	http://eprints.nottingham.ac.uk/56892/ or the paper 
	[Phys. Rev. A 103, 042204 (2021)], input as sets of blocks. The 
	Hamiltonian must have also been block diagonalized. These blocks 
	are assumed to be ordered according to the eigenspace of the
	symmetry it acts on, i.e. if a block acts on the first
	eigenspace of the symmetry, it occurs first in the list of blocks.
	"""

	def __init__(self, parameters):
		self.hamiltonian = parameters['hamiltonians']
		self.jump_operators = parameters['jump_operators']
		self.jump_operator_eigenvalues = parameters['jump_operator_eigenvalues']
		self.eigenspace_pairs = parameters['eigenspace_pairs']
		self.eigenspace_dimensions = parameters['eigenspace_dimensions']
		self.symmetry_rep = parameters['symmetry_rep']
		self.eigenspace_number = len(self.eigenspace_dimensions)
		self.hilbert_space_dimension = sum(self.eigenspace_dimensions)
		self._generate_matrix_representation()

	def _hamiltonian_term(self, block_index):
		"""Returns a block of the Hamiltonian commutator term in the Lindbladian."""
		hamiltonian_term_blocks = []
		for symmetry_index in range(self.eigenspace_number):
			pair_index = self.eigenspace_pairs[block_index][symmetry_index]
			hamiltonian_term_blocks.append(-1j * np.kron(
				self.hamiltonian[symmetry_index], 
				np.eye(self.eigenspace_dimensions[pair_index])))
			hamiltonian_term_blocks[-1] += 1j * np.kron(
				np.eye(self.eigenspace_dimensions[symmetry_index]),
				self.hamiltonian[pair_index].T)
		hamiltonian_term = scipy.linalg.block_diag(*hamiltonian_term_blocks)
		return hamiltonian_term

	def _normalization_term(self, block_index):
		"""Returns a block of the anti-commutator term in the Lindbladian."""
		conjugated_jumps = np.conjugate(self.jump_operators)
		jump_products = [np.zeros((self.eigenspace_dimensions[i],
								   self.eigenspace_dimensions[i]), dtype = complex) 
						 for i in range(self.eigenspace_number)]
		for jump_index in range(len(self.jump_operators)):
			for symmetry_index in range(self.eigenspace_number):
				jump_products[symmetry_index] += (
					conjugated_jumps[jump_index][symmetry_index].T 
					@ self.jump_operators[jump_index][symmetry_index])
		normalization_term_blocks = []
		for symmetry_index in range(self.eigenspace_number):
			pair_index = self.eigenspace_pairs[block_index][symmetry_index]
			normalization_term_blocks.append(-0.5 * np.kron(
				jump_products[symmetry_index], 
				np.eye(self.eigenspace_dimensions[pair_index])))
			normalization_term_blocks[-1] -= 0.5 * np.kron(
				np.eye(self.eigenspace_dimensions[symmetry_index]),
				jump_products[pair_index].T)
		normalization_term = scipy.linalg.block_diag(*normalization_term_blocks)
		return normalization_term


	def _jump_term(self, block_index):
		"""Returns a block of the jump term in the Lindbladian."""
		adjoint_eigenspace_dimensions = []
		for eigenspace_index in range(self.eigenspace_number):
			pair_index = self.eigenspace_pairs[block_index][eigenspace_index]
			adjoint_eigenspace_dimensions.append(
				self.eigenspace_dimensions[eigenspace_index] 
				* self.eigenspace_dimensions[pair_index])
		jump_term = [[np.zeros((adjoint_eigenspace_dimensions[i],
								adjoint_eigenspace_dimensions[j]), dtype = complex) 
					  for j in range(self.eigenspace_number)]
					 for i in range(self.eigenspace_number)]
		conjugated_jump_operators = np.conjugate(self.jump_operators)
		for jump_index in range(len(self.jump_operators)):
			for symmetry_index in range(self.eigenspace_number):
				pair_index = self.eigenspace_pairs[block_index][symmetry_index]
				jump_term[(symmetry_index 
						   - self.jump_operator_eigenvalues[jump_index]) 
						  % self.eigenspace_number][symmetry_index] += np.kron(
					self.jump_operators[jump_index][symmetry_index],
					conjugated_jump_operators[jump_index][pair_index])
		return np.block(jump_term)

	def _generate_matrix_representation(self):
		"""Constructs the matrix representation of the Lindblad operator.

		When run for the first time, dimensionless components will be
		saved for future updates if the program has been instructed to.
		"""
		print(self.eigenspace_pairs)
		self.matrix_representation = []
		for block_index in range(self.eigenspace_number):
			block_dimension = 0
			for eigenspace_index in range(self.eigenspace_number):
				pair_index = self.eigenspace_pairs[block_index][eigenspace_index]
				block_dimension += (self.eigenspace_dimensions[eigenspace_index] 
								* self.eigenspace_dimensions[pair_index])
			self.matrix_representation.append(np.zeros((block_dimension, block_dimension),
											  dtype = complex))
			self.matrix_representation[-1] += self._hamiltonian_term(block_index)
			self.matrix_representation[-1] += self._normalization_term(block_index)
			self.matrix_representation[-1] += self._jump_term(block_index)

	def _matrix_embedding(self, vectors, matrix_eigenspace):
		"""Embeds a symmetric eigenvector in the full matrix hilbert space."""
		eigenspace_pairs = self.eigenspace_pairs[matrix_eigenspace]
		matrices = []
		for vector in vectors:
			matrix = [[np.zeros((self.eigenspace_dimensions[i], 
								 self.eigenspace_dimensions[j]), dtype = complex)
					   for j in range(self.eigenspace_number)] 
					  for i in range(self.eigenspace_number)]
			vblock_start = 0
			vblock_end = 0
			for left, right in enumerate(eigenspace_pairs):
				vblock_start = vblock_end
				vblock_end += (self.eigenspace_dimensions[left]
							   * self.eigenspace_dimensions[right])
				matrix[left][right] += np.reshape(
					vector[vblock_start:vblock_end],
					(self.eigenspace_dimensions[left], self.eigenspace_dimensions[right]))
			matrices.append(np.block(matrix))
		return matrices

	def spectrum(self, return_number = None, extra_eigenvalues = 0, rounding = 10):
		"""Diagonalized the Lindblad matrix and reshapes the eigenvectors."""
		if return_number == None:
			return_number = self.hilbert_space_dimension**2
		eigenvalues = []
		left_eigenmatrices = []
		right_eigenmatrices = []
		symmetry_indices = []
		for sym_ind in range(self.eigenspace_number):
			evals, left_eigenvectors, right_eigenvectors = scipy.linalg.eig(
				self.matrix_representation[sym_ind], left = True)
			sorting_index = evals.argsort()[::-1]
			evals = np.around(evals[sorting_index], rounding)[0:return_number]
			left_evecs = left_eigenvectors[:, sorting_index].T[0:return_number]
			right_evecs = right_eigenvectors[:, sorting_index].T[0:return_number]
			left_emats = self._matrix_embedding(left_evecs, sym_ind)
			right_emats = self._matrix_embedding(right_evecs, sym_ind)
			if sym_ind == 0:
				print(np.trace(right_emats[0]))
				right_emats[0] = right_emats[0] / np.trace(right_emats[0])
			norm_factor = 1/np.einsum("ikj,ikj->i", np.conjugate(left_emats), right_emats)
			left_emats = np.einsum("ijk,i->ijk", left_emats, np.conjugate(norm_factor))
			left_emats = np.around(left_emats, rounding)
			right_emats = np.around(right_emats, rounding)
			eigenvalues.extend(evals)
			left_eigenmatrices.extend(left_emats)
			right_eigenmatrices.extend(right_emats)
			symmetry_indices.extend([sym_ind for i in range(return_number)])
		sorting_index = np.array(eigenvalues).argsort()[::-1]
		eigenvalues = np.array(eigenvalues)[sorting_index]
		left_eigenmatrices = np.array(left_eigenmatrices)[sorting_index][0:return_number]
		right_eigenmatrices = np.array(right_eigenmatrices)[sorting_index][
																		0:return_number]
		symmetry_indices = np.array(symmetry_indices)[sorting_index][0:return_number]
		return (eigenvalues[0 : return_number + extra_eigenvalues], 
				left_eigenmatrices, right_eigenmatrices, symmetry_indices)