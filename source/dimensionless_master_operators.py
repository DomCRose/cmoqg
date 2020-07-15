import sys
import os
import numpy as np


class lindbladian(object):

	"""Stores, applies and constructs Lindblad master operators.

	
	"""

	def __init__(self, parameters):
		self.hamiltonians = parameters['hamiltonians']
		self.hamiltonian_number = len(self.hamiltonians)
		self.jump_operators = parameters['jump_operators']
		self.jump_group_number = len(self.jump_operators)
		if self.hamiltonian_number != 0:
			self.hilbert_space_dimension = len(self.hamiltonians[0])
		else:
			self.hilbert_space_dimension = len(self.jump_operators[0][0])
		if 'current_parameters' in parameters:
			self.current_parameters = parameters['current_parameters']
			if (len(self.current_parameters[0]) != self.hamiltonian_number
				or len(self.current_parameters[1]) != self.jump_group_number):
				raise ValueError(
					"current_parameters must be a list of two lists of "
					"length equal to the length of the variables hamiltonians"
					"and jump_operators.")
		else:
			self.current_parameters = [[1 for i in range(self.hamiltonian_number)],
									   [1 for i in range(self.jump_group_number)]]
		if 'save_component' in parameters:
			self.save_component = parameters['save_component']
			if (len(self.save_component[0]) != self.hamiltonian_number
				or len(self.save_component[1]) != self.jump_group_number):
				raise ValueError(
					"save_component must be a list of two lists of "
					"length equal to the length of the variables hamiltonians"
					"and jump_operators.")
		else:
			self.save_component = [[False for i in range(self.hamiltonian_number)],
								   [False for i in range(self.jump_group_number)]]
		self.hamiltonians_saved = [False for i in range(self.hamiltonian_number)]
		self.hamiltonian_components = [[] for i in range(self.hamiltonian_number)]
		self.jump_groups_saved = [False for i in range(self.jump_group_number)]
		self.jump_group_components = [[] for i in range(self.jump_group_number)]

	def _hamiltonian_component(self, component_index):
		"""Returns a dimensionless hamiltonian matrix.

		Each component corresponds to a matrix in the choi 
		representation of one of the dimensionless Hamiltonians 
		defining the model. These can then be combined linearly with 
		the desired dimensionful parametrization to get the matrix 
		representation of the models full Lindblad operator.
		"""
		if self.hamiltonians_saved[component_index]:
			return self.hamiltonian_components[component_index]
		else:
			hamiltonian_component = -1j * np.kron(self.hamiltonians[component_index],
												  np.eye(self.hilbert_space_dimension))
			hamiltonian_component += 1j * np.kron(np.eye(self.hilbert_space_dimension),
												  self.hamiltonians[component_index].T)
			if self.save_component[0][component_index]:
				self.hamiltonian_components[component_index] = hamiltonian_component
				self.hamiltonians_saved[component_index] = True
			return hamiltonian_component

	def _jump_component(self, component_index):
		"""Returns a dimensionless jump group matrix.

		Each component corresponds to a matrix in the choi 
		representation of one of the dimensionless jump operator groups
		defining the model. These can then be combined linearly with 
		the desired dimensionful parametrization to get the matrix 
		representation of the models full Lindblad operator.
		"""
		if self.jump_groups_saved[component_index]:
			return self.jump_group_components[component_index]
		else:
			conjugated_jump_operators = np.conjugate(self.jump_operators[component_index])
			jump_operator_number = len(conjugated_jump_operators)
			jump_group_component = np.kron(self.jump_operators[component_index][0],
										   conjugated_jump_operators[0])
			for i in range(1, jump_operator_number):
				jump_group_component += np.kron(self.jump_operators[component_index][i],
												conjugated_jump_operators[i])
			trace_preservation_term = np.tensordot(conjugated_jump_operators,
												   self.jump_operators[component_index],
												   axes = ([0, 1], [0, 1]))
			jump_group_component -= 0.5*np.kron(trace_preservation_term,
												np.eye(self.hilbert_space_dimension))
			jump_group_component -= 0.5*np.kron(np.eye(self.hilbert_space_dimension),
												trace_preservation_term.T)
			if self.save_component[1][component_index]:
				self.jump_group_components[component_index] = jump_group_component
				self.jump_groups_saved[component_index] = True
			return jump_group_component

	def action(self, matrix, parameters = None):
		"""Applies the usual action of the Lindblad equation."""
		if parameters != None:
			self.current_parameters = parameters
		return matrix

	def adjoint_action(self, matrix, parameters = None):
		"""Applies the adjoint action of the Lindblad equation."""
		if parameters != None:
			self.current_parameters = parameters
		return matrix

	def generate_matrix_representation(self, parameters = None):
		"""Constructs the matrix representation of the Lindblad operator.

		When run for the first time, dimensionless components will be
		saved for future updates if the program has been instructed to.
		"""
		if parameters != None:
			self.current_parameters = parameters
		self.matrix_representation = np.zeros((self.hilbert_space_dimension**2,
											   self.hilbert_space_dimension**2), 
											  dtype = complex)
		for i in range(self.hamiltonian_number):
			self.matrix_representation += (self.current_parameters[0][i]
										   * self._hamiltonian_component(i))
		for i in range(self.jump_group_number):
			self.matrix_representation += (self.current_parameters[1][i]
										   * self._jump_component(i))

	def update_matrix_representation(self, new_parameters, rounding = 10):
		"""Updates the matrix representation of the Lindblad operator.

		Only updates with components for which the associated parameter
		has been changed. If the corresponding dimensionless component
		has been saved this will be used instead of reconstruction.
		"""
		parameter_change = [np.around(np.array(new_parameters)[0]
									  - np.array(self.current_parameters[0]),
									  rounding),
							np.around(np.array(new_parameters)[1]
									  - np.array(self.current_parameters[1]),
									  rounding)]
		self.current_parameters = new_parameters
		for i in range(self.hamiltonian_number):
			if parameter_change[0][i]:
				self.matrix_representation += (parameter_change[0][i]
											   * self._hamiltonian_component(i))
		for i in range(self.jump_group_number):
			if parameter_change[1][i]:
				self.matrix_representation += (parameter_change[1][i]
											   * self._jump_component(i))

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
		self.hamiltonians = parameters['hamiltonians']
		self.hamiltonian_number = len(self.hamiltonians)
		self.jump_operators = parameters['jump_operators']
		self.jump_operator_mappings = parameters['jump_operator_mappings']
		self.jump_group_number = len(self.jump_operators)
		self.eigenspace_number = parameters['eigenspace_number']
		self.eigenspace_pairs = parameters['eigenspace_pairs']
		self.eigenspace_dimensions = parameters['eigenspace_dimensions']
		if 'current_parameters' in parameters:
			self.current_parameters = parameters['current_parameters']
			if (len(self.current_parameters[0]) != self.hamiltonian_number
				or len(self.current_parameters[1]) != self.jump_group_number):
				raise ValueError(
					"current_parameters must be a list of two lists of "
					"length equal to the length of the variables hamiltonians"
					"and jump_operators.")
		else:
			self.current_parameters = [[1 for i in range(self.hamiltonian_number)],
									   [1 for i in range(self.jump_group_number)]]
		if 'save_component' in parameters:
			self.save_component = parameters['save_component']
			if (len(self.save_component[0]) != self.hamiltonian_number
				or len(self.save_component[1]) != self.jump_group_number):
				raise ValueError(
					"save_component must be a list of two lists of "
					"length equal to the length of the variables hamiltonians"
					"and jump_operators.")
		else:
			self.save_component = [[[False for i in range(self.hamiltonian_number)],
									[False for i in range(self.jump_group_number)]]
								   for j in range(self.eigenspace_number)]
		self.hamiltonians_saved = [[False for i in range(self.hamiltonian_number)]
								   for j in range(self.eigenspace_number)]
		self.hamiltonian_components = [[[] for i in range(self.hamiltonian_number)]
									   for j in range(self.eigenspace_number)]
		self.jump_groups_saved = [[False for i in range(self.jump_group_number)]
								  for j in range(self.eigenspace_number)]
		self.jump_group_components = [[[] for i in range(self.jump_group_number)]
									  for j in range(self.eigenspace_number)]

	def _hamiltonian_component(self, component_index, block_index):
		"""Returns a dimensionless hamiltonian matrix.

		Each component corresponds to a matrix in the choi 
		representation of one of the dimensionless Hamiltonians 
		defining the model. These can then be combined linearly with 
		the desired dimensionful parametrization to get the matrix 
		representation of the models full Lindblad operator.
		"""
		if self.hamiltonians_saved[block_index][component_index]:
			return self.hamiltonian_components[block_index][component_index]
		else:
			hamiltonian_component = -1j * np.kron(
				self.hamiltonians[component_index][0],
				np.eye(self.eigenspace_dimensions[
					self.eigenspace_pairs[block_index][0]]))
			hamiltonian_component += 1j * np.kron(
				np.eye(self.eigenspace_dimensions[
					self.eigenspace_pairs[block_index][0]]),
				self.hamiltonians[component_index][0].T)
			for symmetry_index in range(1, self.eigenspace_number):
				hamiltonian_component += -1j * np.kron(
					self.hamiltonians[component_index][symmetry_index],
					np.eye(self.eigenspace_dimensions[
						self.eigenspace_pairs[block_index][symmetry_index]]))
				hamiltonian_component += 1j * np.kron(
					np.eye(self.eigenspace_dimensions[
						self.eigenspace_pairs[block_index][symmetry_index]]),
					self.hamiltonians[component_index][symmetry_index].T)
			if self.save_component[block_index][0][component_index]:
				self.hamiltonian_components[block_index][component_index] = (
					hamiltonian_component)
				self.hamiltonians_saved[block_index][component_index] = True
			return hamiltonian_component

	def _jump_component(self, component_index, block_index):
		"""Returns a dimensionless jump group matrix.

		Each component corresponds to a matrix in the choi 
		representation of one of the dimensionless jump operator groups
		defining the model. These can then be combined linearly with 
		the desired dimensionful parametrization to get the matrix 
		representation of the models full Lindblad operator.
		"""
		if self.jump_groups_saved[block_index][component_index]:
			return self.jump_group_components[block_index][component_index]
		else:
			conjugated_jump_operators = np.conjugate(self.jump_operators[component_index])
			jump_operator_number = len(conjugated_jump_operators)
			pair_index = self.eigenspace_pairs[block_index][0]
			block_dimension = (self.eigenspace_dimensions[0] 
							   * self.eigenspace_dimensions[pair_index])
			jump_group_component = np.zeros((block_dimension, block_dimension),
											dtype = complex)
			trace_preservation_term = np.einsum('ijkl,ijkm->jlm',
												conjugated_jump_operators,
												self.jump_operators[component_index])
			for symmetry_index in range(self.eigenspace_number):
				pair_index = self.eigenspace_pairs[block_index][symmetry_index]
				for i in range(jump_operator_number):
					jump_group_component += np.kron(
						self.jump_operators[component_index][i][symmetry_index],
						conjugated_jump_operators[i][pair_index])
				jump_group_component -= 0.5*np.kron(
					trace_preservation_term[symmetry_index],
					np.eye(self.eigenspace_dimensions[pair_index]))
				jump_group_component -= 0.5*np.kron(
					np.eye(self.eigenspace_dimensions[symmetry_index]),
					trace_preservation_term[pair_index].T)
			if self.save_component[block_index][1][component_index]:
				self.jump_group_components[block_index][component_index] = (
					jump_group_component)
				self.jump_groups_saved[block_index][component_index] = True
			return jump_group_component

	def action(self, matrix, parameters = None):
		"""Applies the usual action of the Lindblad equation."""
		if parameters != None:
			self.current_parameters = parameters
		return matrix

	def adjoint_action(self, matrix, parameters = None):
		"""Applies the adjoint action of the Lindblad equation."""
		if parameters != None:
			self.current_parameters = parameters
		return matrix

	def generate_matrix_representation(self, parameters = None):
		"""Constructs the matrix representation of the Lindblad operator.

		When run for the first time, dimensionless components will be
		saved for future updates if the program has been instructed to.
		"""
		if parameters != None:
			self.current_parameters = parameters
			self.matrix_representation = []
		for block_index in range(self.eigenspace_number):
			pair_index = self.eigenspace_pairs[block_index][0]
			block_dimension = (self.eigenspace_dimensions[0] 
							   * self.eigenspace_dimensions[pair_index])
			self.matrix_representation.append(np.zeros((block_dimension, block_dimension),
											  dtype = complex))
			for i in range(self.hamiltonian_number):
				self.matrix_representation[-1] += (self.current_parameters[0][i]
											* self._hamiltonian_component(i, block_index))
			for i in range(self.jump_group_number):
				self.matrix_representation[-1] += (self.current_parameters[1][i]
											* self._jump_component(i, block_index))

	def update_matrix_representation(self, new_parameters, rounding = 10):
		"""Updates the matrix representation of the Lindblad operator.

		Only updates with components for which the associated parameter
		has been changed. If the corresponding dimensionless component
		has been saved this will be used instead of reconstruction.
		"""
		parameter_change = [np.around(np.array(new_parameters)[0]
									  - np.array(self.current_parameters[0]),
									  rounding),
							np.around(np.array(new_parameters)[1]
									  - np.array(self.current_parameters[1]),
									  rounding)]
		self.current_parameters = new_parameters
		for block_index in range(self.eigenspace_number):
			for i in range(self.hamiltonian_number):
				if parameter_change[0][i]:
					self.matrix_representation += (
						parameter_change[0][i]
						* self._hamiltonian_component(i, block_index))
			for i in range(self.jump_group_number):
				if parameter_change[1][i]:
					self.matrix_representation += (
						parameter_change[1][i]
						* self._jump_component(i, block_index))