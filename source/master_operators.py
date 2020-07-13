import sys
import os
import scipy
import numpy as np


class lindbladian(object):

	"""Stores, applies and constructs Lindblad master operators.

	
	"""

	def __init__(self, initialization_variables):
		self.hamiltonians = initialization_variables['hamiltonians']
		self.hamiltonian_number = len(self.hamiltonians)
		self.jump_operators = initialization_variables['jump_operators']
		self.jump_group_number = len(self.jump_operators)
		if self.hamiltonian_number != 0:
			self.hilbert_space_dimension = len(self.hamiltonians[0])
		else:
			self.hilbert_space_dimension = len(self.jump_operators[0][0])
		if 'current_parameters' in initialization_variables:
			self.current_parameters = initialization_variables['current_parameters']
			if (len(self.current_parameters[0]) != self.hamiltonian_number
				or len(self.current_parameters[1]) != self.jump_group_number):
				raise ValueError(
					"current_parameters must be a list of two lists of "
					"length equal to the length of the variables hamiltonians"
					"and jump_operators.")
		else:
			self.current_parameters = [[1 for i in range(self.hamiltonian_number)],
									   [1 for i in range(self.jump_group_number)]]
		if 'save_component' in initialization_variables:
			self.save_component = initialization_variables['save_component']
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