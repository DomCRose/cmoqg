import sys
import os
import scipy
import numpy as np
import master_operators


class activity_biased_lindbladian(master_operators.lindbladian):

	"""Stores, applies and constructs Lindblad master operators.

	
	"""

	def __init__(self, lindbladian_variables, bias_variables):
		super().__init__(lindbladian_variables)
		if 'conjugate_field' in bias_variables:
			self.conjugate_field = bias_variables['conjugate_field']
		else:
			self.conjugate_field = 0
		if 'save_component' in bias_variables:
			self.save_bias_component = bias_variables['save_component']
		else:
			self.save_bias_component = False
		self.tilting_component_saved = False
		self.tilting_component = []

	def _tilting_subcomponent(self, component_index):
		conjugated_jump_operators = np.conjugate(self.jump_operators[component_index])
		jump_operator_number = len(conjugated_jump_operators)
		tilting_subcomponent = np.kron(self.jump_operators[component_index][0],
									   conjugated_jump_operators[0])
		for i in range(1,jump_operator_number):
			tilting_subcomponent += np.kron(self.jump_operators[component_index][i],
											conjugated_jump_operators[i])
		return tilting_subcomponent

	def _tilting_component(self):
		if self.tilting_component_saved:
			return self.tilting_component
		else:
			tilting_component = (self.current_parameters[1][0] 
								 * self._tilting_subcomponent(0))
			for i in range(1, self.jump_group_number):
				tilting_component += (self.current_parameters[1][i] 
									  * self._tilting_subcomponent(i))
			if self.save_bias_component:
				self.tilting_component = tilting_component
				self.tilting_component_saved = True
			return tilting_component

	def action(self, matrix, parameters = None, conjugate_field = None):
		"""Applies the usual action of the Lindblad equation."""
		if parameters != None:
			self.current_parameters = parameters
		if conjugate_field != None:
			self.conjugate_field = conjugate_field
		acted_matrix = super().action(matrix)
		return acted_matrix

	def adjoint_action(self, matrix, parameters = None, conjugate_field = None):
		"""Applies the adjoint action of the Lindblad equation."""
		if parameters != None:
			self.current_parameters = parameters
		if conjugate_field != None:
			self.conjugate_field = conjugate_field
		acted_matrix = super().adjoint_action(matrix)
		return acted_matrix

	def generate_matrix_representation(self, parameters = None, conjugate_field = None):
		"""Constructs the matrix representation of the tilted operator.

		Either constructs the operator from scratch, discarding each
		dimensionless component after use, or uses stored dimensionless
		components if available.
		"""
		if conjugate_field != None:
			self.conjugate_field = conjugate_field
		super().generate_matrix_representation(parameters = parameters)
		self.matrix_representation += ((np.exp(-self.conjugate_field)-1)
									   * self._tilting_component())

	def update_matrix_representation(self, new_parameters, 
									  new_conjugate_field, rounding = 10):
		parameter_change = [np.around(np.array(new_parameters)[0]
									  - np.array(self.current_parameters[0]),
									  rounding),
							np.around(np.array(new_parameters)[1]
									  - np.array(self.current_parameters[1]),
									  rounding)]
		conjugate_field_change = round(self.conjugate_field - new_conjugate_field, 
									   rounding)
		if parameter_change[0].any() or parameter_change[1].any():
			super().update_matrix_representation(new_parameters)
		for i in range(self.jump_group_number):
			if parameter_change[1][i]:
				tilting_subcomponent = self._tilting_subcomponent(i)
				self.matrix_representation += ((np.exp(-self.conjugate_field)-1)
											   * parameter_change[1][i]
											   * tilting_subcomponent)
				if self.tilting_component_saved:
					self.tilting_component += (parameter_change[1][i] 
											   * tilting_subcomponent)
		if conjugate_field_change:
			self.matrix_representation += ((np.exp(-new_conjugate_field)
											- np.exp(-self.conjugate_field))
										   * self._tilting_component())
			self.conjugate_field = new_conjugate_field



class quadrature_biased_lindbladian(master_operators.lindbladian):

	"""Stores, applies and constructs Lindblad master operators.

	
	"""

	def __init__(self, lindbladian_variables, bias_variables):
		super().__init__(lindbladian_variables)
		if 'conjugate_field' in bias_variables:
			self.conjugate_field = bias_variables['conjugate_field']
		else:
			self.conjugate_field = 0
		if 'quadrature_angle' in bias_variables:
			self.quadrature_angle = bias_variables['quadrature_angle']
		else:
			self.quadrature_angle = 0
		if 'save_component' in bias_variables:
			self.save_bias_component = bias_variables['save_component']
		else:
			self.save_bias_component = False
		self.tilting_component_saved = False
		self.tilting_component = []

	def _tilting_subcomponent(self, component_index):
		Identity = np.eye(self.hilbert_space_dimension)
		tilting_subcomponent = (
			(np.kron(self.jump_operators[component_index][0], Identity)
			 * np.exp(-1j*self.quadrature_angle))
			+ (np.exp(1j*self.quadrature_angle)
			   * np.kron(Identity, 
			   			 np.conjugate(self.jump_operators[component_index][0]))))
		jump_operator_number = len(self.jump_operators[component_index])
		for i in range(1,jump_operator_number):
			tilting_subcomponent += (
				(np.kron(self.jump_operators[component_index][i], Identity)
				 * np.exp(-1j*self.quadrature_angle))
				+ (np.exp(1j*self.quadrature_angle)
				   * np.kron(Identity, 
				   			 np.conjugate(self.jump_operators[component_index][i]))))
		return tilting_subcomponent

	def _tilting_component(self):
		if self.tilting_component_saved:
			return self.tilting_component
		else:
			tilting_component = (np.sqrt(self.current_parameters[1][0])
								 * self._tilting_subcomponent(0))
			for i in range(1, self.jump_group_number):
				tilting_component += (np.sqrt(self.current_parameters[1][i])
									  * self._tilting_subcomponent(i))
			if self.save_bias_component:
				self.tilting_component = tilting_component
				self.tilting_component_saved = True
			return tilting_component

	def action(self, matrix, parameters = None, conjugate_field = None):
		"""Applies the usual action of the Lindblad equation."""
		if parameters != None:
			self.current_parameters = parameters
		if conjugate_field != None:
			self.conjugate_field = conjugate_field
		acted_matrix = super().action(matrix)
		return acted_matrix

	def adjoint_action(self, matrix, parameters = None, conjugate_field = None):
		"""Applies the adjoint action of the Lindblad equation."""
		if parameters != None:
			self.current_parameters = parameters
		if conjugate_field != None:
			self.conjugate_field = conjugate_field
		acted_matrix = super().adjoint_action(matrix)
		return acted_matrix

	def generate_matrix_representation(self, parameters = None, conjugate_field = None):
		"""Constructs the matrix representation of the tilted operator.

		Either constructs the operator from scratch, discarding each
		dimensionless component after use, or uses stored dimensionless
		components if available.
		"""
		if conjugate_field != None:
			self.conjugate_field = conjugate_field
		super().generate_matrix_representation(parameters = parameters)
		self.matrix_representation += self.conjugate_field*0.5 * self._tilting_component()
		self.matrix_representation += (self.conjugate_field**2/8.0
									   * np.eye(self.hilbert_space_dimension**2))

	def update_matrix_representation(self, new_parameters, 
									  new_conjugate_field, rounding = 10):
		old_parameters = self.current_parameters
		parameter_change = [np.around(np.array(new_parameters)[0]
									  - np.array(self.current_parameters[0]),
									  rounding),
							np.around(np.array(new_parameters)[1]
									  - np.array(self.current_parameters[1]),
									  rounding)]
		conjugate_field_change = round(new_conjugate_field - self.conjugate_field, 
									   rounding)
		if parameter_change[0].any() or parameter_change[1].any():
			super().update_matrix_representation(new_parameters)
		for i in range(self.jump_group_number):
			if parameter_change[1][i]:
				tilting_subcomponent = self._tilting_subcomponent(i)
				self.matrix_representation += (self.conjugate_field
											   * (np.sqrt(self.current_parameters[1][i])
											   	  - np.sqrt(old_parameters[1][i]))
											   * tilting_subcomponent)
				if self.tilting_component_saved:
					self.tilting_component += ((np.sqrt(self.current_parameters[1][i])
											   	- np.sqrt(old_parameters[1][i])) 
											   * tilting_subcomponent)
		if conjugate_field_change:
			self.matrix_representation += (conjugate_field_change/2.0
										   * self._tilting_component())
			self.matrix_representation += ((new_conjugate_field**2
											-self.conjugate_field**2)/8.0
										   * np.eye(self.hilbert_space_dimension**2))
			self.conjugate_field = new_conjugate_field