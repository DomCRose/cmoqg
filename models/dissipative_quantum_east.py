import sys
import os
import math
import scipy
import numpy as np
source_path = os.path.join(os.pardir, "source/")
sys.path.insert(0, source_path)
import special_unitary_groups as su2
import cyclic_representations
import master_operators

def local_operator(operator, index, sites):
	identity = np.eye(len(operator))
	if index == 1:
		output = np.array(operator)
		for i in range(sites - 1):
			output = np.kron(output, identity)
	else:
		output = identity
		for i in range(index - 2):
			output = np.kron(output, identity)
		output = np.kron(output, np.array(operator))
		for i in range(sites - index):
			output = np.kron(output, identity)
	return output


class master_operator(master_operators.lindbladian):

	def __init__(
			self,
			sites,
			decay_rate,
			field,
			temperature,
			hardness):
		self.sites = sites
		self.decay_rate = decay_rate
		self.field = field
		self.temperature = temperature
		self.hardness = hardness
		self.hilbert_space_dimension = 2**sites
		self._operators()
		self._generate_matrix_representation()

	def _constraint_operator(self):
		delta = math.sqrt((self.temperature + self.decay_rate)**2 + 16*self.field**2)
		projection = 0.5*np.array(
			[[1 + (self.temperature + self.decay_rate)/delta, 
			  (4j*self.field)/delta],
			 [(-4j*self.field)/delta, 
			  1 - (self.temperature + self.decay_rate)/delta]])
		self.constraint_operator = (self.hardness*projection 
									+ (1 - self.hardness)*np.eye(2, dtype = complex))

	def _hamiltonian(self):
		sigma_x = su2.pauli('x')
		self.hamiltonian = np.zeros((self.hilbert_space_dimension,
									 self.hilbert_space_dimension), 
									dtype = complex)
		for site in range(1, self.sites + 1):
			local_sigma_x = local_operator(sigma_x, site, self.sites)
			local_constraint = local_operator(
				self.constraint_operator, (site % self.sites) + 1, self.sites)
			self.hamiltonian += local_sigma_x @ local_constraint @ local_constraint
		self.hamiltonian *= self.field

	def _jump_operators(self):
		self.jump_operators = []
		sqrt_decay = math.sqrt(self.decay_rate)
		sqrt_temperature = math.sqrt(self.temperature)
		sigma_minus = su2.pauli('-')
		sigma_plus = su2.pauli('+')
		for site in range(1, self.sites + 1):
			local_constraint = local_operator(
				self.constraint_operator, (site % self.sites) + 1, self.sites)
			self.jump_operators.append(sqrt_decay * (
				local_operator(sigma_minus, site, self.sites) @ local_constraint))
			self.jump_operators.append(sqrt_temperature * (
				local_operator(sigma_plus, site, self.sites) @ local_constraint))

	def _operators(self):
		self._constraint_operator()
		self._hamiltonian()
		self._jump_operators()

	def update_parameters(self, decay_rate = None, field = None, temperature = None,
						  hardness = None):
		if decay_rate != None:
			self.decay_rate = decay_rate
		if field != None:
			self.field = field
		if temperature != None:
			self.temperature = temperature
		if hardness != None:
			self.hardness = hardness
		self._operators()
		self._generate_matrix_representation()
		

class symmetrized_master_operator(master_operators.weakly_symmetric_lindbladian,
								  master_operator):

	def __init__(
			self,
			sites,
			decay_rate,
			field,
			temperature,
			hardness):
		self.sites = sites
		self.decay_rate = decay_rate
		self.field = field
		self.temperature = temperature
		self.hardness = hardness
		self.hilbert_space_dimension = 2**sites
		self._operators()
		self.jump_number = len(self.jump_operators)
		self.spin_rep = cyclic_representations.spin_representation(self.sites)
		self._blocks()
		self._generate_matrix_representation()

	def _blocks(self):
		self.hamiltonian = self.spin_rep.block_diagonalize(self.hamiltonian)
		jump_blocks = []
		for new_index in range(self.jump_number):
			jump_operator = self.jump_operators[0]
			for jump_index in range(1, self.jump_number):
				jump_operator += np.exp(complex(2*np.pi*jump_index*new_index*1j)
										/ self.jump_number)*self.jump_operators[jump_index]
			jump_operator /= np.sqrt(self.jump_number)
			jump_blocks.append(self.spin_rep.block_diagonalize(jump_operator, new_index))
		self.jump_operators = jump_blocks


	def update_parameters(self, decay_rate = None, field = None, temperature = None,
						  hardness = None):
		if decay_rate != None:
			self.decay_rate = decay_rate
		if field != None:
			self.field = field
		if temperature != None:
			self.temperature = temperature
		if hardness != None:
			self.hardness = hardness
		self._operators()
		self._blocks()
		self._generate_matrix_representation()