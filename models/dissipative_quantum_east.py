import sys
import os
import math
import scipy
import numpy as np
import numpy.linalg as la
source_path = os.path.join(os.pardir, "source/")
sys.path.insert(0, source_path)
import cyclic_representations
import master_operators
from scipy import linalg

def pauli(label):
	"""Returns the specified Pauli matrix."""
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

def local_operator(operator, index, sites):
	"""Constructs a local operator based on the given operator for a particular site."""
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

def spatial_average(operator, sites):
	"""Constructs an operator giving the spatial average of a local operators."""
	observable = np.zeros((2**sites, 2**sites), dtype = complex)
	for site in range(1, sites + 1):
		observable += local_operator(operator, site, sites)
	return observable/sites

def local_operators(operator, sites):
	"""Constructs local operators based on the given operator for each site."""
	observables = []
	for site in range(1, sites + 1):
		observables.append(local_operator(operator, site, sites))
	return np.array(observables)


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
		sigma_x = pauli('x')
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
		sigma_minus = pauli('-')
		sigma_plus = pauli('+')
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

class tilted_master_operator(master_operators.activity_biased_lindbladian,
							 master_operator):

	def __init__(
			self,
			sites,
			decay_rate,
			field,
			temperature,
			hardness,
			biases):
		self.sites = sites
		self.decay_rate = decay_rate
		self.field = field
		self.temperature = temperature
		self.hardness = hardness
		self.hilbert_space_dimension = 2**sites
		self.biases = biases
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
		self.jumps = len(self.jump_operators)
		self.symmetry_rep = cyclic_representations.spin_representation(self.sites)
		self._blocks()
		self._generate_matrix_representation()

	def _blocks(self):
		self.hamiltonian = self.symmetry_rep.block_diagonalize(self.hamiltonian)
		jump_blocks = []
		jump_block_eigenvalues = []
		for new_index in range(int(self.jumps/2)):
			decay_jump_operator = np.copy(self.jump_operators[0])
			temp_jump_operator = np.copy(self.jump_operators[1])
			for jump_index in range(2, self.jumps, 2):
				prefactor = np.exp(complex(2*np.pi*jump_index*new_index*1j) / self.jumps)
				decay_jump_operator += prefactor * self.jump_operators[jump_index]
				temp_jump_operator += prefactor * self.jump_operators[jump_index + 1]
			decay_jump_operator /= np.sqrt(self.jumps/2)
			temp_jump_operator /= np.sqrt(self.jumps/2)
			jump_blocks.append(
				self.symmetry_rep.block_diagonalize(decay_jump_operator, new_index))
			jump_block_eigenvalues.append(new_index)
			jump_blocks.append(
				self.symmetry_rep.block_diagonalize(temp_jump_operator, new_index))
			jump_block_eigenvalues.append(new_index)
		self.jump_operators = jump_blocks
		self.jump_operator_eigenvalues = jump_block_eigenvalues
		self.symmetry_rep.eigenspace_pairs()
		self.eigenspace_pairs = self.symmetry_rep.adjoint_eigenspace_pairs
		self.eigenspace_dimensions = self.symmetry_rep.eigenspace_dimensions
		self.eigenspace_number = len(self.eigenspace_dimensions)

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

def hermitian_basis(evals, lemats, remats, sym_inds, N, rounding = 8):
	"""Constructs a hermitian basis of the provided eigenmatrices.
	
	Requires knowledge of the system size, and the eigenvalues and 
	symmetry eigenspace of each eigenmatrix.
	"""
	evals = list(evals)
	remats = list(remats)
	lemats = list(lemats)
	sym_inds = list(sym_inds)
	herm_lemats = []
	herm_remats = []
	while len(lemats) >= 1:
		curr_eval = evals.pop(0)
		curr_left = lemats.pop(0)
		curr_right = remats.pop(0)
		curr_sym = sym_inds.pop(0)
		if curr_sym == 0 or curr_sym == float(N)/2.0:
			if round(np.trace(
				la.matrix_power(curr_right + np.conjugate(curr_right).T, 2)), 5) == 0:
				herm_remats.append((curr_right - np.conjugate(curr_right).T)/2j)
				herm_lemats.append((curr_left - np.conjugate(curr_left).T)/2j)
			else:
				herm_remats.append((curr_right + np.conjugate(curr_right).T)/2)
				herm_lemats.append((curr_left + np.conjugate(curr_left).T)/2)
		else:
			for i in range(len(sym_inds)):
				if (sym_inds[i] == float(N) - curr_sym 
					and round(abs(curr_eval - np.conjugate(evals[i])), rounding) == 0):
					evals.pop(i)
					sym_inds.pop(i)
					second_left = lemats.pop(i)
					second_right = remats.pop(i)
					herm_remats.append(curr_right + second_right)
					herm_remats.append((curr_right - second_right)/1j)
					herm_lemats.append((curr_left + second_left)/2)
					herm_lemats.append((curr_left - second_left)/2j)
					break
			else:
				herm_remats.append((curr_right + np.conjugate(curr_right).T)/2)
				herm_lemats.append((curr_left + np.conjugate(curr_left).T)/2)
	return herm_lemats, herm_remats