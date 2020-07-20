import numpy as np
import scipy.linalg


class spin_representation(object):

	def __init__(self, sites):
		self.sites = sites
		self._state_eigenspaces

	def _left_translation(self, number):
		"""Cycles the bits of a number left, wrapping at sites."""
		final_site_test = number & 2**(self.sites-1)
		if final_site_test == 2**(self.sites-1):
			shifted_number = number << 1
			cycled_number = shifted_number - 2**self.sites + 1
		else:
			cycled_number = number << 1
		return cycled_number

	def _cycles(self):
		"""Constructs invariant sets of spin states under translation."""
		spin_states = [N for N in range(2**self.sites)][::-1]
		state_cycles = []
		while len(spin_states) != 0:
			current_cycle = []
			highest_configuration = spin_states.pop(0)
			current_cycle.append(highest_configuration)
			next_configuration = self._left_translation(highest_configuration)
			while next_configuration != highest_configuration:
				spin_states.remove(next_configuration)
				current_cycle.append(next_configuration)
				next_configuration = self._left_translation(next_configuration)
			state_cycles.append(current_cycle)
		return state_cycles

	def _state_eigenspaces(self):
		"""Constructs the translation eigenspace projection operators."""
		state_cycles = self._cycles()
		eigenspaces = [[] for i in range(self.sites)]
		for cycle in state_cycles:
			for vector_index in range(len(cycle)):
				vector = np.zeros((2**self.sites), complex)
				for cycle_index in range(len(cycle)):
					spin_index = 2**self.sites - 1 - cycle[cycle_index]
					vector[spin_index] += np.round(
						np.exp(complex(2*np.pi*cycle_index*vector_index*1j)
							/ complex(len(cycle)))/complex(np.sqrt(len(cycle))),8)
				eigenspaces[int(vector_index*self.sites/len(cycle))].append(vector)
		self.projectors = []
		self.eigenspace_dimensions = []
		for eigenspace in eigenspaces:
			self.projectors.append(np.array(eigenspace))
			self.eigenspace_dimensions.append(len(eigenspace))

	def adjoint_eigenspaces(self):
		"""Constructs eigenspaces of the translation superoperator."""
		adjoint_eigenvalues = [[i-j for j in range(self.sites)] 
							   for i in range(self.sites)]
		adjoint_eigenspace_pairs = []
		for block_index in range(-self.sites+1, self.sites):
			eigenspace_pairs = []
			for row_index in range(self.sites):
				for column_index in range(self.sites):
					if adjoint_eigenvalues[row_index][column_index] == block_index:
						eigenspace_pairs.append((row_index, column_index))
			adjoint_eigenspace_pairs.append(eigenspace_pairs)
		adjoint_eigenspaces = []
		for eigenspace_pairs in adjoint_eigenspace_pairs:
			eigenspace = []
			for pair in eigenspace_pairs:
				for state in self.projectors[pair[0]]:
					for conjugate_state in self.projectors[pair[1]]:
						eigenspace.append(np.outer(state, np.conjugate(conjugate_state)))
			adjoint_eigenspaces.append(eigenspace)
		return adjoint_eigenspaces

	def block_diagonalize(self, operator, diagonal = 0):
		"""Produces blocks of operator along specified diagonal."""
		blocks = []
		for space_index in range(self.sites):
			dual_space_index = (space_index + diagonal) % self.sites
			blocks.append((self.projectors[space_index] @ operator @ 
						   @ np.conjugate(self.projectors[dual_space_index]).T))
		return blocks