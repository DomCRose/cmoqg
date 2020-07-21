import sys
import os
import itertools
import numpy as np
import random_linear_algebra as rla

def hermitian_basis(left_eigenmatrices, right_eigenmatrices):
	pass

def simplex_vertices(left_eigenmatrices, mode = 'random', rotations = 3, 
					 threshold = 0.05, right_eigenmatrices = None, 
					 symmetry_transformation = None, power_for_identity = None):
	"""Wrapper for calculating the best simplex approximation."""
	if mode == 'random':
		return _simplex_vertices_random(left_eigenmatrices, rotations, 
										threshold)
	elif mode == 'cyclic':
		return _simplex_vertices_cyclic(left_eigenmatrices, rotations, 
										threshold, right_eigenmatrices, 
										symmetry_transformation, power_for_identity)
	else:
		raise ValueError("mode must be either 'random' or 'cyclic'. "
						 "mode given was {}".format(mode))

def _extreme_state_coefficients(left_basis, coefficient_matrices):
	"""Calculates extreme states and their coefficients."""
	number_of_matrices = len(left_basis)
	eigenvalues, eigenvectors = np.linalg.eig(left_basis)
	extreme_states = np.concatenate((eigenvectors[np.arange(number_of_matrices), 
												  :, eigenvalues.argmax(axis = 1)],
									 eigenvectors[np.arange(number_of_matrices), 
									 			  :, eigenvalues.argmin(axis = 1)]))
	extreme_state_coefficients = np.einsum('li,kij,lj->lk', 
										   extreme_states.conjugate(), 
										   coefficient_matrices[:-1], 
										   extreme_states).real
	return extreme_state_coefficients

def _candidates(left_eigenmatrices, rotations, number_of_vertices):
	"""Extracts candidate vertices a set of rotated left bases."""
	candidate_vertices = _extreme_state_coefficients(left_eigenmatrices[:-1], 
													 left_eigenmatrices)
	for i in range(rotations):
		random_rotation = rla.random_orthogonal_matrix(number_of_vertices - 1)
		rotated_basis = np.einsum('ij,jkl', random_rotation, left_eigenmatrices[:-1])
		candidate_vertices = np.concatenate(
			(candidate_vertices, 
			 _extreme_state_coefficients(rotated_basis, left_eigenmatrices)))
	return candidate_vertices

def _clustering(vectors, threshold):
	"""Groups vectors by those they are first found closest too."""
	clustered_vectors = []
	vectors = list(vectors)
	while len(vectors) > 0:
		clustered_vectors.append(vectors.pop(0))
		for i in range(len(vectors)):
			vector = vectors.pop(0)
			if np.linalg.norm(vector - clustered_vectors[-1]) > threshold:
				vectors.append(vector)
	return clustered_vectors

def _maximal_volume_simplex(candidate_vertices, number_of_vertices):
	"""Finds candidate combination with the largest volume simplex."""
	combinations = itertools.combinations(candidate_vertices, number_of_vertices)
	current_volume = 0
	for combination in combinations:
		combination = np.array(combination)
		shifted_vertices = combination[:-1] - combination[-1]
		combination_volume = abs(np.linalg.det(shifted_vertices))
		if combination_volume > current_volume:
			current_vertices = combination
			current_volume = combination_volume
	return np.concatenate((current_vertices, np.ones((number_of_vertices, 1))), axis = 1)

def _simplex_vertices_random(left_eigenmatrices, rotations, threshold):
	"""Uses random rotations to calculate the best simplex vertices.

	Collects extreme eigenvectors of a set of randomly rotated left 
	matrix bases, reduced to distinct set of candidates, and compares
	volumes of combinations to find the best simplex approximation.
	"""
	number_of_vertices = len(left_eigenmatrices)
	candidate_vertices = _candidates(left_eigenmatrices, rotations, number_of_vertices)
	clustered_candidates = _clustering(candidate_vertices, threshold)
	simplex_vertices = _maximal_volume_simplex(clustered_candidates, 
											   number_of_vertices)
	return simplex_vertices

def _divisors(number):
	"""Finds a numbers divisors and returns them in ascending order."""
	if np.sqrt(number)%1 == 0:
		possible_divisors = np.arange(1, int(np.sqrt(number)) + 1)
		lower_divisors = possible_divisors[np.mod(number, possible_divisors) == 0]
		divisors = np.concatenate((lower_divisors, number/lower_divisors, 
								   [int(np.sqrt(number))]))
	else:
		possible_divisors = np.arange(1, int(np.sqrt(number)) + 1)
		print(possible_divisors)
		lower_divisors = possible_divisors[np.mod(number, possible_divisors) == 0]
		print(lower_divisors)
		divisors = np.concatenate((lower_divisors, number/lower_divisors))
	return sorted(divisors)

def _invariant_sets(vectors, threshold, transformation, power_for_identity):
	"""Groups vectors according to their transformations."""
	set_lengths = _divisors(power_for_identity)
	print(set_lengths)
	invariant_sets = [[] for i in range(len(set_lengths))]
	set_length_counts = np.zeros(len(set_lengths))
	vectors = list(vectors)
	while len(vectors) > 0:
		invariant_set = vectors.pop(0)[np.newaxis]
		for i in range(len(vectors)):
			vector = vectors.pop(0)
			if np.linalg.norm(vector - invariant_set[-1]) > threshold:
				vectors.append(vector)
		for power in range(power_for_identity):
			transformed_vector = np.dot(transformation, invariant_set[-1])
			if np.linalg.norm(invariant_set[0] - transformed_vector) < threshold:
				invariant_sets[set_lengths.index(power + 1)].append(invariant_set)
				set_length_counts[set_lengths.index(power + 1)] += 1
				break
			for i in range(len(vectors)):
				vector = vectors.pop(0)
				if np.linalg.norm(vector - invariant_set[-1]) > threshold:
					vectors.append(vector)
			invariant_set = np.concatenate((invariant_set, 
											transformed_vector[np.newaxis]))
	return invariant_sets, set_length_counts

def _valid_sum_combinations(numbers, number_counts, total, partial_count = None):
	if type(partial_count) != np.ndarray:
		partial_count = np.zeros(len(numbers))
	combination_counts = []
	current_total = np.dot(numbers, partial_count)
	if current_total == total:
		combination_counts.append(partial_count)
	if current_total >= total:
		return combination_counts
	for index in range(len(number_counts)):
		if number_counts[index] > 0:
			number_counts[index] -= 1
			partial_count[index + (len(numbers) - len(number_counts))] += 1
			combination_counts.extend(_valid_sum_combinations(
				numbers, np.copy(number_counts[index:]), total, np.copy(partial_count)))
			partial_count[index + (len(numbers) - len(number_counts))] -= 1
	return combination_counts

def _combination_volumes(combination, candidate_groups, set_length_counts, 
						 set_lengths, volume = 0, vertices = [],
						 current_vertices = [], length_index = 0):
	if len(combination) != length_index + 1:
		if combination[length_index] > 0:
			group_combinations = itertools.combinations(candidate_groups[length_index],
														int(combination[length_index]))
			for groups in group_combinations:
				if len(current_vertices) == 0:
					partial_vertices = np.concatenate(groups)
				else:
					partial_vertices = np.concatenate((current_vertices,
													   np.concatenate(groups)))
				volume, vertices = _combination_volumes(
					combination, candidate_groups, set_length_counts, set_lengths,
					volume, vertices, current_vertices = partial_vertices, 
					length_index = length_index + 1)
		else:
			volume, vertices = _combination_volumes(
				combination, candidate_groups, set_length_counts, set_lengths,
				volume, vertices, current_vertices, length_index + 1)
		return volume, vertices

	else:
		if combination[length_index] > 0:
			group_combinations = itertools.combinations(candidate_groups[length_index],
														int(combination[length_index]))
			for groups in group_combinations:
				if len(current_vertices) == 0:
					partial_vertices = np.concatenate(groups)
				else:
					partial_vertices = np.concatenate((current_vertices,
													   np.concatenate(groups)))
				shifted_vertices = partial_vertices[:-1] - partial_vertices[-1]
				combination_volume = abs(np.linalg.det(shifted_vertices))
				if combination_volume > volume:
					vertices = partial_vertices
					volume = combination_volume
		else:
			shifted_vertices = current_vertices[:-1] - current_vertices[-1]
			combination_volume = abs(np.linalg.det(shifted_vertices))
			if combination_volume > volume:
				vertices = current_vertices
				volume = combination_volume
		return volume, vertices

def _maximal_volume_combination(candidate_groups, set_length_counts,
								power_for_identity, number_of_vertices):
	set_lengths = _divisors(power_for_identity)
	set_length_combinations = _valid_sum_combinations(set_lengths, set_length_counts,
													  number_of_vertices)
	current_volume = 0
	current_vertices = []
	for combination in set_length_combinations[::-1]:
		current_volume, current_vertices = _combination_volumes(
			combination, candidate_groups, set_length_counts, set_lengths,
			current_volume, current_vertices)
	print(current_volume)
	return np.concatenate((current_vertices, np.ones((number_of_vertices, 1))), axis = 1)


def _simplex_vertices_cyclic(left_eigenmatrices, rotations,
							 threshold, right_eigenmatrices, 
							 cyclic_transformation, power_for_identity):
	"""Uses symmetries to calculate the best simplex vertices."""
	if right_eigenmatrices == None:
		raise ValueError("right_eigenmatrices must be provided in 'cyclic' mode.")
	#if cyclic_transformation == None:
	#	raise ValueError("cyclic_transformation must be specified in 'cyclic' mode.")
	if power_for_identity == None:
		raise ValueError("power_for_identity must be specified in 'cyclic' mode.")
	number_of_vertices = len(left_eigenmatrices)
	candidate_vertices = _candidates(left_eigenmatrices, rotations, number_of_vertices)
	projected_cyclic_transformation = np.einsum(
		'ikl,lm,jmn,nk->ij', 
		left_eigenmatrices[:-1],
		cyclic_transformation, 
		right_eigenmatrices[:-1], 
		np.conjugate(cyclic_transformation).T)
	candidate_sets, set_length_counts = _invariant_sets(candidate_vertices, threshold,
														projected_cyclic_transformation, 
														power_for_identity)
	simplex_vertices = _maximal_volume_combination(candidate_sets, set_length_counts,
												   power_for_identity, number_of_vertices)
	return simplex_vertices

def probability_operators(left_eigenmatrices, simplex_vertices):
	"""Constructs the probability operators dual to the extreme states."""
	return np.einsum('ji,jkl', np.linalg.inv(simplex_vertices), left_eigenmatrices)

def extreme_metastable_states(right_eigenmatrices, simplex_vertices):
	"""Constructs the metastable phases on the vertices of the simplex."""
	return np.einsum('ij,jkl', simplex_vertices, right_eigenmatrices)

def classicality(probability_operators):
	"""Calculate a bound on the average error from the simplex."""
	eigenvalues = np.linalg.eigvals(probability_operators)
	return np.sum(-eigenvalues[eigenvalues < 0])/len(probability_operators[0])