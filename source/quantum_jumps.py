import numpy as np
from scipy import linalg

class jump_trajectory_generator(object):

	"""Generates quantum jump trajectories using a binary search.

	"""

	def __init__(self, lindbladian, smallest_time, evolver_number):
		self.model = lindbladian
		self.smallest_time = smallest_time
		self.evolver_number = evolver_number
		self._effective_hamiltonian()
		self._binary_evolution_operators()

	def _effective_hamiltonian(self):
		"""The non-hermitian generator for evolution between jumps."""
		self.effective_hamiltonian = np.array(self.model.hamiltonian)
		for jump in self.model.jump_operators:
			self.effective_hamiltonian -= 0.5j * jump.conjugate().T @ jump

	def _binary_evolution_operators(self):
		"""Evolution operators for the binary search of jump times."""
		smallest_evolver = linalg.expm(-1j*self.smallest_time*self.effective_hamiltonian)
		evolvers = [smallest_evolver]
		for i in range(self.evolver_number - 1):
			evolvers.append(evolvers[-1] @ evolvers[-1])
		self.evolver_steps = [2**(self.evolver_number - 1 - i) 
							  for i in range(self.evolver_number)]
		self.evolvers = evolvers[::-1]

	def _evolver_combination(self, steps):
		"""Calculates the combination of evolvers to reach the next observation."""
		combination = []
		for evolver_step in self.evolver_steps:
			combination.append(int(steps / evolver_step))
			steps = steps % evolver_step
			if steps == 0:
				combination.extend([0 for i in range(len(self.evolver_steps)-len(combination))])
				break
		return combination

	def _jump_search(self, evolver_index):
		self.current_state = np.dot(self.evolvers[evolver_index+1], self.previous_state)
		probability = linalg.norm(self.current_state)**2
		for i in range(evolver_index+2, len(self.evolvers)):
			if probability > self.random:
				self.current_steps += self.evolver_steps[i - 1]
				self.previous_state = np.array(self.current_state)
				self.current_state = np.dot(self.evolvers[i], self.previous_state)
			else:
				self.current_state = np.dot(self.evolvers[i], self.previous_state)
			probability = linalg.norm(self.current_state)**2
		if probability > self.random:
			self.current_steps += self.evolver_steps[-1]
			#self.current_state = np.dot(Evolvers[-1], self.current_state)
			return self.current_state, self.current_steps
		else:
			#self.current_steps += self.evolver_steps[-1]
			#self.current_state = np.dot(Evolvers[-1], self.previous_state)
			return self.previous_state, self.current_steps

	def _jump(LindbladOperators, State):
		#print(State)
		UnnormedJumpStates = [np.dot(np.array(Op), State) for Op in LindbladOperators]
		UnnormedProbabilities = np.array([linalg.norm(UnnormedJumpState)**2 for UnnormedJumpState in UnnormedJumpStates])
		#print(UnnormedProbabilities)
		NormedProbabilities = UnnormedProbabilities/sum(UnnormedProbabilities)
		StackedProbabilities = [sum(NormedProbabilities[:(index+1)]) for index in range(len(NormedProbabilities))]
		RandomJump = np.random.random()
		Jump = np.searchsorted(StackedProbabilities, RandomJump, side = 'right')
		PostJumpState = UnnormedJumpStates[Jump]/linalg.norm(UnnormedJumpStates[Jump])
		return PostJumpState

	def _binary_evolution_step(LindbladOperators, self.previous_state, self.current_state, 
		Probability, EvolverIndex, Evolvers, Evolutions, self.current_steps, Steps, self.random):
		for Step in range(Evolutions):
			#print(Step)
			#print(Probability)
			#print(self.random)
			#print(Probability)
			if Probability <= self.random:
				EvolvedState, self.current_steps = self._jump_search(self.previous_state, 
					Evolvers, EvolverIndex, self.current_steps, Steps, self.random)
				JumpedState = self._jump(LindbladOperators, EvolvedState)
				self.random = np.random.random()
				#print("Jump")
				#print(self.random)
				return JumpedState, self.current_steps, self.random, True
			self.current_steps += Steps[EvolverIndex] 
			self.previous_state = np.array(self.current_state)
			self.current_state = np.dot(Evolvers[EvolverIndex], self.previous_state)
			Probability = linalg.norm(self.current_state)**2
		return self.current_state, self.current_steps, self.random, False

	def _binary_evolution(LindbladOperators, self.previous_state, Evolvers, self.random, 
		SmallestStepMultiple):
		EvolverNumber = len(Evolvers)
		Steps = [2**(EvolverNumber-1-i) for i in range(EvolverNumber)]
		self.current_steps = 0
		Combination = self._evolver_combination(SmallestStepMultiple, Steps)
		#print(Combination)
		for EvolverIndex in range(EvolverNumber):
			if Combination[EvolverIndex] >= 1:
				self.current_state = np.dot(Evolvers[EvolverIndex], self.previous_state)
				#print(EvolverIndex)
				#print(self.current_state)
				Probability = linalg.norm(self.current_state)**2
				#print(Probability)
				self.current_state, self.current_steps, self.random, JumpOccured = self._binary_evolution_step(
					LindbladOperators, self.previous_state, self.current_state, Probability, EvolverIndex, 
					Evolvers, Combination[EvolverIndex]-1, self.current_steps, Steps, self.random)
			if JumpOccured:
				#print("jump")
				del Combination
				del Steps
				del EvolverNumber
				#print(self.current_steps)
				#print("Rand")
				#print(self.random)
				#print(self.current_state)
				self.current_state, self.random = self._binary_evolution(LindbladOperators, self.current_state, 
					Evolvers, self.random, SmallestStepMultiple-self.current_steps)
				break
			self.previous_state = np.array(self.current_state)
		return self.current_state, self.random

	def trajectory(LindbladOperators, UnnormedState, OutputNumber, OutputSmallestStepMultiple, 
		Evolvers, ObservationFunction, *ObservationFunctionArgs):
		TimeDependentAmplitude = [UnnormedState]
		self.random = np.random.random()
		Results = [ObservationFunction(UnnormedState, *ObservationFunctionArgs)]
		for i in range(OutputNumber):
			print(i)
			UnnormedState, self.random = self._binary_evolution(LindbladOperators, UnnormedState, Evolvers, 
				self.random, OutputSmallestStepMultiple)
			Results.append(ObservationFunction(UnnormedState/linalg.norm(UnnormedState), *ObservationFunctionArgs))
		return Results

	def expectation(State, Observable):
		return np.dot(np.conjugate(State).T,np.dot(Observable,State))

	def stochastic_average(LindbladOperators, UnnormedState, OutputNumber, OutputSmallestStepMultiple, 
		Evolvers, Samples, ObservationFunction, *ObservationFunctionArgs):
		Results = np.array(self.trajectory(LindbladOperators, UnnormedState, OutputNumber, OutputSmallestStepMultiple, 
		Evolvers, ObservationFunction, *ObservationFunctionArgs))
		for i in range(Samples-1):
			print(i)
			Results += np.array(self.trajectory(LindbladOperators, UnnormedState, OutputNumber, OutputSmallestStepMultiple, 
		Evolvers, ObservationFunction, *ObservationFunctionArgs))
		return Results/Samples
