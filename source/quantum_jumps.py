import numpy as np
from scipy import linalg

class jump_trajectory_generator(object):

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
		for i in range(self.evolver_number-1):
			evolvers.append(evolvers[-1] @ evolvers[-1])
		return evolvers[::-1]

	def EvolverCombination(SmallestStepMultiple, Steps):
		Combination = []
		for Step in Steps:
			Combination.append(int(SmallestStepMultiple/Step))
			SmallestStepMultiple = SmallestStepMultiple%Step
			if SmallestStepMultiple == 0:
				Combination.extend([0 for i in range(len(Steps)-len(Combination))])
				break
		return Combination

	def TrajectoryJumpBinarySearch(PreviousUnnormedState, Evolvers, EvolverInd, 
		CurrentSteps, Steps, RandomProbability):
		CurrentUnnormedState = np.dot(Evolvers[EvolverInd+1], PreviousUnnormedState)
		Probability = linalg.norm(CurrentUnnormedState)**2
		for Index in range(EvolverInd+2,len(Evolvers)):
			if Probability > RandomProbability:
				CurrentSteps += Steps[Index-1]
				PreviousUnnormedState = np.array(CurrentUnnormedState)
				CurrentUnnormedState = np.dot(Evolvers[Index], PreviousUnnormedState)
			else:
				CurrentUnnormedState = np.dot(Evolvers[Index], PreviousUnnormedState)
			Probability = linalg.norm(CurrentUnnormedState)**2
		if Probability > RandomProbability:
			CurrentSteps += Steps[-1]
			#CurrentUnnormedState = np.dot(Evolvers[-1], CurrentUnnormedState)
			return CurrentUnnormedState, CurrentSteps
		else:
			#CurrentSteps += Steps[-1]
			#CurrentUnnormedState = np.dot(Evolvers[-1], PreviousUnnormedState)
			return PreviousUnnormedState, CurrentSteps

	def TrajectoryJump(LindbladOperators, State):
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

	def TrajectoryBinaryStepEvolution(LindbladOperators, PreviousUnnormedState, CurrentUnnormedState, 
		Probability, EvolverIndex, Evolvers, Evolutions, CurrentSteps, Steps, RndmProbability):
		for Step in range(Evolutions):
			#print(Step)
			#print(Probability)
			#print(RndmProbability)
			#print(Probability)
			if Probability <= RndmProbability:
				EvolvedState, CurrentSteps = TrajectoryJumpBinarySearch(PreviousUnnormedState, 
					Evolvers, EvolverIndex, CurrentSteps, Steps, RndmProbability)
				JumpedState = TrajectoryJump(LindbladOperators, EvolvedState)
				RndmProbability = np.random.random()
				#print("Jump")
				#print(RndmProbability)
				return JumpedState, CurrentSteps, RndmProbability, True
			CurrentSteps += Steps[EvolverIndex] 
			PreviousUnnormedState = np.array(CurrentUnnormedState)
			CurrentUnnormedState = np.dot(Evolvers[EvolverIndex], PreviousUnnormedState)
			Probability = linalg.norm(CurrentUnnormedState)**2
		return CurrentUnnormedState, CurrentSteps, RndmProbability, False

	def TrajectoryBinaryEvolution(LindbladOperators, PreviousUnnormedState, Evolvers, RandomProbability, 
		SmallestStepMultiple):
		EvolverNumber = len(Evolvers)
		Steps = [2**(EvolverNumber-1-i) for i in range(EvolverNumber)]
		CurrentSteps = 0
		Combination = EvolverCombination(SmallestStepMultiple, Steps)
		#print(Combination)
		for EvolverIndex in range(EvolverNumber):
			if Combination[EvolverIndex] >= 1:
				CurrentUnnormedState = np.dot(Evolvers[EvolverIndex], PreviousUnnormedState)
				#print(EvolverIndex)
				#print(CurrentUnnormedState)
				Probability = linalg.norm(CurrentUnnormedState)**2
				#print(Probability)
				CurrentUnnormedState, CurrentSteps, RandomProbability, JumpOccured = TrajectoryBinaryStepEvolution(
					LindbladOperators, PreviousUnnormedState, CurrentUnnormedState, Probability, EvolverIndex, 
					Evolvers, Combination[EvolverIndex]-1, CurrentSteps, Steps, RandomProbability)
			if JumpOccured:
				#print("jump")
				del Combination
				del Steps
				del EvolverNumber
				#print(CurrentSteps)
				#print("Rand")
				#print(RandomProbability)
				#print(CurrentUnnormedState)
				CurrentUnnormedState, RandomProbability = TrajectoryBinaryEvolution(LindbladOperators, CurrentUnnormedState, 
					Evolvers, RandomProbability, SmallestStepMultiple-CurrentSteps)
				break
			PreviousUnnormedState = np.array(CurrentUnnormedState)
		return CurrentUnnormedState, RandomProbability

	def TrajectoryAmplitudesBinary(LindbladOperators, UnnormedState, OutputNumber, OutputSmallestStepMultiple, 
		Evolvers, ObservationFunction, *ObservationFunctionArgs):
		TimeDependentAmplitude = [UnnormedState]
		RandomProbability = np.random.random()
		Results = [ObservationFunction(UnnormedState, *ObservationFunctionArgs)]
		for i in range(OutputNumber):
			print(i)
			UnnormedState, RandomProbability = TrajectoryBinaryEvolution(LindbladOperators, UnnormedState, Evolvers, 
				RandomProbability, OutputSmallestStepMultiple)
			Results.append(ObservationFunction(UnnormedState/linalg.norm(UnnormedState), *ObservationFunctionArgs))
		return Results

	def Expectation(State, Observable):
		return np.dot(np.conjugate(State).T,np.dot(Observable,State))

	def StochasticAverage(LindbladOperators, UnnormedState, OutputNumber, OutputSmallestStepMultiple, 
		Evolvers, Samples, ObservationFunction, *ObservationFunctionArgs):
		Results = np.array(TrajectoryAmplitudesBinary(LindbladOperators, UnnormedState, OutputNumber, OutputSmallestStepMultiple, 
		Evolvers, ObservationFunction, *ObservationFunctionArgs))
		for i in range(Samples-1):
			print(i)
			Results += np.array(TrajectoryAmplitudesBinary(LindbladOperators, UnnormedState, OutputNumber, OutputSmallestStepMultiple, 
		Evolvers, ObservationFunction, *ObservationFunctionArgs))
		return Results/Samples
