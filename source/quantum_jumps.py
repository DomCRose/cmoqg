import numpy as np
from scipy import linalg

def EffectiveHamiltonian(Hamiltonian, LindbladOps):
	EffectiveHam = np.array(Hamiltonian)
	for Op in LindbladOps:
		EffectiveHam = EffectiveHam - 0.5j * np.dot(np.array(Op).conjugate().T, np.array(Op))
	return EffectiveHam
def BinaryEvolvers(EffectiveHamiltonian, SmallestTime, NumberOfEvolvers):
	SmallestEvolver = linalg.expm(-1j*SmallestTime*EffectiveHamiltonian)
	Evolvers = [SmallestEvolver]
	for i in range(NumberOfEvolvers-1):
		Evolvers.append(np.dot(Evolvers[-1],Evolvers[-1]))
	return Evolvers[::-1]
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
