import sys
import os
import time
import numpy as np
source_path = os.path.join("source/")
sys.path.insert(0,source_path)
import metastability
source_path = os.path.join("models/")
sys.path.insert(0,source_path)
import OpenQuantumMetastabilityMethods as kcqg
import dissipative_quantum_east as dqe
from scipy import linalg

Sites = 3
DecayRate = 1
FieldRange = [0.1,0.1]
FieldStep = 1
TempRange = [0.0001,0.0001]
TempStep = 1
Hardness = 0.999
eMSNumber = Sites + 1

sites = Sites
decay_rate = DecayRate
field = FieldRange[0]
temperature = TempRange[0]
hardness = Hardness
print("Start new")
current_time = time.time()
model = dqe.master_operator(sites, decay_rate, field, temperature, hardness)
print(sorted(linalg.eigvals(model.matrix_representation))[-1:-5:-1])
current_time = time.time() - current_time
print("New diagonalization time: %s"%(current_time))

print("Start symmetrized")
current_time = time.time()
model2 = dqe.symmetrized_master_operator(sites, decay_rate, field, temperature, hardness)
print(sorted(linalg.eigvals(model2.matrix_representation[0]))[-1:-5:-1])
print(sorted(linalg.eigvals(model2.matrix_representation[1]))[-1:-5:-1])
print(sorted(linalg.eigvals(model2.matrix_representation[2]))[-1:-5:-1])
current_time = time.time() - current_time
print("Symmetrized diagonalization time: %s"%(current_time))

print("Start")
current_time = time.time()
Evals, LeftEvecs, RightEvecs, BlockIndicies, FieldVals, TemperatureVals, ExtendedEvals = kcqg.GlassMasterOpSpectrumVs_FieldPlusTemperatureEvecs(
	DecayRate, FieldRange, FieldStep, TempRange, TempStep, Hardness, Sites, eMSNumber,
	ExtendEvals = True, Extension = 1)
print(Evals)
current_time = time.time() - current_time
print("Diagonalization time: %s"%(current_time))

LeftEmats, RightEmats = kcqg.KCQG_EigenvectorBasistoHermitianMatrixBasis2DParameterSpace(
	Evals, LeftEvecs, RightEvecs, BlockIndicies, Sites)

current_time = time.time()
simplex_vertices = metastability.simplex_vertices(LeftEmats[0][0],
												  rotations = 3)
print("Standard simplex algorithm time: %s" %(time.time() - current_time))

probability_operators = metastability.probability_operators(LeftEmats[0][0], 
															simplex_vertices)
classicality = metastability.classicality(probability_operators)
print("Vertices:")
for i in range(eMSNumber):
	print(np.around(simplex_vertices[i].real,3))
print("Classicality:")
print(classicality)



StateSpaceSymmetryTransformation = kcqg.SpinBasisCyclicUnitaryGenerator(Sites)
ObservableNewBasisVector = kcqg.StdBMCycBVMap(StateSpaceSymmetryTransformation, Sites)
ObservableVectorFull = np.array(ObservableNewBasisVector[0])
for index in range(1, len(ObservableNewBasisVector)):
	ObservableVectorFull = np.concatenate((ObservableVectorFull, ObservableNewBasisVector[index]))
StateSpaceSymmetryTransformation = kcqg.CycBVCycBMMap(ObservableVectorFull, Sites)

current_time = time.time()
simplex_vertices = metastability.simplex_vertices(
	LeftEmats[0][0], 
	mode = 'cyclic',
	rotations = 3, 
	right_eigenmatrices = RightEmats[0][0],
	symmetry_transformation = StateSpaceSymmetryTransformation,
	power_for_identity = Sites)
print("Symmetrized algorithm time: %s" %(time.time() - current_time))

probability_operators = metastability.probability_operators(LeftEmats[0][0], 
															simplex_vertices)
classicality = metastability.classicality(probability_operators)
print("Symmetrized algorithm vertices:")
for i in range(eMSNumber):
	print(np.around(simplex_vertices[i].real,3))
print("Symmetrized algorithm classicality")
print(classicality)
