"""This file contains the original code for the project, being reworked into clearer, more
efficient code in the other source files."""


import sys
import time
import numpy as np
import itertools
import scipy
import copy
from numpy import linalg as la
"""
from matplotlib import pyplot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
"""
from scipy import linalg
"""
from matplotlib.colors import SymLogNorm
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
"""
from scipy.sparse import linalg as spla
"""
def Viridis():
	_viridis_data = [[0.267004, 0.004874, 0.329415],[0.268510, 0.009605, 0.335427],[0.269944, 0.014625, 0.341379],[0.271305, 0.019942, 0.347269],[0.272594, 0.025563, 0.353093],[0.273809, 0.031497, 0.358853],[0.274952, 0.037752, 0.364543],[0.276022, 0.044167, 0.370164],[0.277018, 0.050344, 0.375715],[0.277941, 0.056324, 0.381191],[0.278791, 0.062145, 0.386592],[0.279566, 0.067836, 0.391917],[0.280267, 0.073417, 0.397163],[0.280894, 0.078907, 0.402329],[0.281446, 0.084320, 0.407414],[0.281924, 0.089666, 0.412415],[0.282327, 0.094955, 0.417331],[0.282656, 0.100196, 0.422160],[0.282910, 0.105393, 0.426902],[0.283091, 0.110553, 0.431554],[0.283197, 0.115680, 0.436115],[0.283229, 0.120777, 0.440584],[0.283187, 0.125848, 0.444960],[0.283072, 0.130895, 0.449241],[0.282884, 0.135920, 0.453427],[0.282623, 0.140926, 0.457517],[0.282290, 0.145912, 0.461510],[0.281887, 0.150881, 0.465405],[0.281412, 0.155834, 0.469201],[0.280868, 0.160771, 0.472899],[0.280255, 0.165693, 0.476498],[0.279574, 0.170599, 0.479997],[0.278826, 0.175490, 0.483397],[0.278012, 0.180367, 0.486697],[0.277134, 0.185228, 0.489898],[0.276194, 0.190074, 0.493001],[0.275191, 0.194905, 0.496005],[0.274128, 0.199721, 0.498911],[0.273006, 0.204520, 0.501721],[0.271828, 0.209303, 0.504434],[0.270595, 0.214069, 0.507052],[0.269308, 0.218818, 0.509577],[0.267968, 0.223549, 0.512008],[0.266580, 0.228262, 0.514349],[0.265145, 0.232956, 0.516599],[0.263663, 0.237631, 0.518762],[0.262138, 0.242286, 0.520837],[0.260571, 0.246922, 0.522828],[0.258965, 0.251537, 0.524736],[0.257322, 0.256130, 0.526563],[0.255645, 0.260703, 0.528312],[0.253935, 0.265254, 0.529983],[0.252194, 0.269783, 0.531579],[0.250425, 0.274290, 0.533103],[0.248629, 0.278775, 0.534556],[0.246811, 0.283237, 0.535941],[0.244972, 0.287675, 0.537260],[0.243113, 0.292092, 0.538516],[0.241237, 0.296485, 0.539709],[0.239346, 0.300855, 0.540844],[0.237441, 0.305202, 0.541921],[0.235526, 0.309527, 0.542944],[0.233603, 0.313828, 0.543914],[0.231674, 0.318106, 0.544834],[0.229739, 0.322361, 0.545706],[0.227802, 0.326594, 0.546532],[0.225863, 0.330805, 0.547314],[0.223925, 0.334994, 0.548053],[0.221989, 0.339161, 0.548752],[0.220057, 0.343307, 0.549413],[0.218130, 0.347432, 0.550038],[0.216210, 0.351535, 0.550627],[0.214298, 0.355619, 0.551184],[0.212395, 0.359683, 0.551710],[0.210503, 0.363727, 0.552206],[0.208623, 0.367752, 0.552675],[0.206756, 0.371758, 0.553117],[0.204903, 0.375746, 0.553533],[0.203063, 0.379716, 0.553925],[0.201239, 0.383670, 0.554294],[0.199430, 0.387607, 0.554642],[0.197636, 0.391528, 0.554969],[0.195860, 0.395433, 0.555276],[0.194100, 0.399323, 0.555565],[0.192357, 0.403199, 0.555836],[0.190631, 0.407061, 0.556089],[0.188923, 0.410910, 0.556326],[0.187231, 0.414746, 0.556547],[0.185556, 0.418570, 0.556753],[0.183898, 0.422383, 0.556944],[0.182256, 0.426184, 0.557120],[0.180629, 0.429975, 0.557282],[0.179019, 0.433756, 0.557430],[0.177423, 0.437527, 0.557565],[0.175841, 0.441290, 0.557685],[0.174274, 0.445044, 0.557792],[0.172719, 0.448791, 0.557885],[0.171176, 0.452530, 0.557965],[0.169646, 0.456262, 0.558030],[0.168126, 0.459988, 0.558082],[0.166617, 0.463708, 0.558119],[0.165117, 0.467423, 0.558141],[0.163625, 0.471133, 0.558148],[0.162142, 0.474838, 0.558140],[0.160665, 0.478540, 0.558115],[0.159194, 0.482237, 0.558073],[0.157729, 0.485932, 0.558013],[0.156270, 0.489624, 0.557936],[0.154815, 0.493313, 0.557840],[0.153364, 0.497000, 0.557724],[0.151918, 0.500685, 0.557587],[0.150476, 0.504369, 0.557430],[0.149039, 0.508051, 0.557250],[0.147607, 0.511733, 0.557049],[0.146180, 0.515413, 0.556823],[0.144759, 0.519093, 0.556572],[0.143343, 0.522773, 0.556295],[0.141935, 0.526453, 0.555991],[0.140536, 0.530132, 0.555659],[0.139147, 0.533812, 0.555298],[0.137770, 0.537492, 0.554906],[0.136408, 0.541173, 0.554483],[0.135066, 0.544853, 0.554029],[0.133743, 0.548535, 0.553541],[0.132444, 0.552216, 0.553018],[0.131172, 0.555899, 0.552459],[0.129933, 0.559582, 0.551864],[0.128729, 0.563265, 0.551229],[0.127568, 0.566949, 0.550556],[0.126453, 0.570633, 0.549841],[0.125394, 0.574318, 0.549086],[0.124395, 0.578002, 0.548287],[0.123463, 0.581687, 0.547445],[0.122606, 0.585371, 0.546557],[0.121831, 0.589055, 0.545623],[0.121148, 0.592739, 0.544641],[0.120565, 0.596422, 0.543611],[0.120092, 0.600104, 0.542530],[0.119738, 0.603785, 0.541400],[0.119512, 0.607464, 0.540218],[0.119423, 0.611141, 0.538982],[0.119483, 0.614817, 0.537692],[0.119699, 0.618490, 0.536347],[0.120081, 0.622161, 0.534946],[0.120638, 0.625828, 0.533488],[0.121380, 0.629492, 0.531973],[0.122312, 0.633153, 0.530398],[0.123444, 0.636809, 0.528763],[0.124780, 0.640461, 0.527068],[0.126326, 0.644107, 0.525311],[0.128087, 0.647749, 0.523491],[0.130067, 0.651384, 0.521608],[0.132268, 0.655014, 0.519661],[0.134692, 0.658636, 0.517649],[0.137339, 0.662252, 0.515571],[0.140210, 0.665859, 0.513427],[0.143303, 0.669459, 0.511215],[0.146616, 0.673050, 0.508936],[0.150148, 0.676631, 0.506589],[0.153894, 0.680203, 0.504172],[0.157851, 0.683765, 0.501686],[0.162016, 0.687316, 0.499129],[0.166383, 0.690856, 0.496502],[0.170948, 0.694384, 0.493803],[0.175707, 0.697900, 0.491033],[0.180653, 0.701402, 0.488189],[0.185783, 0.704891, 0.485273],[0.191090, 0.708366, 0.482284],[0.196571, 0.711827, 0.479221],[0.202219, 0.715272, 0.476084],[0.208030, 0.718701, 0.472873],[0.214000, 0.722114, 0.469588],[0.220124, 0.725509, 0.466226],[0.226397, 0.728888, 0.462789],[0.232815, 0.732247, 0.459277],[0.239374, 0.735588, 0.455688],[0.246070, 0.738910, 0.452024],[0.252899, 0.742211, 0.448284],[0.259857, 0.745492, 0.444467],[0.266941, 0.748751, 0.440573],[0.274149, 0.751988, 0.436601],[0.281477, 0.755203, 0.432552],[0.288921, 0.758394, 0.428426],[0.296479, 0.761561, 0.424223],[0.304148, 0.764704, 0.419943],[0.311925, 0.767822, 0.415586],[0.319809, 0.770914, 0.411152],[0.327796, 0.773980, 0.406640],[0.335885, 0.777018, 0.402049],[0.344074, 0.780029, 0.397381],[0.352360, 0.783011, 0.392636],[0.360741, 0.785964, 0.387814],[0.369214, 0.788888, 0.382914],[0.377779, 0.791781, 0.377939],[0.386433, 0.794644, 0.372886],[0.395174, 0.797475, 0.367757],[0.404001, 0.800275, 0.362552],[0.412913, 0.803041, 0.357269],[0.421908, 0.805774, 0.351910],[0.430983, 0.808473, 0.346476],[0.440137, 0.811138, 0.340967],[0.449368, 0.813768, 0.335384],[0.458674, 0.816363, 0.329727],[0.468053, 0.818921, 0.323998],[0.477504, 0.821444, 0.318195],[0.487026, 0.823929, 0.312321],[0.496615, 0.826376, 0.306377],[0.506271, 0.828786, 0.300362],[0.515992, 0.831158, 0.294279],[0.525776, 0.833491, 0.288127],[0.535621, 0.835785, 0.281908],[0.545524, 0.838039, 0.275626],[0.555484, 0.840254, 0.269281],[0.565498, 0.842430, 0.262877],[0.575563, 0.844566, 0.256415],[0.585678, 0.846661, 0.249897],[0.595839, 0.848717, 0.243329],[0.606045, 0.850733, 0.236712],[0.616293, 0.852709, 0.230052],[0.626579, 0.854645, 0.223353],[0.636902, 0.856542, 0.216620],[0.647257, 0.858400, 0.209861],[0.657642, 0.860219, 0.203082],[0.668054, 0.861999, 0.196293],[0.678489, 0.863742, 0.189503],[0.688944, 0.865448, 0.182725],[0.699415, 0.867117, 0.175971],[0.709898, 0.868751, 0.169257],[0.720391, 0.870350, 0.162603],[0.730889, 0.871916, 0.156029],[0.741388, 0.873449, 0.149561],[0.751884, 0.874951, 0.143228],[0.762373, 0.876424, 0.137064],[0.772852, 0.877868, 0.131109],[0.783315, 0.879285, 0.125405],[0.793760, 0.880678, 0.120005],[0.804182, 0.882046, 0.114965],[0.814576, 0.883393, 0.110347],[0.824940, 0.884720, 0.106217],[0.835270, 0.886029, 0.102646],[0.845561, 0.887322, 0.099702],[0.855810, 0.888601, 0.097452],[0.866013, 0.889868, 0.095953],[0.876168, 0.891125, 0.095250],[0.886271, 0.892374, 0.095374],[0.896320, 0.893616, 0.096335],[0.906311, 0.894855, 0.098125],[0.916242, 0.896091, 0.100717],[0.926106, 0.897330, 0.104071],[0.935904, 0.898570, 0.108131],[0.945636, 0.899815, 0.112838],[0.955300, 0.901065, 0.118128],[0.964894, 0.902323, 0.123941],[0.974417, 0.903590, 0.130215],[0.983868, 0.904867, 0.136897],[0.993248, 0.906157, 0.143936]]
	viridis = ListedColormap(_viridis_data, name='viridis')
	pyplot.register_cmap(name='viridis', cmap=viridis)
	pyplot.set_cmap(viridis)
	return _viridis_data
"""
def LinbladEquation(Density, Hamiltonian, LindbladOperators, LindbladCouplings):
	JumpedDensity = np.zeros((len(Density), len(Density)), dtype = complex)
	for i in range(len(LindbladOperators)):
		for j in range(len(LindbladOperators)):
			JumpedDensity += LindbladCouplings[i][j]*(np.dot(LindbladOperators[i], np.dot(Density, LindbladOperators[j].conjugate().T)) - 0.5*(np.dot(Density, np.dot(LindbladOperators[j].conjugate().T, LindbladOperators[i])) + np.dot(LindbladOperators[j].conjugate().T, np.dot(LindbladOperators[i], Density))))
	return -1j*(np.dot(Hamiltonian, Density)-np.dot(Density, np.conjugate(Hamiltonian).T)) + JumpedDensity
def LinbladEquationS(Density, Hamiltonian, LindbladOperators, LindbladCouplings, ConjugateField):
	JumpedDensity = np.zeros((len(Density), len(Density)), dtype = complex)
	for i in range(len(LindbladOperators)):
		for j in range(len(LindbladOperators)):
			JumpedDensity += LindbladCouplings[i][j]*(np.exp(-ConjugateField)*np.dot(LindbladOperators[i], np.dot(Density, LindbladOperators[j].conjugate().T)) - 0.5*(np.dot(Density, np.dot(LindbladOperators[j].conjugate().T, LindbladOperators[i])) + np.dot(LindbladOperators[j].conjugate().T, np.dot(LindbladOperators[i], Density))))
	return -1j*(np.dot(Hamiltonian, Density)-np.dot(Density, Hamiltonian)) + JumpedDensity
def LinbladEquationMultipleS(Density, Hamiltonian, LindbladOperators, LindbladCouplings, ConjugateFields):
	JumpedDensity = np.zeros((len(Density), len(Density)), dtype = complex)
	for i in range(len(LindbladOperators)):
		for j in range(len(LindbladOperators)):
			JumpedDensity += LindbladCouplings[i][j]*(np.exp(-ConjugateFields[i])*np.dot(LindbladOperators[i], np.dot(Density, LindbladOperators[j].conjugate().T)) - 0.5*(np.dot(Density, np.dot(LindbladOperators[j].conjugate().T, LindbladOperators[i])) + np.dot(LindbladOperators[j].conjugate().T, np.dot(LindbladOperators[i], Density))))
	return -1j*(np.dot(Hamiltonian, Density)-np.dot(Density, Hamiltonian)) + JumpedDensity
def MasterOperatorMatrixMultipleS(Hamiltonian, LindbladOperators, LindbladCouplings, ConjugateFields):
	MasterOperator = []
	BasisMatrix = np.zeros((len(Hamiltonian), len(Hamiltonian)), dtype = complex)
	for i in range(len(Hamiltonian)):
		for j in range(len(Hamiltonian)):
			MasterOperatorColumn = []
			BasisMatrix[i][j] += 1
			Coefficients = LinbladEquationMultipleS(BasisMatrix, Hamiltonian, LindbladOperators, LindbladCouplings, ConjugateFields)
			MasterOperatorColumn = Coefficients.flatten()
			MasterOperator.append(MasterOperatorColumn)
			BasisMatrix[i][j] = 0
	return np.array(MasterOperator).T
"""
Methods for spin representations of the cyclic group.
"""
def RightBinaryCycle(Number, Sites):
	DigitTest = Number & 1
	if DigitTest == 1:
		PreCycledNumber = Number + 2**Sites - 1
		CycledNumber = PreCycledNumber >> 1
	else:
		CycledNumber = Number >> 1
	return CycledNumber
def LeftBinaryCycle(Number, Sites):
	DigitTest = Number & 2**(Sites-1)
	if DigitTest == 2**(Sites-1):
		PreCycledNumber = Number << 1
		CycledNumber = PreCycledNumber - 2**Sites + 1
	else:
		CycledNumber = Number << 1
	return CycledNumber
def SpinBasisCyclicUnitaryGenerator(Sites):
	UnitaryTranslation = np.zeros((2**Sites,2**Sites))
	for i in range(2**Sites):
		for j in range(2**Sites):
			if 2**Sites-1-i == RightBinaryCycle(2**Sites-1-j, Sites):
				UnitaryTranslation[i][j] = 1
	return UnitaryTranslation
def IrreducibleCyclicBasisL(Sites):
	SpinBasis = [N for N in range(2**Sites)][::-1]
	Irreps = []
	while len(SpinBasis) != 0:
		CurrentIrrep = []
		IrrepLargestElement = SpinBasis.pop(0)
		CurrentIrrep.append(IrrepLargestElement)
		NextElement = RightBinaryCycle(IrrepLargestElement, Sites)
		while NextElement != IrrepLargestElement:
			SpinBasis.remove(NextElement)
			CurrentIrrep.append(NextElement)
			NextElement = RightBinaryCycle(NextElement, Sites)
		Irreps.append(CurrentIrrep)
	return Irreps
def IrreducibleCyclicBasisR(Sites):
	SpinBasis = [N for N in range(2**Sites)][::-1]
	Irreps = []
	while len(SpinBasis) != 0:
		CurrentIrrep = []
		IrrepLargestElement = SpinBasis.pop(0)
		CurrentIrrep.append(IrrepLargestElement)
		NextElement = LeftBinaryCycle(IrrepLargestElement, Sites)
		while NextElement != IrrepLargestElement:
			SpinBasis.remove(NextElement)
			CurrentIrrep.append(NextElement)
			NextElement = LeftBinaryCycle(NextElement, Sites)
		Irreps.append(CurrentIrrep)
	return Irreps
def VectorEigenspacesR(Sites):
	Irreps = IrreducibleCyclicBasisR(Sites)
	SpinBasis = [N for N in range(2**Sites)][::-1]
	EigenSpaces = [[] for i in range(Sites)]
	for Irrep in Irreps:
		for vector_index in range(len(Irrep)):
			Vector = np.zeros((2**Sites), complex)
			for irrep_index in range(len(Irrep)):
				spin_index = 2**Sites - 1 - Irrep[irrep_index]
				Vector[spin_index] += np.round(np.exp(complex(2*np.pi*irrep_index*vector_index*1j)/complex(len(Irrep)))/complex(np.sqrt(len(Irrep))),8)
			EigenSpaces[int(vector_index*Sites/len(Irrep))].append(Vector)
	return EigenSpaces
def DensityEigenspaces(Sites):
	EigenSpaces = VectorEigenspacesR(Sites)
	AdjointEvals = [[i-j for j in range(Sites)] for i in range(Sites)]
	AdjointEigenDecomps = []
	for BlockIndex in range(-Sites+1, Sites):
		AdjointEigenDecomp = []
		for row_index in range(Sites):
			for column_index in range(Sites):
				if AdjointEvals[row_index][column_index] == BlockIndex:
					AdjointEigenDecomp.append((row_index, column_index))
		AdjointEigenDecomps.append(AdjointEigenDecomp)
	DensityEspaces = []
	for AdjointEigenDecomp in AdjointEigenDecomps:
		Espace = []
		for TensorRep1 in AdjointEigenDecomp:
			for State1 in EigenSpaces[TensorRep1[0]]:
				for ConjugateState1 in EigenSpaces[TensorRep1[1]]:
					Espace.append(np.outer(State1, np.conjugate(ConjugateState1)))
		DensityEspaces.append(Espace)
	return DensityEspaces
def VectorEigenspaceDims(Sites):
	Irreps = IrreducibleCyclicBasisR(Sites)
	SpinBasis = [N for N in range(2**Sites)][::-1]
	EigenSpaces = [[] for i in range(Sites)]
	EspaceDims = []
	for Irrep in Irreps:
		for vector_index in range(len(Irrep)):
			Vector = np.zeros((2**Sites), complex)
			for irrep_index in range(len(Irrep)):
				spin_index = 2**Sites - 1 - Irrep[irrep_index]
				Vector[spin_index] += np.round(np.exp(complex(2*np.pi*irrep_index*vector_index*1j)/complex(len(Irrep)))/complex(np.sqrt(len(Irrep))),8)
			EigenSpaces[int(vector_index*Sites/len(Irrep))].append(Vector)
	for Space in EigenSpaces:
		EspaceDims.append(len(Space))
	return EspaceDims
def MatrixEigenspaceDims(EspaceDims, Sites):
	AdjointEvals = [[j-i for j in range(Sites)] for i in range(Sites)]
	MatrixEspaceDims = []
	for BlockIndex in range(Sites):
		Dim = 0
		for row_index in range(Sites):
			for column_index in range(Sites):
				if AdjointEvals[row_index][column_index] == BlockIndex or AdjointEvals[row_index][column_index] == BlockIndex - Sites:
					Dim += EspaceDims[row_index]*EspaceDims[column_index]
		MatrixEspaceDims.append(Dim)
	return MatrixEspaceDims
def EigenVectorBlocktoMatrixEmbedding(EigenvectorBlock, Block, Sites):
	EspaceDims = VectorEigenspaceDims(Sites)
	MatrixEspaceDims = MatrixEigenspaceDims(EspaceDims, Sites)
	FullState = np.array([])
	for BlockIndex in range(Sites):
		if BlockIndex == Block:
			FullState = np.concatenate((FullState, EigenvectorBlock))
		else:
			FullState = np.concatenate((FullState, np.zeros(MatrixEspaceDims[BlockIndex])))
	MatrixEmbedding = CycBVCycBMMap(FullState, Sites)
	return MatrixEmbedding
def EigenVectorBlocktoMatrixEmbeddingMultiple(LeftEvecs, RightEvecs, BlockIndicies, Sites):
	RightEmats = [EigenVectorBlocktoMatrixEmbedding(RightEvecs[index], BlockIndicies[index], Sites) for index in range(len(LeftEvecs))]
	LeftEmats = [EigenVectorBlocktoMatrixEmbedding(LeftEvecs[index], BlockIndicies[index], Sites) for index in range(len(LeftEvecs))]
	return LeftEmats, RightEmats
def EigenVectorBlocktoMatrixEmbedding2DParamSpace(LeftEvecs, RightEvecs, BlockIndicies, Sites):
	Parameter1DataPoints = len(LeftEvecs)
	Parameter2DataPoints = len(LeftEvecs[0])
	RightEmats = [[[] for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)]
	LeftEmats = [[[] for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)]
	for i in range(Parameter1DataPoints):
		for j in range(Parameter2DataPoints):
			RightEmats[i][j] = [EigenVectorBlocktoMatrixEmbedding(RightEvecs[i][j][index], BlockIndicies[i][j][index], Sites) for index in range(len(LeftEvecs[i][j]))]
			LeftEmats[i][j] = [EigenVectorBlocktoMatrixEmbedding(LeftEvecs[i][j][index], BlockIndicies[i][j][index], Sites) for index in range(len(LeftEvecs[i][j]))]
	return LeftEmats, RightEmats
"""
General linear algebra methods.
"""
def HSProduct(Matrix1, Matrix2):
	return np.trace(np.dot(np.conjugate(Matrix1).T,Matrix2))
def AdjointAction(Operator, Density):
	return np.dot(Operator,np.dot(Density,np.conjugate(Operator).T))
def SuperOperatorMatrixMapper(Operator, Type):
	SuperOpMatrix = []
	BasisMatrix = np.zeros((len(Operator), len(Operator)))
	if Type == "O":
		for i in range(len(Operator)):
			for j in range(len(Operator)):
				MatrixColumn = []
				BasisMatrix[i][j] += 1
				Coefficients = np.dot(Operator, BasisMatrix)
				MatrixColumn = Coefficients.flatten()
				SuperOpMatrix.append(MatrixColumn)
				BasisMatrix[i][j] = 0
	elif Type == "M":
		for i in range(len(Operator)):
			for j in range(len(Operator)):
				MatrixColumn = []
				BasisMatrix[i][j] += 1
				Coefficients = np.dot(Operator, np.dot(BasisMatrix, np.conjugate(Operator).T))
				MatrixColumn = Coefficients.flatten()
				SuperOpMatrix.append(MatrixColumn)
				BasisMatrix[i][j] = 0
	else:
		print("Please input valid types of superoperator: O for a standard operator multiplication and M for a measurement superoperator.")
		return
	return np.array(SuperOpMatrix).T
def MeasurementSuperOperatorAction(Projections, Measurements, Density):
	MeasuredDensity = Measurements[0]*np.dot(Projections[0],np.dot(Density, Projections[0]))
	for i in range(1,len(Projections)):
		MeasuredDensity += Measurements[i]*np.dot(Projections[i],np.dot(Density, Projections[i]))
	return MeasuredDensity
"""
General exact methods for block Lindblad operators.
"""
def MasterOperatorBlockR(Hamiltonian, LindbladOperators, LindbladCouplings, BlockIndex, Sites):
	EigenSpaces = VectorEigenspacesR(Sites)
	AdjointEvals = [[j-i for j in range(Sites)] for i in range(Sites)]
	AdjointEigenDecomp = []
	for row_index in range(Sites):
		for column_index in range(Sites):
			if AdjointEvals[row_index][column_index] == BlockIndex or AdjointEvals[row_index][column_index] == BlockIndex - Sites:
				AdjointEigenDecomp.append((row_index, column_index))
	MasterOperator = []
	for TensorRep1 in AdjointEigenDecomp:
		for State1 in EigenSpaces[TensorRep1[0]]:
			for ConjugateState1 in EigenSpaces[TensorRep1[1]]:
				MasterOperatorColumn = []
				TensorRepBasisElement1 = np.outer(State1, np.conjugate(ConjugateState1))
				MasterOpAction = LinbladEquation(TensorRepBasisElement1, Hamiltonian, LindbladOperators, LindbladCouplings)
				for TensorRep2 in AdjointEigenDecomp:
					for State2 in EigenSpaces[TensorRep2[0]]:
						for ConjugateState2 in EigenSpaces[TensorRep2[1]]:
							TensorRepBasisElement2 = np.outer(State2, np.conjugate(ConjugateState2))
							Coefficient = HSProduct(TensorRepBasisElement2, MasterOpAction)
							MasterOperatorColumn.append(Coefficient)
				MasterOperator.append(MasterOperatorColumn)
	return np.array(MasterOperator).T
def MasterOperatorBlockRS(Hamiltonian, LindbladOperators, LindbladCouplings, BlockIndex, Sites, ConjugateField):
	EigenSpaces = VectorEigenspacesR(Sites)
	AdjointEvals = [[j-i for j in range(Sites)] for i in range(Sites)]
	AdjointEigenDecomp = []
	for row_index in range(Sites):
		for column_index in range(Sites):
			if AdjointEvals[row_index][column_index] == BlockIndex or AdjointEvals[row_index][column_index] == BlockIndex - Sites:
				AdjointEigenDecomp.append((row_index, column_index))
	MasterOperator = []
	for TensorRep1 in AdjointEigenDecomp:
		for State1 in EigenSpaces[TensorRep1[0]]:
			for ConjugateState1 in EigenSpaces[TensorRep1[1]]:
				MasterOperatorColumn = []
				TensorRepBasisElement1 = np.outer(State1, np.conjugate(ConjugateState1))
				MasterOpAction = LinbladEquationS(TensorRepBasisElement1, Hamiltonian, LindbladOperators, LindbladCouplings, ConjugateField)
				for TensorRep2 in AdjointEigenDecomp:
					for State2 in EigenSpaces[TensorRep2[0]]:
						for ConjugateState2 in EigenSpaces[TensorRep2[1]]:
							TensorRepBasisElement2 = np.outer(State2, np.conjugate(ConjugateState2))
							Coefficient = HSProduct(TensorRepBasisElement2, MasterOpAction)
							MasterOperatorColumn.append(Coefficient)
				MasterOperator.append(MasterOperatorColumn)
	return np.array(MasterOperator).T
def StdBMCycBVMap(Matrix, Sites):
	EigenSpaces = VectorEigenspacesR(Sites)
	AdjointEvals = [[j-i for j in range(Sites)] for i in range(Sites)]
	AdjointEigenDecomps = []
	for BlockIndex in range(Sites):
		AdjointEigenDecomp = []
		for row_index in range(Sites):
			for column_index in range(Sites):
				if AdjointEvals[row_index][column_index] == BlockIndex or AdjointEvals[row_index][column_index] == BlockIndex - Sites:
					AdjointEigenDecomp.append((row_index, column_index))
		AdjointEigenDecomps.append(AdjointEigenDecomp)
	MatrixVectorBlocks = []
	for AdjointEigenDecomp in AdjointEigenDecomps:
		MatrixVectorBlock = []
		for TensorRep in AdjointEigenDecomp:
			for State in EigenSpaces[TensorRep[0]]:
				for ConjugateState in EigenSpaces[TensorRep[1]]:
					TensorRepBasisElement = np.outer(State, np.conjugate(ConjugateState))
					MatrixVectorBlock.append(HSProduct(TensorRepBasisElement, Matrix))
		MatrixVectorBlocks.append(MatrixVectorBlock)
	return MatrixVectorBlocks
def CycBVCycBBMap(DensityVector, Sites):
	DensityBlocks = []
	BlockLabels = []
	VecEspaces = VectorEigenspacesR(Sites)
	EspaceDims = [len(Espace) for Espace in VecEspaces]
	AdjointEvals = [[j-i for j in range(Sites)] for i in range(Sites)]
	LastIndex = 0
	for BlockIndex in range(Sites):
		AdjointEigenDecomp = []
		for row_index in range(Sites):
			for column_index in range(Sites):
				if AdjointEvals[row_index][column_index] == BlockIndex or AdjointEvals[row_index][column_index] == BlockIndex - Sites:
					AdjointEigenDecomp.append((row_index, column_index))
					BlockLabels.append((row_index, column_index))
		for EigenPair in AdjointEigenDecomp:
			FirstIndex = LastIndex
			LastIndex += EspaceDims[EigenPair[0]]*EspaceDims[EigenPair[1]]
			CurrentBlock = DensityVector[FirstIndex:LastIndex]
			DensityBlocks.append(np.resize(CurrentBlock,(EspaceDims[EigenPair[0]],EspaceDims[EigenPair[1]])))
	return DensityBlocks, BlockLabels
def CycBBCycBMMap(DensityBlocks, BlockLabels, Sites):
	DensityMatrix = DensityBlocks[0]
	for BlockIndex2 in range(1, Sites):
		DensityMatrix = np.concatenate((DensityMatrix, DensityBlocks[BlockLabels.index((0,BlockIndex2))]), axis = 1)
	for BlockIndex1 in range(1, Sites):
		Row = DensityBlocks[BlockLabels.index((BlockIndex1,0))]
		for BlockIndex2 in range(1, Sites):
			Row = np.concatenate((Row, DensityBlocks[BlockLabels.index((BlockIndex1,BlockIndex2))]), axis = 1)
		DensityMatrix = np.concatenate((DensityMatrix, Row))
	return DensityMatrix
def CycBVCycBMMap(DensityVector, Sites):
	DensityBlocks = []
	BlockLabels = []
	VecEspaces = VectorEigenspacesR(Sites)
	EspaceDims = [len(Espace) for Espace in VecEspaces]
	AdjointEvals = [[j-i for j in range(Sites)] for i in range(Sites)]
	LastIndex = 0
	for BlockIndex in range(Sites):
		AdjointEigenDecomp = []
		for row_index in range(Sites):
			for column_index in range(Sites):
				if AdjointEvals[row_index][column_index] == BlockIndex or AdjointEvals[row_index][column_index] == BlockIndex - Sites:
					AdjointEigenDecomp.append((row_index, column_index))
					BlockLabels.append((row_index, column_index))
		for EigenPair in AdjointEigenDecomp:
			FirstIndex = LastIndex
			LastIndex += EspaceDims[EigenPair[0]]*EspaceDims[EigenPair[1]]
			CurrentBlock = DensityVector[FirstIndex:LastIndex]
			DensityBlocks.append(np.resize(CurrentBlock,(EspaceDims[EigenPair[0]],EspaceDims[EigenPair[1]])))
	DensityMatrix = DensityBlocks[0]
	for BlockIndex2 in range(1, Sites):
		DensityMatrix = np.concatenate((DensityMatrix, DensityBlocks[BlockLabels.index((0,BlockIndex2))]), axis = 1)
	for BlockIndex1 in range(1, Sites):
		Row = DensityBlocks[BlockLabels.index((BlockIndex1,0))]
		for BlockIndex2 in range(1, Sites):
			Row = np.concatenate((Row, DensityBlocks[BlockLabels.index((BlockIndex1,BlockIndex2))]), axis = 1)
		DensityMatrix = np.concatenate((DensityMatrix, Row))
	return DensityMatrix
def DensityBlockEvolution(Density, Hamiltonian, LindbladOperators, LindbladCouplings, Sites, Time, TimeStep):
	BlockProjection = StdBMCycBVMap(Density)
	TimeAxis = [i*TimeStep for i in range(int(float(Time)/float(TimeStep)) + 1)]
	TimeDepDensityVector = np.array([[] for i in range(len(TimeAxis))])
	for BlockIndex in range(Sites):
		TimeDepDensityBlock = [BlockProjection[BlockIndex]]
		MastOpBlock = MasterOperatorBlockR(Hamiltonian, LindbladOperators, LindbladCouplings, BlockIndex, Sites)
		StepBlockEvolution = linalg.expm(TimeStep*MastOpBlock)
		for i in range(1, len(TimeAxis)):
			EvolvedDensityBlock = np.dot(StepBlockEvolution, TimeDepDensityBlock[i-1])
			TimeDepDensityBlock.append(EvolvedDensityBlock)
		for i in range(len(TimeDepDensityBlock)):
			TimeDepDensityVector[i] = np.append(TimeDepDensityVector[i], TimeDepDensityBlock[i])
	return TimeDepDensityVector
def DynamicBlockExpectation(InitialDensity, Observable, Hamiltonian, LindbladOperators, LindbladCouplings, Sites, Time, TimeStep, ConvertInitial = True):
	if ConvertInitial == True:
		BlockProjection = StdBMCycBVMap(InitialDensity, Sites)
	else:
		BlockProjection = InitialDensity
	ObservableNewBasisVector = StdBMCycBVMap(Observable, Sites)
	ObservableVectorFull = np.array(ObservableNewBasisVector[0])
	for index in range(1, len(ObservableNewBasisVector)):
		ObservableVectorFull = np.concatenate((ObservableVectorFull, ObservableNewBasisVector[index]))
	ObservableNewBasisMatrix = CycBVCycBMMap(ObservableVectorFull, Sites)
	TimeAxis = [i*TimeStep for i in range(int(float(Time)/float(TimeStep)) + 1)]
	BlockEvolvers = []
	for BlockIndex in range(Sites):
		MastOpBlock = MasterOperatorBlockR(Hamiltonian, LindbladOperators, LindbladCouplings, BlockIndex, Sites)
		StepBlockEvolution = linalg.expm(TimeStep*MastOpBlock)
		BlockEvolvers.append(StepBlockEvolution)
		del MastOpBlock
	TimeDepExpectation = []
	CurrentState = BlockProjection
	CurrentFullState = []
	for Block in CurrentState:
		CurrentFullState = np.append(CurrentFullState, Block)
	CurrentMatrix = CycBVCycBMMap(np.array(CurrentFullState), Sites)
	TimeDepExpectation.append(np.trace(np.dot(ObservableNewBasisMatrix, CurrentMatrix)))
	for i in range(1, len(TimeAxis)):
		NextState = []
		CurrentFullState = np.array([])
		for BlockIndex in range(Sites):
			NextState.append(np.dot(BlockEvolvers[BlockIndex], CurrentState[BlockIndex]))
			CurrentFullState = np.append(CurrentFullState, NextState[BlockIndex])
		CurrentState = NextState
		TimeDepExpectation.append(np.trace(np.dot(ObservableNewBasisMatrix, CycBVCycBMMap(np.array(CurrentFullState), Sites))))
	FinalState = CurrentState
	return TimeDepExpectation, TimeAxis, FinalState
def DynamicBlockExpectations(InitialDensity, Observables, Hamiltonian, LindbladOperators, LindbladCouplings, Sites, Time, TimeStep):
	BlockProjection = StdBMCycBVMap(InitialDensity, Sites)
	ObservablesNewBasisVector = [StdBMCycBVMap(Observable, Sites) for Observable in Observables]
	ObservablesVectorFull = [np.array(ObservableNewBasisVector[0]) for ObservableNewBasisVector in ObservablesNewBasisVector]
	for index in range(1, len(ObservableNewBasisVector)):
		ObservablesVectorFull = [np.concatenate((ObservablesVectorFull[i], ObservablesNewBasisVector[i][index])) for i in range(len(ObservablesVectorFull))]
	ObservablesNewBasisMatrix = [CycBVCycBMMap(ObservableVectorFull, Sites) for ObservableVectorFull in ObservablesVectorFull]
	TimeAxis = [i*TimeStep for i in range(int(float(Time)/float(TimeStep)) + 1)]
	BlockEvolvers = []
	for BlockIndex in range(Sites):
		MastOpBlock = MasterOperatorBlockR(Hamiltonian, LindbladOperators, LindbladCouplings, BlockIndex, Sites)
		StepBlockEvolution = linalg.expm(TimeStep*MastOpBlock)
		BlockEvolvers.append(StepBlockEvolution)
		del MastOpBlock
	TimeDepExpectations = [np.array([]) for i in range(len(ObservablesNewBasisMatrix))]
	CurrentState = BlockProjection
	CurrentFullState = []
	for Block in CurrentState:
		CurrentFullState = np.append(CurrentFullState, Block)
	CurrentMatrix = CycBVCycBMMap(np.array(CurrentFullState), Sites)
	TimeDepExpectations = [np.append(TimeDepExpectations[i], np.trace(np.dot(ObservablesNewBasisMatrix[i], CurrentMatrix))) for i in range(len(ObservablesNewBasisMatrix))]
	for i in range(1, len(TimeAxis)):
		NextState = []
		CurrentFullState = np.array([])
		for BlockIndex in range(Sites):
			NextState.append(np.dot(BlockEvolvers[BlockIndex], CurrentState[BlockIndex]))
			CurrentFullState = np.append(CurrentFullState, NextState[BlockIndex])
		CurrentState = NextState
		CurrentMatrix = CycBVCycBMMap(np.array(CurrentFullState), Sites)
		TimeDepExpectations = [np.append(TimeDepExpectations[i], np.trace(np.dot(ObservablesNewBasisMatrix[i], CurrentMatrix))) for i in range(len(ObservablesNewBasisMatrix))] 
	return TimeDepExpectations, TimeAxis
def DynamicInvariantBlockExpectationMultipleScale(InitialDensities, Observables, Hamiltonian, LindbladOperators, LindbladCouplings, Sites, TimeSteps, StepNumbers, ConvertInitial = True):
	NextStates = []
	for Density in InitialDensities:
		if ConvertInitial == True:
			NextStates.append(StdBMCycBVMap(Density, Sites)[0])
		else:
			NextStates.append(Density)
	NextStates = np.array(NextStates).T
	ObservableVectors = []
	for Observable in Observables:
		ObservableVectors.append(StdBMCycBVMap(Observable, Sites)[0])
	ObservableVectors = np.array(ObservableVectors)
	StepEvolvers = []
	MastOpBlock = MasterOperatorBlockR(Hamiltonian, LindbladOperators, LindbladCouplings, 0, Sites)
	for Step in TimeSteps:
		StepBlockEvolution = linalg.expm(Step*MastOpBlock)
		StepEvolvers.append(StepBlockEvolution)
	del MastOpBlock
	TimeAxis = [0 for i in range(int(sum(StepNumbers))+1)]
	TimeDepExpectations = np.dot(ObservableVectors, NextStates)
	i = 0
	k = 0
	for StepNumber in StepNumbers:
		for j in range(StepNumber):
			k += 1
			NextStates = np.dot(StepEvolvers[i],NextStates)
			TimeDepExpectations= np.dstack((TimeDepExpectations,np.dot(ObservableVectors, NextStates)))
			TimeAxis[k] = TimeAxis[k-1] + TimeSteps[i]
		i += 1
	return TimeDepExpectations, TimeAxis
def DynamicInvariantBlockExpectationMultipleScaleExponentAction(InitialDensity, Observable, Hamiltonian, LindbladOperators, LindbladCouplings, Sites, TimeSteps, StepNumbers, ConvertInitial = True):
	if ConvertInitial == True:
		NextState = StdBMCycBVMap(InitialDensity, Sites)[0]
	else:
		NextState = InitialDensity
	ObservableNewBasisVector = StdBMCycBVMap(Observable, Sites)[0]
	MastOpBlock = MasterOperatorBlockR(Hamiltonian, LindbladOperators, LindbladCouplings, 0, Sites)
	TimeAxis = [0 for i in range(int(sum(StepNumbers))+1)]
	TimeDepExpectation = [0 for i in range(int(sum(StepNumbers))+1)]
	TimeDepExpectation[0] = np.dot(ObservableNewBasisVector, NextState)
	i = 0
	k = 0
	for StepNumber in StepNumbers:
		for j in range(StepNumber):
			k += 1
			NextState = spla.expm_multiply(MastOpBlock*TimeSteps[i], NextState)
			TimeDepExpectation[k] = np.dot(ObservableNewBasisVector, NextState)
			TimeAxis[k] = TimeAxis[k-1] + TimeSteps[i]
			if k % 10 == 0:
				print(TimeAxis[k])
		i += 1
	return TimeDepExpectation, TimeAxis
def InvariantBlockTwoPointCorrelation(InitialDensity, Observables, Hamiltonian, LindbladOperators, LindbladCouplings, Sites, Time, TimeStep):
	""" Observable 0 fixed to 0 time, observable 1 inserted at each time step. """
	BlockProjection = StdBMCycBVMap(InitialDensity, Sites)[0]
	ObservableNewBasisVector1 = StdBMCycBVMap(Observables[0], Sites)[0]
	ObservableNewBasisVector2 = StdBMCycBVMap(Observables[1], Sites)[0]
	EspaceDims = [len(Espace) for Espace in VectorEigenspacesR(Sites)]
	ObservableEspaceBlocks1 = []
	ObservableEspaceBlocks2 = []
	CurrentIndex = 0
	for EspaceIndex in range(Sites):
		NextIndex = CurrentIndex + EspaceDims[EspaceIndex]**2
		CurrentBlock1 = ObservableNewBasisVector1[CurrentIndex:NextIndex]
		CurrentBlock2 = ObservableNewBasisVector2[CurrentIndex:NextIndex]
		ObservableEspaceBlocks1.append(np.resize(CurrentBlock1, (EspaceDims[EspaceIndex], EspaceDims[EspaceIndex])))
		ObservableEspaceBlocks2.append(np.resize(CurrentBlock2, (EspaceDims[EspaceIndex], EspaceDims[EspaceIndex])))
		CurrentIndex = NextIndex
	TimeAxis = [i*TimeStep for i in range(int(float(Time)/float(TimeStep)) + 1)]
	MastOpBlock = MasterOperatorBlockR(Hamiltonian, LindbladOperators, LindbladCouplings, 0, Sites)
	BlockEvolver = StepBlockEvolution = linalg.expm(TimeStep*MastOpBlock)
	del MastOpBlock
	BlockProjection = np.dot(la.matrix_power(BlockEvolver, 500000), BlockProjection)
	TimeDepCorrelation = [0 for i in range(len(TimeAxis))]
	ObservableStateProduct = np.array([])
	Observable1SteadyStateExpectation = 0
	Observable2SteadyStateExpectation = 0
	CurrentIndex = 0
	for EspaceIndex in range(Sites):
		NextIndex = CurrentIndex + EspaceDims[EspaceIndex]**2
		CurrentBlockVector = BlockProjection[CurrentIndex:NextIndex]
		CurrentBlock = np.resize(CurrentBlockVector, (EspaceDims[EspaceIndex], EspaceDims[EspaceIndex]))
		Observable1StateProductBlock = np.dot(ObservableEspaceBlocks1[EspaceIndex], CurrentBlock)
		Observable2StateProductBlock = np.dot(ObservableEspaceBlocks2[EspaceIndex], CurrentBlock)
		Observable1SteadyStateExpectation += np.trace(Observable1StateProductBlock)
		Observable2SteadyStateExpectation += np.trace(Observable2StateProductBlock)
		TimeDepCorrelation[0] += np.trace(np.dot(ObservableEspaceBlocks2[EspaceIndex], Observable1StateProductBlock))
		ObservableStateProduct = np.concatenate((ObservableStateProduct, Observable1StateProductBlock.flatten()))
		CurrentIndex = NextIndex
	TimeDepCorrelation[0] += -Observable1SteadyStateExpectation*Observable2SteadyStateExpectation
	CurrentState = ObservableStateProduct
	for i in range(1, len(TimeAxis)):
		NextState = np.dot(BlockEvolver, CurrentState)
		CurrentState = NextState
		CurrentIndex = 0
		for EspaceIndex in range(Sites):
			NextIndex = CurrentIndex + EspaceDims[EspaceIndex]**2
			CurrentBlockVector = CurrentState[CurrentIndex:NextIndex]
			CurrentBlock = np.resize(CurrentBlockVector, (EspaceDims[EspaceIndex], EspaceDims[EspaceIndex]))
			TimeDepCorrelation[i] += np.trace(np.dot(ObservableEspaceBlocks2[EspaceIndex], CurrentBlock))
			CurrentIndex = NextIndex
		print(i*TimeStep)
		TimeDepCorrelation[i] += -Observable1SteadyStateExpectation*Observable2SteadyStateExpectation
	return TimeDepCorrelation, TimeAxis
def InvariantBlockTwoPointMeasurementCorrelation(InitialDensity, MeasurementProjections, MeasurementValues, Hamiltonian, LindbladOperators, LindbladCouplings, Sites, Time, TimeStep, Normalized = True):
	""" Measurement 0 fixed to 0 time, Measurement 1 inserted at each time step. """
	BlockProjection = StdBMCycBVMap(InitialDensity, Sites)[0]
	Measurement1NewBasisVectors = []
	for i in range(len(MeasurementProjections[0])):
		Measurement1NewBasisVectors.append(StdBMCycBVMap(MeasurementProjections[0][i], Sites)[0])
	Measurement2NewBasisVectors = []
	for i in range(len(MeasurementProjections[1])):
		Measurement2NewBasisVectors.append(StdBMCycBVMap(MeasurementProjections[1][i], Sites)[0])
	EspaceDims = [len(Espace) for Espace in VectorEigenspacesR(Sites)]
	Measurement1EspaceBlocks = [[] for i in range(Sites)]
	Measurement2EspaceBlocks = [[] for i in range(Sites)]
	CurrentIndex = 0
	for EspaceIndex in range(Sites):
		NextIndex = CurrentIndex + EspaceDims[EspaceIndex]**2
		for i in range(len(MeasurementProjections[0])):
			CurrentBlock1 = Measurement1NewBasisVectors[i][CurrentIndex:NextIndex]
			Measurement1EspaceBlocks[EspaceIndex].append(np.resize(CurrentBlock1, (EspaceDims[EspaceIndex], EspaceDims[EspaceIndex])))
		for i in range(len(MeasurementProjections[1])):
			CurrentBlock2 = Measurement2NewBasisVectors[i][CurrentIndex:NextIndex]
			Measurement2EspaceBlocks[EspaceIndex].append(np.resize(CurrentBlock2, (EspaceDims[EspaceIndex], EspaceDims[EspaceIndex])))
		CurrentIndex = NextIndex
	TimeAxis = [i*TimeStep for i in range(int(float(Time)/float(TimeStep)) + 1)]
	MastOpBlock = MasterOperatorBlockR(Hamiltonian, LindbladOperators, LindbladCouplings, 0, Sites)
	BlockEvolver = StepBlockEvolution = linalg.expm(TimeStep*MastOpBlock)
	del MastOpBlock
	BlockProjection = np.dot(la.matrix_power(BlockEvolver, 500000), BlockProjection)
	TimeDepCorrelation = [0 for i in range(len(TimeAxis))]
	MeasuredState = np.array([])
	Measurement1SteadyStateExpectation = 0
	Measurement2SteadyStateExpectation = 0
	CurrentIndex = 0
	for EspaceIndex in range(Sites):
		NextIndex = CurrentIndex + EspaceDims[EspaceIndex]**2
		CurrentBlockVector = BlockProjection[CurrentIndex:NextIndex]
		CurrentBlock = np.resize(CurrentBlockVector, (EspaceDims[EspaceIndex], EspaceDims[EspaceIndex]))
		Measurement1StateBlock = MeasurementSuperOperatorAction(Measurement1EspaceBlocks[EspaceIndex], MeasurementValues[0], CurrentBlock)
		Measurement2StateBlock = MeasurementSuperOperatorAction(Measurement2EspaceBlocks[EspaceIndex], MeasurementValues[1], CurrentBlock)
		Measurement1SteadyStateExpectation += np.trace(Measurement1StateBlock)
		Measurement2SteadyStateExpectation += np.trace(Measurement2StateBlock)
		TimeDepCorrelation[0] += np.trace(MeasurementSuperOperatorAction(Measurement2EspaceBlocks[EspaceIndex], MeasurementValues[1], Measurement1StateBlock))
		MeasuredState = np.concatenate((MeasuredState, Measurement1StateBlock.flatten()))
		CurrentIndex = NextIndex
	TimeDepCorrelation[0] += -Measurement1SteadyStateExpectation*Measurement2SteadyStateExpectation
	CurrentState = MeasuredState
	for i in range(1, len(TimeAxis)):
		NextState = np.dot(BlockEvolver, CurrentState)
		CurrentState = NextState
		CurrentIndex = 0
		for EspaceIndex in range(Sites):
			NextIndex = CurrentIndex + EspaceDims[EspaceIndex]**2
			CurrentBlockVector = CurrentState[CurrentIndex:NextIndex]
			CurrentBlock = np.resize(CurrentBlockVector, (EspaceDims[EspaceIndex], EspaceDims[EspaceIndex]))
			TimeDepCorrelation[i] += np.trace(MeasurementSuperOperatorAction(Measurement1EspaceBlocks[EspaceIndex], MeasurementValues[1], CurrentBlock))
			CurrentIndex = NextIndex
		print(i*TimeStep)
		TimeDepCorrelation[i] += -Measurement1SteadyStateExpectation*Measurement2SteadyStateExpectation
	if Normalized == True:
		TimeDepCorrelation = np.array(TimeDepCorrelation)/TimeDepCorrelation[0]
	return TimeDepCorrelation, TimeAxis
def InvariantBlockTwoPointCorrelationStat(InitialDensity, Observables, Hamiltonian, LindbladOperators, LindbladCouplings, Sites, Time, TimeStep):
	""" Observable 0 fixed to 0 time, observable 1 inserted at each time step. """
	BlockProjection = StdBMCycBVMap(InitialDensity, Sites)[0]
	ObservableNewBasisVector1 = StdBMCycBVMap(Observables[0], Sites)[0]
	ObservableNewBasisVector2 = StdBMCycBVMap(Observables[1], Sites)[0]
	EspaceDims = [len(Espace) for Espace in VectorEigenspacesR(Sites)]
	ObservableEspaceBlocks1 = []
	ObservableEspaceBlocks2 = []
	CurrentIndex = 0
	for EspaceIndex in range(Sites):
		NextIndex = CurrentIndex + EspaceDims[EspaceIndex]
		CurrentBlock1 = ObservableNewBasisVector1[CurrentIndex:NextIndex]
		CurrentBlock2 = ObservableNewBasisVector2[CurrentIndex:NextIndex]
		ObservableEspaceBlocks1.append(np.resize(CurrentBlock1, (EspaceDims[EspaceIndex], EspaceDims[EspaceIndex])))
		ObservableEspaceBlocks2.append(np.resize(CurrentBlock2, (EspaceDims[EspaceIndex], EspaceDims[EspaceIndex])))
		CurrentIndex = NextIndex
	TimeAxis = [i*TimeStep for i in range(int(float(Time)/float(TimeStep)) + 1)]
	MastOpBlock = MasterOperatorBlockR(Hamiltonian, LindbladOperators, LindbladCouplings, 0, Sites)
	BlockEvolver = StepBlockEvolution = linalg.expm(TimeStep*MastOpBlock)
	del MastOpBlock
	BlockProjection = np.dot(la.matrix_power(BlockEvolver, 10000), BlockProjection)
	TimeDepCorrelation = [0 for i in range(len(TimeAxis))]
	ObservableStateProduct = np.array([])
	Observable1SteadyStateExpectation = 0
	Observable2SteadyStateExpectation = 0
	Observable1SteadyStateStandDev = 0
	Observable2SteadyStateStandDev = 0
	CurrentIndex = 0
	for EspaceIndex in range(Sites):
		NextIndex = CurrentIndex + EspaceDims[EspaceIndex]
		CurrentBlockVector = BlockProjection[CurrentIndex:NextIndex]
		CurrentBlock = np.resize(CurrentBlockVector, (EspaceDims[EspaceIndex], EspaceDims[EspaceIndex]))
		Observable1StateProductBlock = np.dot(ObservableEspaceBlocks1[EspaceIndex], CurrentBlock)
		Observable2StateProductBlock = np.dot(ObservableEspaceBlocks2[EspaceIndex], CurrentBlock)
		Observable1SteadyStateExpectation += np.trace(Observable1StateProductBlock)
		Observable2SteadyStateExpectation += np.trace(Observable2StateProductBlock)
		Observable1SteadyStateStandDev += np.trace(np.dot(Observable1StateProductBlock, Observable1StateProductBlock))
		Observable2SteadyStateStandDev += np.trace(np.dot(Observable2StateProductBlock, Observable2StateProductBlock))
		TimeDepCorrelation[0] += np.trace(np.dot(ObservableEspaceBlocks2[EspaceIndex], Observable1StateProductBlock))
		ObservableStateProduct = np.concatenate((ObservableStateProduct, Observable1StateProductBlock.flatten()))
		CurrentIndex = NextIndex
	Observable1SteadyStateStandDev = np.sqrt(Observable1SteadyStateStandDev-Observable1SteadyStateExpectation**2)
	Observable2SteadyStateStandDev = np.sqrt(Observable2SteadyStateStandDev-Observable2SteadyStateExpectation**2)
	TimeDepCorrelation[0] = (TimeDepCorrelation[0]-Observable1SteadyStateExpectation*Observable2SteadyStateExpectation)/float(Observable1SteadyStateStandDev*Observable2SteadyStateStandDev)
	CurrentState = ObservableStateProduct
	for i in range(1, len(TimeAxis)):
		NextState = np.dot(BlockEvolver, CurrentState)
		CurrentState = NextState
		CurrentIndex = 0
		for EspaceIndex in range(Sites):
			NextIndex = CurrentIndex + EspaceDims[EspaceIndex]
			CurrentBlockVector = CurrentState[CurrentIndex:NextIndex]
			CurrentBlock = np.resize(CurrentBlockVector, (EspaceDims[EspaceIndex], EspaceDims[EspaceIndex]))
			TimeDepCorrelation[i] += np.trace(np.dot(ObservableEspaceBlocks2[EspaceIndex], CurrentBlock))
			CurrentIndex = NextIndex
		TimeDepCorrelation[i] = (TimeDepCorrelation[i]-Observable1SteadyStateExpectation*Observable2SteadyStateExpectation)/float(Observable1SteadyStateStandDev*Observable2SteadyStateStandDev)
		print(i*TimeStep)
	return TimeDepCorrelation, TimeAxis
"""
Methods for tensor operator construction.
"""
def CompositeOperator(Operator, Index, Sites):
	Identity = np.eye(len(Operator))
	if Index == 1:
		CompOp = np.array(Operator)
		for index in range(Sites -1):
			CompOp = np.kron(CompOp, Identity)
	else:
		CompOp = Identity
		for index in range(Index - 2):
			CompOp = np.kron(CompOp, Identity)
		CompOp = np.kron(CompOp, np.array(Operator))
		for index in range(Sites - Index):
			CompOp = np.kron(CompOp, Identity)
	return CompOp
def CompositeJumps(Operator, DampingConstant, Sites):
	CompJumps = []
	DampConstRoot = np.sqrt(DampingConstant)
	for index in range(Sites):
		CompJumps.append(DampConstRoot*CompositeOperator(Operator, index + 1, Sites))
	return CompJumps
def ObservableProjections(Operator):
	Projectors = []
	Measurements = []
	Dim = len(Operator)
	Spectrum = linalg.eigh(Operator)
	EvalSpectrum = [round(eigval, 5) for eigval in Spectrum[0]]
	EvecSpectrum = []
	for i in range(Dim):
		EvecSpectrum.append(Spectrum[1].T[i])
	zipped = zip(list(EvalSpectrum), list(EvecSpectrum))
	zipped = sorted(zipped, key=lambda double: np.real(double[0]))
	SortedEvalSpectrum, SortedEvecSpectrum = zip(*zipped)
	currentProjector = np.zeros((Dim,Dim))
	currentMeasurement = SortedEvalSpectrum[0]
	for index in range(len(SortedEvalSpectrum)):
		if currentMeasurement != SortedEvalSpectrum[index]:
			Projectors.append(currentProjector)
			Measurements.append(currentMeasurement)
			currentProjector = np.zeros((Dim,Dim))
			currentMeasurement = SortedEvalSpectrum[index]
		currentProjector += np.outer(SortedEvecSpectrum[index], np.conjugate(SortedEvecSpectrum[index]))
	else:
		Projectors.append(currentProjector)
		Measurements.append(currentMeasurement)
	return Projectors, Measurements
"""
Effective evolution methods.
"""
def ExtremeMetastableStates2D(LeftEvec, RightEvec, StationaryState):
	LeftEvecEvals = sorted(la.eigvals(LeftEvec))
	MaxEval = LeftEvecEvals[-1]
	MinEval = LeftEvecEvals[0]
	eMS1 = StationaryState + MaxEval*RightEvec
	eMS2 = StationaryState + MinEval*RightEvec
	return eMS1, eMS2
def MetastableManifoldProjectors2D(LeftEvec):
	dim = len(LeftEvec)
	Identity = np.eye(dim)
	LeftEvecEvals = sorted(la.eigvals(LeftEvec))
	MaxEval = LeftEvecEvals[-1]
	MinEval = LeftEvecEvals[0]
	EvalDifference = MaxEval - MinEval
	Projection1 = (float(1)/complex(EvalDifference))*(LeftEvec-MinEval*Identity)
	Projection2 = (float(1)/complex(EvalDifference))*(-LeftEvec+MaxEval*Identity)
	return Projection1, Projection2
def MetastableManifoldProjection(InitialDensity, LeftEvec, RightEvec, StationaryState, ReturnProbs = False):
	eMS1, eMS2 = ExtremeMetastableStates2D(LeftEvec, RightEvec, StationaryState)
	Projection1, Projection2 = MetastableManifoldProjectors2D(LeftEvec)
	prob1 = np.trace(np.dot(Projection1, InitialDensity))
	prob2 = np.trace(np.dot(Projection2, InitialDensity))
	MetastableDensityProjection = prob1*eMS1 + prob2*eMS2
	if ReturnProbs == True:
		return MetastableDensityProjection, prob1, prob2
	return MetastableDensityProjection
def EffectiveEvolutionExpectation(InitialDensity, Observable, LeftEvec, RightEvec, StationaryState, SpectralGap, TimeStep, Time, TimeStart = 0):
	eMS1, eMS2 = ExtremeMetastableStates2D(LeftEvec, RightEvec, StationaryState)
	Projection1, Projection2 = MetastableManifoldProjectors2D(LeftEvec)
	CurrentProbs = []
	CurrentProbs.append(np.trace(np.dot(Projection1, InitialDensity)))
	CurrentProbs.append(np.trace(np.dot(Projection2, InitialDensity)))
	CurrentProbs = np.array(CurrentProbs)
	MetastableExps = []
	MetastableExps.append(np.trace(np.dot(Observable, eMS1)))
	MetastableExps.append(np.trace(np.dot(Observable, eMS2)))
	MetastableExps = np.array(MetastableExps)
	LeftEvecEvals = sorted(la.eigvals(LeftEvec))
	MaxEval = LeftEvecEvals[-1]
	MinEval = LeftEvecEvals[0]
	EvalDifference = MaxEval - MinEval
	StepEvolution = np.array([[MaxEval*np.exp(TimeStep*SpectralGap)-MinEval, MinEval*(np.exp(TimeStep*SpectralGap)-1)], [-MaxEval*(np.exp(TimeStep*SpectralGap)-1), MaxEval-MinEval*np.exp(TimeStep*SpectralGap)]])/float(EvalDifference)
	Expectations = [np.dot(MetastableExps, CurrentProbs)]
	TimeAxis = [TimeStart]
	for Step in range(int(float(Time)/float(TimeStep))):
		TimeAxis.append(TimeStep*(Step+1) + TimeStart)
		CurrentProbs = np.dot(StepEvolution, CurrentProbs)
		Expectations.append(np.dot(MetastableExps, CurrentProbs))
	return Expectations, TimeAxis
def EffectiveClassicalObservable(Observable, Projectors, ExtremeMetastableStates):
	ClassicalObservable = [[0,0],[0,0]]
	for i in range(2):
		for j in range(2):
			ClassicalObservable[i][j] += np.trace(np.dot(Projectors[i],np.dot(Observable, ExtremeMetastableStates[j])))
	return np.array(ClassicalObservable)
def EffectiveSteadyStateCorrelation(Observables, LeftEvec, RightEvec, StationaryState, SpectralGap, TimeStep, Time, TimeStart = 0):
	eMS1, eMS2 = ExtremeMetastableStates2D(LeftEvec, RightEvec, StationaryState)
	Projection1, Projection2 = MetastableManifoldProjectors2D(LeftEvec)
	ClassicalObservable = EffectiveClassicalObservable(Observables[0], [Projection1, Projection2], [eMS1, eMS2])
	CurrentProbs = []
	CurrentProbs.append(np.trace(np.dot(Projection1, StationaryState)))
	CurrentProbs.append(np.trace(np.dot(Projection2, StationaryState)))
	CurrentProbs = np.array(CurrentProbs)
	
	CurrentObservableProbProduct = np.dot(ClassicalObservable, CurrentProbs)
	
	MetastableExps = []
	MetastableExps.append(np.trace(np.dot(Observables[1], eMS1)))
	MetastableExps.append(np.trace(np.dot(Observables[1], eMS2)))
	MetastableExps = np.array(MetastableExps)
	SteadyStateExp1 = np.trace(np.dot(Observables[0], StationaryState))
	SteadyStateExp2 = np.trace(np.dot(Observables[1], StationaryState))

	LeftEvecEvals = sorted(la.eigvals(LeftEvec))
	MaxEval = LeftEvecEvals[-1]
	MinEval = LeftEvecEvals[0]
	EvalDifference = MaxEval - MinEval
	StepEvolution = np.array([[MaxEval*np.exp(TimeStep*SpectralGap)-MinEval, MinEval*(np.exp(TimeStep*SpectralGap)-1)], [-MaxEval*(np.exp(TimeStep*SpectralGap)-1), MaxEval-MinEval*np.exp(TimeStep*SpectralGap)]])/float(EvalDifference)
	
	Correlations = [np.dot(MetastableExps, CurrentObservableProbProduct)-SteadyStateExp1*SteadyStateExp2]
	TimeAxis = [TimeStart]
	for Step in range(int(float(Time)/float(TimeStep))):
		TimeAxis.append(TimeStep*(Step+1) + TimeStart)
		CurrentObservableProbProduct = np.dot(StepEvolution, CurrentObservableProbProduct)
		Correlations.append(np.dot(MetastableExps, CurrentObservableProbProduct)-SteadyStateExp1*SteadyStateExp2)
	return Correlations, TimeAxis
"""
Methods for Ising model construction.
"""
def TransIsingHamiltonian1D(Field, Coupling, Sites, Periodic = False):
	SpinX = np.array([[0,0.5],[0.5,0]])
	Hamiltonian = CompositeOperator(SpinX, 1, Sites)
	for index in range(Sites - 1):
		Hamiltonian += CompositeOperator(SpinX, index + 2, Sites)
	Hamiltonian = Field*Hamiltonian
	SpinZ = np.array([[0.5,0],[0,-0.5]])
	CurrentSite = CompositeOperator(SpinZ, 1, Sites)
	for index in range(Sites - 1):
		NextSite = CompositeOperator(SpinZ, index + 2, Sites)
		Hamiltonian += Coupling*np.dot(CurrentSite, NextSite)
		CurrentSite = NextSite
	if Periodic == True:
		Hamiltonian += Coupling*np.dot(CompositeOperator(SpinZ, Sites, Sites), CompositeOperator(SpinZ, 1, Sites))
	return Hamiltonian
def TransIsingHamiltonian1DField(Sites):
	SpinX = np.array([[0,0.5],[0.5,0]])
	HamiltonianField = CompositeOperator(SpinX, 1, Sites)
	for index in range(Sites - 1):
		HamiltonianField += CompositeOperator(SpinX, index + 2, Sites)
	return HamiltonianField
def TransIsingHamiltonian1DInteraction(Sites, Periodic = False):
	HamiltonianInt = np.zeros((2**Sites, 2**Sites))
	SpinZ = np.array([[0.5,0],[0,-0.5]])
	CurrentSite = CompositeOperator(SpinZ, 1, Sites)
	for index in range(Sites - 1):
		NextSite = CompositeOperator(SpinZ, index + 2, Sites)
		HamiltonianInt += np.dot(CurrentSite, NextSite)
		CurrentSite = NextSite
	if Periodic == True:
		HamiltonianInt += np.dot(CompositeOperator(SpinZ, Sites, Sites), CompositeOperator(SpinZ, 1, Sites))
	return HamiltonianInt
"""
Methods for Z2 Ising model construction.
"""
def Z2TransIsingHamiltonian1DField(Sites):
	SpinZ = np.array([[0.5,0],[0,-0.5]])
	HamiltonianField = CompositeOperator(SpinZ, 1, Sites)
	for index in range(Sites - 1):
		HamiltonianField += CompositeOperator(SpinZ, index + 2, Sites)
	return HamiltonianField
def Z2TransIsingHamiltonian1DInteraction(Sites, Periodic = False):
	HamiltonianInt = np.zeros((2**Sites, 2**Sites))
	SpinX = np.array([[0,0.5],[0.5,0]])
	CurrentSite = CompositeOperator(SpinX, 1, Sites)
	for index in range(Sites - 1):
		NextSite = CompositeOperator(SpinX, index + 2, Sites)
		HamiltonianInt -= np.dot(CurrentSite, NextSite)
		CurrentSite = NextSite
	if Periodic == True:
		HamiltonianInt -= np.dot(CompositeOperator(SpinX, Sites, Sites), CompositeOperator(SpinX, 1, Sites))
	return HamiltonianInt
"""
Methods for kinetically constrained quantum glass model.
"""
def KCQG_OneSiteStationaryState(DecayRate, Field, Temperature):
	SteadyState11 = float(4*(Field**2)+Temperature*(DecayRate+Temperature))
	SteadyState22 = float(4*(Field**2)+DecayRate*(DecayRate+Temperature))
	SteadyState12 = float(-2*Field*(DecayRate-Temperature))*1j
	SteadyState21 = float(2*Field*(DecayRate-Temperature))*1j
	SteadyStateNorm = float(8*(Field**2)+(DecayRate+Temperature)**2)
	return np.array([[SteadyState11,SteadyState12],[SteadyState21,SteadyState22]])/SteadyStateNorm
def KCQG_ExcitedStateProjection(DecayRate, Field, Temperature):
	""" Only works if Field is small relative to DecayRate. (As otherwise there is no "excited" state) """
	SteadyState = KCQG_OneSiteStationaryState(DecayRate, Field, Temperature)
	Probabilities, Eigenstates = linalg.eig(SteadyState)
	Zipped = zip(list(Probabilities), list(Eigenstates))
	Zipped = sorted(Zipped, key = lambda double: double[0])
	SortedProbs, SortedEvecs = zip(*Zipped)
	return np.outer(SortedEvecs[0],np.conjugate(SortedEvecs[0]))
def KCQG_SoftConstraint(DecayRate, Field, Temperature, p):
	Projection = KCQG_ExcitedStateProjection(DecayRate, Field, Temperature)
	return p*Projection+(1-p)*np.eye(2, dtype = complex)
def KCQG_Hamiltonian1D(DecayRate, Field, p, Sites, Temperature = 0):
	SigmaX = np.array([[0.0,1.0],[1.0,0.0]])
	KinConstraint = KCQG_SoftConstraint(DecayRate, Field, Temperature, p)
	Hamiltonian = np.dot(CompositeOperator(SigmaX, 1, Sites),np.dot(CompositeOperator(KinConstraint, 2, Sites),CompositeOperator(KinConstraint, 2, Sites)))
	for index in range(1, Sites-1):
		Hamiltonian += np.dot(CompositeOperator(SigmaX, index+1, Sites),np.dot(CompositeOperator(KinConstraint, index+2, Sites),CompositeOperator(KinConstraint, index+2, Sites)))
	Hamiltonian += np.dot(CompositeOperator(SigmaX, Sites, Sites),np.dot(CompositeOperator(KinConstraint, 1, Sites),CompositeOperator(KinConstraint, 1, Sites)))
	Hamiltonian = Field*Hamiltonian
	return Hamiltonian
def KCQG_JumpOps(DecayRate, Field, p, Sites, Temperature = 0):
	JumpOps = []
	SigmaMinus = np.array([[0.0,0.0],[1.0,0.0]])
	SigmaPlus = np.array([[0.0,1.0],[0.0,0.0]])
	KinConstraint = KCQG_SoftConstraint(DecayRate, Field, Temperature, p)
	for index in range(Sites-1):
		JumpOp1 = np.dot(np.sqrt(DecayRate)*CompositeOperator(SigmaMinus, index+1, Sites),CompositeOperator(KinConstraint, index+2, Sites))
		JumpOp2 = np.dot(np.sqrt(Temperature)*CompositeOperator(SigmaPlus, index+1, Sites),CompositeOperator(KinConstraint, index+2, Sites))
		JumpOps.append(JumpOp1)
		JumpOps.append(JumpOp2)
	JumpOps.append(np.dot(np.sqrt(DecayRate)*CompositeOperator(SigmaMinus, Sites, Sites),CompositeOperator(KinConstraint, 1, Sites)))
	JumpOps.append(np.dot(np.sqrt(Temperature)*CompositeOperator(SigmaPlus, Sites, Sites),CompositeOperator(KinConstraint, 1, Sites)))
	return JumpOps
def Q_Operator(DecayRate, Field):
	SigmaZ = np.array([[1.0,0.0],[0.0,-1.0]])
	SigmaY = np.array([[0.0,-1.0j],[1.0j,0.0]])
	omega = np.sqrt(16*(Field**2)+DecayRate**2)
	Q = np.eye(2)/float(2)+DecayRate*SigmaZ/float(2*omega)-2*Field*SigmaY/float(omega)
	return Q
def KineticConstraint(DecayRate, Field, p):
	Q = Q_Operator(DecayRate, Field)
	KinConstraint = p*Q+(1-p)*np.eye(2)
	return KinConstraint
def KCQG_Hamiltonian1DOld(DecayRate, Field, p, Sites):
	SigmaX = np.array([[0.0,1.0],[1.0,0.0]])
	KinConstraint = KineticConstraint(DecayRate, Field, p)
	Hamiltonian = np.dot(CompositeOperator(SigmaX, 1, Sites),np.dot(CompositeOperator(KinConstraint, 2, Sites),CompositeOperator(KinConstraint, 2, Sites)))
	for index in range(1, Sites-1):
		Hamiltonian += np.dot(CompositeOperator(SigmaX, index+1, Sites),np.dot(CompositeOperator(KinConstraint, index+2, Sites),CompositeOperator(KinConstraint, index+2, Sites)))
	Hamiltonian += np.dot(CompositeOperator(SigmaX, Sites, Sites),np.dot(CompositeOperator(KinConstraint, 1, Sites),CompositeOperator(KinConstraint, 1, Sites)))
	Hamiltonian = Field*Hamiltonian
	return Hamiltonian
def KCQG_Hamiltonian1DUnsquared(DecayRate, Field, p, Sites):
	SigmaX = np.array([[0.0,1.0],[1.0,0.0]])
	KinConstraint = KineticConstraint(DecayRate, Field, p)
	Hamiltonian = np.dot(CompositeOperator(SigmaX, 1, Sites),CompositeOperator(KinConstraint, 2, Sites))
	for index in range(1, Sites-1):
		Hamiltonian += np.dot(CompositeOperator(SigmaX, index+1, Sites),CompositeOperator(KinConstraint, index+2, Sites))
	Hamiltonian += np.dot(CompositeOperator(SigmaX, Sites, Sites),CompositeOperator(KinConstraint, 1, Sites))
	Hamiltonian = Field*Hamiltonian
	return Hamiltonian
def KCQG_JumpOpsOld(DecayRate, Field, p, Sites):
	JumpOps = []
	SigmaMinus = np.array([[0.0,0.0],[1.0,0.0]])
	KinConstraint = KineticConstraint(DecayRate, Field, p)
	for index in range(Sites-1):
		JumpOp = np.dot(np.sqrt(DecayRate)*CompositeOperator(SigmaMinus, index+1, Sites),CompositeOperator(KinConstraint, index+2, Sites))
		JumpOps.append(JumpOp)
	JumpOps.append(np.dot(np.sqrt(DecayRate)*CompositeOperator(SigmaMinus, Sites, Sites),CompositeOperator(KinConstraint, 1, Sites)))
	return JumpOps
"""
Methods for spectral analysis of Ising model.
"""
def IsingMasterOpSpectrumVsRatioRange(DampingConstant, CouplingRatio, FieldRatioRange, FieldRatioStep, Sites, Periodic = False):
	Ratios = [i*FieldRatioStep + FieldRatioRange[0] for i in range(int(float(FieldRatioRange[1]-FieldRatioRange[0])/float(FieldRatioStep)))]
	Evals = [[] for i in range(len(Ratios))]
	SMinus = np.array([[0,0],[1,0]])
	Coupling = CouplingRatio*DampingConstant
	FieldHam = TransIsingHamiltonian1DField(Sites)
	IntHam = TransIsingHamiltonian1DInteraction(Sites, Periodic)
	Jumps = CompositeJumps(SMinus, DampingConstant, Sites)
	for BlockIndex in range(Sites):
		print(BlockIndex)
		FieldMast = MasterOperatorBlockR(FieldHam, [], [], BlockIndex, Sites)
		IntMast = MasterOperatorBlockR(IntHam, [], [], BlockIndex, Sites)
		JumpMast = MasterOperatorBlockR(np.eye(2**Sites), CompositeJumps(SMinus, 1, Sites), np.eye(Sites), BlockIndex, Sites)
		PartialMast = CouplingRatio*DampingConstant*IntMast + DampingConstant*JumpMast
		for Field_Step in range(len(Ratios)):
			MastBlock = Ratios[Field_Step]*DampingConstant*FieldMast + PartialMast
			Evalset = np.array(sorted(linalg.eigvals(MastBlock)))
			Evals[Field_Step] = np.append(Evals[Field_Step], Evalset)
	return Evals, Ratios
def IsingMasterOpBlockSpectrumVsRatioRange(DampingConstant, CouplingRatio, FieldRatioRange, FieldRatioStep, Sites, BlockIndex, Periodic = False):
	Ratios = [i*FieldRatioStep + FieldRatioRange[0] for i in range(int(float(FieldRatioRange[1]-FieldRatioRange[0])/float(FieldRatioStep)))]
	Evals = [[] for i in range(len(Ratios))]
	SMinus = np.array([[0,0],[1,0]])
	Coupling = CouplingRatio*DampingConstant
	FieldHam = TransIsingHamiltonian1DField(Sites)
	IntHam = TransIsingHamiltonian1DInteraction(Sites, Periodic)
	Jumps = CompositeJumps(SMinus, DampingConstant, Sites)
	FieldMast = MasterOperatorBlockR(FieldHam, [], [], BlockIndex, Sites)
	IntMast = MasterOperatorBlockR(IntHam, [], [], BlockIndex, Sites)
	JumpMast = MasterOperatorBlockR(np.eye(2**Sites), CompositeJumps(SMinus, 1, Sites), np.eye(Sites), BlockIndex, Sites)
	PartialMast = CouplingRatio*DampingConstant*IntMast + DampingConstant*JumpMast
	for Field_Step in range(len(Ratios)):
		MastBlock = Ratios[Field_Step]*DampingConstant*FieldMast + PartialMast
		Evalset = np.array(sorted(linalg.eigvals(MastBlock)))
		Evals[Field_Step] = np.append(Evals[Field_Step], Evalset)
	return Evals, Ratios
def IsingMasterOpBlockPartialSpectrumVsRatioRange(DampingConstant, CouplingRatio, FieldRatioRange, FieldRatioStep, Sites, BlockIndex, SpectrumSaved, Periodic = False):
	Ratios = [i*FieldRatioStep + FieldRatioRange[0] for i in range(int(float(FieldRatioRange[1]-FieldRatioRange[0])/float(FieldRatioStep))+1)]
	Evals = []
	LeftEvecs = []
	RightEvecs = []
	SMinus = np.array([[0,0],[1,0]])
	Coupling = CouplingRatio*DampingConstant
	FieldHam = TransIsingHamiltonian1DField(Sites)
	IntHam = TransIsingHamiltonian1DInteraction(Sites, Periodic)
	Jumps = CompositeJumps(SMinus, DampingConstant, Sites)
	FieldMast = MasterOperatorBlockR(FieldHam, [], [], BlockIndex, Sites)
	IntMast = MasterOperatorBlockR(IntHam, [], [], BlockIndex, Sites)
	JumpMast = MasterOperatorBlockR(np.eye(2**Sites), CompositeJumps(SMinus, 1, Sites), np.eye(Sites), BlockIndex, Sites)
	PartialMast = CouplingRatio*DampingConstant*IntMast + DampingConstant*JumpMast
	Dim = len(PartialMast)
	for Field_Step in range(len(Ratios)):
		MastBlock = Ratios[Field_Step]*DampingConstant*FieldMast + PartialMast
		Spectrum = linalg.eig(MastBlock, left = True)
		EvalSpectrum = Spectrum[0]
		LeftEvecSpectrum = []
		for i in range(Dim):
			LeftEvecSpectrum.append(Spectrum[1].T[i])
		RightEvecSpectrum = []
		for i in range(Dim):
			RightEvecSpectrum.append(Spectrum[2].T[i])
		zipped = zip(list(EvalSpectrum), list(LeftEvecSpectrum), list(RightEvecSpectrum))
		zipped = sorted(zipped, key=lambda triple: np.real(triple[0]))
		SortedEvalSpectrum, SortedLeftEvecSpectrum, SortedRightEvecSpectrum = zip(*zipped)
		Evals.append(SortedEvalSpectrum[len(SortedEvalSpectrum)-SpectrumSaved : len(SortedEvalSpectrum)])
		LeftEvecs.append(SortedLeftEvecSpectrum[len(SortedEvalSpectrum)-SpectrumSaved : len(SortedEvalSpectrum)])
		RightEvecs.append(SortedRightEvecSpectrum[len(SortedEvalSpectrum)-SpectrumSaved : len(SortedEvalSpectrum)])
	return Evals, LeftEvecs, RightEvecs, Ratios
def IsingMasterOpBlockPartialSpectrumVsComplexField(DecayRate, Interaction, RealFieldRange, RealFieldStep, ImaginaryFieldRange, ImaginaryFieldStep, Sites, BlockIndex, SpectrumSaved, Periodic = False):
	RealField = [i*RealFieldStep + RealFieldRange[0] for i in range(int(float(RealFieldRange[1]-RealFieldRange[0])/float(RealFieldStep))+1)]
	ImaginaryField = [i*ImaginaryFieldStep + ImaginaryFieldRange[0] for i in range(int(float(ImaginaryFieldRange[1]-ImaginaryFieldRange[0])/float(ImaginaryFieldStep))+1)]
	Evals = []
	LeftEvecs = []
	RightEvecs = []
	SMinus = np.array([[0,0],[1,0]])
	FieldHam = TransIsingHamiltonian1DField(Sites)
	IntHam = TransIsingHamiltonian1DInteraction(Sites, Periodic)
	Jumps = CompositeJumps(SMinus, DecayRate, Sites)
	FieldMast = MasterOperatorBlockR(FieldHam, [], [], BlockIndex, Sites)
	IntMast = MasterOperatorBlockR(IntHam, [], [], BlockIndex, Sites)
	JumpMast = MasterOperatorBlockR(np.eye(2**Sites), CompositeJumps(SMinus, 1, Sites), np.eye(Sites), BlockIndex, Sites)
	PartialMast = Interaction*IntMast + DecayRate*JumpMast
	Dim = len(PartialMast)
	for Real in RealField:
		EvalsRow = []
		LeftEvecsRow = []
		RightEvecsRow = []
		for Imag in ImaginaryField:
			MastBlock = (Real+Imag*1j)*FieldMast + PartialMast
			Spectrum = linalg.eig(MastBlock, left = True)
			EvalSpectrum = Spectrum[0]
			LeftEvecSpectrum = []
			for i in range(Dim):
				LeftEvecSpectrum.append(Spectrum[1].T[i])
			RightEvecSpectrum = []
			for i in range(Dim):
				RightEvecSpectrum.append(Spectrum[2].T[i])
			zipped = zip(list(EvalSpectrum), list(LeftEvecSpectrum), list(RightEvecSpectrum))
			zipped = sorted(zipped, key=lambda triple: np.real(triple[0]))
			SortedEvalSpectrum, SortedLeftEvecSpectrum, SortedRightEvecSpectrum = zip(*zipped)
			EvalsRow.append(SortedEvalSpectrum[len(SortedEvalSpectrum)-SpectrumSaved : len(SortedEvalSpectrum)])
			LeftEvecsRow.append(SortedLeftEvecSpectrum[len(SortedEvalSpectrum)-SpectrumSaved : len(SortedEvalSpectrum)])
			RightEvecsRow.append(SortedRightEvecSpectrum[len(SortedEvalSpectrum)-SpectrumSaved : len(SortedEvalSpectrum)])
		Evals.append(EvalsRow)
		LeftEvecs.append(LeftEvecsRow)
		RightEvecs.append(RightEvecsRow)
	return Evals, LeftEvecs, RightEvecs, RealField, ImaginaryField
def IsingMasterOpBlockEvalVsComplexField(DecayRate, Interaction, RealFieldRange, RealFieldStep, ImaginaryFieldRange, ImaginaryFieldStep, Sites, BlockIndex, SpectrumSaved, Scale = 1, Periodic = False):
	RealField = [i*RealFieldStep + RealFieldRange[0] for i in range(int(float(RealFieldRange[1]-RealFieldRange[0])/float(RealFieldStep))+1)]
	ImaginaryField = [i*ImaginaryFieldStep + ImaginaryFieldRange[0] for i in range(int(float(ImaginaryFieldRange[1]-ImaginaryFieldRange[0])/float(ImaginaryFieldStep))+1)]
	Evals = []
	SMinus = np.array([[0,0],[1,0]])
	FieldHam = TransIsingHamiltonian1DField(Sites)
	IntHam = TransIsingHamiltonian1DInteraction(Sites, Periodic)
	Jumps = CompositeJumps(SMinus, DecayRate, Sites)
	FieldMast = MasterOperatorBlockR(FieldHam, [], [], BlockIndex, Sites)
	FieldMastImaginary = MasterOperatorBlockR(1j*FieldHam, [], [], BlockIndex, Sites)
	IntMast = MasterOperatorBlockR(IntHam, [], [], BlockIndex, Sites)
	JumpMast = MasterOperatorBlockR(np.eye(2**Sites), CompositeJumps(SMinus, 1, Sites), np.eye(Sites), BlockIndex, Sites)
	PartialMast = Interaction*IntMast + DecayRate*JumpMast
	Dim = len(PartialMast)
	for Real in RealField:
		EvalsRow = []
		for Imag in ImaginaryField:
			MastBlock = Real*FieldMast + Imag*FieldMastImaginary + PartialMast
			Spectrum = linalg.eigvals(Scale*MastBlock)
			EvalSpectrum = list(np.real(Spectrum))
			SortedEvalSpectrum = sorted(EvalSpectrum)
			EvalsRow.append(SortedEvalSpectrum[len(SortedEvalSpectrum)-SpectrumSaved : len(SortedEvalSpectrum)])
		Evals.append(EvalsRow)
	return Evals, RealField, ImaginaryField
def IsingMasterOpBlockEvalVsComplexFieldHermRoot(DecayRate, Interaction, RealFieldRange, RealFieldStep, ImaginaryFieldRange, ImaginaryFieldStep, Sites, BlockIndex, SpectrumSaved, Periodic = False):
	RealField = [i*RealFieldStep + RealFieldRange[0] for i in range(int(float(RealFieldRange[1]-RealFieldRange[0])/float(RealFieldStep))+1)]
	ImaginaryField = [i*ImaginaryFieldStep + ImaginaryFieldRange[0] for i in range(int(float(ImaginaryFieldRange[1]-ImaginaryFieldRange[0])/float(ImaginaryFieldStep))+1)]
	Evals = []
	SMinus = np.array([[0,0],[1,0]])
	FieldHam = TransIsingHamiltonian1DField(Sites)
	IntHam = TransIsingHamiltonian1DInteraction(Sites, Periodic)
	Jumps = CompositeJumps(SMinus, DecayRate, Sites)
	FieldMast = MasterOperatorBlockR(FieldHam, [], [], BlockIndex, Sites)
	IntMast = MasterOperatorBlockR(IntHam, [], [], BlockIndex, Sites)
	JumpMast = MasterOperatorBlockR(np.eye(2**Sites), CompositeJumps(SMinus, 1, Sites), np.eye(Sites), BlockIndex, Sites)
	PartialMast = Interaction*IntMast + DecayRate*JumpMast
	Dim = len(PartialMast)
	for Real in RealField:
		EvalsRow = []
		for Imag in ImaginaryField:
			MastBlock = (Real+Imag*1j)*FieldMast + PartialMast
			HermMast = np.dot(np.conjugate(MastBlock).T,MastBlock)
			Spectrum = linalg.eigvalsh(linalg.fractional_matrix_power(HermMast,0.5))
			EvalSpectrum = list(np.real(Spectrum))
			SortedEvalSpectrum = sorted(EvalSpectrum)
			EvalsRow.append(SortedEvalSpectrum[len(SortedEvalSpectrum)-SpectrumSaved : len(SortedEvalSpectrum)])
		Evals.append(EvalsRow)
	return Evals, RealField, ImaginaryField
"""
Methods for spectral analysis of Z2 Ising model.
"""
def Z2IsingMasterOpSpectrumVsRatioRange(DampingConstant, CouplingRatio, FieldRatioRange, FieldRatioStep, Sites, Periodic = False):
	Ratios = [i*FieldRatioStep + FieldRatioRange[0] for i in range(int(float(FieldRatioRange[1]-FieldRatioRange[0])/float(FieldRatioStep))+1)]
	Evals = [[] for i in range(len(Ratios))]
	SMinus = np.array([[0,0],[1,0]])
	Coupling = CouplingRatio*DampingConstant
	FieldHam = Z2TransIsingHamiltonian1DField(Sites)
	IntHam = Z2TransIsingHamiltonian1DInteraction(Sites, Periodic)
	Jumps = CompositeJumps(SMinus, DampingConstant, Sites)
	for BlockIndex in range(Sites):
		print(BlockIndex)
		FieldMast = MasterOperatorBlockR(FieldHam, [], [], BlockIndex, Sites)
		IntMast = MasterOperatorBlockR(IntHam, [], [], BlockIndex, Sites)
		JumpMast = MasterOperatorBlockR(np.eye(2**Sites), CompositeJumps(SMinus, 1, Sites), np.eye(Sites), BlockIndex, Sites)
		PartialMast = CouplingRatio*DampingConstant*IntMast + DampingConstant*JumpMast
		for Field_Step in range(len(Ratios)):
			MastBlock = Ratios[Field_Step]*DampingConstant*FieldMast + PartialMast
			Evalset = np.array(sorted(linalg.eigvals(MastBlock)))
			Evals[Field_Step] = np.append(Evals[Field_Step], Evalset)
	return Evals, Ratios
"""
Methods for spectral analysis of KCQG model.
"""
def GlassMasterOpSpectrumVs_p(DecayRate, Field, pRange, pStep, Sites):
	pVals = [i*pStep + pRange[0] for i in range(int(float(pRange[1]-pRange[0])/float(pStep))+1)]
	Evals = [[] for i in range(len(pVals))]
	for BlockIndex in range(Sites):
		i = 0
		for pVal in pVals:
			Hamiltonian = KCQG_Hamiltonian1D(DecayRate, Field, pVal, Sites)
			JumpOps = KCQG_JumpOps(DecayRate, Field, pVal, Sites)
			MastOpBlock = MasterOperatorBlockR(Hamiltonian, JumpOps, np.eye(Sites*2), BlockIndex, Sites)
			Evalset = np.array(sorted(np.real(linalg.eigvals(MastOpBlock))))
			Evals[i] = np.append(Evals[i], Evalset)
			i += 1
	return Evals, pVals
def GlassMasterOpSpectrumVs_Field(DecayRate, FieldRange, FieldStep, p, Sites):
	FieldVals = [i*FieldStep + FieldRange[0] for i in range(int(float(FieldRange[1]-FieldRange[0])/float(FieldStep))+1)]
	Evals = [[] for i in range(len(FieldVals))]
	for BlockIndex in range(Sites):
		i = 0
		for FieldVal in FieldVals:
			Hamiltonian = KCQG_Hamiltonian1D(DecayRate, FieldVal, p, Sites)
			JumpOps = KCQG_JumpOps(DecayRate, FieldVal, p, Sites)
			MastOpBlock = MasterOperatorBlockR(Hamiltonian, JumpOps, np.eye(Sites*2), BlockIndex, Sites)
			Evalset = np.array(sorted(np.real(linalg.eigvals(MastOpBlock))))
			Evals[i] = np.append(Evals[i], Evalset)
			i += 1
	return Evals, FieldVals
def GlassMasterOpSpectrumVs_FieldPlusTemperature(DecayRate, FieldRange, FieldStep, TemperatureRange, TemperatureStep, p, Sites):
	FieldVals = [i*FieldStep + FieldRange[0] for i in range(int(float(FieldRange[1]-FieldRange[0])/float(FieldStep))+1)]
	TemperatureVals = [i*TemperatureStep + TemperatureRange[0] for i in range(int(float(TemperatureRange[1]-TemperatureRange[0])/float(TemperatureStep))+1)]
	Evals = [[[] for j in range(len(TemperatureVals))] for i in range(len(FieldVals))]
	i = 0
	for FieldVal in FieldVals:
		j = 0
		for TemperatureVal in TemperatureVals:
			Hamiltonian = KCQG_Hamiltonian1D(DecayRate, FieldVal, p, Sites, Temperature = TemperatureVal)
			JumpOps = KCQG_JumpOps(DecayRate, FieldVal, p, Sites, Temperature = TemperatureVal)
			Evalset = np.array([])
			for BlockIndex in range(Sites):
				MastOpBlock = MasterOperatorBlockR(Hamiltonian, JumpOps, np.eye(Sites*2), BlockIndex, Sites)
				BlockEvals = np.array(np.real(linalg.eigvals(MastOpBlock)))
				Evalset = np.append(Evalset,BlockEvals)
			Evals[i][j] = sorted(Evalset)
			j += 1
		i += 1
	return Evals, FieldVals, TemperatureVals
def GlassMasterOpSpectrumVs_FieldPlusP(DecayRate, FieldRange, FieldStep, pRange, pStep, Temperature, Sites):
	FieldVals = [i*FieldStep + FieldRange[0] for i in range(int(float(FieldRange[1]-FieldRange[0])/float(FieldStep))+1)]
	pVals = [i*pStep + pRange[0] for i in range(int(float(pRange[1]-pRange[0])/float(pStep))+1)]
	Evals = [[[] for j in range(len(pVals))] for i in range(len(FieldVals))]
	i = 0
	for FieldVal in FieldVals:
		j = 0
		for pVal in pVals:
			Hamiltonian = KCQG_Hamiltonian1D(DecayRate, FieldVal, pVal, Sites, Temperature = Temperature)
			JumpOps = KCQG_JumpOps(DecayRate, FieldVal, pVal, Sites, Temperature = Temperature)
			Evalset = np.array([])
			for BlockIndex in range(Sites):
				MastOpBlock = MasterOperatorBlockR(Hamiltonian, JumpOps, np.eye(Sites*2), BlockIndex, Sites)
				BlockEvals = np.array(np.real(linalg.eigvals(MastOpBlock)))
				Evalset = np.append(Evalset,BlockEvals)
			Evals[i][j] = sorted(Evalset)
			j += 1
		i += 1
	return Evals, FieldVals, pVals
def GlassMasterOpSpectrumVs_FieldPlusTemperatureEvecs(DecayRate, FieldRange, FieldStep, TemperatureRange, TemperatureStep, p, Sites, SpectrumSaved, ExtendEvals = False, Extension = 0):
	FieldVals = [i*FieldStep + FieldRange[0] for i in range(int(round(float(FieldRange[1]-FieldRange[0])/float(FieldStep)))+1)]
	TemperatureVals = [i*TemperatureStep + TemperatureRange[0] for i in range(int(round(float(TemperatureRange[1]-TemperatureRange[0])/float(TemperatureStep)))+1)]
	Evals = [[[] for j in range(len(TemperatureVals))] for i in range(len(FieldVals))]
	LeftEvecs = [[[] for j in range(len(TemperatureVals))] for i in range(len(FieldVals))]
	RightEvecs = [[[] for j in range(len(TemperatureVals))] for i in range(len(FieldVals))]
	BlockIndicies = [[[] for j in range(len(TemperatureVals))] for i in range(len(FieldVals))]
	if ExtendEvals == True:
		ExtendedEvals = [[[] for j in range(len(TemperatureVals))] for i in range(len(FieldVals))]
	i = 0
	for FieldVal in FieldVals:
		j = 0
		for TemperatureVal in TemperatureVals:
			Hamiltonian = KCQG_Hamiltonian1D(DecayRate, FieldVal, p, Sites, Temperature = TemperatureVal)
			JumpOps = KCQG_JumpOps(DecayRate, FieldVal, p, Sites, Temperature = TemperatureVal)
			Evalset = np.array([])
			LeftEvecSet = []
			RightEvecSet = []
			BlockIndiciesTemp = np.array([])
			for BlockIndex in range(Sites):
				MastOpBlock = MasterOperatorBlockR(Hamiltonian, JumpOps, np.eye(Sites*2), BlockIndex, Sites)
				Spectrum = linalg.eig(MastOpBlock, left = True)
				Evalset = np.append(Evalset,Spectrum[0])
				Dim = len(Spectrum[0])
				for n in range(Dim):
					LeftEvecSet.append(Spectrum[1].T[n])
				for n in range(Dim):
					RightEvecSet.append(Spectrum[2].T[n])
				BlockIndiciesTemp = np.append(BlockIndiciesTemp, [BlockIndex for n in range(Dim)])
			Zipped = zip(list(Evalset),LeftEvecSet,RightEvecSet,list(BlockIndiciesTemp))
			Evalset, LeftEvecSet, RightEvecSet, BlockIndiciesTemp = zip(*sorted(Zipped, key = lambda Quad: np.real(Quad[0])))
			Evals[i][j] = Evalset[2**(2*Sites)-SpectrumSaved : 2**(2*Sites)]
			if ExtendEvals == True:
				ExtendedEvals[i][j] = Evalset[2**(2*Sites)-SpectrumSaved-Extension : 2**(2*Sites)]
			LeftEvecs[i][j] = LeftEvecSet[2**(2*Sites)-SpectrumSaved : 2**(2*Sites)]
			RightEvecs[i][j] = RightEvecSet[2**(2*Sites)-SpectrumSaved : 2**(2*Sites)]
			BlockIndicies[i][j] = BlockIndiciesTemp[2**(2*Sites)-SpectrumSaved : 2**(2*Sites)]
			j += 1
		i += 1
	if ExtendEvals == True:
		return Evals, LeftEvecs, RightEvecs, BlockIndicies, FieldVals, TemperatureVals, ExtendedEvals
	else:
		return Evals, LeftEvecs, RightEvecs, BlockIndicies, FieldVals, TemperatureVals
def GlassMasterOpSpectrumVs_FieldPlusPEvecs(DecayRate, FieldRange, FieldStep, pRange, pStep, Temperature, Sites, SpectrumSaved):
	FieldVals = [i*FieldStep + FieldRange[0] for i in range(int(float(FieldRange[1]-FieldRange[0])/float(FieldStep))+1)]
	pVals = [i*pStep + pRange[0] for i in range(int(float(pRange[1]-pRange[0])/float(pStep))+1)]
	Evals = [[[] for j in range(len(pVals))] for i in range(len(FieldVals))]
	LeftEvecs = [[[] for j in range(len(pVals))] for i in range(len(FieldVals))]
	RightEvecs = [[[] for j in range(len(pVals))] for i in range(len(FieldVals))]
	BlockIndicies = [[[] for j in range(len(pVals))] for i in range(len(FieldVals))]
	i = 0
	for FieldVal in FieldVals:
		j = 0
		for pVal in pVals:
			Hamiltonian = KCQG_Hamiltonian1D(DecayRate, FieldVal, pVal, Sites, Temperature = Temperature)
			JumpOps = KCQG_JumpOps(DecayRate, FieldVal, pVal, Sites, Temperature = Temperature)
			Evalset = np.array([])
			LeftEvecSet = []
			RightEvecSet = []
			BlockIndiciesTemp = np.array([])
			for BlockIndex in range(Sites):
				MastOpBlock = MasterOperatorBlockR(Hamiltonian, JumpOps, np.eye(Sites*2), BlockIndex, Sites)
				Spectrum = linalg.eig(MastOpBlock, left = True)
				Evalset = np.append(Evalset,Spectrum[0])
				Dim = len(Spectrum[0])
				for n in range(Dim):
					LeftEvecSet.append(Spectrum[1].T[n])
				for n in range(Dim):
					RightEvecSet.append(Spectrum[2].T[n])
				BlockIndiciesTemp = np.append(BlockIndiciesTemp, [BlockIndex for n in range(Dim)])
			Zipped = zip(list(Evalset),LeftEvecSet,RightEvecSet,list(BlockIndiciesTemp))
			Evalset, LeftEvecSet, RightEvecSet, BlockIndiciesTemp = zip(*sorted(Zipped, key = lambda Quad: np.real(Quad[0])))
			Evals[i][j] = Evalset[2**(2*Sites)-SpectrumSaved : 2**(2*Sites)]
			LeftEvecs[i][j] = LeftEvecSet[2**(2*Sites)-SpectrumSaved : 2**(2*Sites)]
			RightEvecs[i][j] = RightEvecSet[2**(2*Sites)-SpectrumSaved : 2**(2*Sites)]
			BlockIndicies[i][j] = BlockIndiciesTemp[2**(2*Sites)-SpectrumSaved : 2**(2*Sites)]
			j += 1
		i += 1
	return Evals, LeftEvecs, RightEvecs, BlockIndicies, FieldVals, pVals
def GlassMasterOpSpectrumVs_pEvecs(DecayRate, Field, pRange, pStep, Sites, SpectrumSaved):
	pVals = [i*pStep + pRange[0] for i in range(int(float(pRange[1]-pRange[0])/float(pStep))+1)]
	Evals = [[] for i in range(len(pVals))]
	LeftEvecs = [[] for i in range(len(pVals))]
	RightEvecs = [[] for i in range(len(pVals))]
	BlockIndicies = [[] for i in range(len(pVals))]
	for BlockIndex in range(Sites):
		i = 0
		for pVal in pVals:
			Hamiltonian = KCQG_Hamiltonian1D(DecayRate, Field, pVal, Sites)
			JumpOps = KCQG_JumpOps(DecayRate, Field, pVal, Sites)
			MastOpBlock = MasterOperatorBlockR(Hamiltonian, JumpOps, np.eye(Sites*2), BlockIndex, Sites)
			Spectrum = linalg.eig(MastOpBlock, left = True)
			EvalSpectrum = Spectrum[0]
			Dim = len(EvalSpectrum)
			for j in range(Dim):
				LeftEvecs[i].append(Spectrum[1].T[j])
			for j in range(Dim):
				RightEvecs[i].append(Spectrum[2].T[j])
			Evals[i] = np.append(Evals[i], EvalSpectrum)
			BlockIndicies[i] = np.append(BlockIndicies[i], [BlockIndex for j in range(len(EvalSpectrum))])
			i += 1
	SortedEvals = [[] for i in range(len(pVals))]
	SortedLeftEvecs = [[] for i in range(len(pVals))]
	SortedRightEvecs = [[] for i in range(len(pVals))]
	SortedBlockIndicies = [[] for i in range(len(pVals))]
	for index in range(len(pVals)):
		zipped = zip(list(Evals[index]), list(LeftEvecs[index]), list(RightEvecs[index]), BlockIndicies[index])
		zipped = sorted(zipped, key=lambda triple: np.real(triple[0]))
		EvalSpectrum, LeftEvecSpectrum, RightEvecSpectrum, SortedBlocks = zip(*zipped)
		SortedEvals[i] = EvalSpectrum[len(EvalSpectrum)-SpectrumSaved : len(EvalSpectrum)]
		SortedLeftEvecs[i] = LeftEvecSpectrum[len(EvalSpectrum)-SpectrumSaved : len(EvalSpectrum)]
		SortedRightEvecs[i] = RightEvecSpectrum[len(EvalSpectrum)-SpectrumSaved : len(EvalSpectrum)]
		SortedBlockIndicies[i] = SortedBlocks[len(EvalSpectrum)-SpectrumSaved : len(EvalSpectrum)]
	return SortedEvals, SortedLeftEvecs, SortedRightEvecs, SortedBlockIndicies, pVals
def GlassMasterOpSpectrumVs_S(DecayRate, Field, p, sRange, sStep, Sites):
	sVals = [i*sStep + sRange[0] for i in range(int(float(sRange[1]-sRange[0])/float(sStep))+3)]
	Evals = [0 for i in range(len(sVals))]
	i = 0
	for sVal in sVals:
		Hamiltonian = KCQG_Hamiltonian1D(DecayRate, Field, p, Sites)
		JumpOps = KCQG_JumpOps(DecayRate, Field, p, Sites)
		MastOpBlock = MasterOperatorBlockRS(Hamiltonian, JumpOps, np.eye(Sites*2), 0, Sites, sVal)
		Evalset = np.array(sorted(np.real(linalg.eigvals(MastOpBlock))))
		Evals[i] = Evalset[-1]
		i += 1
	Activity = [0 for i in range(len(Evals)-1)]
	for index in range(len(Activity)):
		Activity[index] = -(Evals[index]-Evals[index+1])/(sVals[index]-sVals[index+1])
	Mandel = [0 for i in range(len(Activity)-1)]
	for index in range(len(Mandel)):
		Mandel[index] = ((Activity[index]-Activity[index+1])/((sVals[index]-sVals[index+1])*Activity[index]))-1
	return Evals[0:len(sVals)-2], sVals[0:len(sVals)-2], Activity[0:len(sVals)-2], Mandel[0:len(sVals)-2]
def GlassMasterOpSpectrumVs_SFull(DecayRate, Field, p, sRange, sStep, Sites):
	sVals = [i*sStep + sRange[0] for i in range(int(float(sRange[1]-sRange[0])/float(sStep))+1)]
	Evals = [0 for i in range(len(sVals))]
	i = 0
	for sVal in sVals:
		Evalset = np.array([])
		for BlockIndex in range(Sites):
			Hamiltonian = KCQG_Hamiltonian1D(DecayRate, Field, p, Sites)
			JumpOps = KCQG_JumpOps(DecayRate, Field, p, Sites)
			MastOpBlock = MasterOperatorBlockRS(Hamiltonian, JumpOps, np.eye(Sites*2), BlockIndex, Sites, sVal)
			Evalset = np.append(Evalset, np.array(np.real(linalg.eigvals(MastOpBlock))))
		Evalset = sorted(Evalset)
		Evals[i] = Evalset[-1]
		i += 1
	Activity = [0 for i in range(len(Evals)-1)]
	for index in range(len(Activity)):
		Activity[index] = -(Evals[index]-Evals[index+1])/(sVals[index]-sVals[index+1])
	return Evals, sVals, Activity
def GlassMasterOpSpectrumVs_SingleSSteadyState(DecayRate, Field, p, sRange, sStep, sIndex, Sites):
	Snumber = int(float(sRange[1]-sRange[0])/float(sStep))+1
	sVals = np.zeros((Snumber,Sites), complex)
	for i in range(int(float(sRange[1]-sRange[0])/float(sStep))+1):
		sVals[i][sIndex] = i*sStep + sRange[0]
	Evals = [0 for i in range(len(sVals))]
	Evecs = []
	i = 0
	for sVal in sVals:
		Hamiltonian = KCQG_Hamiltonian1D(DecayRate, Field, p, Sites)
		JumpOps = KCQG_JumpOps(DecayRate, Field, p, Sites)
		MastOp = MasterOperatorMatrixMultipleS(Hamiltonian, JumpOps, np.eye(Sites*2), sVal)
		Spectrum = linalg.eig(MastOp)
		Evalset = Spectrum[0]
		Evecset = []
		for j in range(len(Spectrum[0])):
			Evecset.append(Spectrum[1].T[j])
		zipped = zip(list(Evalset), list(Evecset))
		zipped = sorted(zipped, key=lambda double: np.real(double[0]))
		Evalset, Evecset = zip(*zipped)
		Evals[i] = Evalset[-1]
		Evecs.append(Evecset[-1])
		i += 1
	return Evals, Evecs
def GlassMasterOpSpectrumVs_TwoSSteadyState(DecayRate, Field, p, sRanges, sSteps, sIndicies, Sites):
	Snumber1 = int(float(sRanges[0][1]-sRanges[0][0])/float(sSteps[0]))+1
	Snumber2 = int(float(sRanges[1][1]-sRanges[1][0])/float(sSteps[1]))+1
	sVals = np.zeros((Snumber1*Snumber2,Sites), complex)
	for i in range(Snumber1):
		for j in range(Snumber2):
			sVals[Snumber1*i+j][sIndicies[0]] = i*sSteps[0] + sRanges[0][0]
			sVals[Snumber1*i+j][sIndicies[1]] = j*sSteps[1] + sRanges[0][0]
	Evals = [0 for i in range(len(sVals))]
	Evecs = []
	i = 0
	for sVal in sVals:
		Hamiltonian = KCQG_Hamiltonian1D(DecayRate, Field, p, Sites)
		JumpOps = KCQG_JumpOps(DecayRate, Field, p, Sites)
		MastOp = MasterOperatorMatrixMultipleS(Hamiltonian, JumpOps, np.eye(Sites*2), sVal)
		Spectrum = linalg.eig(MastOp)
		Evalset = Spectrum[0]
		Evecset = []
		for j in range(len(Spectrum[0])):
			Evecset.append(Spectrum[1].T[j])
		zipped = zip(list(Evalset), list(Evecset))
		zipped = sorted(zipped, key=lambda double: np.real(double[0]))
		Evalset, Evecset = zip(*zipped)
		Evals[i] = Evalset[-1]
		Evecs.append(Evecset[-1])
		i += 1
	return Evals, Evecs
def SaveFieldPlusTempSpectralData(Eigenvalues, LeftEigenvectors, RightEigenvectors, BlockIndicies, FieldValues, TemperatureValues):
	for i in range(len(FieldValues)):
		for j in range(len(TemperatureValues)):
			Field = FieldValues[i]
			Temp = TemperatureValues[j]
			file_name = "%sSites%sF%sT%sP%sD" %(Sites,Field,Temp,p,DataStored)
			np.savetxt(file_name+"Evals.txt",np.array(Evals[i][j]).view(float))
			np.savetxt(file_name+"Left.txt",np.concatenate(LeftEvecs[i][j]).view(float))
			np.savetxt(file_name+"Right.txt",np.concatenate(RightEvecs[i][j]).view(float))
			np.savetxt(file_name+"Blocks.txt",np.array(BlockIndicies[i][j]).view(float))
	return
def EigenvectorUnpacking(LeftEigenvectors, RightEigenvectors, BlockIndicies, Sites, DataRequired):
	EspaceDims = MatrixEigenspaceDims(VectorEigenspaceDims(Sites), Sites)
	LeftEvecs = []
	RightEvecs = []
	Current = 0
	for Index in BlockIndicies:
		Dim = EspaceDims[int(Index)]
		LeftEvecs.append(LeftEigenvectors[Current:Current+Dim])
		RightEvecs.append(RightEigenvectors[Current:Current+Dim])
		Current += Dim
	return LeftEvecs[len(BlockIndicies)-DataRequired:len(BlockIndicies)], RightEvecs[len(BlockIndicies)-DataRequired:len(BlockIndicies)]
"""
Metastable manifold analysis/construction techniques.
"""
def Divisors(Number):
	Divisors = []
	if np.sqrt(Number)%1 == 0:
		for i in range(1,int(np.sqrt(Number))):
			if Number%i == 0:
				Divisors.append(i)
				Divisors.append(Number/i)
		Divisors.append(int(np.sqrt(Number)))
	else:
		for i in range(1,int(np.sqrt(Number))+1):
			if Number%i == 0:
				Divisors.append(i)
				Divisors.append(Number/i)
	return Divisors
def ConstrainedSumCombinations(Numbers, Total, PartialCombination = []):
	Combinations = []
	Sum = sum(PartialCombination)
	if Sum == Total:
		Combinations.append(PartialCombination)
	if Sum >= Total:
		return Combinations
	for i in range(len(Numbers)):
		number = Numbers[i]
		Combinations.extend(ConstrainedSumCombinations(
			Numbers[i+1:], Total, PartialCombination+[number]))
	return Combinations
def UniqueOrderedConstrainedSumCombinations(Numbers, Total, PartialCombination = []):
	PreviousNumber = 0
	Combinations = []
	Sum = sum(PartialCombination)
	if Sum == Total:
		Combinations.append(PartialCombination)
	if Sum >= Total:
		return Combinations
	for i in range(len(Numbers)):
		if PreviousNumber != Numbers[i]:
			Combinations.extend(UniqueOrderedConstrainedSumCombinations(
				Numbers[i+1:], Total, PartialCombination+[Numbers[i]]))
			PreviousNumber = Numbers[i]
	return Combinations
def VectorIrrepGrouping(Vectors, Transformation, PowerForIdentity, Rounding = 8):
	IrrepDimensions = sorted(Divisors(PowerForIdentity))
	Irreps = [[] for i in range(len(IrrepDimensions))]
	IrrepDimensionCounts = [0 for i in range(len(IrrepDimensions))]
	while len(Vectors) > 0:
		Irrep = [Vectors.pop(0)]
		for i in range(len(Vectors)):
			Vector = Vectors.pop(0)
			Magnitude = round(la.norm(Irrep[0]-Vector), Rounding)
			if Magnitude == 0:
				del Vector
			else:
				Vectors.append(Vector)
		TransformedVector = np.dot(Transformation,Irrep[-1])
		for j in range(PowerForIdentity):
			TransformationPresentCheck = False
			if round(la.norm(Irrep[0]-TransformedVector), Rounding) == 0:
				Irreps[IrrepDimensions.index(len(Irrep))].append(Irrep)
				IrrepDimensionCounts[IrrepDimensions.index(len(Irrep))] += 1
				break
			for i in range(len(Vectors)):
				Vector = Vectors.pop(0)
				Magnitude = round(la.norm(TransformedVector-Vector), Rounding)
				if Magnitude == 0:
					if not TransformationPresentCheck:
						Irrep.append(Vector)
						TransformationPresentCheck = True
					else:
						del Vector
				else:
					Vectors.append(Vector)
			if not TransformationPresentCheck:
				break
				#Irrep.append(Vector)
			TransformedVector = np.dot(Transformation,TransformedVector)
	print(IrrepDimensionCounts)
	return Irreps, IrrepDimensionCounts
def VectorIrrepGrouping2(Vectors, Transformation, PowerForIdentity, Rounding = 8):
	IrrepDimensions = sorted(Divisors(PowerForIdentity))
	Irreps = [[] for i in range(len(IrrepDimensions))]
	IrrepDimensionCounts = [0 for i in range(len(IrrepDimensions))]
	while len(Vectors) > 0:
		Irrep = [Vectors.pop(0)]
		for i in range(len(Vectors)):
			Vector = Vectors.pop(0)
			Magnitude = round(la.norm(Irrep[0]-Vector), Rounding)
			if Magnitude == 0:
				del Vector
			else:
				Vectors.append(Vector)
		TransformedVector = np.dot(Transformation,Irrep[-1])
		for j in range(PowerForIdentity):
			TransformationPresentCheck = False
			if round(la.norm(Irrep[0]-TransformedVector), Rounding) == 0:
				Irreps[IrrepDimensions.index(len(Irrep))].append(Irrep)
				IrrepDimensionCounts[IrrepDimensions.index(len(Irrep))] += 1
				break
			for i in range(len(Vectors)):
				Vector = Vectors.pop(0)
				Magnitude = round(la.norm(TransformedVector-Vector), Rounding)
				if Magnitude != 0:
					Vectors.append(Vector)
			Irrep.append(TransformedVector)
			TransformedVector = np.dot(Transformation,TransformedVector)
	print(IrrepDimensionCounts)
	return Irreps, IrrepDimensionCounts
def EigenvalueGapAnalysis2dParamSpace(EigenvalueData, ParameterSpaceDataPointNumbers, RatioForGap):
	GapPositions = []
	GapRatios = []
	for i in range(ParameterSpaceDataPointNumbers[0]):
		for j in range(ParameterSpaceDataPointNumbers[1]):
			for k in range(2,len(EigenvalueData[i][j])):
				if np.around(EigenvalueData[i][j][-k],14) != 0:
					if EigenvalueData[i][j][-(k+1)]/EigenvalueData[i][j][-k] > RatioForGap:
						if k in GapPositions:
							GapRatios[GapPositions.index(k)][i][j] = EigenvalueData[i][j][-(k+1)]/EigenvalueData[i][j][-k]
						else:
							GapPositions.append(k)
							GapRatios.append([[0 for n in range(ParameterSpaceDataPointNumbers[1])] for m in range(ParameterSpaceDataPointNumbers[0])])
							GapRatios[-1][i][j] = EigenvalueData[i][j][-(k+1)]/EigenvalueData[i][j][-k]
	Zipped = zip(list(GapPositions),list(GapRatios))
	Zipped = sorted(Zipped, key = lambda Double: Double[0])
	GapPositions, GapRatios = zip(*Zipped)
	return np.array(GapPositions), np.array(GapRatios)
def RandomPureRealState(Dimension):
	RandomUnnormedState = np.random.randn(Dimension)
	return RandomUnnormedState/float(linalg.norm(RandomUnnormedState))
def RandomPureState(Dimension):
	RandomUnnormedState = np.random.randn(Dimension)
	RandomNormedState = RandomUnnormedState/float(linalg.norm(RandomUnnormedState))
	RandomComplexState = [0 for i in range(Dimension)]
	for i in range(Dimension):
		phase = np.exp(1j*2*np.pi*np.random.random())
		RandomComplexState[i] = RandomNormedState[i]*phase
	return np.array(RandomComplexState)
def IncorrectRandomPureState(Dimension):
	RandomUnnormedState = (np.random.rand(Dimension)-0.5)*2
	return RandomUnnormedState/float(linalg.norm(RandomUnnormedState))
def SimplexVolume(Vertices):
	if len(Vertices) != len(Vertices[0])+1:
		print(Vertices)
	VertexNumber = len(Vertices)
	VerticesMatrix = Vertices[1]-Vertices[0]
	for i in range(2,VertexNumber):
		AffineVertex = Vertices[i]-Vertices[0]
		VerticesMatrix = np.vstack((VerticesMatrix, AffineVertex))
	Volume = linalg.det(VerticesMatrix)
	#/float(scipy.misc.factorial(len(Vertices)-1))
	return abs(Volume)
def MetastableManifoldVertexSelector(PossibleVertices, SimplexVertexNumber):
	Combinations = itertools.combinations(range(len(PossibleVertices)), SimplexVertexNumber)
	LargestVolume = 0
	for combination in Combinations:
		Vertices = [PossibleVertices[combination[i]] for i in range(SimplexVertexNumber)]
		CurrentVolume = SimplexVolume(Vertices)
		if CurrentVolume > LargestVolume:
			LargestSimplex = Vertices
			LargestVolume = CurrentVolume
	return LargestVolume, LargestSimplex
def DegenerateMetastableManifoldVertexSelector(LeftEigenmatrices, RoundingError = 10, Iterations = None):
	HilbertDim = len(LeftEigenmatrices[0])
	ProjectedLeftEigenvectors = []
	for index in range(len(LeftEigenmatrices)-1):
		Eigenvalues, Eigenvectors = linalg.eig(LeftEigenmatrices[index])
		Evectors = []
		for j in range(len(Eigenvalues)):
			Evectors.append(Eigenvectors[:,j])
		zipped = zip(list(Eigenvalues), list(Evectors))
		zipped = sorted(zipped, key=lambda triple: np.real(triple[0]))
		SortedEvals, SortedEvecs = zip(*zipped)
		ProjectedEvecs = []
		for Evec in SortedEvecs:
			Density = np.outer(Evec, np.conjugate(Evec).T)
			ProjectedEvecs.append(InitialStateProjection(Density, LeftEigenmatrices[0:len(LeftEigenmatrices)-1]))
		ProjectedLeftEigenvectors.append(ProjectedEvecs)
	InitialPossibleVertices = []
	for LeftEmatsEvecs in ProjectedLeftEigenvectors:
		InitialPossibleVertices.append(LeftEmatsEvecs[0])
		InitialPossibleVertices.append(LeftEmatsEvecs[-1])
	CurrentVolume, CurrentVertices = MetastableManifoldVertexSelector(InitialPossibleVertices, len(LeftEigenmatrices))
	if Iterations == None or Iterations >= HilbertDim/2:
		for i in range(1,HilbertDim/2):
			CurrentPossibleVertices = CurrentVertices
			AffineTrafos = np.vstack((np.array(CurrentVertices).T,np.ones((len(CurrentVertices)),dtype=complex)))
			InverseTrafos = linalg.inv(AffineTrafos)
			j = 0
			for LeftEmatsEvecs in ProjectedLeftEigenvectors:
				Probabilities1 = np.dot(InverseTrafos,np.append(LeftEmatsEvecs[i],1))
				Probabilities2 = np.dot(InverseTrafos,np.append(LeftEmatsEvecs[-(i+1)],1))
				for prob in Probabilities1:
					if round(prob, RoundingError) > 1 or round(prob, RoundingError) < 0:
						CurrentPossibleVertices.append(LeftEmatsEvecs[i])
						j+=1
						break
				for prob in Probabilities2:
					if round(prob, RoundingError) > 1 or round(prob, RoundingError) < 0:
						CurrentPossibleVertices.append(LeftEmatsEvecs[-(i+1)])
						j+= 1
						break
			if j != 0:
				CurrentVolume, CurrentVertices = MetastableManifoldVertexSelector(CurrentPossibleVertices, len(LeftEigenmatrices))
	else:
		for i in range(1,Iterations):
			CurrentPossibleVertices = CurrentVertices
			AffineTrafos = np.vstack((np.array(CurrentVertices).T,np.ones((len(CurrentVertices)),dtype=complex)))
			InverseTrafos = linalg.inv(AffineTrafos)
			for LeftEmatsEvecs in ProjectedLeftEigenvectors:
				Probabilities1 = np.dot(InverseTrafos,np.append(LeftEmatsEvecs[i],1))
				Probabilities2 = np.dot(InverseTrafos,np.append(LeftEmatsEvecs[-(i+1)],1))
				for prob in Probabilities1:
					if round(prob, RoundingError) > 1 or round(prob, RoundingError) < 0:
						CurrentPossibleVertices.append(LeftEmatsEvecs[i])
						j+=1
						break
				for prob in Probabilities2:
					if round(prob, RoundingError) > 1 or round(prob, RoundingError) < 0:
						CurrentPossibleVertices.append(LeftEmatsEvecs[-(i+1)])
						j+= 1
						break
			if j != 0:
				CurrentVolume, CurrentVertices = MetastableManifoldVertexSelector(CurrentPossibleVertices, len(LeftEigenmatrices))
	return CurrentVertices
def CombinationMetastableManifoldVertexSelector(
	Combination, IrrepCounts, EigenvectorIrreps, IrrepDimensions, 
	IrrepConsidered = 0, PartialVertices = [], CurrentVolume = 0, 
	CurrentVertices = np.array([])):
	if len(Combination) != IrrepConsidered + 1:
		Eigenvectors = EigenvectorIrreps[IrrepConsidered]
		IrrepDim = IrrepDimensions[IrrepConsidered]
		IrrepMultiplicity = Combination[IrrepConsidered]
		Combinations = itertools.combinations(range(IrrepCounts[IrrepConsidered]), 
											  IrrepMultiplicity)
		for Combi in Combinations:
			for I in Combi:
				PartialVertices.extend(Eigenvectors[I])
			CurrentVolume, CurrentVertices = CombinationMetastableManifoldVertexSelector(
				Combination, IrrepCounts, EigenvectorIrreps, IrrepDimensions, 
				IrrepConsidered + 1, PartialVertices, CurrentVolume, CurrentVertices)
			del PartialVertices[int(len(PartialVertices)-IrrepMultiplicity*IrrepDim):]
		return CurrentVolume, CurrentVertices
	else:
		Eigenvectors = EigenvectorIrreps[IrrepConsidered]
		IrrepDim = IrrepDimensions[IrrepConsidered]
		IrrepMultiplicity = Combination[IrrepConsidered]
		Combinations = itertools.combinations(range(IrrepCounts[IrrepConsidered]), 
											  IrrepMultiplicity)
		for Combi in Combinations:
			for I in Combi:
				PartialVertices.extend(Eigenvectors[I])
			NewVolume = SimplexVolume(PartialVertices)
			if NewVolume >= CurrentVolume:
				CurrentVertices = list(PartialVertices)
				CurrentVolume = NewVolume
			del PartialVertices[int(len(PartialVertices)-IrrepMultiplicity*IrrepDim):]
		return CurrentVolume, CurrentVertices
def SymmetryMetastableManifoldVertexSelector(
	LeftEigenmatrices, RightEigenmatrices, StateSpaceSymmetryTransformation, 
	PowerForIdentity, RoundingError = 10, InitialPairsPerMatrix = 2):
	ProjectedLeftEigenvectors = []
	for index in range(len(LeftEigenmatrices)-1):
		Eigenvalues, Eigenvectors = linalg.eig(LeftEigenmatrices[index])
		Evectors = []
		for j in range(len(Eigenvalues)):
			Evectors.append(Eigenvectors[:,j])
		zipped = zip(list(Eigenvalues), list(Evectors))
		zipped = sorted(zipped, key=lambda triple: np.real(triple[0]))
		SortedEvals, SortedEvecs = zip(*zipped)
		ProjectedEvecs = []
		for Evec in SortedEvecs:
			Density = np.outer(Evec, np.conjugate(Evec).T)
			ProjectedEvecs.append(InitialStateProjection(Density, LeftEigenmatrices[0:len(LeftEigenmatrices)-1]))
		ProjectedLeftEigenvectors.append(ProjectedEvecs)
	ProjectedTransformation = []
	for index1 in range(len(LeftEigenmatrices)-1):
		ProjectedTransformationRow = []
		for index2 in range(len(LeftEigenmatrices)-1):
			TransformedRight = np.dot(StateSpaceSymmetryTransformation,
									  np.dot(RightEigenmatrices[index2],
									  		 np.conjugate(StateSpaceSymmetryTransformation).T))
			ProjectedTransformationRow.append(np.trace(np.dot(LeftEigenmatrices[index1],
															  TransformedRight)))
		ProjectedTransformation.append(ProjectedTransformationRow)
	ProjectedTransformation = np.array(ProjectedTransformation)
	InitialPossibleVertices = []
	for index in range(InitialPairsPerMatrix):
		for LeftEmatsEvecs in ProjectedLeftEigenvectors:
			InitialPossibleVertices.append(LeftEmatsEvecs[index])
			InitialPossibleVertices.append(LeftEmatsEvecs[-(1+index)])
	EigenvectorIrreps, IrrepCounts = VectorIrrepGrouping(
		InitialPossibleVertices,ProjectedTransformation,PowerForIdentity, 
		Rounding = RoundingError)
	IrrepDims = sorted(Divisors(PowerForIdentity))

	Numbers = []
	for index in range(len(IrrepDims)):
		Max = int(len(LeftEigenmatrices)/IrrepDims[index])
		for i in range(min([Max, IrrepCounts[index]])):
			Numbers.append(IrrepDims[index])
	ValidCombinations = UniqueOrderedConstrainedSumCombinations(
		Numbers, len(LeftEigenmatrices))
	if len(ValidCombinations) == 0:
		return [np.array([0.01*(i==j) for i in range(len(LeftEigenmatrices)-1)]) 
				for j in range(len(LeftEigenmatrices))]
	CompactCombinations = [[Combi.count(IrrepDim) for IrrepDim in IrrepDims] 
						   for Combi in ValidCombinations]

	CurrentVol = 0
	CurrentVert = []
	for Combination in CompactCombinations:
		CurrentVol, CurrentVert = CombinationMetastableManifoldVertexSelector(
			Combination, IrrepCounts, EigenvectorIrreps, IrrepDims, 
			CurrentVolume = CurrentVol, CurrentVertices = CurrentVert)
	print(CurrentVol)
	return CurrentVert
def SimplexVerticesVs2DParameterSpace(LeftEigenmatrices, RoundingError = 10, Iterations = None):
	Parameter1DataPoints = len(LeftEigenmatrices)
	Parameter2DataPoints = len(LeftEigenmatrices[0])
	SimplexVertices = [[[] for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)]
	for i in range(Parameter1DataPoints):
		for j in range(Parameter2DataPoints):
			SimplexVertices[i][j] = DegenerateMetastableManifoldVertexSelector(LeftEigenmatrices[i][j], RoundingError = RoundingError, Iterations = Iterations)
	return SimplexVertices
def SimplexVerticesVs2DParameterSpaceSymmetry(LeftEigenmatrices, RightEigenmatrices, StateSpaceSymmetryTransformation, PowerForIdentity, RndError = 10, InitPairsPerMatrix = None):
	Parameter1DataPoints = len(LeftEigenmatrices)
	Parameter2DataPoints = len(LeftEigenmatrices[0])
	SimplexVertices = [[[] for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)]
	for i in range(Parameter1DataPoints):
		for j in range(Parameter2DataPoints):
			SimplexVertices[i][j] = SymmetryMetastableManifoldVertexSelector(LeftEigenmatrices[i][j], RightEigenmatrices[i][j], 
				StateSpaceSymmetryTransformation, PowerForIdentity, RoundingError = RndError, InitialPairsPerMatrix = InitPairsPerMatrix)
	return SimplexVertices
def ApproximatePOVMConstructor(LeftEigenmatrices, SimplexVertices):
	AffineTrafos = np.vstack((np.array(SimplexVertices).T,np.ones((len(SimplexVertices)),dtype=complex)))
	InverseTrafos = linalg.inv(AffineTrafos)
	POVMs = []
	for index in range(len(SimplexVertices)):
		POVM = InverseTrafos[index][0]*LeftEigenmatrices[0]
		for index2 in range(1,len(SimplexVertices)):
			POVM += InverseTrafos[index][index2]*LeftEigenmatrices[index2]
		POVMs.append(POVM)
	return POVMs
def ApproximatePOVMConstructor2DParameterSpace(LeftEigenmatrices, SimplexVertices):
	Parameter1DataPoints = len(LeftEigenmatrices)
	Parameter2DataPoints = len(LeftEigenmatrices[0])
	POVMs = [[[] for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)]
	for i in range(Parameter1DataPoints):
		for j in range(Parameter2DataPoints):
			POVMs[i][j] = ApproximatePOVMConstructor(LeftEigenmatrices[i][j], SimplexVertices[i][j])
	return POVMs
def ClassicalityTest(POVMs):
	TotalNonClassicality = 0
	for POVM in POVMs:
		for Eigenvalue in np.real(linalg.eigvals(POVM)):
			if Eigenvalue < 0:
				TotalNonClassicality += abs(Eigenvalue)
			elif Eigenvalue > 1:
				TotalNonClassicality += Eigenvalue - 1
	return TotalNonClassicality
def ClassicalityTest2DParameterSpace(POVMs):
	Parameter1DataPoints = len(POVMs)
	Parameter2DataPoints = len(POVMs[0])
	TotalNonClassicality = [[[] for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)]
	for i in range(Parameter1DataPoints):
		for j in range(Parameter2DataPoints):
			TotalNonClassicality[i][j] = ClassicalityTest(POVMs[i][j])
	return TotalNonClassicality
def eMSConstructor(RightEigenmatrices, SimplexVertices):
	AffineTrafos = np.vstack((np.array(SimplexVertices).T,np.ones((len(SimplexVertices)),dtype=complex)))
	TransposeTrafos = AffineTrafos.T
	eMSs = []
	for index in range(len(SimplexVertices)):
		eMS = TransposeTrafos[index][0]*RightEigenmatrices[0]
		for index2 in range(1,len(SimplexVertices)):
			eMS += TransposeTrafos[index][index2]*RightEigenmatrices[index2]
		eMSs.append(eMS)
	return eMSs
def eMSConstructor2DParameterSpace(RightEigenmatrices, SimplexVertices):
	Parameter1DataPoints = len(RightEigenmatrices)
	Parameter2DataPoints = len(RightEigenmatrices[0])
	eMSs = [[[] for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)]
	for i in range(Parameter1DataPoints):
		for j in range(Parameter2DataPoints):
			eMSs[i][j] = eMSConstructor(RightEigenmatrices[i][j], SimplexVertices[i][j])
	return eMSs
def InitialStateProjection(InitialDensity, POVMs):
	return np.array([np.trace(np.dot(POVM, InitialDensity)) for POVM in POVMs])
def EffectiveMasterOperator(Eigenvalues, SimplexVertices):
	AffineTrafos = np.vstack((np.array(SimplexVertices).T,np.ones((len(SimplexVertices)),dtype=complex)))
	InverseTrafos = linalg.inv(AffineTrafos)
	EigenvalueMatrix = np.diag(np.matrix(Eigenvalues).A1)
	EffectiveOp = np.dot(InverseTrafos, np.dot(EigenvalueMatrix, AffineTrafos))
	return EffectiveOp
def EffectiveMasterOperator2DParameterSpace(Eigenvalues, SimplexVertices):
	Parameter1DataPoints = len(Eigenvalues)
	Parameter2DataPoints = len(Eigenvalues[0])
	EffectiveMasterOperators = [[[] for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)]
	for i in range(Parameter1DataPoints):
		for j in range(Parameter2DataPoints):
			EffectiveMasterOperators[i][j] = EffectiveMasterOperator(Eigenvalues[i][j], SimplexVertices[i][j])
	return EffectiveMasterOperators
def EffectiveSteadyStateRatioDifference(Eigenvalues,SimplexVertices,RatioIndicies):
	EffectiveMastOp = EffectiveMasterOperator(Eigenvalues,SimplexVertices)
	Spectrum = linalg.eig(MastOpBlock, left = True)
	EvalSpectrum = Spectrum[0]
	Evecs = []
	Dim = len(EvalSpectrum)
	for j in range(Dim):
		Evecs.append(Spectrum[1].T[j])
	Spectrum = zip(*sorted(zip(Spectrum,Evecs), key = lambda double: np.real(double[0])))
	SteadyState = Spectrum[1][-1]
	UnnormedProb = sum(list(SteadyState))
	SteadyState = np.array(SteadyState)/UnnormedProb
	RatioDifference = SteadyState[RatioIndicies[1]]/SteadyState[RatioIndicies[0]]-SteadyState[RatioIndicies[2]]/SteadyState[RatioIndicies[1]]
	return RatioDifference
def EffectiveSteadyStateRatioDifference2D(Eigenvalues,SimplexVertices,RatioIndicies):
	Parameter1DataPoints = len(Eigenvalues)
	Parameter2DataPoints = len(Eigenvalues[0])
	RatioDifferences = [[[] for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)]
	for i in range(Parameter1DataPoints):
		for j in range(Parameter2DataPoints):
			RatioDifferences[i][j] = EffectiveSteadyStateRatioDifference(Eigenvalues[i][j], SimplexVertices[i][j], RatioIndicies)
	return RatioDifferences
def SimilarityNonHermiticity2D(EffectiveMasterOperators):
	Parameter1DataPoints = len(EffectiveMasterOperators)
	Parameter2DataPoints = len(EffectiveMasterOperators[0])
	SimilarityNonHermiticity = [[0 for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)]
	for i in range(Parameter1DataPoints):
		for j in range(Parameter2DataPoints):
			Eigvals, Evecset = linalg.eig(EffectiveMasterOperators[i][j])
			Eigvecs = []
			for k in range(len(Eigvals)):
				if sum(Evecset.T[k]) != 0:
					Eigvecs.append(Evecset.T[k]/sum(Evecset.T[k]))
				else:
					Eigvecs.append(Evecset.T[k])
			zipped = zip(list(Eigvals), list(Eigvecs))
			zipped = sorted(zipped, key=lambda double: np.real(double[0]))
			Eigvals, Eigvecs = zip(*zipped)
			A = np.diag(np.sqrt(Eigvecs[-1]))
			n = 0
			for element in Eigvecs[-1]:
				if element == 0:
					n = 1
					break
			if n == 1:
				SimilarityNonHermiticity[i][j] = 10**(-18)
			else:
				Ainv = linalg.inv(A)
				SimilarityTransformed = np.dot(Ainv,np.dot(EffectiveMasterOperators[i][j],A))
				SimilarityNonHermiticity[i][j] = abs(np.trace(linalg.fractional_matrix_power(np.dot(np.conjugate(SimilarityTransformed-np.conjugate(SimilarityTransformed).T),SimilarityTransformed-np.conjugate(SimilarityTransformed).T),0.5)))/(2*np.trace(linalg.fractional_matrix_power(np.dot(np.conjugate(SimilarityTransformed).T,SimilarityTransformed),0.5)))
	return SimilarityNonHermiticity
def EffectiveStateEvolution(InitialProbability, EffectiveMasterOp, TimeStep, Time):
	StepEvolution = linalg.expm(TimeStep*EffectiveMasterOp)
	TimeDepProbability =[np.array(InitialProbability)]
	TimeAxis = [i*TimeStep for i in range(int(float(Time)/float(TimeStep)) + 1)]
	for i in range(len(TimeAxis)-1):
		TimeDepProbability.append(np.dot(StepEvolution,TimeDepProbability[-1]))
	return TimeDepProbability
def EffectiveObservableEvolution(InitialProbability, EffectiveMasterOp, TimeStep, Time, ObservableProjection):
	StepEvolution = linalg.expm(TimeStep*EffectiveMasterOp)
	CurrentState = InitialProbability
	TimeDepObservable =[np.dot(InitialProbability, ObservableProjection)]
	for i in range(int(float(Time)/float(TimeStep))):
		CurrentState = np.dot(StepEvolution,CurrentState)
		TimeDepObservable.append(np.dot(CurrentState,ObservableProjection))
	return TimeDepObservable
def ProbabilityToCoefficientConversion(TimeDepProbability, Eigenvalues, SimplexVertices):
	AffineTrafos = np.vstack((np.array(SimplexVertices).T,np.ones((len(SimplexVertices)),dtype=complex)))
	return np.array([np.dot(AffineTrafos,TimeDepProb) for TimeDepProb in TimeDepProbability])
def KCQGHermitianBasisConstruction(Eigenvalues, LeftEigenmatrices, RightEigenmatrices, BlockIndicies, Sites, Rounding = 8):
	LeftEigenmatrices[-1] = (LeftEigenmatrices[-1]/np.trace(LeftEigenmatrices[-1]))*len(LeftEigenmatrices[-1])
	RightEigenmatrices[-1] = RightEigenmatrices[-1]/np.trace(RightEigenmatrices[-1])
	for i in range(len(Eigenvalues)-1):
		LeftEigenmatrices[i] = LeftEigenmatrices[i]/np.conjugate(np.trace(np.dot(np.conjugate(LeftEigenmatrices[i]).T, RightEigenmatrices[i])))
	Eigenvalues = list(Eigenvalues)
	RightEigenmatrices = list(RightEigenmatrices)
	LeftEigenmatrices = list(LeftEigenmatrices)
	BlockIndicies = list(BlockIndicies)
	HermitianLeftEigenmatrices = []
	HermitianRightEigenmatrices = []
	NewBlockIndicies = []
	while len(LeftEigenmatrices) >= 1:
		CurrentEval = Eigenvalues.pop(0)
		CurrentLeft = LeftEigenmatrices.pop(0)
		CurrentRight = RightEigenmatrices.pop(0)
		CurrentBlock = BlockIndicies.pop(0)
		if CurrentBlock == 0 or CurrentBlock == float(Sites)/2.0:
			if round(np.trace(la.matrix_power(CurrentRight+np.conjugate(CurrentRight).T,2)),5) == 0:
				HermitianRightEigenmatrices.append((CurrentRight-np.conjugate(CurrentRight).T)/2j)
				HermitianLeftEigenmatrices.append((CurrentLeft-np.conjugate(CurrentLeft).T)/2j)
			else:
				HermitianRightEigenmatrices.append((CurrentRight+np.conjugate(CurrentRight).T)/2)
				HermitianLeftEigenmatrices.append((CurrentLeft+np.conjugate(CurrentLeft).T)/2)
		else:
			for i in range(len(BlockIndicies)):
				if BlockIndicies[i] == float(Sites) - CurrentBlock and round(abs(CurrentEval - np.conjugate(Eigenvalues[i])),Rounding) == 0:
					SecondEval = Eigenvalues.pop(i)
					SecondBlock = BlockIndicies.pop(i)
					SecondLeft = LeftEigenmatrices.pop(i)
					SecondRight = RightEigenmatrices.pop(i)
					HermitianRightEigenmatrices.append(CurrentRight+SecondRight)
					HermitianRightEigenmatrices.append((CurrentRight-SecondRight)/1j)
					HermitianLeftEigenmatrices.append((CurrentLeft+SecondLeft)/2)
					HermitianLeftEigenmatrices.append((CurrentLeft-SecondLeft)/2j)
					break
			else:
				HermitianRightEigenmatrices.append((CurrentRight+np.conjugate(CurrentRight).T)/2)
				HermitianLeftEigenmatrices.append((CurrentLeft+np.conjugate(CurrentLeft).T)/2)
	return HermitianLeftEigenmatrices, HermitianRightEigenmatrices
def KCQG_EigenvectorBasistoHermitianMatrixBasis2DParameterSpace(Eigenvalues, LeftEvecs, RightEvecs, BlockIndicies, Sites, Rounding = 9):
	Parameter1DataPoints = len(LeftEvecs)
	Parameter2DataPoints = len(LeftEvecs[0])
	LeftEmats, RightEmats = EigenVectorBlocktoMatrixEmbedding2DParamSpace(LeftEvecs,RightEvecs,BlockIndicies,Sites)
	HermitianLeftEigenmatrices = [[[] for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)]
	HermitianRightEigenmatrices = [[[] for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)]
	for i in range(Parameter1DataPoints):
		for j in range(Parameter2DataPoints):
			HermitianLeftEigenmatrices[i][j], HermitianRightEigenmatrices[i][j] = KCQGHermitianBasisConstruction(Eigenvalues[i][j], LeftEmats[i][j], RightEmats[i][j], BlockIndicies[i][j], Sites, Rounding = 9)
	return HermitianLeftEigenmatrices, HermitianRightEigenmatrices
def MultipleStatesObservablesExpectation2dParameterSpace(States, Observables, SortingFunction = None, CalculatePurities = False):
	Parameter1DataPoints = len(States)
	Parameter2DataPoints = len(States[0])
	if CalculatePurities == True:
		Purities = [[[0 for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)] for k in range(len(States[0][0]))]
		Expectations = [[[[0 for k in range(Parameter2DataPoints)] for l in range(Parameter1DataPoints)]for j in range(len(Observables))] for i in range(len(States[0][0]))]
		for i in range(Parameter1DataPoints):
			for j in range(Parameter2DataPoints):
				if SortingFunction == None:
					for k in range(len(States[i][j])):
						for l in range(len(Observables)):
							Expectations[k][l][i][j] = np.trace(np.dot(States[i][j][k],Observables[l]))
						Purities[k][i][j] = np.trace(np.dot(States[i][j][k],States[i][j][k]))
				else:
					Exps = [[0 for a in range(len(Observables))] for b in range(len(States[0][0]))]
					Pure = [0 for b in range(len(States[0][0]))]
					for k in range(len(States[i][j])):
						for l in range(len(Observables)):
							Exps[k][l] = np.trace(np.dot(States[i][j][k],Observables[l]))
						Pure[k] = np.trace(np.dot(States[i][j][k],States[i][j][k]))
					Exps, Pure = SortingFunction(Exps, AdditionalQuantityToSort = Pure)
					for k in range(len(States[i][j])):
						for l in range(len(Observables)):
							Expectations[k][l][i][j] = Exps[k][l]
						Purities[k][i][j] = Pure[k]
		return Expectations, Purities
	else:
		Expectations = [[[[0 for k in range(Parameter2DataPoints)] for l in range(Parameter1DataPoints)]for j in range(len(Observables))] for i in range(len(States[0][0]))]
		for i in range(Parameter1DataPoints):
			for j in range(Parameter2DataPoints):
				if SortingFunction == None:
					for k in range(len(States[i][j])):
						for l in range(len(Observables)):
							Expectations[k][l][i][j] = np.trace(np.dot(States[i][j][k],Observables[l]))
				else:
					Exps = [[0 for a in range(len(Observables))] for b in range(len(States[0][0]))]
					for k in range(len(States[i][j])):
						for l in range(len(Observables)):
							Exps[k][l] = np.trace(np.dot(States[i][j][k],Observables[l]))
					Exps = SortingFunction(Exps)
					for k in range(len(States[i][j])):
						for l in range(len(Observables)):
							Expectations[k][l][i][j] = Exps[k][l]
		return Expectations
def MultipleStatesObservablesExpectation2dParameterSpacePlusSimplexSort(States, Observables, SortingFunction, SimplexVertices, CalculatePurities = False):
	Parameter1DataPoints = len(States)
	Parameter2DataPoints = len(States[0])
	SortedSimplexVertices = [[[] for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)]
	if CalculatePurities == True:
		Purities = [[[0 for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)] for k in range(len(States[0][0]))]
		Expectations = [[[[0 for k in range(Parameter2DataPoints)] for l in range(Parameter1DataPoints)]for j in range(len(Observables))] for i in range(len(States[0][0]))]
		for i in range(Parameter1DataPoints):
			for j in range(Parameter2DataPoints):
				Exps = [[0 for a in range(len(Observables))] for b in range(len(States[0][0]))]
				Pure = [0 for b in range(len(States[0][0]))]
				for k in range(len(States[i][j])):
					for l in range(len(Observables)):
						Exps[k][l] = np.trace(np.dot(States[i][j][k],Observables[l]))
					Pure[k] = np.trace(np.dot(States[i][j][k],States[i][j][k]))
				Exps, Pure, SortedSimplexVertices[i][j] = SortingFunction(Exps, AdditionalQuantityToSort = Pure, SimpVerts = SimplexVertices[i][j])
				for k in range(len(States[i][j])):
					for l in range(len(Observables)):
						Expectations[k][l][i][j] = Exps[k][l]
					Purities[k][i][j] = Pure[k]
		return Expectations, Purities, SortedSimplexVertices
	else:
		Expectations = [[[[0 for k in range(Parameter2DataPoints)] for l in range(Parameter1DataPoints)]for j in range(len(Observables))] for i in range(len(States[0][0]))]
		for i in range(Parameter1DataPoints):
			for j in range(Parameter2DataPoints):
				Exps = [[0 for a in range(len(Observables))] for b in range(len(States[0][0]))]
				for k in range(len(States[i][j])):
					for l in range(len(Observables)):
						Exps[k][l] = np.trace(np.dot(States[i][j][k],Observables[l]))
				Exps, SortedSimplexVertices[i][j] = SortingFunction(Exps, SimpVerts = SimplexVertices[i][j])
				for k in range(len(States[i][j])):
					for l in range(len(Observables)):
						Expectations[k][l][i][j] = Exps[k][l]
		return Expectations, SortedSimplexVertices
def SSEigenvectorMag(Sites, DecayRate, Field, Temperature):
	ExcitedStateProjector = KCQG_ExcitedStateProjection(DecayRate, Field, Temperature)
	UnexcitedStateProjector = np.eye(2)-ExcitedStateProjector
	SingleSpinMag = 0.5*ExcitedStateProjector-0.5*UnexcitedStateProjector
	SSSiteMagnetisationOps = CompositeJumps(SingleSpinMag, 1, Sites)
	SSCyclicBasisSiteMagnetisationOps = []
	for Op in SSSiteMagnetisationOps:
		ObservableNewBasisVector = StdBMCycBVMap(Op, Sites)
		ObservableVectorFull = np.array(ObservableNewBasisVector[0])
		for index in range(1, len(ObservableNewBasisVector)):
			ObservableVectorFull = np.concatenate((ObservableVectorFull, ObservableNewBasisVector[index]))
		SSCyclicBasisSiteMagnetisationOps.append(CycBVCycBMMap(ObservableVectorFull, Sites))
	return SSCyclicBasisSiteMagnetisationOps
def MultipleStatesSSEigenvectorMagExpectation2dParameterSpacePlusSimplexSort(States, SortingFunction, SimplexVertices, Parameter1, Parameter2, Sites, DecayRate):
	Parameter1DataPoints = len(States)
	Parameter2DataPoints = len(States[0])
	eMSNumber = len(States[0][0])
	SortedSimplexVertices = [[[] for i in range(Parameter2DataPoints)] for j in range(Parameter1DataPoints)]
	Expectations = [[[[0 for k in range(Parameter2DataPoints)] for l in range(Parameter1DataPoints)] for j in range(Sites)] for i in range(len(States[0][0]))]
	for i in range(Parameter1DataPoints):
		for j in range(Parameter2DataPoints):
			Observables = SSEigenvectorMag(Sites, DecayRate, Parameter1[i], Parameter2[j])
			Exps = [[0 for a in range(len(Observables))] for b in range(len(States[0][0]))]
			for k in range(len(States[i][j])):
				for l in range(len(Observables)):
					Exps[k][l] = np.trace(np.dot(States[i][j][k],Observables[l]))
			Exps, Add, SortedSimplexVertices[i][j] = SortingFunction(Exps, AdditionalQuantityToSort = [0 for t in range(eMSNumber)], SimpVerts = SimplexVertices[i][j])
			for k in range(len(States[i][j])):
				for l in range(len(Observables)):
					Expectations[k][l][i][j] = Exps[k][l]
	return Expectations, SortedSimplexVertices
"""
def MultipleStatesObservablesExpectation2dParamPlotting(EvalRatios, Expectations, Parameter1Values, Parameter2Values, Parameter1Label, Parameter2Label, fontSize, ColorBarLevels, Extension = 0, 
	FigSize = (10,15)):
	Max1 = Parameter1Values[-1]
	Min1 = Parameter1Values[0]
	Max2 = Parameter2Values[-1]
	Min2 = Parameter2Values[0]
	Parameter2Values, Parameter1Values = np.meshgrid(Parameter2Values, Parameter1Values)
	States = len(Expectations)
	Observables = len(Expectations[0])
	font = {'family' : 'cmr10','serif' : 'cmr10','weight' : 'normal','size'   : fontSize}
	pyplot.rc('font', **font)
	fig = pyplot.figure(figsize = FigSize)
	for i  in range(States):
		for j in range(Observables):
			if j == Observables-1:
				ax = pyplot.subplot2grid((States,400*(Observables-1)+490+Extension),(i,400*j), colspan=490)
			else:
				ax = pyplot.subplot2grid((States,400*(Observables-1)+490+Extension),(i,400*j), colspan=400)
			pyplot.pcolor(Parameter1Values,Parameter2Values,np.array(Expectations[i][j]).T, cmap = 'viridis', vmax = 0.5, vmin = -0.5)
			if j == Observables-1:
				cbar = pyplot.colorbar(ticks = [-0.5,0.0,0.5])
				cbar.set_ticklabels([str(-0.5),str(0.0),str(0.5)])
			pyplot.contour(Parameter1Values,Parameter2Values,np.array(EvalRatios).T, levels = [20,20.1])
			pyplot.xlim(Min1,Max1)
			if j == 0:
				pyplot.ylim(Min2,Max2)
				pyplot.yticks([Min2,Max2],[str(Min2),str(Max2)])
				pyplot.ylabel(Parameter2Label, labelpad = -21)
			else:
				pyplot.setp(ax.get_yticklabels(), visible = False)
			if (j+1)%2 == 0 and i == 0:
				pyplot.xlabel(Parameter1Label, labelpad = -12)
				pyplot.xticks([Min1,Max1],[str(Min1),str(Max1)])
				ax.xaxis.tick_top()
				ax.xaxis.set_label_position('top')
			elif (j+1)%2 != 0 and i == States-1:
				pyplot.xticks([Min1,Max1],[str(Min1),str(Max1)])
				pyplot.xlabel(Parameter1Label, labelpad = -18)
			else:
				pyplot.setp(ax.get_xticklabels(), visible = False)
	fig.subplots_adjust(hspace = 0.15, wspace = 0.0, left = 0.05, bottom = 0.06, right = 0.95, top = 0.94)
"""
def SiteMagnetisationSort(eMSSiteMag, AdditionalQuantityToSort = None, SimpVerts = None):
	UpSites = []
	for Mags in eMSSiteMag:
		Ups = []
		i = 1
		for Mag in Mags:
			if Mag > 0:
				Ups.append(i)
			i += 1
		if len(Ups) == 1:
			UpSites.append(Ups[0])
		else:
			UpSites.append(0)
	if SimpVerts == None:
		if AdditionalQuantityToSort == None:
			zipped = zip(list(UpSites), list(eMSSiteMag))
			zipped = sorted(zipped, key=lambda double: np.real(double[0]))
			SortedUpSites, SortedeMSSiteMag = zip(*zipped)
			return SortedeMSSiteMag
		else:
			zipped = zip(list(UpSites), list(eMSSiteMag), list(AdditionalQuantityToSort))
			zipped = sorted(zipped, key=lambda Triple: np.real(Triple[0]))
			SortedUpSites, SortedeMSSiteMag, SortedAdditional = zip(*zipped)
			return SortedeMSSiteMag, SortedAdditional
	else:
		if AdditionalQuantityToSort == None:
			zipped = zip(list(UpSites), list(eMSSiteMag), list(SimpVerts))
			zipped = sorted(zipped, key=lambda double: np.real(double[0]))
			SortedUpSites, SortedeMSSiteMag, SortedSimplex = zip(*zipped)
			return SortedeMSSiteMag, SortedSimplex
		else:
			zipped = zip(list(UpSites), list(eMSSiteMag), list(AdditionalQuantityToSort), list(SimpVerts))
			zipped = sorted(zipped, key=lambda Triple: np.real(Triple[0]))
			SortedUpSites, SortedeMSSiteMag, SortedAdditional, SortedSimplex = zip(*zipped)
			return SortedeMSSiteMag, SortedAdditional, SortedSimplex
def SiteMagnetisationSortTwo(eMSSiteMag, AdditionalQuantityToSort = None, SimpVerts = None):
	UpSites = []
	for Mags in eMSSiteMag:
		Ups = 0
		NumUp = 0
		Sites = len(Mags)
		i = 0
		for Mag in Mags:
			if Mag > 0:
				Ups += 2**i
				NumUp += 1
			i += 1
		if NumUp == 0:
			UpSites.append(0)
		elif NumUp == 1:
			UpSites.append(Ups)
		else:
			UpSites.append(Ups+2**((NumUp-1)*Sites))
	if SimpVerts == None:
		if AdditionalQuantityToSort == None:
			zipped = zip(list(UpSites), list(eMSSiteMag))
			zipped = sorted(zipped, key=lambda double: np.real(double[0]))
			SortedUpSites, SortedeMSSiteMag = zip(*zipped)
			return SortedeMSSiteMag
		else:
			zipped = zip(list(UpSites), list(eMSSiteMag), list(AdditionalQuantityToSort))
			zipped = sorted(zipped, key=lambda Triple: np.real(Triple[0]))
			SortedUpSites, SortedeMSSiteMag, SortedAdditional = zip(*zipped)
			return SortedeMSSiteMag, SortedAdditional
	else:
		if AdditionalQuantityToSort == None:
			zipped = zip(list(UpSites), list(eMSSiteMag), list(SimpVerts))
			zipped = sorted(zipped, key=lambda double: np.real(double[0]))
			SortedUpSites, SortedeMSSiteMag, SortedSimplex = zip(*zipped)
			return SortedeMSSiteMag, SortedSimplex
		else:
			zipped = zip(list(UpSites), list(eMSSiteMag), list(AdditionalQuantityToSort), list(SimpVerts))
			zipped = sorted(zipped, key=lambda Triple: np.real(Triple[0]))
			SortedUpSites, SortedeMSSiteMag, SortedAdditional, SortedSimplex = zip(*zipped)
			return SortedeMSSiteMag, SortedAdditional, SortedSimplex
def SiteMagnetisationAdaptedSort(eMSSiteMag, AdditionalQuantityToSort = None, SimpVerts = None):
	SortedeMSSiteMag = []
	SortedAdditional = []
	SortedSimplex = []
	OneUpSite = []
	TwoUpSiteLarge = []
	TwoUpSiteSmall = []
	Sites = len(eMSSiteMag[0])
	for Index in range(len(eMSSiteMag)):
		Ups = []
		NumUp = 0
		SiteIndex = 0
		for Mag in eMSSiteMag[Index]:
			if Mag > 0:
				Ups.append(SiteIndex)
				NumUp += 1
			SiteIndex += 1
		if NumUp == 0:
			SortedeMSSiteMag.append(eMSSiteMag[Index])
			SortedAdditional.append(AdditionalQuantityToSort[Index])
			SortedSimplex.append(SimpVerts[Index])
		elif NumUp == 1:
			OneUpSite.append((eMSSiteMag[Index],AdditionalQuantityToSort[Index],SimpVerts[Index],Ups[0]))
		elif NumUp == 2:
			Distance1 = max(Ups)-min(Ups)-1
			Distance2 = Sites - 2 - Distance1
			Distances = [Distance1,Distance2]
			if min(Distances) == 2:
				SortingSite = Ups[Distances.index(2)]
				TwoUpSiteLarge.append((eMSSiteMag[Index],AdditionalQuantityToSort[Index],SimpVerts[Index],SortingSite))
			if min(Distances) == 1:
				SortingSite = Ups[Distances.index(1)]
				TwoUpSiteSmall.append((eMSSiteMag[Index],AdditionalQuantityToSort[Index],SimpVerts[Index],SortingSite))
		else:
			return SiteMagnetisationSortTwo(eMSSiteMag, AdditionalQuantityToSort = AdditionalQuantityToSort, SimpVerts = SimpVerts)
	if len(OneUpSite) != 0:
		OneSortedeMSSiteMag, OneSortedAdditional, OneSortedSimplex, OneSortedSites = zip(*sorted(OneUpSite, key = lambda Quad: Quad[-1]))
		SortedeMSSiteMag.extend(OneSortedeMSSiteMag)
		SortedAdditional.extend(OneSortedAdditional)
		SortedSimplex.extend(OneSortedSimplex)
	if len(TwoUpSiteLarge) != 0:
		TwoLargeSortedeMSSiteMag, TwoLargeSortedAdditional, TwoLargeSortedSimplex, TwoLargeSortedSites = zip(*sorted(TwoUpSiteLarge, key = lambda Quad: Quad[-1]))
		SortedeMSSiteMag.extend(TwoLargeSortedeMSSiteMag)
		SortedAdditional.extend(TwoLargeSortedAdditional)
		SortedSimplex.extend(TwoLargeSortedSimplex)
	if len(TwoUpSiteSmall) != 0:
		TwoSmallSortedeMSSiteMag, TwoSmallSortedAdditional, TwoSmallSortedSimplex, TwoSmallSortedSites = zip(*sorted(TwoUpSiteSmall, key = lambda Quad: Quad[-1]))
		SortedeMSSiteMag.extend(TwoSmallSortedeMSSiteMag)
		SortedAdditional.extend(TwoSmallSortedAdditional)
		SortedSimplex.extend(TwoSmallSortedSimplex)
	return SortedeMSSiteMag, SortedAdditional, SortedSimplex
def KCQG_Input_Call():
	args = sys.argv
	Sites = int(args[1])
	InitialField = float(args[2])
	FinalField = float(args[3])
	FieldStep = float(args[4])
	InitialTemp = float(args[5])
	FinalTemp = float(args[6])
	TempStep = float(args[7])
	Hardness = float(args[8])
	eMSNumber = int(args[9])
	eigenvalueNumber = int(args[10])
	return Sites, [InitialField,FinalField], FieldStep, [InitialTemp, FinalTemp], TempStep, Hardness, eMSNumber, eigenvalueNumber
def KCQG_1MM_Data_Save(Sites, FieldRange, FieldStep, TempRange, TempStep, Hardness, Eigenvalues, eMSSiteMags, SSeMSSiteMags, Purities, SimplexVertices, Classicality):
	Data = np.array([])
	Data = np.concatenate((Data, np.real(np.array(Eigenvalues).flatten())))
	Data = np.concatenate((Data, np.real(np.array(eMSSiteMags).flatten())))
	Data = np.concatenate((Data, np.real(np.array(SSeMSSiteMags).flatten())))
	Data = np.concatenate((Data, np.real(np.array(Purities).flatten())))
	Data = np.concatenate((Data, np.array(SimplexVertices).flatten().view(float)))
	Data = np.concatenate((Data, np.real(np.array(Classicality).flatten())))
	file_name = "%s"%(Sites) + "Site" + "%sFi%sFf%sFs%sTi%sTf%sTs%sp"%(FieldRange[0], FieldRange[1], FieldStep, TempRange[0], TempRange[1], TempStep, Hardness) + "Data.txt"
	np.savetxt(file_name, Data)
def KCQG_1MM_Data_Load_Init(Sites, FieldRange, FieldStep, TempRange, TempStep, Hardness, eigenvalueNumber, eMSNumber):
	file_name = "%s"%(Sites) + "Site" + "%sFi%sFf%sFs%sTi%sTf%sTs%sp"%(FieldRange[0], FieldRange[1], FieldStep, TempRange[0], TempRange[1], TempStep, Hardness) + "Data.txt"
	FieldNumber = int(round(float(FieldRange[1]-FieldRange[0])/float(FieldStep)))+1
	Data = np.loadtxt(file_name)
	NewEigenvalues = Data[0:eigenvalueNumber*FieldNumber]
	NewEigenvalues = np.resize(NewEigenvalues,(FieldNumber,eigenvalueNumber))
	NeweMSSiteMags = Data[eigenvalueNumber*FieldNumber:(eigenvalueNumber+eMSNumber*Sites)*FieldNumber]
	NeweMSSiteMags = np.resize(NeweMSSiteMags,(eMSNumber,Sites,FieldNumber))
	NewPurities = Data[(eigenvalueNumber+eMSNumber*Sites)*FieldNumber:(eigenvalueNumber+eMSNumber*Sites+Sites)*FieldNumber]
	NewPurities = np.resize(NewPurities,(eMSNumber,FieldNumber))
	NewSimplexVertices = Data[(eigenvalueNumber+eMSNumber*Sites+Sites)*FieldNumber:(eigenvalueNumber+eMSNumber*Sites+Sites+eMSNumber*(eMSNumber-1))*FieldNumber].view(complex)
	NewSimplexVertices = np.resize(NewSimplexVertices,(FieldNumber,eMSNumber,eMSNumber-1))
	NewClassicality = Data[(eigenvalueNumber+eMSNumber*Sites+Sites+eMSNumber*(eMSNumber-1))*FieldNumber:(eigenvalueNumber+eMSNumber*Sites+Sites+eMSNumber*(eMSNumber-1)+1)*FieldNumber]
	return NewEigenvalues, NeweMSSiteMags, NewPurities, NewSimplexVertices, NewClassicality
def KCQG_1MM_Data_Load(Sites, FieldRange, FieldStep, TempRange, TempStep, Hardness, eigenvalueNumber, eMSNumber, CurrentEigenvalues, 
	CurrenteMSSiteMags, CurrentPurities, CurrentSimplexVertices, CurrentClassicality):
	file_name = "%s"%(Sites) + "Site" + "%sFi%sFf%sFs%sTi%sTf%sTs%sp"%(FieldRange[0], FieldRange[1], FieldStep, TempRange[0], TempRange[1], TempStep, Hardness) + "Data.txt"
	print(float(FieldRange[1]-FieldRange[0]))
	FieldNumber = int(round(float(FieldRange[1]-FieldRange[0])/float(FieldStep)))+1
	print(FieldNumber)
	Data = np.loadtxt(file_name)
	NewEigenvalues = Data[0:eigenvalueNumber*FieldNumber]
	NewEigenvalues = np.resize(NewEigenvalues,(FieldNumber,eigenvalueNumber))
	NeweMSSiteMags = Data[eigenvalueNumber*FieldNumber:(eigenvalueNumber+eMSNumber*Sites)*FieldNumber]
	NeweMSSiteMags = np.resize(NeweMSSiteMags,(eMSNumber,Sites,FieldNumber))
	NewPurities = Data[(eigenvalueNumber+eMSNumber*Sites)*FieldNumber:(eigenvalueNumber+eMSNumber*Sites+Sites)*FieldNumber]
	NewPurities = np.resize(NewPurities,(eMSNumber,FieldNumber))
	NewSimplexVertices = Data[(eigenvalueNumber+eMSNumber*Sites+Sites)*FieldNumber:(eigenvalueNumber+eMSNumber*Sites+Sites+eMSNumber*(eMSNumber-1))*FieldNumber].view(complex)
	NewSimplexVertices = np.resize(NewSimplexVertices,(FieldNumber,eMSNumber,eMSNumber-1))
	NewClassicality = Data[(eigenvalueNumber+eMSNumber*Sites+Sites+eMSNumber*(eMSNumber-1))*FieldNumber:(eigenvalueNumber+eMSNumber*Sites+Sites+eMSNumber*(eMSNumber-1)+1)*FieldNumber]
	CurrentEigenvalues = np.concatenate((CurrentEigenvalues,NewEigenvalues))
	CurrenteMSSiteMags = np.concatenate((CurrenteMSSiteMags,NeweMSSiteMags), axis = 2)
	CurrentPurities = np.concatenate((CurrentPurities,NewPurities), axis = 1)
	CurrentSimplexVertices = np.concatenate((CurrentSimplexVertices,NewSimplexVertices))
	CurrentClassicality = np.concatenate((CurrentClassicality,NewClassicality))
	return CurrentEigenvalues, CurrenteMSSiteMags, CurrentPurities, CurrentSimplexVertices, CurrentClassicality
"""
Sites = 6
DecayRate = 1
FieldRange = [np.sqrt(0.03),np.sqrt(0.03)]
FieldStep = 1
TempRange = [0.001,0.001]
TempStep = 1
Hardness = 0.999
eMSNumber = Sites + 1 + 3

StateSpaceSymmetryTransformation = SpinBasisCyclicUnitaryGenerator(Sites)
ObservableNewBasisVector = StdBMCycBVMap(StateSpaceSymmetryTransformation, Sites)
ObservableVectorFull = np.array(ObservableNewBasisVector[0])
for index in range(1, len(ObservableNewBasisVector)):
	ObservableVectorFull = np.concatenate((ObservableVectorFull, ObservableNewBasisVector[index]))
StateSpaceSymmetryTransformation = CycBVCycBMMap(ObservableVectorFull, Sites)

SpinZ = np.array([[0.5,0],[0,-0.5]])
SiteMagnetisationOps = CompositeJumps(SpinZ, 1, Sites)
CyclicBasisSiteMagnetisationOps = []
for Op in SiteMagnetisationOps:
	ObservableNewBasisVector = StdBMCycBVMap(Op, Sites)
	ObservableVectorFull = np.array(ObservableNewBasisVector[0])
	for index in range(1, len(ObservableNewBasisVector)):
		ObservableVectorFull = np.concatenate((ObservableVectorFull, ObservableNewBasisVector[index]))
	CyclicBasisSiteMagnetisationOps.append(CycBVCycBMMap(ObservableVectorFull, Sites))

Evals, LeftEvecs, RightEvecs, BlockIndicies, FieldVals, TemperatureVals = GlassMasterOpSpectrumVs_FieldPlusTemperatureEvecs(
	DecayRate, FieldRange, FieldStep, TempRange, TempStep, Hardness, Sites, eMSNumber)
LeftEmats, RightEmats = KCQG_EigenvectorBasistoHermitianMatrixBasis2DParameterSpace(
	Evals, LeftEvecs, RightEvecs, BlockIndicies, Sites)
SimplexVertices = SimplexVerticesVs2DParameterSpaceSymmetry(LeftEmats, RightEmats, StateSpaceSymmetryTransformation, Sites, RndError = 1, InitPairsPerMatrix = 5)
POVMs = ApproximatePOVMConstructor2DParameterSpace(LeftEmats, SimplexVertices)
Classicality = ClassicalityTest2DParameterSpace(POVMs)
eMSs = eMSConstructor2DParameterSpace(RightEmats,SimplexVertices)
eMSSiteMags, Purities, SimplexVertices2 = MultipleStatesObservablesExpectation2dParameterSpacePlusSimplexSort(eMSs, CyclicBasisSiteMagnetisationOps, 
	SiteMagnetisationAdaptedSort, SimplexVertices, CalculatePurities = True)
SSeMSSiteMags, SimplexVs = MultipleStatesSSEigenvectorMagExpectation2dParameterSpacePlusSimplexSort(eMSs, SiteMagnetisationAdaptedSort, SimplexVertices, FieldVals, TemperatureVals, Sites, DecayRate)
for i in range(eMSNumber):
	print(np.around(SimplexVertices[0][0][i].real,3))
print(Classicality)
"""
