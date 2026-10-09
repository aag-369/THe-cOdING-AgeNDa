"""Pure-NumPy quantum simulation: state vectors, density matrices, MPS, PQCs, feature maps.

No quantum SDK is used anywhere; every gate, channel and measurement is
implemented in this package on top of NumPy linear algebra.
"""

from .circuit import ParamRef, QuantumCircuit, SimulationReport, run_density, run_statevector
from .density import DensityMatrix
from .entanglement import entanglement_entropy, mutual_information, purity, von_neumann_entropy
from .feature_map import QuantumFeatureExtractor, QuantumFeatureMap, QuantumFeatures
from .gates import GATES, unitarity_error
from .kernels import MemoryBudget, norm_error_bound
from .measurement import PauliString
from .mps import MPS
from .noise import NoiseModel
from .pqc import ParameterizedQuantumCircuit, reuploading_ansatz, ring_observables
from .statevector import StateVector

__all__ = [
    "ParamRef",
    "QuantumCircuit",
    "SimulationReport",
    "run_density",
    "run_statevector",
    "DensityMatrix",
    "entanglement_entropy",
    "mutual_information",
    "purity",
    "von_neumann_entropy",
    "QuantumFeatureExtractor",
    "QuantumFeatureMap",
    "QuantumFeatures",
    "GATES",
    "unitarity_error",
    "MemoryBudget",
    "norm_error_bound",
    "PauliString",
    "MPS",
    "NoiseModel",
    "ParameterizedQuantumCircuit",
    "reuploading_ansatz",
    "ring_observables",
    "StateVector",
]
