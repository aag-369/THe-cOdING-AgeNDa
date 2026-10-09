"""Mixed-state (density-matrix) simulator with noise channels.

Memory: ``16 * 4^n`` bytes per state -- 8 qubits: 1 MiB, 10 qubits: 16 MiB,
12 qubits: 256 MiB (warned), 13 qubits: 1 GiB.  The hot paths (gates and the
built-in noise channels) cost one or two passes over ``rho`` each.

Numerical notes: after every simulated circuit ``rho`` is re-Hermitised
(``(rho + rho^dagger) / 2``) which removes the anti-Hermitian residue that
accumulates at ``O(G u)``.  The trace is preserved to ``O(G u)`` by every
channel; :meth:`DensityMatrix.trace_error` reports the observed deviation.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ..utils.mathx import clip_probabilities
from . import entanglement, kernels, measurement
from .circuit import QuantumCircuit, run_density
from .gates import GATES
from .measurement import PauliString
from .noise import NoiseModel
from .statevector import StateVector

__all__ = ["DensityMatrix"]


class DensityMatrix:
    """An ``n``-qubit density operator ``rho`` (Hermitian, PSD, unit trace)."""

    def __init__(
        self,
        n_qubits: int,
        rho: Optional[np.ndarray] = None,
        *,
        budget: kernels.MemoryBudget = kernels.DEFAULT_BUDGET,
        allow_large: bool = False,
    ) -> None:
        budget.check(f"{n_qubits}-qubit density matrix", (4**n_qubits) * 16, allow_large)
        self.n_qubits = n_qubits
        d = 2**n_qubits
        if rho is None:
            self._rho = np.zeros((1, d, d), dtype=np.complex128)
            self._rho[0, 0, 0] = 1.0
        else:
            r = np.asarray(rho, dtype=np.complex128)
            if r.shape != (d, d):
                raise ValueError("rho has wrong shape")
            self._rho = r[None].copy()

    @classmethod
    def from_statevector(cls, sv: Union[StateVector, np.ndarray], n_qubits: Optional[int] = None) -> "DensityMatrix":
        psi = sv.amplitudes if isinstance(sv, StateVector) else np.asarray(sv, dtype=np.complex128)
        n = sv.n_qubits if isinstance(sv, StateVector) else int(n_qubits or np.log2(psi.size))
        return cls(n, np.outer(psi, psi.conj()))

    @classmethod
    def maximally_mixed(cls, n_qubits: int) -> "DensityMatrix":
        d = 2**n_qubits
        return cls(n_qubits, np.eye(d, dtype=np.complex128) / d)

    @property
    def matrix(self) -> np.ndarray:
        return self._rho[0].copy()

    def copy(self) -> "DensityMatrix":
        return DensityMatrix(self.n_qubits, self._rho[0])

    # ---------------------------------------------------------------- dynamics
    def apply(self, gate: Union[str, np.ndarray], qubits: Sequence[int], *params: float) -> "DensityMatrix":
        if isinstance(gate, str):
            spec = GATES[gate]
            u = spec.matrix(*params)
            self._rho = kernels.apply_gate_dm(self._rho, u, qubits, self.n_qubits, diagonal=spec.diagonal)
        else:
            self._rho = kernels.apply_gate_dm(self._rho, np.asarray(gate, dtype=np.complex128), qubits, self.n_qubits)
        return self

    def apply_channel(self, kraus: Sequence[np.ndarray], qubits: Sequence[int]) -> "DensityMatrix":
        """Apply an arbitrary CPTP map given by Kraus operators."""
        self._rho = kernels.apply_kraus_dm(self._rho, kraus, qubits, self.n_qubits)
        return self

    def depolarize(self, p: float, qubits: Sequence[int]) -> "DensityMatrix":
        self._rho = kernels.depolarize_dm(self._rho, p, qubits, self.n_qubits)
        return self

    def amplitude_damp(self, gamma: float, qubit: int) -> "DensityMatrix":
        self._rho = kernels.amplitude_damp_dm(self._rho, gamma, qubit, self.n_qubits)
        return self

    def phase_damp(self, lam: float, qubit: int) -> "DensityMatrix":
        self._rho = kernels.phase_damp_dm(self._rho, lam, qubit, self.n_qubits)
        return self

    def run(self, circuit: QuantumCircuit, params: Optional[np.ndarray] = None, noise: Optional[NoiseModel] = None) -> "DensityMatrix":
        self._rho = run_density(circuit, params, noise=noise, initial_rho=self._rho)[:1]
        return self

    # ------------------------------------------------------------ observables
    def trace_error(self) -> float:
        return float(abs(np.trace(self._rho[0]) - 1.0))

    def probabilities(self) -> np.ndarray:
        return clip_probabilities(np.real(np.diagonal(self._rho[0])))

    def expectation(self, pauli: Union[PauliString, str]) -> float:
        p = PauliString.from_label(pauli) if isinstance(pauli, str) else pauli
        return float(measurement.expectation_density(self._rho, p, self.n_qubits)[0])

    def sample(self, shots: int, rng: np.random.Generator, readout_error: float = 0.0) -> np.ndarray:
        out = measurement.sample_outcomes(self.probabilities(), shots, rng)
        return measurement.apply_readout_error(out, self.n_qubits, readout_error, rng)

    def partial_trace(self, keep: Sequence[int]) -> np.ndarray:
        return kernels.partial_trace(self._rho, keep, self.n_qubits)[0]

    def purity(self) -> float:
        return entanglement.purity(self._rho[0])

    def von_neumann_entropy(self) -> float:
        return entanglement.von_neumann_entropy(self._rho[0])

    def subsystem_entropy(self, qubits: Sequence[int]) -> float:
        return entanglement.von_neumann_entropy(self.partial_trace(qubits))

    def fidelity_with(self, sv: StateVector) -> float:
        return entanglement.fidelity_pure_mixed(sv.amplitudes, self._rho[0])

    def is_valid(self, atol: float = 1e-9) -> bool:
        """Hermitian, unit trace and positive semi-definite (to ``atol``)."""
        r = self._rho[0]
        herm = np.allclose(r, r.conj().T, atol=atol)
        eig = np.linalg.eigvalsh(0.5 * (r + r.conj().T))
        return bool(herm and self.trace_error() < atol and eig.min() > -atol)
