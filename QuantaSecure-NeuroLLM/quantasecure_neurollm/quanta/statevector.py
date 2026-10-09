"""Object-oriented pure-state simulator (wraps the batched kernels with batch size 1).

Supports any number of qubits that fits in memory; with complex128 amplitudes
an ``n``-qubit state needs ``16 * 2^n`` bytes (16 qubits: 1 MiB, 24 qubits:
256 MiB, 28 qubits: 4 GiB).  :class:`~quantasecure_neurollm.quanta.kernels.MemoryBudget`
warns above 256 MiB and refuses above 4 GiB unless ``allow_large=True``; use
:class:`~quantasecure_neurollm.quanta.mps.MPS` for larger, weakly-entangled states.

Example::

    sv = StateVector(3).h(0).cnot(0, 1).cnot(1, 2)       # GHZ
    sv.entanglement_entropy([0])                         # -> 1.0 bit
    sv.sample_counts(1000, rng)                          # ~{'000': 500, '111': 500}
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ..utils.mathx import clip_probabilities
from . import entanglement, kernels, measurement
from .circuit import QuantumCircuit, run_statevector
from .gates import GATES
from .measurement import PauliString

__all__ = ["StateVector"]


class StateVector:
    """A pure ``n``-qubit state ``|psi> in C^(2^n)`` with big-endian qubit order."""

    def __init__(
        self,
        n_qubits: int,
        amplitudes: Optional[np.ndarray] = None,
        *,
        budget: kernels.MemoryBudget = kernels.DEFAULT_BUDGET,
        allow_large: bool = False,
    ) -> None:
        if n_qubits < 1:
            raise ValueError("need at least one qubit")
        budget.check(f"{n_qubits}-qubit state vector", (2**n_qubits) * 16, allow_large)
        self.n_qubits = n_qubits
        self._budget = budget
        self._allow_large = allow_large
        if amplitudes is None:
            self._psi = np.zeros((1, 2**n_qubits), dtype=np.complex128)
            self._psi[0, 0] = 1.0
        else:
            a = np.asarray(amplitudes, dtype=np.complex128).reshape(1, -1)
            if a.shape[1] != 2**n_qubits:
                raise ValueError("amplitude vector has wrong length")
            nrm = np.linalg.norm(a)
            if nrm == 0:
                raise ValueError("zero vector is not a state")
            self._psi = a / nrm
        self.gate_count = 0

    # ---------------------------------------------------------------- creation
    @classmethod
    def from_bitstring(cls, bits: str) -> "StateVector":
        sv = cls(len(bits))
        sv._psi[0, 0] = 0.0
        sv._psi[0, int(bits, 2)] = 1.0
        return sv

    def copy(self) -> "StateVector":
        out = StateVector.__new__(StateVector)
        out.n_qubits = self.n_qubits
        out._budget = self._budget
        out._allow_large = self._allow_large
        out._psi = self._psi.copy()
        out.gate_count = self.gate_count
        return out

    # ------------------------------------------------------------------ access
    @property
    def amplitudes(self) -> np.ndarray:
        return self._psi[0].copy()

    @property
    def dim(self) -> int:
        return 2**self.n_qubits

    def norm(self) -> float:
        return float(np.linalg.norm(self._psi))

    def norm_drift(self) -> float:
        """``| ||psi|| - 1 |`` -- compare against :func:`kernels.norm_error_bound`."""
        return abs(self.norm() - 1.0)

    def renormalize(self) -> "StateVector":
        self._psi /= np.linalg.norm(self._psi)
        return self

    # ------------------------------------------------------------------- gates
    def apply(self, gate: Union[str, np.ndarray], qubits: Sequence[int], *params: float) -> "StateVector":
        """Apply a named gate (``"rx"``, ``"cx"``, ...) or an explicit unitary matrix."""
        if isinstance(gate, str):
            spec = GATES[gate]
            if len(qubits) != spec.n_qubits or len(params) != spec.n_params:
                raise ValueError(f"{gate} expects {spec.n_qubits} qubits and {spec.n_params} params")
            u = spec.matrix(*params)
            if spec.diagonal:
                self._psi = kernels.apply_diagonal(self._psi, np.diagonal(u), qubits, self.n_qubits)
            else:
                self._psi = kernels.apply_gate(self._psi, u, qubits, self.n_qubits)
        else:
            u = np.asarray(gate, dtype=np.complex128)
            self._psi = kernels.apply_gate(self._psi, u, qubits, self.n_qubits)
        self.gate_count += 1
        return self

    def h(self, q: int) -> "StateVector":
        return self.apply("h", [q])

    def x(self, q: int) -> "StateVector":
        return self.apply("x", [q])

    def y(self, q: int) -> "StateVector":
        return self.apply("y", [q])

    def z(self, q: int) -> "StateVector":
        return self.apply("z", [q])

    def s(self, q: int) -> "StateVector":
        return self.apply("s", [q])

    def t(self, q: int) -> "StateVector":
        return self.apply("t", [q])

    def rx(self, q: int, theta: float) -> "StateVector":
        return self.apply("rx", [q], theta)

    def ry(self, q: int, theta: float) -> "StateVector":
        return self.apply("ry", [q], theta)

    def rz(self, q: int, theta: float) -> "StateVector":
        return self.apply("rz", [q], theta)

    def cnot(self, control: int, target: int) -> "StateVector":
        return self.apply("cx", [control, target])

    def cz(self, a: int, b: int) -> "StateVector":
        return self.apply("cz", [a, b])

    def swap(self, a: int, b: int) -> "StateVector":
        return self.apply("swap", [a, b])

    def toffoli(self, c1: int, c2: int, target: int) -> "StateVector":
        return self.apply("ccx", [c1, c2, target])

    def unitary(self, u: np.ndarray, qubits: Sequence[int]) -> "StateVector":
        return self.apply(u, qubits)

    def run(self, circuit: QuantumCircuit, params: Optional[np.ndarray] = None, renormalize_tol: float = 1e-10) -> "StateVector":
        """Evolve under ``circuit`` (single parameter vector)."""
        if circuit.n_qubits != self.n_qubits:
            raise ValueError("circuit width does not match state")
        psi, report = run_statevector(circuit, params, initial=self._psi, renormalize_tol=renormalize_tol)
        self._psi = psi[:1]
        self.gate_count += report.n_gates
        return self

    # ------------------------------------------------------------- measurement
    def probabilities(self) -> np.ndarray:
        return measurement.born_probabilities(self._psi)[0]

    def expectation(self, pauli: Union[PauliString, str]) -> float:
        p = PauliString.from_label(pauli) if isinstance(pauli, str) else pauli
        return float(measurement.expectation(self._psi, p, self.n_qubits)[0])

    def sample(self, shots: int, rng: np.random.Generator, readout_error: float = 0.0) -> np.ndarray:
        """Basis-state indices of ``shots`` independent measurements (state not collapsed)."""
        out = measurement.sample_outcomes(self.probabilities(), shots, rng)
        return measurement.apply_readout_error(out, self.n_qubits, readout_error, rng)

    def sample_counts(self, shots: int, rng: np.random.Generator) -> dict[str, int]:
        idx = self.sample(shots, rng)
        counts = np.bincount(idx, minlength=self.dim)
        return {format(i, f"0{self.n_qubits}b"): int(c) for i, c in enumerate(counts) if c}

    def measure(self, qubit: int, rng: np.random.Generator) -> int:
        """Projective Z measurement of one qubit with state collapse (Born rule)."""
        p1 = float(measurement.marginal_probabilities(self.probabilities(), self.n_qubits)[qubit])
        outcome = int(rng.random() < p1)
        t = self._psi.reshape((1,) + (2,) * self.n_qubits).copy()
        idx: list[object] = [slice(None)] * (self.n_qubits + 1)
        idx[1 + qubit] = 1 - outcome
        t[tuple(idx)] = 0.0
        self._psi = t.reshape(1, -1)
        self.renormalize()
        return outcome

    # ----------------------------------------------------------- entanglement
    def reduced_density_matrix(self, qubits: Sequence[int]) -> np.ndarray:
        rho = self._psi[0][:, None] * self._psi[0][None, :].conj()
        return kernels.partial_trace(rho[None], qubits, self.n_qubits)[0]

    def entanglement_entropy(self, subsystem: Sequence[int]) -> float:
        return entanglement.entanglement_entropy(self._psi[0], subsystem, self.n_qubits)

    def fidelity(self, other: "StateVector") -> float:
        return float(abs(np.vdot(self._psi[0], other._psi[0])) ** 2)

    def __repr__(self) -> str:
        p = clip_probabilities(np.abs(self._psi[0]) ** 2)
        top = np.argsort(p)[::-1][:4]
        terms = ", ".join(f"|{format(i, f'0{self.n_qubits}b')}>:{p[i]:.3f}" for i in top if p[i] > 1e-6)
        return f"StateVector(n={self.n_qubits}, {terms})"
