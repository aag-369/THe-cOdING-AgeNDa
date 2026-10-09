"""Parameterised quantum circuits (PQCs) as differentiable layers.

A :class:`ParameterizedQuantumCircuit` maps per-sample inputs ``x`` and shared
weights ``w`` to a vector of expectation values ``f(x, w) = <O_k>``.  Its
:meth:`~ParameterizedQuantumCircuit.jacobian` returns *exact* gradients with
respect to both ``x`` and ``w`` via the **parameter-shift rule**.

Parameter-shift rule
--------------------
If a gate is ``exp(-i theta G)`` and ``G`` has exactly two eigenvalues
``+-r`` (here always ``r = 1/2``), then for any observable

    d<O>/dtheta = r [ <O>(theta + s) - <O>(theta - s) ],   s = pi / (4 r) = pi / 2,

with no truncation error (Mitarai et al. 2018; Schuld et al. 2019).  A circuit
parameter ``p`` that enters several gates with angles ``c_m * p`` gets
``d<O>/dp = sum_m c_m d<O>/dtheta_m`` (chain rule), so we first *unroll* the
circuit so that every occurrence has its own angle, shift each occurrence
individually and contract with the scale matrix.

All ``1 + 2M`` circuits (``M`` = number of occurrences) for all ``B`` samples
are simulated in a single batched sweep.  Cost per Jacobian:
``B (1 + 2M)`` state-vector simulations, versus ``B (1 + P)`` for one-sided
finite differences -- but with exact (not ``O(h)``) gradients and no
step-size tuning.  With shot-based expectations the estimator stays unbiased.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .circuit import Operation, ParamRef, QuantumCircuit, iter_parameter_occurrences, run_statevector
from .gates import GATES
from .measurement import PauliString, expectation, sample_outcomes, z_signs

__all__ = ["ParameterizedQuantumCircuit", "reuploading_ansatz", "ring_observables"]

_SHIFT = np.pi / 2


def ring_observables(n: int) -> list[PauliString]:
    """``Z_q`` for every qubit plus ``Z_q Z_{q+1}`` around a ring (or one pair if n == 2)."""
    obs = [PauliString.z(q) for q in range(n)]
    if n == 2:
        obs.append(PauliString.z(0, 1))
    elif n > 2:
        obs += [PauliString.z(q, (q + 1) % n) for q in range(n)]
    return obs


def reuploading_ansatz(n_qubits: int, n_layers: int) -> tuple[QuantumCircuit, int, int]:
    """Hardware-efficient data re-uploading ansatz.

    Layout of the parameter vector: ``[x_0 .. x_{n-1} | w_0 .. w_{W-1}]``.  Each
    layer re-uploads the same inputs (``RY(x_q)``), then applies trainable
    ``RZ(w) RY(w)`` rotations and a CNOT ring; a final trainable ``RY`` layer
    closes the circuit.  Returns ``(circuit, n_inputs, n_weights)``.
    """
    n_in = n_qubits
    n_w = n_layers * 2 * n_qubits + n_qubits
    qc = QuantumCircuit(n_qubits, n_in + n_w, name=f"reupload_{n_qubits}x{n_layers}")
    w = n_in
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(q, qc.p(q))
        for q in range(n_qubits):
            qc.rz(q, qc.p(w))
            qc.ry(q, qc.p(w + 1))
            w += 2
        if n_qubits > 1:
            for q in range(n_qubits if n_qubits > 2 else 1):
                qc.cx(q, (q + 1) % n_qubits)
    for q in range(n_qubits):
        qc.ry(q, qc.p(w))
        w += 1
    assert w == n_in + n_w
    return qc, n_in, n_w


class ParameterizedQuantumCircuit:
    """Batched PQC ``f(x, w) = [<O_1>, ..., <O_K>]`` with parameter-shift Jacobians."""

    def __init__(self, circuit: QuantumCircuit, n_inputs: int, observables: Sequence[PauliString]) -> None:
        if not 0 <= n_inputs <= circuit.n_params:
            raise ValueError("n_inputs out of range")
        self.circuit = circuit
        self.n_qubits = circuit.n_qubits
        self.n_inputs = n_inputs
        self.n_weights = circuit.n_params - n_inputs
        self.observables = list(observables)
        occ = list(iter_parameter_occurrences(circuit))
        for op_idx, _, _ in occ:
            name = circuit.ops[op_idx].name
            if not GATES[name].shiftable:
                raise ValueError(f"gate {name} does not admit the two-term parameter-shift rule")
        self.n_occurrences = len(occ)
        # Unrolled circuit: occurrence m gets its own parameter index m with unit scale.
        unrolled = QuantumCircuit(circuit.n_qubits, max(1, len(occ)), name=circuit.name + "_unrolled")
        m = 0
        for op in circuit.ops:
            new_params = []
            for a in op.params:
                if isinstance(a, ParamRef):
                    new_params.append(ParamRef(m, 1.0))
                    m += 1
                else:
                    new_params.append(a)
            unrolled.ops.append(Operation(op.name, op.qubits, tuple(new_params), op.matrix))
        self._unrolled = unrolled
        self._occ_index = np.array([ref.index for _, _, ref in occ], dtype=np.int64)
        self._occ_scale = np.array([ref.scale for _, _, ref in occ], dtype=np.float64)
        # Chain-rule matrix: d/dparam_p = sum_m scale_m [occ_index_m == p] d/dangle_m.
        self._chain = np.zeros((len(occ), circuit.n_params))
        self._chain[np.arange(len(occ)), self._occ_index] = self._occ_scale
        self.n_evaluations = 0  # number of circuit simulations performed (for logging)

    @property
    def n_outputs(self) -> int:
        return len(self.observables)

    # ----------------------------------------------------------------- forward
    def _angles(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if x.shape[1] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} inputs, got {x.shape[1]}")
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if w.size != self.n_weights:
            raise ValueError(f"expected {self.n_weights} weights, got {w.size}")
        full = np.concatenate([x, np.broadcast_to(w, (x.shape[0], self.n_weights))], axis=1)
        return full[:, self._occ_index] * self._occ_scale  # (B, M)

    def _expectations(self, angles: np.ndarray, shots: Optional[int], rng: Optional[np.random.Generator]) -> np.ndarray:
        states, _ = run_statevector(self._unrolled, angles if self.n_occurrences else None)
        self.n_evaluations += states.shape[0]
        n = self.n_qubits
        if shots is None:
            return np.stack([expectation(states, o, n) for o in self.observables], axis=1)
        if rng is None:
            raise ValueError("shot-based estimation needs an rng")
        if not all(o.is_diagonal for o in self.observables):
            raise ValueError("shot mode supports Z-type observables only")
        signs = np.stack([o.coeff * z_signs(n, [q for q, _ in o.terms]) for o in self.observables], axis=1)
        probs = np.abs(states) ** 2
        out = np.empty((states.shape[0], len(self.observables)))
        for b in range(states.shape[0]):
            idx = sample_outcomes(probs[b], shots, rng)
            out[b] = signs[idx].mean(axis=0)
        return out

    def forward(
        self, x: np.ndarray, w: np.ndarray, shots: Optional[int] = None, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Expectation values ``(B, K)``."""
        return self._expectations(self._angles(x, w), shots, rng)

    # ---------------------------------------------------------------- gradient
    def jacobian(
        self, x: np.ndarray, w: np.ndarray, shots: Optional[int] = None, rng: Optional[np.random.Generator] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(f, J_x, J_w)`` with shapes ``(B, K)``, ``(B, K, n_inputs)``, ``(B, K, n_weights)``."""
        a = self._angles(x, w)
        bsz, m = a.shape
        k = self.n_outputs
        if m == 0:
            f = self._expectations(a, shots, rng)
            return f, np.zeros((bsz, k, self.n_inputs)), np.zeros((bsz, k, self.n_weights))
        shifted = np.repeat(a[:, None, :], 1 + 2 * m, axis=1)  # (B, 1+2M, M)
        ar = np.arange(m)
        shifted[:, 1 + 2 * ar, ar] += _SHIFT
        shifted[:, 2 + 2 * ar, ar] -= _SHIFT
        e = self._expectations(shifted.reshape(-1, m), shots, rng).reshape(bsz, 1 + 2 * m, k)
        f = e[:, 0]
        d_angle = 0.5 * (e[:, 1::2] - e[:, 2::2])  # (B, M, K)
        d_param = np.einsum("bmk,mp->bkp", d_angle, self._chain)  # (B, K, P)
        return f, d_param[:, :, : self.n_inputs], d_param[:, :, self.n_inputs :]

    def finite_difference_jacobian(self, x: np.ndarray, w: np.ndarray, h: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """Central differences (testing only): ``(J_x, J_w)``."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        w = np.asarray(w, dtype=np.float64)
        jx = np.zeros((x.shape[0], self.n_outputs, self.n_inputs))
        jw = np.zeros((x.shape[0], self.n_outputs, self.n_weights))
        for i in range(self.n_inputs):
            d = np.zeros_like(x)
            d[:, i] = h
            jx[:, :, i] = (self.forward(x + d, w) - self.forward(x - d, w)) / (2 * h)
        for j in range(self.n_weights):
            d = np.zeros_like(w)
            d[j] = h
            jw[:, :, j] = (self.forward(x, w + d) - self.forward(x, w - d)) / (2 * h)
        return jx, jw
