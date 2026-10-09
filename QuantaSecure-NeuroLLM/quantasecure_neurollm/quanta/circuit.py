"""Symbolic circuits, parameter binding and batched simulation drivers.

A :class:`QuantumCircuit` is an ordered list of :class:`Operation` objects.
Angles are either Python floats or :class:`ParamRef` symbols
(``angle = scale * params[..., index]``), so a single circuit object can be
evaluated for a whole batch of parameter vectors at once::

    qc = QuantumCircuit(2, n_params=1)
    qc.h(0).cx(0, 1).ry(1, qc.p(0))
    states, report = run_statevector(qc, params=np.linspace(0, np.pi, 16)[:, None])  # (16, 4)

The drivers :func:`run_statevector` and :func:`run_density` are the only places
where circuits meet the numerical kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from . import kernels
from .gates import GATES, GateSpec, unitarity_error
from .noise import NoiseModel

__all__ = [
    "ParamRef",
    "Operation",
    "BoundOp",
    "QuantumCircuit",
    "run_statevector",
    "run_density",
    "SimulationReport",
    "iter_parameter_occurrences",
]


@dataclass(frozen=True)
class ParamRef:
    """Symbolic angle ``scale * params[..., index]``."""

    index: int
    scale: float = 1.0

    def resolve(self, params: np.ndarray) -> np.ndarray:
        return self.scale * params[..., self.index]


Angle = Union[float, ParamRef]


@dataclass(frozen=True)
class Operation:
    """A gate application.  ``matrix`` is used only for ``name == "unitary"``."""

    name: str
    qubits: tuple[int, ...]
    params: tuple[Angle, ...] = ()
    matrix: Optional[np.ndarray] = field(default=None, compare=False)

    @property
    def spec(self) -> Optional[GateSpec]:
        return GATES.get(self.name)

    @property
    def is_parameterized(self) -> bool:
        return any(isinstance(p, ParamRef) for p in self.params)


@dataclass
class BoundOp:
    """An operation with its numerical matrix: ``(d, d)`` shared or ``(B, d, d)`` per batch row."""

    name: str
    qubits: tuple[int, ...]
    matrix: np.ndarray
    diagonal: bool


@dataclass
class SimulationReport:
    """Diagnostics returned by :func:`run_statevector`."""

    n_gates: int
    max_norm_drift: float  # observed max_b | ||psi_b|| - 1 | before any re-normalisation
    renormalized: bool
    error_bound: float  # a-priori bound G (2^k + 2) u from kernels.norm_error_bound


class QuantumCircuit:
    """Ordered list of gates on ``n_qubits`` with ``n_params`` symbolic parameters."""

    def __init__(self, n_qubits: int, n_params: int = 0, name: str = "circuit") -> None:
        if n_qubits < 1:
            raise ValueError("need at least one qubit")
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.name = name
        self.ops: list[Operation] = []

    # ------------------------------------------------------------------ building
    def p(self, index: int, scale: float = 1.0) -> ParamRef:
        """Reference to parameter ``index`` (optionally scaled)."""
        if not 0 <= index < self.n_params:
            raise IndexError(f"parameter {index} out of range ({self.n_params} declared)")
        return ParamRef(index, scale)

    def append(self, name: str, qubits: Sequence[int], *params: Angle) -> "QuantumCircuit":
        spec = GATES.get(name)
        if spec is None:
            raise KeyError(f"unknown gate {name!r}")
        qubits = tuple(int(q) for q in qubits)
        if len(qubits) != spec.n_qubits:
            raise ValueError(f"{name} acts on {spec.n_qubits} qubits, got {qubits}")
        if len(params) != spec.n_params:
            raise ValueError(f"{name} takes {spec.n_params} parameters, got {len(params)}")
        self._check_qubits(qubits)
        for prm in params:
            if isinstance(prm, ParamRef) and not 0 <= prm.index < self.n_params:
                raise IndexError(f"parameter index {prm.index} out of range")
        self.ops.append(Operation(spec.name, qubits, tuple(params)))
        return self

    def unitary(self, matrix: np.ndarray, qubits: Sequence[int], check: bool = True) -> "QuantumCircuit":
        """Append an arbitrary (fixed) unitary on ``qubits``."""
        qubits = tuple(int(q) for q in qubits)
        m = np.asarray(matrix, dtype=np.complex128)
        if m.shape != (2 ** len(qubits),) * 2:
            raise ValueError(f"matrix shape {m.shape} does not match {len(qubits)} qubits")
        if check and unitarity_error(m) > 1e-9:
            raise ValueError("matrix is not unitary")
        self._check_qubits(qubits)
        m = m.copy()
        m.setflags(write=False)
        self.ops.append(Operation("unitary", qubits, (), m))
        return self

    def _check_qubits(self, qubits: Sequence[int]) -> None:
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"repeated qubit in {tuple(qubits)}")
        for q in qubits:
            if not 0 <= q < self.n_qubits:
                raise ValueError(f"qubit {q} out of range")

    # Named-gate conveniences (each returns self so calls chain).
    def h(self, q: int) -> "QuantumCircuit":
        return self.append("h", [q])

    def x(self, q: int) -> "QuantumCircuit":
        return self.append("x", [q])

    def y(self, q: int) -> "QuantumCircuit":
        return self.append("y", [q])

    def z(self, q: int) -> "QuantumCircuit":
        return self.append("z", [q])

    def s(self, q: int) -> "QuantumCircuit":
        return self.append("s", [q])

    def sdg(self, q: int) -> "QuantumCircuit":
        return self.append("sdg", [q])

    def t(self, q: int) -> "QuantumCircuit":
        return self.append("t", [q])

    def tdg(self, q: int) -> "QuantumCircuit":
        return self.append("tdg", [q])

    def rx(self, q: int, theta: Angle) -> "QuantumCircuit":
        return self.append("rx", [q], theta)

    def ry(self, q: int, theta: Angle) -> "QuantumCircuit":
        return self.append("ry", [q], theta)

    def rz(self, q: int, theta: Angle) -> "QuantumCircuit":
        return self.append("rz", [q], theta)

    def phase(self, q: int, theta: Angle) -> "QuantumCircuit":
        return self.append("phase", [q], theta)

    def u3(self, q: int, theta: Angle, phi: Angle, lam: Angle) -> "QuantumCircuit":
        return self.append("u3", [q], theta, phi, lam)

    def cx(self, control: int, target: int) -> "QuantumCircuit":
        return self.append("cx", [control, target])

    cnot = cx

    def cz(self, a: int, b: int) -> "QuantumCircuit":
        return self.append("cz", [a, b])

    def swap(self, a: int, b: int) -> "QuantumCircuit":
        return self.append("swap", [a, b])

    def rzz(self, a: int, b: int, theta: Angle) -> "QuantumCircuit":
        return self.append("rzz", [a, b], theta)

    def rxx(self, a: int, b: int, theta: Angle) -> "QuantumCircuit":
        return self.append("rxx", [a, b], theta)

    def ryy(self, a: int, b: int, theta: Angle) -> "QuantumCircuit":
        return self.append("ryy", [a, b], theta)

    def ccx(self, c1: int, c2: int, target: int) -> "QuantumCircuit":
        return self.append("ccx", [c1, c2, target])

    toffoli = ccx

    # ------------------------------------------------------------- inspection
    def __len__(self) -> int:
        return len(self.ops)

    def count_ops(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for op in self.ops:
            counts[op.name] = counts.get(op.name, 0) + 1
        return counts

    def depth(self) -> int:
        level = [0] * self.n_qubits
        for op in self.ops:
            d = max(level[q] for q in op.qubits) + 1
            for q in op.qubits:
                level[q] = d
        return max(level) if level else 0

    def max_gate_qubits(self) -> int:
        return max((len(op.qubits) for op in self.ops), default=1)

    def draw(self) -> str:
        """Plain-text listing (one gate per line)."""
        lines = [f"{self.name}: {self.n_qubits} qubits, {len(self.ops)} gates, depth {self.depth()}"]
        for i, op in enumerate(self.ops):
            ps = ", ".join(f"p[{a.index}]*{a.scale:g}" if isinstance(a, ParamRef) else f"{a:.4f}" for a in op.params)
            lines.append(f"  {i:4d}  {op.name:<7} q{list(op.qubits)}" + (f"  ({ps})" if ps else ""))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"QuantumCircuit(n_qubits={self.n_qubits}, n_params={self.n_params}, gates={len(self.ops)})"

    # ------------------------------------------------------------ transforms
    def copy(self) -> "QuantumCircuit":
        out = QuantumCircuit(self.n_qubits, self.n_params, self.name)
        out.ops = list(self.ops)
        return out

    def compose(self, other: "QuantumCircuit", qubit_map: Optional[Sequence[int]] = None) -> "QuantumCircuit":
        """Append ``other`` (its parameters are shared by index with ``self``)."""
        qmap = list(qubit_map) if qubit_map is not None else list(range(other.n_qubits))
        if other.n_params > self.n_params:
            raise ValueError("composed circuit references more parameters than declared")
        out = self.copy()
        for op in other.ops:
            out.ops.append(Operation(op.name, tuple(qmap[q] for q in op.qubits), op.params, op.matrix))
        return out

    def inverse(self) -> "QuantumCircuit":
        """Adjoint circuit ``C^dagger`` (gate order reversed, each gate inverted)."""
        out = QuantumCircuit(self.n_qubits, self.n_params, self.name + "_dg")

        def neg(a: Angle) -> Angle:
            return ParamRef(a.index, -a.scale) if isinstance(a, ParamRef) else -a

        selfinv = {"id", "x", "y", "z", "h", "cx", "cz", "swap", "ccx"}
        for op in reversed(self.ops):
            if op.name == "unitary":
                assert op.matrix is not None
                out.ops.append(Operation("unitary", op.qubits, (), op.matrix.conj().T.copy()))
            elif op.name in selfinv:
                out.ops.append(op)
            elif op.name in ("s", "t", "sdg", "tdg"):
                inv = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}[op.name]
                out.ops.append(Operation(inv, op.qubits))
            elif op.name == "sx":
                out.ops.append(Operation("unitary", op.qubits, (), GATES["sx"].matrix().conj().T.copy()))
            elif op.name == "u3":
                th, ph, la = op.params
                out.ops.append(Operation("u3", op.qubits, (neg(th), neg(la), neg(ph))))
            else:  # single-parameter rotations: R(theta)^dagger = R(-theta)
                out.ops.append(Operation(op.name, op.qubits, tuple(neg(a) for a in op.params)))
        return out

    # ----------------------------------------------------------------- binding
    def bind(self, params: Optional[np.ndarray] = None) -> tuple[list[BoundOp], int]:
        """Resolve all symbols to matrices.  Returns ``(bound_ops, batch_size)``."""
        batch = 1
        prm: Optional[np.ndarray] = None
        if params is not None:
            prm = np.asarray(params, dtype=np.float64)
            if prm.ndim == 1:
                prm = prm[None, :]
            if prm.shape[-1] != self.n_params:
                raise ValueError(f"expected {self.n_params} parameters, got {prm.shape[-1]}")
            batch = prm.shape[0]
        elif self.n_params and any(op.is_parameterized for op in self.ops):
            raise ValueError("circuit is parameterized; pass params")
        bound: list[BoundOp] = []
        for op in self.ops:
            if op.name == "unitary":
                assert op.matrix is not None
                bound.append(BoundOp(op.name, op.qubits, op.matrix, False))
                continue
            spec = GATES[op.name]
            if spec.n_params == 0:
                bound.append(BoundOp(op.name, op.qubits, spec.matrix(), spec.diagonal))
                continue
            args = []
            for a in op.params:
                if isinstance(a, ParamRef):
                    assert prm is not None
                    args.append(a.resolve(prm))
                else:
                    args.append(np.float64(a))
            bound.append(BoundOp(op.name, op.qubits, spec.matrix(*args), spec.diagonal))
        return bound, batch

    def to_unitary(self, params: Optional[np.ndarray] = None) -> np.ndarray:
        """Dense ``2^n x 2^n`` unitary of the circuit (for testing; n <= 10)."""
        if self.n_qubits > 10:
            raise ValueError("to_unitary is limited to 10 qubits")
        dim = 2**self.n_qubits
        prm = None if params is None else np.asarray(params, dtype=np.float64).reshape(1, -1)
        cols, _ = run_statevector(self, params=prm, initial=np.eye(dim, dtype=np.complex128), renormalize_tol=None)
        return cols.T


def _initial_states(n: int, batch: int, initial: Optional[np.ndarray]) -> np.ndarray:
    dim = 2**n
    if initial is None:
        psi = np.zeros((batch, dim), dtype=np.complex128)
        psi[:, 0] = 1.0
        return psi
    psi = np.asarray(initial, dtype=np.complex128)
    if psi.ndim == 1:
        psi = psi[None, :]
    if psi.shape[-1] != dim:
        raise ValueError("initial state has wrong dimension")
    if psi.shape[0] == 1 and batch > 1:
        psi = np.repeat(psi, batch, axis=0)
    elif batch > 1 and psi.shape[0] != batch:
        raise ValueError("initial batch does not match parameter batch")
    return psi.copy()


def run_statevector(
    circuit: QuantumCircuit,
    params: Optional[np.ndarray] = None,
    initial: Optional[np.ndarray] = None,
    renormalize_tol: Optional[float] = 1e-10,
    budget: kernels.MemoryBudget = kernels.DEFAULT_BUDGET,
    allow_large: bool = False,
) -> tuple[np.ndarray, SimulationReport]:
    """Simulate ``circuit`` on a batch of pure states.

    Returns ``(states, report)`` with ``states`` of shape ``(B, 2**n)``.  If the
    norm drift of any state exceeds ``renormalize_tol`` it is re-normalised (the
    drift observed *before* re-normalisation is reported).  A shared parameter
    row (``params`` of batch 1) broadcasts against a batch of initial states.
    """
    n = circuit.n_qubits
    bound, batch = circuit.bind(params)
    if initial is not None and np.asarray(initial).ndim == 2:
        batch = max(batch, np.asarray(initial).shape[0])
    budget.check(f"state vector batch ({batch} x 2^{n})", batch * (2**n) * 16, allow_large)
    psi = _initial_states(n, batch, initial)
    for op in bound:
        if op.diagonal:
            psi = kernels.apply_diagonal(psi, np.diagonal(op.matrix, axis1=-2, axis2=-1), op.qubits, n)
            continue
        perm = kernels.permutation_indices(op.matrix, op.qubits, n)
        psi = psi[:, perm] if perm is not None else kernels.apply_gate(psi, op.matrix, op.qubits, n)
    norms = np.linalg.norm(psi, axis=1)
    drift = float(np.max(np.abs(norms - 1.0))) if psi.size else 0.0
    renorm = False
    if renormalize_tol is not None and drift > renormalize_tol:
        psi = psi / norms[:, None]
        renorm = True
    report = SimulationReport(len(bound), drift, renorm, kernels.norm_error_bound(len(bound), circuit.max_gate_qubits()))
    return psi, report


def run_density(
    circuit: QuantumCircuit,
    params: Optional[np.ndarray] = None,
    noise: Optional[NoiseModel] = None,
    initial_rho: Optional[np.ndarray] = None,
    initial_state: Optional[np.ndarray] = None,
    budget: kernels.MemoryBudget = kernels.DEFAULT_BUDGET,
    allow_large: bool = False,
) -> np.ndarray:
    """Simulate ``circuit`` on a batch of density matrices with optional gate noise.

    The input is ``initial_rho`` (``(d, d)`` or ``(B, d, d)``), or pure
    ``initial_state`` (``(d,)`` or ``(B, d)``), or neither (``|0..0><0..0|``).
    Returns ``(B, d, d)``.
    """
    n = circuit.n_qubits
    dim = 2**n
    bound, batch = circuit.bind(params)
    budget.check(f"density-matrix batch ({batch} x 4^{n})", batch * dim * dim * 16, allow_large)
    if initial_rho is not None:
        rho = np.asarray(initial_rho, dtype=np.complex128)
        rho = rho if rho.ndim == 3 else rho[None]
        if rho.shape[0] == 1 and batch > 1:
            rho = np.repeat(rho, batch, axis=0)
        rho = rho.copy()
    else:
        psi = _initial_states(n, batch, initial_state)
        rho = psi[:, :, None] * psi[:, None, :].conj()
    noisy = noise is not None and not noise.is_noiseless
    for op in bound:
        rho = kernels.apply_gate_dm(rho, op.matrix, op.qubits, n, diagonal=op.diagonal)
        if noisy:
            assert noise is not None
            rho = noise.apply_after_gate(rho, op.qubits, n, inplace=True)  # rho is a fresh array we own
    # Enforce exact Hermiticity (removes the O(u) anti-Hermitian rounding residue).
    return 0.5 * (rho + np.conj(np.swapaxes(rho, -1, -2)))


def iter_parameter_occurrences(circuit: QuantumCircuit) -> Iterable[tuple[int, int, ParamRef]]:
    """Yield ``(op_index, slot, ref)`` for every symbolic angle in the circuit."""
    for i, op in enumerate(circuit.ops):
        for j, a in enumerate(op.params):
            if isinstance(a, ParamRef):
                yield i, j, a
