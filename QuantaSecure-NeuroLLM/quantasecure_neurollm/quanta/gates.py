"""Gate library.

Conventions
-----------
* **Qubit order is big-endian**: in an ``n``-qubit basis state ``|b_0 b_1 ... b_{n-1}>``
  qubit 0 is the *most* significant bit of the integer index.  For a k-qubit
  gate acting on ``qubits=(q_0, ..., q_{k-1})`` the first listed qubit is the
  most significant bit of the gate matrix index, so ``CNOT`` on ``(c, t)`` is
  ``|0><0| (x) I + |1><1| (x) X`` with ``c`` the control.
* Rotation gates follow ``R_P(theta) = exp(-i theta P / 2)``.  Every
  parameterised gate in :data:`GATES` whose ``shiftable`` flag is set has a
  generator with exactly two eigenvalues separated by 1, so the two-term
  parameter-shift rule ``df/dtheta = [f(theta + pi/2) - f(theta - pi/2)] / 2`` is
  *exact* (see :mod:`quantasecure_neurollm.quanta.pqc`).
* All matrix constructors are vectorised: given parameter arrays of shape
  ``S`` they return arrays of shape ``S + (d, d)``.  This is what lets the
  simulator run a whole batch of differently-parameterised circuits (e.g. all
  parameter-shift evaluations) in a single sweep.

Precision: matrices are built in complex128.  ``cos``/``sin`` are accurate to
within 1 ulp on IEEE platforms, so ``||U^dagger U - I||_max`` is ``O(u) ~ 1e-16``
for every gate here (checked by :func:`unitarity_error` in the tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "I2",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "SDG",
    "T",
    "TDG",
    "SX",
    "CNOT",
    "CZ",
    "SWAP",
    "TOFFOLI",
    "rx",
    "ry",
    "rz",
    "phase",
    "u3",
    "rzz",
    "rxx",
    "ryy",
    "controlled",
    "GateSpec",
    "GATES",
    "unitarity_error",
    "is_unitary",
]

C128 = np.complex128
_SQ2 = 1.0 / np.sqrt(2.0)

I2 = np.eye(2, dtype=C128)
X = np.array([[0, 1], [1, 0]], dtype=C128)
Y = np.array([[0, -1j], [1j, 0]], dtype=C128)
Z = np.array([[1, 0], [0, -1]], dtype=C128)
H = np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=C128)
S = np.array([[1, 0], [0, 1j]], dtype=C128)
SDG = S.conj().T
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=C128)
TDG = T.conj().T
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=C128)


def controlled(u: np.ndarray, n_controls: int = 1) -> np.ndarray:
    """Return the matrix of ``u`` controlled on ``n_controls`` leading qubits."""
    d = u.shape[-1]
    total = d * (2**n_controls)
    out = np.eye(total, dtype=C128)
    out[total - d :, total - d :] = u
    return out


CNOT = controlled(X)
CZ = np.diag(np.array([1, 1, 1, -1], dtype=C128))
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=C128)
TOFFOLI = controlled(X, 2)


def _as_angles(theta: np.ndarray | float) -> np.ndarray:
    return np.asarray(theta, dtype=np.float64)


def rx(theta: np.ndarray | float) -> np.ndarray:
    t = _as_angles(theta)
    c, s = np.cos(t / 2), np.sin(t / 2)
    out = np.empty(t.shape + (2, 2), dtype=C128)
    out[..., 0, 0] = c
    out[..., 0, 1] = -1j * s
    out[..., 1, 0] = -1j * s
    out[..., 1, 1] = c
    return out


def ry(theta: np.ndarray | float) -> np.ndarray:
    t = _as_angles(theta)
    c, s = np.cos(t / 2), np.sin(t / 2)
    out = np.empty(t.shape + (2, 2), dtype=C128)
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    return out


def rz(theta: np.ndarray | float) -> np.ndarray:
    t = _as_angles(theta)
    out = np.zeros(t.shape + (2, 2), dtype=C128)
    out[..., 0, 0] = np.exp(-0.5j * t)
    out[..., 1, 1] = np.exp(0.5j * t)
    return out


def phase(theta: np.ndarray | float) -> np.ndarray:
    """``diag(1, e^{i theta})`` (equals ``RZ`` up to a global phase)."""
    t = _as_angles(theta)
    out = np.zeros(t.shape + (2, 2), dtype=C128)
    out[..., 0, 0] = 1.0
    out[..., 1, 1] = np.exp(1j * t)
    return out


def u3(theta: np.ndarray | float, phi: np.ndarray | float, lam: np.ndarray | float) -> np.ndarray:
    """General single-qubit unitary ``RZ(phi) RY(theta) RZ(lam)`` up to global phase."""
    t, p, l = np.broadcast_arrays(_as_angles(theta), _as_angles(phi), _as_angles(lam))
    c, s = np.cos(t / 2), np.sin(t / 2)
    out = np.empty(t.shape + (2, 2), dtype=C128)
    out[..., 0, 0] = c
    out[..., 0, 1] = -np.exp(1j * l) * s
    out[..., 1, 0] = np.exp(1j * p) * s
    out[..., 1, 1] = np.exp(1j * (p + l)) * c
    return out


def rzz(theta: np.ndarray | float) -> np.ndarray:
    """``exp(-i theta Z(x)Z / 2)`` -- the IQP / ZZ-feature-map entangler."""
    t = _as_angles(theta)
    out = np.zeros(t.shape + (4, 4), dtype=C128)
    a, b = np.exp(-0.5j * t), np.exp(0.5j * t)
    out[..., 0, 0] = a
    out[..., 1, 1] = b
    out[..., 2, 2] = b
    out[..., 3, 3] = a
    return out


def _pauli_pair_rotation(p: np.ndarray, theta: np.ndarray | float) -> np.ndarray:
    t = _as_angles(theta)
    pp = np.kron(p, p)
    c, s = np.cos(t / 2)[..., None, None], np.sin(t / 2)[..., None, None]
    return c * np.eye(4, dtype=C128) - 1j * s * pp


def rxx(theta: np.ndarray | float) -> np.ndarray:
    return _pauli_pair_rotation(X, theta)


def ryy(theta: np.ndarray | float) -> np.ndarray:
    return _pauli_pair_rotation(Y, theta)


@dataclass(frozen=True)
class GateSpec:
    """Static description of a named gate."""

    name: str
    n_qubits: int
    n_params: int
    matrix: Callable[..., np.ndarray]
    diagonal: bool = False  # diagonal in the computational basis (enables a fast path)
    shiftable: bool = False  # exact two-term parameter-shift rule applies to every parameter


def _const(m: np.ndarray) -> Callable[..., np.ndarray]:
    m = m.copy()
    m.setflags(write=False)
    return lambda: m


GATES: dict[str, GateSpec] = {
    "id": GateSpec("id", 1, 0, _const(I2), diagonal=True),
    "x": GateSpec("x", 1, 0, _const(X)),
    "y": GateSpec("y", 1, 0, _const(Y)),
    "z": GateSpec("z", 1, 0, _const(Z), diagonal=True),
    "h": GateSpec("h", 1, 0, _const(H)),
    "s": GateSpec("s", 1, 0, _const(S), diagonal=True),
    "sdg": GateSpec("sdg", 1, 0, _const(SDG), diagonal=True),
    "t": GateSpec("t", 1, 0, _const(T), diagonal=True),
    "tdg": GateSpec("tdg", 1, 0, _const(TDG), diagonal=True),
    "sx": GateSpec("sx", 1, 0, _const(SX)),
    "rx": GateSpec("rx", 1, 1, rx, shiftable=True),
    "ry": GateSpec("ry", 1, 1, ry, shiftable=True),
    "rz": GateSpec("rz", 1, 1, rz, diagonal=True, shiftable=True),
    "phase": GateSpec("phase", 1, 1, phase, diagonal=True, shiftable=True),
    "u3": GateSpec("u3", 1, 3, u3, shiftable=True),
    "cx": GateSpec("cx", 2, 0, _const(CNOT)),
    "cz": GateSpec("cz", 2, 0, _const(CZ), diagonal=True),
    "swap": GateSpec("swap", 2, 0, _const(SWAP)),
    "rzz": GateSpec("rzz", 2, 1, rzz, diagonal=True, shiftable=True),
    "rxx": GateSpec("rxx", 2, 1, rxx, shiftable=True),
    "ryy": GateSpec("ryy", 2, 1, ryy, shiftable=True),
    "ccx": GateSpec("ccx", 3, 0, _const(TOFFOLI)),
}
GATES["cnot"] = GATES["cx"]
GATES["toffoli"] = GATES["ccx"]


def unitarity_error(u: np.ndarray) -> float:
    """``max |U^dagger U - I|`` (batched matrices allowed)."""
    d = u.shape[-1]
    prod = np.swapaxes(u.conj(), -1, -2) @ u
    return float(np.max(np.abs(prod - np.eye(d))))


def is_unitary(u: np.ndarray, atol: float = 1e-10) -> bool:
    return unitarity_error(u) <= atol
