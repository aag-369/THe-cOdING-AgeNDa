"""Low-level, batched linear-algebra kernels for state-vector and density-matrix simulation.

All kernels operate on a *batch* of states so that many differently
parameterised circuits (chunks of a document, parameter-shift evaluations) are
simulated in one vectorised sweep:

* pure states:      ``psi`` of shape ``(B, 2**n)``
* density matrices: ``rho`` of shape ``(B, 2**n, 2**n)``

Gate application
----------------
A k-qubit gate ``U`` is applied by viewing ``psi`` as a rank-``(n+1)`` tensor
``(B, 2, ..., 2)``, moving the target axes to the end, contracting with ``U``
(``psi @ U^T``) and moving the axes back.  Cost: ``O(B * 2^n * 2^k)`` flops and
one temporary copy of the state.  Diagonal gates (``RZ``, ``CZ``, ``RZZ``, ...)
take an ``O(B * 2^n)`` broadcasting fast path with no transpose.

A density matrix is treated as a ``2n``-qubit "vectorised" state: the row
index occupies qubit slots ``0..n-1`` and the column index slots ``n..2n-1``, so
``U rho U^dagger`` is ``U`` on the row slots followed by ``conj(U)`` on the
column slots.

Floating-point error analysis
-----------------------------
Each output amplitude of a dense k-qubit gate is an inner product of length
``d = 2^k``; by the standard bound (Higham, *Accuracy and Stability of
Numerical Algorithms*, Thm 3.5) its computed value satisfies
``|fl(y) - y| <= gamma_d * |U| |x|`` with ``gamma_d = d u / (1 - d u)`` and
``u = 2^-53``.  Because ``U`` is unitary, the 2-norm of the state changes by at
most ``~ (d + 2) u`` per gate (the ``+2`` accounts for the rounding of the
matrix entries themselves).  After ``G`` gates the norm drift is therefore
bounded by ``G (2^k + 2) u`` -- about ``7e-13`` for ``G = 10^3`` two-qubit
gates.  The drivers monitor ``| ||psi|| - 1 |``, re-normalise above a
configurable tolerance and report the observed drift next to this bound.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "MemoryBudget",
    "DEFAULT_BUDGET",
    "apply_gate",
    "apply_diagonal",
    "apply_gate_dm",
    "permutation_indices",
    "apply_kraus_dm",
    "depolarize_dm",
    "amplitude_damp_dm",
    "phase_damp_dm",
    "damp_dm",
    "partial_trace",
    "norm_error_bound",
]


class MemoryBudget:
    """Guards against accidental exponential memory blow-up.

    ``bytes_needed`` is the size of the primary array; kernels allocate up to
    ~2x this again in temporaries (one ``moveaxis`` copy plus the matmul output).
    """

    def __init__(self, warn_bytes: int = 256 * 2**20, max_bytes: int = 4 * 2**30) -> None:
        self.warn_bytes = warn_bytes
        self.max_bytes = max_bytes

    def check(self, what: str, bytes_needed: int, allow_large: bool = False) -> None:
        if bytes_needed > self.max_bytes and not allow_large:
            raise MemoryError(
                f"{what} needs {bytes_needed / 2**30:.2f} GiB (> limit {self.max_bytes / 2**30:.2f} GiB); "
                "use the MPS backend or pass allow_large=True"
            )
        if bytes_needed > self.warn_bytes:
            warnings.warn(
                f"{what} needs {bytes_needed / 2**20:.0f} MiB (+ ~2x temporaries during gate application)",
                ResourceWarning,
                stacklevel=3,
            )


DEFAULT_BUDGET = MemoryBudget()


def _check_qubits(qubits: Sequence[int], n: int) -> None:
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"repeated qubit in {tuple(qubits)}")
    for q in qubits:
        if not 0 <= q < n:
            raise ValueError(f"qubit {q} out of range for {n} qubits")


def apply_gate(psi: np.ndarray, u: np.ndarray, qubits: Sequence[int], n: int) -> np.ndarray:
    """Apply a dense ``k``-qubit gate to a batch of states.

    Parameters
    ----------
    psi:
        ``(B, 2**n)`` complex amplitudes.
    u:
        ``(2**k, 2**k)`` shared matrix, or ``(B, 2**k, 2**k)`` / ``(1, 2**k, 2**k)`` per-row matrices.
    qubits:
        Target qubits; ``qubits[0]`` is the most significant bit of ``u``'s index.
    """
    _check_qubits(qubits, n)
    k = len(qubits)
    b = psi.shape[0]
    t = psi.reshape((b,) + (2,) * n)
    src = [1 + q for q in qubits]
    dst = list(range(n + 1 - k, n + 1))
    t = np.moveaxis(t, src, dst)
    moved_shape = t.shape
    t = t.reshape(b, -1, 2**k)
    if u.ndim == 2:
        t = t @ u.T
    else:
        t = np.matmul(t, np.swapaxes(u, -1, -2))
    t = np.moveaxis(t.reshape(moved_shape), dst, src)
    return np.ascontiguousarray(t).reshape(b, 2**n)


def apply_diagonal(psi: np.ndarray, diag: np.ndarray, qubits: Sequence[int], n: int) -> np.ndarray:
    """Fast path for gates diagonal in the computational basis.

    ``diag`` has shape ``(2**k,)`` or ``(B, 2**k)``.  Cost ``O(B 2^n)``, no transpose.
    """
    _check_qubits(qubits, n)
    k = len(qubits)
    b = psi.shape[0]
    d = diag.reshape((-1,) + (2,) * k)  # (B or 1, 2, ..., 2) with axes in the order of `qubits`
    order = np.argsort(qubits)
    d = np.transpose(d, [0] + [1 + int(i) for i in order])  # sort target axes ascending
    shape = [d.shape[0]] + [1] * n
    for q in qubits:
        shape[1 + q] = 2
    t = psi.reshape((b,) + (2,) * n) * d.reshape(shape)
    return t.reshape(b, 2**n)


@lru_cache(maxsize=4096)
def _permutation_cached(key: bytes, k: int, qubits: tuple[int, ...], n: int) -> Optional[np.ndarray]:
    u = np.frombuffer(key, dtype=np.complex128).reshape(2**k, 2**k)
    if not (np.all((u == 0) | (u == 1)) and np.all(u.sum(axis=0) == 1) and np.all(u.sum(axis=1) == 1)):
        return None
    src_of = np.argmax(u, axis=1)  # (U psi)[out] = psi[src_of[out]] on the k-qubit subspace
    idx = np.arange(2**n, dtype=np.int64)
    shifts = np.array([n - 1 - q for q in qubits], dtype=np.int64)
    sub_out = np.zeros(2**n, dtype=np.int64)
    for i, s in enumerate(shifts):
        sub_out |= ((idx >> s) & 1) << (k - 1 - i)
    sub_in = src_of[sub_out]
    perm = idx.copy()
    for i, s in enumerate(shifts):
        bit = (sub_in >> (k - 1 - i)) & 1
        perm = (perm & ~(1 << s)) | (bit << s)
    return perm


def permutation_indices(u: np.ndarray, qubits: Sequence[int], n: int) -> Optional[np.ndarray]:
    """If ``u`` is a 0/1 permutation matrix, return ``perm`` with ``(U psi)[i] = psi[perm[i]]`` on n qubits."""
    if u.ndim != 2:
        return None
    return _permutation_cached(np.ascontiguousarray(u, dtype=np.complex128).tobytes(), len(qubits), tuple(qubits), n)


def apply_gate_dm(rho: np.ndarray, u: np.ndarray, qubits: Sequence[int], n: int, diagonal: bool = False) -> np.ndarray:
    """``rho -> U rho U^dagger`` for a batch ``(B, 2**n, 2**n)`` in a single pass over ``rho``.

    * permutation gates (X, CNOT, SWAP, Toffoli): one index gather on rows and columns;
    * diagonal gates: one multiplication by ``d_row (x) conj(d_col)``;
    * otherwise the superoperator ``U (x) conj(U)`` is applied to the (row, column)
      axes of the target qubits at once -- equivalent to ``U`` on the rows then
      ``conj(U)`` on the columns, but with one pass instead of two.
    """
    b = rho.shape[0]
    k = len(qubits)
    perm = permutation_indices(u, qubits, n) if not diagonal else None
    if perm is not None:
        return rho[:, perm[:, None], perm[None, :]]
    targets = list(qubits) + [q + n for q in qubits]
    flat = rho.reshape(b, 4**n)
    if diagonal:
        d = np.diagonal(u, axis1=-2, axis2=-1)
        dd = (d[..., :, None] * d.conj()[..., None, :]).reshape(d.shape[:-1] + (4**k,))
        return apply_diagonal(flat, dd, targets, 2 * n).reshape(b, 2**n, 2**n)
    if u.ndim == 2:
        sup = np.kron(u, u.conj())
    else:
        sup = np.einsum("bij,bkl->bikjl", u, u.conj()).reshape(u.shape[0], 4**k, 4**k)
    return apply_gate(flat, sup, targets, 2 * n).reshape(b, 2**n, 2**n)


def apply_kraus_dm(rho: np.ndarray, kraus: Sequence[np.ndarray], qubits: Sequence[int], n: int) -> np.ndarray:
    """General CPTP map ``rho -> sum_m K_m rho K_m^dagger`` on ``qubits``."""
    b = rho.shape[0]
    flat = rho.reshape(b, 4**n)
    out = np.zeros_like(flat)
    for k in kraus:
        tmp = apply_gate(flat, k, qubits, 2 * n)
        out += apply_gate(tmp, k.conj(), [q + n for q in qubits], 2 * n)
    return out.reshape(b, 2**n, 2**n)


def _qubit_view(rho: np.ndarray, q: int, n: int) -> np.ndarray:
    """View ``rho`` as ``(B, L, 2, R, L, 2, R)`` exposing qubit ``q`` on rows and columns."""
    b = rho.shape[0]
    left, right = 2**q, 2 ** (n - q - 1)
    return rho.reshape(b, left, 2, right, left, 2, right)


def depolarize_dm(rho: np.ndarray, p: float, qubits: Sequence[int], n: int, inplace: bool = False) -> np.ndarray:
    """``rho -> (1-p) rho + p (I/2^k) (x) Tr_qubits(rho)`` (closed form, no Kraus loop).

    For one qubit this equals the Kraus form with ``K_0 = sqrt(1 - 3p/4) I``,
    ``K_{1,2,3} = sqrt(p/4) {X, Y, Z}``; for two qubits it is the uniform
    two-qubit depolarising channel.  Implemented on a strided view whose last
    ``2k`` axes are (row bits, column bits) of the target qubits, so the partial
    trace and the update touch only the ``2^k`` diagonal blocks.
    """
    if p <= 0.0:
        return rho
    out = rho if inplace else rho.copy()
    b, k = out.shape[0], len(qubits)
    t = out.reshape((b,) + (2,) * (2 * n))
    src = [1 + q for q in qubits] + [1 + n + q for q in qubits]
    t = np.moveaxis(t, src, list(range(-2 * k, 0)))  # a view: writes go to `out`
    blocks = [(Ellipsis,) + bits + bits for bits in np.ndindex(*([2] * k))]
    reduced = sum(t[idx] for idx in blocks)  # Tr over the target qubits
    t *= 1.0 - p
    for idx in blocks:
        t[idx] += (p / 2**k) * reduced
    return out


def amplitude_damp_dm(rho: np.ndarray, gamma: float, q: int, n: int, inplace: bool = False) -> np.ndarray:
    """Amplitude damping (``|1> -> |0>`` with probability ``gamma``), closed form.

    Kraus: ``K_0 = [[1,0],[0,sqrt(1-g)]]``, ``K_1 = [[0,sqrt(g)],[0,0]]`` giving
    ``rho_00 += g rho_11``, ``rho_11 *= (1-g)``, coherences ``*= sqrt(1-g)``.
    """
    if gamma <= 0.0:
        return rho
    out = rho if inplace else rho.copy()
    v = _qubit_view(out, q, n)
    s = math.sqrt(1.0 - gamma)
    v[:, :, 0, :, :, 0, :] += gamma * v[:, :, 1, :, :, 1, :]
    v[:, :, 1, :, :, 1, :] *= 1.0 - gamma
    v[:, :, 0, :, :, 1, :] *= s
    v[:, :, 1, :, :, 0, :] *= s
    return out


def damp_dm(rho: np.ndarray, gamma: float, lam: float, q: int, n: int, inplace: bool = False) -> np.ndarray:
    """Amplitude damping followed by phase damping on qubit ``q``, fused into one pass.

    Exact composition: populations follow amplitude damping only, coherences are
    scaled by ``sqrt(1 - gamma) * sqrt(1 - lambda)``.
    """
    if gamma <= 0.0 and lam <= 0.0:
        return rho
    out = rho if inplace else rho.copy()
    v = _qubit_view(out, q, n)
    s = math.sqrt(1.0 - gamma) * math.sqrt(1.0 - lam)
    if gamma > 0.0:
        v[:, :, 0, :, :, 0, :] += gamma * v[:, :, 1, :, :, 1, :]
        v[:, :, 1, :, :, 1, :] *= 1.0 - gamma
    v[:, :, 0, :, :, 1, :] *= s
    v[:, :, 1, :, :, 0, :] *= s
    return out


def phase_damp_dm(rho: np.ndarray, lam: float, q: int, n: int, inplace: bool = False) -> np.ndarray:
    """Pure dephasing: coherences of qubit ``q`` scaled by ``sqrt(1 - lambda)``."""
    if lam <= 0.0:
        return rho
    out = rho if inplace else rho.copy()
    v = _qubit_view(out, q, n)
    s = math.sqrt(1.0 - lam)
    v[:, :, 0, :, :, 1, :] *= s
    v[:, :, 1, :, :, 0, :] *= s
    return out


def partial_trace(rho: np.ndarray, keep: Sequence[int], n: int) -> np.ndarray:
    """Reduced density matrix on ``keep`` (in the given order) for a batch of ``rho``."""
    keep = list(keep)
    _check_qubits(keep, n)
    b = rho.shape[0]
    traced = [q for q in range(n) if q not in keep]
    t = rho.reshape((b,) + (2,) * (2 * n))
    perm = [0] + [1 + q for q in keep] + [1 + q for q in traced] + [1 + n + q for q in keep] + [1 + n + q for q in traced]
    t = np.transpose(t, perm)
    dk, dt = 2 ** len(keep), 2 ** len(traced)
    t = t.reshape(b, dk, dt, dk, dt)
    return np.einsum("bitjt->bij", t)


def norm_error_bound(n_gates: int, max_gate_qubits: int = 2) -> float:
    """A-priori bound ``G (2^k + 2) u`` on the 2-norm drift after ``G`` gates (see module doc)."""
    u = np.finfo(np.float64).eps / 2
    return n_gates * (2**max_gate_qubits + 2) * u
