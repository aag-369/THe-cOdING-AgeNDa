"""Matrix-product-state (MPS) backend -- the "limited tensor-network mode".

An ``n``-qubit state is stored as site tensors ``A[i]`` of shape
``(chi_{i}, 2, chi_{i+1})`` with ``chi_0 = chi_n = 1``.  Memory is
``O(n chi^2)`` instead of ``O(2^n)``, so weakly-entangled states on 40+ qubits
are cheap.  Supported operations:

* single-qubit gates (local contraction, exact);
* two-qubit gates on any pair (non-adjacent pairs are routed with SWAPs),
  followed by an SVD that truncates the bond to ``max_bond`` and drops singular
  values whose discarded *weight* is below ``cutoff``;
* Toffoli via its standard 6-CNOT Clifford+T decomposition;
* exact Pauli-string expectations by transfer-matrix contraction, ``O(n chi^3)``;
* bond (entanglement) entropies via mixed-canonical form;
* perfect sampling of computational-basis outcomes (sequential conditional
  sampling on the right-canonical form), vectorised over shots.

Truncation error
----------------
Each truncation discards a normalised weight ``eps_k = sum_{i > chi} s_i^2 / sum_i s_i^2``
and re-normalises.  The fidelity with the untruncated state then satisfies
``F >= prod_k (1 - eps_k) >= 1 - sum_k eps_k`` to first order, and
:attr:`MPS.truncation_error` accumulates ``sum_k eps_k`` so users can bound the
deviation of any bounded observable by ``2 sqrt(sum_k eps_k)`` (trace-distance
bound).  With no truncation the backend is exact up to ``O(G u)`` rounding.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..utils.mathx import entropy_from_eigenvalues
from .circuit import QuantumCircuit
from .gates import GATES, SWAP
from .measurement import PauliString

__all__ = ["MPS", "toffoli_decomposition"]

_PAULI = {"X": GATES["x"].matrix(), "Y": GATES["y"].matrix(), "Z": GATES["z"].matrix()}


def toffoli_decomposition(a: int, b: int, c: int) -> list[tuple[str, tuple[int, ...]]]:
    """Toffoli(a, b -> c) as 1- and 2-qubit gates (Nielsen & Chuang, Fig. 4.9)."""
    return [
        ("h", (c,)), ("cx", (b, c)), ("tdg", (c,)), ("cx", (a, c)), ("t", (c,)), ("cx", (b, c)),
        ("tdg", (c,)), ("cx", (a, c)), ("t", (b,)), ("t", (c,)), ("h", (c,)), ("cx", (a, b)),
        ("t", (a,)), ("tdg", (b,)), ("cx", (a, b)),
    ]  # fmt: skip


class MPS:
    """Open-boundary matrix product state with SVD truncation."""

    def __init__(self, n_qubits: int, max_bond: int = 64, cutoff: float = 1e-12) -> None:
        if n_qubits < 1:
            raise ValueError("need at least one qubit")
        self.n_qubits = n_qubits
        self.max_bond = max_bond
        self.cutoff = cutoff
        self.tensors: list[np.ndarray] = []
        for _ in range(n_qubits):
            a = np.zeros((1, 2, 1), dtype=np.complex128)
            a[0, 0, 0] = 1.0
            self.tensors.append(a)
        self.truncation_error = 0.0
        self.n_truncations = 0

    # ------------------------------------------------------------- creation
    @classmethod
    def from_statevector(cls, psi: np.ndarray, n_qubits: int, max_bond: int = 64, cutoff: float = 1e-12) -> "MPS":
        """MPS of a dense state by successive SVDs (exact unless truncated)."""
        mps = cls(n_qubits, max_bond, cutoff)
        rest = np.asarray(psi, dtype=np.complex128).reshape(1, -1)
        chi = 1
        for i in range(n_qubits - 1):
            mat = rest.reshape(chi * 2, -1)
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            keep = mps._choose_rank(s)
            mps._record_truncation(s, keep)
            u, s, vh = u[:, :keep], s[:keep], vh[:keep]
            mps.tensors[i] = u.reshape(chi, 2, keep)
            rest = s[:, None] * vh
            chi = keep
        mps.tensors[-1] = rest.reshape(chi, 2, 1)
        nrm = mps.norm()
        if nrm > 0:
            mps.tensors[-1] = mps.tensors[-1] / nrm
        return mps

    def copy(self) -> "MPS":
        out = MPS(self.n_qubits, self.max_bond, self.cutoff)
        out.tensors = [t.copy() for t in self.tensors]
        out.truncation_error = self.truncation_error
        out.n_truncations = self.n_truncations
        return out

    # ------------------------------------------------------------- internals
    def _choose_rank(self, s: np.ndarray) -> int:
        w = s**2
        total = w.sum()
        if total == 0:
            return 1
        # tail[k] = normalised weight discarded if we keep rank k; pick the smallest k with tail <= cutoff.
        tail = np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]]) / total
        rank = int(np.argmax(tail <= self.cutoff))
        return max(1, min(rank, self.max_bond))

    def _record_truncation(self, s: np.ndarray, keep: int) -> None:
        w = s**2
        total = w.sum()
        if total > 0 and keep < len(s):
            eps = float(w[keep:].sum() / total)
            if eps > 0:
                self.truncation_error += eps
                self.n_truncations += 1

    # ------------------------------------------------------------------ gates
    def apply_1q(self, u: np.ndarray, q: int) -> "MPS":
        self.tensors[q] = np.einsum("ij,ljr->lir", u, self.tensors[q])
        return self

    def _apply_adjacent(self, u: np.ndarray, q: int) -> None:
        """Gate ``u`` on sites ``(q, q+1)`` with ``q`` the most significant index of ``u``."""
        a, b = self.tensors[q], self.tensors[q + 1]
        chi_l, chi_r = a.shape[0], b.shape[2]
        theta = np.einsum("lir,rjs->lijs", a, b)
        theta = np.einsum("abij,lijs->labs", u.reshape(2, 2, 2, 2), theta)
        mat = theta.reshape(chi_l * 2, 2 * chi_r)
        uu, s, vh = np.linalg.svd(mat, full_matrices=False)
        keep = self._choose_rank(s)
        self._record_truncation(s, keep)
        s_kept = s[:keep]
        s_kept = s_kept / np.linalg.norm(s_kept) * np.linalg.norm(s)  # keep the global norm unchanged
        self.tensors[q] = uu[:, :keep].reshape(chi_l, 2, keep)
        self.tensors[q + 1] = (s_kept[:, None] * vh[:keep]).reshape(keep, 2, chi_r)

    def apply_2q(self, u: np.ndarray, q1: int, q2: int) -> "MPS":
        """Two-qubit gate on any pair; non-adjacent targets are routed with SWAP gates."""
        if q1 == q2:
            raise ValueError("repeated qubit")
        if q1 > q2:  # reorder so the first index of u is the lower site
            u = SWAP @ u @ SWAP
            q1, q2 = q2, q1
        for k in range(q2 - 1, q1, -1):  # bring q2 down next to q1
            self._apply_adjacent(SWAP, k)
        self._apply_adjacent(u, q1)
        for k in range(q1 + 1, q2):  # and route it back
            self._apply_adjacent(SWAP, k)
        return self

    def apply(self, name: str, qubits: Sequence[int], *params: float) -> "MPS":
        if name in ("ccx", "toffoli"):
            for g, qs in toffoli_decomposition(*qubits):
                self.apply(g, qs)
            return self
        spec = GATES[name]
        u = spec.matrix(*params)
        if spec.n_qubits == 1:
            return self.apply_1q(u, qubits[0])
        if spec.n_qubits == 2:
            return self.apply_2q(u, qubits[0], qubits[1])
        raise NotImplementedError("MPS backend supports 1- and 2-qubit gates (and Toffoli)")

    def run(self, circuit: QuantumCircuit, params: Optional[np.ndarray] = None) -> "MPS":
        """Apply every gate of ``circuit`` (a single parameter vector)."""
        if circuit.n_qubits != self.n_qubits:
            raise ValueError("circuit width does not match state")
        bound, batch = circuit.bind(params)
        if batch != 1:
            raise ValueError("MPS.run takes a single parameter vector")
        for op in bound:
            m = op.matrix[0] if op.matrix.ndim == 3 else op.matrix
            if len(op.qubits) == 1:
                self.apply_1q(m, op.qubits[0])
            elif len(op.qubits) == 2:
                self.apply_2q(m, op.qubits[0], op.qubits[1])
            elif op.name == "ccx":
                self.apply("ccx", op.qubits)
            else:
                raise NotImplementedError(f"{op.name} on {len(op.qubits)} qubits not supported by MPS")
        return self

    # -------------------------------------------------------------- contraction
    def bond_dimensions(self) -> list[int]:
        return [t.shape[2] for t in self.tensors[:-1]]

    def memory_bytes(self) -> int:
        return int(sum(t.nbytes for t in self.tensors))

    def to_statevector(self) -> np.ndarray:
        if self.n_qubits > 22:
            raise ValueError("refusing to densify more than 22 qubits")
        out = self.tensors[0].reshape(2, -1)
        for t in self.tensors[1:]:
            out = np.einsum("pl,lir->pir", out, t).reshape(-1, t.shape[2])
        return out.reshape(-1)

    def norm(self) -> float:
        env = np.ones((1, 1), dtype=np.complex128)
        for a in self.tensors:
            env = np.einsum("ab,aic,bid->cd", env, a.conj(), a)
        return float(np.sqrt(abs(env[0, 0])))

    def expectation(self, pauli: PauliString | str) -> float:
        """Exact ``<psi|P|psi> / <psi|psi>`` by transfer-matrix contraction."""
        p = PauliString.from_label(pauli) if isinstance(pauli, str) else pauli
        ops = dict(p.terms)
        env = np.ones((1, 1), dtype=np.complex128)
        for i, a in enumerate(self.tensors):
            ket = np.einsum("ij,ljr->lir", _PAULI[ops[i]], a) if i in ops else a
            env = np.einsum("ab,aic,bid->cd", env, a.conj(), ket)
        return float(p.coeff * np.real(env[0, 0]) / self.norm() ** 2)

    # ------------------------------------------------------------ canonical forms
    def _right_canonical(self) -> list[np.ndarray]:
        """Copy of the tensors with sites 1..n-1 right-orthonormal (centre at site 0)."""
        ts = [t.copy() for t in self.tensors]
        for i in range(self.n_qubits - 1, 0, -1):
            chi_l, _, chi_r = ts[i].shape
            mat = ts[i].reshape(chi_l, 2 * chi_r)
            q, r = np.linalg.qr(mat.T)  # mat = r^T q^T  (an LQ decomposition)
            k = q.shape[1]
            ts[i] = q.T.reshape(k, 2, chi_r)
            ts[i - 1] = np.einsum("lir,rk->lik", ts[i - 1], r.T)
        return ts

    def bond_entropy(self, cut: int) -> float:
        """Entanglement entropy (bits) between sites ``[0, cut)`` and ``[cut, n)``."""
        if not 0 < cut < self.n_qubits:
            return 0.0
        ts = self._right_canonical()
        for i in range(cut):  # move the orthogonality centre from site 0 to site `cut`
            chi_l, _, chi_r = ts[i].shape
            q, r = np.linalg.qr(ts[i].reshape(chi_l * 2, chi_r))
            k = q.shape[1]
            ts[i] = q.reshape(chi_l, 2, k)
            ts[i + 1] = np.einsum("kr,rjs->kjs", r, ts[i + 1])
        centre = ts[cut]
        s = np.linalg.svd(centre.reshape(centre.shape[0], -1), compute_uv=False)
        return entropy_from_eigenvalues((s / np.linalg.norm(s)) ** 2)

    def sample(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Perfect sampling of basis-state indices (big-endian), vectorised over shots."""
        ts = self._right_canonical()
        n = self.n_qubits
        v = np.ones((shots, 1), dtype=np.complex128)
        bits = np.zeros((shots, n), dtype=np.int64)
        rows = np.arange(shots)
        for i, a in enumerate(ts):
            cand = np.einsum("sl,lir->sir", v, a)  # (shots, 2, chi_r)
            w = np.sum(np.abs(cand) ** 2, axis=2)  # unnormalised conditional weights
            tot = w.sum(axis=1)
            p1 = np.where(tot > 0, w[:, 1] / np.where(tot > 0, tot, 1.0), 0.0)
            b = (rng.random(shots) < p1).astype(np.int64)
            bits[:, i] = b
            v = cand[rows, b] / np.sqrt(np.maximum(w[rows, b], 1e-300))[:, None]
        weights = 1 << np.arange(n - 1, -1, -1, dtype=np.int64)
        return bits @ weights
