"""Entropy and entanglement measures.

Pure states
-----------
For a bipartition ``A | B`` of a pure state, the amplitudes reshaped as a
``2^|A| x 2^|B|`` matrix ``M`` have singular values ``s_i`` equal to the Schmidt
coefficients, and the entanglement entropy is
``S(A) = -sum_i s_i^2 log2 s_i^2``.  We use the SVD rather than
``eigvalsh(rho_A)``: singular values are computed with *absolute* error
``O(u ||M||_2) = O(u)``, whereas forming ``rho_A = M M^dagger`` squares the
condition number and loses half the significant digits of small Schmidt
coefficients (errors ``O(sqrt(u)) ~ 1e-8`` in ``s_i``).

Mixed states
------------
For density matrices we diagonalise the (Hermitian) reduced state with
``eigvalsh``.  Its eigenvalues carry absolute error ``O(n u)``; values below
``1e-12`` are dropped (see :func:`~quantasecure_neurollm.utils.mathx.entropy_from_eigenvalues`),
which perturbs the entropy by at most ``~4e-11`` bits per discarded eigenvalue.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..utils.mathx import entropy_from_eigenvalues
from . import kernels

__all__ = [
    "schmidt_coefficients",
    "entanglement_entropy",
    "entanglement_entropies",
    "von_neumann_entropy",
    "renyi2_entropy",
    "purity",
    "mutual_information",
    "fidelity_pure_mixed",
]


def _bipartition_matrix(psi: np.ndarray, subsystem: Sequence[int], n: int) -> np.ndarray:
    a = list(subsystem)
    b = [q for q in range(n) if q not in a]
    t = np.asarray(psi).reshape((2,) * n)
    t = np.transpose(t, a + b)
    return t.reshape(2 ** len(a), 2 ** len(b))


def schmidt_coefficients(psi: np.ndarray, subsystem: Sequence[int], n: int) -> np.ndarray:
    """Singular values of the ``A|B`` reshaping (sorted descending, normalised)."""
    s = np.linalg.svd(_bipartition_matrix(psi, subsystem, n), compute_uv=False)
    return s / np.linalg.norm(s)


def entanglement_entropy(psi: np.ndarray, subsystem: Sequence[int], n: int) -> float:
    """Von Neumann entanglement entropy ``S(A)`` in bits of a pure state."""
    if len(subsystem) == 0 or len(subsystem) == n:
        return 0.0
    s = schmidt_coefficients(psi, subsystem, n)
    return entropy_from_eigenvalues(s**2)


def entanglement_entropies(psi: np.ndarray, n: int) -> np.ndarray:
    """Entropies of every contiguous cut ``[0..k) | [k..n)`` for ``k = 1..n-1``."""
    return np.array([entanglement_entropy(psi, list(range(k)), n) for k in range(1, n)])


def von_neumann_entropy(rho: np.ndarray) -> float:
    """``S(rho) = -Tr rho log2 rho`` in bits."""
    return entropy_from_eigenvalues(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)))


def renyi2_entropy(rho: np.ndarray) -> float:
    """``S_2(rho) = -log2 Tr rho^2``."""
    return float(-np.log2(max(purity(rho), 1e-300)))


def purity(rho: np.ndarray) -> float:
    """``Tr rho^2 = ||rho||_F^2`` (Hermitian rho), computed without a matrix product."""
    return float(np.real(np.vdot(rho, rho)))


def mutual_information(rho: np.ndarray, a: Sequence[int], b: Sequence[int], n: int) -> float:
    """``I(A:B) = S(A) + S(B) - S(AB)`` in bits."""
    batch = rho[None]
    ra = kernels.partial_trace(batch, list(a), n)[0]
    rb = kernels.partial_trace(batch, list(b), n)[0]
    rab = kernels.partial_trace(batch, list(a) + list(b), n)[0]
    return von_neumann_entropy(ra) + von_neumann_entropy(rb) - von_neumann_entropy(rab)


def fidelity_pure_mixed(psi: np.ndarray, rho: np.ndarray) -> float:
    """``F = <psi| rho |psi>`` (Uhlmann fidelity when one argument is pure)."""
    return float(np.real(np.vdot(psi, rho @ psi)))
