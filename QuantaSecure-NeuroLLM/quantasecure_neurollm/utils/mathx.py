"""Numerically careful math helpers shared by the quantum and neural modules."""

from __future__ import annotations

import numpy as np

__all__ = [
    "EPS64",
    "stable_softmax",
    "logsumexp",
    "shannon_entropy_bits",
    "entropy_from_eigenvalues",
    "clip_probabilities",
]

#: Machine epsilon of IEEE-754 binary64 (``2^-52``); the unit roundoff is ``u = EPS64 / 2``.
EPS64: float = float(np.finfo(np.float64).eps)


def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """``log(sum(exp(x)))`` without overflow (shift by the max)."""
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return out if keepdims else np.squeeze(out, axis=axis)


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax with max-subtraction; exact up to ``O(u)`` relative error per entry."""
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def clip_probabilities(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Project a nearly-valid probability vector back onto the simplex.

    Born probabilities computed as ``|amp|^2`` are non-negative by construction,
    but probabilities read off the diagonal of a density matrix that went
    through many Kraus maps can be ``-1e-17`` instead of ``0``.  We clip at zero
    and renormalise; the induced change is bounded by the clipped mass, which is
    itself ``O(G * u)`` for ``G`` applied operations.
    """
    q = np.clip(np.real(p), 0.0, None)
    s = np.sum(q, axis=axis, keepdims=True)
    return q / np.where(s > 0, s, 1.0)


def shannon_entropy_bits(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in bits with the convention ``0 log 0 = 0``."""
    q = np.clip(p, 0.0, None)
    safe = np.where(q > 0, q, 1.0)
    return np.sum(-q * np.log2(safe), axis=axis)


def entropy_from_eigenvalues(eigs: np.ndarray, tol: float = 1e-12) -> float:
    """Von Neumann entropy ``-sum l log2 l`` from (possibly noisy) eigenvalues.

    ``numpy.linalg.eigvalsh`` returns eigenvalues of a Hermitian PSD matrix with
    absolute error ``~ ||rho||_2 * n * u``; tiny negative values are therefore
    rounding artefacts.  Eigenvalues below ``tol`` contribute at most
    ``tol * log2(1/tol) ~ 4e-11`` bits each and are dropped, then the remainder
    is renormalised so the result is the entropy of a valid spectrum.
    """
    lam = np.real(np.asarray(eigs, dtype=np.float64))
    lam = lam[lam > tol]
    if lam.size == 0:
        return 0.0
    lam = lam / lam.sum()
    return float(-np.sum(lam * np.log2(lam)))
