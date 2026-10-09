"""Observables, exact expectation values and shot-based sampling.

Exact expectations
------------------
For a Pauli string ``P`` the exact value ``<psi|P|psi>`` is computed without
building a ``2^n x 2^n`` matrix: Z-type strings are diagonal, so
``<P> = sum_b |psi_b|^2 (-1)^{parity of the selected bits of b}``; strings
containing X/Y are evaluated as ``<psi| (P|psi>)`` by applying the single-qubit
Paulis with the batched kernel.  Both are ``O(B 2^n)``.

Shot sampling
-------------
:func:`sample_outcomes` draws computational-basis outcomes by inverse-CDF
sampling from the Born distribution.  The estimator of an expectation from
``N`` shots has standard error ``sqrt((1 - <P>^2) / N)``; the parameter-shift
rule applied to shot estimates is unbiased.

Readout error is modelled as an independent symmetric bit flip per qubit
applied *after* sampling, matching the confusion matrix of
:meth:`~quantasecure_neurollm.quanta.noise.NoiseModel.readout_confusion`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..utils.mathx import clip_probabilities
from . import kernels
from .gates import X, Y, Z

__all__ = [
    "PauliString",
    "born_probabilities",
    "expectation",
    "expectation_density",
    "sample_outcomes",
    "apply_readout_error",
    "outcome_counts",
    "marginal_frequencies",
    "marginal_probabilities",
    "parity_correlators",
    "z_signs",
]

_PAULI = {"X": X, "Y": Y, "Z": Z}


@dataclass(frozen=True)
class PauliString:
    """Tensor product of single-qubit Paulis, e.g. ``PauliString.from_dict({0: "Z", 3: "X"})``."""

    terms: tuple[tuple[int, str], ...]
    coeff: float = 1.0

    @classmethod
    def from_dict(cls, d: Mapping[int, str], coeff: float = 1.0) -> "PauliString":
        items = tuple(sorted((int(q), p.upper()) for q, p in d.items() if p.upper() != "I"))
        for _, p in items:
            if p not in _PAULI:
                raise ValueError(f"bad Pauli {p!r}")
        return cls(items, coeff)

    @classmethod
    def from_label(cls, label: str, coeff: float = 1.0) -> "PauliString":
        """``label[q]`` is the Pauli on qubit ``q`` (big-endian), e.g. ``"ZZIX"``."""
        return cls.from_dict({q: c for q, c in enumerate(label)}, coeff)

    @classmethod
    def z(cls, *qubits: int) -> "PauliString":
        return cls.from_dict({q: "Z" for q in qubits})

    @property
    def is_diagonal(self) -> bool:
        return all(p == "Z" for _, p in self.terms)

    def __str__(self) -> str:
        body = " ".join(f"{p}{q}" for q, p in self.terms) or "I"
        return body if self.coeff == 1.0 else f"{self.coeff:g}*{body}"


def born_probabilities(psi: np.ndarray) -> np.ndarray:
    """``|psi|^2`` row-wise, clipped and renormalised onto the simplex."""
    return clip_probabilities(np.abs(psi) ** 2)


def z_signs(n: int, qubits: Iterable[int]) -> np.ndarray:
    """``(-1)^{sum of the given qubit bits}`` for every basis index (big-endian)."""
    idx = np.arange(2**n, dtype=np.int64)
    parity = np.zeros(2**n, dtype=np.int64)
    for q in qubits:
        parity ^= (idx >> (n - 1 - q)) & 1
    return 1.0 - 2.0 * parity


def expectation(psi: np.ndarray, pauli: PauliString, n: int) -> np.ndarray:
    """Exact ``<psi|P|psi>`` for a batch ``(B, 2**n)`` -> ``(B,)`` real."""
    psi = np.atleast_2d(psi)
    if pauli.is_diagonal:
        probs = np.abs(psi) ** 2
        return pauli.coeff * probs @ z_signs(n, [q for q, _ in pauli.terms])
    phi = psi
    for q, p in pauli.terms:
        phi = kernels.apply_gate(phi, _PAULI[p], [q], n)
    return pauli.coeff * np.real(np.sum(psi.conj() * phi, axis=1))


def expectation_density(rho: np.ndarray, pauli: PauliString, n: int) -> np.ndarray:
    """Exact ``Tr(rho P)`` for a batch ``(B, d, d)`` -> ``(B,)`` real."""
    rho = rho if rho.ndim == 3 else rho[None]
    if pauli.is_diagonal:
        diag = np.real(np.diagonal(rho, axis1=1, axis2=2))
        return pauli.coeff * diag @ z_signs(n, [q for q, _ in pauli.terms])
    b, d = rho.shape[0], rho.shape[1]
    # Form P rho by applying P to the row index: treat each column of rho as a state vector.
    cols = np.ascontiguousarray(np.swapaxes(rho, 1, 2)).reshape(b * d, d)  # row index is now last
    for q, p in pauli.terms:
        cols = kernels.apply_gate(cols, _PAULI[p], [q], n)
    p_rho = np.swapaxes(cols.reshape(b, d, d), 1, 2)
    return pauli.coeff * np.real(np.trace(p_rho, axis1=1, axis2=2))


def sample_outcomes(probs: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``shots`` basis-state indices from a single Born distribution (inverse CDF)."""
    p = clip_probabilities(np.asarray(probs, dtype=np.float64).ravel())
    cdf = np.cumsum(p)
    cdf[-1] = 1.0  # guard against the final partial sum rounding to 1 - O(u)
    u = rng.random(shots)
    return np.searchsorted(cdf, u, side="right").astype(np.int64)


def apply_readout_error(outcomes: np.ndarray, n: int, p_flip: float, rng: np.random.Generator) -> np.ndarray:
    """Flip each measured bit independently with probability ``p_flip``."""
    if p_flip <= 0:
        return outcomes
    flips = rng.random((outcomes.size, n)) < p_flip
    weights = 1 << np.arange(n - 1, -1, -1, dtype=np.int64)
    return outcomes ^ (flips.astype(np.int64) @ weights)


def outcome_counts(outcomes: np.ndarray, n: int) -> np.ndarray:
    """Histogram of basis-state indices, length ``2**n``."""
    return np.bincount(outcomes, minlength=2**n).astype(np.int64)


def marginal_frequencies(outcomes: np.ndarray, n: int) -> np.ndarray:
    """Empirical ``P(qubit q = 1)`` for each qubit from sampled indices."""
    bits = (np.asarray(outcomes)[:, None] >> (n - 1 - np.arange(n))) & 1
    return bits.mean(axis=0)


def marginal_probabilities(probs: np.ndarray, n: int) -> np.ndarray:
    """Exact ``P(qubit q = 1)`` from a Born distribution (batched on leading axes)."""
    idx = np.arange(2**n, dtype=np.int64)
    bits = ((idx[:, None] >> (n - 1 - np.arange(n))) & 1).astype(np.float64)  # (2^n, n)
    return np.asarray(probs) @ bits


def parity_correlators(probs: np.ndarray, n: int, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Exact ``<Z_a Z_b>`` for each pair from a Born distribution (batched)."""
    cols = np.stack([z_signs(n, pair) for pair in pairs], axis=1)  # (2^n, n_pairs)
    return np.asarray(probs) @ cols
