"""Noise channels (Kraus form) and a per-gate noise model for density-matrix simulation.

Each channel is available in two equivalent forms:

* an explicit Kraus list (:func:`depolarizing_kraus`, ...) -- general, used by
  :meth:`DensityMatrix.apply_channel` and by the tests;
* a closed-form kernel (:func:`~quantasecure_neurollm.quanta.kernels.depolarize_dm`, ...)
  that the simulator uses on the hot path because it needs one or two passes
  over ``rho`` instead of ``2 * len(kraus)`` dense gate applications.

The two are checked against each other in the test-suite, and every Kraus set
is checked for trace preservation ``sum_m K_m^dagger K_m = I``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from . import kernels
from .gates import I2, X, Y, Z

__all__ = [
    "depolarizing_kraus",
    "amplitude_damping_kraus",
    "phase_damping_kraus",
    "bit_flip_kraus",
    "phase_flip_kraus",
    "is_trace_preserving",
    "NoiseModel",
]


def _check_prob(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability {p} outside [0, 1]")


def depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Single-qubit ``rho -> (1-p) rho + p I/2``."""
    _check_prob(p)
    return [np.sqrt(1 - 3 * p / 4) * I2, np.sqrt(p / 4) * X, np.sqrt(p / 4) * Y, np.sqrt(p / 4) * Z]


def amplitude_damping_kraus(gamma: float) -> list[np.ndarray]:
    _check_prob(gamma)
    k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    k1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
    return [k0, k1]


def phase_damping_kraus(lam: float) -> list[np.ndarray]:
    _check_prob(lam)
    k0 = np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=np.complex128)
    k1 = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=np.complex128)
    return [k0, k1]


def bit_flip_kraus(p: float) -> list[np.ndarray]:
    _check_prob(p)
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * X]


def phase_flip_kraus(p: float) -> list[np.ndarray]:
    _check_prob(p)
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * Z]


def is_trace_preserving(kraus: Sequence[np.ndarray], atol: float = 1e-12) -> bool:
    d = kraus[0].shape[0]
    s = sum(k.conj().T @ k for k in kraus)
    return bool(np.allclose(s, np.eye(d), atol=atol))


@dataclass(frozen=True)
class NoiseModel:
    """Gate-level Markovian noise applied *after* every gate on the qubits it touched.

    The order is: depolarising (1q or 2q, according to the gate's arity), then
    amplitude damping and phase damping on each touched qubit.  Gates on three
    or more qubits receive independent single-qubit depolarising noise, at the
    two-qubit rate, on every touched qubit.
    """

    depolarizing_1q: float = 0.0
    depolarizing_2q: float = 0.0
    amplitude_damping: float = 0.0
    phase_damping: float = 0.0
    readout_error: float = 0.0

    @classmethod
    def from_config(cls, cfg: Any) -> "NoiseModel":
        if not getattr(cfg, "enabled", True):
            return cls()
        return cls(
            depolarizing_1q=cfg.depolarizing_1q,
            depolarizing_2q=cfg.depolarizing_2q,
            amplitude_damping=cfg.amplitude_damping,
            phase_damping=cfg.phase_damping,
            readout_error=cfg.readout_error,
        )

    @property
    def is_noiseless(self) -> bool:
        return (
            self.depolarizing_1q == 0
            and self.depolarizing_2q == 0
            and self.amplitude_damping == 0
            and self.phase_damping == 0
        )

    def apply_after_gate(self, rho: np.ndarray, qubits: Sequence[int], n: int, inplace: bool = False) -> np.ndarray:
        """Apply the post-gate channels to a batch of density matrices.

        With ``inplace=True`` the channels overwrite ``rho`` (the simulator uses
        this on arrays it owns, avoiding one copy of ``rho`` per channel).
        """
        if not inplace:
            rho = rho.copy()
        if len(qubits) == 1:
            kernels.depolarize_dm(rho, self.depolarizing_1q, qubits, n, inplace=True)
        elif len(qubits) == 2:
            kernels.depolarize_dm(rho, self.depolarizing_2q, qubits, n, inplace=True)
        else:
            for q in qubits:
                kernels.depolarize_dm(rho, self.depolarizing_2q, [q], n, inplace=True)
        for q in qubits:
            kernels.damp_dm(rho, self.amplitude_damping, self.phase_damping, q, n, inplace=True)
        return rho

    def readout_confusion(self) -> np.ndarray:
        """Single-qubit confusion matrix ``M[measured, true]``."""
        e = self.readout_error
        return np.array([[1 - e, e], [e, 1 - e]])
