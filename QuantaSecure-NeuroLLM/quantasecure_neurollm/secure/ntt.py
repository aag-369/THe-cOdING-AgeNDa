"""Vectorised number-theoretic transforms for ML-KEM and ML-DSA.

Both schemes work in ``R_q = Z_q[X] / (X^256 + 1)``.  The NTT is the
Cooley-Tukey butterfly network of FIPS 203 Alg. 9/10 (ML-KEM) and FIPS 204
Alg. 41/42 (ML-DSA); the two differ only in the modulus, the root of unity and
the number of layers:

========  ==========  =====  ======  ===============================================
scheme    q           zeta   layers  ring splits into
========  ==========  =====  ======  ===============================================
ML-KEM    3329        17     7       128 quadratic factors  X^2 - zeta^(2 brv(i)+1)
ML-DSA    8380417     1753   8       256 linear factors
========  ==========  =====  ======  ===============================================

Each butterfly layer is executed as a single NumPy operation over a
``(polys, blocks, 2, len)`` view, so a vector of ``k`` polynomials is
transformed with ``layers`` vectorised steps.  All intermediate products are
``< q^2 < 2^46`` and therefore exact in int64.
"""

from __future__ import annotations

import numpy as np

__all__ = ["NTT", "bitrev", "schoolbook_negacyclic"]


def bitrev(x: int, bits: int) -> int:
    out = 0
    for _ in range(bits):
        out = (out << 1) | (x & 1)
        x >>= 1
    return out


class NTT:
    """Forward / inverse NTT over ``Z_q[X]/(X^256+1)`` with ``layers`` butterfly layers."""

    def __init__(self, q: int, zeta: int, layers: int, n: int = 256) -> None:
        self.q = q
        self.n = n
        self.layers = layers
        self.zetas = np.array([pow(zeta, bitrev(k, layers), q) for k in range(2**layers)], dtype=np.int64)
        self.scale = pow(2**layers, -1, q)  # 128^-1 = 3303 (ML-KEM), 256^-1 = 8347681 (ML-DSA)

    def forward(self, f: np.ndarray) -> np.ndarray:
        q = self.q
        a = (np.asarray(f, dtype=np.int64) % q).reshape(-1, self.n).copy()
        k, length = 1, self.n // 2
        for _ in range(self.layers):
            nb = self.n // (2 * length)
            z = self.zetas[k : k + nb][None, :, None]
            v = a.reshape(-1, nb, 2, length)
            t = (z * v[:, :, 1, :]) % q
            lo = (v[:, :, 0, :] + t) % q
            hi = (v[:, :, 0, :] - t) % q
            v[:, :, 0, :] = lo
            v[:, :, 1, :] = hi
            k += nb
            length //= 2
        return a.reshape(np.shape(f))

    def inverse(self, f: np.ndarray) -> np.ndarray:
        q = self.q
        a = (np.asarray(f, dtype=np.int64) % q).reshape(-1, self.n).copy()
        length = self.n >> self.layers
        k = 2**self.layers - 1
        for _ in range(self.layers):
            nb = self.n // (2 * length)
            z = self.zetas[k - np.arange(nb)][None, :, None]
            v = a.reshape(-1, nb, 2, length)
            t = v[:, :, 0, :].copy()
            u = v[:, :, 1, :]
            v[:, :, 0, :] = (t + u) % q
            v[:, :, 1, :] = (z * (u - t)) % q
            k -= nb
            length *= 2
        return (a * self.scale % q).reshape(np.shape(f))


def schoolbook_negacyclic(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Reference ``a * b mod (X^n + 1, q)`` in O(n^2) (testing only)."""
    n = len(a)
    full = np.zeros(2 * n, dtype=object)
    for i in range(n):
        full[i : i + n] += int(a[i]) * np.asarray(b, dtype=object)
    res = (full[:n] - full[n:]) % q
    return res.astype(np.int64)
