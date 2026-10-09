"""ML-KEM (FIPS 203, formerly CRYSTALS-Kyber): Module-LWE key encapsulation.

This is a from-scratch implementation that follows the FIPS 203 algorithms
step by step (algorithm numbers are cited inline).  The development test-suite
cross-checked it byte-for-byte against an independent reference
implementation, and the resulting known-answer digests are pinned in
``tests/test_secure_primitives.py``.

Security sketch
---------------
K-PKE is IND-CPA under the decision Module-LWE assumption in dimension
``256 k`` (``k = 2, 3, 4`` for ML-KEM-512/768/1024, NIST categories 1/3/5).
ML-KEM applies the Fujisaki-Okamoto transform with *implicit rejection*: on
decapsulation the ciphertext is re-encrypted and, if it differs, a
pseudorandom key ``J(z || c)`` is returned instead of an error.  This yields
IND-CCA2 security in the (quantum) random-oracle model.

Side channels: the re-encryption comparison uses ``hmac.compare_digest`` and
the final key is selected with a mask; the NumPy arithmetic itself is not
guaranteed constant-time (see :mod:`.constant_time`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .constant_time import ct_equal, ct_select_bytes
from .ntt import NTT, bitrev
from .rng import RandomSource, default_rng

__all__ = ["MLKEMParams", "ML_KEM_512", "ML_KEM_768", "ML_KEM_1024", "MLKEM", "PARAMS", "MLKEMError"]

Q = 3329
N = 256
_NTT = NTT(Q, 17, 7)
_GAMMAS = np.array([pow(17, 2 * bitrev(i, 7) + 1, Q) for i in range(128)], dtype=np.int64)


class MLKEMError(ValueError):
    pass


@dataclass(frozen=True)
class MLKEMParams:
    name: str
    k: int
    eta1: int
    eta2: int
    du: int
    dv: int

    @property
    def ek_bytes(self) -> int:
        return 384 * self.k + 32

    @property
    def dk_bytes(self) -> int:
        return 768 * self.k + 96

    @property
    def ct_bytes(self) -> int:
        return 32 * (self.du * self.k + self.dv)


ML_KEM_512 = MLKEMParams("ML-KEM-512", 2, 3, 2, 10, 4)
ML_KEM_768 = MLKEMParams("ML-KEM-768", 3, 2, 2, 10, 4)
ML_KEM_1024 = MLKEMParams("ML-KEM-1024", 4, 2, 2, 11, 5)
PARAMS = {p.name: p for p in (ML_KEM_512, ML_KEM_768, ML_KEM_1024)}


# ------------------------------------------------------------- hash functions
def _G(x: bytes) -> tuple[bytes, bytes]:
    h = hashlib.sha3_512(x).digest()
    return h[:32], h[32:]


def _H(x: bytes) -> bytes:
    return hashlib.sha3_256(x).digest()


def _J(x: bytes) -> bytes:
    return hashlib.shake_256(x).digest(32)


def _prf(eta: int, s: bytes, b: int) -> bytes:
    return hashlib.shake_256(s + bytes([b])).digest(64 * eta)


# ------------------------------------------------------------- encodings
def _byte_encode(f: np.ndarray, d: int) -> bytes:
    """Alg. 5 ByteEncode_d: little-endian bit packing of d-bit coefficients."""
    f = np.asarray(f, dtype=np.int64).reshape(-1)
    bits = ((f[:, None] >> np.arange(d)) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1), bitorder="little").tobytes()


def _byte_decode(b: bytes, d: int) -> np.ndarray:
    """Alg. 6 ByteDecode_d (reduces mod q when d = 12)."""
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder="little").astype(np.int64)
    f = (bits.reshape(-1, d) << np.arange(d)).sum(axis=1)
    return f % Q if d == 12 else f


def _compress(x: np.ndarray, d: int) -> np.ndarray:
    """round(2^d / q * x) mod 2^d, exact integer arithmetic."""
    return (((np.asarray(x, dtype=np.int64) << (d + 1)) + Q) // (2 * Q)) & ((1 << d) - 1)


def _decompress(y: np.ndarray, d: int) -> np.ndarray:
    """round(q / 2^d * y)."""
    return (np.asarray(y, dtype=np.int64) * Q + (1 << (d - 1))) >> d


# ------------------------------------------------------------- sampling
def _sample_ntt(seed: bytes) -> np.ndarray:
    """Alg. 7 SampleNTT: rejection sampling of a uniform NTT-domain polynomial from SHAKE128."""
    length = 840
    while True:
        buf = np.frombuffer(hashlib.shake_128(seed).digest(length), dtype=np.uint8).astype(np.int64)
        b = buf[: (len(buf) // 3) * 3].reshape(-1, 3)
        d1 = b[:, 0] + 256 * (b[:, 1] & 15)
        d2 = (b[:, 1] >> 4) + 16 * b[:, 2]
        cand = np.stack([d1, d2], axis=1).reshape(-1)
        ok = cand[cand < Q]
        if len(ok) >= N:
            return ok[:N]
        length += 168


def _cbd(eta: int, data: bytes) -> np.ndarray:
    """Alg. 8 SamplePolyCBD_eta: centred binomial distribution, returned mod q."""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little").astype(np.int64)
    bits = bits.reshape(N, 2, eta).sum(axis=2)
    return (bits[:, 0] - bits[:, 1]) % Q


def _multiply_ntts(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Alg. 11/12 MultiplyNTTs: 128 base-case products in Z_q[X]/(X^2 - gamma_i)."""
    a0, a1 = f[..., 0::2], f[..., 1::2]
    b0, b1 = g[..., 0::2], g[..., 1::2]
    c0 = (a0 * b0 + (a1 * b1 % Q) * _GAMMAS) % Q
    c1 = (a0 * b1 + a1 * b0) % Q
    out = np.empty(np.broadcast_shapes(f.shape, g.shape), dtype=np.int64)
    out[..., 0::2] = c0
    out[..., 1::2] = c1
    return out


def _matrix(rho: bytes, k: int, transpose: bool = False) -> np.ndarray:
    """A_hat[i][j] = SampleNTT(rho || j || i) (Alg. 13 line 4); optionally its transpose."""
    a = np.empty((k, k, N), dtype=np.int64)
    for i in range(k):
        for j in range(k):
            a[i, j] = _sample_ntt(rho + bytes([j, i]))
    return a.transpose(1, 0, 2) if transpose else a


def _matvec(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _multiply_ntts(a, v[None, :, :]).sum(axis=1) % Q


class MLKEM:
    """ML-KEM key generation, encapsulation and decapsulation for one parameter set."""

    def __init__(self, params: MLKEMParams | str = ML_KEM_768) -> None:
        self.params = PARAMS[params] if isinstance(params, str) else params

    # ---------------------------------------------------------------- K-PKE
    def _pke_keygen(self, d: bytes) -> tuple[bytes, bytes]:
        """Alg. 13 K-PKE.KeyGen."""
        p = self.params
        rho, sigma = _G(d + bytes([p.k]))
        a = _matrix(rho, p.k)
        s = np.stack([_cbd(p.eta1, _prf(p.eta1, sigma, i)) for i in range(p.k)])
        e = np.stack([_cbd(p.eta1, _prf(p.eta1, sigma, p.k + i)) for i in range(p.k)])
        s_hat, e_hat = _NTT.forward(s), _NTT.forward(e)
        t_hat = (_matvec(a, s_hat) + e_hat) % Q
        return _byte_encode(t_hat, 12) + rho, _byte_encode(s_hat, 12)

    def _pke_encrypt(self, ek: bytes, m: bytes, r: bytes) -> bytes:
        """Alg. 14 K-PKE.Encrypt."""
        p = self.params
        k = p.k
        t_hat = _byte_decode(ek[: 384 * k], 12).reshape(k, N)
        rho = ek[384 * k :]
        at = _matrix(rho, k, transpose=True)
        y = np.stack([_cbd(p.eta1, _prf(p.eta1, r, i)) for i in range(k)])
        e1 = np.stack([_cbd(p.eta2, _prf(p.eta2, r, k + i)) for i in range(k)])
        e2 = _cbd(p.eta2, _prf(p.eta2, r, 2 * k))
        y_hat = _NTT.forward(y)
        u = (_NTT.inverse(_matvec(at, y_hat)) + e1) % Q
        mu = _decompress(_byte_decode(m, 1), 1)
        v = (_NTT.inverse(_multiply_ntts(t_hat, y_hat).sum(axis=0) % Q) + e2 + mu) % Q
        return _byte_encode(_compress(u, p.du), p.du) + _byte_encode(_compress(v, p.dv), p.dv)

    def _pke_decrypt(self, dk_pke: bytes, c: bytes) -> bytes:
        """Alg. 15 K-PKE.Decrypt."""
        p = self.params
        k = p.k
        c1, c2 = c[: 32 * p.du * k], c[32 * p.du * k :]
        u = _decompress(_byte_decode(c1, p.du), p.du).reshape(k, N)
        v = _decompress(_byte_decode(c2, p.dv), p.dv)
        s_hat = _byte_decode(dk_pke, 12).reshape(k, N)
        w = (v - _NTT.inverse(_multiply_ntts(s_hat, _NTT.forward(u)).sum(axis=0) % Q)) % Q
        return _byte_encode(_compress(w, 1), 1)

    # ---------------------------------------------------------------- ML-KEM
    def keygen_internal(self, d: bytes, z: bytes) -> tuple[bytes, bytes]:
        """Alg. 16 ML-KEM.KeyGen_internal -> (ek, dk)."""
        ek, dk_pke = self._pke_keygen(d)
        return ek, dk_pke + ek + _H(ek) + z

    def keygen(self, rng: RandomSource | None = None) -> tuple[bytes, bytes]:
        """Alg. 19 ML-KEM.KeyGen."""
        rng = rng or default_rng()
        return self.keygen_internal(rng.random_bytes(32), rng.random_bytes(32))

    def _check_ek(self, ek: bytes) -> None:
        p = self.params
        if len(ek) != p.ek_bytes:
            raise MLKEMError("encapsulation key has wrong length")
        k = p.k
        if _byte_encode(_byte_decode(ek[: 384 * k], 12), 12) != ek[: 384 * k]:
            raise MLKEMError("encapsulation key fails the modulus check")

    def encaps_internal(self, ek: bytes, m: bytes) -> tuple[bytes, bytes]:
        """Alg. 17 ML-KEM.Encaps_internal -> (shared_key, ciphertext)."""
        key, r = _G(m + _H(ek))
        return key, self._pke_encrypt(ek, m, r)

    def encaps(self, ek: bytes, rng: RandomSource | None = None) -> tuple[bytes, bytes]:
        """Alg. 20 ML-KEM.Encaps (with the FIPS 203 sec. 7.2 input checks)."""
        self._check_ek(ek)
        return self.encaps_internal(ek, (rng or default_rng()).random_bytes(32))

    def decaps(self, dk: bytes, c: bytes) -> bytes:
        """Alg. 18/21 ML-KEM.Decaps with implicit rejection."""
        p = self.params
        k = p.k
        if len(c) != p.ct_bytes or len(dk) != p.dk_bytes:
            raise MLKEMError("ciphertext or decapsulation key has wrong length")
        dk_pke = dk[: 384 * k]
        ek = dk[384 * k : 768 * k + 32]
        h = dk[768 * k + 32 : 768 * k + 64]
        z = dk[768 * k + 64 :]
        if _H(ek) != h:
            raise MLKEMError("decapsulation key hash check failed")
        m_prime = self._pke_decrypt(dk_pke, c)
        key_prime, r_prime = _G(m_prime + h)
        key_bar = _J(z + c)
        c_prime = self._pke_encrypt(ek, m_prime, r_prime)
        return ct_select_bytes(ct_equal(c, c_prime), key_prime, key_bar)
