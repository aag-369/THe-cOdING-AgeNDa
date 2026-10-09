"""ML-DSA (FIPS 204, formerly CRYSTALS-Dilithium): Module-LWE/SIS signatures.

A from-scratch implementation following the FIPS 204 algorithms (numbers cited
inline).  During development it was cross-checked byte-for-byte against an
independent reference implementation for key generation and (deterministic)
signing; the resulting known-answer digests are pinned in the tests.

Scheme in one paragraph
-----------------------
The public key is ``t = A s1 + s2`` (rounded: only the high bits ``t1`` are
published) for short secrets ``s1, s2``.  To sign, draw a masking vector ``y``,
commit to ``w1 = HighBits(A y)``, derive the challenge ``c`` (a sparse ternary
polynomial with ``tau`` non-zeros) from ``H(mu || w1)``, and answer
``z = y + c s1``.  "Fiat-Shamir with aborts": the candidate is rejected unless
``||z||_inf < gamma1 - beta`` and the low bits are small, which makes the
distribution of ``z`` independent of the secret key.  Hints ``h`` let the
verifier recompute ``w1`` from ``A z - c t1 2^d`` despite the missing ``t0``.
EUF-CMA security reduces to Module-LWE and SelfTargetMSIS in the QROM.

Side channels: the rejection loop's iteration count is public by design (it
only depends on discarded candidates); arithmetic is NumPy / Python ints and
not guaranteed constant-time (see :mod:`.constant_time`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ntt import NTT
from .rng import RandomSource, default_rng

__all__ = ["MLDSAParams", "ML_DSA_44", "ML_DSA_65", "ML_DSA_87", "MLDSA", "PARAMS", "MLDSAError"]

Q = 8380417
N = 256
D = 13
_NTT = NTT(Q, 1753, 8)


class MLDSAError(ValueError):
    pass


@dataclass(frozen=True)
class MLDSAParams:
    name: str
    tau: int
    lam: int  # collision strength lambda (c_tilde has lam/4 bytes)
    gamma1: int
    gamma2: int
    k: int
    l: int
    eta: int
    omega: int

    @property
    def beta(self) -> int:
        return self.tau * self.eta

    @property
    def pk_bytes(self) -> int:
        return 32 + 32 * self.k * 10

    @property
    def sig_bytes(self) -> int:
        return self.lam // 4 + self.l * 32 * (1 + (self.gamma1 - 1).bit_length()) + self.omega + self.k


ML_DSA_44 = MLDSAParams("ML-DSA-44", 39, 128, 2**17, (Q - 1) // 88, 4, 4, 2, 80)
ML_DSA_65 = MLDSAParams("ML-DSA-65", 49, 192, 2**19, (Q - 1) // 32, 6, 5, 4, 55)
ML_DSA_87 = MLDSAParams("ML-DSA-87", 60, 256, 2**19, (Q - 1) // 32, 8, 7, 2, 75)
PARAMS = {p.name: p for p in (ML_DSA_44, ML_DSA_65, ML_DSA_87)}


def _H(data: bytes, n: int) -> bytes:
    return hashlib.shake_256(data).digest(n)


# ---------------------------------------------------------------- bit packing
def _simple_bit_pack(w: np.ndarray, bits: int) -> bytes:
    """Alg. 16 SimpleBitPack: coefficients in [0, 2^bits), little-endian bit order."""
    w = np.asarray(w, dtype=np.int64).reshape(-1)
    b = ((w[:, None] >> np.arange(bits)) & 1).astype(np.uint8)
    return np.packbits(b.reshape(-1), bitorder="little").tobytes()


def _simple_bit_unpack(data: bytes, bits: int) -> np.ndarray:
    b = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little").astype(np.int64)
    return (b.reshape(-1, bits) << np.arange(bits)).sum(axis=1)


def _bit_pack(w: np.ndarray, a: int, b: int) -> bytes:
    """Alg. 17 BitPack: coefficients in [-a, b] stored as b - w with bitlen(a + b) bits."""
    return _simple_bit_pack(b - np.asarray(w, dtype=np.int64), (a + b).bit_length())


def _bit_unpack(data: bytes, a: int, b: int) -> np.ndarray:
    return b - _simple_bit_unpack(data, (a + b).bit_length())


def _centered(x: np.ndarray) -> np.ndarray:
    """Representative in (-(q-1)/2, (q-1)/2]."""
    x = np.asarray(x, dtype=np.int64) % Q
    return np.where(x > (Q - 1) // 2, x - Q, x)


def _inf_norm(x: np.ndarray) -> int:
    return int(np.max(np.abs(_centered(x))))


# ---------------------------------------------------------------- sampling
def _rej_ntt_poly(seed: bytes) -> np.ndarray:
    """Alg. 30 RejNTTPoly via Alg. 14 CoeffFromThreeBytes (SHAKE128)."""
    length = 840
    while True:
        b = np.frombuffer(hashlib.shake_128(seed).digest(length), dtype=np.uint8).astype(np.int64)
        b = b[: (len(b) // 3) * 3].reshape(-1, 3)
        z = ((b[:, 2] & 127) << 16) | (b[:, 1] << 8) | b[:, 0]
        ok = z[z < Q]
        if len(ok) >= N:
            return ok[:N]
        length += 168


def _rej_bounded_poly(seed: bytes, eta: int) -> np.ndarray:
    """Alg. 31 RejBoundedPoly via Alg. 15 CoeffFromHalfByte (SHAKE256)."""
    length = 272
    while True:
        z = np.frombuffer(hashlib.shake_256(seed).digest(length), dtype=np.uint8).astype(np.int64)
        nib = np.stack([z & 15, z >> 4], axis=1).reshape(-1)
        if eta == 2:
            nib = nib[nib < 15]
            vals = 2 - (nib % 5)
        else:
            nib = nib[nib < 9]
            vals = 4 - nib
        if len(vals) >= N:
            return vals[:N] % Q
        length += 136


def _expand_a(rho: bytes, p: MLDSAParams) -> np.ndarray:
    """Alg. 32 ExpandA: A_hat[r][s] = RejNTTPoly(rho || s || r)."""
    a = np.empty((p.k, p.l, N), dtype=np.int64)
    for r in range(p.k):
        for s in range(p.l):
            a[r, s] = _rej_ntt_poly(rho + bytes([s, r]))
    return a


def _expand_s(rho_prime: bytes, p: MLDSAParams) -> tuple[np.ndarray, np.ndarray]:
    """Alg. 33 ExpandS."""
    s1 = np.stack([_rej_bounded_poly(rho_prime + r.to_bytes(2, "little"), p.eta) for r in range(p.l)])
    s2 = np.stack([_rej_bounded_poly(rho_prime + (r + p.l).to_bytes(2, "little"), p.eta) for r in range(p.k)])
    return s1, s2


def _expand_mask(rho2: bytes, kappa: int, p: MLDSAParams) -> np.ndarray:
    """Alg. 34 ExpandMask: y with coefficients in (-gamma1, gamma1]."""
    c = 1 + (p.gamma1 - 1).bit_length()
    ys = []
    for r in range(p.l):
        v = _H(rho2 + (kappa + r).to_bytes(2, "little"), 32 * c)
        ys.append(_bit_unpack(v, p.gamma1 - 1, p.gamma1) % Q)
    return np.stack(ys)


def _sample_in_ball(c_tilde: bytes, tau: int) -> np.ndarray:
    """Alg. 29 SampleInBall: tau coefficients in {-1, +1}, the rest 0."""
    stream = hashlib.shake_256(c_tilde).digest(8 + 4 * 256)
    signs = int.from_bytes(stream[:8], "little")
    c = np.zeros(N, dtype=np.int64)
    pos = 8
    for i in range(N - tau, N):
        while True:
            if pos >= len(stream):
                stream = hashlib.shake_256(c_tilde).digest(len(stream) * 2)
            j = stream[pos]
            pos += 1
            if j <= i:
                break
        c[i] = c[j]
        c[j] = 1 - 2 * ((signs >> (i + tau - N)) & 1)
    return c % Q


# ---------------------------------------------------------------- rounding
def _power2round(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alg. 35 Power2Round: r = r1 2^d + r0 with r0 in (-2^{d-1}, 2^{d-1}]."""
    rp = np.asarray(r, dtype=np.int64) % Q
    r0 = rp & ((1 << D) - 1)
    r0 = np.where(r0 > (1 << (D - 1)), r0 - (1 << D), r0)
    return (rp - r0) >> D, r0


def _decompose(r: np.ndarray, gamma2: int) -> tuple[np.ndarray, np.ndarray]:
    """Alg. 36 Decompose."""
    rp = np.asarray(r, dtype=np.int64) % Q
    r0 = rp % (2 * gamma2)
    r0 = np.where(r0 > gamma2, r0 - 2 * gamma2, r0)
    edge = (rp - r0) == Q - 1
    r1 = np.where(edge, 0, (rp - r0) // (2 * gamma2))
    r0 = np.where(edge, r0 - 1, r0)
    return r1, r0


def _high_bits(r: np.ndarray, gamma2: int) -> np.ndarray:
    return _decompose(r, gamma2)[0]


def _low_bits(r: np.ndarray, gamma2: int) -> np.ndarray:
    return _decompose(r, gamma2)[1]


def _make_hint(z: np.ndarray, r: np.ndarray, gamma2: int) -> np.ndarray:
    """Alg. 39 MakeHint."""
    return (_high_bits(r, gamma2) != _high_bits(np.asarray(r) + np.asarray(z), gamma2)).astype(np.int64)


def _use_hint(h: np.ndarray, r: np.ndarray, gamma2: int) -> np.ndarray:
    """Alg. 40 UseHint."""
    m = (Q - 1) // (2 * gamma2)
    r1, r0 = _decompose(r, gamma2)
    up = (h == 1) & (r0 > 0)
    down = (h == 1) & (r0 <= 0)
    return np.where(up, (r1 + 1) % m, np.where(down, (r1 - 1) % m, r1))


def _w1_encode(w1: np.ndarray, gamma2: int) -> bytes:
    """Alg. 28 w1Encode."""
    return _simple_bit_pack(w1, ((Q - 1) // (2 * gamma2) - 1).bit_length())


def _hint_bit_pack(h: np.ndarray, p: MLDSAParams) -> bytes:
    """Alg. 20 HintBitPack."""
    y = bytearray(p.omega + p.k)
    idx = 0
    for i in range(p.k):
        for j in np.nonzero(h[i])[0]:
            y[idx] = int(j)
            idx += 1
        y[p.omega + i] = idx
    return bytes(y)


def _hint_bit_unpack(y: bytes, p: MLDSAParams) -> Optional[np.ndarray]:
    """Alg. 21 HintBitUnpack (returns None on malformed / non-canonical input)."""
    h = np.zeros((p.k, N), dtype=np.int64)
    idx = 0
    for i in range(p.k):
        end = y[p.omega + i]
        if end < idx or end > p.omega:
            return None
        first = idx
        while idx < end:
            if idx > first and y[idx - 1] >= y[idx]:
                return None
            h[i, y[idx]] = 1
            idx += 1
    if any(y[i] != 0 for i in range(idx, p.omega)):
        return None
    return h


class MLDSA:
    """ML-DSA key generation, signing and verification for one parameter set."""

    def __init__(self, params: MLDSAParams | str = ML_DSA_44) -> None:
        self.params = PARAMS[params] if isinstance(params, str) else params
        self.last_sign_attempts = 0

    # ------------------------------------------------------------- encodings
    def _pk_encode(self, rho: bytes, t1: np.ndarray) -> bytes:
        return rho + _simple_bit_pack(t1, 10)

    def _pk_decode(self, pk: bytes) -> tuple[bytes, np.ndarray]:
        p = self.params
        if len(pk) != p.pk_bytes:
            raise MLDSAError("public key has wrong length")
        return pk[:32], _simple_bit_unpack(pk[32:], 10).reshape(p.k, N)

    def _sk_encode(self, rho: bytes, key: bytes, tr: bytes, s1: np.ndarray, s2: np.ndarray, t0: np.ndarray) -> bytes:
        p = self.params
        return (
            rho
            + key
            + tr
            + _bit_pack(_centered(s1), p.eta, p.eta)
            + _bit_pack(_centered(s2), p.eta, p.eta)
            + _bit_pack(t0, (1 << (D - 1)) - 1, 1 << (D - 1))
        )

    def _sk_decode(self, sk: bytes) -> tuple[bytes, bytes, bytes, np.ndarray, np.ndarray, np.ndarray]:
        p = self.params
        eb = 32 * (2 * p.eta).bit_length()
        off = 128
        s1 = _bit_unpack(sk[off : off + eb * p.l], p.eta, p.eta).reshape(p.l, N)
        off += eb * p.l
        s2 = _bit_unpack(sk[off : off + eb * p.k], p.eta, p.eta).reshape(p.k, N)
        off += eb * p.k
        t0 = _bit_unpack(sk[off : off + 32 * D * p.k], (1 << (D - 1)) - 1, 1 << (D - 1)).reshape(p.k, N)
        if off + 32 * D * p.k != len(sk) or np.abs(s1).max() > p.eta or np.abs(s2).max() > p.eta:
            raise MLDSAError("malformed secret key")
        return sk[:32], sk[32:64], sk[64:128], s1 % Q, s2 % Q, t0 % Q

    def _sig_encode(self, c_tilde: bytes, z: np.ndarray, h: np.ndarray) -> bytes:
        p = self.params
        return c_tilde + _bit_pack(_centered(z), p.gamma1 - 1, p.gamma1) + _hint_bit_pack(h, p)

    def _sig_decode(self, sig: bytes) -> Optional[tuple[bytes, np.ndarray, np.ndarray]]:
        p = self.params
        if len(sig) != p.sig_bytes:
            return None
        cl = p.lam // 4
        zb = 32 * (1 + (p.gamma1 - 1).bit_length()) * p.l
        z = _bit_unpack(sig[cl : cl + zb], p.gamma1 - 1, p.gamma1).reshape(p.l, N)
        h = _hint_bit_unpack(sig[cl + zb :], p)
        if h is None:
            return None
        return sig[:cl], z % Q, h

    # ------------------------------------------------------------- internal API
    def keygen_internal(self, xi: bytes) -> tuple[bytes, bytes]:
        """Alg. 6 ML-DSA.KeyGen_internal -> (pk, sk)."""
        p = self.params
        seed = _H(xi + bytes([p.k, p.l]), 128)
        rho, rho_prime, key = seed[:32], seed[32:96], seed[96:]
        a_hat = _expand_a(rho, p)
        s1, s2 = _expand_s(rho_prime, p)
        t = (_NTT.inverse((a_hat * _NTT.forward(s1)[None]).sum(axis=1) % Q) + s2) % Q
        t1, t0 = _power2round(t)
        pk = self._pk_encode(rho, t1)
        tr = _H(pk, 64)
        return pk, self._sk_encode(rho, key, tr, s1, s2, t0)

    def sign_internal(self, sk: bytes, m_prime: bytes, rnd: bytes) -> bytes:
        """Alg. 7 ML-DSA.Sign_internal (Fiat-Shamir with aborts)."""
        p = self.params
        rho, key, tr, s1, s2, t0 = self._sk_decode(sk)
        s1_hat, s2_hat, t0_hat = _NTT.forward(s1), _NTT.forward(s2), _NTT.forward(t0)
        a_hat = _expand_a(rho, p)
        mu = _H(tr + m_prime, 64)
        rho2 = _H(key + rnd + mu, 64)
        kappa = 0
        attempts = 0
        while True:
            attempts += 1
            if attempts > 1000:
                raise MLDSAError("signing did not terminate (should be astronomically unlikely)")
            y = _expand_mask(rho2, kappa, p)
            kappa += p.l
            w = _NTT.inverse((a_hat * _NTT.forward(y)[None]).sum(axis=1) % Q)
            w1 = _high_bits(w, p.gamma2)
            c_tilde = _H(mu + _w1_encode(w1, p.gamma2), p.lam // 4)
            c_hat = _NTT.forward(_sample_in_ball(c_tilde, p.tau))
            cs1 = _NTT.inverse(c_hat[None] * s1_hat % Q)
            cs2 = _NTT.inverse(c_hat[None] * s2_hat % Q)
            z = (y + cs1) % Q
            r0 = _low_bits((w - cs2) % Q, p.gamma2)
            if _inf_norm(z) >= p.gamma1 - p.beta or int(np.max(np.abs(r0))) >= p.gamma2 - p.beta:
                continue
            ct0 = _NTT.inverse(c_hat[None] * t0_hat % Q)
            h = _make_hint((-ct0) % Q, (w - cs2 + ct0) % Q, p.gamma2)
            if _inf_norm(ct0) >= p.gamma2 or int(h.sum()) > p.omega:
                continue
            self.last_sign_attempts = attempts
            return self._sig_encode(c_tilde, z, h)

    def verify_internal(self, pk: bytes, m_prime: bytes, sig: bytes) -> bool:
        """Alg. 8 ML-DSA.Verify_internal."""
        p = self.params
        try:
            rho, t1 = self._pk_decode(pk)
        except MLDSAError:
            return False
        dec = self._sig_decode(sig)
        if dec is None:
            return False
        c_tilde, z, h = dec
        if _inf_norm(z) >= p.gamma1 - p.beta:
            return False
        a_hat = _expand_a(rho, p)
        tr = _H(pk, 64)
        mu = _H(tr + m_prime, 64)
        c_hat = _NTT.forward(_sample_in_ball(c_tilde, p.tau))
        az = (a_hat * _NTT.forward(z)[None]).sum(axis=1) % Q
        ct1 = c_hat[None] * _NTT.forward((t1 << D) % Q) % Q
        w_approx = _NTT.inverse((az - ct1) % Q)
        w1 = _use_hint(h, w_approx, p.gamma2)
        return _H(mu + _w1_encode(w1, p.gamma2), p.lam // 4) == c_tilde

    # ------------------------------------------------------------- external API
    def keygen(self, rng: RandomSource | None = None) -> tuple[bytes, bytes]:
        """Alg. 1 ML-DSA.KeyGen."""
        return self.keygen_internal((rng or default_rng()).random_bytes(32))

    @staticmethod
    def _message_prime(msg: bytes, ctx: bytes) -> bytes:
        if len(ctx) > 255:
            raise MLDSAError("context string longer than 255 bytes")
        return bytes([0, len(ctx)]) + ctx + msg

    def sign(self, sk: bytes, msg: bytes, ctx: bytes = b"", rng: RandomSource | None = None, deterministic: bool = False) -> bytes:
        """Alg. 2 ML-DSA.Sign (hedged by default; ``deterministic`` uses rnd = 0^32)."""
        rnd = bytes(32) if deterministic else (rng or default_rng()).random_bytes(32)
        return self.sign_internal(sk, self._message_prime(msg, ctx), rnd)

    def verify(self, pk: bytes, msg: bytes, sig: bytes, ctx: bytes = b"") -> bool:
        """Alg. 3 ML-DSA.Verify."""
        if len(ctx) > 255:
            return False
        return self.verify_internal(pk, self._message_prime(msg, ctx), sig)
