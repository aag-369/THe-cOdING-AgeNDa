"""Hybrid (classical + post-quantum) signatures and key combination.

Signatures use the AND-combiner: a hybrid signature is a pair
``(Ed25519 sig, ML-DSA sig)`` over the same domain-separated message and is
valid only if *both* verify.  Forging it requires forging both, so it is
EUF-CMA as long as either component is (Bindel et al. 2017).  Classical
security survives a break of the young lattice scheme; post-quantum security
survives Shor.

Key combination for the hybrid KEM (X25519 + ML-KEM) is
``HKDF-Extract(salt = transcript_hash, IKM = ss_mlkem || ss_x25519)``.  The
transcript hash binds both public keys and the ML-KEM ciphertext, so the
derived secret is pseudorandom if either shared secret is (dual-PRF argument),
and mix-and-match attacks across sessions are excluded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..utils.serialization import decode, encode
from .ed25519 import Ed25519PrivateKey, ed25519_verify
from .kdf import hkdf_extract
from .mldsa import MLDSA
from .rng import RandomSource, default_rng

__all__ = ["HybridPublicKey", "HybridSigningKey", "hybrid_verify", "combine_shared_secrets"]

_SIG_DOMAIN = b"QSNL-hybrid-signature-v1"


def _framed(label: str, msg: bytes) -> bytes:
    lb = label.encode()
    return _SIG_DOMAIN + len(lb).to_bytes(2, "big") + lb + msg


@dataclass(frozen=True)
class HybridPublicKey:
    """Ed25519 + ML-DSA verification key bound to an identity name."""

    owner: str
    ed25519: bytes
    mldsa: bytes
    mldsa_params: str

    def to_dict(self) -> dict[str, Any]:
        return {"owner": self.owner, "ed25519": self.ed25519, "mldsa": self.mldsa, "mldsa_params": self.mldsa_params}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HybridPublicKey":
        return cls(d["owner"], d["ed25519"], d["mldsa"], d["mldsa_params"])

    def fingerprint(self) -> str:
        return hashlib.sha3_256(encode(self.to_dict())).hexdigest()[:16]


class HybridSigningKey:
    """Holds both secret keys; produces hybrid signatures."""

    def __init__(self, owner: str, mldsa_params: str = "ML-DSA-44", rng: RandomSource | None = None) -> None:
        rng = rng or default_rng()
        self._ed = Ed25519PrivateKey.generate(rng)
        self._mldsa = MLDSA(mldsa_params)
        self._rng = rng
        pk, self._mldsa_sk = self._mldsa.keygen(rng)
        self.public = HybridPublicKey(owner, self._ed.public_key, pk, mldsa_params)

    def sign(self, label: str, msg: bytes) -> bytes:
        m = _framed(label, msg)
        ed_sig = self._ed.sign(m)
        pq_sig = self._mldsa.sign(self._mldsa_sk, m, ctx=label.encode()[:255], rng=self._rng)
        return encode({"ed25519": ed_sig, "mldsa": pq_sig})

    def __repr__(self) -> str:
        return f"HybridSigningKey(owner={self.public.owner!r}, fp={self.public.fingerprint()}, secrets=<redacted>)"


def hybrid_verify(public: HybridPublicKey, label: str, msg: bytes, signature: bytes) -> bool:
    """Valid iff *both* the Ed25519 and the ML-DSA signature verify."""
    try:
        parts = decode(signature)
        ed_sig, pq_sig = parts["ed25519"], parts["mldsa"]
    except Exception:
        return False
    m = _framed(label, msg)
    ok_ed = ed25519_verify(public.ed25519, m, ed_sig)
    ok_pq = MLDSA(public.mldsa_params).verify(public.mldsa, m, pq_sig, ctx=label.encode()[:255])
    return ok_ed and ok_pq


def combine_shared_secrets(ss_kem: bytes, ss_dh: bytes, transcript_hash: bytes) -> bytes:
    """PRK = HKDF-Extract(transcript_hash, ss_kem || ss_dh)."""
    return hkdf_extract(transcript_hash, ss_kem + ss_dh)
