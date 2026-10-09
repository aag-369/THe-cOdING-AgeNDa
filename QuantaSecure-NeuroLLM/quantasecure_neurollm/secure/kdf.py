"""HMAC-SHA-256 and HKDF (RFC 5869), plus TLS-1.3-style labelled expansion.

HMAC and SHA-2/SHA-3 come from the Python standard library (``hmac``,
``hashlib``), which the project constraints explicitly allow; HKDF itself is
implemented here.  Security: HKDF-Extract is a randomness extractor / PRF in
its salt (Krawczyk 2010), and HKDF-Expand is a PRF keyed by the PRK.  When the
input keying material is a concatenation of several shared secrets (hybrid
X25519 || ML-KEM), the output stays pseudorandom as long as *any one* component
is unpredictable, under the dual-PRF assumption on HMAC (Bindel et al., PQCrypto 2019).
"""

from __future__ import annotations

import hashlib
import hmac

__all__ = ["hmac_sha256", "hkdf_extract", "hkdf_expand", "hkdf", "expand_label", "key_fingerprint"]

HASH_LEN = 32


def hmac_sha256(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def hkdf_extract(salt: bytes | None, ikm: bytes) -> bytes:
    """``PRK = HMAC(salt, IKM)``; an absent salt is ``HashLen`` zero bytes."""
    return hmac_sha256(salt if salt else bytes(HASH_LEN), ikm)


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """``T(i) = HMAC(PRK, T(i-1) || info || i)``; output the first ``length`` bytes."""
    if length > 255 * HASH_LEN:
        raise ValueError("HKDF output too long")
    out, t = b"", b""
    i = 1
    while len(out) < length:
        t = hmac_sha256(prk, t + info + bytes([i]))
        out += t
        i += 1
    return out[:length]


def hkdf(ikm: bytes, salt: bytes | None, info: bytes, length: int) -> bytes:
    return hkdf_expand(hkdf_extract(salt, ikm), info, length)


def expand_label(secret: bytes, label: str, context: bytes, length: int) -> bytes:
    """Domain-separated expansion: ``info = len || "qsnl " + label || len || context``."""
    full = b"qsnl " + label.encode()
    info = length.to_bytes(2, "big") + bytes([len(full)]) + full + bytes([len(context)]) + context
    return hkdf_expand(secret, info, length)


def key_fingerprint(key: bytes) -> str:
    """One-way, loggable 64-bit fingerprint of a key (never reveals the key)."""
    return hashlib.sha3_256(b"QSNL-key-fingerprint" + key).hexdigest()[:16]
