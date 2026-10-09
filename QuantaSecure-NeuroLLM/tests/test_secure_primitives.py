"""Known-answer and property tests for the from-scratch cryptographic primitives.

The ML-KEM / ML-DSA digests below were produced by this implementation and
confirmed byte-for-byte against independent reference implementations
(kyber-py 1.2 / dilithium-py 1.4) during development; AES-GCM, ChaCha20-Poly1305,
HKDF, X25519 and Ed25519 vectors are the published RFC / NIST values.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from quantasecure_neurollm.secure import (
    MLDSA,
    MLKEM,
    AEADAuthenticationError,
    DeterministicDRBG,
    HybridSigningKey,
    X25519KeyPair,
    hkdf,
    hybrid_verify,
    make_aead,
)
from quantasecure_neurollm.secure.aes_gcm import AES256
from quantasecure_neurollm.secure.chacha20_poly1305 import poly1305_mac
from quantasecure_neurollm.secure.ed25519 import Ed25519PrivateKey, ed25519_verify
from quantasecure_neurollm.secure.mlkem import Q as KQ
from quantasecure_neurollm.secure.mlkem import _NTT as KYBER_NTT
from quantasecure_neurollm.secure.mlkem import _multiply_ntts
from quantasecure_neurollm.secure.mldsa import Q as DQ
from quantasecure_neurollm.secure.mldsa import _NTT as DIL_NTT
from quantasecure_neurollm.secure.ntt import schoolbook_negacyclic
from quantasecure_neurollm.secure.x25519 import X25519Error, x25519
from quantasecure_neurollm.utils.serialization import SerializationError, decode, encode

# ------------------------------------------------------------------ symmetric


def test_aes256_fips197_c3() -> None:
    out = AES256(bytes(range(32))).encrypt_block(bytes.fromhex("00112233445566778899aabbccddeeff"))
    assert out.hex() == "8ea2b7ca516745bfeafc49904b496089"


def test_aes_gcm_known_answers() -> None:
    gcm = make_aead("AES-256-GCM", bytes(32))
    assert gcm.encrypt(bytes(12), b"").hex() == "530f8afbc74536b9a963b4f1c4cb738b"
    assert gcm.encrypt(bytes(12), bytes(16)).hex() == "cea7403d4d606b6e074ec5d3baf39d18d0d1c8a799996bf0265b98b5d48ab919"


def test_chacha20_poly1305_rfc8439() -> None:
    key = bytes(range(0x80, 0xA0))
    nonce = bytes.fromhex("070000004041424344454647")
    aad = bytes.fromhex("50515253c0c1c2c3c4c5c6c7")
    pt = (
        b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, "
        b"sunscreen would be it."
    )
    out = make_aead("ChaCha20-Poly1305", key).encrypt(nonce, pt, aad)
    assert out.hex() == (
        "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d63dbea45e8ca9671282fafb69da92728b"
        "1a71de0a9e060b2905d6a5b67ecd3b3692ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc"
        "3ff4def08e4b7a9de576d26586cec64b61161ae10b594f09e26a7e902ecbd0600691"
    )


def test_poly1305_rfc8439() -> None:
    key = bytes.fromhex("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b")
    assert poly1305_mac(key, b"Cryptographic Forum Research Group").hex() == "a8061dc1305136c6c22b8baf0c0127a9"


@pytest.mark.parametrize("suite", ["AES-256-GCM", "ChaCha20-Poly1305"])
def test_aead_roundtrip_and_tamper_detection(suite: str) -> None:
    rng = DeterministicDRBG(1)
    key, nonce = rng.random_bytes(32), rng.random_bytes(12)
    aead = make_aead(suite, key)
    pt, aad = b"confidential tensor bytes" * 7, b"header"
    ct = aead.encrypt(nonce, pt, aad)
    assert aead.decrypt(nonce, ct, aad) == pt
    for i in (0, len(ct) // 2, len(ct) - 1):
        bad = bytearray(ct)
        bad[i] ^= 0x01
        with pytest.raises(AEADAuthenticationError):
            aead.decrypt(nonce, bytes(bad), aad)
    with pytest.raises(AEADAuthenticationError):
        aead.decrypt(nonce, ct, b"other header")


def test_hkdf_rfc5869_case1() -> None:
    okm = hkdf(b"\x0b" * 22, bytes(range(13)), bytes(range(0xF0, 0xFA)), 42)
    assert okm.hex() == "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"


# ------------------------------------------------------------------ X25519 / Ed25519


def test_x25519_rfc7748_vector() -> None:
    k = bytes.fromhex("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4")
    u = bytes.fromhex("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c")
    assert x25519(k, u).hex() == "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552"


def test_x25519_agreement_and_low_order_rejection() -> None:
    rng = DeterministicDRBG(2)
    a, b = X25519KeyPair.generate(rng), X25519KeyPair.generate(rng)
    assert a.exchange(b.public) == b.exchange(a.public)
    with pytest.raises(X25519Error):
        a.exchange(bytes(32))  # the identity / small-order point


def test_ed25519_rfc8032_test1() -> None:
    sk = Ed25519PrivateKey(bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"))
    assert sk.public_key.hex() == "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    sig = sk.sign(b"")
    assert sig.hex() == (
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd"
        "25bf5f0595bbe24655141438e7a100b"
    )
    assert ed25519_verify(sk.public_key, b"", sig)
    assert not ed25519_verify(sk.public_key, b"x", sig)


# ------------------------------------------------------------------ NTT


@pytest.mark.parametrize("ntt,q", [(KYBER_NTT, KQ), (DIL_NTT, DQ)])
def test_ntt_inverse_roundtrip(ntt, q) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    f = rng.integers(0, q, (3, 256))
    np.testing.assert_array_equal(ntt.inverse(ntt.forward(f)), f)


def test_ntt_multiplication_matches_schoolbook() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, KQ, 256), rng.integers(0, KQ, 256)
    prod = KYBER_NTT.inverse(_multiply_ntts(KYBER_NTT.forward(a), KYBER_NTT.forward(b)))
    np.testing.assert_array_equal(prod, schoolbook_negacyclic(a, b, KQ))
    a, b = rng.integers(0, DQ, 256), rng.integers(0, DQ, 256)
    prod = DIL_NTT.inverse(DIL_NTT.forward(a) * DIL_NTT.forward(b) % DQ)
    np.testing.assert_array_equal(prod, schoolbook_negacyclic(a, b, DQ))


# ------------------------------------------------------------------ ML-KEM


def test_mlkem768_known_answer() -> None:
    kem = MLKEM("ML-KEM-768")
    ek, dk = kem.keygen_internal(bytes(range(32)), bytes(range(32, 64)))
    key, ct = kem.encaps_internal(ek, bytes(range(64, 96)))
    assert hashlib.sha3_256(ek).hexdigest() == "a24e16d8f8f9383a95b77050f4d9fd2f5733eec1d63ef3c23ebf9918173669a7"
    assert hashlib.sha3_256(dk).hexdigest() == "1149f17c3c4ac6ab1e3e2d9d8bd0171355ac0fa31bb8855c48ceade874c0864b"
    assert hashlib.sha3_256(ct).hexdigest() == "b4cfbd24cef67afd3764276c6980e0f88f8e9ca57f59b7f12fe1a9c1e72f4710"
    assert key.hex() == "9cddd089ffe70e3996e76f7c8d06746df34d07e8657bc0fcf2bb0e1c3084aea1"
    assert kem.decaps(dk, ct) == key


@pytest.mark.parametrize("name", ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"])
def test_mlkem_roundtrip_sizes_and_implicit_rejection(name: str) -> None:
    kem = MLKEM(name)
    rng = DeterministicDRBG(3)
    ek, dk = kem.keygen(rng)
    assert (len(ek), len(dk)) == (kem.params.ek_bytes, kem.params.dk_bytes)
    key, ct = kem.encaps(ek, rng)
    assert len(ct) == kem.params.ct_bytes and len(key) == 32
    assert kem.decaps(dk, ct) == key
    bad = bytearray(ct)
    bad[10] ^= 0x40
    k_bad = kem.decaps(dk, bytes(bad))
    assert k_bad != key and len(k_bad) == 32  # implicit rejection: pseudorandom key, no error
    assert kem.decaps(dk, bytes(bad)) == k_bad  # ... deterministic in (z, c)


def test_mlkem_rejects_non_canonical_encapsulation_key() -> None:
    kem = MLKEM("ML-KEM-512")
    ek, _ = kem.keygen(DeterministicDRBG(4))
    bad = bytearray(ek)
    bad[0], bad[1] = 0xFF, bad[1] | 0x0F  # first 12-bit coefficient = 4095 >= q
    with pytest.raises(ValueError):
        kem.encaps(bytes(bad))


# ------------------------------------------------------------------ ML-DSA


def test_mldsa44_known_answer() -> None:
    sig = MLDSA("ML-DSA-44")
    pk, sk = sig.keygen_internal(bytes(range(32)))
    s = sig.sign(sk, b"QuantaSecure KAT message", deterministic=True)
    assert hashlib.sha3_256(pk).hexdigest() == "373c7bf2cac5bd2a6c35933bab0fa1c951f22247e1333383fcb618822080373f"
    assert hashlib.sha3_256(sk).hexdigest() == "da0665a6d807669d06aeb7010507aa6bd4640641055462e38e37344a274f36a6"
    assert hashlib.sha3_256(s).hexdigest() == "29e1bb0d7b3a0891adc631e4a016407c0731be488d004fd5920c1d00810b90e2"
    assert sig.verify(pk, b"QuantaSecure KAT message", s)


@pytest.mark.parametrize("name", ["ML-DSA-44", "ML-DSA-65"])
def test_mldsa_sign_verify_and_forgeries(name: str) -> None:
    sig = MLDSA(name)
    rng = DeterministicDRBG(5)
    pk, sk = sig.keygen(rng)
    msg = b"stage record: quantum features committed"
    s = sig.sign(sk, msg, ctx=b"QSNL", rng=rng)
    assert len(s) == sig.params.sig_bytes and len(pk) == sig.params.pk_bytes
    assert sig.verify(pk, msg, s, ctx=b"QSNL")
    assert not sig.verify(pk, msg + b".", s, ctx=b"QSNL")
    assert not sig.verify(pk, msg, s, ctx=b"other")
    bad = bytearray(s)
    bad[100] ^= 0x01
    assert not sig.verify(pk, msg, bytes(bad), ctx=b"QSNL")
    pk2, _ = sig.keygen(rng)
    assert not sig.verify(pk2, msg, s, ctx=b"QSNL")


def test_hybrid_signature_requires_both_components() -> None:
    rng = DeterministicDRBG(6)
    key = HybridSigningKey("quantum-node", rng=rng)
    other = HybridSigningKey("mallory", rng=rng)
    sig = key.sign("label", b"payload")
    assert hybrid_verify(key.public, "label", b"payload", sig)
    assert not hybrid_verify(key.public, "other-label", b"payload", sig)
    parts = decode(sig)
    forged = encode({"ed25519": parts["ed25519"], "mldsa": decode(other.sign("label", b"payload"))["mldsa"]})
    assert not hybrid_verify(key.public, "label", b"payload", forged)  # classical half alone is not enough


# ------------------------------------------------------------------ serialization


def test_canonical_serialization_roundtrip_and_strictness() -> None:
    obj = {"b": [1, -1, 2**70, 0.5, None, True], "a": np.arange(6, dtype=np.float64).reshape(2, 3), "c": b"\x00\xff", "d": "μ"}
    raw = encode(obj)
    back = decode(raw)
    assert back["b"] == obj["b"] and back["c"] == obj["c"] and back["d"] == obj["d"]
    np.testing.assert_array_equal(back["a"], obj["a"])
    assert encode({"x": 1, "y": 2}) == encode({"y": 2, "x": 1})  # key order does not matter
    with pytest.raises(SerializationError):
        decode(raw + b"\x00")
    with pytest.raises(SerializationError):
        encode({"o": object()})
    with pytest.raises(SerializationError):
        encode(np.array(["strings are not allowed"]))
