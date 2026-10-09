"""From-scratch cryptography and the secure inter-module channel.

Only the Python standard library (``hashlib``, ``hmac``, ``os``) and NumPy are
used.  Components:

=====================  ===========================================================
``mlkem``              ML-KEM-512/768/1024 (FIPS 203) -- Module-LWE KEM
``mldsa``              ML-DSA-44/65/87 (FIPS 204) -- Module-LWE signatures
``x25519``             X25519 ECDH (RFC 7748)
``ed25519``            edwards25519 group + Ed25519 signatures (RFC 8032)
``aes_gcm``            AES-256-GCM (FIPS 197 + SP 800-38D)
``chacha20_poly1305``  ChaCha20-Poly1305 (RFC 8439)
``kdf``                HMAC-SHA-256, HKDF (RFC 5869)
``hybrid``             hybrid signatures (Ed25519 AND ML-DSA), KEM combiner
``handshake``          signed hybrid X25519 + ML-KEM handshake with key confirmation
``ratchet``            Double Ratchet with ML-KEM mixing at every DH step
``channel``            framed AEAD sessions, recording bus, egress guard
``commitments``        salted SHA3 commitments, Pedersen commitments
``zkp``                NIZK proof of measurement consistency
=====================  ===========================================================

See :mod:`.constant_time` for the side-channel model.
"""

from .aead import AEADAuthenticationError, make_aead
from .channel import ChannelError, EgressGuard, Frame, SecureBus, SecureSession, WireRecord
from .commitments import HashCommitment, commit_object, pedersen_commit, verify_object
from .handshake import HandshakeError, HandshakeInitiator, HandshakeResponder, HandshakeResult, Suite
from .hybrid import HybridPublicKey, HybridSigningKey, hybrid_verify
from .kdf import hkdf, key_fingerprint
from .mldsa import MLDSA
from .mlkem import MLKEM
from .ratchet import DoubleRatchet, RatchetError
from .rng import DeterministicDRBG, RandomSource, SystemRandom, default_rng
from .x25519 import X25519KeyPair
from .zkp import MeasurementProof, ZKVerification, prove_measurement, verify_measurement

__all__ = [
    "AEADAuthenticationError",
    "make_aead",
    "ChannelError",
    "EgressGuard",
    "Frame",
    "SecureBus",
    "SecureSession",
    "WireRecord",
    "HashCommitment",
    "commit_object",
    "pedersen_commit",
    "verify_object",
    "HandshakeError",
    "HandshakeInitiator",
    "HandshakeResponder",
    "HandshakeResult",
    "Suite",
    "HybridPublicKey",
    "HybridSigningKey",
    "hybrid_verify",
    "hkdf",
    "key_fingerprint",
    "MLDSA",
    "MLKEM",
    "DoubleRatchet",
    "RatchetError",
    "DeterministicDRBG",
    "RandomSource",
    "SystemRandom",
    "default_rng",
    "X25519KeyPair",
    "MeasurementProof",
    "ZKVerification",
    "prove_measurement",
    "verify_measurement",
]
