"""Shared helpers: canonical serialization, logging/audit, math, seeding, timing."""

from .logging import AuditLog, AuditRecord, configure_logging, get_logger, short_hex
from .mathx import EPS64, clip_probabilities, entropy_from_eigenvalues, logsumexp, shannon_entropy_bits, stable_softmax
from .rng import derive_seed, numpy_rng
from .serialization import SerializationError, decode, encode
from .timing import Stopwatch

__all__ = [
    "AuditLog",
    "AuditRecord",
    "configure_logging",
    "get_logger",
    "short_hex",
    "EPS64",
    "clip_probabilities",
    "entropy_from_eigenvalues",
    "logsumexp",
    "shannon_entropy_bits",
    "stable_softmax",
    "derive_seed",
    "numpy_rng",
    "SerializationError",
    "decode",
    "encode",
    "Stopwatch",
]
