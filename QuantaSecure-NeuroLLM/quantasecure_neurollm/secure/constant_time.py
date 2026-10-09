"""Side-channel notes and the few constant-time helpers Python can honestly offer.

What this prototype does
------------------------
* All secret comparisons (AEAD tags, key-confirmation MACs, ML-KEM
  re-encryption checks) use :func:`hmac.compare_digest`, whose running time
  depends only on the length of its inputs.
* Branch-free formulations are used where they are natural: the X25519 ladder
  uses an arithmetic conditional swap, GHASH masks instead of branching on key
  bits, ML-KEM implicit rejection selects the output with a mask.
* Secret-dependent *table lookups* are avoided in ChaCha20 (pure ARX), which is
  why it is the recommended AEAD for software deployments.

What it cannot do (and a production implementation must)
--------------------------------------------------------
* CPython big integers have data-dependent running time (number of limbs,
  Karatsuba thresholds, early-exit normalisation), so modular arithmetic in
  X25519 / Ed25519 / Pedersen leaks timing in principle.
* NumPy vector operations used by the lattice code (NTT, sampling, rejection
  checks) are not guaranteed constant-time, and ML-DSA's rejection loop
  reveals the number of iterations (by design this leaks only information about
  *rejected* candidates, which are independent of the key; FIPS 204 sec. 3.6).
* The AES implementation uses 1 KiB T-tables indexed by secret bytes and is
  therefore vulnerable to cache-timing attacks (Bernstein 2005; Osvik-Shamir-
  Tromer 2006).  Use AES-NI or a bitsliced implementation in production.
* Python cannot reliably zeroise memory: ``bytes`` are immutable and the
  allocator may copy them.  Ratchet keys are dropped as soon as possible, which
  shortens but does not eliminate their lifetime in RAM.

The threat model of this artifact is therefore *network* attackers (on-path
adversaries who see, drop, replay and modify frames), not co-resident attackers
with timing or cache access.
"""

from __future__ import annotations

import hmac

__all__ = ["ct_equal", "ct_select_bytes"]


def ct_equal(a: bytes, b: bytes) -> bool:
    """Length-dependent-only comparison (wraps :func:`hmac.compare_digest`)."""
    return hmac.compare_digest(a, b)


def ct_select_bytes(condition: bool, if_true: bytes, if_false: bytes) -> bytes:
    """Select between two equal-length byte strings with a mask instead of a branch."""
    if len(if_true) != len(if_false):
        raise ValueError("ct_select_bytes needs equal lengths")
    mask = -int(bool(condition)) & 0xFF
    return bytes((x & mask) | (y & ~mask & 0xFF) for x, y in zip(if_true, if_false))
