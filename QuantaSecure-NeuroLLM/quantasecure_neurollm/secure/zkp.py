"""NIZK proof that reported measurement shots are consistent with a committed quantum state.

Statement
---------
The quantum node holds a state ``|psi>`` on ``n`` qubits.  It publishes

* Pedersen commitments ``C_i = p_i G + r_i H`` to the *quantised* Born
  probabilities ``p_i = round(2^B |<i|psi>|^2)`` (integers summing to ``2^B``),
* ``N`` computational-basis shots, and
* claimed linear functionals of the hidden distribution: the total mass, the
  per-qubit marginals ``m_q = sum_{i: bit_q(i)=1} p_i`` and the even-parity masses
  ``e_ab = sum_{i: bit_a(i)=bit_b(i)} p_i`` (equivalently ``<Z_a Z_b>``).

It proves, without revealing any ``p_i`` or ``r_i``:

1. **Knowledge of openings** of all ``C_i`` -- a batched Okamoto proof on the
   random linear combination ``sum_i rho_i C_i`` with Fiat-Shamir weights ``rho_i``.
2. **Correctness of every claimed functional** ``v_k = <a_k, p>`` -- for each
   ``k``, ``D_k = sum_{i in S_k} C_i - v_k G`` must equal ``R_k H`` and the prover
   gives a Schnorr proof of knowledge of ``R_k``.  If ``<a_k, p> != v_k`` this
   would reveal ``log_H G``, contradicting the discrete-log assumption.
3. **Total mass = 2^B** (one of the functionals), so the commitments describe
   a normalised distribution.

The shots are sampled from the committed distribution with a seed derived from
the transcript *after* the commitments are fixed, so they cannot be chosen
adaptively.  The verifier finally runs a Hoeffding test: each empirical
marginal / parity frequency must lie within
``eps = sqrt(ln(2K/delta) / (2N))`` of the proven value (union bound over the
``K`` tests), so an honest prover fails with probability at most ``delta``.

Zero knowledge: Pedersen commitments are perfectly hiding and the Schnorr /
Okamoto proofs are honest-verifier zero-knowledge, hence NIZK in the random
oracle model.  The only information released is what is claimed: the marginals,
parities and shots themselves.

Limitations (documented, not hidden): there is no range proof that each
``p_i >= 0``; the soundness of the shot test is statistical, and a prover can
grind the blinding factors to re-roll the shot seed (each re-roll still has to
pass the Hoeffding test, which bounds the gain to ``log2(#trials)`` bits of
selection power).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ..quanta.measurement import sample_outcomes
from ..utils.serialization import encode
from .commitments import pedersen_commit_vector, pedersen_tables
from .ed25519 import (
    IDENTITY,
    L,
    Ed25519Error,
    Point,
    decode_point,
    encode_point,
    point_add,
    point_equal,
    point_neg,
    random_scalar,
    scalar_mult,
)
from .rng import RandomSource, default_rng
from .transcript import Transcript

__all__ = ["MeasurementProof", "ZKVerification", "quantize_probabilities", "prove_measurement", "verify_measurement"]

_PROTOCOL = "QSNL-measurement-consistency-v1"


def quantize_probabilities(probs: np.ndarray, scale_bits: int) -> np.ndarray:
    """Integers ``p_i >= 0`` with ``sum p_i = 2^B`` exactly (largest-remainder rounding).

    The quantisation error per entry is ``< 2^-B`` in probability.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, None)
    p = p / p.sum()
    total = 1 << scale_bits
    scaled = p * total
    base = np.floor(scaled).astype(np.int64)
    deficit = int(total - base.sum())
    if deficit:
        order = np.argsort(-(scaled - base), kind="stable")
        base[order[:deficit]] += 1
    return base


def _statements(n: int, pairs: Sequence[tuple[int, int]]) -> list[tuple[str, np.ndarray]]:
    idx = np.arange(2**n, dtype=np.int64)
    bit = lambda q: (idx >> (n - 1 - q)) & 1  # noqa: E731
    out: list[tuple[str, np.ndarray]] = [("total", np.ones(2**n, dtype=bool))]
    out += [(f"marginal_q{q}", bit(q).astype(bool)) for q in range(n)]
    out += [(f"even_parity_{a}{b}", (bit(a) == bit(b))) for a, b in pairs]
    return out


def _weights(seed: bytes, count: int) -> list[int]:
    """Odd 64-bit Fiat-Shamir weights for the batched opening proof."""
    stream = hashlib.shake_256(b"QSNL-pok-weights" + seed).digest(8 * count)
    return [int.from_bytes(stream[8 * i : 8 * i + 8], "little") | 1 for i in range(count)]


def _msm_small(scalars: Sequence[int], points: Sequence[Point], bits: int = 64) -> Point:
    """``sum_i s_i P_i`` for ``bits``-bit scalars, sharing the doublings (Straus without windows)."""
    acc = IDENTITY
    for b in range(bits - 1, -1, -1):
        acc = point_add(acc, acc)
        for s, pt in zip(scalars, points):
            if (s >> b) & 1:
                acc = point_add(acc, pt)
    return acc


def _cofactor_eq(a: Point, b: Point) -> bool:
    return point_equal(scalar_mult(8, a), scalar_mult(8, b))


@dataclass
class MeasurementProof:
    n_qubits: int
    scale_bits: int
    pairs: list[tuple[int, int]]
    context: bytes
    commitments: list[bytes]
    shot_seed: bytes
    shots: np.ndarray
    claims: list[int]
    pok_t: bytes
    pok_s_m: int
    pok_s_r: int
    nonce_points: list[bytes]
    responses: list[int]
    delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "scale_bits": self.scale_bits,
            "pairs": [list(p) for p in self.pairs],
            "context": self.context,
            "commitments": self.commitments,
            "shot_seed": self.shot_seed,
            "shots": np.asarray(self.shots, dtype=np.int64),
            "claims": list(self.claims),
            "pok_t": self.pok_t,
            "pok_s_m": self.pok_s_m,
            "pok_s_r": self.pok_s_r,
            "nonce_points": self.nonce_points,
            "responses": list(self.responses),
            "delta": self.delta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MeasurementProof":
        return cls(
            int(d["n_qubits"]),
            int(d["scale_bits"]),
            [tuple(p) for p in d["pairs"]],  # type: ignore[misc]
            d["context"],
            list(d["commitments"]),
            d["shot_seed"],
            np.asarray(d["shots"], dtype=np.int64),
            [int(x) for x in d["claims"]],
            d["pok_t"],
            int(d["pok_s_m"]),
            int(d["pok_s_r"]),
            list(d["nonce_points"]),
            [int(x) for x in d["responses"]],
            float(d["delta"]),
        )

    def statement_names(self) -> list[str]:
        return [name for name, _ in _statements(self.n_qubits, self.pairs)]

    def proven_values(self) -> dict[str, float]:
        """Claimed functionals as probabilities (value / 2^B)."""
        return {k: v / 2**self.scale_bits for k, v in zip(self.statement_names(), self.claims)}


@dataclass
class ZKVerification:
    ok: bool
    checks: dict[str, bool]
    epsilon: float
    max_deviation: float
    n_statements: int
    details: dict[str, Any] = field(default_factory=dict)


def _base_transcript(n: int, scale_bits: int, pairs: Sequence[tuple[int, int]], context: bytes, commitments: Sequence[bytes]) -> Transcript:
    tr = Transcript(_PROTOCOL)
    tr.append("context", context)
    tr.append_int("n_qubits", n)
    tr.append_int("scale_bits", scale_bits)
    tr.append("pairs", encode([list(p) for p in pairs]))
    tr.append("commitments", b"".join(commitments))
    return tr


def prove_measurement(
    probs: np.ndarray,
    n_qubits: int,
    shots: int,
    context: bytes,
    pairs: Optional[Sequence[tuple[int, int]]] = None,
    scale_bits: int = 20,
    delta: float = 1e-9,
    rng: RandomSource | None = None,
) -> MeasurementProof:
    """Commit to the Born distribution, sample shots, and prove consistency."""
    rng = rng or default_rng()
    pairs = list(pairs) if pairs is not None else [(q, (q + 1) % n_qubits) for q in range(n_qubits)] if n_qubits > 2 else []
    g_tab, h_tab = pedersen_tables()
    p_int = quantize_probabilities(probs, scale_bits)
    comm_pts, blinds = pedersen_commit_vector([int(x) for x in p_int], rng)
    comm = [encode_point(c) for c in comm_pts]
    tr = _base_transcript(n_qubits, scale_bits, pairs, context, comm)

    shot_seed = tr.challenge_bytes("shot-seed", 32)
    outcomes = sample_outcomes(p_int / float(1 << scale_bits), shots, np.random.default_rng(int.from_bytes(shot_seed, "big")))
    tr.append("shots", outcomes.astype("<i8").tobytes())

    # (1) batched proof of knowledge of all openings (Okamoto).
    weights = _weights(tr.challenge_bytes("pok-weights", 32), len(comm))
    m_comb = sum(w * int(p) for w, p in zip(weights, p_int)) % L
    r_comb = sum(w * r for w, r in zip(weights, blinds)) % L
    a, b = random_scalar(rng), random_scalar(rng)
    t_pt = encode_point(point_add(g_tab.mult(a), h_tab.mult(b)))
    tr.append("pok-t", t_pt)
    c0 = tr.challenge_scalar("pok-challenge")
    s_m, s_r = (a + c0 * m_comb) % L, (b + c0 * r_comb) % L

    # (2) Schnorr proofs for every linear functional.
    stmts = _statements(n_qubits, pairs)
    claims = [int(p_int[mask].sum()) for _, mask in stmts]
    r_k = [sum(blinds[i] for i in np.nonzero(mask)[0]) % L for _, mask in stmts]
    t_k = [random_scalar(rng) for _ in stmts]
    nonces = [encode_point(h_tab.mult(t)) for t in t_k]
    tr.append("claims", encode(claims))
    tr.append("nonces", b"".join(nonces))
    c = tr.challenge_scalar("linear-challenge")
    responses = [(t + c * r) % L for t, r in zip(t_k, r_k)]
    return MeasurementProof(
        n_qubits, scale_bits, pairs, context, comm, shot_seed, outcomes, claims, t_pt, s_m, s_r, nonces, responses, delta
    )


def verify_measurement(proof: MeasurementProof, expected_context: Optional[bytes] = None) -> ZKVerification:
    """Check all algebraic relations and the statistical consistency of the shots."""
    checks: dict[str, bool] = {}
    n, B = proof.n_qubits, proof.scale_bits
    g_tab, h_tab = pedersen_tables()
    stmts = _statements(n, proof.pairs)
    fail = ZKVerification(False, checks, float("nan"), float("nan"), len(stmts))
    checks["context"] = expected_context is None or proof.context == expected_context
    checks["shapes"] = (
        len(proof.commitments) == 2**n
        and len(proof.claims) == len(stmts)
        and len(proof.responses) == len(stmts)
        and len(proof.nonce_points) == len(stmts)
        and proof.shots.ndim == 1
        and proof.shots.size > 0
        and bool(np.all((proof.shots >= 0) & (proof.shots < 2**n)))
    )
    if not checks["shapes"]:
        return fail
    try:
        comm_pts = [decode_point(c) for c in proof.commitments]
        t_pt = decode_point(proof.pok_t)
        nonce_pts = [decode_point(t) for t in proof.nonce_points]
    except Ed25519Error:
        checks["encodings"] = False
        return fail
    checks["encodings"] = True

    tr = _base_transcript(n, B, proof.pairs, proof.context, proof.commitments)
    checks["shot_seed_bound_to_commitments"] = tr.challenge_bytes("shot-seed", 32) == proof.shot_seed
    tr.append("shots", proof.shots.astype("<i8").tobytes())

    weights = _weights(tr.challenge_bytes("pok-weights", 32), len(comm_pts))
    combo = _msm_small(weights, comm_pts)
    tr.append("pok-t", proof.pok_t)
    c0 = tr.challenge_scalar("pok-challenge")
    lhs = point_add(g_tab.mult(proof.pok_s_m), h_tab.mult(proof.pok_s_r))
    checks["opening_knowledge"] = _cofactor_eq(lhs, point_add(t_pt, scalar_mult(c0, combo)))

    tr.append("claims", encode(list(proof.claims)))
    tr.append("nonces", b"".join(proof.nonce_points))
    c = tr.challenge_scalar("linear-challenge")
    linear_ok = True
    for (name, mask), v, t_k, s_k in zip(stmts, proof.claims, nonce_pts, proof.responses):
        d = IDENTITY
        for i in np.nonzero(mask)[0]:
            d = point_add(d, comm_pts[i])
        d = point_add(d, point_neg(g_tab.mult(v)))
        linear_ok &= _cofactor_eq(h_tab.mult(s_k), point_add(t_k, scalar_mult(c, d)))
    checks["linear_relations"] = bool(linear_ok)
    checks["normalised"] = proof.claims[0] == (1 << B)

    # Statistical consistency of the shots with the *proven* functionals.
    shots = proof.shots
    big_n = shots.size
    k_tests = len(stmts) - 1
    eps = math.sqrt(math.log(2 * k_tests / proof.delta) / (2 * big_n)) if k_tests else 0.0
    bits = (shots[:, None] >> (n - 1 - np.arange(n))) & 1
    scale = float(1 << B)
    # Statement order is fixed by _statements: [total, marginal_q0..q_{n-1}, even_parity per pair].
    devs = [abs(bits[:, q].mean() - proof.claims[1 + q] / scale) for q in range(n)]
    devs += [abs((bits[:, a] == bits[:, b]).mean() - proof.claims[1 + n + j] / scale) for j, (a, b) in enumerate(proof.pairs)]
    max_dev = float(max(devs)) if devs else 0.0
    checks["shots_consistent"] = max_dev <= eps
    ok = all(checks.values())
    return ZKVerification(ok, checks, eps, max_dev, len(stmts), {"shots": int(big_n), "delta": proof.delta})
