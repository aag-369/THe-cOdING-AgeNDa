"""End-to-end demonstration.

    python -m quantasecure_neurollm.demo            # default configuration (trains once, then cached)
    python -m quantasecure_neurollm.demo --fast     # reduced configuration (same code paths)
    python -m quantasecure_neurollm.demo --retrain  # ignore the weight cache
    python -m quantasecure_neurollm.demo --aead ChaCha20-Poly1305

Exit status is 0 only if every security and correctness check passed.
The confidential document itself is never printed: the console belongs to the
operator, and the demo treats it like any other untrusted sink.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from .config import PipelineConfig, default_config, fast_config
from .corpus import CONFIDENTIAL_DOCUMENT, CONFIDENTIAL_TOPIC
from .pipeline import PipelineResult, SecureQuantumReasoningPipeline
from .quanta import MPS, ParameterizedQuantumCircuit, StateVector, reuploading_ansatz, ring_observables
from .training import load_or_train
from .utils.logging import configure_logging

RULE = "=" * 100


def _section(title: str) -> None:
    print(f"\n{RULE}\n  {title}\n{RULE}")


def simulator_self_test() -> dict[str, float]:
    """Quick numerical sanity checks shown at start-up (each also covered by the unit tests)."""
    out: dict[str, float] = {}
    sv = StateVector(16).h(0)
    for q in range(15):
        sv.cnot(q, q + 1)
    out["ghz16_entropy_bits"] = sv.entanglement_entropy(list(range(8)))
    mps = MPS(40, max_bond=4)
    mps.apply("h", [0])
    for q in range(39):
        mps.apply("cx", [q, q + 1])
    out["mps40_bond_entropy_bits"] = mps.bond_entropy(20)
    qc, n_in, n_w = reuploading_ansatz(3, 2)
    pqc = ParameterizedQuantumCircuit(qc, n_in, ring_observables(3))
    rng = np.random.default_rng(0)
    x, w = rng.uniform(-1, 1, (2, n_in)), rng.uniform(-np.pi, np.pi, n_w)
    _, jx, jw = pqc.jacobian(x, w)
    fx, fw = pqc.finite_difference_jacobian(x, w)
    out["param_shift_vs_fd_max_err"] = float(max(np.abs(jx - fx).max(), np.abs(jw - fw).max()))
    return out


def print_result(result: PipelineResult) -> None:
    _section("RESULT (decrypted by the data owner)")
    ans = result.answer
    if ans:
        print(f"  Topic           : {ans['topic']}  (encoder confidence {ans['topic_confidence']:.2f})")
        print(f"  Generated text  : {ans['text']}")
        print(f"  Generation      : {ans['tokens_generated']} tokens, mean log-prob {ans['mean_logprob']:.3f}, "
              f"stopped on <eos>: {ans['stopped_on_eos']}, sampling {ans['sampling']}")  # fmt: skip
    q = result.quantum_stats
    if q:
        print(f"  Quantum evidence: {q['n_windows']} windows on {q['n_qubits']} qubits, half-chain entanglement "
              f"{q['half_chain_entropy']:.3f} bits, noisy purity {q['purity']:.3f}, fidelity {q['fidelity']:.3f}")  # fmt: skip
        print(f"                    ZK-proven qubit marginals P(1) = {q['proven_marginals']}")

    _section("PROVENANCE CHAIN (hybrid Ed25519 + ML-DSA signed, hash-commitment linked)")
    for r in result.provenance:
        print(f"  {r['stage']:<17} by {r['node']:<17} in={r['input_commitment'][:16]}..  out={r['output_commitment'][:16]}..  "
              f"model={r.get('model_digest', '-')}")  # fmt: skip

    _section("VERIFICATION")
    for group, checks in (("owner", result.owner_checks), ("zero-knowledge", result.zk_checks), ("wire audit", result.audit.checks)):
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {group:<15} {k}")
    d = result.audit.details
    if d:
        c = d["ciphertext"]
        print(f"\n  wire: {d['frames']} frames ({d['frames_by_channel']}), {d['wire_bytes']:,} bytes; "
              f"{d['message_keys']} distinct message keys; DH+KEM ratchet steps per endpoint: {d['dh_ratchet_steps']}")  # fmt: skip
        print(f"  ciphertext: {c['bytes']:,} bytes, entropy {c['entropy_bits_per_byte']:.4f} bits/byte, "
              f"chi2={c['chi2']:.1f} (p={c['chi2_p_value']:.3f}); plaintext n-gram hits on wire: {d['leak_hits']}")  # fmt: skip
    print("\n  timings: " + ", ".join(f"{k} {v:.2f}s" for k, v in result.timings.items()))
    if result.errors:
        print("\n  ERRORS: " + "; ".join(result.errors))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QuantaSecure-NeuroLLM end-to-end demonstration")
    ap.add_argument("--fast", action="store_true", help="reduced configuration (smaller models, 6 qubits)")
    ap.add_argument("--retrain", action="store_true", help="ignore cached weights and retrain")
    ap.add_argument("--aead", choices=["AES-256-GCM", "ChaCha20-Poly1305"], default=None)
    ap.add_argument("--quiet", action="store_true", help="only warnings from the component loggers")
    args = ap.parse_args(argv)

    configure_logging(logging.WARNING if args.quiet else logging.INFO)
    config: PipelineConfig = fast_config() if args.fast else default_config()
    if args.aead:
        config = replace(config, crypto=replace(config.crypto, aead=args.aead))

    _section("QuantaSecure-NeuroLLM :: secure quantum-enhanced reasoning over a confidential document")
    print("  configuration:")
    print("\n".join("    " + line for line in config.to_yaml().splitlines()))

    _section("PHASE 0 :: simulator self-test")
    for k, v in simulator_self_test().items():
        print(f"  {k:<28} {v:.3e}" if "err" in k else f"  {k:<28} {v:.6f}")

    _section("PHASE 1 :: model preparation (public corpus only; no pre-trained weights)")
    t0 = time.perf_counter()
    art = load_or_train(config, retrain=args.retrain, progress=lambda s: print("  " + s, flush=True))
    print(f"  ready in {time.perf_counter() - t0:.1f}s  (weights fingerprint {art.fingerprint}, cached={art.from_cache})")
    print(f"  held-out metrics: {art.metrics}")

    _section("PHASE 2 :: secure pipeline run")
    doc_bytes = CONFIDENTIAL_DOCUMENT.encode()
    print(f"  confidential document loaded: {len(doc_bytes)} bytes, SHA3-256 {hashlib.sha3_256(doc_bytes).hexdigest()[:24]}.. (content not displayed)")
    result = SecureQuantumReasoningPipeline(config, art).run(CONFIDENTIAL_DOCUMENT)

    print_result(result)
    relevant = result.answer.get("topic") == CONFIDENTIAL_TOPIC
    _section("OUTCOME")
    print(f"  answer topic matches the document's true topic: {relevant}")
    print(f"\n  >>> {result.message}\n")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
