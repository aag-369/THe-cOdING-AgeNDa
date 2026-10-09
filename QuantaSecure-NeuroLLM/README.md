# QuantaSecure-NeuroLLM

A self-contained research prototype that pushes a **confidential scientific document** through

1. a **pure-NumPy quantum simulator** (state vector, density matrix with noise, MPS tensor network,
   parameterised circuits with exact parameter-shift gradients),
2. a **hybrid quantum-classical neural encoder** (attention, RMSNorm/LayerNorm, SwiGLU, a
   norm-preserving unitary MZI-mesh layer, and a trainable PQC layer back-propagated with the
   parameter-shift rule),
3. a **from-scratch decoder-only language model** (byte-level BPE, RoPE, causal attention with KV
   cache, SwiGLU, RMSNorm, soft-prompt conditioning, temperature / top-k / nucleus sampling),

where **every hop between modules is protected by a from-scratch post-quantum secure channel**:
ML-KEM + X25519 hybrid handshake signed with Ed25519 + ML-DSA, a Double Ratchet that mixes a fresh
ML-KEM secret into every DH step, AES-256-GCM or ChaCha20-Poly1305, salted hash commitments on every
intermediate tensor, and a **zero-knowledge proof** that the reported measurement shots are
consistent with a Pedersen-committed quantum state.

No quantum SDK, no ML framework, no pre-trained weights and no third-party crypto library is used at
run time: only the Python standard library, NumPy and SciPy.

```
data-owner --(token ids)--> quantum-processor --(features + ZK proof)--> neural-encoder
     ^                                                                         |
     +-----------------(answer + signed provenance)-- language-model <--(latent)+
            every arrow: AEAD frame under a fresh ratchet key; every frame recorded and audited
```

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python run_demo.py                # default config: 8-qubit feature map, ML-KEM-768, ML-DSA-44, AES-256-GCM
python run_demo.py --fast         # smaller models / 6 qubits, same code paths
python run_demo.py --aead ChaCha20-Poly1305
python run_demo.py --retrain      # ignore the weight cache

python -m pytest                  # full suite (107 tests; the end-to-end tests train the fast config once)
python -m pytest -m "not slow"    # unit tests only (~10 s)
```

The first run trains the encoder and the language model in-process on the **public** corpus
(about 2 minutes on a laptop CPU for the default configuration, ~1 minute for `--fast`) and caches
the weights under `.qsnl_cache/<config-fingerprint>.npz`. Subsequent runs start in seconds. The
secure pipeline run itself takes a few seconds (about 1.5 s for `--fast`, about 3.5 s by default).

The demo exits with status 0 only if every check passes, and then prints:

> Secure quantum-enhanced reasoning completed. Original document never exposed in plaintext. All
> channel keys rotated. Proof of correct measurement verified.

The confidential document is never printed. The console belongs to the operator, and the demo
treats it like any other untrusted sink.

---

## Package layout

```
quantasecure_neurollm/
  config.py            frozen dataclasses for every hyper-parameter (+ YAML-like dump, cache fingerprint)
  corpus.py            hard-coded public corpus (6 abstracts + summaries), the confidential preprint,
                       held-out evaluation passages (all original prose)
  training.py          in-process pre-training on the public corpus, weight cache
  pipeline.py          enclave nodes, provenance chain, owner-side verification, orchestrator
  audit.py             post-run wire audit: plaintext-leak scan, key freshness, active probes
  demo.py              runnable demonstration (also: ../run_demo.py)
  quanta/              quantum simulation
    gates.py             gate library (vectorised constructors, parameter-shift metadata)
    kernels.py           batched gate / channel kernels, memory budget, error bound
    circuit.py           symbolic circuits, parameter binding, state-vector / density drivers
    statevector.py       StateVector class (measurement with collapse, sampling, entropies)
    density.py           DensityMatrix class (Kraus channels, purity, entropies)
    noise.py             Kraus sets + per-gate NoiseModel (depolarising, T1, T2, readout)
    measurement.py       Pauli strings, exact expectations, shot sampling, marginals
    entanglement.py      Schmidt / von Neumann / Renyi-2 / mutual information
    mps.py               matrix-product-state backend (SVD truncation, perfect sampling)
    pqc.py               parameterised circuits + exact parameter-shift Jacobians
    feature_map.py       text -> LSA angles -> re-uploading circuit -> measurement features
  neuro/               neural network
    autograd.py          reverse-mode autodiff on NumPy (fused softmax / cross-entropy)
    layers.py            Module system, Linear, RMSNorm, LayerNorm, SwiGLU, MHA, UnitaryMeshLayer
    quantum_layer.py     PQC as an autograd op (parameter-shift backward)
    encoder.py           HybridQuantumEncoder
    optim.py             AdamW, gradient clipping, cosine schedule
  llm/                 language model
    tokenizer.py         byte-level BPE (deterministic training) ; bpe_merges.py = hard-coded table
    rope.py              rotary position embeddings
    model.py             decoder-only transformer, soft-prompt adapter, KV cache
    generation.py        temperature / top-k / top-p / repetition-penalty sampling
  secure/              cryptography and the channel
    mlkem.py  mldsa.py  ntt.py           FIPS 203 / FIPS 204 lattice schemes, vectorised NTT
    x25519.py  ed25519.py                 RFC 7748 / RFC 8032 (+ edwards25519 group for Pedersen)
    aes_gcm.py  chacha20_poly1305.py  aead.py  kdf.py
    hybrid.py  handshake.py  ratchet.py  channel.py
    commitments.py  transcript.py  zkp.py
    constant_time.py  rng.py
  utils/               canonical serialization, hash-chained audit log, numerics, seeding
scripts/build_bpe_merges.py   regenerates the hard-coded BPE table from the public corpus
tests/                        quanta, neuro, llm, crypto primitives, crypto protocols, end-to-end
```

---

## Mathematical foundations

### Quantum simulator (`quanta/`)

**State vector.** An `n`-qubit state is `psi in C^(2^n)` with **big-endian** qubit order (qubit 0 is
the most significant bit). A `k`-qubit gate `U` is applied by viewing `psi` as a rank-`n` tensor,
moving the `k` target axes last, multiplying by `U^T` and moving the axes back: `O(2^n 2^k)` flops.
Diagonal gates (`RZ`, `CZ`, `RZZ`, phase) use an `O(2^n)` broadcast, and permutation gates (`X`,
`CNOT`, `SWAP`, Toffoli) use a single index gather. Every kernel is **batched**: `B` circuits with
different parameters run in one sweep. The feature map evaluates every window of every document at
once, and the parameter-shift rule evaluates all `1 + 2M` shifted circuits at once. Default memory
guard: warn above 256 MiB, refuse above 4 GiB (16 qubits = 1 MiB; 24 qubits = 256 MiB).

**Density matrices and noise.** `rho` is treated as a `2n`-qubit vector, so `U rho U^dagger` is the
superoperator `U (x) conj(U)` acting on the (row, column) axes of the target qubits in one pass.
Channels are given in Kraus form `rho -> sum_m K_m rho K_m^dagger` (trace preservation
`sum K^dagger K = I` is tested) and in equivalent closed forms used on the hot path:

| channel | action |
|---|---|
| depolarising (k qubits) | `rho -> (1-p) rho + p I/2^k (x) Tr_k rho` |
| amplitude damping (T1) | `rho_00 += g rho_11`, `rho_11 *= 1-g`, coherences `*= sqrt(1-g)` |
| phase damping (T2) | coherences `*= sqrt(1-l)` |
| readout error | independent classical bit flips after sampling |

**Measurement.** Born probabilities are `|psi_i|^2` (clipped and renormalised onto the simplex).
Shots use inverse-CDF sampling. Exact Pauli expectations never form a `2^n x 2^n` matrix. Projective
single-qubit measurement collapses the state.

**Entanglement.** For a bipartition `A|B` of a pure state, the Schmidt coefficients are the singular
values of `psi` reshaped to `2^|A| x 2^|B|`, and `S(A) = -sum s_i^2 log2 s_i^2`. The SVD keeps
absolute accuracy `O(u)`, whereas `eigvalsh(rho_A)` would square the condition number. Mixed-state
entropies use `eigvalsh` with sub-`1e-12` eigenvalues dropped (an effect of at most ~4e-11 bits each).

**Tensor networks.** `MPS` stores site tensors `(chi, 2, chi')`. Two-qubit gates contract
neighbouring sites, apply `U`, SVD, and truncate to `max_bond`. Non-adjacent gates are SWAP-routed.
Truncation accumulates the discarded weight `eps = sum_k eps_k`, giving fidelity
`F >= 1 - eps` to first order. Expectations are exact transfer-matrix contractions, and sampling is
perfect (sequential conditionals on the right-canonical form). The tests run a 40-qubit GHZ state
with bond dimension 2.

**Parameter-shift rule.** Every parameterised gate is `exp(-i theta G)` with `G` having eigenvalues
`+-1/2`, so `d<O>/dtheta = [<O>(theta+pi/2) - <O>(theta-pi/2)] / 2` **exactly**. Parameters that
appear in several gates (with scales `c_m`) get `sum_m c_m d<O>/dtheta_m`. The tests agree with
central finite differences to `1e-8`, and with shot-based estimates in expectation.

**Floating-point error analysis.** A dense `k`-qubit gate computes each amplitude as a length-`2^k`
inner product. By Higham's bound, the 2-norm drifts by at most `~(2^k + 2) u` per gate (`u = 2^-53`),
so after `G` gates the drift is at most `G (2^k + 2) u` (7e-13 for 1000 two-qubit gates). Every
simulation reports the observed drift next to this a-priori bound and renormalises above `1e-10`. The
tests check `observed <= bound` on an 800-gate, 10-qubit random circuit, and the demo logs both values
(e.g. `6.7e-16` vs `4.4e-14`).

**Feature map.** Token windows become sub-linear TF-IDF vectors. These are projected onto the top
`nL` **LSA** directions of the public corpus (a random projection to only ~16 dimensions distorts
cosine similarity by `~1/sqrt(16)`, which swamps the topic signal) and squashed to angles
`alpha = pi * sigmoid(1.5 z)`. Each of the `L` layers then applies `RY(alpha)`, IQP-style
`RZZ((pi - alpha_q)(pi - alpha_q+1)/pi)` on a ring, a fixed seeded reservoir of `RX`/`RZ`
rotations, and a CNOT ring. Per-window features are Bloch vectors, ring `<ZZ>`, entanglement entropy
of every contiguous cut, and readout-noisy shot marginals. The document-level noisy density-matrix
probe adds noisy Bloch vectors, purity, fidelity with the ideal state, half-register entropy and
noisy marginals.

### Neural encoder (`neuro/`)

A reverse-mode autograd engine (iterative topological sort; fused softmax / log-softmax /
cross-entropy / SiLU; `custom_op` for external Jacobians) underlies everything. Its gradients are
verified against finite differences in `tests/test_neuro.py`.

* **UnitaryMeshLayer.** The `d` real features are read as `d/2` complex modes, and a brick-wall mesh
  of Mach-Zehnder interferometers applies
  `U(theta, phi) = [[e^{i phi} cos theta, -sin theta], [e^{i phi} sin theta, cos theta]]`.
  Each block is unitary, so the layer preserves norms exactly (tested to `1e-12`) and, at sufficient
  depth, can represent any unitary (Reck/Clements).
* **QuantumLayer.** A data re-uploading PQC whose backward pass contracts the incoming gradient with
  the **exact parameter-shift Jacobians** w.r.t. both inputs and weights.
* **HybridQuantumEncoder.** Standardise features; add a global-probe token; run blocks of
  `x + MHA(RMSNorm x)`, `x + Mesh(LayerNorm x)`, `x + SwiGLU(RMSNorm x)`; masked attention pooling;
  `QuantumLayer((pi/2) tanh(W pooled))`; fuse into a latent `z`; add an auxiliary topic head. The LM
  loss back-propagates *through* the PQC.

### Language model (`llm/`)

Byte-level BPE (256 base bytes, 320 merges learned deterministically from the public corpus,
5 special tokens). Pre-norm decoder blocks:
`h = x + W_o Attn(RoPE(q), RoPE(k), v)` and `y = h + SwiGLU(RMSNorm h)`, with an output head tied to
the embedding. RoPE rotates pairs by `m * base^(-2i/d)`, so `<R_m q, R_n k>` depends only on `n - m`
(tested). The encoder latent `z` is mapped by a `SoftPromptAdapter` to `P` prefix vectors.
Generation uses a KV cache (tested equal to the full forward pass to `1e-10`) with repetition
penalty, temperature, top-k and nucleus filtering.

### Cryptography (`secure/`)

| component | construction | assumption |
|---|---|---|
| KEM | **ML-KEM-512/768/1024** (FIPS 203): Module-LWE K-PKE + Fujisaki-Okamoto with implicit rejection | MLWE, QROM |
| signatures | **ML-DSA-44/65/87** (FIPS 204): Fiat-Shamir with aborts, hints, `Power2Round` | MLWE + SelfTargetMSIS, QROM |
| classical | **X25519** (RFC 7748), **Ed25519** (RFC 8032) | CDH / DL on Curve25519 |
| AEAD | **AES-256-GCM** (FIPS 197 + SP 800-38D) or **ChaCha20-Poly1305** (RFC 8439) | PRP / PRF + universal hashing |
| KDF | HMAC-SHA-256, **HKDF** (RFC 5869) | HMAC is a (dual) PRF |
| hybrid signature | `Ed25519 AND ML-DSA` over a domain-separated message | either scheme unforgeable |
| handshake | signed `X25519 + ML-KEM` ephemeral exchange; `PRK = HKDF-Extract(H(transcript), ss_kem || ss_dh)`; Finished MACs both ways | either KEM secure **and** signatures unforgeable |
| ratchet | Signal Double Ratchet; each DH step mixes `DH || ML-KEM(peer ratchet key)` into the root KDF; fresh message key per frame; bounded skipped keys; transactional decrypt | as above |
| commitments | salted SHA3-256 hash commitments on every intermediate tensor; **Pedersen** `mG + rH` on edwards25519 with hash-to-curve `H` | collision resistance / ROM; DL |
| ZK proof | see below | DL, ROM |

Both lattice schemes share one vectorised NTT (7 layers mod 3329 with zeta=17 for ML-KEM, 8 layers
mod 8380417 with zeta=1753 for ML-DSA), cross-checked against schoolbook negacyclic multiplication.

**Measurement-consistency NIZK** (`secure/zkp.py`). The quantum node quantises the Born distribution
of its document state to integers `p_i` summing exactly to `2^20` and publishes Pedersen commitments
`C_i = p_i G + r_i H`. It samples the shots from the *committed* distribution, using a seed derived
by Fiat-Shamir **after** the commitments, and proves:

1. knowledge of all openings, with a batched Okamoto proof on `sum rho_i C_i` with FS weights;
2. for every claimed linear functional (total mass = `2^20`, every qubit marginal, every ring parity
   `<Z_a Z_b>`), that `sum_{i in S} C_i - v G = R H` (Schnorr proof of knowledge of `R`). A false
   `v` would reveal `log_H G`.

The verifier finally checks that each empirical shot marginal or parity lies within
`eps = sqrt(ln(2K/delta) / 2N)` of the proven value (Hoeffding + union bound, `delta = 1e-9`).
Pedersen commitments are perfectly hiding and the Sigma protocols are HVZK, so nothing beyond the
claimed statistics is revealed. The neural node refuses to proceed if the proof fails, and the data
owner re-verifies it independently.

---

## Security claims

**Threat model.** A network adversary that reads, drops, reorders, replays and modifies every frame
on the bus, a *future* adversary with a quantum computer recording traffic today, and an
authenticated but **dishonest quantum node** that might misreport measurement statistics.

| claim | mechanism | how it is checked |
|---|---|---|
| confidentiality of the document, tokens and all tensors in transit | hybrid PQ handshake + ratchet + AEAD | wire auditor: zero 12-byte n-grams of the document / token ids (int16/32/64) in any frame; per-seal egress guard; independent check in `test_end_to_end.py`; ciphertext entropy ~8 bits/byte |
| harvest-now-decrypt-later resistance | ML-KEM in the handshake **and** in every ratchet step | KATs + byte-exact cross-validation |
| mutual authentication, no MITM | pinned hybrid identity keys, signed hellos, Finished MACs | tests: impostor, swapped ephemeral key, re-addressed hello, downgrade, bad Finished |
| a fresh key for every message; post-compromise recovery | symmetric chain per message, DH+KEM ratchet per direction change | auditor: all message-key fingerprints distinct; every endpoint's ratchet advanced |
| integrity, replay protection | AEAD over header + routing metadata; consumed keys; state rollback on failure | live probes: tampered frame rejected, genuine accepted afterwards, replay rejected |
| auditable chain of custody | signed stage records linked by hash commitments; hash-chained audit log | owner verification; `audit_log_chain_intact` |
| correct measurement reporting | Pedersen-committed state + NIZK + Hoeffding test | lying-prover test fails closed |

**Not claimed** (read this before reusing any code):

* **Side channels.** Pure-Python big integers, NumPy and table-based AES are **not constant-time**.
  The model covers network attackers only. See `secure/constant_time.py` for exactly what is and
  isn't mitigated.
* **Enclaves are logical, not hardware.** All four nodes run in one Python process. A node
  necessarily sees its own decrypted input (the quantum node sees token ids). Hardware isolation (SGX,
  SEV, TDX) is out of scope.
* **Metadata** (who talks to whom, sizes, timing) is visible, as in TLS.
* **The ZK proof** has no range proofs (`p_i >= 0` is not proven), its shot test is statistical, and
  a prover can grind blinding factors to re-roll the shot seed. It proves consistency with a
  *committed* distribution, not that the circuit was the advertised one.
* **Trust store provisioning** is assumed honest (the orchestrator pins keys out of band).
* Implementations follow FIPS 203/204 and the RFCs and were cross-checked byte-for-byte against
  independent references during development (`cryptography`, `kyber-py`, `dilithium-py`). The
  resulting known-answer digests are pinned in `tests/test_secure_primitives.py`. This is still
  research code and has not been audited.

---

## What the language model can and cannot do

The LM is a **~150k-parameter** transformer trained for a couple of minutes on six public abstracts. It
does not summarise arbitrary text. It learns to map regions of the quantum-feature latent space to
fluent, topic-faithful scientific prose (here: structured *Finding / Method / Implication* summaries)
and to continue scientific text in the style of its corpus. For the confidential preprint (a held-out
document on surface-code error correction), the pipeline produces the corresponding domain summary.
The tests check that the answer is non-empty, names the correct topic, and shares at least four
domain keywords with the document, and that at least two of the three unseen evaluation passages from
other topics are routed to the right topic. Novel,
document-specific abstraction would need more data and compute. The architecture and code paths need
no change.

"Quantum-enhanced" here means *quantum-simulated features* (entanglement entropies, correlators and
noisy-hardware statistics of an encoding circuit). No quantum advantage is claimed.

---

## Configuration

All hyper-parameters live in `quantasecure_neurollm/config.py` as frozen dataclasses
(`QuantumConfig`, `NoiseConfig`, `NeuroConfig`, `LLMConfig`, `TrainingConfig`, `CryptoConfig`). The
demo prints the full configuration as YAML at start-up. `PipelineConfig.training_fingerprint()`
hashes every training-relevant field plus the corpus and code version, so a stale weight cache can
never be loaded silently. Weights are loaded with `np.load(allow_pickle=False)`.

```python
from dataclasses import replace
from quantasecure_neurollm.config import default_config
from quantasecure_neurollm.training import load_or_train
from quantasecure_neurollm.pipeline import SecureQuantumReasoningPipeline

cfg = default_config()
cfg = replace(cfg, crypto=replace(cfg.crypto, kem="ML-KEM-1024", aead="ChaCha20-Poly1305"))
art = load_or_train(cfg)
result = SecureQuantumReasoningPipeline(cfg, art).run("your confidential text ...")
print(result.success, result.answer["text"], result.audit.checks)
```

Setting `crypto.deterministic_seed` switches all key generation to a SHAKE-256 DRBG.
**This is for tests only.**

## Reproducing the hard-coded vocabulary

```bash
python scripts/build_bpe_merges.py
```

This regenerates `llm/bpe_merges.py` from the public corpus. `tests/test_llm.py` asserts that the
committed table equals a fresh deterministic training run and records the corpus hash it came from.
