"""Quantum feature map: text tokens -> quantum states -> measurement features.

Pipeline for one document (all windows of all documents are simulated in a
single batched sweep):

1. **Windowing.**  The token sequence is cut into overlapping windows
   (``chunk_tokens`` long, ``chunk_stride`` apart).  Window 0 is the whole
   document (a "global" window, like a CLS token).
2. **Classical pre-processing.**  Each window becomes a sub-linear TF-IDF
   vector over the vocabulary, L2-normalised and projected to ``nL``
   coordinates -- by latent semantic analysis fitted on the public corpus (the
   default in the pipeline), or by a seeded Gaussian random projection.  A
   random projection to only ``nL ~ 16`` dimensions distorts cosine
   similarities by ``O(1/sqrt(nL)) ~ 0.25`` (Johnson-Lindenstrauss), which
   swamps the topic signal; LSA keeps the topical directions instead.
3. **Angle + ZZ encoding with data re-uploading.**  For each of ``L`` layers:
   ``RY(alpha_{l,q})`` on every qubit with ``alpha = pi * sigmoid(1.5 z)``,
   ``RZZ(beta_{l,q})`` on ring neighbours with the IQP product phase
   ``beta = (pi - alpha_q)(pi - alpha_{q+1}) / pi``, followed by a fixed random
   "reservoir" layer (``RX``, ``RZ`` with seeded angles) and a CNOT ring.
4. **Measurement features per window (exact state vector):**
   single-qubit Bloch vectors ``(<X_q>, <Y_q>, <Z_q>)``, ring correlators
   ``<Z_q Z_{q+1}>``, entanglement entropies of all contiguous cuts, plus
   shot-based marginals ``P(q = 1)`` with readout error.
5. **Noisy hardware probe (document window only, density matrix):** the same
   circuit under the configured gate-noise model; features are noisy Bloch
   vectors, purity ``Tr rho^2``, fidelity with the ideal state, the von Neumann
   entropy of the first half of the register and readout-noisy shot marginals.

The ideal document state ``|psi_doc>`` is also returned so the quantum node can
commit to its Born distribution and prove measurement consistency
(:mod:`quantasecure_neurollm.secure.zkp`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ..utils.mathx import clip_probabilities
from ..utils.rng import numpy_rng
from . import entanglement, kernels, measurement
from .circuit import QuantumCircuit, run_density, run_statevector
from .noise import NoiseModel

__all__ = ["QuantumFeatureMap", "QuantumFeatures", "QuantumFeatureExtractor"]


class QuantumFeatureMap:
    """Classical pre-processing + the parameterised encoding circuit.

    The projection from TF-IDF space to the ``n L`` encoding angles is either a
    seeded Gaussian random projection or, when ``fit_docs`` (public token
    sequences) are given, **latent semantic analysis**: the top ``n L`` right
    singular vectors of the TF-IDF matrix of those documents.  LSA keeps the
    directions along which public scientific text actually varies (topics), so
    the few available rotation angles carry semantic rather than random
    coordinates; any shortfall in rank is filled with seeded random directions
    orthogonal to the LSA subspace.  Projected coordinates are standardised with
    statistics of the fitting documents before the sigmoid squashing.
    """

    def __init__(
        self,
        n_qubits: int,
        layers: int,
        vocab_size: int,
        idf: Optional[np.ndarray] = None,
        projection_seed: int = 11,
        reservoir_seed: int = 7,
        fit_docs: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.vocab_size = vocab_size
        self.idf = np.ones(vocab_size) if idf is None else np.asarray(idf, dtype=np.float64)
        if self.idf.shape != (vocab_size,):
            raise ValueError("idf must have one entry per vocabulary item")
        k = n_qubits * layers
        rng = numpy_rng("feature-map-projection", projection_seed, vocab_size, n_qubits, layers)
        random_dirs = rng.standard_normal((vocab_size, k))
        self.proj_mean = np.zeros(k)
        self.proj_std = np.ones(k)
        if fit_docs is None:
            self.projection = random_dirs
            self.projection_kind = "gaussian-random"
        else:
            x = self.window_vectors(list(fit_docs))
            _, s, vt = np.linalg.svd(x, full_matrices=False)
            rank = int(np.sum(s > 1e-10 * s[0]))
            basis = vt[: min(k, rank)].T  # (V, r) orthonormal columns
            if basis.shape[1] < k:  # pad with random directions orthogonal to the LSA subspace
                extra = random_dirs[:, : k - basis.shape[1]]
                extra = extra - basis @ (basis.T @ extra)
                extra, _ = np.linalg.qr(extra)
                basis = np.concatenate([basis, extra], axis=1)
            signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(k)])
            self.projection = basis * np.where(signs == 0, 1.0, signs)  # deterministic sign convention
            z = x @ self.projection
            self.proj_mean = z.mean(axis=0)
            self.proj_std = z.std(axis=0) + 1e-6
            self.projection_kind = f"lsa(rank={min(k, rank)})"
        res = numpy_rng("feature-map-reservoir", reservoir_seed, n_qubits, layers)
        self.reservoir = res.uniform(-np.pi, np.pi, size=(layers, n_qubits, 2))
        self.n_ring = n_qubits if n_qubits > 2 else (1 if n_qubits == 2 else 0)
        self.circuit = self._build_circuit()

    @property
    def n_params(self) -> int:
        return self.layers * (self.n_qubits + self.n_ring)

    def _ring(self) -> list[tuple[int, int]]:
        n = self.n_qubits
        return [(q, (q + 1) % n) for q in range(self.n_ring)]

    def _build_circuit(self) -> QuantumCircuit:
        n, L = self.n_qubits, self.layers
        qc = QuantumCircuit(n, self.n_params, name=f"feature_map_{n}q_{L}L")
        k = 0
        for q in range(n):
            qc.h(q)
        for layer in range(L):
            for q in range(n):
                qc.ry(q, qc.p(k))
                k += 1
            for a, b in self._ring():
                qc.rzz(a, b, qc.p(k))
                k += 1
            for q in range(n):
                qc.rx(q, float(self.reservoir[layer, q, 0]))
                qc.rz(q, float(self.reservoir[layer, q, 1]))
            for a, b in self._ring():
                qc.cx(a, b)
        assert k == self.n_params
        return qc

    def window_vectors(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        """Sub-linear TF-IDF, L2-normalised: ``(B, V)``."""
        tf = np.zeros((len(windows), self.vocab_size))
        for i, w in enumerate(windows):
            ids, counts = np.unique(np.asarray(w, dtype=np.int64), return_counts=True)
            ids_ok = ids[(ids >= 0) & (ids < self.vocab_size)]
            counts_ok = counts[(ids >= 0) & (ids < self.vocab_size)]
            tf[i, ids_ok] = 1.0 + np.log(counts_ok)
        v = tf * self.idf
        nrm = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(nrm > 0, nrm, 1.0)

    def encode(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        """Circuit parameters ``(B, n_params)`` for a list of token windows."""
        z = (self.window_vectors(windows) @ self.projection - self.proj_mean) / self.proj_std
        alpha = np.pi / (1.0 + np.exp(-1.5 * z))  # (B, n L) in (0, pi)
        n, L = self.n_qubits, self.layers
        out = []
        for layer in range(L):
            a = alpha[:, layer * n : (layer + 1) * n]
            out.append(a)
            if self.n_ring:
                idx = np.array(self._ring())
                beta = (np.pi - a[:, idx[:, 0]]) * (np.pi - a[:, idx[:, 1]]) / np.pi
                out.append(beta)
        return np.concatenate(out, axis=1)


@dataclass
class QuantumFeatures:
    """Everything the quantum stage produces for one document."""

    chunk_features: np.ndarray  # (T, F) -- window 0 is the whole document
    global_features: np.ndarray  # (G,)  -- noisy density-matrix probe of the document window
    doc_state: np.ndarray  # (2^n,) ideal document state (stays inside the quantum node)
    stats: dict[str, Any] = field(default_factory=dict)


class QuantumFeatureExtractor:
    """Runs the feature map for many documents at once and assembles features."""

    def __init__(
        self,
        feature_map: QuantumFeatureMap,
        noise: NoiseModel,
        chunk_tokens: int = 40,
        chunk_stride: int = 32,
        max_chunks: int = 12,
        chunk_shots: int = 256,
        probe_shots: int = 1024,
        density_batch: int = 16,
    ) -> None:
        self.fmap = feature_map
        self.noise = noise
        self.chunk_tokens = chunk_tokens
        self.chunk_stride = chunk_stride
        self.max_chunks = max_chunks
        self.chunk_shots = chunk_shots
        self.probe_shots = probe_shots
        self.density_batch = density_batch
        n = feature_map.n_qubits
        self._pairs = [(q, (q + 1) % n) for q in range(feature_map.n_ring)]

    # ---------------------------------------------------------------- sizes
    @property
    def n_qubits(self) -> int:
        return self.fmap.n_qubits

    @property
    def chunk_dim(self) -> int:
        n = self.n_qubits
        return 3 * n + len(self._pairs) + (n - 1) + n

    @property
    def global_dim(self) -> int:
        return 4 * self.n_qubits + 3

    def feature_names(self) -> tuple[list[str], list[str]]:
        n = self.n_qubits
        chunk = [f"bloch_{a}{q}" for q in range(n) for a in "xyz"]
        chunk += [f"zz_{a}{b}" for a, b in self._pairs]
        chunk += [f"S_cut{k}" for k in range(1, n)]
        chunk += [f"shot_p1_q{q}" for q in range(n)]
        glob = [f"noisy_bloch_{a}{q}" for q in range(n) for a in "xyz"]
        glob += ["purity", "fidelity", "S_half_noisy"]
        glob += [f"noisy_shot_p1_q{q}" for q in range(n)]
        return chunk, glob

    # -------------------------------------------------------------- windowing
    def windows(self, token_ids: np.ndarray) -> list[np.ndarray]:
        ids = np.asarray(token_ids, dtype=np.int64)
        out = [ids]
        if len(ids) > self.chunk_tokens:
            starts = range(0, len(ids) - self.chunk_tokens + self.chunk_stride, self.chunk_stride)
            for s in starts:
                w = ids[s : s + self.chunk_tokens]
                if len(w) >= self.chunk_tokens // 4:
                    out.append(w)
                if len(out) >= self.max_chunks:
                    break
        return out

    # ---------------------------------------------------------------- features
    def _bloch(self, psi: np.ndarray) -> np.ndarray:
        """Bloch vectors of every qubit from pure states ``(B, 2^n)`` -> ``(B, 3n)``."""
        n = self.n_qubits
        b = psi.shape[0]
        out = np.empty((b, n, 3))
        for q in range(n):
            t = psi.reshape(b, 2**q, 2, 2 ** (n - q - 1))
            r01 = np.einsum("bas,bas->b", t[:, :, 0, :], t[:, :, 1, :].conj())
            p0 = np.sum(np.abs(t[:, :, 0, :]) ** 2, axis=(1, 2))
            p1 = np.sum(np.abs(t[:, :, 1, :]) ** 2, axis=(1, 2))
            out[:, q, 0] = 2 * r01.real
            out[:, q, 1] = -2 * r01.imag
            out[:, q, 2] = p0 - p1
        return out.reshape(b, 3 * n)

    def _bloch_dm(self, rho: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        b = rho.shape[0]
        out = np.empty((b, n, 3))
        for q in range(n):
            r = kernels.partial_trace(rho, [q], n)  # (B, 2, 2)
            out[:, q, 0] = 2 * r[:, 0, 1].real
            out[:, q, 1] = -2 * r[:, 0, 1].imag
            out[:, q, 2] = (r[:, 0, 0] - r[:, 1, 1]).real
        return out.reshape(b, 3 * n)

    def _cut_entropies(self, psi: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        b = psi.shape[0]
        out = np.zeros((b, n - 1))
        for k in range(1, n):
            s = np.linalg.svd(psi.reshape(b, 2**k, 2 ** (n - k)), compute_uv=False)
            w = s**2
            w = w / w.sum(axis=1, keepdims=True)
            safe = np.where(w > 1e-15, w, 1.0)
            out[:, k - 1] = -np.sum(np.where(w > 1e-15, w * np.log2(safe), 0.0), axis=1)
        return out

    def extract_many(self, docs: Sequence[np.ndarray], seed: int = 0) -> list[QuantumFeatures]:
        """Features for many documents (token-id arrays) in one batched simulation."""
        n = self.n_qubits
        all_windows: list[np.ndarray] = []
        spans: list[tuple[int, int]] = []
        for d in docs:
            w = self.windows(d)
            spans.append((len(all_windows), len(all_windows) + len(w)))
            all_windows.extend(w)
        params = self.fmap.encode(all_windows)
        psi, report = run_statevector(self.fmap.circuit, params)
        probs = measurement.born_probabilities(psi)
        bloch = self._bloch(psi)
        zz = measurement.parity_correlators(probs, n, self._pairs) if self._pairs else np.zeros((len(psi), 0))
        ent = self._cut_entropies(psi)
        rng = numpy_rng("chunk-shots", seed, len(all_windows))
        shot_marg = np.empty((len(psi), n))
        for i in range(len(psi)):
            idx = measurement.sample_outcomes(probs[i], self.chunk_shots, rng)
            idx = measurement.apply_readout_error(idx, n, self.noise.readout_error, rng)
            shot_marg[i] = measurement.marginal_frequencies(idx, n)
        chunk_feats = np.concatenate([bloch, zz, ent, shot_marg], axis=1)

        # Noisy density-matrix probe of each document window (window 0 of every span).
        doc_rows = np.array([s for s, _ in spans])
        glob = np.empty((len(docs), self.global_dim))
        probe_stats: list[dict[str, float]] = []
        prng = numpy_rng("probe-shots", seed, len(docs))
        for start in range(0, len(doc_rows), self.density_batch):
            rows = doc_rows[start : start + self.density_batch]
            rho = run_density(self.fmap.circuit, params[rows], noise=self.noise)
            nb = self._bloch_dm(rho)
            for j, r in enumerate(rows):
                di = start + j
                pur = entanglement.purity(rho[j])
                fid = entanglement.fidelity_pure_mixed(psi[r], rho[j])
                half = kernels.partial_trace(rho[j : j + 1], list(range(n // 2)), n)[0]
                s_half = entanglement.von_neumann_entropy(half)
                p = clip_probabilities(np.real(np.diagonal(rho[j])))
                idx = measurement.sample_outcomes(p, self.probe_shots, prng)
                idx = measurement.apply_readout_error(idx, n, self.noise.readout_error, prng)
                glob[di] = np.concatenate([nb[j], [pur, fid, s_half], measurement.marginal_frequencies(idx, n)])
                probe_stats.append({"purity": pur, "fidelity": fid, "S_half_noisy": s_half})

        results = []
        for di, (s, e) in enumerate(spans):
            stats = {
                "n_windows": e - s,
                "n_qubits": n,
                "gates": len(self.fmap.circuit),
                "depth": self.fmap.circuit.depth(),
                "max_norm_drift": report.max_norm_drift,
                "norm_error_bound": report.error_bound,
                "half_chain_entropy": float(ent[s, n // 2 - 1]) if n > 1 else 0.0,
                "mean_window_entropy": float(ent[s:e, n // 2 - 1].mean()) if n > 1 else 0.0,
                **probe_stats[di],
            }
            results.append(QuantumFeatures(chunk_feats[s:e].copy(), glob[di].copy(), psi[s].copy(), stats))
        return results

    def extract(self, token_ids: np.ndarray, seed: int = 0) -> QuantumFeatures:
        return self.extract_many([token_ids], seed=seed)[0]
