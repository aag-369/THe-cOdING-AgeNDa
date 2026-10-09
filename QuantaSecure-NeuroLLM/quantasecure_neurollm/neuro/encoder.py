"""Hybrid quantum-classical encoder: measurement features -> fixed-size latent.

Input
    * a variable-length sequence of per-window quantum feature vectors
      (Bloch vectors, ``<ZZ>`` correlators, entanglement entropies, shot
      marginals) -- shape ``(B, T, F)`` with a padding mask;
    * the document-level noisy-hardware probe -- shape ``(B, G)``.

Architecture::

    standardise features (statistics fitted on the public corpus, stored as buffers)
    tokens = [global_proj(g)] ++ chunk_proj(chunks)                         (B, 1+T, d)
    repeat n_blocks:
        x = x + MHA(RMSNorm(x), key_mask)          # classical multi-head attention
        x = x + UnitaryMesh(LayerNorm(x))          # norm-preserving quantum-inspired mixing
        x = x + SwiGLU(RMSNorm(x))
    pooled = masked attention pooling with a learned query                  (B, d)
    q      = QuantumLayer( (pi/2) tanh(W_q pooled) )                         (B, 2 n_q)   <- parameter shift
    z      = RMSNorm( W_f [pooled ; q] )                                     (B, d_latent)
    topic  = W_t z                                                           (auxiliary head)

The PQC sits on the main path to the latent, so the language-model loss
back-propagates *through* the quantum circuit (exact parameter-shift
Jacobians) into the classical layers below it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import NeuroConfig
from .autograd import Tensor, concat, softmax
from .layers import LayerNorm, Linear, Module, MultiHeadAttention, Parameter, RMSNorm, SwiGLU, UnitaryMeshLayer
from .quantum_layer import QuantumLayer

__all__ = ["EncoderBlock", "HybridQuantumEncoder", "EncoderOutput", "pad_chunks"]


def pad_chunks(chunks: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack variable-length ``(T_i, F)`` arrays into ``(B, T_max, F)`` plus a validity mask."""
    t_max = max(c.shape[0] for c in chunks)
    f = chunks[0].shape[1]
    out = np.zeros((len(chunks), t_max, f))
    mask = np.zeros((len(chunks), t_max), dtype=bool)
    for i, c in enumerate(chunks):
        out[i, : c.shape[0]] = c
        mask[i, : c.shape[0]] = True
    return out, mask


@dataclass
class EncoderOutput:
    latent: Tensor
    topic_logits: Tensor
    quantum_expectations: np.ndarray
    pool_weights: np.ndarray


class EncoderBlock(Module):
    def __init__(self, cfg: NeuroConfig, rng: np.random.Generator, out_gain: float) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, rng, out_gain=out_gain)
        self.norm_mesh = LayerNorm(cfg.d_model)
        self.mesh = UnitaryMeshLayer(cfg.d_model, cfg.mesh_depth, rng)
        self.norm_ffn = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, rng, out_gain=out_gain)

    def forward(self, x: Tensor, key_mask: np.ndarray) -> Tensor:
        x = x + self.attn(self.norm_attn(x), key_mask)
        x = x + self.mesh(self.norm_mesh(x)) * 0.5
        return x + self.ffn(self.norm_ffn(x))


class HybridQuantumEncoder(Module):
    def __init__(self, cfg: NeuroConfig, chunk_dim: int, global_dim: int, n_topics: int, seed: int = 0) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        out_gain = 1.0 / math.sqrt(2 * cfg.n_blocks)  # GPT-2 style residual scaling
        self.cfg = cfg
        self.chunk_proj = Linear(chunk_dim, cfg.d_model, rng)
        self.global_proj = Linear(global_dim, cfg.d_model, rng)
        self.blocks = [EncoderBlock(cfg, rng, out_gain) for _ in range(cfg.n_blocks)]
        self.final_norm = RMSNorm(cfg.d_model)
        self.pool_query = Parameter(rng.standard_normal(cfg.d_model) / math.sqrt(cfg.d_model))
        self.to_quantum = Linear(cfg.d_model, cfg.quantum_qubits, rng)
        self.quantum = QuantumLayer(cfg.quantum_qubits, cfg.quantum_layers, rng)
        self.fuse = Linear(cfg.d_model + self.quantum.n_outputs, cfg.d_latent, rng)
        self.out_norm = RMSNorm(cfg.d_latent)
        self.topic_head = Linear(cfg.d_latent, n_topics, rng)
        self.register_buffer("chunk_mean", np.zeros(chunk_dim))
        self.register_buffer("chunk_std", np.ones(chunk_dim))
        self.register_buffer("global_mean", np.zeros(global_dim))
        self.register_buffer("global_std", np.ones(global_dim))

    def fit_normalizer(self, chunks: Sequence[np.ndarray], globals_: np.ndarray) -> None:
        """Feature standardisation statistics (public corpus only)."""
        allc = np.concatenate(list(chunks), axis=0)
        self.register_buffer("chunk_mean", allc.mean(axis=0))
        self.register_buffer("chunk_std", allc.std(axis=0) + 1e-3)
        self.register_buffer("global_mean", globals_.mean(axis=0))
        self.register_buffer("global_std", globals_.std(axis=0) + 1e-3)

    def forward(self, chunks: np.ndarray, mask: np.ndarray, globals_: np.ndarray) -> EncoderOutput:
        c = Tensor((chunks - self.buffer("chunk_mean")) / self.buffer("chunk_std") * mask[..., None])
        g = Tensor((globals_ - self.buffer("global_mean")) / self.buffer("global_std"))
        b = chunks.shape[0]
        tokens = concat([self.global_proj(g).reshape(b, 1, -1), self.chunk_proj(c)], axis=1)
        key_mask = np.concatenate([np.ones((b, 1), dtype=bool), mask], axis=1)
        x = tokens
        for blk in self.blocks:
            x = blk(x, key_mask)
        x = self.final_norm(x)
        scores = (x @ self.pool_query.reshape(-1, 1)).reshape(b, -1) * (1.0 / math.sqrt(self.cfg.d_model))
        scores = scores + np.where(key_mask, 0.0, -1e9)
        w = softmax(scores, axis=-1)
        pooled = (w.reshape(b, -1, 1) * x).sum(axis=1)
        q_in = self.to_quantum(pooled).tanh() * (math.pi / 2)
        q_out = self.quantum(q_in)
        latent = self.out_norm(self.fuse(concat([pooled, q_out], axis=-1)))
        return EncoderOutput(latent, self.topic_head(latent), q_out.data.copy(), w.data.copy())

    def encode(self, chunks: Sequence[np.ndarray], globals_: np.ndarray) -> EncoderOutput:
        padded, mask = pad_chunks(chunks)
        return self.forward(padded, mask, np.atleast_2d(globals_))

    def topic_probabilities(self, out: EncoderOutput) -> np.ndarray:
        z = out.topic_logits.data
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
