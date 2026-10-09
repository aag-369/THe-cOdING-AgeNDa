"""All hyper-parameters, hard-coded as frozen dataclasses.

The configuration is a tree of immutable dataclasses.  :meth:`PipelineConfig.to_yaml`
renders it as YAML-like text (logged at start-up so every run is
self-describing) and :meth:`PipelineConfig.training_fingerprint` hashes exactly
the fields that influence trained weights, which keys the on-disk model cache.

Two presets are provided:

* :func:`default_config` -- the demo configuration (8-qubit feature map,
  ML-KEM-768, ML-DSA-44, AES-256-GCM, ~1-2 minutes of in-process training).
* :func:`fast_config` -- a reduced configuration for the tests (same code
  paths, smaller models and fewer training steps).
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from .utils.serialization import encode

__all__ = [
    "NoiseConfig",
    "QuantumConfig",
    "NeuroConfig",
    "LLMConfig",
    "TrainingConfig",
    "CryptoConfig",
    "PipelineConfig",
    "default_config",
    "fast_config",
]


@dataclass(frozen=True)
class NoiseConfig:
    """Hardware-noise model for the density-matrix probe (per-gate channels)."""

    enabled: bool = True
    depolarizing_1q: float = 0.002  # p in  rho -> (1-p) rho + p I/2  after each 1-qubit gate
    depolarizing_2q: float = 0.012  # p in  rho -> (1-p) rho + p I/4  after each 2-qubit gate
    amplitude_damping: float = 0.004  # gamma (T1-type decay) after every gate, on its qubits
    phase_damping: float = 0.003  # lambda (pure dephasing, T2-type) after every gate
    readout_error: float = 0.01  # symmetric classical bit-flip probability at measurement


@dataclass(frozen=True)
class QuantumConfig:
    """Quantum feature map / simulator settings."""

    n_qubits: int = 8
    reupload_layers: int = 2  # data re-uploading depth of the feature map
    chunk_tokens: int = 40  # tokens per chunk (sliding window)
    chunk_stride: int = 32
    max_chunks: int = 12
    shots: int = 2048  # shots of the committed (ideal) document circuit, used by the ZK proof
    probe_shots: int = 1024  # shots of the noisy density-matrix probe
    reservoir_seed: int = 7  # seeds the fixed (non-trained) variational angles of the feature map
    projection_seed: int = 11  # seeds the classical random projection of token statistics
    renormalize_tol: float = 1e-10  # re-normalise a state when | ||psi|| - 1 | exceeds this
    mps_max_bond: int = 32
    mps_cutoff: float = 1e-12


@dataclass(frozen=True)
class NeuroConfig:
    """Hybrid quantum-classical encoder."""

    d_model: int = 64
    n_heads: int = 4
    n_blocks: int = 2
    ffn_hidden: int = 128
    mesh_depth: int = 4  # brick-wall depth of the unitary (MZI-mesh) layer
    d_latent: int = 64
    quantum_qubits: int = 4  # qubits of the trainable PQC layer inside the encoder
    quantum_layers: int = 2  # data re-uploading layers of that PQC


@dataclass(frozen=True)
class LLMConfig:
    """Decoder-only transformer + sampling defaults."""

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ffn_hidden: int = 176
    max_seq_len: int = 160
    n_prefix: int = 4  # soft-prompt vectors produced from the encoder latent
    rope_base: float = 10000.0
    temperature: float = 0.6
    top_k: int = 12
    top_p: float = 0.9
    max_new_tokens: int = 110
    repetition_penalty: float = 1.05


@dataclass(frozen=True)
class TrainingConfig:
    """In-process pre-training on the *public* corpus only."""

    seed: int = 1234
    views_per_doc: int = 10
    steps: int = 320
    batch_size: int = 12
    lr: float = 3e-3
    min_lr: float = 2e-4
    warmup_steps: int = 30
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    topic_loss_weight: float = 0.5
    text_loss_weight: float = 0.35
    text_window: int = 64  # tokens per raw-abstract crop used for conditioned language modelling
    latent_noise: float = 0.3  # std of Gaussian noise added to the latent before it conditions the LM
    log_every: int = 50
    cache_dir: str = ".qsnl_cache"


@dataclass(frozen=True)
class CryptoConfig:
    """Secure-channel suite."""

    kem: str = "ML-KEM-768"  # or "ML-KEM-512" / "ML-KEM-1024"
    signature: str = "ML-DSA-44"  # or "ML-DSA-65" / "ML-DSA-87"
    aead: str = "AES-256-GCM"  # or "ChaCha20-Poly1305"
    kem_ratchet: bool = True  # mix a fresh ML-KEM secret into every DH-ratchet step
    max_skip: int = 64  # Double Ratchet bound on skipped message keys
    pedersen_scale_bits: int = 20  # Born probabilities committed as integers summing to 2^bits
    zk_delta: float = 1e-9  # failure probability of the Hoeffding consistency test
    leak_window: int = 12  # n-gram length used by the plaintext-leak auditor
    #: TEST ONLY.  When set, all key generation uses a deterministic SHAKE-256 DRBG.
    deterministic_seed: Optional[int] = None


@dataclass(frozen=True)
class PipelineConfig:
    """Root of the configuration tree."""

    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    neuro: NeuroConfig = field(default_factory=NeuroConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    generation_seed: int = 2024

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        """Render the configuration as YAML-like text."""

        def render(d: dict[str, Any], indent: int) -> list[str]:
            lines: list[str] = []
            for k, v in d.items():
                if isinstance(v, dict):
                    lines.append(" " * indent + f"{k}:")
                    lines.extend(render(v, indent + 2))
                else:
                    lines.append(" " * indent + f"{k}: {v}")
            return lines

        return "\n".join(render(self.as_dict(), 0))

    def training_fingerprint(self, corpus_digest: bytes, code_version: str) -> str:
        """Hash of every field that influences trained weights (model-cache key)."""
        relevant = {
            "quantum": dataclasses.asdict(self.quantum),
            "noise": dataclasses.asdict(self.noise),
            "neuro": dataclasses.asdict(self.neuro),
            "llm": {k: v for k, v in dataclasses.asdict(self.llm).items() if k not in _SAMPLING_ONLY},
            "training": {k: v for k, v in dataclasses.asdict(self.training).items() if k not in ("log_every", "cache_dir")},
            "corpus": corpus_digest,
            "code": code_version,
        }
        return hashlib.sha3_256(encode(relevant)).hexdigest()[:24]


_SAMPLING_ONLY = ("temperature", "top_k", "top_p", "max_new_tokens", "repetition_penalty")


def default_config() -> PipelineConfig:
    """Configuration used by ``demo.py``."""
    return PipelineConfig()


def fast_config() -> PipelineConfig:
    """Smaller-but-complete configuration for tests (identical code paths)."""
    base = PipelineConfig()
    return replace(
        base,
        quantum=replace(base.quantum, n_qubits=6, shots=1024, probe_shots=512, max_chunks=8),
        neuro=replace(base.neuro, d_model=32, n_heads=2, n_blocks=1, ffn_hidden=64, d_latent=32, mesh_depth=2, quantum_qubits=3),
        llm=replace(base.llm, d_model=48, n_heads=2, n_layers=2, ffn_hidden=128, max_seq_len=144),
        training=replace(base.training, views_per_doc=6, steps=260, batch_size=12, cache_dir=".qsnl_cache/test"),
        crypto=replace(base.crypto, kem="ML-KEM-512"),
    )
