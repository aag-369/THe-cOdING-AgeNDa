"""QuantaSecure-NeuroLLM: a self-contained hybrid quantum / neural / LLM pipeline behind a
post-quantum secure channel.

Sub-packages
------------
``quanta``   pure-NumPy quantum simulator (state vector, density matrix, MPS, PQC, feature map)
``secure``   from-scratch cryptography (ML-KEM, ML-DSA, X25519, Ed25519, AES-GCM, ChaCha20-Poly1305,
             HKDF, hybrid Double Ratchet, Pedersen commitments, measurement-consistency NIZK)
``neuro``    reverse-mode autograd + hybrid quantum-classical encoder
``llm``      byte-level BPE tokenizer + decoder-only transformer (RoPE, RMSNorm, SwiGLU)
``utils``    canonical serialization, audit logging, numerics

Entry points: :mod:`quantasecure_neurollm.demo` (``python -m quantasecure_neurollm.demo``)
and :class:`quantasecure_neurollm.pipeline.SecureQuantumReasoningPipeline`.
"""

__version__ = "0.1.0"
CODE_VERSION = "qsnl-0.1.0"

__all__ = ["__version__", "CODE_VERSION"]
