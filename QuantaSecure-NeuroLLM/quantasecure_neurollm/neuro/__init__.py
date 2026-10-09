"""Neural-network module: reverse-mode autograd, layers, quantum layer, hybrid encoder."""

from .autograd import Tensor, concat, cross_entropy, custom_op, log_softmax, no_grad, softmax
from .encoder import EncoderOutput, HybridQuantumEncoder, pad_chunks
from .layers import (
    Embedding,
    LayerNorm,
    Linear,
    Module,
    MultiHeadAttention,
    Parameter,
    RMSNorm,
    SwiGLU,
    UnitaryMeshLayer,
    scaled_dot_product_attention,
)
from .optim import AdamW, clip_grad_norm, cosine_schedule
from .quantum_layer import QuantumLayer

__all__ = [
    "Tensor",
    "concat",
    "cross_entropy",
    "custom_op",
    "log_softmax",
    "no_grad",
    "softmax",
    "EncoderOutput",
    "HybridQuantumEncoder",
    "pad_chunks",
    "Embedding",
    "LayerNorm",
    "Linear",
    "Module",
    "MultiHeadAttention",
    "Parameter",
    "RMSNorm",
    "SwiGLU",
    "UnitaryMeshLayer",
    "scaled_dot_product_attention",
    "AdamW",
    "clip_grad_norm",
    "cosine_schedule",
    "QuantumLayer",
]
