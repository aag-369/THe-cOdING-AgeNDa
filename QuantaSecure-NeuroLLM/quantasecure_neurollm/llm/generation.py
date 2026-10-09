"""Autoregressive decoding with temperature, top-k, nucleus (top-p) and repetition penalty.

Sampling pipeline for each step (applied to the last position's logits):

1. special tokens other than ``<eos>`` are masked out;
2. repetition penalty (Keskar et al. 2019): logits of already-generated tokens
   are divided by ``penalty`` if positive and multiplied if negative;
3. ``temperature`` scaling (``T -> 0`` = greedy);
4. **top-k**: keep the ``k`` largest logits;
5. **nucleus / top-p** (Holtzman et al. 2020): keep the smallest prefix of
   the sorted distribution with cumulative mass ``>= p``;
6. renormalise and draw from the remaining tokens.

Decoding uses the model's KV cache, so the soft prompt and the text prompt
are processed once and each new token costs a single-position forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..neuro.autograd import Tensor, no_grad
from ..utils.mathx import stable_softmax
from .model import TinyDecoderLM
from .tokenizer import BPETokenizer

__all__ = ["SamplingConfig", "GenerationResult", "filter_logits", "sample_token", "generate"]


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.9
    max_new_tokens: int = 100
    repetition_penalty: float = 1.0
    seed: int = 0


@dataclass
class GenerationResult:
    token_ids: list[int]
    text: str
    mean_logprob: float
    stopped_on_eos: bool
    step_entropies: list[float] = field(default_factory=list)


def filter_logits(logits: np.ndarray, cfg: SamplingConfig, history: Sequence[int], banned: Sequence[int]) -> np.ndarray:
    """Return a probability vector after penalty / temperature / top-k / top-p."""
    z = np.array(logits, dtype=np.float64)
    z[list(banned)] = -np.inf
    if cfg.repetition_penalty != 1.0 and len(history):
        idx = np.unique(np.asarray(history, dtype=np.int64))
        z[idx] = np.where(z[idx] > 0, z[idx] / cfg.repetition_penalty, z[idx] * cfg.repetition_penalty)
    if cfg.temperature <= 0:
        p = np.zeros_like(z)
        p[int(np.argmax(z))] = 1.0
        return p
    z = z / cfg.temperature
    if 0 < cfg.top_k < z.size:
        kth = np.partition(z, -cfg.top_k)[-cfg.top_k]
        z = np.where(z >= kth, z, -np.inf)
    p = stable_softmax(z)
    if 0 < cfg.top_p < 1.0:
        order = np.argsort(-p, kind="stable")
        cum = np.cumsum(p[order])
        keep = order[: int(np.searchsorted(cum, cfg.top_p) + 1)]
        mask = np.zeros_like(p, dtype=bool)
        mask[keep] = True
        p = np.where(mask, p, 0.0)
        p /= p.sum()
    return p


def sample_token(p: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(p.size, p=p))


def generate(
    model: TinyDecoderLM,
    tokenizer: BPETokenizer,
    latent: Optional[np.ndarray],
    prompt_ids: Sequence[int],
    cfg: SamplingConfig,
) -> GenerationResult:
    """Sample a continuation of ``prompt_ids`` conditioned on ``latent`` (``(1, d_latent)`` or None)."""
    rng = np.random.default_rng(cfg.seed)
    banned = [i for i in tokenizer.special_ids if i != tokenizer.eos_id]
    out: list[int] = []
    logps: list[float] = []
    ents: list[float] = []
    eos = False
    with no_grad():
        model.eval()
        cache = model.new_cache()
        z = None if latent is None else Tensor(np.atleast_2d(latent))
        logits = model.forward(np.asarray([list(prompt_ids)], dtype=np.int64), z, start=0, cache=cache).data[0, -1]
        pos = cache.length
        limit = model.cfg.max_seq_len - pos
        for _ in range(min(cfg.max_new_tokens, limit)):
            p = filter_logits(logits, cfg, list(prompt_ids) + out, banned)
            tok = sample_token(p, rng)
            full = stable_softmax(np.where(np.isin(np.arange(p.size), banned), -np.inf, logits))
            logps.append(float(np.log(max(full[tok], 1e-300))))
            ents.append(float(-np.sum(np.where(full > 0, full * np.log(np.where(full > 0, full, 1.0)), 0.0))))
            if tok == tokenizer.eos_id:
                eos = True
                break
            out.append(tok)
            logits = model.forward(np.asarray([[tok]], dtype=np.int64), None, start=pos, cache=cache).data[0, -1]
            pos += 1
    return GenerationResult(out, tokenizer.decode(out), float(np.mean(logps)) if logps else 0.0, eos, ents)
