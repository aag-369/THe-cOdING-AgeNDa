"""In-process pre-training of the hybrid encoder and the language model on the *public* corpus.

Nothing is downloaded and no pre-trained weights exist: every parameter starts
from a seeded random initialisation and is fitted here, in ~1-2 minutes on a
laptop CPU for the default configuration.

Data
----
For each public document we generate ``views_per_doc`` augmented *views*
(random sentence subsets, optionally with the title) and run every view through
the quantum feature map (state-vector features per window + noisy
density-matrix probe).  The confidential document is never touched.

Objective (joint, end-to-end through the parameter-shift quantum layer)
----------------------------------------------------------------------
``L = CE(summary | soft_prompt(z))                     # conditioned summary generation
    + w_text  * CE(abstract crop | soft_prompt(z))      # conditioned language modelling
    + w_topic * CE(topic | z)``                         # auxiliary topic head

where ``z = Encoder(quantum features)``.  Gradients of the language-model loss
flow through the soft-prompt adapter into the encoder and through the PQC
(exact parameter-shift Jacobians) into the classical layers below it.

Caching
-------
Trained weights are cached in ``<cache_dir>/<fingerprint>.npz`` where the
fingerprint hashes every training-relevant config field, the corpus and the
code version, so a stale cache can never be loaded silently.  Loading uses
``np.load(allow_pickle=False)``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from . import CODE_VERSION
from .config import PipelineConfig
from .corpus import EVALUATION_DOCUMENTS, PUBLIC_CORPUS, TOPICS, corpus_digest
from .llm.generation import SamplingConfig, generate
from .llm.model import TinyDecoderLM
from .llm.tokenizer import BPETokenizer
from .neuro.autograd import cross_entropy, no_grad
from .neuro.encoder import HybridQuantumEncoder, pad_chunks
from .neuro.optim import AdamW, clip_grad_norm, cosine_schedule
from .quanta.feature_map import QuantumFeatureExtractor, QuantumFeatureMap, QuantumFeatures
from .quanta.noise import NoiseModel
from .utils.logging import get_logger
from .utils.rng import numpy_rng

__all__ = ["TrainedArtifacts", "build_quantum_components", "make_views", "pretrain", "load_or_train", "summary_prompt"]

_log = get_logger("training")


@dataclass
class TrainedArtifacts:
    tokenizer: BPETokenizer
    feature_map: QuantumFeatureMap
    extractor: QuantumFeatureExtractor
    encoder: HybridQuantumEncoder
    lm: TinyDecoderLM
    topics: tuple[str, ...]
    fingerprint: str
    history: dict[str, list[float]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    train_seconds: float = 0.0
    from_cache: bool = False


def summary_prompt(tok: BPETokenizer) -> list[int]:
    return [tok.bos_id, tok.sum_id]


def _idf(tok: BPETokenizer) -> np.ndarray:
    docs = [set(tok.encode(d.title + " " + d.abstract)) for d in PUBLIC_CORPUS]
    df = np.zeros(tok.vocab_size)
    for s in docs:
        df[list(s)] += 1
    return np.log((len(docs) + 1) / (df + 1)) + 1.0


_SENT = re.compile(r"(?<=\.)\s+")


def _public_sentences() -> list[str]:
    out: list[str] = []
    for d in PUBLIC_CORPUS:
        out.append(d.title)
        out += [s for s in _SENT.split(d.abstract) if s]
        out += [s for s in _SENT.split(d.summary) if s]
    return out


def build_quantum_components(config: PipelineConfig, tok: BPETokenizer) -> tuple[QuantumFeatureMap, QuantumFeatureExtractor]:
    """Feature map (LSA projection fitted on public sentences) and extractor; deterministic in the config."""
    qc = config.quantum
    fit_docs = [np.array(tok.encode(s)) for s in _public_sentences()]
    fmap = QuantumFeatureMap(
        qc.n_qubits, qc.reupload_layers, tok.vocab_size, _idf(tok), qc.projection_seed, qc.reservoir_seed, fit_docs=fit_docs
    )
    ext = QuantumFeatureExtractor(
        fmap,
        NoiseModel.from_config(config.noise),
        chunk_tokens=qc.chunk_tokens,
        chunk_stride=qc.chunk_stride,
        max_chunks=qc.max_chunks,
        chunk_shots=256,
        probe_shots=qc.probe_shots,
    )
    return fmap, ext


def make_views(title: str, abstract: str, summary: str, n_views: int, rng: np.random.Generator) -> list[str]:
    """Augmented views of one public document.

    View 0 is the full abstract and view 1 the title + summary (different
    wording, same topic).  The rest draw a random subset of abstract *and*
    summary sentences (order kept), sometimes prefixed by the title, and drop
    ~12% of the words, so the encoder cannot key on any single sentence.
    """
    sents = [s for s in _SENT.split(abstract) if s] + [s for s in _SENT.split(summary) if s]
    views = [abstract, title + ". " + summary][:n_views]
    while len(views) < n_views:
        p_keep = rng.uniform(0.3, 0.8)  # short views too: unseen documents can be only a few sentences long
        keep = [s for s in sents if rng.random() < p_keep]
        if len(keep) < 2:
            continue
        words = [w for w in " ".join(keep).split(" ") if rng.random() > 0.12]
        text = " ".join(words)
        if rng.random() < 0.3:
            text = title + ". " + text
        views.append(text)
    return views


def _batch_sequences(seqs: Sequence[Sequence[int]], pad: int, loss_from: int) -> tuple[np.ndarray, np.ndarray]:
    width = max(len(s) for s in seqs)
    ids = np.full((len(seqs), width), pad, dtype=np.int64)
    mask = np.zeros((len(seqs), width))
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
        mask[i, loss_from : len(s)] = 1.0
    return ids, mask


def _encode_batch(enc: HybridQuantumEncoder, feats: Sequence[QuantumFeatures]):  # type: ignore[no-untyped-def]
    padded, mask = pad_chunks([f.chunk_features for f in feats])
    return enc.forward(padded, mask, np.stack([f.global_features for f in feats]))


def pretrain(config: PipelineConfig, progress: Optional[Callable[[str], None]] = None) -> TrainedArtifacts:
    t0 = time.perf_counter()
    tc = config.training
    say = progress or _log.info
    tok = BPETokenizer.default()
    fmap, ext = build_quantum_components(config, tok)
    rng = numpy_rng("training-views", tc.seed)

    texts, labels = [], []
    for k, d in enumerate(PUBLIC_CORPUS):
        for v in make_views(d.title, d.abstract, d.summary, tc.views_per_doc, rng):
            texts.append(v)
            labels.append(k)
    say(f"[train] quantum feature extraction for {len(texts)} public views ({config.quantum.n_qubits} qubits) ...")
    feats = ext.extract_many([np.array(tok.encode(t)) for t in texts], seed=tc.seed)
    labels_arr = np.array(labels)

    held_rng = numpy_rng("heldout-views", tc.seed)
    held_texts, held_labels = [], []
    for k, d in enumerate(PUBLIC_CORPUS):
        for v in make_views(d.title, d.abstract, d.summary, 5, held_rng)[2:]:
            held_texts.append(v)
            held_labels.append(k)
    held_feats = ext.extract_many([np.array(tok.encode(t)) for t in held_texts], seed=tc.seed + 1)
    eval_feats = ext.extract_many([np.array(tok.encode(t)) for _, t in EVALUATION_DOCUMENTS], seed=tc.seed + 2)

    enc = HybridQuantumEncoder(config.neuro, ext.chunk_dim, ext.global_dim, len(TOPICS), seed=tc.seed)
    enc.fit_normalizer([f.chunk_features for f in feats], np.stack([f.global_features for f in feats]))
    lm = TinyDecoderLM(config.llm, tok.vocab_size, config.neuro.d_latent, seed=tc.seed + 7)
    params = enc.parameters() + lm.parameters()
    say(f"[train] encoder {enc.n_params():,} params (incl. {enc.quantum.weights.size} PQC angles), LM {lm.n_params():,} params")

    summaries = [summary_prompt(tok) + tok.encode(d.summary) + [tok.eos_id] for d in PUBLIC_CORPUS]
    abstracts = [tok.encode(d.abstract) for d in PUBLIC_CORPUS]
    opt = AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    history: dict[str, list[float]] = {"loss": [], "summary": [], "text": [], "topic": [], "grad_norm": []}
    brng = numpy_rng("training-batches", tc.seed)
    enc.train()
    lm.train()
    for step in range(tc.steps):
        opt.lr = cosine_schedule(step, tc.steps, tc.lr, tc.min_lr, tc.warmup_steps)
        idx = brng.choice(len(feats), size=tc.batch_size, replace=False)
        out = _encode_batch(enc, [feats[i] for i in idx])
        # Latent noise: the LM must map a whole neighbourhood of each topic's latent cluster to
        # fluent text, which keeps generation stable for unseen documents.
        z = out.latent + brng.standard_normal(out.latent.shape) * tc.latent_noise
        ids_s, mask_s = _batch_sequences([summaries[labels_arr[i]] for i in idx], tok.pad_id, 2)
        loss_s = lm.sequence_loss(ids_s, z, mask_s)
        half = max(1, len(idx) // 2)  # the auxiliary text objective uses half the batch (cost control)
        crops = []
        for i in idx[:half]:
            a = abstracts[labels_arr[i]]
            st = int(brng.integers(0, max(1, len(a) - tc.text_window)))
            crops.append([tok.bos_id, tok.txt_id] + a[st : st + tc.text_window])
        ids_t, mask_t = _batch_sequences(crops, tok.pad_id, 2)
        loss_t = lm.sequence_loss(ids_t, z[:half], mask_t)
        loss_c = cross_entropy(out.topic_logits, labels_arr[idx])
        loss = loss_s + loss_t * tc.text_loss_weight + loss_c * tc.topic_loss_weight
        opt.zero_grad()
        loss.backward()
        gn = clip_grad_norm(params, tc.grad_clip)
        opt.step()
        for k, v in (("loss", loss.item()), ("summary", loss_s.item()), ("text", loss_t.item()), ("topic", loss_c.item()), ("grad_norm", gn)):
            history[k].append(float(v))
        if step % tc.log_every == 0 or step == tc.steps - 1:
            say(
                f"[train] step {step:4d}/{tc.steps}  loss {loss.item():.3f}  summary {loss_s.item():.3f}  "
                f"text {loss_t.item():.3f}  topic {loss_c.item():.3f}  |g| {gn:.2f}  lr {opt.lr:.2e}"
            )

    enc.eval()
    lm.eval()
    metrics = _evaluate(enc, lm, tok, held_feats, np.array(held_labels), eval_feats, config)
    say(
        f"[train] held-out view topic accuracy {metrics['heldout_topic_accuracy']:.2f}, "
        f"evaluation-document accuracy {metrics['evaluation_topic_accuracy']:.2f}, "
        f"summary exact-match (greedy) {metrics['summary_exact_match']:.2f}"
    )
    fp = config.training_fingerprint(corpus_digest(), CODE_VERSION)
    return TrainedArtifacts(tok, fmap, ext, enc, lm, TOPICS, fp, history, metrics, time.perf_counter() - t0, False)


def _evaluate(
    enc: HybridQuantumEncoder,
    lm: TinyDecoderLM,
    tok: BPETokenizer,
    held: Sequence[QuantumFeatures],
    held_labels: np.ndarray,
    eval_feats: Sequence[QuantumFeatures],
    config: PipelineConfig,
) -> dict[str, Any]:
    with no_grad():
        held_pred = enc.topic_probabilities(_encode_batch(enc, held)).argmax(axis=1)
        ev_pred = enc.topic_probabilities(_encode_batch(enc, eval_feats)).argmax(axis=1)
        exact = 0
        for k, d in enumerate(PUBLIC_CORPUS):
            idx = int(np.nonzero(held_labels == k)[0][0])
            z = _encode_batch(enc, [held[idx]]).latent.data
            res = generate(lm, tok, z, summary_prompt(tok), SamplingConfig(temperature=0.0, max_new_tokens=config.llm.max_new_tokens))
            exact += int(res.text.strip() == d.summary.strip())
    ev_true = np.array([TOPICS.index(t) for t, _ in EVALUATION_DOCUMENTS])
    return {
        "heldout_topic_accuracy": float(np.mean(held_pred == held_labels)),
        "evaluation_topic_accuracy": float(np.mean(ev_pred == ev_true)),
        "summary_exact_match": exact / len(PUBLIC_CORPUS),
    }


# ----------------------------------------------------------------------------- cache
def _cache_paths(config: PipelineConfig) -> tuple[Path, Path, str]:
    fp = config.training_fingerprint(corpus_digest(), CODE_VERSION)
    root = Path(config.training.cache_dir)
    return root / f"{fp}.npz", root / f"{fp}.json", fp


def save_artifacts(art: TrainedArtifacts, config: PipelineConfig) -> Path:
    npz, meta, _ = _cache_paths(config)
    npz.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"enc/{k}": v for k, v in art.encoder.state_dict().items()}
    arrays.update({f"lm/{k}": v for k, v in art.lm.state_dict().items()})
    np.savez_compressed(npz, **arrays)
    meta.write_text(json.dumps({"history": art.history, "metrics": art.metrics, "train_seconds": art.train_seconds}), encoding="utf-8")
    return npz


def load_artifacts(config: PipelineConfig) -> Optional[TrainedArtifacts]:
    npz, meta, fp = _cache_paths(config)
    if not npz.exists() or not meta.exists():
        return None
    tok = BPETokenizer.default()
    fmap, ext = build_quantum_components(config, tok)
    enc = HybridQuantumEncoder(config.neuro, ext.chunk_dim, ext.global_dim, len(TOPICS), seed=config.training.seed)
    lm = TinyDecoderLM(config.llm, tok.vocab_size, config.neuro.d_latent, seed=config.training.seed + 7)
    with np.load(npz, allow_pickle=False) as data:
        enc.load_state_dict({k[4:]: data[k] for k in data.files if k.startswith("enc/")})
        lm.load_state_dict({k[3:]: data[k] for k in data.files if k.startswith("lm/")})
    m = json.loads(meta.read_text(encoding="utf-8"))
    enc.eval()
    lm.eval()
    return TrainedArtifacts(tok, fmap, ext, enc, lm, TOPICS, fp, m["history"], m["metrics"], m["train_seconds"], True)


def load_or_train(config: PipelineConfig, retrain: bool = False, progress: Optional[Callable[[str], None]] = None) -> TrainedArtifacts:
    """Load cached weights for this exact configuration, or train and cache them."""
    if not retrain:
        art = load_artifacts(config)
        if art is not None:
            (progress or _log.info)(f"[train] loaded cached weights {art.fingerprint} (trained in {art.train_seconds:.0f}s)")
            return art
    art = pretrain(config, progress)
    save_artifacts(art, config)
    return art
