"""Byte-level BPE tokenizer with a hard-coded merge table.

* **Base alphabet**: the 256 byte values, so *any* UTF-8 text round-trips
  exactly and there is no unknown token.
* **Pre-tokenisation**: a GPT-2-style regex splits text into words with their
  leading space, digit runs and punctuation runs; merges never cross word
  boundaries.
* **Merges**: learned by :func:`train_bpe` from the public corpus only, then
  frozen into :mod:`.bpe_merges` (the "hard-coded vocabulary").  Training is
  deterministic: the most frequent adjacent pair wins, ties are broken by the
  smallest ``(left, right)`` id pair, and merging stops when no pair occurs at
  least ``min_frequency`` times.
* **Special tokens** follow the merged tokens: ``<pad> <bos> <eos> <sum> <txt>``.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Iterable, Sequence

__all__ = ["BPETokenizer", "train_bpe", "SPECIAL_TOKENS"]

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<sum>", "<txt>")
_PRETOKENIZE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def _words(text: str) -> list[bytes]:
    return [m.group(0).encode("utf-8") for m in _PRETOKENIZE.finditer(text)]


def _merge(seq: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


def train_bpe(texts: Iterable[str], n_merges: int, min_frequency: int = 2) -> list[tuple[int, int]]:
    """Learn up to ``n_merges`` merges (deterministic, see module docstring)."""
    freq: Counter[tuple[int, ...]] = Counter()
    for t in texts:
        for w in _words(t):
            freq[tuple(w)] += 1
    merges: list[tuple[int, int]] = []
    words = dict(freq)
    for k in range(n_merges):
        pairs: Counter[tuple[int, int]] = Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))
        if best[1] < min_frequency:
            break
        pair = best[0]
        new_id = 256 + k
        merges.append(pair)
        words = {_merge(w, pair, new_id): c for w, c in words.items()}
    return merges


class BPETokenizer:
    """Encode text to token ids and back."""

    def __init__(self, merges: Sequence[tuple[int, int]]) -> None:
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.token_bytes: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        base = len(self.token_bytes)
        self.special = {tok: base + i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.special_ids = frozenset(self.special.values())
        self.pad_id = self.special["<pad>"]
        self.bos_id = self.special["<bos>"]
        self.eos_id = self.special["<eos>"]
        self.sum_id = self.special["<sum>"]
        self.txt_id = self.special["<txt>"]
        self._encode_word = lru_cache(maxsize=65536)(self._encode_word_uncached)

    @classmethod
    def default(cls) -> "BPETokenizer":
        """Tokenizer with the hard-coded merge table."""
        from .bpe_merges import MERGES

        return cls(MERGES)

    @property
    def vocab_size(self) -> int:
        return len(self.token_bytes) + len(SPECIAL_TOKENS)

    def _encode_word_uncached(self, word: bytes) -> tuple[int, ...]:
        seq = tuple(word)
        while len(seq) > 1:
            best = min(((self.ranks.get(p, 1 << 30), p) for p in zip(seq, seq[1:])), default=(1 << 30, None))
            if best[1] is None or best[0] == 1 << 30:
                break
            seq = _merge(seq, best[1], 256 + best[0])
        return seq

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = [self.bos_id] if add_bos else []
        for w in _words(text):
            ids.extend(self._encode_word(w))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out = bytearray()
        names = {v: k for k, v in self.special.items()}
        for i in ids:
            i = int(i)
            if i in self.special_ids:
                if not skip_special:
                    out += names[i].encode()
                continue
            out += self.token_bytes[i]
        return out.decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        if i in self.special_ids:
            return {v: k for k, v in self.special.items()}[i]
        return self.token_bytes[i].decode("utf-8", errors="replace")
