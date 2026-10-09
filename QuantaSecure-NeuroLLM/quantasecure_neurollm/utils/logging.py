"""Structured logging and a tamper-evident (hash-chained) audit log.

Two distinct facilities live here:

* :func:`get_logger` / :func:`configure_logging` -- ordinary human-readable
  logging through the standard :mod:`logging` package, namespaced under
  ``qsnl.*``.
* :class:`AuditLog` -- an append-only list of structured events where every
  record carries ``SHA3-256(prev_hash || canonical(record))``.  Any later edit,
  deletion or reordering of a record breaks the chain, which
  :meth:`AuditLog.verify_chain` detects.  The log is what the final auditor
  replays to show that every handshake, ratchet step and measurement happened.

Security note: audit events must never contain secrets or plaintext payloads.
Key material is only ever logged as a *fingerprint* (a truncated domain-separated
hash), which is one-way and safe to disclose.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .serialization import encode

__all__ = ["get_logger", "configure_logging", "AuditLog", "AuditRecord", "short_hex"]

_ROOT = "qsnl"
_GENESIS = hashlib.sha3_256(b"QSNL-audit-genesis-v1").digest()


def get_logger(name: str) -> logging.Logger:
    """Return a logger in the ``qsnl`` namespace (``qsnl.<name>``)."""
    return logging.getLogger(f"{_ROOT}.{name}" if name else _ROOT)


def configure_logging(level: int = logging.INFO, stream: Any = None) -> None:
    """Configure the ``qsnl`` logger tree with a compact single-line format."""
    root = logging.getLogger(_ROOT)
    root.handlers.clear()
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-5s %(name)-16s %(message)s", "%H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def short_hex(data: bytes, n: int = 8) -> str:
    """First ``n`` bytes of ``data`` as hex (for log lines)."""
    return bytes(data[:n]).hex()


@dataclass(frozen=True)
class AuditRecord:
    """One hash-chained audit event."""

    index: int
    timestamp: float
    category: str
    event: str
    fields: dict[str, Any]
    prev_hash: bytes
    record_hash: bytes


@dataclass
class AuditLog:
    """Append-only, hash-chained event log (tamper-evident, not tamper-proof)."""

    records: list[AuditRecord] = field(default_factory=list)

    @staticmethod
    def _hash(prev: bytes, index: int, ts: float, category: str, event: str, fields: dict[str, Any]) -> bytes:
        body = encode({"i": index, "t": ts, "c": category, "e": event, "f": fields})
        return hashlib.sha3_256(prev + body).digest()

    def append(self, category: str, event: str, **fields: Any) -> AuditRecord:
        prev = self.head
        idx = len(self.records)
        ts = time.time()
        h = self._hash(prev, idx, ts, category, event, fields)
        rec = AuditRecord(idx, ts, category, event, dict(fields), prev, h)
        self.records.append(rec)
        return rec

    @property
    def head(self) -> bytes:
        return self.records[-1].record_hash if self.records else _GENESIS

    def verify_chain(self) -> bool:
        prev = _GENESIS
        for i, rec in enumerate(self.records):
            if rec.index != i or rec.prev_hash != prev:
                return False
            if self._hash(prev, rec.index, rec.timestamp, rec.category, rec.event, rec.fields) != rec.record_hash:
                return False
            prev = rec.record_hash
        return True

    def filter(self, category: str | None = None, event: str | None = None) -> list[AuditRecord]:
        return [
            r for r in self.records if (category is None or r.category == category) and (event is None or r.event == event)
        ]
