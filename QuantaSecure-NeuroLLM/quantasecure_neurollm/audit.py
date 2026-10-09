"""Post-run security audit of everything that crossed a module boundary.

The claim "the original document never left the secure channel in plaintext"
is *checked*, not assumed: :class:`WireAuditor` replays the bus's wire log --
exactly the bytes an on-path adversary saw -- and verifies:

* **No plaintext leakage.**  No ``window``-byte substring of the document's
  UTF-8 encoding, nor of its token-id sequence in any common integer encoding
  (int32/int64/uint16, little-endian), occurs in any frame (handshake, data or
  receipt).  With ``window = 12`` a chance match on ciphertext has probability
  ``~ (bytes on wire) * (secret length) * 2^-96``, i.e. zero for practical
  purposes, so a hit can only mean real leakage.
* **Fresh keys.**  Every sealed frame used a distinct message key (compared by
  one-way fingerprints), and every session's Diffie-Hellman/KEM ratchet
  advanced at least once at each endpoint.
* **Complete handshakes** and an **intact hash-chained audit log**.
* **Ciphertext statistics**: byte entropy and a chi-square uniformity test of
  all AEAD outputs (a sanity check that payloads are not stored in the clear,
  not a proof of security).
* **Active probes** on a live session: a bit-flipped frame is rejected, the
  genuine frame is accepted, and replaying it is rejected -- integrity,
  transactional rollback and replay protection observed end to end.

This is evidence about one execution, complementary to (not a substitute for)
the cryptographic arguments in the module docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np
from scipy import stats as sstats

from .secure.channel import ChannelError, Frame, SecureBus
from .utils.logging import AuditLog, get_logger

__all__ = ["AuditReport", "WireAuditor"]

_log = get_logger("audit")


@dataclass
class AuditReport:
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


class WireAuditor:
    def __init__(self, window: int = 12) -> None:
        self.window = window

    # ------------------------------------------------------------ leak scan
    def leak_scan(self, frames: list[bytes], secrets: Mapping[str, bytes]) -> dict[str, int]:
        """Number of ``window``-grams of each secret found anywhere on the wire."""
        w = self.window
        wire_grams: set[bytes] = set()
        for f in frames:
            wire_grams.update(f[i : i + w] for i in range(len(f) - w + 1))
        hits: dict[str, int] = {}
        for name, secret in secrets.items():
            if len(secret) < w:
                hits[name] = sum(secret in f for f in frames)
                continue
            grams = {secret[i : i + w] for i in range(len(secret) - w + 1)}
            hits[name] = len(grams & wire_grams)
        return hits

    @staticmethod
    def ciphertext_statistics(bus: SecureBus) -> dict[str, float]:
        bodies = []
        for rec in bus.wire:
            if rec.channel != "handshake":
                bodies.append(Frame.from_bytes(rec.data).ciphertext)
        data = np.frombuffer(b"".join(bodies), dtype=np.uint8)
        if data.size == 0:
            return {"bytes": 0, "entropy_bits_per_byte": 0.0, "chi2": float("nan"), "chi2_p_value": float("nan")}
        counts = np.bincount(data, minlength=256).astype(np.float64)
        p = counts / counts.sum()
        entropy = float(-np.sum(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0)))
        expected = counts.sum() / 256
        chi2 = float(np.sum((counts - expected) ** 2 / expected))
        return {"bytes": int(data.size), "entropy_bits_per_byte": entropy, "chi2": chi2, "chi2_p_value": float(sstats.chi2.sf(chi2, 255))}

    # ------------------------------------------------------------ main entry
    def run(
        self,
        bus: SecureBus,
        audit: AuditLog,
        nodes: Mapping[str, Any],
        secrets: Mapping[str, bytes],
        probe_link: Optional[tuple[str, str]] = None,
    ) -> AuditReport:
        rep = AuditReport()
        frames = [r.data for r in bus.wire]
        hits = self.leak_scan(frames, secrets)
        rep.details["leak_hits"] = hits
        rep.checks["no_plaintext_document_on_wire"] = hits.get("document", 0) == 0
        rep.checks["no_plaintext_tokens_on_wire"] = all(v == 0 for k, v in hits.items() if k.startswith("tokens"))

        fingerprints: list[str] = []
        dh_steps: dict[str, int] = {}
        for node in nodes.values():
            for peer, sess in node.sessions.items():
                fingerprints += sess.ratchet.used_key_fingerprints
                dh_steps[f"{node.name}->{peer}"] = sess.ratchet.dh_ratchet_steps
        seals = audit.filter("channel", "seal")
        rep.checks["fresh_key_per_message"] = len(fingerprints) == len(set(fingerprints)) == len(seals) > 0
        rep.checks["dh_ratchet_advanced_everywhere"] = bool(dh_steps) and all(v >= 1 for v in dh_steps.values())
        n_links = len(dh_steps) // 2
        rep.checks["handshakes_complete"] = len(audit.filter("handshake", "established")) == 2 * n_links > 0
        rep.details.update(
            {
                "message_keys": len(fingerprints),
                "dh_ratchet_steps": dh_steps,
                "frames": len(frames),
                "wire_bytes": sum(map(len, frames)),
                "frames_by_channel": {c: sum(r.channel == c for r in bus.wire) for c in ("handshake", "data", "ack")},
            }
        )
        cstats = self.ciphertext_statistics(bus)
        rep.details["ciphertext"] = cstats
        rep.checks["ciphertext_looks_uniform"] = cstats["bytes"] > 0 and cstats["entropy_bits_per_byte"] > 7.9

        if probe_link is not None:
            rep.checks.update(self._active_probes(nodes, *probe_link))
        rep.checks["audit_log_chain_intact"] = audit.verify_chain()
        _log.info(
            "wire audit: %d frames / %d bytes, leak hits %s, %d unique message keys, entropy %.4f bits/byte",
            rep.details["frames"], rep.details["wire_bytes"], hits, len(set(fingerprints)), cstats["entropy_bits_per_byte"],
        )  # fmt: skip
        return rep

    @staticmethod
    def _active_probes(nodes: Mapping[str, Any], a: str, b: str) -> dict[str, bool]:
        sender = nodes[a].sessions[b]
        receiver = nodes[b].sessions[a]
        wire = sender.seal({"probe": "integrity-check", "nonce": nodes[a].rng.random_bytes(16)}, kind="probe")
        tampered = bytearray(wire)
        tampered[-24] ^= 0x01  # inside the AEAD ciphertext/tag
        out: dict[str, bool] = {}
        try:
            receiver.open(bytes(tampered))
            out["tampered_frame_rejected"] = False
        except ChannelError:
            out["tampered_frame_rejected"] = True
        try:
            out["genuine_frame_accepted_after_tamper"] = receiver.open(wire)[1].get("probe") == "integrity-check"
        except ChannelError:
            out["genuine_frame_accepted_after_tamper"] = False
        try:
            receiver.open(wire)
            out["replayed_frame_rejected"] = False
        except ChannelError:
            out["replayed_frame_rejected"] = True
        return out
