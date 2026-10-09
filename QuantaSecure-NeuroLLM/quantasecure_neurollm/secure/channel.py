"""Secure inter-module channel: framed AEAD sessions over a recording in-memory bus.

Every byte that moves between pipeline modules is a :class:`Frame` on the
:class:`SecureBus`.  The bus is deliberately dumb and observable -- it records
every frame exactly as an on-path network adversary would see it -- so the
auditor can later *check* (rather than assume) that no plaintext crossed a
module boundary.

Frame layout (canonical encoding)::

    {v, kind, sid, src, dst, seq, hdr: RatchetHeader, ct: AEAD(ciphertext || tag)}

The associated data of the AEAD covers ``v, kind, sid, src, dst, seq`` and the
ratchet header, so routing metadata cannot be altered or frames re-addressed
without detection.  Metadata (who talks to whom, sizes, timing) is *not*
hidden -- as in TLS.

:class:`EgressGuard` is a data-loss-prevention check run on every outgoing
frame: it verifies that no ``window``-byte substring of the plaintext payload
appears anywhere in the frame bytes.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..utils.logging import AuditLog, get_logger
from ..utils.serialization import decode, encode
from .ratchet import DoubleRatchet, RatchetError, RatchetHeader

__all__ = ["Frame", "SecureSession", "SecureBus", "WireRecord", "EgressGuard", "ChannelError"]

_log = get_logger("secure.channel")
FRAME_VERSION = 1


class ChannelError(Exception):
    pass


@dataclass(frozen=True)
class Frame:
    kind: str
    session_id: bytes
    src: str
    dst: str
    seq: int
    header: RatchetHeader
    ciphertext: bytes

    def associated_data(self) -> bytes:
        return encode({"v": FRAME_VERSION, "kind": self.kind, "sid": self.session_id, "src": self.src, "dst": self.dst, "seq": self.seq})

    def to_bytes(self) -> bytes:
        return encode(
            {
                "v": FRAME_VERSION,
                "kind": self.kind,
                "sid": self.session_id,
                "src": self.src,
                "dst": self.dst,
                "seq": self.seq,
                "hdr": self.header.to_dict(),
                "ct": self.ciphertext,
            }
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Frame":
        try:
            d = decode(raw)
            if d["v"] != FRAME_VERSION:
                raise ChannelError("unsupported frame version")
            return cls(d["kind"], d["sid"], d["src"], d["dst"], int(d["seq"]), RatchetHeader.from_dict(d["hdr"]), d["ct"])
        except ChannelError:
            raise
        except Exception as exc:
            raise ChannelError(f"malformed frame: {exc}") from exc


class EgressGuard:
    """Asserts that no ``window``-byte run of a plaintext occurs inside the outgoing bytes.

    ``public`` is the frame's routing metadata (session id, node names, ratchet
    header), which is deliberately unencrypted and built only from fixed fields,
    never from the payload.  Windows that also occur in it are ignored -- a
    payload that happens to mention a node name is not a leak (the canonical
    encoding of ``"data-owner"`` appears in both the signed provenance record and
    the ``src`` header field).
    """

    def __init__(self, window: int = 16) -> None:
        self.window = window
        self.checks = 0
        self.bytes_checked = 0

    def _grams(self, data: bytes) -> set[bytes]:
        w = self.window
        return {data[i : i + w] for i in range(len(data) - w + 1)}

    def check(self, plaintext: bytes, wire: bytes, public: bytes = b"") -> bool:
        w = self.window
        self.checks += 1
        self.bytes_checked += len(plaintext)
        if len(plaintext) < w or len(wire) < w:
            return True
        grams = self._grams(wire) - self._grams(public)
        return not any(plaintext[i : i + w] in grams for i in range(len(plaintext) - w + 1))


class SecureSession:
    """One endpoint of an established session: seals / opens payload dictionaries."""

    def __init__(
        self,
        local: str,
        peer: str,
        session_id: bytes,
        ratchet: DoubleRatchet,
        audit: AuditLog,
        guard: Optional[EgressGuard] = None,
    ) -> None:
        self.local = local
        self.peer = peer
        self.session_id = session_id
        self.ratchet = ratchet
        self.audit = audit
        self.guard = guard or EgressGuard()
        self._send_seq = 0
        self.frames_sent = 0
        self.frames_received = 0

    def seal(self, payload: dict[str, Any], kind: str = "data") -> bytes:
        plaintext = encode(payload)
        stub = Frame(kind, self.session_id, self.local, self.peer, self._send_seq, RatchetHeader(b"", 0, 0, b""), b"")
        header, ct = self.ratchet.encrypt(plaintext, stub.associated_data())
        frame = Frame(kind, self.session_id, self.local, self.peer, self._send_seq, header, ct)
        wire = frame.to_bytes()
        public = Frame(kind, self.session_id, self.local, self.peer, self._send_seq, header, b"").to_bytes()
        if not self.guard.check(plaintext, wire, public):
            raise ChannelError("egress guard: plaintext found in outgoing frame")
        self._send_seq += 1
        self.frames_sent += 1
        self.audit.append(
            "channel",
            "seal",
            src=self.local,
            dst=self.peer,
            kind=kind,
            sid=self.session_id.hex(),
            n=header.n,
            key_fp=self.ratchet.used_key_fingerprints[-1],
            plaintext_bytes=len(plaintext),
            wire_bytes=len(wire),
            plaintext_sha3=hashlib.sha3_256(plaintext).hexdigest()[:16],
        )
        return wire

    def open(self, raw: bytes) -> tuple[str, dict[str, Any]]:
        frame = Frame.from_bytes(raw)
        if frame.session_id != self.session_id or frame.dst != self.local or frame.src != self.peer:
            raise ChannelError("frame does not belong to this session")
        steps_before = self.ratchet.dh_ratchet_steps
        try:
            plaintext = self.ratchet.decrypt(frame.header, frame.ciphertext, frame.associated_data())
        except RatchetError as exc:
            self.audit.append("channel", "reject", src=frame.src, dst=frame.dst, sid=self.session_id.hex(), reason=str(exc))
            raise ChannelError(f"frame rejected: {exc}") from exc
        self.frames_received += 1
        self.audit.append(
            "channel",
            "open",
            src=frame.src,
            dst=frame.dst,
            kind=frame.kind,
            sid=self.session_id.hex(),
            n=frame.header.n,
            dh_ratchet_steps=self.ratchet.dh_ratchet_steps - steps_before,
        )
        return frame.kind, decode(plaintext)


@dataclass
class WireRecord:
    index: int
    src: str
    dst: str
    channel: str  # "handshake" | "data" | "ack"
    data: bytes


@dataclass
class SecureBus:
    """In-memory transport between modules.  Records every frame as seen on the wire.

    ``interceptor`` (tests / demos only) may inspect or modify bytes in flight,
    modelling an active network adversary.
    """

    wire: list[WireRecord] = field(default_factory=list)
    interceptor: Optional[Callable[[WireRecord], bytes]] = None
    _queues: dict[str, deque[tuple[str, str, bytes]]] = field(default_factory=lambda: defaultdict(deque))

    def send(self, src: str, dst: str, data: bytes, channel: str) -> None:
        rec = WireRecord(len(self.wire), src, dst, channel, bytes(data))
        self.wire.append(rec)
        if self.interceptor is not None:
            data = self.interceptor(rec)
        self._queues[dst].append((src, channel, bytes(data)))
        _log.debug("wire #%d %s -> %s [%s] %d bytes", rec.index, src, dst, channel, len(data))

    def receive(self, dst: str) -> Optional[tuple[str, str, bytes]]:
        q = self._queues[dst]
        return q.popleft() if q else None

    def pending(self, dst: str) -> int:
        return len(self._queues[dst])

    @property
    def total_bytes(self) -> int:
        return sum(len(r.data) for r in self.wire)
