"""Canonical, deterministic, pickle-free binary serialization.

Every byte string that crosses a module boundary, and every value that is
hashed into a commitment, a signature or a Fiat-Shamir transcript, is produced
by :func:`encode`.  Two properties matter for the security argument:

* **Canonicality** -- a given Python value has exactly one encoding (dict keys
  are sorted by their UTF-8 bytes, integers use a minimal-length rule, arrays
  are forced to little-endian C order).  Without canonicality two parties could
  hash "the same" object to different digests, or an adversary could find two
  encodings of different objects that collide syntactically.
* **Safety** -- decoding never executes code (contrast :mod:`pickle`, which is a
  remote-code-execution primitive on untrusted input), enforces a whitelist of
  NumPy dtypes, bounds recursion depth and rejects trailing bytes.

Wire format (all multi-byte lengths are big-endian)::

    None    : 0x00
    bool    : 0x01 <u8 0|1>
    int     : 0x02 <u16 nbytes> <two's-complement big-endian>
    float   : 0x03 <IEEE-754 binary64 big-endian>
    str     : 0x04 <u32 nbytes> <utf-8>
    bytes   : 0x05 <u32 nbytes> <raw>
    list    : 0x06 <u32 count> <item>*
    dict    : 0x07 <u32 count> (<str key> <value>)*   keys sorted by utf-8 bytes
    ndarray : 0x08 <u8 len> <dtype str> <u8 ndim> <u64 dim>* <u64 nbytes> <raw LE bytes>
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

__all__ = ["encode", "decode", "SerializationError"]

_T_NONE, _T_BOOL, _T_INT, _T_FLOAT, _T_STR, _T_BYTES, _T_LIST, _T_DICT, _T_ARRAY = range(9)

_ALLOWED_DTYPES = frozenset(
    {"|b1", "|i1", "|u1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "<f4", "<f8", "<c8", "<c16"}
)
_MAX_DEPTH = 64
_MAX_LEN = 1 << 31


class SerializationError(ValueError):
    """Raised when a value cannot be encoded or a byte string is malformed."""


def _int_nbytes(x: int) -> int:
    # Room for the sign bit; deterministic so the decoder can enforce it.
    return (x.bit_length() + 8) // 8


def _encode(obj: Any, out: bytearray, depth: int) -> None:
    if depth > _MAX_DEPTH:
        raise SerializationError("maximum nesting depth exceeded")
    if obj is None:
        out.append(_T_NONE)
    elif isinstance(obj, (bool, np.bool_)):
        out.append(_T_BOOL)
        out.append(1 if bool(obj) else 0)
    elif isinstance(obj, (int, np.integer)):
        x = int(obj)
        n = _int_nbytes(x)
        if n > 0xFFFF:
            raise SerializationError("integer too large")
        out.append(_T_INT)
        out += struct.pack(">H", n)
        out += x.to_bytes(n, "big", signed=True)
    elif isinstance(obj, (float, np.floating)):
        out.append(_T_FLOAT)
        out += struct.pack(">d", float(obj))
    elif isinstance(obj, str):
        raw = obj.encode("utf-8")
        out.append(_T_STR)
        out += struct.pack(">I", len(raw))
        out += raw
    elif isinstance(obj, (bytes, bytearray, memoryview)):
        raw = bytes(obj)
        out.append(_T_BYTES)
        out += struct.pack(">I", len(raw))
        out += raw
    elif isinstance(obj, (list, tuple)):
        out.append(_T_LIST)
        out += struct.pack(">I", len(obj))
        for item in obj:
            _encode(item, out, depth + 1)
    elif isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            if not isinstance(k, str):
                raise SerializationError(f"dict keys must be str, got {type(k).__name__}")
            items.append((k.encode("utf-8"), v))
        items.sort(key=lambda t: t[0])
        out.append(_T_DICT)
        out += struct.pack(">I", len(items))
        for raw_k, v in items:
            out.append(_T_STR)
            out += struct.pack(">I", len(raw_k))
            out += raw_k
            _encode(v, out, depth + 1)
    elif isinstance(obj, np.ndarray):
        arr = obj
        if arr.dtype.byteorder == ">":
            arr = arr.astype(arr.dtype.newbyteorder("<"))
        arr = np.ascontiguousarray(arr)
        dt = arr.dtype.str
        if dt not in _ALLOWED_DTYPES:
            raise SerializationError(f"dtype {dt} not allowed")
        if arr.ndim > 32:
            raise SerializationError("too many array dimensions")
        out.append(_T_ARRAY)
        out.append(len(dt))
        out += dt.encode("ascii")
        out.append(arr.ndim)
        for dim in arr.shape:
            out += struct.pack(">Q", dim)
        raw = arr.tobytes(order="C")
        out += struct.pack(">Q", len(raw))
        out += raw
    else:
        raise SerializationError(f"cannot encode type {type(obj).__name__}")


def encode(obj: Any) -> bytes:
    """Encode ``obj`` canonically.  Tuples encode as lists."""
    out = bytearray()
    _encode(obj, out, 0)
    return bytes(out)


class _Reader:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.buf):
            raise SerializationError("truncated input")
        chunk = self.buf[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def u8(self) -> int:
        return self.take(1)[0]


def _decode(r: _Reader, depth: int) -> Any:
    if depth > _MAX_DEPTH:
        raise SerializationError("maximum nesting depth exceeded")
    tag = r.u8()
    if tag == _T_NONE:
        return None
    if tag == _T_BOOL:
        b = r.u8()
        if b not in (0, 1):
            raise SerializationError("non-canonical bool")
        return bool(b)
    if tag == _T_INT:
        (n,) = struct.unpack(">H", r.take(2))
        x = int.from_bytes(r.take(n), "big", signed=True)
        if _int_nbytes(x) != n:
            raise SerializationError("non-canonical integer length")
        return x
    if tag == _T_FLOAT:
        return struct.unpack(">d", r.take(8))[0]
    if tag == _T_STR:
        (n,) = struct.unpack(">I", r.take(4))
        try:
            return r.take(n).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SerializationError("invalid utf-8") from exc
    if tag == _T_BYTES:
        (n,) = struct.unpack(">I", r.take(4))
        return r.take(n)
    if tag == _T_LIST:
        (n,) = struct.unpack(">I", r.take(4))
        if n > _MAX_LEN:
            raise SerializationError("list too long")
        return [_decode(r, depth + 1) for _ in range(n)]
    if tag == _T_DICT:
        (n,) = struct.unpack(">I", r.take(4))
        out: dict[str, Any] = {}
        prev: bytes | None = None
        for _ in range(n):
            if r.u8() != _T_STR:
                raise SerializationError("dict key must be str")
            (kn,) = struct.unpack(">I", r.take(4))
            raw_k = r.take(kn)
            if prev is not None and raw_k <= prev:
                raise SerializationError("dict keys not strictly sorted (non-canonical)")
            prev = raw_k
            out[raw_k.decode("utf-8")] = _decode(r, depth + 1)
        return out
    if tag == _T_ARRAY:
        dlen = r.u8()
        dt = r.take(dlen).decode("ascii")
        if dt not in _ALLOWED_DTYPES:
            raise SerializationError(f"dtype {dt} not allowed")
        ndim = r.u8()
        shape = tuple(struct.unpack(">Q", r.take(8))[0] for _ in range(ndim))
        (nbytes,) = struct.unpack(">Q", r.take(8))
        dtype = np.dtype(dt)
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize if shape else dtype.itemsize
        if nbytes != expected:
            raise SerializationError("array byte length does not match shape")
        return np.frombuffer(r.take(nbytes), dtype=dtype).reshape(shape).copy()
    raise SerializationError(f"unknown tag 0x{tag:02x}")


def decode(data: bytes) -> Any:
    """Decode bytes produced by :func:`encode`; rejects trailing garbage."""
    r = _Reader(bytes(data))
    obj = _decode(r, 0)
    if r.pos != len(r.buf):
        raise SerializationError("trailing bytes after encoded value")
    return obj
