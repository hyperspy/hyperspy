"""
tests/drawing/anyplotlib_playwright/_png_utils.py
=================================================

Vendored from anyplotlib (anyplotlib/tests/_png_utils.py) so the hyperspy
Playwright suite does not depend on anyplotlib's test tree being installed.

Minimal PNG encode / decode and pixel-level comparison utilities built on
top of numpy + Python stdlib only (struct, zlib).  No PIL, no matplotlib.

Public API
----------
decode_png(data)            bytes  → H×W×C uint8 ndarray  (RGB or RGBA)
encode_png(arr)             H×W×C uint8 ndarray → bytes    (filter-0 PNG)
compare_arrays(a, b, ...)   two uint8 arrays    → (ok: bool, message: str)
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

# ---------------------------------------------------------------------------
# PNG magic / chunk helpers
# ---------------------------------------------------------------------------

_PNG_SIG = b"\x89PNG\r\n\x1a\n"

# color-type → channel count (8-bit only)
_CT_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _iter_chunks(data: bytes):
    """Yield (chunk_type: bytes, chunk_data: bytes) for every PNG chunk."""
    pos = 8  # skip signature
    while pos + 12 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        cdata = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        yield ctype, cdata
        if ctype == b"IEND":
            break


def _make_chunk(ctype: bytes, cdata: bytes) -> bytes:
    length = struct.pack(">I", len(cdata))
    body = ctype + cdata
    crc = struct.pack(">I", zlib.crc32(body) & 0xFFFF_FFFF)
    return length + body + crc


# ---------------------------------------------------------------------------
# Reconstruction filters
# ---------------------------------------------------------------------------


def _recon_none(row: np.ndarray, _prev: np.ndarray, _bpp: int) -> np.ndarray:
    return row


def _recon_sub(row: np.ndarray, _prev: np.ndarray, bpp: int) -> np.ndarray:
    """Sub: Recon[x] = Filt[x] + Recon[x-bpp]  (mod 256).

    Equivalent to a cumulative-sum along pixel axis, mod 256 — fully vectorised
    for bpp-aligned rows (the common case for 8-bit RGB / RGBA).
    """
    n = len(row)
    if n % bpp == 0:
        # Reshape to (npixels, bpp), cumsum along pixels, mask to 8-bit
        arr = row.reshape(-1, bpp)
        return (np.cumsum(arr, axis=0) & 0xFF).ravel().astype(np.uint8)
    # Fallback (edge case: row not bpp-aligned)
    recon = row.copy()
    for i in range(bpp, n):
        recon[i] = (recon[i] + recon[i - bpp]) & 0xFF
    return recon.astype(np.uint8)


def _recon_up(row: np.ndarray, prev: np.ndarray, _bpp: int) -> np.ndarray:
    """Up: Recon[x] = Filt[x] + Prior[x]  (mod 256)."""
    return ((row + prev) & 0xFF).astype(np.uint8)


def _recon_avg(row: np.ndarray, prev: np.ndarray, bpp: int) -> np.ndarray:
    """Average: Recon[x] = Filt[x] + floor((Recon[x-bpp] + Prior[x]) / 2)."""
    recon = row.copy()
    n = len(recon)
    for i in range(n):
        a = int(recon[i - bpp]) if i >= bpp else 0
        b = int(prev[i])
        recon[i] = (int(row[i]) + (a + b) // 2) & 0xFF
    return recon.astype(np.uint8)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _recon_paeth(row: np.ndarray, prev: np.ndarray, bpp: int) -> np.ndarray:
    """Paeth: Recon[x] = Filt[x] + PaethPredictor(Recon[x-bpp], Prior[x], Prior[x-bpp])."""
    recon = row.copy()
    n = len(recon)
    for i in range(n):
        a = int(recon[i - bpp]) if i >= bpp else 0
        b = int(prev[i])
        c = int(prev[i - bpp]) if i >= bpp else 0
        recon[i] = (int(row[i]) + _paeth(a, b, c)) & 0xFF
    return recon.astype(np.uint8)


_RECON = [_recon_none, _recon_sub, _recon_up, _recon_avg, _recon_paeth]


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def decode_png(data: bytes) -> np.ndarray:
    """Decode a PNG file to a ``(H, W, C)`` uint8 numpy array.

    Supports 8-bit colour types 0 (grayscale), 2 (RGB), 4 (grayscale+alpha),
    and 6 (RGBA) — which cover all screenshots Playwright can produce.

    Parameters
    ----------
    data : bytes
        Raw PNG file contents.

    Returns
    -------
    np.ndarray
        Shape ``(height, width, channels)``, dtype ``uint8``.
    """
    if data[:8] != _PNG_SIG:
        raise ValueError("Not a valid PNG file (bad signature)")

    ihdr: tuple | None = None
    idat_parts: list[bytes] = []

    for ctype, cdata in _iter_chunks(data):
        if ctype == b"IHDR":
            w, h, bd, ct = struct.unpack(">IIBB", cdata[:10])
            ihdr = (w, h, bd, ct)
        elif ctype == b"IDAT":
            idat_parts.append(cdata)
        elif ctype == b"IEND":
            break

    if ihdr is None:
        raise ValueError("PNG has no IHDR chunk")

    w, h, bd, ct = ihdr
    if bd != 8:
        raise ValueError(f"Only 8-bit depth supported, got {bd}")

    channels = _CT_CHANNELS.get(ct)
    if channels is None:
        raise ValueError(f"Unsupported PNG colour type: {ct}")

    raw = zlib.decompress(b"".join(idat_parts))
    bpp = channels  # bytes per pixel for 8-bit
    stride = w * channels  # bytes per row (without filter byte)

    result = np.empty((h, w, channels), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)

    for y in range(h):
        base = y * (stride + 1)
        filt = raw[base]
        row = np.frombuffer(raw[base + 1 : base + 1 + stride], dtype=np.uint8).astype(
            np.int32
        )

        if filt > 4:
            raise ValueError(f"Unknown PNG filter type {filt} at row {y}")

        recon = _RECON[filt](row, prev, bpp)
        prev = recon.astype(np.int32)
        result[y] = recon.reshape(w, channels)

    return result


# ---------------------------------------------------------------------------
# Encoder  (filter 0 — no filtering; only used for writing baseline PNGs)
# ---------------------------------------------------------------------------


def encode_png(arr: np.ndarray) -> bytes:
    """Encode a ``(H, W, C)`` or ``(H, W)`` uint8 array as a PNG file.

    Uses filter type 0 (None) on every row — correct but not maximally
    compressed.  Intended for writing golden baselines, not production use.

    Parameters
    ----------
    arr : np.ndarray
        Shape ``(H, W, C)`` or ``(H, W)``, dtype ``uint8``.

    Returns
    -------
    bytes
        Raw PNG file contents.
    """
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    h, w, channels = arr.shape

    ct_map = {1: 0, 2: 0, 3: 2, 4: 6}  # channel count → PNG colour type
    # Override: 1 ch → grayscale (0), 2 ch → grayscale+alpha (4)
    ct_map = {1: 0, 2: 4, 3: 2, 4: 6}
    ct = ct_map.get(channels)
    if ct is None:
        raise ValueError(f"Unsupported channel count: {channels}")

    # Build raw scanlines: prepend a 0x00 filter byte to each row
    rows_bytes = bytearray()
    for y in range(h):
        rows_bytes.append(0)  # filter = None
        rows_bytes += arr[y].tobytes()

    compressed = zlib.compress(bytes(rows_bytes), level=1)

    sig = _PNG_SIG
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, ct, 0, 0, 0)
    return (
        sig
        + _make_chunk(b"IHDR", ihdr_data)
        + _make_chunk(b"IDAT", compressed)
        + _make_chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_arrays(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    tol: int = 8,
    max_diff_frac: float = 0.02,
) -> tuple[bool, str]:
    """Compare two uint8 image arrays pixel-by-pixel.

    A pixel is considered *different* when any of its channel values differ by
    more than *tol*.  The comparison passes as long as the fraction of
    different pixels is at most *max_diff_frac*.

    Parameters
    ----------
    actual, expected : np.ndarray
        ``(H, W, C)`` uint8 arrays.  They must have the same shape.
    tol : int
        Per-channel absolute tolerance (default 8, i.e. ~3 % of 255).
    max_diff_frac : float
        Maximum fraction of pixels allowed to differ (default 0.02 = 2 %).

    Returns
    -------
    (ok, message) : (bool, str)
    """
    if actual.shape != expected.shape:
        return False, (
            f"shape mismatch: actual {actual.shape} vs expected {expected.shape}"
        )

    # Work in int32 to avoid uint8 wrap-around
    diff = np.abs(actual.astype(np.int32) - expected.astype(np.int32))

    # A pixel fails if ANY channel exceeds tol
    bad_pixels = (diff > tol).any(axis=-1)  # (H, W) bool
    n_bad = int(bad_pixels.sum())
    n_total = bad_pixels.size
    frac = n_bad / n_total

    if frac > max_diff_frac:
        max_diff = int(diff.max())
        return False, (
            f"{n_bad}/{n_total} pixels ({frac:.1%}) differ by >{tol}; "
            f"max channel diff = {max_diff}"
        )

    return True, f"ok — {n_bad}/{n_total} pixels ({frac:.2%}) differ by >{tol}"
