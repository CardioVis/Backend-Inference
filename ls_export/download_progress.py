"""Optional tqdm progress for Label Studio export byte streams."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from typing import BinaryIO


def _tqdm_disabled() -> bool:
    return os.environ.get("LABEL_STUDIO_NO_TQDM", "").strip().lower() in ("1", "true", "yes")


def write_download_stream(
    stream: Iterator[bytes],
    out: BinaryIO,
    *,
    desc: str,
) -> int:
    """Write chunks from `stream` to `out`; show tqdm on stderr unless disabled."""
    if _tqdm_disabled():
        n = 0
        for chunk in stream:
            out.write(chunk)
            n += len(chunk)
        return n
    try:
        from tqdm import tqdm
    except ImportError:
        n = 0
        for chunk in stream:
            out.write(chunk)
            n += len(chunk)
        return n
    n = 0
    with tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
        file=sys.stderr,
        mininterval=0.3,
    ) as pbar:
        for chunk in stream:
            out.write(chunk)
            ln = len(chunk)
            n += ln
            pbar.update(ln)
    return n


def download_stream_to_bytes(stream: Iterator[bytes], *, desc: str) -> bytes:
    """Consume `stream` into a single bytes object with tqdm on stderr unless disabled."""
    if _tqdm_disabled():
        return b"".join(stream)
    try:
        from tqdm import tqdm
    except ImportError:
        return b"".join(stream)
    buf = bytearray()
    with tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
        file=sys.stderr,
        mininterval=0.3,
    ) as pbar:
        for chunk in stream:
            buf.extend(chunk)
            pbar.update(len(chunk))
    return bytes(buf)
