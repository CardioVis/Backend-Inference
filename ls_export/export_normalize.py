"""Rename Label Studio YOLO export files to `frame_<digits>.<ext>`."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# e.g. d2afa4a8__dac34a71-frame_03979 → frame_03979
_STEM_RE = re.compile(
    r"^(?:[0-9a-f]{8}__)?[0-9a-f]{8}-(frame_\d+)$",
    re.IGNORECASE,
)


def _normalize_stem(stem: str) -> str | None:
    m = _STEM_RE.match(stem)
    return m.group(1) if m else None


def normalize_ls_yolo_filenames(extracted_dir: Path) -> int:
    """Rename files under extracted/images and extracted/labels to frame_NNNNN.*

    Returns count of files renamed. Exits on collision (two sources → same target).
    """
    n = 0
    for sub in ("images", "labels"):
        d = extracted_dir / sub
        if not d.is_dir():
            continue
        # Collect renames to apply (avoid partial renames while iterating)
        ops: list[tuple[Path, Path]] = []
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            new_stem = _normalize_stem(p.stem)
            if new_stem is None or new_stem == p.stem:
                continue
            target = d / f"{new_stem}{p.suffix}"
            ops.append((p, target))

        seen_targets: set[Path] = set()
        for src, dst in ops:
            if dst in seen_targets:
                print(
                    f"Error: duplicate rename target {dst} (collision). Fix export or names.\n",
                    file=sys.stderr,
                )
                sys.exit(1)
            seen_targets.add(dst)

        for src, dst in ops:
            if dst.exists() and dst.resolve() != src.resolve():
                print(
                    f"Error: cannot rename {src.name} → {dst.name}: target exists.\n",
                    file=sys.stderr,
                )
                sys.exit(1)
            src.rename(dst)
            n += 1
    return n
