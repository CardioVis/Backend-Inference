"""Shared checkpoint helpers (notebook + overlay_video_segment)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def normalize_state_dict(ckpt: Any) -> dict[str, Any]:
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
    out: dict[str, Any] = {}
    for k, v in ckpt.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def infer_num_classes(state: dict[str, Any]) -> int:
    key = "classifier.classifier.3.weight"
    if key not in state:
        raise KeyError(
            f"Cannot infer num_classes (missing {key!r}). Sample keys: {list(state.keys())[:10]}"
        )
    return int(state[key].shape[0])


def load_checkpoint_state(path: Path | str, map_location: torch.device | str) -> dict[str, Any]:
    path = Path(path)
    try:
        raw = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        raw = torch.load(path, map_location=map_location)
    return normalize_state_dict(raw)
