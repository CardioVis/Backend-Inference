#!/usr/bin/env python3
"""One-off generator: notebook cell2 -> guideline_overlay_core.py (run from repo root)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
src = Path("/tmp/cell2_extract.py").read_text(encoding="utf-8")

# Longest token first (word boundaries).
TOKENS = [
    "OVERLAY_ALPHA_CLASS_IDS",
    "GUIDELINE_ANCHOR_CLASS_ID",
    "BLINK_WARNING_PERIOD_FRAMES",
    "BLINK_WARNING_CLASS_ID",
    "MAX_CENTERLINE_JUMP_PX",
    "CALLOUT_MIN_MASK_PIXELS",
    "DISPLAY_CLASS_NAMES",
    "DANGER_TRIANGLE_OUTLINE",
    "DANGER_TRIANGLE_COLOR",
    "CALLOUT_ARROW_COLOR",
    "CALLOUT_TEXT_COLOR",
    "CALLOUT_BG_COLOR",
    "CALLOUT_CLASS_IDS",
    "CALLOUT_FONT_SCALE",
    "CALLOUT_THICKNESS",
    "CALLOUT_PADDING",
    "CALLOUT_OFFSET_CM",
    "CLASS_COLORS_BGR",
    "BAND_WIDTH_SCALE",
    "GUIDELINE_COLOR_BGR",
    "TEMPORAL_ALPHA",
    "LINE_THICKNESS",
    "PIXELS_PER_CM",
    "OFFSET_CM",
    "BAND_COLOR_BGR",
    "BAND_ALPHA",
    "IMG_SIZE",
    "DASH_LEN",
    "GAP_LEN",
    "TV_WEIGHT",
    "CLASS_NAMES",
    "DEVICE",
    "ALPHA",
]

for t in TOKENS:
    src = re.sub(rf"\b{re.escape(t)}\b", f"_cfg().{t}", src)

header = '''"""Guideline overlay geometry + visualization (from predict_frames_guideline_overlay notebook).

Call :func:`set_overlay_runtime` with the same keys the notebook used as UPPER_SNAKE globals
before using :func:`preprocess_frame` or :func:`overlay_guideline`.
"""
from __future__ import annotations

import types
from typing import Any, Optional

import cv2
import numpy as np
import torch
from skimage.restoration import denoise_tv_chambolle

_RT: Any = None


def set_overlay_runtime(**kwargs: Any) -> None:
    """Bind DEVICE, IMG_SIZE, CLASS_NAMES, GUIDELINE_ANCHOR_CLASS_ID, … (notebook globals)."""
    global _RT
    _RT = types.SimpleNamespace(**kwargs)


def _cfg() -> Any:
    if _RT is None:
        raise RuntimeError("guideline_overlay_core: call set_overlay_runtime(...) first")
    return _RT


'''

out = ROOT / "guideline_overlay_core.py"
out.write_text(header + src, encoding="utf-8")
print("Wrote", out, "lines", len(out.read_text().splitlines()))
