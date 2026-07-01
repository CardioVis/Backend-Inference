"""Default CLASS_NAMES / EXPORT_MASK_LAYERS / overlay constants (18 logits: 0=bg, 1–17 foreground)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

TAG_TO_CLASS = {
    "phrenic nerve": 1,
    "pericardium": 2,
    "epicardial adipose tissue": 3,
    "aortic root": 4,
    "auricles": 5,
    "pericardium boundary": 6,
    "needle holders": 7,
    "Cardioplegia cannula": 8,
    "Surgical pledget": 9,
    "Suction": 10,
    "Chitwood aortic cross-lamp": 11,
    "Left suction tube": 12,
    "Snare tubing": 13,
    "Electrocautery": 14,
    "Right Ventricle": 15,
    "Long Forceps": 16,
    "Epicardial fat on aortic": 17,
}

CLASS_NAMES = [
    "Background",
    "phrenic nerve",
    "pericardium",
    "epicardial adipose tissue",
    "aortic root",
    "auricles",
    "pericardium boundary",
    "needle holders",
    "Cardioplegia cannula",
    "Surgical pledget",
    "Suction",
    "Chitwood aortic cross-lamp",
    "Left suction tube",
    "Snare tubing",
    "Electrocautery",
    "Right Ventricle",
    "Long Forceps",
    "Epicardial fat on aortic",
]

CLASS_COLORS_BGR = [
    (0, 0, 0),           # 0 Background
    (0, 165, 255),       # 1 phrenic nerve
    (0, 200, 0),         # 2 pericardium (guideline anchor; band uses GUIDELINE_COLOR_BGR)
    (255, 0, 0),         # 3 epicardial adipose tissue
    (255, 128, 0),       # 4 aortic root
    (255, 0, 255),       # 5 auricles
    (0, 255, 255),       # 6 pericardium boundary
    (180, 105, 255),     # 7 needle holders
    (42, 42, 165),       # 8 Cardioplegia cannula
    (255, 0, 255),       # 9 Surgical pledget
    (0, 255, 0),         # 10 Suction
    (0, 140, 255),       # 11 Chitwood aortic cross-lamp
    (255, 191, 0),       # 12 Left suction tube
    (203, 192, 255),     # 13 Snare tubing
    (0, 0, 255),         # 14 Electrocautery
    (128, 0, 128),       # 15 Right Ventricle
    (19, 69, 139),       # 16 Long Forceps
    (32, 165, 218),      # 17 Epicardial fat on aortic
]

# Optional uint8 LUT of length NUM_MODEL_CLASSES when checkpoint head size != len(CLASS_NAMES).
MODEL_CLASS_TO_SEMANTIC: np.ndarray | None = None

assert len(CLASS_COLORS_BGR) == len(CLASS_NAMES)


def _export_slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _validate_tag_to_class() -> None:
    seen: dict[int, str] = {}
    for tag, class_id in TAG_TO_CLASS.items():
        if class_id <= 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(
                f"TAG_TO_CLASS[{tag!r}]={class_id} out of range [1, {len(CLASS_NAMES) - 1}]"
            )
        if class_id in seen:
            raise ValueError(
                f"Duplicate class id {class_id} for tags {seen[class_id]!r} and {tag!r}"
            )
        seen[class_id] = tag


_validate_tag_to_class()

ALL_FOREGROUND_IDS = tuple(range(1, len(CLASS_NAMES)))
GUIDELINE_ANCHOR_CLASS_ID = TAG_TO_CLASS["pericardium"]
OVERLAY_ALPHA_CLASS_IDS = tuple(c for c in ALL_FOREGROUND_IDS if c != GUIDELINE_ANCHOR_CLASS_ID)
CALLOUT_CLASS_IDS = ALL_FOREGROUND_IDS
DISPLAY_CLASS_NAMES: dict[int, str] = {}
BLINK_WARNING_CLASS_ID = TAG_TO_CLASS["phrenic nerve"]
BLINK_WARNING_PERIOD_FRAMES = 8


def build_export_mask_layers(
    class_names: list[str] | None = None,
    guideline_anchor_class_id: int | None = None,
) -> list[dict[str, Any]]:
    """One export layer per foreground class (ids 1 .. len(class_names)-1)."""
    names = class_names if class_names is not None else CLASS_NAMES
    anchor = (
        guideline_anchor_class_id
        if guideline_anchor_class_id is not None
        else GUIDELINE_ANCHOR_CLASS_ID
    )
    layers: list[dict[str, Any]] = []
    for class_id in range(1, len(names)):
        label = names[class_id]
        slug = _export_slug(label)
        note = f"0/255 binary, model id {class_id} ({label})."
        if class_id == anchor:
            note += " Guideline anchor (centerline / band)."
        layers.append(
            {
                "class_id": class_id,
                "subdir": f"class{class_id:02d}_{slug}",
                "file_prefix": f"class{class_id:02d}_",
                "json_key": f"class_{class_id}_{slug}",
                "note": note,
            }
        )
    return layers


EXPORT_MASK_LAYERS = build_export_mask_layers()

# Downstream JSON uses key "pericardium_class"; value is the guideline anchor id.
PERICARDIUM_CLASS = GUIDELINE_ANCHOR_CLASS_ID

ALPHA = 0.5
GUIDELINE_COLOR_BGR = (0, 255, 0)
OFFSET_CM = 1.2
PIXELS_PER_CM = 20.0
OFFSET_COLOR_POS_BGR = (0, 140, 255)
OFFSET_COLOR_NEG_BGR = (255, 0, 0)
DASH_LEN = 12
GAP_LEN = 8
LINE_THICKNESS = 2
OFFSET_LINE_THICKNESS = 2
TV_WEIGHT = 0.15
TEMPORAL_ALPHA = 0.25
MAX_CENTERLINE_JUMP_PX = 25
BAND_WIDTH_SCALE = 12.0
BAND_ALPHA = 0.18
BAND_COLOR_BGR = (0, 255, 0)
CALLOUT_TEXT_COLOR = (255, 255, 255)
CALLOUT_BG_COLOR = (20, 20, 20)
CALLOUT_ARROW_COLOR = (255, 255, 255)
CALLOUT_FONT_SCALE = 0.55
CALLOUT_THICKNESS = 1
CALLOUT_PADDING = 6
CALLOUT_MIN_MASK_PIXELS = 1
CALLOUT_OFFSET_CM = 2.0
DANGER_TRIANGLE_COLOR = (0, 0, 255)
DANGER_TRIANGLE_OUTLINE = (255, 255, 255)


def make_overlay_runtime_kwargs(device: torch.device, img_size: int) -> dict[str, Any]:
    """Keyword args for guideline_overlay_core.set_overlay_runtime (notebook globals)."""
    return {
        "DEVICE": device,
        "IMG_SIZE": img_size,
        "CLASS_NAMES": CLASS_NAMES,
        "CLASS_COLORS_BGR": CLASS_COLORS_BGR,
        "GUIDELINE_ANCHOR_CLASS_ID": GUIDELINE_ANCHOR_CLASS_ID,
        "OVERLAY_ALPHA_CLASS_IDS": OVERLAY_ALPHA_CLASS_IDS,
        "CALLOUT_CLASS_IDS": CALLOUT_CLASS_IDS,
        "DISPLAY_CLASS_NAMES": DISPLAY_CLASS_NAMES,
        "BLINK_WARNING_CLASS_ID": BLINK_WARNING_CLASS_ID,
        "BLINK_WARNING_PERIOD_FRAMES": BLINK_WARNING_PERIOD_FRAMES,
        "ALPHA": ALPHA,
        "GUIDELINE_COLOR_BGR": GUIDELINE_COLOR_BGR,
        "OFFSET_CM": OFFSET_CM,
        "PIXELS_PER_CM": PIXELS_PER_CM,
        "OFFSET_COLOR_POS_BGR": OFFSET_COLOR_POS_BGR,
        "OFFSET_COLOR_NEG_BGR": OFFSET_COLOR_NEG_BGR,
        "DASH_LEN": DASH_LEN,
        "GAP_LEN": GAP_LEN,
        "LINE_THICKNESS": LINE_THICKNESS,
        "OFFSET_LINE_THICKNESS": OFFSET_LINE_THICKNESS,
        "TV_WEIGHT": TV_WEIGHT,
        "TEMPORAL_ALPHA": TEMPORAL_ALPHA,
        "MAX_CENTERLINE_JUMP_PX": MAX_CENTERLINE_JUMP_PX,
        "BAND_WIDTH_SCALE": BAND_WIDTH_SCALE,
        "BAND_ALPHA": BAND_ALPHA,
        "BAND_COLOR_BGR": BAND_COLOR_BGR,
        "CALLOUT_TEXT_COLOR": CALLOUT_TEXT_COLOR,
        "CALLOUT_BG_COLOR": CALLOUT_BG_COLOR,
        "CALLOUT_ARROW_COLOR": CALLOUT_ARROW_COLOR,
        "CALLOUT_FONT_SCALE": CALLOUT_FONT_SCALE,
        "CALLOUT_THICKNESS": CALLOUT_THICKNESS,
        "CALLOUT_PADDING": CALLOUT_PADDING,
        "CALLOUT_MIN_MASK_PIXELS": CALLOUT_MIN_MASK_PIXELS,
        "CALLOUT_OFFSET_CM": CALLOUT_OFFSET_CM,
        "DANGER_TRIANGLE_COLOR": DANGER_TRIANGLE_COLOR,
        "DANGER_TRIANGLE_OUTLINE": DANGER_TRIANGLE_OUTLINE,
    }


def configure_notebook_overlay(device: torch.device, img_size: int) -> None:
    """Use from notebook after DEVICE / IMG_SIZE and config globals are defined.

    Any matching names already defined on the notebook ``__main__`` module (e.g.
    ``CLASS_NAMES``, ``EXPORT_MASK_LAYERS``) override the defaults from this module.
    """
    import __main__ as _main

    from guideline_overlay_core import set_overlay_runtime

    g = vars(_main)
    kw = make_overlay_runtime_kwargs(device, img_size)
    for k in list(kw.keys()):
        if k in g and k not in ("DEVICE", "IMG_SIZE"):
            kw[k] = g[k]
    set_overlay_runtime(**kw)


def build_model_pred_to_semantic(
    num_model_classes: int,
    class_names: list[str],
    model_class_to_semantic: np.ndarray | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Map argmax class ids to uint8 semantic ids (same rules as the notebook model cell)."""

    def model_pred_to_semantic(pred: np.ndarray) -> np.ndarray:
        p = np.asarray(pred, dtype=np.int32)
        if model_class_to_semantic is not None:
            lut = np.asarray(model_class_to_semantic, dtype=np.uint8)
            if lut.shape[0] != num_model_classes:
                raise ValueError(
                    f"MODEL_CLASS_TO_SEMANTIC length {lut.shape[0]} != NUM_MODEL_CLASSES {num_model_classes}"
                )
            return lut[p]
        if len(class_names) == num_model_classes:
            return p.astype(np.uint8)
        raise ValueError(
            f"NUM_MODEL_CLASSES={num_model_classes} but len(CLASS_NAMES)={len(class_names)}. "
            "Set CLASS_NAMES to match checkpoint order/size, or set MODEL_CLASS_TO_SEMANTIC to a "
            "uint8 ndarray of shape (NUM_MODEL_CLASSES,) mapping model class ids to semantic ids."
        )

    return model_pred_to_semantic


def assert_class_names_match_checkpoint(
    num_model_classes: int,
    class_names: list[str],
    model_class_to_semantic: np.ndarray | None,
) -> None:
    if model_class_to_semantic is None:
        assert len(class_names) == num_model_classes, (
            f"len(CLASS_NAMES)={len(class_names)} != NUM_MODEL_CLASSES={num_model_classes}; "
            "align CLASS_NAMES with the checkpoint or set MODEL_CLASS_TO_SEMANTIC."
        )
