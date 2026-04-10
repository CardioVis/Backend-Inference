"""Default CLASS_NAMES / EXPORT_MASK_LAYERS / overlay constants (aligned with predict_frames_guideline_overlay cell 0)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

TAG_TO_CLASS = {
    "epicardial adipose tissue": 1,
    "pericardium": 2,
    "phrenic nerve": 3,
    "aortic root": 4,
    "auricles": 5,
    "epicardial fat on aortic": 6,
    "grasper": 7,
    "needle holders": 8,
    "pericardial stay sutures": 9,
    "pericardium boundary": 10,
}

CLASS_NAMES = [
    "Background",
    "Epicardial adipose tissue",
    "Pericardium",
    "Phrenic nerve",
    "Aortic root",
    "Auricles",
    "Epicardial fat on aortic",
    "Grasper",
    "Needle holders",
    "Pericardial stay sutures",
    "Pericardium boundary",
]

CLASS_COLORS_BGR = [
    (0, 0, 0),
    (0, 165, 255),
    (0, 200, 0),
    (255, 0, 0),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
    (180, 105, 255),
    (42, 42, 165),
    (255, 0, 255),
    (0, 255, 0),
]

ALL_FOREGROUND_IDS = tuple(range(1, len(CLASS_NAMES)))
GUIDELINE_ANCHOR_CLASS_ID = TAG_TO_CLASS["pericardium"]
OVERLAY_ALPHA_CLASS_IDS = tuple(c for c in ALL_FOREGROUND_IDS if c != GUIDELINE_ANCHOR_CLASS_ID)
CALLOUT_CLASS_IDS = ALL_FOREGROUND_IDS
DISPLAY_CLASS_NAMES: dict[int, str] = {}
BLINK_WARNING_CLASS_ID = TAG_TO_CLASS["phrenic nerve"]
BLINK_WARNING_PERIOD_FRAMES = 8

EXPORT_MASK_LAYERS = [
    {
        "class_id": 1,
        "subdir": "class01_epicardial_adipose",
        "file_prefix": "class01_",
        "json_key": "class_1_epicardial_adipose_tissue",
        "note": "0/255 binary, model id 1 (epicardial adipose tissue).",
    },
    {
        "class_id": 2,
        "subdir": "class02_pericardium",
        "file_prefix": "class02_",
        "json_key": "class_2_pericardium_full_segmentation",
        "note": "0/255 binary, model id 2 (pericardium); guideline anchor.",
    },
    {
        "class_id": 3,
        "subdir": "class03_phrenic",
        "file_prefix": "class03_",
        "json_key": "class_3_phrenic_nerve",
        "note": "0/255 binary, model id 3 (phrenic nerve).",
    },
    {
        "class_id": 4,
        "subdir": "class04_aortic_root",
        "file_prefix": "class04_",
        "json_key": "class_4_aortic_root",
        "note": "0/255 binary, model id 4 (aortic root).",
    },
    {
        "class_id": 5,
        "subdir": "class05_auricles",
        "file_prefix": "class05_",
        "json_key": "class_5_auricles",
        "note": "0/255 binary, model id 5 (auricles).",
    },
    {
        "class_id": 6,
        "subdir": "class06_epicardial_fat_aortic",
        "file_prefix": "class06_",
        "json_key": "class_6_epicardial_fat_on_aortic",
        "note": "0/255 binary, model id 6 (epicardial fat on aortic).",
    },
    {
        "class_id": 7,
        "subdir": "class07_grasper",
        "file_prefix": "class07_",
        "json_key": "class_7_grasper",
        "note": "0/255 binary, model id 7 (grasper).",
    },
    {
        "class_id": 8,
        "subdir": "class08_needle_holders",
        "file_prefix": "class08_",
        "json_key": "class_8_needle_holders",
        "note": "0/255 binary, model id 8 (needle holders).",
    },
    {
        "class_id": 9,
        "subdir": "class09_pericardial_stay_sutures",
        "file_prefix": "class09_",
        "json_key": "class_9_pericardial_stay_sutures",
        "note": "0/255 binary, model id 9 (pericardial stay sutures).",
    },
    {
        "class_id": 10,
        "subdir": "class10_pericardium_boundary",
        "file_prefix": "class10_",
        "json_key": "class_10_pericardium_boundary",
        "note": "0/255 binary, model id 10 (pericardium boundary).",
    },
]

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
