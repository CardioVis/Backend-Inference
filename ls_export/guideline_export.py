"""Build guideline_seq_export-style ann_*.json + frames + masks from normalized YOLO extract."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ls_export.guideline_geometry import get_pericardium_centerline


def _cv2():
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "Guideline export requires OpenCV: pip install opencv-python-headless"
        ) from e
    return cv2

_DEFAULT_CLASS_NAMES: dict[str, str] = {
    "0": "Background",
    "1": "Epicardial adipose tissue",
    "2": "Pericardium",
    "3": "Phrenic nerve",
}

_DEFAULT_PARAMS: dict[str, float] = {
    "OFFSET_CM": 1.2,
    "PIXELS_PER_CM": 20.0,
    "offset_px": 24.0,
    "BAND_WIDTH_SCALE": 12.0,
    "band_halfwidth_px": 144.0,
}

_DEFAULT_MASK_NOTES: dict[str, str] = {
    "band_guideline_between_parallel": "MVP: empty band (all-zero PNG); no parallel geometry from LS YOLO export.",
    "class_1_epicardial_adipose": "0/255 from YOLO boxes mapped to class 1 (axis-aligned rectangles).",
    "class_2_pericardium_full_segmentation": "0/255 from YOLO boxes mapped to class 2 (axis-aligned rectangles).",
    "class_3_phrenic_nerve": "0/255 from YOLO boxes mapped to class 3 (axis-aligned rectangles).",
}


def _frame_sort_key(stem: str) -> tuple[int, str]:
    m = re.search(r"frame_(\d+)", stem, re.IGNORECASE)
    return (int(m.group(1)), stem) if m else (10**9, stem)


def _load_mapping(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("yolo_to_guideline") or data.get("yoloToGuideline")
    if not isinstance(raw, dict) or not raw:
        print(
            "guideline mapping JSON must contain non-empty object 'yolo_to_guideline': "
            '{ "0": 1, "1": 2, ... } (YOLO class id → guideline mask class 1–3).\n',
            file=sys.stderr,
        )
        sys.exit(1)
    yolo_to_guideline: dict[int, int] = {}
    for k, v in raw.items():
        try:
            yi = int(k)
            gi = int(v)
        except (TypeError, ValueError):
            continue
        if gi not in (1, 2, 3):
            print(f"guideline class must be 1–3, got {gi} for YOLO class {k}\n", file=sys.stderr)
            sys.exit(1)
        yolo_to_guideline[yi] = gi
    data["_parsed_yolo_to_guideline"] = yolo_to_guideline
    return data


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _parse_yolo_lines(text: str) -> list[tuple[int, float, float, float, float]]:
    out: list[tuple[int, float, float, float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        out.append((cid, cx, cy, w, h))
    return out


def _draw_yolo_boxes_on_masks(
    h: int,
    w: int,
    boxes: list[tuple[int, float, float, float, float]],
    yolo_to_guideline: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m1 = np.zeros((h, w), dtype=np.uint8)
    m2 = np.zeros((h, w), dtype=np.uint8)
    m3 = np.zeros((h, w), dtype=np.uint8)
    masks = {1: m1, 2: m2, 3: m3}
    for cid, cx, cy, bw, bh in boxes:
        g = yolo_to_guideline.get(cid)
        if g is None:
            continue
        x1 = max(0, min(w - 1, int(round((cx - bw / 2.0) * w))))
        x2 = max(0, min(w - 1, int(round((cx + bw / 2.0) * w))))
        y1 = max(0, min(h - 1, int(round((cy - bh / 2.0) * h))))
        y2 = max(0, min(h - 1, int(round((cy + bh / 2.0) * h))))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x1 == x2 or y1 == y2:
            continue
        masks[g][y1 : y2 + 1, x1 : x2 + 1] = 255
    return m1, m2, m3


def write_guideline_seq_from_extract(
    extracted_dir: Path,
    mapping_path: Path,
    run_dir: Path,
    *,
    subdir: str = "guideline_seq",
) -> Path:
    """Read extracted/images, extracted/labels (normalized names), write run_dir/subdir/{frames,masks,json}."""
    cv2 = _cv2()
    mapping = _load_mapping(mapping_path)
    yolo_to_guideline: dict[int, int] = mapping["_parsed_yolo_to_guideline"]

    images_dir = extracted_dir / "images"
    labels_dir = extracted_dir / "labels"
    if not labels_dir.is_dir():
        print(f"No labels dir: {labels_dir}\n", file=sys.stderr)
        sys.exit(1)

    out_root = run_dir / subdir
    frames_out = out_root / "frames"
    masks_band = out_root / "masks" / "band"
    masks_c1 = out_root / "masks" / "class01_epicardial"
    masks_c2 = out_root / "masks" / "class02_pericardium"
    masks_c3 = out_root / "masks" / "class03_phrenic"
    json_out = out_root / "json"
    for d in (frames_out, masks_band, masks_c1, masks_c2, masks_c3, json_out):
        d.mkdir(parents=True, exist_ok=True)

    label_files = [p for p in labels_dir.iterdir() if p.suffix.lower() == ".txt" and p.is_file()]
    label_files.sort(key=lambda p: _frame_sort_key(p.stem))

    class_names = mapping.get("class_names")
    if not isinstance(class_names, dict):
        class_names = dict(_DEFAULT_CLASS_NAMES)

    params = mapping.get("params")
    if not isinstance(params, dict):
        params = dict(_DEFAULT_PARAMS)
    else:
        merged = dict(_DEFAULT_PARAMS)
        for k, v in params.items():
            if isinstance(v, (int, float)):
                merged[str(k)] = float(v)
        params = merged

    task = str(mapping.get("task") or "label_studio_yolo_export")
    frame_note = str(mapping.get("frame_note") or "From Label Studio YOLO export (normalized frame_* names).")
    predict_note = str(
        mapping.get("predict_note")
        or "MVP: masks rasterized from YOLO boxes; centerline from pericardium (class 2) mask."
    )
    mask_notes = mapping.get("mask_notes")
    if not isinstance(mask_notes, dict):
        mask_notes = dict(_DEFAULT_MASK_NOTES)
    else:
        mn = dict(_DEFAULT_MASK_NOTES)
        mn.update({str(k): str(v) for k, v in mask_notes.items()})
        mask_notes = mn

    pericardium_class = int(mapping.get("pericardium_class", 2))

    for idx, lab_path in enumerate(label_files):
        stem = lab_path.stem
        img_path = _find_image(images_dir, stem)
        if img_path is None:
            print(f"Warning: no image for {lab_path.name}, skipping.\n", file=sys.stderr)
            continue

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Warning: could not read {img_path}, skipping.\n", file=sys.stderr)
            continue
        h, w = bgr.shape[:2]

        boxes = _parse_yolo_lines(lab_path.read_text(encoding="utf-8", errors="replace"))
        m1, m2, m3 = _draw_yolo_boxes_on_masks(h, w, boxes, yolo_to_guideline)
        pcm = {1: m1, 2: m2, 3: m3}.get(pericardium_class, m2)
        pericardium_mask = pcm
        segments = get_pericardium_centerline(pericardium_mask, gap_threshold=5, smooth_win=5)

        frame_rel = f"frames/frame_{idx:06d}.png"
        cv2.imwrite(str(frames_out / f"frame_{idx:06d}.png"), bgr)

        band_path = masks_band / f"band_{idx:06d}.png"
        cv2.imwrite(str(band_path), np.zeros((h, w), dtype=np.uint8))
        cv2.imwrite(str(masks_c1 / f"class01_{idx:06d}.png"), m1)
        cv2.imwrite(str(masks_c2 / f"class02_{idx:06d}.png"), m2)
        cv2.imwrite(str(masks_c3 / f"class03_{idx:06d}.png"), m3)

        src_template = mapping.get("source_path_template")
        if isinstance(src_template, str) and "{stem}" in src_template:
            source_path = src_template.replace("{stem}", stem).replace("{index}", str(idx))
        elif isinstance(src_template, str):
            source_path = src_template
        else:
            source_path = str(img_path.resolve())

        ann: dict[str, Any] = {
            "frame_index": idx,
            "source_path": source_path,
            "task": task,
            "frame_file": frame_rel,
            "frame_note": frame_note,
            "predict_note": predict_note,
            "mask_files": {
                "band_guideline_between_parallel": f"masks/band/band_{idx:06d}.png",
                "class_1_epicardial_adipose": f"masks/class01_epicardial/class01_{idx:06d}.png",
                "class_2_pericardium_full_segmentation": f"masks/class02_pericardium/class02_{idx:06d}.png",
                "class_3_phrenic_nerve": f"masks/class03_phrenic/class03_{idx:06d}.png",
            },
            "mask_notes": mask_notes,
            "image_size": {"width": w, "height": h},
            "class_names": class_names,
            "pericardium_class": pericardium_class,
            "params": params,
            "main_centerline_segment_index": 0,
            "centerline_segments": segments,
            "parallel_lines_offset_cm": {"positive_normal_side": [], "negative_normal_side": []},
            "band_polygons_between_parallel_bounds": [],
        }

        out_json = json_out / f"ann_{idx:06d}.json"
        out_json.write_text(json.dumps(ann, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return out_root
