"""Task 2 export: frames/, masks/*, json/ann_*.json, optional zip (notebook parity)."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch

import guideline_export_config as gcfg
from guideline_overlay_core import _cfg, overlay_guideline, preprocess_frame


def _write_one_export_frame(
    *,
    export_dir: Path,
    frame: np.ndarray,
    path: Path,
    pred: np.ndarray,
    geom: dict,
    band_mask: np.ndarray,
    stem: str,
    frame_index: int,
    image_width: int,
    image_height: int,
    mask_as_bgr: bool,
    export_mask_layers: list[dict],
    class_names: list[str],
    guideline_anchor_class_id: int,
    offset_cm: float,
    pixels_per_cm: float,
    band_width_scale: float,
    offset_px: float,
    band_halfwidth_px: float,
    task_json_label: str,
) -> None:
    mr = cv2.resize(
        np.asarray(pred, dtype=np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(str(export_dir / "frames" / f"frame_{stem}.png"), frame)
    _band_out = (
        cv2.cvtColor(band_mask, cv2.COLOR_GRAY2BGR)
        if mask_as_bgr and band_mask.ndim == 2
        else band_mask
    )
    cv2.imwrite(str(export_dir / "masks" / "band" / f"band_{stem}.png"), _band_out)
    _fg = ((mr > 0).astype(np.uint8) * 255)
    _fg_out = cv2.cvtColor(_fg, cv2.COLOR_GRAY2BGR) if mask_as_bgr else _fg
    cv2.imwrite(str(export_dir / "masks" / "foreground" / f"foreground_{stem}.png"), _fg_out)
    mask_files = {
        "band_guideline_between_parallel": f"masks/band/band_{stem}.png",
        "foreground_any_non_background": f"masks/foreground/foreground_{stem}.png",
    }
    mask_notes = {
        "band_guideline_between_parallel": (
            "0/255 band region (guideline), not full anchor-class mask."
        ),
        "foreground_any_non_background": (
            "0/255 union of all predicted classes (mr>0); BGR PNG if mask_as_bgr."
        ),
    }
    for layer in export_mask_layers:
        cid = int(layer["class_id"])
        sub = layer["subdir"]
        pfx = layer["file_prefix"]
        jk = layer["json_key"]
        note = layer.get("note", "")
        bin_m = ((mr == cid).astype(np.uint8) * 255)
        _bin_out = cv2.cvtColor(bin_m, cv2.COLOR_GRAY2BGR) if mask_as_bgr else bin_m
        fn = f"{pfx}{stem}.png"
        cv2.imwrite(str(export_dir / "masks" / sub / fn), _bin_out)
        mask_files[jk] = f"masks/{sub}/{fn}"
        mask_notes[jk] = note
    payload = {
        "frame_index": frame_index,
        "source_path": str(path),
        "task": task_json_label,
        "frame_file": f"frames/frame_{stem}.png",
        "frame_note": "raw BGR, không overlay",
        "predict_note": "Single model.forward; class masks from argmax (no second inference).",
        "mask_files": mask_files,
        "mask_notes": mask_notes,
        "image_size": {"width": int(image_width), "height": int(image_height)},
        "class_names": {str(k): v for k, v in enumerate(class_names)},
        "pericardium_class": int(guideline_anchor_class_id),
        "params": {
            "OFFSET_CM": float(offset_cm),
            "PIXELS_PER_CM": float(pixels_per_cm),
            "offset_px": float(offset_px),
            "BAND_WIDTH_SCALE": float(band_width_scale),
            "band_halfwidth_px": float(band_halfwidth_px),
        },
        **geom,
    }
    with open(export_dir / "json" / f"ann_{stem}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _ensure_export_tree(export_dir: Path, export_mask_layers: list[dict]) -> None:
    (export_dir / "frames").mkdir(parents=True, exist_ok=True)
    (export_dir / "masks").mkdir(parents=True, exist_ok=True)
    (export_dir / "masks" / "band").mkdir(parents=True, exist_ok=True)
    (export_dir / "masks" / "foreground").mkdir(parents=True, exist_ok=True)
    for layer in export_mask_layers:
        (export_dir / "masks" / layer["subdir"]).mkdir(parents=True, exist_ok=True)
    (export_dir / "json").mkdir(parents=True, exist_ok=True)


def run_frames_guideline_pipeline(
    frame_paths: Sequence[Path],
    model: torch.nn.Module,
    model_pred_to_semantic: Callable[[np.ndarray], np.ndarray],
    *,
    video_writer: cv2.VideoWriter | None = None,
    export_dir: Path | None = None,
    export_zip: Path | None = None,
    clean_export: bool = True,
    write_zip: bool = True,
    mask_as_bgr: bool = True,
    export_mask_layers: list[dict] | None = None,
    class_names: list[str] | None = None,
    guideline_anchor_class_id: int | None = None,
    progress_every: int = 100,
    task_json_label: str = "export_500",
    video_progress_message: str = "Processed",
) -> None:
    """One inference loop: optional guideline overlay MP4 + optional Task 2 folder + zip."""
    layers = export_mask_layers if export_mask_layers is not None else gcfg.EXPORT_MASK_LAYERS
    names = class_names if class_names is not None else gcfg.CLASS_NAMES
    anchor = (
        guideline_anchor_class_id
        if guideline_anchor_class_id is not None
        else gcfg.GUIDELINE_ANCHOR_CLASS_ID
    )

    rt = _cfg()
    offset_cm = float(rt.OFFSET_CM)
    pixels_per_cm = float(rt.PIXELS_PER_CM)
    band_width_scale = float(rt.BAND_WIDTH_SCALE)
    offset_px = offset_cm * pixels_per_cm
    band_half = offset_px * (band_width_scale / 2.0)

    h_e = w_e = 0
    if export_dir is not None:
        first_e = cv2.imread(str(frame_paths[0]))
        if first_e is None:
            raise SystemExit(f"Cannot read first export frame: {frame_paths[0]}")
        h_e, w_e = first_e.shape[:2]
        if clean_export:
            shutil.rmtree(export_dir, ignore_errors=True)
        _ensure_export_tree(export_dir, layers)

    prev_centerline_seg = None
    with torch.no_grad():
        for i, path in enumerate(frame_paths):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            img_t, (orig_h, orig_w) = preprocess_frame(frame)
            logits = model(img_t)
            pred = model_pred_to_semantic(
                torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            )
            overlay_vis, prev_centerline_seg, geom, _mask_resized, band_mask = overlay_guideline(
                frame, pred, orig_h, orig_w, prev_centerline_seg, frame_idx=i
            )
            if video_writer is not None:
                video_writer.write(overlay_vis)
            if export_dir is not None:
                stem = f"{i:06d}"
                _write_one_export_frame(
                    export_dir=export_dir,
                    frame=frame,
                    path=path,
                    pred=pred,
                    geom=geom,
                    band_mask=band_mask,
                    stem=stem,
                    frame_index=i,
                    image_width=w_e,
                    image_height=h_e,
                    mask_as_bgr=mask_as_bgr,
                    export_mask_layers=layers,
                    class_names=names,
                    guideline_anchor_class_id=anchor,
                    offset_cm=offset_cm,
                    pixels_per_cm=pixels_per_cm,
                    band_width_scale=band_width_scale,
                    offset_px=offset_px,
                    band_halfwidth_px=band_half,
                    task_json_label=task_json_label,
                )
            if progress_every > 0 and (i + 1) % progress_every == 0:
                n = len(frame_paths)
                parts = []
                if video_writer is not None:
                    parts.append(f"{video_progress_message} {i + 1}/{n}")
                if export_dir is not None:
                    parts.append(f"Task 2: {i + 1}/{n}")
                if parts:
                    print(" | ".join(parts))

    if export_dir is not None and write_zip and export_zip is not None:
        export_zip = Path(export_zip)
        export_zip.parent.mkdir(parents=True, exist_ok=True)
        if export_zip.exists():
            export_zip.unlink()
        export_dir = Path(export_dir)
        with zipfile.ZipFile(export_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in export_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(export_dir))


def run_task2_feature_zip_export(
    *,
    raw_dir: Path,
    frame_paths: Sequence[Path] | None,
    model: torch.nn.Module,
    model_pred_to_semantic: Callable[[np.ndarray], np.ndarray],
    output_seq_dir: Path,
    output_seq_zip: Path,
    max_frames: int = 1000,
    clean_export: bool = True,
    mask_as_bgr: bool = True,
    write_zip: bool = True,
    progress_every: int = 100,
    export_mask_layers: list[dict] | None = None,
    class_names: list[str] | None = None,
    guideline_anchor_class_id: int | None = None,
) -> None:
    """Notebook-style entry: glob frame_*.jpg under raw_dir, cap count, export + zip."""
    if frame_paths is None:
        _all_exp = sorted(raw_dir.glob("frame_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
        export_paths = _all_exp[:max_frames]
    else:
        export_paths = list(frame_paths)[:max_frames]
    if not export_paths:
        raise SystemExit("No frames to export (frame_*.jpg or empty frame_paths).")
    print(f"Task 2: exporting {len(export_paths)} frames → {output_seq_dir}")
    run_frames_guideline_pipeline(
        export_paths,
        model,
        model_pred_to_semantic,
        video_writer=None,
        export_dir=output_seq_dir,
        export_zip=output_seq_zip,
        clean_export=clean_export,
        write_zip=write_zip,
        mask_as_bgr=mask_as_bgr,
        progress_every=progress_every,
        export_mask_layers=export_mask_layers,
        class_names=class_names,
        guideline_anchor_class_id=guideline_anchor_class_id,
    )
    print("Task 2 done. Folder:", output_seq_dir)
    if write_zip:
        print("ZIP:", output_seq_zip)
