"""
Load DeepLabV3+ checkpoint, segment frames in [start, end], write guideline-overlay MP4.
Optional Task 2 export: frames/, masks/, json/, feature zip (notebook parity).

Run: python overlay_video_segment.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from checkpoint_loading import infer_num_classes, load_checkpoint_state  # noqa: E402
from feature_zip_export import run_frames_guideline_pipeline  # noqa: E402
import guideline_export_config as gcfg  # noqa: E402
from guideline_export_config import (  # noqa: E402
    assert_class_names_match_checkpoint,
    build_model_pred_to_semantic,
    make_overlay_runtime_kwargs,
)
from guideline_overlay_core import set_overlay_runtime  # noqa: E402
from Models.DeepLabV3Plus.modeling import deeplabv3plus_resnet101  # noqa: E402


def _diagnose_non_image_file(path: Path) -> str:
    """When cv2.imread returns None: explain common cases (e.g. HTML saved as .jpg)."""
    try:
        raw = path.read_bytes()[:512]
    except OSError as e:
        return f"read_error:{e!r}"
    if not raw:
        return "empty_file"
    head = raw.lstrip()[:200]
    if head.startswith(b"<") or b"<!DOCTYPE" in raw[:300].upper():
        low = raw.lower()
        if b"cloudflare" in low or b"cf-ray" in low or b"cf-access" in low:
            return "html_cloudflare_access_login"
        return "html_not_jpeg"
    if raw.startswith(b"\xff\xd8"):
        return "jpeg_magic_but_decode_failed"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png_magic_but_decode_failed"
    return "unknown_bytes_not_decoded_as_image"


def frame_sort_key(p: Path) -> int:
    stem = p.stem
    if stem.startswith("frame_"):
        return int(stem[6:], 10)
    return int(stem, 10)


def main() -> None:
    ap = argparse.ArgumentParser(description="Guideline overlay video (+ optional Task 2 export)")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "cardio" / "cardio_run" / "fold1" / "best_fea2.pth",
    )
    ap.add_argument(
        "--frames-dir",
        type=Path,
        default=ROOT / "exports" / "export-4-10" / "extracted" / "images",
        help="Folder with frame_01044.jpg style files",
    )
    ap.add_argument("--frame-start", type=int, default=2704)
    ap.add_argument("--frame-end", type=int, default=3500, help="Inclusive end index")
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "exports" / "export-4-10" / "guideline_overlay.mp4",
    )
    ap.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override head size; default: infer from checkpoint (classifier.classifier.3.weight)",
    )
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--fps", type=float, default=18.0)
    ap.add_argument(
        "--export",
        action="store_true",
        help="Write Task 2 layout under --export-dir and zip to --export-zip",
    )
    ap.add_argument(
        "--export-dir",
        type=Path,
        default=ROOT / "features_data",
    )
    ap.add_argument(
        "--export-zip",
        type=Path,
        default=ROOT / "features_data" / "feature_2_data.zip",
    )
    ap.add_argument(
        "--no-clean-export",
        action="store_true",
        help="Do not shutil.rmtree(export_dir) before writing",
    )
    ap.add_argument(
        "--no-export-zip",
        action="store_true",
        help="Write export folder only (no zip)",
    )
    ap.add_argument(
        "--no-mask-bgr",
        action="store_true",
        help="Write single-channel PNG masks instead of 3-channel BGR",
    )
    ap.add_argument(
        "--semantic-lut",
        type=Path,
        default=None,
        help="Optional .npy uint8 vector of length num_classes mapping logits → semantic ids",
    )
    args = ap.parse_args()

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_paths = [
        p
        for p in args.frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts and p.stem.lower().startswith("frame_")
    ]
    all_paths.sort(key=frame_sort_key)
    frame_paths = [p for p in all_paths if args.frame_start <= frame_sort_key(p) <= args.frame_end]

    if not frame_paths:
        raise SystemExit(
            f"No frames in [{args.frame_start}, {args.frame_end}] under {args.frames_dir}"
        )

    print(f"Frames: {len(frame_paths)} ({frame_paths[0].name} .. {frame_paths[-1].name})")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        diag = _diagnose_non_image_file(frame_paths[0])
        hint = ""
        if diag == "html_cloudflare_access_login":
            hint = (
                "\nThese files look like Cloudflare Access HTML saved as .jpg. "
                "Re-download the export with CF_ACCESS_CLIENT_ID and "
                "CF_ACCESS_CLIENT_SECRET set (see README.md), then re-extract the zip.\n"
            )
        elif diag.startswith("html"):
            hint = (
                "\nFile content is HTML, not image data. Re-export frames from Label Studio "
                "or fix the download so each frame_*.jpg is a real image.\n"
            )
        raise SystemExit(
            f"Cannot read: {frame_paths[0]}\n"
            f"Diagnosis: {diag}{hint}"
        )
    h, w = first.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_overlay_runtime(**make_overlay_runtime_kwargs(device, args.img_size))

    state = load_checkpoint_state(args.checkpoint, device)
    num_model_classes = args.num_classes if args.num_classes is not None else infer_num_classes(state)
    lut = None
    if args.semantic_lut is not None:
        lut = np.load(args.semantic_lut)
        if lut.ndim != 1:
            raise SystemExit("--semantic-lut must be a 1-D .npy array")
    assert_class_names_match_checkpoint(num_model_classes, gcfg.CLASS_NAMES, lut)
    model_pred_to_semantic = build_model_pred_to_semantic(num_model_classes, gcfg.CLASS_NAMES, lut)

    model = deeplabv3plus_resnet101(
        num_classes=num_model_classes, output_stride=8, pretrained_backbone=False
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    print("Loaded:", args.checkpoint, f"(num_classes={num_model_classes})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(args.output), fourcc, args.fps, (w, h))
    if not out.isOpened():
        raise SystemExit("VideoWriter failed; try another --output path or codec.")

    export_dir = Path(args.export_dir) if args.export else None
    export_zip = Path(args.export_zip) if args.export else None

    run_frames_guideline_pipeline(
        frame_paths,
        model,
        model_pred_to_semantic,
        video_writer=out,
        export_dir=export_dir,
        export_zip=export_zip,
        clean_export=not args.no_clean_export,
        write_zip=args.export and not args.no_export_zip,
        mask_as_bgr=not args.no_mask_bgr,
        progress_every=100,
        video_progress_message="Processed",
    )

    out.release()
    print("Done video:", args.output)
    if args.export:
        print("Export dir:", export_dir)
        if not args.no_export_zip:
            print("ZIP:", export_zip)


if __name__ == "__main__":
    main()
