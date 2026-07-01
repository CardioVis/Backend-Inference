"""
Load DeepLabV3+ checkpoint (cardio_p28, 17 classes), segment frames, write overlay MP4.

Đọc / trích frame từ video bằng MoviePy; ghi MP4 overlay bằng OpenCV VideoWriter.
Nguồn ảnh: --video (MP4) hoặc --frames-dir (frame_#####.jpg).
Run: python overlay_video_segment.py
"""
from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from Models.DeepLabV3Plus.modeling import deeplabv3plus_resnet101

# Khớp TAG_TO_CLASS trong prepare_cardio_data_from_labelstudio_json.py (id 0..16)
CLASS_NAMES = [
    "Background",
    "Phrenic nerve",
    "Pericardium",
    "Epicardial adipose tissue",
    "Epicardial fat on aortic",
    "Aortic root",
    "Auricles",
    "Pericardium boundary",
    "Needle holders",
    "Cardioplegia cannula",
    "Surgical pledget",
    "Electrocautery",
    "Chitwood aortic cross-clamp",
    "Left suction tube",
    "Snare tubing",
    "Right ventricle",
    "Long forceps",
]

# BGR, gần màu Label Studio project B
CLASS_COLORS_BGR = [
    (0, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (0, 128, 255),
    (44, 226, 214),
    (71, 232, 17),
    (255, 71, 206),
    (255, 158, 165),
    (255, 255, 170),
    (255, 158, 234),
    (143, 153, 0),
    (0, 139, 173),
    (255, 100, 96),
    (13, 56, 212),
    (105, 192, 255),
    (83, 91, 242),
    (96, 97, 102),
]

NUM_CLASSES = len(CLASS_NAMES)


def frame_sort_key(p: Path) -> int:
    stem = p.stem
    if stem.startswith("frame_"):
        return int(stem[6:], 10)
    return int(stem, 10)


def preprocess_frame(frame: np.ndarray, device: torch.device, img_size: int):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (img_size, img_size))
    img = np.clip(img, 0, 255).astype(np.float32) / 255.0
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device)
    return t, (h, w)


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    num_classes: int,
    alpha: float = 0.5,
) -> np.ndarray:
    overlay = frame.copy()
    mask_resized = cv2.resize(
        mask.astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    for c in range(1, num_classes):
        color = CLASS_COLORS_BGR[c] if c < len(CLASS_COLORS_BGR) else (200, 200, 200)
        m = mask_resized == c
        if not np.any(m):
            continue
        overlay[m] = (
            (1 - alpha) * overlay[m].astype(np.float32) + alpha * np.array(color, dtype=np.float32)
        ).astype(np.uint8)

    line_h = 18
    box_x, box_y = 10, 10
    box_w = 440
    n_legend = num_classes - 1
    box_h = 8 + n_legend * line_h + 10
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (40, 40, 40), -1)
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (200, 200, 200), 1)
    for c in range(1, num_classes):
        y = box_y + 8 + (c - 1) * line_h + 14
        cv2.rectangle(overlay, (box_x + 8, y - 12), (box_x + 26, y + 2), CLASS_COLORS_BGR[c], -1)
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
        cv2.putText(
            overlay, name, (box_x + 32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
    return overlay.astype(np.uint8)


def load_model(checkpoint: Path, num_classes: int, device: torch.device):
    model = deeplabv3plus_resnet101(
        num_classes=num_classes, output_stride=8, pretrained_backbone=False
    )
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def rgb_to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    """MoviePy trả RGB uint8; OpenCV dùng BGR."""
    if frame_rgb.ndim == 2:
        return cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


@contextmanager
def open_video_clip(video_path: Path):
    clip = VideoFileClip(str(video_path))
    try:
        yield clip
    finally:
        clip.close()


def video_frame_count(clip: VideoFileClip) -> int:
    n = getattr(clip.reader, "nframes", None)
    if n is not None and int(n) > 0:
        return int(n)
    if clip.fps and clip.duration:
        return max(1, int(round(clip.fps * clip.duration)))
    return 1


def read_video_frame_bgr(clip: VideoFileClip, frame_idx: int) -> np.ndarray:
    """Đọc một frame theo chỉ số (0-based) qua MoviePy get_frame."""
    if clip.fps is None or clip.fps <= 0:
        raise ValueError("Video FPS không hợp lệ.")
    t = frame_idx / float(clip.fps)
    if clip.duration is not None:
        t = min(max(0.0, t), max(0.0, clip.duration - 1e-6))
    return rgb_to_bgr(clip.get_frame(t))


def iter_video_frames(
    video_path: Path,
    frame_start: int | None,
    frame_end: int | None,
):
    """Yield (frame_idx, frame_bgr) từ video bằng MoviePy."""
    with open_video_clip(video_path) as clip:
        total = video_frame_count(clip)
        start = frame_start if frame_start is not None else 0
        end = frame_end if frame_end is not None else max(0, total - 1)
        end = min(end, total - 1)
        if start > end:
            return
        for idx in range(start, end + 1):
            yield idx, read_video_frame_bgr(clip, idx)


def extract_video_frames(
    video_path: Path,
    out_dir: Path,
    frame_start: int | None = None,
    frame_end: int | None = None,
    prefix: str = "frame_",
    digits: int = 5,
) -> int:
    """Trích frame từ video ra JPG (frame_00000.jpg, ...). Trả về số frame đã lưu."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    with open_video_clip(video_path) as clip:
        total = video_frame_count(clip)
        start = frame_start if frame_start is not None else 0
        end = frame_end if frame_end is not None else max(0, total - 1)
        end = min(end, total - 1)
        print(f"Extract {video_path.name} -> {out_dir} | frames {start}..{end}")
        for idx in range(start, end + 1):
            frame = read_video_frame_bgr(clip, idx)
            name = f"{prefix}{idx:0{digits}d}.jpg"
            cv2.imwrite(str(out_dir / name), frame)
            n_saved += 1
            if n_saved % 100 == 0:
                print(f"  saved {n_saved}")
    print(f"Extracted {n_saved} frames -> {out_dir}")
    return n_saved


def collect_frame_paths(frames_dir: Path, frame_start: int | None, frame_end: int | None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_paths = [
        p
        for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts and p.stem.lower().startswith("frame_")
    ]
    all_paths.sort(key=frame_sort_key)
    if frame_start is None and frame_end is None:
        return all_paths
    lo = frame_start if frame_start is not None else frame_sort_key(all_paths[0])
    hi = frame_end if frame_end is not None else frame_sort_key(all_paths[-1])
    return [p for p in all_paths if lo <= frame_sort_key(p) <= hi]


def main():
    ap = argparse.ArgumentParser(description="Segmentation overlay video (MP4 or frame folder)")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        # default=ROOT / "checkpoints" / "cardio_p28" / "tmp" / "fold1" / "best.pth",
        default = "checkpoints/best.pth"
    )
    ap.add_argument(
        "--video",
        type=Path,
        # default=ROOT / "2025-04-01_163954_VID001.mp4",
        default = "overlay_vid1_patient10_10fps.mp4",
        help="Input MP4. Bỏ qua nếu dùng --frames-dir-only.",
    )
    ap.add_argument(
        "--frames-dir-only",
        action="store_true",
        help="Dùng thư mục frame thay vì --video.",
    )
    ap.add_argument(
        "--frames-dir",
        type=Path,
        default=ROOT / "frames_for_label_vid1_patient10",
        help="Folder with frame_#####.jpg (khi --frames-dir-only).",
    )
    ap.add_argument(
        "--frame-start",
        type=int,
        default=None,
        help="Chỉ số frame bắt đầu (trong video hoặc số frame_#####). None = từ đầu.",
    )
    ap.add_argument(
        "--frame-end",
        type=int,
        default=None,
        help="Chỉ số frame kết thúc (inclusive). None = đến hết.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4. Mặc định: overlay_<tên_video>.mp4 cạnh file input.",
    )
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--fps", type=float, default=None, help="FPS output; mặc định = FPS video hoặc 18.")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument(
        "--extract-dir",
        type=Path,
        default=None,
        help="Chỉ trích frame JPG từ --video ra thư mục này (không chạy model).",
    )
    ap.add_argument(
        "--extract-only",
        action="store_true",
        help="Chỉ extract frame (--extract-dir bắt buộc), không overlay.",
    )
    args = ap.parse_args()

    if args.extract_only:
        if args.extract_dir is None:
            raise SystemExit("Cần --extract-dir khi dùng --extract-only.")
        if not args.video.is_file():
            raise SystemExit(f"Video không tồn tại: {args.video}")
        extract_video_frames(
            args.video, args.extract_dir, args.frame_start, args.frame_end
        )
        return

    if args.num_classes != NUM_CLASSES:
        print(f"Cảnh báo: --num-classes={args.num_classes}, CLASS_NAMES có {NUM_CLASSES} phần tử.")

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint không tồn tại: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args.num_classes, device)
    print("Loaded:", args.checkpoint, "| device:", device)

    use_video = not args.frames_dir_only and args.video is not None and args.video.is_file()
    out_fps = args.fps or 18.0

    if use_video:
        if args.output is None:
            args.output = args.video.with_name(f"overlay_{args.video.stem}.mp4")
        with open_video_clip(args.video) as clip:
            if args.fps is None:
                out_fps = float(clip.fps) if clip.fps else 18.0
            total = video_frame_count(clip)
            start = args.frame_start if args.frame_start is not None else 0
            end = args.frame_end if args.frame_end is not None else max(0, total - 1)
            end = min(end, total - 1)
            print(
                f"Video (MoviePy): {args.video.name} | frames {start}..{end} "
                f"(total {total}) | fps {out_fps:.2f}"
            )
            first = read_video_frame_bgr(clip, start)

        h, w = first.shape[:2]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(args.output), fourcc, out_fps, (w, h))
        if not out.isOpened():
            raise SystemExit("VideoWriter failed.")

        n = 0
        with torch.no_grad():
            for i, frame in iter_video_frames(args.video, args.frame_start, args.frame_end):
                img_t, _ = preprocess_frame(frame, device, args.img_size)
                logits = model(img_t)
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
                vis = overlay_mask(frame, pred, args.num_classes, alpha=args.alpha)
                out.write(vis)
                n += 1
                if n % 100 == 0:
                    print(f"  frame {i} ({n} written)")
        out.release()
        print(f"Done: {args.output} ({n} frames)")
        return

    # --- frame folder mode ---
    frame_paths = collect_frame_paths(args.frames_dir, args.frame_start, args.frame_end)
    if not frame_paths:
        raise SystemExit(f"No frames under {args.frames_dir}")

    if args.output is None:
        tag = f"{args.frame_start}_{args.frame_end}" if args.frame_start is not None else "all"
        args.output = ROOT / f"overlay_p28_{tag}.mp4"

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise SystemExit(f"Cannot read: {frame_paths[0]}")
    h, w = first.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(args.output), fourcc, out_fps, (w, h))
    if not out.isOpened():
        raise SystemExit("VideoWriter failed.")

    print(f"Frames: {len(frame_paths)} ({frame_paths[0].name} .. {frame_paths[-1].name})")
    with torch.no_grad():
        for i, path in enumerate(frame_paths):
            frame = cv2.imread(str(path))
            if frame is None:
                print("Skip (read error):", path.name)
                continue
            img_t, _ = preprocess_frame(frame, device, args.img_size)
            logits = model(img_t)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            vis = overlay_mask(frame, pred, args.num_classes, alpha=args.alpha)
            out.write(vis)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(frame_paths)}")
    out.release()
    print("Done:", args.output)


if __name__ == "__main__":
    main()
