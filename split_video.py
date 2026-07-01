"""
Split or trim a video into smaller clips of a fixed length.

Uses ffmpeg when available (fast). Falls back to MoviePy if ffmpeg is missing.

Examples:
  python split_video.py 2025-10-24_150300_VID001.mp4 --length 60
  python split_video.py --input video.mp4 --length 5m --output-dir clips
  python split_video.py video.mp4 --length 30s --start 2m --end 10m
  python split_video.py video.mp4 --length 1m --reencode
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "2025-10-24_150300_VID001.mp4"


def parse_duration(text: str) -> float:
    """Parse seconds or compact durations like 30s, 5m, 1h30m into seconds."""
    text = text.strip().lower()
    if not text:
        raise ValueError("Duration must not be empty.")

    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)

    total = 0.0
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", text):
        value = float(amount)
        if unit == "h":
            total += value * 3600
        elif unit == "m":
            total += value * 60
        else:
            total += value

    if total <= 0:
        raise ValueError(f"Could not parse duration: {text!r}")
    return total


def format_timestamp(seconds: float) -> str:
    whole = max(0, int(seconds))
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_duration_ffprobe(video: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def get_duration_moviepy(video: Path) -> float:
    try:
        from moviepy import VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip  # type: ignore[no-redef]

    with VideoFileClip(str(video)) as clip:
        if clip.duration is None or clip.duration <= 0:
            raise ValueError(f"Could not read duration for {video}")
        return float(clip.duration)


def get_duration(video: Path) -> float:
    duration = get_duration_ffprobe(video)
    if duration is not None and duration > 0:
        return duration
    return get_duration_moviepy(video)


def build_output_path(
    output_dir: Path,
    stem: str,
    index: int,
    start_s: float,
    end_s: float,
) -> Path:
    return output_dir / f"{stem}_part{index:03d}_{int(start_s):04d}-{int(end_s):04d}s.mp4"


def split_with_ffmpeg(
    video: Path,
    output_dir: Path,
    segment_length: float,
    start_s: float,
    end_s: float,
    reencode: bool,
) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH.")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index = 1
    cursor = start_s

    while cursor < end_s - 1e-6:
        chunk_end = min(cursor + segment_length, end_s)
        chunk_len = chunk_end - cursor
        out_path = build_output_path(output_dir, video.stem, index, cursor, chunk_end)

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{cursor:.3f}",
            "-i",
            str(video),
            "-t",
            f"{chunk_len:.3f}",
        ]
        if reencode:
            cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac"])
        else:
            cmd.extend(["-c", "copy", "-avoid_negative_ts", "make_zero"])

        cmd.append(str(out_path))
        subprocess.run(cmd, check=True)
        written.append(out_path)
        print(f"  wrote {out_path.name} ({format_timestamp(cursor)} -> {format_timestamp(chunk_end)})")
        cursor = chunk_end
        index += 1

    return written


def split_with_moviepy(
    video: Path,
    output_dir: Path,
    segment_length: float,
    start_s: float,
    end_s: float,
) -> list[Path]:
    try:
        from moviepy import VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip  # type: ignore[no-redef]

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index = 1
    cursor = start_s

    with VideoFileClip(str(video)) as clip:
        while cursor < end_s - 1e-6:
            chunk_end = min(cursor + segment_length, end_s)
            out_path = build_output_path(output_dir, video.stem, index, cursor, chunk_end)
            if hasattr(clip, "subclipped"):
                sub = clip.subclipped(cursor, chunk_end)
            else:
                sub = clip.subclip(cursor, chunk_end)  # moviepy 1.x
            sub.write_videofile(
                str(out_path),
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
            sub.close()
            written.append(out_path)
            print(f"  wrote {out_path.name} ({format_timestamp(cursor)} -> {format_timestamp(chunk_end)})")
            cursor = chunk_end
            index += 1

    return written


def split_video(
    video: Path,
    segment_length: float,
    output_dir: Path | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    reencode: bool = False,
) -> list[Path]:
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if segment_length <= 0:
        raise ValueError("Segment length must be positive.")

    duration = get_duration(video)
    start = 0.0 if start_s is None else max(0.0, start_s)
    end = duration if end_s is None else min(duration, end_s)
    if start >= end:
        raise ValueError(f"Invalid range: start={start:.2f}s, end={end:.2f}s, duration={duration:.2f}s")

    out_dir = output_dir or video.with_name(f"{video.stem}_clips")
    print(
        f"Splitting {video.name}\n"
        f"  source duration: {duration:.2f}s ({format_timestamp(duration)})\n"
        f"  range: {format_timestamp(start)} -> {format_timestamp(end)}\n"
        f"  segment length: {segment_length:.2f}s\n"
        f"  output dir: {out_dir}"
    )

    if shutil.which("ffmpeg"):
        mode = "re-encode (exact cuts)" if reencode else "stream copy (fast, keyframe-aligned)"
        print(f"  mode: ffmpeg, {mode}")
        return split_with_ffmpeg(video, out_dir, segment_length, start, end, reencode)

    print("  mode: moviepy fallback (ffmpeg not found; this is slower)")
    return split_with_moviepy(video, out_dir, segment_length, start, end)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Split a video into smaller clips of a fixed length.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Duration formats:
  60       60 seconds
  30s      30 seconds
  5m       5 minutes
  1h30m    1 hour 30 minutes
""",
    )
    p.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input video (default: {DEFAULT_INPUT.name})",
    )
    p.add_argument(
        "-l",
        "--length",
        required=True,
        help="Clip length in seconds or compact form (e.g. 60, 30s, 5m)",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output clips (default: <video_stem>_clips next to input)",
    )
    p.add_argument(
        "--start",
        default=None,
        help="Start time before splitting (seconds or 2m, 1h30m, ...)",
    )
    p.add_argument(
        "--end",
        default=None,
        help="End time before splitting (seconds or 2m, 1h30m, ...)",
    )
    p.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode with ffmpeg for exact cut points (slower, larger files)",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    try:
        segment_length = parse_duration(args.length)
        start_s = None if args.start is None else parse_duration(args.start)
        end_s = None if args.end is None else parse_duration(args.end)
        written = split_video(
            args.input,
            segment_length,
            output_dir=args.output_dir,
            start_s=start_s,
            end_s=end_s,
            reencode=args.reencode,
        )
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Done: {len(written)} clip(s) written.")


if __name__ == "__main__":
    main()
