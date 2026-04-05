import json
import re
import urllib.parse
from pathlib import Path


def _looks_like_html_body(b: bytes) -> bool:
    s = b[:512].lstrip().lower()
    return s.startswith(b"<!doctype") or s.startswith(b"<html") or s.startswith(b"<head")


def _is_jpeg_magic(b: bytes) -> bool:
    return len(b) >= 3 and b[0] == 0xFF and b[1] == 0xD8 and b[2] == 0xFF


def _is_png_magic(b: bytes) -> bool:
    return len(b) >= 8 and b[:8] == b"\x89PNG\r\n\x1a\n"


def _is_webp_magic(b: bytes) -> bool:
    return len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP"


def _is_valid_raster_image_payload(b: bytes) -> bool:
    if not b:
        return False
    head = b[: min(512, len(b))]
    if _looks_like_html_body(head):
        return False
    return _is_jpeg_magic(b) or _is_png_magic(b) or _is_webp_magic(b)


def _output_path_for_image_bytes(dest_jpg_path: Path, data: bytes) -> Path:
    if _is_png_magic(data):
        return dest_jpg_path.with_suffix(".png")
    if _is_webp_magic(data):
        return dest_jpg_path.with_suffix(".webp")
    return dest_jpg_path


def _resolve_media_url(raw: str, base_no_slash: str) -> str:
    r = (raw or "").strip()
    if not r:
        return ""
    if r.startswith("http://") or r.startswith("https://"):
        return r
    return urllib.parse.urljoin(base_no_slash + "/", r.lstrip("/"))


def _normalize_ls_upload_path_to_data_upload(abs_url: str) -> str:
    """Exports often reference /upload/<project>/<file>; Django serves under /data/upload/...."""
    pu = urllib.parse.urlsplit(abs_url)
    path = pu.path or ""
    if not path.startswith("/upload/"):
        return abs_url
    rest = path[len("/upload/") :].lstrip("/")
    new_path = "/data/upload/" + rest
    return urllib.parse.urlunsplit((pu.scheme, pu.netloc, new_path, pu.query, pu.fragment))


def _rewrite_data_upload_to_storage_proxy(abs_url: str) -> str:
    """Many LS deployments (nginx / cloud storage) 404 on /data/upload/... but serve the same
    bytes via /storage-data/uploaded/?filepath=upload/<project>/<file> (see LS tools io.py)."""
    if not abs_url:
        return abs_url
    pu = urllib.parse.urlsplit(abs_url)
    path = pu.path or ""
    if path.startswith("/storage-data/uploaded"):
        return abs_url
    if not path.startswith("/data/upload/"):
        return abs_url
    inner = path[len("/data/") :]  # upload/<project>/<filename>
    query = urllib.parse.urlencode({"filepath": inner}, safe="/")
    return urllib.parse.urlunsplit((pu.scheme, pu.netloc, "/storage-data/uploaded/", query, ""))


def _frame_index_from_stem(name: str) -> int | None:
    m = re.search(r"frame_(\d+)", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _tasks_from_json_payload(body: bytes) -> list:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        t = data.get("tasks")
        if isinstance(t, list):
            return [x for x in t if isinstance(x, dict)]
    return []


def _task_image_field(task: dict) -> str | None:
    d = task.get("data") or {}
    for k in ("image", "img", "ocr"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for v in d.values():
        if isinstance(v, str) and v.strip():
            vs = v.strip()
            low = vs.lower()
            if "/upload/" in vs or "frame_" in low or low.endswith((".jpg", ".jpeg", ".png", ".webp")):
                return vs
    return None


def _build_frame_to_image_url(tasks: list, base_no_slash: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for t in tasks:
        raw = _task_image_field(t)
        if not raw:
            continue
        path = urllib.parse.urlparse(raw).path if raw.startswith("http") else raw
        base_name = Path(path).name
        fi = _frame_index_from_stem(base_name)
        if fi is None:
            continue
        full = _resolve_media_url(raw, base_no_slash)
        if full:
            out[fi] = full
    return out
