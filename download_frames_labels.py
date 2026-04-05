import json
import os
import re
import shutil
import sys
import tempfile
import time
import types
import urllib.parse
import zipfile
from pathlib import Path

import numpy as np
import requests
from label_studio_sdk import LabelStudio
from label_studio_sdk.core.api_error import ApiError
from label_studio_sdk.types.lse_task_filter_options_request import (
    LseTaskFilterOptionsRequest,
)
from label_studio_sdk.types.serialization_option_request import SerializationOptionRequest
from label_studio_sdk.types.serialization_options_request import SerializationOptionsRequest
from label_studio_sdk.types.token_refresh_response import TokenRefreshResponse


def _debug_log_path() -> Path:
    """Prefer repo .cursor/; fall back to system temp if that dir is not writable."""
    primary = Path(__file__).resolve().parent / ".cursor" / "debug-988c41.log"
    try:
        primary.parent.mkdir(parents=True, exist_ok=True)
        return primary
    except OSError:
        return Path(tempfile.gettempdir()) / "cardiovis-debug-988c41.ndjson"


# #region agent log
def _agent_log(message: str, data: dict, hypothesis_id: str) -> None:
    line = json.dumps(
        {
            "sessionId": "988c41",
            "timestamp": int(time.time() * 1000),
            "location": "download_frames_labels.py",
            "message": message,
            "data": data,
            "hypothesisId": hypothesis_id,
        },
        ensure_ascii=False,
    )
    try:
        with open(_debug_log_path(), "a", encoding="utf-8") as _f:
            _f.write(line + "\n")
    except OSError:
        pass


# #endregion


def _patch_token_refresh_to_include_cf_access(ls: LabelStudio, base_url: str, cf: dict) -> None:
    """label_studio_sdk TokensClientExt.refresh() POSTs via httpx with only Content-Type, dropping
    custom headers. Behind Cloudflare Access, /api/token/refresh then returns HTML → JSONDecodeError.
    """
    _tc = ls._client_wrapper._tokens_client

    def _refresh_cf(self) -> TokenRefreshResponse:
        _url = f"{base_url.rstrip('/')}/api/token/refresh/"
        _h = {"Content-Type": "application/json", **cf}
        _r = requests.post(
            _url,
            json={"refresh": self._api_key},
            headers=_h,
            timeout=120,
        )
        _agent_log(
            "token_refresh_response",
            {
                "status_code": _r.status_code,
                "content_type": _r.headers.get("Content-Type", ""),
                "body_len": len(_r.text or ""),
                "body_preview": (_r.text or "")[:400].replace("\n", "\\n"),
            },
            "H-refresh-CF",
        )
        if _r.status_code != 200:
            raise ApiError(status_code=_r.status_code, body=_r.text[:2000])
        try:
            _data = _r.json()
        except json.JSONDecodeError:
            raise ApiError(
                status_code=_r.status_code,
                body=_r.text[:2000] or "empty body (often Cloudflare HTML without CF-Access headers on refresh)",
            ) from None
        try:
            return TokenRefreshResponse.model_validate(_data)
        except AttributeError:
            return TokenRefreshResponse.parse_obj(_data)

    _tc.refresh = types.MethodType(_refresh_cf, _tc)


def _log_api_error(name: str, err: ApiError, hypothesis_id: str) -> None:
    body = err.body
    preview = ""
    if isinstance(body, (dict, list)):
        preview = json.dumps(body)[:800]
    elif body is not None:
        preview = str(body)[:800]
    _agent_log(
        name,
        {"status_code": err.status_code, "body_preview": preview.replace("\n", "\\n")},
        hypothesis_id,
    )


def _export_task_filter_options() -> LseTaskFilterOptionsRequest:
    """Default: export all tasks (includes unlabeled / not finished). Optional Data Manager view.
    Set LABEL_STUDIO_EXPORT_ANNOTATED_ONLY=1 and/or LABEL_STUDIO_EXPORT_FINISHED_ONLY=1 to narrow."""
    kwargs: dict = {}
    if os.environ.get("LABEL_STUDIO_EXPORT_FINISHED_ONLY", "").strip().lower() in ("1", "true", "yes"):
        kwargs["finished"] = "only"
    if os.environ.get("LABEL_STUDIO_EXPORT_ANNOTATED_ONLY", "").strip().lower() in ("1", "true", "yes"):
        kwargs["annotated"] = "only"
    raw = os.environ.get("LABEL_STUDIO_EXPORT_VIEW_ID", "").strip()
    if raw and raw.lower() not in ("none", "false", "0"):
        try:
            kwargs["view"] = int(raw)
        except ValueError:
            kwargs["view"] = 1
    return LseTaskFilterOptionsRequest(**kwargs)


def _export_serialization_options() -> SerializationOptionsRequest:
    """Keep drafts/predictions compact; do not use only_id on annotations__completed_by — that can
    strip nested annotation fields and yield empty YOLO label files from the server converter."""
    return SerializationOptionsRequest(
        drafts=SerializationOptionRequest(only_id=True),
        predictions=SerializationOptionRequest(only_id=True),
        interpolate_key_frames=False,
    )


def _wait_for_converted_format(
    export_pk: int,
    project_id: int,
    export_type: str,
) -> None:
    """POST /exports/.../convert is async; download must run after ConvertedFormat is completed."""
    _deadline = time.time() + float(os.environ.get("LABEL_STUDIO_CONVERT_TIMEOUT_SEC", "3600"))
    while time.time() < _deadline:
        try:
            _exp = ls.projects.exports.get(project_id, export_pk)
        except ApiError as _e:
            _log_api_error("convert_poll_get_error", _e, "H-convert-poll")
            raise
        _match = None
        for _c in _exp.converted_formats or []:
            if (_c.export_type or "").upper() == export_type.upper():
                _match = _c
                break
        _agent_log(
            "convert_poll",
            {
                "export_type": export_type,
                "found": _match is not None,
                "status": getattr(_match, "status", None),
            },
            "H-convert-async",
        )
        if _match is None:
            time.sleep(1.0)
            continue
        if _match.status == "failed":
            print(
                f"Export conversion to {export_type} failed. "
                f"Traceback (if any): {getattr(_match, 'traceback', None)}",
                file=sys.stderr,
            )
            sys.exit(1)
        if _match.status == "completed":
            return
        time.sleep(1.0)
    print(
        f"Timed out waiting for {export_type} conversion (>{os.environ.get('LABEL_STUDIO_CONVERT_TIMEOUT_SEC', '3600')}s).",
        file=sys.stderr,
    )
    sys.exit(1)


def _ls_bearer_headers(ls: LabelStudio, cf: dict) -> dict:
    tok = getattr(ls._client_wrapper._tokens_client, "api_key", "") or ""
    h = {"Authorization": f"Bearer {tok}"}
    h.update(cf or {})
    return h


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


def _ensure_json_export_ready(ls: LabelStudio, project_id: int, export_id: int) -> None:
    try:
        _exp = ls.projects.exports.get(project_id, export_id)
    except ApiError as _e:
        _log_api_error("repair_json_get_export", _e, "H-repair")
        raise
    for _c in _exp.converted_formats or []:
        if (_c.export_type or "").upper() == "JSON" and _c.status == "completed":
            _agent_log("json_export_already_ready", {"export_id": export_id}, "H-repair")
            return
    try:
        ls.projects.exports.convert(
            id=project_id,
            export_pk=export_id,
            export_type="JSON",
            download_resources=False,
        )
    except ApiError as _e:
        _prev = ""
        if isinstance(_e.body, str):
            _prev = _e.body[:400]
        elif _e.body is not None:
            _prev = json.dumps(_e.body)[:400]
        _agent_log(
            "json_convert_attempt",
            {"status": _e.status_code, "preview": _prev.replace(chr(10), " ")},
            "H-repair",
        )
    _wait_for_converted_format(export_id, project_id, "JSON")


def _download_json_export_bytes(ls: LabelStudio, project_id: int, export_id: int) -> bytes:
    return b"".join(ls.projects.exports.download(project_id, export_id, export_type="JSON"))


def _access_bit(data: bytes, num: int) -> int:
    base = num // 8
    shift = 7 - (num % 8)
    return (data[base] & (1 << shift)) >> shift


def _bytes_to_bit_string(data: bytes) -> str:
    return "".join(str(_access_bit(data, i)) for i in range(len(data) * 8))


class _LsRleBitReader:
    """Bit stream reader for Label Studio brush RLE (label-studio-converter/brush.py)."""

    def __init__(self, bits: str) -> None:
        self._bits = bits
        self._i = 0

    def read(self, nbits: int) -> int:
        chunk = self._bits[self._i : self._i + nbits]
        self._i += nbits
        return int(chunk, 2) if chunk else 0


def _decode_ls_brush_rle(rle: list | bytes | tuple) -> np.ndarray:
    if isinstance(rle, str):
        raise TypeError("rle string not supported")
    raw = bytes(int(x) & 0xFF for x in rle) if not isinstance(rle, bytes) else rle
    inp = _LsRleBitReader(_bytes_to_bit_string(raw))
    num = inp.read(32)
    word_size = inp.read(5) + 1
    rle_sizes = [inp.read(4) + 1 for _ in range(4)]
    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = inp.read(1)
        j = i + 1 + inp.read(rle_sizes[inp.read(2)])
        if x:
            val = inp.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = inp.read(word_size)
                out[i] = val
                i += 1
    return out


def _brush_result_to_percent_tlwh(result: dict) -> tuple[float, float, float, float] | None:
    val = result.get("value") or {}
    rle = val.get("rle")
    if rle is None:
        return None
    ow = int(result.get("original_width") or val.get("original_width") or 0)
    oh = int(result.get("original_height") or val.get("original_height") or 0)
    if ow <= 0 or oh <= 0:
        return None
    try:
        flat = _decode_ls_brush_rle(rle)
        arr = np.reshape(flat, (oh, ow, 4))
        alpha = arr[:, :, 3]
        ys, xs = np.where(alpha > 0)
        if ys.size == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        xp = 100.0 * x0 / ow
        yp = 100.0 * y0 / oh
        wp = 100.0 * (x1 - x0 + 1) / ow
        hp = 100.0 * (y1 - y0 + 1) / oh
        return xp, yp, wp, hp
    except Exception as e:
        _agent_log("brush_rle_bbox_fail", {"err": str(e)[:240]}, "H-yolo-labels")
        return None


def _polygon_value_to_percent_tlwh(value: dict) -> tuple[float, float, float, float] | None:
    pts_raw = value.get("points")
    xs: list[float] = []
    ys: list[float] = []
    if isinstance(pts_raw, str):
        for seg in pts_raw.split(";"):
            seg = seg.strip()
            if "," not in seg:
                continue
            a, b = seg.split(",", 1)
            try:
                xs.append(float(a))
                ys.append(float(b))
            except ValueError:
                continue
    elif isinstance(pts_raw, list):
        for pt in pts_raw:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
            elif isinstance(pt, dict):
                xs.append(float(pt.get("x", 0)))
                ys.append(float(pt.get("y", 0)))
    if len(xs) < 1:
        return None
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w, h = maxx - minx, maxy - miny
    if w <= 0 or h <= 0:
        return None
    return minx, miny, w, h


def _task_frame_index(task: dict) -> int | None:
    raw = _task_image_field(task)
    if not raw:
        return None
    path = urllib.parse.urlparse(raw).path if raw.startswith("http") else raw
    return _frame_index_from_stem(Path(path).name)


def _rectangle_class_names_in_value(value: dict) -> list[str]:
    names: list[str] = []
    rl = value.get("rectanglelabels")
    if isinstance(rl, list):
        names.extend(str(x) for x in rl if x is not None)
    lb = value.get("labels")
    if isinstance(lb, list):
        names.extend(str(x) for x in lb if x is not None)
    return names


def _all_yolo_class_names_from_tasks(tasks: list) -> set[str]:
    found: set[str] = set()
    for t in tasks:
        for ann in t.get("annotations") or []:
            if ann.get("was_cancelled"):
                continue
            for r in ann.get("result") or []:
                rt = (r.get("type") or "").lower()
                val = r.get("value") or {}
                if rt in ("rectanglelabels", "rectangle"):
                    found.update(_rectangle_class_names_in_value(val))
                elif rt == "brushlabels":
                    bl = val.get("brushlabels")
                    if isinstance(bl, list):
                        found.update(str(x) for x in bl if x is not None)
                elif rt == "polygonlabels":
                    pl = val.get("polygonlabels")
                    if isinstance(pl, list):
                        found.update(str(x) for x in pl if x is not None)
    return found


def _yaml_parse_names_to_map(content: str) -> dict[str, int]:
    """Best-effort parse of data.yaml / dataset.yaml `names` without PyYAML."""
    name_to_id: dict[str, int] = {}
    m = re.search(r"names:\s*\[([^\]]*)\]", content, re.DOTALL)
    if m:
        idx = 0
        for a, b in re.findall(r"'([^']*)'|\"([^\"]*)\"", m.group(1)):
            name = (a or b).strip()
            if name:
                name_to_id[name] = idx
                idx += 1
        if name_to_id:
            return name_to_id
    sec = re.search(r"names:\s*\n((?:\s*\d+\s*:[^\n]+\n?)+)", content)
    if sec:
        for line in sec.group(1).splitlines():
            mm = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", line.strip())
            if mm:
                name_to_id[mm.group(2).strip().strip("'\"")] = int(mm.group(1))
        if name_to_id:
            return name_to_id
    sec2 = re.search(r"names:\s*\n((?:\s*-\s+[^\n]+\n?)+)", content)
    if sec2:
        idx = 0
        for line in sec2.group(1).splitlines():
            mm = re.match(r"\s*-\s*(.+)\s*$", line)
            if mm:
                name_to_id[mm.group(1).strip().strip("'\"")] = idx
                idx += 1
    return name_to_id


def _find_data_yaml_in_tree(root: Path) -> Path | None:
    for name in ("data.yaml", "dataset.yaml"):
        p = root / name
        if p.is_file():
            return p
    for p in root.rglob("*.yaml"):
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if re.search(r"(?m)^names:\s*(\[|\n)", t):
            return p
    return None


def _build_name_to_id_map(yaml_text: str | None, tasks: list) -> dict[str, int]:
    from_yaml: dict[str, int] = {}
    if yaml_text:
        from_yaml = _yaml_parse_names_to_map(yaml_text)
    ann_names = _all_yolo_class_names_from_tasks(tasks)
    out = dict(from_yaml)
    next_id = max(out.values(), default=-1) + 1
    for n in sorted(ann_names):
        if n not in out:
            out[n] = next_id
            next_id += 1
    return out


def _percent_tlwh_to_yolo_line(x: float, y: float, w: float, h: float, class_id: int) -> str | None:
    if w <= 0 or h <= 0:
        return None
    cx = (x + w / 2.0) / 100.0
    cy = (y + h / 2.0) / 100.0
    nw = w / 100.0
    nh = h / 100.0
    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    nw = min(1.0, max(0.0, nw))
    nh = min(1.0, max(0.0, nh))
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def _ls_percent_rect_to_yolo_line(value: dict, class_id: int) -> str | None:
    return _percent_tlwh_to_yolo_line(
        float(value.get("x", 0)),
        float(value.get("y", 0)),
        float(value.get("width", 0)),
        float(value.get("height", 0)),
        class_id,
    )


def _yolo_lines_for_task(task: dict, name_to_id: dict[str, int]) -> list[str]:
    lines: list[str] = []
    for ann in task.get("annotations") or []:
        if ann.get("was_cancelled"):
            continue
        for r in ann.get("result") or []:
            rt = (r.get("type") or "").lower()
            val = r.get("value") or {}
            if rt in ("rectanglelabels", "rectangle"):
                rnames = _rectangle_class_names_in_value(val)
                if not rnames:
                    continue
                cls_name = rnames[0]
                cid = name_to_id.get(cls_name)
                if cid is None:
                    continue
                line = _ls_percent_rect_to_yolo_line(val, cid)
                if line:
                    lines.append(line)
            elif rt == "brushlabels":
                bl = val.get("brushlabels")
                if not isinstance(bl, list) or not bl:
                    continue
                cls_name = str(bl[0])
                cid = name_to_id.get(cls_name)
                if cid is None:
                    continue
                tlwh = _brush_result_to_percent_tlwh(r)
                if tlwh is None:
                    continue
                line = _percent_tlwh_to_yolo_line(*tlwh, cid)
                if line:
                    lines.append(line)
            elif rt == "polygonlabels":
                pl = val.get("polygonlabels")
                if not isinstance(pl, list) or not pl:
                    continue
                cls_name = str(pl[0])
                cid = name_to_id.get(cls_name)
                if cid is None:
                    continue
                tlwh = _polygon_value_to_percent_tlwh(val)
                if tlwh is None:
                    continue
                line = _percent_tlwh_to_yolo_line(*tlwh, cid)
                if line:
                    lines.append(line)
    return lines


def _log_label_patch_annotation_sample(tasks: list) -> None:
    types: dict[str, int] = {}
    for t in tasks[:3]:
        for ann in t.get("annotations") or []:
            for r in ann.get("result") or []:
                k = str(r.get("type") or "?")
                types[k] = types.get(k, 0) + 1
    _agent_log(
        "label_patch_result_types_sample",
        {"first_tasks_types": types, "task_count": len(tasks)},
        "H-yolo-labels",
    )


def _atomic_rezip_directory(src_dir: Path, dest_zip: Path) -> None:
    tmp = dest_zip.with_suffix(dest_zip.suffix + ".tmp")
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for fp in src_dir.rglob("*"):
            if fp.is_file():
                zout.write(fp, fp.relative_to(src_dir))
    tmp.replace(dest_zip)


def _ensure_empty_label_files_for_all_images(unpack: Path) -> None:
    """Create labels/ (if needed) and labels/<stem>.txt for every images/<stem>.* (unlabeled tasks)."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    for img_root, lbl_root in (
        (unpack / "images", unpack / "labels"),
        (unpack / "images" / "train", unpack / "labels" / "train"),
        (unpack / "images" / "val", unpack / "labels" / "val"),
    ):
        if not img_root.is_dir():
            continue
        lbl_root.mkdir(parents=True, exist_ok=True)
        for imgp in img_root.iterdir():
            if not imgp.is_file() or imgp.suffix.lower() not in exts:
                continue
            tx = lbl_root / f"{imgp.stem}.txt"
            if not tx.exists():
                tx.write_text("", encoding="utf-8")


def _patch_yolo_zip_labels_from_json(
    ls: LabelStudio,
    project_id: int,
    export_id: int,
    zip_path: Path,
    base_no_slash: str,
) -> None:
    """Overwrite labels/**/*.txt from JSON (rectangles, brush masks → AABB, polygons → AABB)."""
    unpack = zip_path.parent / f"{zip_path.stem}_labelpatch_tmp"
    if unpack.exists():
        shutil.rmtree(unpack, ignore_errors=True)
    unpack.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(unpack)
        _ensure_empty_label_files_for_all_images(unpack)
        label_txts = [p for p in unpack.rglob("*.txt") if "labels" in p.parts]
        if not label_txts:
            _agent_log("label_patch_skip", {"reason": "no_txt_under_labels_after_image_sync"}, "H-yolo-labels")
            _atomic_rezip_directory(unpack, zip_path)
            return
        _ensure_json_export_ready(ls, project_id, export_id)
        jbody = _download_json_export_bytes(ls, project_id, export_id)
        tasks = _tasks_from_json_payload(jbody)
        _log_label_patch_annotation_sample(tasks)
        yaml_path = _find_data_yaml_in_tree(unpack)
        yaml_text = yaml_path.read_text(encoding="utf-8", errors="replace") if yaml_path else None
        name_to_id = _build_name_to_id_map(yaml_text, tasks)
        frame_to_lines: dict[int, list[str]] = {}
        for task in tasks:
            fi = _task_frame_index(task)
            if fi is None:
                continue
            for line in _yolo_lines_for_task(task, name_to_id):
                frame_to_lines.setdefault(fi, []).append(line)
        written = 0
        nonempty = 0
        for txt_path in label_txts:
            fi = _frame_index_from_stem(txt_path.name)
            if fi is None:
                continue
            lines = frame_to_lines.get(fi, [])
            body = "\n".join(lines)
            if lines:
                nonempty += 1
            txt_path.write_text(body + ("\n" if body else ""), encoding="utf-8")
            written += 1
        _total_lines = sum(len(v) for v in frame_to_lines.values())
        _agent_log(
            "label_patch_done",
            {
                "label_files_touched": written,
                "nonempty_label_files": nonempty,
                "frames_with_boxes": len(frame_to_lines),
                "total_yolo_lines": _total_lines,
                "class_count": len(name_to_id),
            },
            "H-yolo-labels",
        )
        if nonempty == 0 and tasks:
            print(
                "Warning: label patch produced no non-empty YOLO lines (check brush RLE decode logs "
                "or use rectangle/polygon/brush tools supported by the patch).\n",
                file=sys.stderr,
            )
        _atomic_rezip_directory(unpack, zip_path)
    finally:
        shutil.rmtree(unpack, ignore_errors=True)


def _repair_yolo_zip_with_cf_auth(
    ls: LabelStudio,
    project_id: int,
    export_id: int,
    zip_path: Path,
    base_no_slash: str,
    cf: dict,
) -> Path | None:
    """Rebuild a sibling *_repaired.zip when images/ contains HTML placeholders or empty files."""
    unpack = zip_path.parent / f"{zip_path.stem}_unpack_tmp"
    if unpack.exists():
        shutil.rmtree(unpack, ignore_errors=True)
    unpack.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(unpack)
        images_dir = unpack / "images"
        if not images_dir.is_dir():
            _agent_log("repair_skip_no_images_dir", {"unpack": str(unpack)}, "H-repair")
            print(
                "Image repair skipped: export zip has no images/ directory. "
                "Use YOLO_WITH_IMAGES (default) or set LABEL_STUDIO_DOWNLOAD_RESOURCES=1 if you need "
                "server-embedded files (then rely on *_repaired.zip for real image bytes).",
                file=sys.stderr,
            )
            return None
        bad_files: list[Path] = []
        for p in sorted(images_dir.iterdir()):
            if not p.is_file():
                continue
            try:
                b = p.read_bytes()[:4096]
            except OSError:
                continue
            if _is_valid_raster_image_payload(b):
                continue
            if len(b) == 0:
                bad_files.append(p)
                continue
            bad_files.append(p)
        _agent_log(
            "repair_scan",
            {
                "bad_count": len(bad_files),
                "sample_bad": [x.name for x in bad_files[:5]],
            },
            "H-repair",
        )
        if not bad_files:
            return None
        _ensure_json_export_ready(ls, project_id, export_id)
        jbody = _download_json_export_bytes(ls, project_id, export_id)
        tasks = _tasks_from_json_payload(jbody)
        frame_map = _build_frame_to_image_url(tasks, base_no_slash)
        _agent_log(
            "repair_json_stats",
            {"tasks": len(tasks), "frame_keys": len(frame_map), "json_bytes": len(jbody)},
            "H-repair",
        )
        hdr = _ls_bearer_headers(ls, cf)
        replaced = 0
        failed = 0
        _logged_fetch_sample = False
        for p in bad_files:
            fi = _frame_index_from_stem(p.name)
            url = frame_map.get(fi) if fi is not None else None
            if not url:
                failed += 1
                continue
            resolved = _normalize_ls_upload_path_to_data_upload(_resolve_media_url(url, base_no_slash))
            primary = _rewrite_data_upload_to_storage_proxy(resolved)
            try_urls = [primary]
            if primary != resolved:
                try_urls.append(resolved)
            data = b""
            r = None
            for attempt in try_urls:
                if not _logged_fetch_sample:
                    _agent_log(
                        "repair_fetch_url_sample",
                        {
                            "file": p.name,
                            "resolved_path": urllib.parse.urlsplit(resolved).path,
                            "attempt_path": urllib.parse.urlsplit(attempt).path,
                            "attempt_qs_prefix": (urllib.parse.urlsplit(attempt).query or "")[:100],
                        },
                        "H-repair",
                    )
                    _logged_fetch_sample = True
                try:
                    r = requests.get(attempt, headers=hdr, timeout=120, allow_redirects=True)
                except requests.RequestException as e:
                    _agent_log("repair_get_err", {"file": p.name, "err": str(e)[:120]}, "H-repair")
                    r = None
                    break
                data = r.content
                if r.status_code == 200 and _is_valid_raster_image_payload(data):
                    break
            if r is None:
                failed += 1
                continue
            if r.status_code != 200 or not _is_valid_raster_image_payload(data):
                _agent_log(
                    "repair_bad_response",
                    {
                        "file": p.name,
                        "status": r.status_code,
                        "len": len(data),
                        "ct": r.headers.get("Content-Type", ""),
                    },
                    "H-repair",
                )
                failed += 1
                continue
            out_path = _output_path_for_image_bytes(p, data)
            if out_path != p and p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
            out_path.write_bytes(data)
            replaced += 1
        _agent_log("repair_write_done", {"replaced": replaced, "failed": failed}, "H-repair")
        repaired = zip_path.with_name(zip_path.stem + "_repaired.zip")
        if repaired.exists():
            repaired.unlink()
        with zipfile.ZipFile(repaired, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for fp in unpack.rglob("*"):
                if fp.is_file():
                    zout.write(fp, fp.relative_to(unpack))
        return repaired
    finally:
        shutil.rmtree(unpack, ignore_errors=True)


def _label_studio_url() -> str:
    u = (os.environ.get("LABEL_STUDIO_URL") or "https://labeling.cardiovis.com/").strip()
    return u if u.endswith("/") else u + "/"


def _label_studio_api_key() -> str:
    return (
        os.environ.get("LABEL_STUDIO_API_KEY")
        or os.environ.get("LABEL_STUDIO_TOKEN")
        or ""
    ).strip()


LABEL_STUDIO_URL = _label_studio_url()
API_KEY = _label_studio_api_key()
if not API_KEY:
    print(
        "Set LABEL_STUDIO_API_KEY (or LABEL_STUDIO_TOKEN) to your Label Studio "
        "access token (Account → Access Token). Do not commit it.",
        file=sys.stderr,
    )
    sys.exit(1)


def _cf_env_value(raw: str | None) -> str:
    """Strip whitespace; allow mistaken 'CF-Access-Client-Id: <value>' pastes."""
    s = (raw or "").strip()
    if not s:
        return ""
    low = s.lower()
    for prefix in (
        "cf-access-client-id:",
        "cf-access-client-secret:",
    ):
        if low.startswith(prefix):
            return s[len(prefix) :].strip()
    return s


def _cf_access_headers():
    """Optional machine auth when Label Studio is behind Cloudflare Access."""
    cid = _cf_env_value(
        os.environ.get("CF_ACCESS_CLIENT_ID")
        or os.environ.get("CLOUDFLARE_ACCESS_CLIENT_ID")
    )
    csec = _cf_env_value(
        os.environ.get("CF_ACCESS_CLIENT_SECRET")
        or os.environ.get("CLOUDFLARE_ACCESS_CLIENT_SECRET")
    )
    if cid and csec:
        return {"CF-Access-Client-Id": cid, "CF-Access-Client-Secret": csec}
    return {}


_cf_headers = _cf_access_headers()
_agent_log(
    "startup_env",
    {
        "label_studio_url": LABEL_STUDIO_URL,
        "api_key_len": len(API_KEY),
        "api_key_jwt_shape": API_KEY.count(".") == 2,
        "cf_headers_configured": bool(_cf_headers),
    },
    "H2-H5",
)

_base_no_slash = LABEL_STUDIO_URL.rstrip("/")
_version_url = _base_no_slash + "/api/version"
_probe_headers = {"Authorization": f"Token {API_KEY}"}
_probe_headers.update(_cf_headers)
_resp = requests.get(
    _version_url,
    headers=_probe_headers,
    timeout=30,
    allow_redirects=True,
)
_text = _resp.text or ""
_final = _resp.url or ""
_is_cf_gate = (
    "cloudflareaccess.com" in _final
    or "cdn-cgi/access/login" in _final
    or "Cloudflare Access" in _text
)
_ct = (_resp.headers.get("Content-Type") or "").lower()
_looks_like_json = "application/json" in _ct or _text.lstrip().startswith("{")
_agent_log(
    "cf_probe_result",
    {
        "status_code": _resp.status_code,
        "final_url_host": _final.split("//", 1)[-1].split("/", 1)[0] if _final else "",
        "is_cf_gate": _is_cf_gate,
        "looks_like_json": _looks_like_json,
        "content_type": _resp.headers.get("Content-Type", ""),
    },
    "H1",
)
if _is_cf_gate and not _looks_like_json:
    _cf_set = bool(_cf_headers)
    print(
        "This host is behind Cloudflare Access: /api/version returned the Access "
        "login HTML instead of JSON.\n\n"
        f"In this process, CF_ACCESS_* service token env vars look "
        f"{'set (both ID and secret non-empty)' if _cf_set else 'unset or incomplete'}.\n"
        + (
            "\n"
            "Cloudflare is still returning the IdP login page. Common causes:\n"
            "  • Policy Action must be 'Service Auth', not 'Allow'.\n"
            "  • Confirm CF_ACCESS_CLIENT_ID matches the token in your Access policy.\n"
            "  • Docs: https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/\n\n"
            if _cf_set
            else ""
        )
        + "Export CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET (see script comments).\n",
        file=sys.stderr,
    )
    sys.exit(1)

_ls_headers = dict(_cf_headers) if _cf_headers else None
try:
    ls = LabelStudio(base_url=_base_no_slash, api_key=API_KEY, headers=_ls_headers)
except ApiError as _e:
    _log_api_error("label_studio_client_init_api_error", _e, "H2")
    print(_e, file=sys.stderr)
    sys.exit(1)

_patch_token_refresh_to_include_cf_access(ls, _base_no_slash, dict(_cf_headers))

try:
    ls.users.whoami()
except ApiError as _e:
    _log_api_error("whoami_api_error", _e, "H2")
    if _e.status_code == 401:
        print(
            "Label Studio returned 401 on /api/current-user/whoami.\n"
            "Personal Access Tokens (JWT-shaped) must use the modern SDK path: this "
            "script now uses LabelStudio(), which calls /api/token/refresh and sends "
            "Authorization: Bearer <access>.\n"
            "If you still see 401: create a new PAT, or use a Legacy token "
            "(Account settings) if your org enables it — legacy tokens use "
            "Authorization: Token … and also work with LabelStudio().\n",
            file=sys.stderr,
        )
    else:
        print(_e, file=sys.stderr)
    sys.exit(1)

_agent_log("whoami_ok", {}, "H2")

PROJECT_ID = int(os.environ.get("LABEL_STUDIO_PROJECT_ID", "13"))
# Plain YOLO often omits image bytes; use *_WITH_IMAGES for zip with photos (Label Studio ≥2.21-style).
EXPORT_TYPE = os.environ.get("LABEL_STUDIO_EXPORT_TYPE", "YOLO_WITH_IMAGES")
# Server-side resource fetch for the zip does not send Cloudflare Access headers → HTML saved as .jpg.
_cf_active = bool(_cf_headers)
_dl_raw = os.environ.get("LABEL_STUDIO_DOWNLOAD_RESOURCES", "").strip().lower()
if _dl_raw in ("1", "true", "yes"):
    _download_resources = True
elif _dl_raw in ("0", "false", "no"):
    _download_resources = False
else:
    _download_resources = not _cf_active
_repair_raw = os.environ.get("LABEL_STUDIO_REPAIR_CF_IMAGES", "").strip().lower()
if _repair_raw in ("0", "false", "no"):
    _repair_cf_images = False
elif _repair_raw in ("1", "true", "yes"):
    _repair_cf_images = True
else:
    _repair_cf_images = _cf_active
if _cf_active and not _download_resources:
    print(
        "Note: LABEL_STUDIO_DOWNLOAD_RESOURCES defaults to false when CF_ACCESS_* is set, "
        "so the server does not embed Cloudflare login HTML as image bytes. "
        "Images are re-fetched with your Bearer token + service headers when needed.\n",
        file=sys.stderr,
    )

if not os.environ.get("LABEL_STUDIO_EXPORT_VIEW_ID", "").strip():
    print(
        "Tip: set LABEL_STUDIO_EXPORT_VIEW_ID to the Data Manager tab id from the URL "
        "(…/data?tab=14 → export LABEL_STUDIO_EXPORT_VIEW_ID=14) so the export matches "
        "the tasks you see in the UI.",
        file=sys.stderr,
    )
print(
    "Export task filter: all tasks (unlabeled + unfinished included by default). "
    "To restore the old scope, set LABEL_STUDIO_EXPORT_ANNOTATED_ONLY=1 and "
    "LABEL_STUDIO_EXPORT_FINISHED_ONLY=1.\n",
    file=sys.stderr,
)

try:
    _created = ls.projects.exports.create(
        id=PROJECT_ID,
        title="Feature 3: Patient 10",
        task_filter_options=_export_task_filter_options(),
        serialization_options=_export_serialization_options(),
    )
except ApiError as _e:
    _log_api_error("export_create_api_error", _e, "H4")
    print(_e, file=sys.stderr)
    sys.exit(1)

_export_id = _created.id
if _export_id is None:
    print("Export create returned no id.", file=sys.stderr)
    sys.exit(1)

_agent_log("export_created", {"export_id": _export_id}, "H4")

while True:
    try:
        _exp = ls.projects.exports.get(PROJECT_ID, _export_id)
    except ApiError as _e:
        _log_api_error("export_get_api_error", _e, "H4")
        print(_e, file=sys.stderr)
        sys.exit(1)
    _st = _exp.status or ""
    if _st == "failed":
        _agent_log("export_failed", {"export_id": _export_id, "status": _st}, "H4")
        print(f"Export {_export_id} failed on the server.", file=sys.stderr)
        sys.exit(1)
    if _st in ("completed",):
        break
    time.sleep(1.0)

_agent_log("export_completed", {"export_id": _export_id}, "H4")

try:
    ls.projects.exports.convert(
        id=PROJECT_ID,
        export_pk=_export_id,
        export_type=EXPORT_TYPE,
        download_resources=_download_resources,
    )
except ApiError as _e:
    _log_api_error("export_convert_api_error", _e, "H4")
    print(_e, file=sys.stderr)
    sys.exit(1)

_wait_for_converted_format(_export_id, PROJECT_ID, EXPORT_TYPE)

_out_dir = Path(".")
_out_dir.mkdir(parents=True, exist_ok=True)
_default_name = f"export-{PROJECT_ID}-{_export_id}-{EXPORT_TYPE}.zip"
_out_path = _out_dir / _default_name

try:
    with open(_out_path, "wb") as _out:
        for _chunk in ls.projects.exports.download(
            PROJECT_ID, _export_id, export_type=EXPORT_TYPE
        ):
            _out.write(_chunk)
except ApiError as _e:
    _log_api_error("export_download_api_error", _e, "H4")
    print(_e, file=sys.stderr)
    sys.exit(1)

_agent_log("export_download_ok", {"path": str(_out_path)}, "H4")
if "YOLO" in (EXPORT_TYPE or "").upper() and os.environ.get(
    "LABEL_STUDIO_SKIP_LABEL_PATCH", ""
).strip().lower() not in ("1", "true", "yes"):
    _patch_yolo_zip_labels_from_json(ls, PROJECT_ID, _export_id, _out_path, _base_no_slash)
    print(
        "Patched labels/*.txt from JSON (rectangles; brush→axis-aligned box; polygons→box). "
        "Re-unzip the export if you use extracted images/labels folders.\n",
        file=sys.stderr,
    )
_repaired_path: Path | None = None
if _repair_cf_images and _cf_headers and "YOLO" in (EXPORT_TYPE or "").upper():
    _repaired_path = _repair_yolo_zip_with_cf_auth(
        ls,
        PROJECT_ID,
        _export_id,
        _out_path,
        _base_no_slash,
        dict(_cf_headers),
    )
    if _repaired_path is not None:
        _agent_log("export_repaired_zip", {"path": str(_repaired_path)}, "H-repair")
        print(
            f"Repaired images (authenticated fetch; PNG/WebP saved with correct extension): "
            f"{_repaired_path.resolve()}",
            file=sys.stderr,
        )
print(f"Status of the export is OK.\nFile path is {_out_path.resolve()}")
