import re
import shutil
import sys
import urllib.parse
import zipfile
from pathlib import Path

import numpy as np
from label_studio_sdk import LabelStudio

from ls_export.export_api import _download_json_export_bytes, _ensure_json_export_ready
from ls_export.logging_support import _agent_log
from ls_export.media import (
    _frame_index_from_stem,
    _task_image_field,
    _tasks_from_json_payload,
)


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
