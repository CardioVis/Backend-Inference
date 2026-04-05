import shutil
import sys
import urllib.parse
import zipfile
from pathlib import Path

import requests
from label_studio_sdk import LabelStudio

from ls_export.client import _ls_bearer_headers
from ls_export.export_api import _download_json_export_bytes, _ensure_json_export_ready
from ls_export.logging_support import _agent_log
from ls_export.media import (
    _build_frame_to_image_url,
    _frame_index_from_stem,
    _is_valid_raster_image_payload,
    _normalize_ls_upload_path_to_data_upload,
    _output_path_for_image_bytes,
    _resolve_media_url,
    _rewrite_data_upload_to_storage_proxy,
    _tasks_from_json_payload,
)


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
