"""Download project frames + YOLO labels via Tasks API when snapshot export fails."""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.parse
import zipfile
from pathlib import Path

import requests
from label_studio_sdk import LabelStudio

from ls_export.client import _ls_bearer_headers
from ls_export.logging_support import _agent_log
from ls_export.media import (
    _is_valid_raster_image_payload,
    _normalize_ls_upload_path_to_data_upload,
    _output_path_for_image_bytes,
    _resolve_media_url,
    _rewrite_data_upload_to_storage_proxy,
    _task_image_field,
)
from ls_export.export_normalize import normalize_ls_yolo_filenames
from ls_export.guideline_export import write_guideline_seq_from_extract
from ls_export.yolo_patch import (
    _build_name_to_id_map,
    _task_frame_index,
    _yolo_lines_for_task,
)


def _tasks_fallback_enabled(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    raw = os.environ.get("LABEL_STUDIO_TASKS_FALLBACK", "auto").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _filter_tasks_client_side(tasks: list[dict]) -> list[dict]:
    annotated_only = os.environ.get("LABEL_STUDIO_EXPORT_ANNOTATED_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    finished_only = os.environ.get("LABEL_STUDIO_EXPORT_FINISHED_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    out: list[dict] = []
    for t in tasks:
        if finished_only and not t.get("is_labeled"):
            continue
        anns = t.get("annotations") or []
        has_ann = any(not a.get("was_cancelled") for a in anns if isinstance(a, dict))
        if annotated_only and not has_ann:
            continue
        out.append(t)
    return out


def _fetch_tasks_page(
    hdr: dict,
    url: str,
    *,
    page: int,
    page_size: int,
    extra_params: dict | None = None,
) -> tuple[list[dict], bool]:
    params = {"page": page, "page_size": page_size}
    if extra_params:
        params.update(extra_params)
    r = requests.get(
        url,
        headers=hdr,
        params=params,
        timeout=120,
        allow_redirects=True,
    )
    if r.status_code != 200:
        preview = (r.text or "")[:500]
        raise RuntimeError(f"Tasks API HTTP {r.status_code}: {preview}")
    try:
        body = r.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(
            "Tasks API returned non-JSON (check Cloudflare / token headers)."
        ) from e
    if isinstance(body, list):
        items = [x for x in body if isinstance(x, dict)]
        return items, len(items) >= page_size
    items = body.get("tasks") or body.get("results") or []
    if not isinstance(items, list):
        items = []
    items = [x for x in items if isinstance(x, dict)]
    total = body.get("total")
    if isinstance(total, int):
        has_more = page * page_size < total
    else:
        has_more = len(items) >= page_size
    return items, has_more


def _task_to_dict(task) -> dict:
    if isinstance(task, dict):
        return task
    if hasattr(task, "model_dump"):
        return task.model_dump()
    if hasattr(task, "dict"):
        return task.dict()
    return dict(task)


def _list_project_tasks(
    ls: LabelStudio,
    base_no_slash: str,
    hdr: dict,
    project_id: int,
) -> list[dict]:
    view_raw = os.environ.get("LABEL_STUDIO_EXPORT_VIEW_ID", "").strip()
    page_size = int(os.environ.get("LABEL_STUDIO_TASKS_PAGE_SIZE", "100"))
    annotated_only = os.environ.get("LABEL_STUDIO_EXPORT_ANNOTATED_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    list_kwargs: dict = {
        "project": project_id,
        "fields": "all",
        "page_size": page_size,
    }
    if view_raw and view_raw.lower() not in ("none", "false", "0"):
        list_kwargs["view"] = int(view_raw)
        print(
            f"Fetching tasks (SDK, view={list_kwargs['view']}, fields=all)…",
            file=sys.stderr,
        )
    else:
        print(f"Fetching tasks for project {project_id} (SDK, fields=all)…", file=sys.stderr)
    if annotated_only:
        list_kwargs["only_annotated"] = True

    all_tasks: list[dict] = []
    try:
        pager = ls.tasks.list(**list_kwargs)
        for task in pager:
            all_tasks.append(_task_to_dict(task))
            if len(all_tasks) % page_size == 0:
                print(f"  … {len(all_tasks)} tasks", file=sys.stderr)
    except Exception as e:
        _agent_log("tasks_sdk_list_failed", {"err": str(e)[:300]}, "H-tasks-fallback")
        print(f"SDK task list failed ({e}); trying REST fallback…", file=sys.stderr)
        all_tasks = _list_project_tasks_rest(base_no_slash, hdr, project_id, view_raw, page_size)

    filtered = _filter_tasks_client_side(all_tasks)
    if len(filtered) != len(all_tasks):
        print(
            f"Filtered to {len(filtered)} task(s) "
            f"(from {len(all_tasks)}; annotated/finished env filters).",
            file=sys.stderr,
        )
    return filtered


def _list_project_tasks_rest(
    base_no_slash: str,
    hdr: dict,
    project_id: int,
    view_raw: str,
    page_size: int,
) -> list[dict]:
    """REST fallback when SDK task list fails."""
    extra_params: dict | None = None
    if view_raw and view_raw.lower() not in ("none", "false", "0"):
        view_id = int(view_raw)
        url = f"{base_no_slash.rstrip('/')}/api/dm/views/{view_id}/tasks/"
        extra_params = {"project": project_id}
    else:
        url = f"{base_no_slash.rstrip('/')}/api/projects/{project_id}/tasks/"
    all_tasks: list[dict] = []
    page = 1
    while page <= 10_000:
        items, has_more = _fetch_tasks_page(
            hdr, url, page=page, page_size=page_size, extra_params=extra_params
        )
        if not items:
            break
        all_tasks.extend(items)
        if not has_more:
            break
        page += 1
    return all_tasks


def _download_task_image(
    raw_url: str,
    dest: Path,
    base_no_slash: str,
    hdr: dict,
) -> bool:
    resolved = _normalize_ls_upload_path_to_data_upload(
        _resolve_media_url(raw_url, base_no_slash)
    )
    primary = _rewrite_data_upload_to_storage_proxy(resolved)
    try_urls = [primary]
    if primary != resolved:
        try_urls.append(resolved)
    for attempt in try_urls:
        try:
            r = requests.get(attempt, headers=hdr, timeout=120, allow_redirects=True)
        except requests.RequestException:
            continue
        if r.status_code == 200 and _is_valid_raster_image_payload(r.content):
            out = _output_path_for_image_bytes(dest, r.content)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(r.content)
            return True
    return False


def _stem_for_task(task: dict) -> str:
    fi = _task_frame_index(task)
    if fi is not None:
        return f"frame_{fi:05d}"
    tid = task.get("id")
    return f"task_{tid}" if tid is not None else "task_unknown"


def _write_data_yaml(root: Path, name_to_id: dict[str, int]) -> None:
    lines = ["path: .", "train: images", "names:"]
    for name, cid in sorted(name_to_id.items(), key=lambda x: x[1]):
        lines.append(f"  {cid}: {name}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tasks_fallback(
    ls: LabelStudio,
    base_no_slash: str,
    cf_headers: dict,
    project_id: int,
    output_parent: Path,
    *,
    extract: bool,
    normalize_frame_names: bool = True,
    guideline_mapping: Path | None = None,
    guideline_subdir: str = "guideline_seq",
) -> Path:
    """Build a YOLO zip from /api/tasks (or dm view tasks) + authenticated image fetch."""
    hdr = _ls_bearer_headers(ls, cf_headers)
    tasks = _list_project_tasks(ls, base_no_slash, hdr, project_id)
    if not tasks:
        print("No tasks to download after filtering.", file=sys.stderr)
        sys.exit(1)

    run_dir = output_parent / f"export-{project_id}-tasks-fallback"
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    work = run_dir / "yolo_build"
    images_dir = work / "images"
    labels_dir = work / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    name_to_id = _build_name_to_id_map(None, tasks)
    _write_data_yaml(work, name_to_id)

    ok_images = 0
    skip_images = 0
    for i, task in enumerate(tasks, 1):
        stem = _stem_for_task(task)
        raw = _task_image_field(task)
        if raw:
            dest = images_dir / f"{stem}.jpg"
            if _download_task_image(raw, dest, base_no_slash, hdr):
                ok_images += 1
            else:
                skip_images += 1
                print(f"  Warning: could not download image for {stem}", file=sys.stderr)
        else:
            skip_images += 1

        lines = _yolo_lines_for_task(task, name_to_id)
        body = "\n".join(lines)
        (labels_dir / f"{stem}.txt").write_text(body + ("\n" if body else ""), encoding="utf-8")

        if i % 25 == 0 or i == len(tasks):
            print(
                f"  Processed {i}/{len(tasks)} tasks "
                f"({ok_images} images OK, {skip_images} skipped/failed).",
                file=sys.stderr,
            )

    zip_path = run_dir / f"export-{project_id}-tasks-fallback-YOLO_WITH_IMAGES.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in work.rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(work))

    _agent_log(
        "tasks_fallback_done",
        {
            "tasks": len(tasks),
            "ok_images": ok_images,
            "skip_images": skip_images,
            "zip": str(zip_path),
        },
        "H-tasks-fallback",
    )

    extract_dir = run_dir / "extracted"
    if extract:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        if normalize_frame_names:
            n_renamed = normalize_ls_yolo_filenames(extract_dir)
            if n_renamed:
                print(
                    f"Renamed {n_renamed} file(s) to frame_* (tasks fallback).\n",
                    file=sys.stderr,
                )
        if guideline_mapping is not None:
            if not guideline_mapping.is_file():
                print(
                    f"Skipping guideline export: mapping file not found: {guideline_mapping.resolve()}\n",
                    file=sys.stderr,
                )
            else:
                gdir = write_guideline_seq_from_extract(
                    extract_dir,
                    guideline_mapping,
                    run_dir,
                    subdir=guideline_subdir,
                )
                n_json = len(list((gdir / "json").glob("ann_*.json")))
                print(
                    f"Guideline-style export: {gdir.resolve()} ({n_json} ann_*.json)\n",
                    file=sys.stderr,
                )
        print(
            f"Status of the export is OK (tasks fallback).\n"
            f"Zip: {zip_path.resolve()}\n"
            f"Extracted dataset: {extract_dir.resolve()}",
        )
    else:
        print(f"Status of the export is OK (tasks fallback).\nZip: {zip_path.resolve()}")

    return zip_path


def maybe_run_tasks_fallback(
    ls: LabelStudio,
    base_no_slash: str,
    cf_headers: dict,
    project_id: int,
    output_parent: Path,
    *,
    extract: bool,
    normalize_frame_names: bool = True,
    guideline_mapping: Path | None = None,
    guideline_subdir: str = "guideline_seq",
    explicit: bool | None = None,
    reason: str,
) -> Path | None:
    if not _tasks_fallback_enabled(explicit):
        return None
    if reason.startswith("--"):
        print(f"\nTask-by-task download ({reason})…\n", file=sys.stderr)
    else:
        print(
            f"\nSnapshot export failed ({reason}). "
            "Using task-by-task download (LABEL_STUDIO_TASKS_FALLBACK)…\n",
            file=sys.stderr,
        )
    return run_tasks_fallback(
        ls,
        base_no_slash,
        cf_headers,
        project_id,
        output_parent,
        extract=extract,
        normalize_frame_names=normalize_frame_names,
        guideline_mapping=guideline_mapping,
        guideline_subdir=guideline_subdir,
    )
