import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

from label_studio_sdk import LabelStudio
from label_studio_sdk.core.api_error import ApiError

from ls_export.cf_repair import _repair_yolo_zip_with_cf_auth
from ls_export.download_progress import write_download_stream
from ls_export.export_normalize import normalize_ls_yolo_filenames
from ls_export.export_api import (
    _export_serialization_options,
    _export_task_filter_options,
    _wait_for_converted_format,
)
from ls_export.export_progress import poll_export_snapshot
from ls_export.logging_support import _agent_log, _log_api_error
from ls_export.guideline_export import write_guideline_seq_from_extract
from ls_export.tasks_fallback import maybe_run_tasks_fallback
from ls_export.yolo_patch import _patch_yolo_zip_labels_from_json


def _export_resource_flags(cf_headers: dict) -> tuple[bool, bool]:
    """Return (download_resources, repair_cf_images) from env; emit the same notes as before."""
    cf_active = bool(cf_headers)
    dl_raw = os.environ.get("LABEL_STUDIO_DOWNLOAD_RESOURCES", "").strip().lower()
    if dl_raw in ("1", "true", "yes"):
        download_resources = True
    elif dl_raw in ("0", "false", "no"):
        download_resources = False
    else:
        download_resources = not cf_active
    repair_raw = os.environ.get("LABEL_STUDIO_REPAIR_CF_IMAGES", "").strip().lower()
    if repair_raw in ("0", "false", "no"):
        repair_cf_images = False
    elif repair_raw in ("1", "true", "yes"):
        repair_cf_images = True
    else:
        repair_cf_images = cf_active
    if cf_active and not download_resources:
        print(
            "Note: LABEL_STUDIO_DOWNLOAD_RESOURCES defaults to false when CF_ACCESS_* is set, "
            "so the server does not embed Cloudflare login HTML as image bytes. "
            "Images are re-fetched with your Bearer token + service headers when needed.\n",
            file=sys.stderr,
        )
    return download_resources, repair_cf_images


def _extract_zip_to_dir(zip_path: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _api_error_is_server_failure(err: ApiError) -> bool:
    code = err.status_code or 0
    return code >= 500 or code in (408, 429)


def _try_tasks_fallback(
    ls: LabelStudio,
    base_no_slash: str,
    cf_headers: dict,
    project_id: int,
    output_parent: Path,
    *,
    extract: bool,
    normalize_frame_names: bool,
    guideline_mapping: Path | None,
    guideline_subdir: str,
    tasks_fallback: bool | None,
    reason: str,
) -> bool:
    """Return True if fallback completed successfully (caller should return)."""
    result = maybe_run_tasks_fallback(
        ls,
        base_no_slash,
        cf_headers,
        project_id,
        output_parent,
        extract=extract,
        normalize_frame_names=normalize_frame_names,
        guideline_mapping=guideline_mapping,
        guideline_subdir=guideline_subdir,
        explicit=tasks_fallback,
        reason=reason,
    )
    return result is not None


def _run_export(
    ls: LabelStudio,
    base_no_slash: str,
    cf_headers: dict,
    project_id: int,
    export_title: str,
    output_parent: Path,
    *,
    extract: bool,
    normalize_frame_names: bool = True,
    guideline_mapping: Path | None = None,
    guideline_subdir: str = "guideline_seq",
    tasks_fallback: bool | None = None,
) -> None:
    export_type = os.environ.get("LABEL_STUDIO_EXPORT_TYPE", "YOLO_WITH_IMAGES")
    download_resources, repair_cf_images = _export_resource_flags(cf_headers)

    view_id = os.environ.get("LABEL_STUDIO_EXPORT_VIEW_ID", "").strip()
    if not view_id:
        print(
            "Tip: set LABEL_STUDIO_EXPORT_VIEW_ID to the Data Manager tab id from the URL "
            "(…/data?tab=34 → export LABEL_STUDIO_EXPORT_VIEW_ID=34) so the export matches "
            "the tasks you see in the UI.",
            file=sys.stderr,
        )
    else:
        print(f"Using Data Manager view id {view_id} for export / fallback.\n", file=sys.stderr)

    ann = os.environ.get("LABEL_STUDIO_EXPORT_ANNOTATED_ONLY", "").strip().lower() in ("1", "true", "yes")
    fin = os.environ.get("LABEL_STUDIO_EXPORT_FINISHED_ONLY", "").strip().lower() in ("1", "true", "yes")
    if ann or fin:
        print(
            f"Export task filter: annotated_only={ann}, finished_only={fin}.\n",
            file=sys.stderr,
        )
    else:
        print(
            "Export task filter: all tasks (unlabeled + unfinished included by default). "
            "To narrow, set LABEL_STUDIO_EXPORT_ANNOTATED_ONLY=1 and/or "
            "LABEL_STUDIO_EXPORT_FINISHED_ONLY=1.\n",
            file=sys.stderr,
        )

    if tasks_fallback is True:
        print("Using task-by-task download (--tasks-fallback).\n", file=sys.stderr)
        run_tasks_only = maybe_run_tasks_fallback(
            ls,
            base_no_slash,
            cf_headers,
            project_id,
            output_parent,
            extract=extract,
            normalize_frame_names=normalize_frame_names,
            guideline_mapping=guideline_mapping,
            guideline_subdir=guideline_subdir,
            explicit=True,
            reason="--tasks-fallback",
        )
        if run_tasks_only is None:
            sys.exit(1)
        return

    print("Creating export snapshot on Label Studio…", file=sys.stderr)
    try:
        created = ls.projects.exports.create(
            id=project_id,
            title=export_title,
            task_filter_options=_export_task_filter_options(),
            serialization_options=_export_serialization_options(),
        )
    except ApiError as e:
        _log_api_error("export_create_api_error", e, "H4")
        print(e, file=sys.stderr)
        if _api_error_is_server_failure(e) and _try_tasks_fallback(
            ls,
            base_no_slash,
            cf_headers,
            project_id,
            output_parent,
            extract=extract,
            normalize_frame_names=normalize_frame_names,
            guideline_mapping=guideline_mapping,
            guideline_subdir=guideline_subdir,
            tasks_fallback=tasks_fallback,
            reason=f"export create HTTP {e.status_code}",
        ):
            return
        sys.exit(1)

    export_id = created.id
    if export_id is None:
        print("Export create returned no id.", file=sys.stderr)
        sys.exit(1)

    _agent_log("export_created", {"export_id": export_id}, "H4")
    print(f"Export snapshot id={export_id}; waiting for completion…", file=sys.stderr)

    def _snapshot_status() -> str:
        exp = ls.projects.exports.get(project_id, export_id)
        return exp.status or ""

    try:
        st = poll_export_snapshot(_snapshot_status, label="export snapshot")
    except ApiError as e:
        _log_api_error("export_get_api_error", e, "H4")
        print(e, file=sys.stderr)
        if _api_error_is_server_failure(e) and _try_tasks_fallback(
            ls,
            base_no_slash,
            cf_headers,
            project_id,
            output_parent,
            extract=extract,
            normalize_frame_names=normalize_frame_names,
            guideline_mapping=guideline_mapping,
            guideline_subdir=guideline_subdir,
            tasks_fallback=tasks_fallback,
            reason=f"export poll HTTP {e.status_code}",
        ):
            return
        sys.exit(1)

    if st == "failed":
        _agent_log("export_failed", {"export_id": export_id, "status": st}, "H4")
        print(f"Export {export_id} failed on the server.", file=sys.stderr)
        if _try_tasks_fallback(
            ls,
            base_no_slash,
            cf_headers,
            project_id,
            output_parent,
            extract=extract,
            normalize_frame_names=normalize_frame_names,
            guideline_mapping=guideline_mapping,
            guideline_subdir=guideline_subdir,
            tasks_fallback=tasks_fallback,
            reason="export snapshot status=failed",
        ):
            return
        sys.exit(1)

    _agent_log("export_completed", {"export_id": export_id}, "H4")
    print(f"Converting export to {export_type}…", file=sys.stderr)

    try:
        ls.projects.exports.convert(
            id=project_id,
            export_pk=export_id,
            export_type=export_type,
            download_resources=download_resources,
        )
    except ApiError as e:
        _log_api_error("export_convert_api_error", e, "H4")
        print(e, file=sys.stderr)
        if _api_error_is_server_failure(e) and _try_tasks_fallback(
            ls,
            base_no_slash,
            cf_headers,
            project_id,
            output_parent,
            extract=extract,
            normalize_frame_names=normalize_frame_names,
            guideline_mapping=guideline_mapping,
            guideline_subdir=guideline_subdir,
            tasks_fallback=tasks_fallback,
            reason=f"export convert HTTP {e.status_code}",
        ):
            return
        sys.exit(1)

    try:
        _wait_for_converted_format(ls, export_id, project_id, export_type)
    except SystemExit:
        if _try_tasks_fallback(
            ls,
            base_no_slash,
            cf_headers,
            project_id,
            output_parent,
            extract=extract,
            normalize_frame_names=normalize_frame_names,
            guideline_mapping=guideline_mapping,
            guideline_subdir=guideline_subdir,
            tasks_fallback=tasks_fallback,
            reason="export conversion failed or timed out",
        ):
            return
        raise

    output_parent.mkdir(parents=True, exist_ok=True)
    run_dir = output_parent / f"export-{project_id}-{export_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    default_name = f"export-{project_id}-{export_id}-{export_type}.zip"
    out_path = run_dir / default_name

    print(f"Downloading {export_type}…", file=sys.stderr)
    try:
        with open(out_path, "wb") as out_f:
            write_download_stream(
                ls.projects.exports.download(
                    project_id, export_id, export_type=export_type
                ),
                out_f,
                desc=f"Download {export_type}",
            )
    except ApiError as e:
        _log_api_error("export_download_api_error", e, "H4")
        print(e, file=sys.stderr)
        if _api_error_is_server_failure(e) and _try_tasks_fallback(
            ls,
            base_no_slash,
            cf_headers,
            project_id,
            output_parent,
            extract=extract,
            normalize_frame_names=normalize_frame_names,
            guideline_mapping=guideline_mapping,
            guideline_subdir=guideline_subdir,
            tasks_fallback=tasks_fallback,
            reason=f"export download HTTP {e.status_code}",
        ):
            return
        sys.exit(1)

    _agent_log("export_download_ok", {"path": str(out_path)}, "H4")
    if "YOLO" in (export_type or "").upper() and os.environ.get(
        "LABEL_STUDIO_SKIP_LABEL_PATCH", ""
    ).strip().lower() not in ("1", "true", "yes"):
        _patch_yolo_zip_labels_from_json(ls, project_id, export_id, out_path, base_no_slash)
        print(
            "Patched labels/*.txt from JSON (rectangles; brush→axis-aligned box; polygons→box). "
            "Images and labels are under the run folder after extraction.\n",
            file=sys.stderr,
        )

    repaired_path: Path | None = None
    if repair_cf_images and cf_headers and "YOLO" in (export_type or "").upper():
        repaired_path = _repair_yolo_zip_with_cf_auth(
            ls,
            project_id,
            export_id,
            out_path,
            base_no_slash,
            dict(cf_headers),
        )
        if repaired_path is not None:
            _agent_log("export_repaired_zip", {"path": str(repaired_path)}, "H-repair")
            print(
                "Repaired images (authenticated fetch; PNG/WebP saved with correct extension): "
                f"{repaired_path.resolve()}",
                file=sys.stderr,
            )
            if (
                repaired_path.is_file()
                and out_path.is_file()
                and repaired_path.resolve() != out_path.resolve()
            ):
                try:
                    out_path.unlink()
                except OSError:
                    pass

    final_zip = repaired_path if repaired_path is not None and repaired_path.is_file() else out_path
    extract_dir = run_dir / "extracted"
    if extract and final_zip.suffix.lower() == ".zip":
        _extract_zip_to_dir(final_zip, extract_dir)
        print(
            f"Status of the export is OK.\n"
            f"Zip: {final_zip.resolve()}\n"
            f"Extracted dataset: {extract_dir.resolve()}",
        )
        if normalize_frame_names:
            n_renamed = normalize_ls_yolo_filenames(extract_dir)
            if n_renamed:
                print(
                    f"Renamed {n_renamed} file(s) to frame_* (stripped Label Studio id prefix).\n",
                    file=sys.stderr,
                )
        if guideline_mapping is not None:
            if "YOLO" not in (export_type or "").upper():
                print(
                    "Skipping guideline export: export type is not YOLO (use YOLO_WITH_IMAGES or similar).\n",
                    file=sys.stderr,
                )
            elif not guideline_mapping.is_file():
                print(
                    f"Skipping guideline export: mapping file not found: {guideline_mapping}\n",
                    file=sys.stderr,
                )
            else:
                gdir = write_guideline_seq_from_extract(
                    extract_dir,
                    guideline_mapping,
                    run_dir,
                    subdir=guideline_subdir,
                )
                print(
                    f"Guideline-style export: {gdir.resolve()}\n",
                    file=sys.stderr,
                )
    else:
        print(f"Status of the export is OK.\nZip: {final_zip.resolve()}")
        if guideline_mapping is not None and not extract:
            print(
                "Guideline export skipped: extraction disabled (--no-extract).\n",
                file=sys.stderr,
            )
