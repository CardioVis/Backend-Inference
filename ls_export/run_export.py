import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

from label_studio_sdk import LabelStudio
from label_studio_sdk.core.api_error import ApiError

from ls_export.cf_repair import _repair_yolo_zip_with_cf_auth
from ls_export.export_api import (
    _export_serialization_options,
    _export_task_filter_options,
    _wait_for_converted_format,
)
from ls_export.logging_support import _agent_log, _log_api_error
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


def _run_export(
    ls: LabelStudio,
    base_no_slash: str,
    cf_headers: dict,
    project_id: int,
    export_title: str,
    output_parent: Path,
    *,
    extract: bool,
) -> None:
    export_type = os.environ.get("LABEL_STUDIO_EXPORT_TYPE", "YOLO_WITH_IMAGES")
    download_resources, repair_cf_images = _export_resource_flags(cf_headers)

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
        created = ls.projects.exports.create(
            id=project_id,
            title=export_title,
            task_filter_options=_export_task_filter_options(),
            serialization_options=_export_serialization_options(),
        )
    except ApiError as e:
        _log_api_error("export_create_api_error", e, "H4")
        print(e, file=sys.stderr)
        sys.exit(1)

    export_id = created.id
    if export_id is None:
        print("Export create returned no id.", file=sys.stderr)
        sys.exit(1)

    _agent_log("export_created", {"export_id": export_id}, "H4")

    while True:
        try:
            exp = ls.projects.exports.get(project_id, export_id)
        except ApiError as e:
            _log_api_error("export_get_api_error", e, "H4")
            print(e, file=sys.stderr)
            sys.exit(1)
        st = exp.status or ""
        if st == "failed":
            _agent_log("export_failed", {"export_id": export_id, "status": st}, "H4")
            print(f"Export {export_id} failed on the server.", file=sys.stderr)
            sys.exit(1)
        if st in ("completed",):
            break
        time.sleep(1.0)

    _agent_log("export_completed", {"export_id": export_id}, "H4")

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
        sys.exit(1)

    _wait_for_converted_format(ls, export_id, project_id, export_type)

    output_parent.mkdir(parents=True, exist_ok=True)
    run_dir = output_parent / f"export-{project_id}-{export_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    default_name = f"export-{project_id}-{export_id}-{export_type}.zip"
    out_path = run_dir / default_name

    try:
        with open(out_path, "wb") as out_f:
            for chunk in ls.projects.exports.download(
                project_id, export_id, export_type=export_type
            ):
                out_f.write(chunk)
    except ApiError as e:
        _log_api_error("export_download_api_error", e, "H4")
        print(e, file=sys.stderr)
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
    else:
        print(f"Status of the export is OK.\nZip: {final_zip.resolve()}")
