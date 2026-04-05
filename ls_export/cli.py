import argparse
import os
import sys
from pathlib import Path

from ls_export.client import _connect_label_studio, _list_projects_print
from ls_export.run_export import _run_export


def _build_argparser() -> argparse.ArgumentParser:
    epilog = """
Environment (details in README.md):
  LABEL_STUDIO_API_KEY or LABEL_STUDIO_TOKEN     Required access token
  LABEL_STUDIO_URL                               Optional; default CardioVis host
  LABEL_STUDIO_PROJECT_ID                        Used when --project-id is omitted (default: 13)
  LABEL_STUDIO_EXPORT_TITLE                      Default snapshot title if --export-title omitted
  LABEL_STUDIO_EXPORT_TYPE                       e.g. YOLO_WITH_IMAGES
  LABEL_STUDIO_OUTPUT_DIR                        Base directory for export-... run folders
  LABEL_STUDIO_SKIP_EXTRACT=1                    Skip unzipping (with --no-extract)
  CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET  Cloudflare Access service tokens
"""
    p = argparse.ArgumentParser(
        description="Download a Label Studio project export (YOLO with images by default).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument(
        "-p",
        "--project-id",
        type=int,
        default=None,
        help="Label Studio project id (overrides LABEL_STUDIO_PROJECT_ID)",
    )
    p.add_argument(
        "--list-projects",
        action="store_true",
        help="List project ids and titles (tab-separated), then exit",
    )
    p.add_argument(
        "--export-title",
        default=None,
        help="Export snapshot title (overrides LABEL_STUDIO_EXPORT_TITLE)",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Base folder for export-<project>-<export_id>/ (default: LABEL_STUDIO_OUTPUT_DIR or label_studio_exports)",
    )
    p.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep only the zip; do not unzip into export-.../extracted/",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    if args.list_projects:
        ls, base, cf = _connect_label_studio()
        _list_projects_print(ls, base, cf)
        return

    project_id = (
        args.project_id
        if args.project_id is not None
        else int(os.environ.get("LABEL_STUDIO_PROJECT_ID", "13"))
    )
    export_title = (
        (args.export_title or os.environ.get("LABEL_STUDIO_EXPORT_TITLE") or "CardioVis export").strip()
    )
    if args.output_dir is not None:
        output_parent = args.output_dir
    else:
        raw = os.environ.get("LABEL_STUDIO_OUTPUT_DIR", "label_studio_exports").strip()
        output_parent = Path(raw) if raw else Path("label_studio_exports")

    extract = not args.no_extract and os.environ.get(
        "LABEL_STUDIO_SKIP_EXTRACT", ""
    ).strip().lower() not in ("1", "true", "yes")

    ls, base, cf = _connect_label_studio()
    _run_export(ls, base, cf, project_id, export_title, output_parent, extract=extract)
