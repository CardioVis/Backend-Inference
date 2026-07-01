import argparse
import os
import sys
from pathlib import Path

from ls_export.env_loader import load_repo_dotenv
from ls_export.client import _connect_label_studio, _list_projects_print
from ls_export.guideline_only import run_guideline_only
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
  LABEL_STUDIO_GUIDELINE_MAPPING                 Path to JSON mapping for guideline_seq export
  LABEL_STUDIO_NO_NORMALIZE_FRAME_NAMES=1        Keep LS hash-prefixed image/label names
  LABEL_STUDIO_NO_TQDM=1                         Disable tqdm byte progress on stderr
  CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET  Cloudflare Access service tokens
  LABEL_STUDIO_TASKS_FALLBACK                  auto (default) | 1 | 0 — task API if snapshot export fails
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
    p.add_argument(
        "--no-normalize-frame-names",
        action="store_true",
        help="Do not rename images/labels to frame_<digits>.* under extracted/",
    )
    p.add_argument(
        "--guideline-mapping",
        type=Path,
        default=None,
        help="JSON file with yolo_to_guideline map; writes export-.../guideline_seq/ (needs --extract)",
    )
    p.add_argument(
        "--guideline-subdir",
        default="guideline_seq",
        help="Subfolder under export-<project>-<id>/ for guideline output (default: guideline_seq)",
    )
    p.add_argument(
        "--tasks-fallback",
        action="store_true",
        help="Skip snapshot export; download frames via Tasks API (for LS export 500s)",
    )
    p.add_argument(
        "--no-tasks-fallback",
        action="store_true",
        help="Do not fall back to Tasks API when snapshot export fails",
    )
    p.add_argument(
        "--guideline-only",
        action="store_true",
        help="Only build guideline_seq/ from existing export-.../extracted/ (no LS download)",
    )
    p.add_argument(
        "--from-run-dir",
        type=Path,
        default=None,
        help="Export run folder containing extracted/ (with --guideline-only)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    load_repo_dotenv()
    args = _build_argparser().parse_args(argv)
    if args.tasks_fallback and args.no_tasks_fallback:
        print("Use only one of --tasks-fallback and --no-tasks-fallback.", file=sys.stderr)
        sys.exit(2)
    tasks_fallback: bool | None = True if args.tasks_fallback else False if args.no_tasks_fallback else None
    if args.list_projects:
        ls, base, cf = _connect_label_studio()
        _list_projects_print(ls, base, cf)
        return

    if args.guideline_only and not args.guideline_mapping:
        gm = os.environ.get("LABEL_STUDIO_GUIDELINE_MAPPING", "").strip()
        if gm:
            args.guideline_mapping = Path(gm)

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

    guideline_mapping = args.guideline_mapping
    if guideline_mapping is None:
        gm = os.environ.get("LABEL_STUDIO_GUIDELINE_MAPPING", "").strip()
        guideline_mapping = Path(gm) if gm else None

    normalize_frame_names = not args.no_normalize_frame_names
    if os.environ.get("LABEL_STUDIO_NO_NORMALIZE_FRAME_NAMES", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        normalize_frame_names = False

    if args.guideline_only:
        if guideline_mapping is None:
            print(
                "Guideline-only mode requires --guideline-mapping (or LABEL_STUDIO_GUIDELINE_MAPPING).\n",
                file=sys.stderr,
            )
            sys.exit(2)
        run_guideline_only(
            project_id=project_id,
            output_parent=output_parent,
            guideline_mapping=guideline_mapping,
            guideline_subdir=args.guideline_subdir,
            run_dir=args.from_run_dir,
        )
        return

    ls, base, cf = _connect_label_studio()
    _run_export(
        ls,
        base,
        cf,
        project_id,
        export_title,
        output_parent,
        extract=extract,
        normalize_frame_names=normalize_frame_names,
        guideline_mapping=guideline_mapping,
        guideline_subdir=args.guideline_subdir,
        tasks_fallback=tasks_fallback,
    )
