"""Build guideline_seq/ from an existing extracted/ folder (no Label Studio re-download)."""

from __future__ import annotations

import sys
from pathlib import Path

from ls_export.guideline_export import write_guideline_seq_from_extract


def _resolve_run_dir(output_parent: Path, project_id: int, run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir.resolve()
    candidates = [
        output_parent / f"export-{project_id}-tasks-fallback",
        output_parent / f"export-{project_id}-manual",
    ]
    for c in output_parent.glob(f"export-{project_id}-*"):
        if (c / "extracted" / "images").is_dir() and (c / "extracted" / "labels").is_dir():
            candidates.insert(0, c)
    for c in candidates:
        if (c / "extracted" / "images").is_dir():
            return c.resolve()
    return (output_parent / f"export-{project_id}-tasks-fallback").resolve()


def run_guideline_only(
    *,
    project_id: int,
    output_parent: Path,
    guideline_mapping: Path,
    guideline_subdir: str = "guideline_seq",
    run_dir: Path | None = None,
) -> Path:
    mapping = guideline_mapping.resolve()
    if not mapping.is_file():
        print(f"Guideline mapping file not found: {mapping}\n", file=sys.stderr)
        sys.exit(1)

    root = _resolve_run_dir(output_parent, project_id, run_dir)
    extracted = root / "extracted"
    if not (extracted / "labels").is_dir():
        print(
            f"No extracted labels at {extracted / 'labels'}.\n"
            f"Run a download first or pass --from-run-dir to the export folder.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Guideline-only: reading {extracted.resolve()}\n"
        f"Mapping: {mapping}\n",
        file=sys.stderr,
    )
    gdir = write_guideline_seq_from_extract(
        extracted,
        mapping,
        root,
        subdir=guideline_subdir,
    )
    json_dir = gdir / "json"
    n_json = len(list(json_dir.glob("ann_*.json"))) if json_dir.is_dir() else 0
    print(
        f"Guideline export OK.\n"
        f"JSON: {json_dir.resolve()} ({n_json} ann_*.json files)\n"
        f"Root: {gdir.resolve()}",
    )
    return gdir
