# Label Studio export downloader

[`download_frames_labels.py`](download_frames_labels.py) is a thin wrapper; the implementation lives in the [`ls_export/`](ls_export/) package (`client`, `export_api`, `yolo_patch`, `cf_repair`, `run_export`, `guideline_export`, `export_normalize`, `cli`). The script creates a snapshot export of a [Label Studio](https://labelstud.io/) project, downloads it (default format **YOLO with images**), optionally patches YOLO `labels/*.txt` from the JSON export for better geometry coverage, and—when the server sits behind Cloudflare Access—can rebuild image bytes using your API token and service headers. After extraction it can **rename** `images/` and `labels/` to `frame_<digits>.*` (stripping `xxxxxxxx__xxxxxxxx-` prefixes) and optionally emit a **guideline-style** tree (`guideline_seq/`) compatible with [`guideline_seq_export/`](guideline_seq_export/). Progress and tips are printed to **stderr**; the final zip path and extracted folder are printed to **stdout**.

## Dependencies

Install the packages the script imports (your environment may not list all of them in this repo’s `requirements.txt`):

```bash
pip install label-studio-sdk requests numpy tqdm opencv-python-headless
```

Export downloads show a **tqdm** byte counter on stderr (YOLO zip and JSON used for label patch / repair). Set `LABEL_STUDIO_NO_TQDM=1` to disable.

## Required configuration

| Variable | Description |
|----------|-------------|
| `LABEL_STUDIO_API_KEY` or `LABEL_STUDIO_TOKEN` | Personal or legacy access token (Account → Access Token). Do not commit it. |

Optional:

| Variable | Description |
|----------|-------------|
| `LABEL_STUDIO_URL` | Base URL of your instance (default in script: `https://labeling.cardiovis.com/`). |

## Choosing a project

1. List projects your token can access (id and title, tab-separated):

   ```bash
   export LABEL_STUDIO_API_KEY="your_token"
   python download_frames_labels.py --list-projects
   ```

2. Run an export for a specific project:

   ```bash
   python download_frames_labels.py --project-id 42
   ```

CLI `--project-id` / `-p` overrides `LABEL_STUDIO_PROJECT_ID`. If both are omitted, the script falls back to `LABEL_STUDIO_PROJECT_ID` and then to a default project id of `13` (for backward compatibility). Prefer always passing `-p` or setting the env var explicitly.

You can also read the project id from the UI URL: `/projects/<id>/…`.

## Output layout

By default, files are written under **`label_studio_exports/`** (override with `-o` / `--output-dir` or `LABEL_STUDIO_OUTPUT_DIR`):

- `label_studio_exports/export-<project_id>-<export_id>/` — run folder
- `…/export-…-<TYPE>.zip` — downloaded archive (if Cloudflare image repair ran, the pre-repair zip is removed and only the `*_repaired.zip` is kept)
- `…/extracted/` — contents of the **final** zip after download, in-place label patch, and optional Cloudflare repair (extract runs last so this folder matches the zip on disk).
- `…/guideline_seq/` — optional; see [Guideline-style export](#guideline-style-export).

Skip unzipping with `--no-extract` or `LABEL_STUDIO_SKIP_EXTRACT=1`.

### Frame name normalization

By default, after extraction, files under `extracted/images/` and `extracted/labels/` whose names look like `xxxxxxxx__xxxxxxxx-frame_01234.ext` are renamed to `frame_01234.ext` / `frame_01234.txt`. If two files would collide, the script exits with an error.

Disable with `--no-normalize-frame-names` or `LABEL_STUDIO_NO_NORMALIZE_FRAME_NAMES=1`.

### Guideline-style export

For YOLO exports only, you can pass `--guideline-mapping path/to/mapping.json` (or `LABEL_STUDIO_GUIDELINE_MAPPING`) to write `export-…/guideline_seq/` with `frames/`, `masks/…`, and `json/ann_XXXXXX.json` using the same top-level schema as the sample under [`guideline_seq_export/json/`](guideline_seq_export/json/). This is an **MVP**: masks are filled from **YOLO boxes** (axis-aligned rectangles) using your class map; `centerline_segments` are derived from the **pericardium** mask (class **2** by default); `band_polygons_between_parallel_bounds` is `[]`, `parallel_lines_offset_cm` lists are empty, and band PNGs are all zeros.

**Mapping file** (JSON), required field `yolo_to_guideline`: map each YOLO class id (string keys) to guideline mask class `1`, `2`, or `3`:

```json
{
  "yolo_to_guideline": {
    "0": 1,
    "1": 2,
    "2": 3
  },
  "task": "my_export",
  "pericardium_class": 2,
  "frame_note": "optional",
  "predict_note": "optional",
  "params": { "OFFSET_CM": 1.2, "PIXELS_PER_CM": 20.0 }
}
```

Optional keys: `class_names`, `mask_notes`, `source_path_template` (e.g. `"/data/{stem}.jpg"`). Output subdirectory name defaults to `guideline_seq`; override with `--guideline-subdir`.

## Common environment variables

| Variable | Description |
|----------|-------------|
| `LABEL_STUDIO_EXPORT_TITLE` | Default title for the snapshot (overridden by `--export-title`). |
| `LABEL_STUDIO_EXPORT_TYPE` | e.g. `YOLO_WITH_IMAGES` (default). |
| `LABEL_STUDIO_EXPORT_VIEW_ID` | Data Manager tab id from the URL (`…/data?tab=14` → `14`) so the export matches that view. |
| `LABEL_STUDIO_EXPORT_ANNOTATED_ONLY` | Set to `1` to export only annotated tasks. |
| `LABEL_STUDIO_EXPORT_FINISHED_ONLY` | Set to `1` to export only finished tasks. |
| `LABEL_STUDIO_DOWNLOAD_RESOURCES` | Force whether the server embeds files in the export zip. |
| `LABEL_STUDIO_REPAIR_CF_IMAGES` | When behind Cloudflare, control client-side image repair into a `*_repaired.zip`. |
| `LABEL_STUDIO_GUIDELINE_MAPPING` | Path to JSON mapping for `guideline_seq/` export (same as `--guideline-mapping`). |
| `LABEL_STUDIO_NO_NORMALIZE_FRAME_NAMES` | Set to `1` to keep long LS-prefixed filenames under `extracted/`. |
| `CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET` | Service tokens when Label Studio is behind **Cloudflare Access** (see stderr hints from the script if `/api/version` returns HTML). |

## Examples

```bash
export LABEL_STUDIO_API_KEY="…"
python download_frames_labels.py --list-projects
python download_frames_labels.py -p 7 --export-title "Weekly backup"
python download_frames_labels.py -p 7 -o ./my_exports
python download_frames_labels.py -p 7 --no-extract
python download_frames_labels.py -p 13 --guideline-mapping ./my_guideline_map.json
```

Full CLI help:

```bash
python download_frames_labels.py --help
```
