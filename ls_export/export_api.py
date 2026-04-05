import json
import os
import sys
import time

from label_studio_sdk import LabelStudio
from label_studio_sdk.core.api_error import ApiError
from label_studio_sdk.types.lse_task_filter_options_request import (
    LseTaskFilterOptionsRequest,
)
from label_studio_sdk.types.serialization_option_request import SerializationOptionRequest
from label_studio_sdk.types.serialization_options_request import SerializationOptionsRequest

from ls_export.logging_support import _agent_log, _log_api_error


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
    ls: LabelStudio,
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
    _wait_for_converted_format(ls, export_id, project_id, "JSON")


def _download_json_export_bytes(ls: LabelStudio, project_id: int, export_id: int) -> bytes:
    return b"".join(ls.projects.exports.download(project_id, export_id, export_type="JSON"))
