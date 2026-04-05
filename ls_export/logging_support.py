import json
import tempfile
import time
from pathlib import Path

from label_studio_sdk.core.api_error import ApiError


def _repo_root() -> Path:
    """Backend-Inference root (parent of ls_export/)."""
    return Path(__file__).resolve().parent.parent


def _debug_log_path() -> Path:
    """Prefer repo .cursor/; fall back to system temp if that dir is not writable."""
    primary = _repo_root() / ".cursor" / "debug-988c41.log"
    try:
        primary.parent.mkdir(parents=True, exist_ok=True)
        return primary
    except OSError:
        return Path(tempfile.gettempdir()) / "cardiovis-debug-988c41.ndjson"


def _agent_log(message: str, data: dict, hypothesis_id: str) -> None:
    line = json.dumps(
        {
            "sessionId": "988c41",
            "timestamp": int(time.time() * 1000),
            "location": "ls_export",
            "message": message,
            "data": data,
            "hypothesisId": hypothesis_id,
        },
        ensure_ascii=False,
    )
    try:
        with open(_debug_log_path(), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _log_api_error(name: str, err: ApiError, hypothesis_id: str) -> None:
    body = err.body
    preview = ""
    if isinstance(body, (dict, list)):
        preview = json.dumps(body)[:800]
    elif body is not None:
        preview = str(body)[:800]
    _agent_log(
        name,
        {"status_code": err.status_code, "body_preview": preview.replace("\n", "\\n")},
        hypothesis_id,
    )
