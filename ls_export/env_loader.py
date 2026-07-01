"""Load optional ``.env`` from the repo root (never committed)."""

from pathlib import Path


def load_repo_dotenv() -> bool:
    """Load Backend-Inference/.env if present. Returns True when a file was loaded."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        return True
    return False
