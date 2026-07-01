"""Progress logging for long-running Label Studio export polls."""

import sys
import time


def poll_export_snapshot(
    get_status,
    *,
    label: str = "export snapshot",
    interval_sec: float = 1.0,
    log_every_sec: float = 15.0,
) -> str:
    """Call ``get_status()`` until status is ``completed`` or ``failed``. Returns final status."""
    started = time.time()
    last_log = started
    while True:
        st = get_status() or ""
        now = time.time()
        if now - last_log >= log_every_sec:
            print(
                f"  Waiting for {label}… status={st!r} "
                f"({int(now - started)}s elapsed)",
                file=sys.stderr,
            )
            last_log = now
        if st == "failed":
            return st
        if st == "completed":
            return st
        time.sleep(interval_sec)
