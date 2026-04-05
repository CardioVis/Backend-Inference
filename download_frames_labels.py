#!/usr/bin/env python3
"""CLI entrypoint: Label Studio → YOLO export download. Implementation: ``ls_export`` package."""

from ls_export.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
