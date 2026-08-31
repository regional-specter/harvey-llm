#!/usr/bin/env python3
"""Launch the Harvey Specter chat TUI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harvey_chat.app import run

if __name__ == "__main__":
    run()
