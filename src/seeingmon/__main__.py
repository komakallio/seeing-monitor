"""Run the command-line interface with `python -m seeingmon`."""

from __future__ import annotations

import sys

from seeingmon.cli import main

if __name__ == "__main__":
    sys.exit(main())
