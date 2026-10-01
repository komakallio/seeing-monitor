"""A stand-in for `build-astrometry-index`, for tests. It is a script, not a test module.

It accepts the options that the catalog build passes, reads the input FITS table, writes a small
FITS file to the `-o` path, and records what it saw in `<output>.json`. The environment variable
`SHIM_FAIL` makes it exit with an error instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from seeingmon.solvers import fitsio


def main() -> int:
    parser = argparse.ArgumentParser()
    for flag in ("-i", "-o", "-P", "-S", "-A", "-D", "-I"):
        parser.add_argument(flag, required=True)
    parser.add_argument("-E", action="store_true")
    args = parser.parse_args()
    if os.environ.get("SHIM_FAIL"):
        print("the shim was told to fail", file=sys.stderr)
        return 3
    table = fitsio.read_table(args.i)
    output = Path(args.o)
    fitsio.write_table(output, {"STARS": np.array([len(table["RA"])], dtype=np.int64)})
    output.with_name(output.name + ".json").write_text(
        json.dumps(
            {
                "preset": args.P,
                "sort": args.S,
                "ra": args.A,
                "dec": args.D,
                "id": args.I,
                "scan": args.E,
                "columns": sorted(table),
                "stars": len(table["RA"]),
                "ra_head": table["RA"][:3].tolist(),
                "mag_head": table["MAG"][:3].tolist(),
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
