"""A stand-in for `solve-field`, for tests. It is a script, not a test module.

The environment variable `SHIM_SPEC` names a JSON file that says what the shim does:

- `exit_code`: exit with this code and a message on stderr (default 0).
- `sleep_s`: wait this long before anything else.
- `solved`: write the solution files (default true).
- `wcs`: the cards of the `.wcs` file (default: a solution centered on the frame).
- `corr_rows`: how many of the brightest stars go into the `.corr` file (default 5).
- `corr_offset_arcsec`: the declination offset between the field and the index positions.
- `garbage_wcs`: write a `.wcs` file that is not FITS.

The shim records its arguments, the backend configuration, and what it read from the star list
in the JSON file that `SHIM_LOG` names.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from seeingmon.solvers import fitsio


def main() -> int:
    spec = json.loads(Path(os.environ["SHIM_SPEC"]).read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    for option in (
        "--config",
        "--dir",
        "--out",
        "--width",
        "--height",
        "--x-column",
        "--y-column",
        "--sort-column",
        "--scale-units",
        "--scale-low",
        "--scale-high",
        "--cpulimit",
        "--ra",
        "--dec",
        "--radius",
        "--uniformize",
    ):
        parser.add_argument(option)
    for flag in ("--overwrite", "--no-plots", "--no-remove-lines", "--crpix-center"):
        parser.add_argument(flag, action="store_true")
    parser.add_argument("input")
    args, extra = parser.parse_known_args()

    hdus = fitsio.read_fits(args.input)
    table = hdus[1].data
    Path(os.environ["SHIM_LOG"]).write_text(
        json.dumps(
            {
                "arguments": sys.argv[1:],
                "parsed": vars(args),
                "extra": extra,
                "config": Path(args.config).read_text(encoding="ascii"),
                "n_rows": len(table["X"]),
                "x_head": table["X"][:3].tolist(),
                "y_head": table["Y"][:3].tolist(),
                "flux_head": table["FLUX"][:3].tolist(),
                "flux_sorted_descending": bool(np.all(np.diff(table["FLUX"]) <= 0)),
                "imagew": hdus[1].header.get("IMAGEW"),
                "imageh": hdus[1].header.get("IMAGEH"),
            }
        ),
        encoding="utf-8",
    )
    time.sleep(float(spec.get("sleep_s", 0.0)))
    if spec.get("exit_code", 0) != 0:
        print(spec.get("message", "the shim failed on purpose"), file=sys.stderr)
        return int(spec["exit_code"])
    if not spec.get("solved", True):
        return 0

    base = Path(args.dir) / args.out
    if spec.get("garbage_wcs"):
        base.with_suffix(".wcs").write_text("this is not a FITS header", encoding="ascii")
    else:
        cards: fitsio.Header = {
            "WCSAXES": 2,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "IMAGEW": int(args.width),
            "IMAGEH": int(args.height),
        }
        cards.update(spec["wcs"])
        base.with_suffix(".wcs").write_bytes(fitsio.primary_bytes(cards))
    base.with_suffix(".solved").write_bytes(b"\x01")
    rows = int(spec.get("corr_rows", 5))
    if rows:
        # The matched stars are the brightest rows of the star list.
        field_ra = np.linspace(10.0, 20.0, rows)
        field_dec = np.linspace(88.0, 89.0, rows)
        offset = float(spec.get("corr_offset_arcsec", 0.5)) / 3600.0
        fitsio.write_table(
            base.with_suffix(".corr"),
            {
                "field_x": table["X"][:rows],
                "field_y": table["Y"][:rows],
                "index_x": table["X"][:rows],
                "index_y": table["Y"][:rows],
                "field_ra": field_ra,
                "field_dec": field_dec,
                "index_ra": field_ra,
                "index_dec": field_dec + offset,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
