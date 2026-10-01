"""A stand-in for `astap`, for tests. It is a script, not a test module.

The environment variable `SHIM_SPEC` names a JSON file that says what the shim does:

- `exit_code`: exit with this code and a message on stderr (default 0).
- `sleep_s`: wait this long first.
- `ini`: the `KEY=VALUE` lines to write to `<image>.ini` (default: none).
- `wcs`: the cards to write to `<image>.wcs` (default: none).

The shim records its arguments and what it saw in the image (its shape, and the positions of
its five brightest peaks) in the JSON file that `SHIM_LOG` names.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from seeingmon.solvers import fitsio


def peaks(image: np.ndarray, count: int = 5) -> list[list[float]]:
    """The positions (x, y) of the brightest pixels, with 6 pixels kept between them."""
    work = image.astype(np.float64).copy()
    found: list[list[float]] = []
    for _ in range(count):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        found.append([float(x), float(y)])
        work[max(y - 6, 0) : y + 7, max(x - 6, 0) : x + 7] = 0.0
    return found


def main() -> int:
    spec = json.loads(Path(os.environ["SHIM_SPEC"]).read_text(encoding="utf-8"))
    arguments = sys.argv[1:]
    options: dict[str, str] = {}
    flags: list[str] = []
    index = 0
    while index < len(arguments):
        item = arguments[index]
        if item in ("-wcs",):
            flags.append(item)
            index += 1
        else:
            options[item] = arguments[index + 1]
            index += 2
    image_path = Path(options["-f"])
    image = fitsio.read_fits(image_path)[0].data["image"]
    Path(os.environ["SHIM_LOG"]).write_text(
        json.dumps(
            {
                "arguments": arguments,
                "options": options,
                "flags": flags,
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "peaks": peaks(image),
                "median": float(np.median(image)),
                "maximum": float(image.max()),
            }
        ),
        encoding="utf-8",
    )
    time.sleep(float(spec.get("sleep_s", 0.0)))
    if spec.get("exit_code", 0) != 0:
        print(spec.get("message", "the shim failed on purpose"), file=sys.stderr)
        return int(spec["exit_code"])
    if "ini" in spec:
        image_path.with_suffix(".ini").write_text("\n".join(spec["ini"]) + "\n", encoding="utf-8")
    if "wcs" in spec:
        cards: fitsio.Header = {"WCSAXES": 2, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"}
        cards.update(spec["wcs"])
        image_path.with_suffix(".wcs").write_bytes(fitsio.primary_bytes(cards))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
