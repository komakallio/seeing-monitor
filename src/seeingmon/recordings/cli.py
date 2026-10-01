"""The `seeingmon recordings` commands.

`seeingmon recordings info PATH` prints the geometry, the frame count, the pixel depth, the
duration, and the spread of the frame interval of a SER file. It reads the file only. It never
prints the observer, instrument, or telescope text of the header, the path of the file, or
anything from a sidecar.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from seeingmon.cli import CliError, Subparsers, add_command

# A gap is an interval longer than this many median intervals (the replay driver uses 1.5).
GAP_FACTOR = 1.5


def register(subparsers: Subparsers) -> None:
    parser = add_command(
        subparsers,
        "recordings",
        help="Inspect SER recordings.",
        handler=_missing_subcommand,
    )
    commands = parser.add_subparsers(
        dest="recordings_command", metavar="<subcommand>", required=True
    )
    info = commands.add_parser(
        "info",
        help="Print the geometry, frame count, duration, and frame timing of a SER file.",
        description=(
            "Print the geometry, frame count, pixel depth, duration, and the median and spread "
            "of the frame interval of a SER file. The output omits the header text and the path."
        ),
    )
    info.add_argument("path", type=Path, help="the SER file")
    info.set_defaults(handler=_info)


def _missing_subcommand(args: argparse.Namespace) -> int:
    # `required=True` makes argparse reject a missing subcommand before this runs.
    raise CliError("choose a subcommand: info", exit_code=2)


def _info(args: argparse.Namespace) -> int:
    import numpy as np

    from seeingmon.recordings.ser import SerError, SerFile

    try:
        with SerFile(args.path) as ser:
            header = ser.header
            stamps = ser.timestamps_utc_ns()
            rows = [
                ("frames", str(ser.frame_count)),
                ("size", f"{header.width} x {header.height} pixels"),
                ("pixel depth", f"{header.pixel_depth} bits"),
                ("color", _color_name(header.color_id.name)),
            ]
    except SerError as exc:
        raise CliError(str(exc)) from None
    if stamps is None:
        rows.append(("timestamps", "none"))
    else:
        rows.append(("timestamps", "yes"))
        duration_s = float(stamps[-1] - stamps[0]) / 1e9
        rows.append(("duration", f"{duration_s:.3f} s from the first to the last timestamp"))
        if len(stamps) > 1:
            intervals_ms = np.diff(stamps).astype(np.float64) / 1e6
            median = float(np.median(intervals_ms))
            spread = float(np.std(intervals_ms))
            low, high = float(intervals_ms.min()), float(intervals_ms.max())
            rows.append(
                (
                    "frame interval",
                    f"median {median:.3f} ms, standard deviation {spread:.3f} ms, "
                    f"range {low:.3f} to {high:.3f} ms",
                )
            )
            if duration_s > 0:
                rows.append(
                    ("frame rate", f"{(len(stamps) - 1) / duration_s:.2f} frames per second")
                )
            gaps = int(np.count_nonzero(intervals_ms > GAP_FACTOR * median))
            rows.append(("long intervals", f"{gaps} longer than {GAP_FACTOR} median intervals"))
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label.ljust(width)}  {value}")
    return 0


def _color_name(name: str) -> str:
    if name == "MONO":
        return "mono"
    if name in ("RGB", "BGR"):
        return name
    return f"raw mosaic ({name.removeprefix('BAYER_')})"
