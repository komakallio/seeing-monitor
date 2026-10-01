"""The `store` case: result inserts, and the append of per-frame metrics to a segment file.

**Result rows.** `Store.write` adds one record in one transaction, as `core` writes a seeing
window or a survey result. The case writes rows of a realistic seeing window (with the 24-bin
motion spectrum) to a temporary database, and it reports the time of one row, the sustained rate in
rows per second, and the time of a row in a batch of 100 (`Store.write_many`). The architecture
expects about 4 rows a minute, so any figure above a few rows per second leaves the store idle.

**Segment append.** `SegmentWriter.write_metrics` appends the rows of the `frame` record (44 bytes
each) to a segment file. `core` calls it once per second of frame time, so the case appends batches
of one second at 98 frames per second. It reports the time per frame without an fsync, which is
processor time, and the time per frame with the default fsync interval, which includes the waits of
the disk. The writer runs on a virtual clock, so the fsync interval counts frame time.

Both parts use a temporary folder that the case deletes. The disk of the dev machine is far faster
than the SD card of a Pi, so the figures with an fsync are a floor and not an estimate.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from seeingmon.perf.registry import REGISTRY, CaseContext
from seeingmon.perf.report import Measurement
from seeingmon.perf.timing import TimingStats

if TYPE_CHECKING:
    from seeingmon.records.base import Record

_NS_PER_S = 1_000_000_000
_EPOCH_NS = 1_800_000_000 * _NS_PER_S
_RATE_HZ = 98
_BATCH_ROWS = 100
_SPECTRUM_BINS = 24


def window_record() -> Record:
    """A seeing-window record with the fields and the sizes of a real window."""
    from seeingmon.records.samples import sample_record

    frequencies = [0.5 * 1.25**index for index in range(_SPECTRUM_BINS)]
    spectrum = [0.3 / (1.0 + (f / 8.0) ** (17.0 / 3.0)) for f in frequencies]
    return sample_record(
        "seeing_window",
        station_id="perf",
        profile_id="asi294mm-gs250",
        provenance={"algo": "fast-1", "assumptions": "L0=20 m; wind=10 m/s; detrend=2"},
        duration_s=60.0,
        stream_id=1,
        readout_mode="bin1",
        exposure_us=2000,
        gain=0,
        n_frames=5880,
        n_dropped=3,
        valid_fraction=0.99,
        frame_rate_hz=98.0,
        image_motion_rms_x_arcsec=0.41,
        image_motion_rms_y_arcsec=0.39,
        seeing_fwhm_arcsec=1.1,
        r0_cm=9.2,
        seeing_fwhm_structure_arcsec=1.2,
        r0_structure_cm=8.7,
        scintillation_index=0.02,
        centroid_noise_px=0.03,
        motion_psd_freq_hz=frequencies,
        motion_psd_x_arcsec2_per_hz=spectrum,
        motion_psd_y_arcsec2_per_hz=spectrum,
        vibration_lines_hz=[],
        flags=[],
    )


def time_results(ctx: CaseContext, folder: Path) -> list[Measurement]:
    from seeingmon.store.db import Store

    rows = ctx.pick(3000, 60)
    batches = ctx.pick(30, 2)
    template = window_record()
    timer = time.perf_counter_ns
    with Store.open(folder / "results.sqlite") as store:
        singles: list[float] = []
        began = timer()
        for index in range(rows):
            record = template.model_copy(update={"t_utc_ns": _EPOCH_NS + index * 60 * _NS_PER_S})
            start = timer()
            store.write(record)
            singles.append(timer() - start)
        sustained_s = (timer() - began) / _NS_PER_S
        per_batch: list[float] = []
        origin = _EPOCH_NS + rows * 60 * _NS_PER_S
        for batch in range(batches):
            records = [
                template.model_copy(update={"t_utc_ns": origin + (batch * _BATCH_ROWS + i) * 10**9})
                for i in range(_BATCH_ROWS)
            ]
            start = timer()
            store.write_many(records)
            per_batch.append((timer() - start) / _BATCH_ROWS)
    single = TimingStats.from_samples(singles).scaled(1e-3)
    many = TimingStats.from_samples(per_batch).scaled(1e-3)
    return [
        Measurement(
            "results.write",
            "us/row",
            single.median,
            single,
            "interpreter",
            {
                "rows": rows,
                "rows_per_s": round(rows / sustained_s, 1),
                "transaction": "one per row",
            },
        ),
        Measurement(
            "results.write_many",
            "us/row",
            many.median,
            many,
            "interpreter",
            {"rows": batches * _BATCH_ROWS, "batch": _BATCH_ROWS},
        ),
    ]


def time_segments(ctx: CaseContext, folder: Path, *, fsync: bool) -> TimingStats:
    """Append one second of rows per call, and return the time per frame in microseconds."""
    import numpy as np

    from seeingmon.clock import VirtualClock
    from seeingmon.records.segments import segment_dtype
    from seeingmon.store.config import SegmentsConfig
    from seeingmon.store.segments import SegmentWriter

    seconds = ctx.pick(300, 4)
    config = SegmentsConfig() if fsync else SegmentsConfig(fsync_interval_s=1e9)
    dtype = segment_dtype("frame")
    batch = np.zeros(_RATE_HZ, dtype=dtype)
    offsets = np.arange(_RATE_HZ, dtype=np.int64) * (_NS_PER_S // _RATE_HZ)
    clock = VirtualClock()
    per_call: list[float] = []
    timer = time.perf_counter_ns
    root = folder / ("with-fsync" if fsync else "no-fsync")
    with SegmentWriter(
        root, clock, station_id="perf", profile_id="asi294mm", config=config
    ) as writer:
        for second in range(seconds):
            batch["t_utc_ns"] = _EPOCH_NS + second * _NS_PER_S + offsets
            batch["seq"] = np.arange(_RATE_HZ) + second * _RATE_HZ
            start = timer()
            writer.write_metrics(1, batch)
            per_call.append((timer() - start) / _RATE_HZ)
            clock.advance(1.0)
    return TimingStats.from_samples(per_call).scaled(1e-3)


@REGISTRY.case("store", summary="Result inserts per second and the metrics segment append")
def store(ctx: CaseContext) -> list[Measurement]:
    ctx.mark_baseline()
    with tempfile.TemporaryDirectory(prefix="smon-perf-", ignore_cleanup_errors=True) as name:
        folder = Path(name)
        measurements = time_results(ctx, folder)
        plain = time_segments(ctx, folder, fsync=False)
        synced = time_segments(ctx, folder, fsync=True)
    measurements.extend(
        [
            Measurement(
                "segments.append",
                "us/frame",
                plain.median,
                plain,
                "interpreter",
                {"rate_hz": _RATE_HZ, "batch_rows": _RATE_HZ, "fsync": "none"},
            ),
            Measurement(
                "segments.append_fsync",
                "us/frame",
                synced.mean,
                synced,
                "none",
                {"rate_hz": _RATE_HZ, "batch_rows": _RATE_HZ, "fsync": "every 60 s of frame time"},
            ),
            Measurement(
                "segments.share",
                "percent",
                plain.median * _RATE_HZ / 1e4,
                None,
                "interpreter",
                {"rate_hz": _RATE_HZ},
            ),
        ]
    )
    ctx.note(
        "The segment append is timed per call of one second of rows (98 frames), and the figure is "
        "per frame. The value of the fsync figure is the mean, because the fsync is one call in "
        "sixty. The temporary folder is deleted."
    )
    return measurements
