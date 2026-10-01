"""Synthetic records and image files for the web tests.

Every value is made up. The seed holds a clearly synthetic station and no site. The times sit on
round UTC minutes so that a test can compute a bucket by hand.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from seeingmon.clock import NS_PER_S
from seeingmon.records.samples import sample_record
from seeingmon.store.db import Store
from seeingmon.store.layout import DataLayout
from tests.services.web.helpers import tiny_jpeg

STATION = "test-station"
PROFILE = "test-profile"
BASE_NS = int(datetime(2026, 10, 1, 2, 0, tzinfo=UTC).timestamp()) * NS_PER_S  # 02:00:00Z
NOW_NS = BASE_NS + 3600 * NS_PER_S  # 03:00:00Z
MINUTE_NS = 60 * NS_PER_S

# (minute after BASE, seeing, r0, flags, quality for the missing values)
WINDOWS: list[tuple[int, float | None, float | None, list[str], dict[str, str] | None]] = [
    (0, 1.0, 10.1, [], None),
    (1, 1.2, 8.4, ["cloud"], None),
    (2, None, None, ["cloud", "vibration"], {"seeing_fwhm_arcsec": "too few usable frames"}),
    (3, 1.4, 7.2, [], None),
    (4, 1.6, 6.3, ["twilight"], None),
    (5, 1.8, 5.6, [], None),
    (10, 2.0, 5.05, [], None),
    (11, None, None, [], None),
    (20, None, None, [], {"seeing_fwhm_arcsec": "the star is not detected"}),
    (21, None, None, [], None),
]


def at(minutes: float) -> int:
    """The time `minutes` after the base time, in nanoseconds."""
    return BASE_NS + round(minutes * MINUTE_NS)


def common(t_utc_ns: int, **fields: Any) -> dict[str, Any]:
    return {"station_id": STATION, "profile_id": PROFILE, "t_utc_ns": t_utc_ns, **fields}


def seed_records(store: Store) -> None:
    """Write the seeing windows, a few survey records, a health record, and some events."""
    for minute, seeing, r0, flags, quality in WINDOWS:
        store.write(
            sample_record(
                "seeing_window",
                **common(
                    at(minute),
                    seeing_fwhm_arcsec=seeing,
                    r0_cm=r0,
                    flags=flags,
                    quality=quality,
                    n_frames=5400,
                    n_dropped=minute % 3,
                    valid_fraction=0.99,
                    scintillation_index=0.002 * (minute + 1),
                    motion_psd_freq_hz=[1.0, 2.0, 3.0],
                    motion_psd_x_arcsec2_per_hz=[0.1, 0.2, 0.3],
                    motion_psd_y_arcsec2_per_hz=[0.3, 0.2, 0.1],
                    zenith_angle_deg=33.0,
                    stream_id=7,
                    gain=0,
                    exposure_us=2000,
                    readout_mode="bin1",
                ),
            )
        )
    for minute, mag, cloud in [(0, 21.0, 0.0), (3, 20.8, 0.2), (6, 20.2, 0.6)]:
        store.write(
            sample_record(
                "sky_quality",
                **common(at(minute), sky_mag_arcsec2=mag, cloud_fraction=cloud, n_stars_used=50),
            )
        )
    for minute, offset in [(0, 0.1), (3, 0.3), (6, 0.5)]:
        store.write(
            sample_record(
                "pointing",
                **common(
                    at(minute),
                    offset_arcmin=offset,
                    roll_deg=10.0 + minute,
                    attitude=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    n_matched=40,
                    readout_mode="bin2",
                    solver="solver-a",
                ),
            )
        )
    store.write(
        sample_record(
            "health",
            **common(
                NOW_NS - 30 * NS_PER_S,
                state="auto",
                degraded=False,
                components={"acquire": "ok", "core": "ok", "scheduler": "ok", "camera": "ok"},
                dark_due=False,
                free_space_gb=12.5,
            ),
        )
    )
    events = [
        (0, "info", "scheduler.state_change", "The scheduler entered the auto state.", None),
        (1, "warning", "scheduler.cloud", "Clouds crossed the star.", {"active": True}),
        (2, "error", "scheduler.fault", "A camera error ended the activity.", {"failures": 1}),
        (3, "info", "storage.recovered", "Repaired a file.", {"where": "the data folder"}),
        (4, "warning", "scheduler.cloud", "The clouds went away.", {"active": False}),
    ]
    for minute, level, kind, message, detail in events:
        store.write(
            sample_record(
                "event",
                **common(at(minute), level=level, kind=kind, message=message, detail=detail),
            )
        )


def write_preview(
    layout: DataLayout, stamp: str, *, kind: str = "preview", shade: int = 50
) -> Path:
    """Write a small JPEG as the preview with this stamp (`YYYYMMDDTHHMMSS.mmmZ`)."""
    folder = layout.previews_dir / stamp[0:4] / stamp[4:6] / stamp[6:8]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{kind}-{stamp}.jpg"
    path.write_bytes(tiny_jpeg(shade))
    return path


def write_fits(layout: DataLayout, stamp: str, size: int = 2880 * 2) -> Path:
    """Write a file with the FITS name of this stamp. The web process only serves the bytes."""
    folder = layout.survey_dir / stamp[0:4] / stamp[4:6] / stamp[6:8]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}.fits"
    path.write_bytes(b"SIMPLE  =                    T".ljust(80) + b" " * (size - 80))
    return path
