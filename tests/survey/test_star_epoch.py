"""The nightly star summary: the running means, the night boundary, and the packed record."""

from __future__ import annotations

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.records.survey import StarEpochRecord, star_rows
from seeingmon.survey.star_epoch import STAR_EPOCH_COLUMNS, FrameStars, NightAccumulator

EVENING = iso_to_utc_ns("2026-10-01T19:00:00Z")
STEP_NS = 180 * NS_PER_S


def accumulator(n_catalog: int = 50, *, min_frames: int = 3) -> NightAccumulator:
    return NightAccumulator(
        n_catalog,
        station_id="test-station",
        profile_id="profile-1",
        provenance={"algo": "sky-1", "catalog": "abcd1234"},
        min_frames=min_frames,
    )


def frame(rows: list[int], dx: list[float], dy: list[float], mag: list[float]) -> FrameStars:
    return FrameStars.build(rows, dx, dy, mag)


def test_the_stars_of_a_frame_survive_the_trip_between_processes() -> None:
    stars = frame([3, 17, 40], [0.1, -0.2, 0.3], [0.0, 0.5, -0.5], [11.2, 12.4, 9.9])
    back = FrameStars.from_bytes(stars.to_bytes())
    np.testing.assert_array_equal(back.cat_row, stars.cat_row)
    np.testing.assert_array_equal(back.dx_arcsec, stars.dx_arcsec)
    np.testing.assert_array_equal(back.dy_arcsec, stars.dy_arcsec)
    np.testing.assert_array_equal(back.mag, stars.mag)
    assert len(back) == 3
    assert len(FrameStars.from_bytes(FrameStars.empty().to_bytes())) == 0
    with pytest.raises(ValueError, match="wrong size"):
        FrameStars.from_bytes(b"123")


def test_a_night_gives_the_mean_offset_the_mean_magnitude_and_the_scatter() -> None:
    rng = np.random.default_rng(1)
    n_frames = 12
    rows = [4, 9, 20]
    dx = rng.normal(0.2, 0.3, (n_frames, 3)).astype(np.float32)
    dy = rng.normal(-0.1, 0.3, (n_frames, 3)).astype(np.float32)
    mag = (np.array([10.0, 11.5, 12.2]) + rng.normal(0.0, 0.02, (n_frames, 3))).astype(np.float32)
    night = accumulator()
    for index in range(n_frames):
        assert (
            night.add(EVENING + index * STEP_NS, frame(rows, dx[index], dy[index], mag[index]))
            is None
        )
    record = night.flush()
    assert isinstance(record, StarEpochRecord)
    assert record.night == "2026-10-01"
    assert record.t_utc_ns == EVENING  # the first frame
    assert record.n_stars == 3
    assert record.n_frames == n_frames
    assert record.columns == STAR_EPOCH_COLUMNS
    assert record.provenance == {"algo": "sky-1", "catalog": "abcd1234"}
    table = star_rows(record)
    assert table.shape == (3, 6)
    np.testing.assert_array_equal(table[:, 0], rows)
    np.testing.assert_allclose(table[:, 1], dx.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(table[:, 2], dy.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(table[:, 3], mag.mean(axis=0), atol=1e-4)
    np.testing.assert_allclose(table[:, 4], mag.std(axis=0, ddof=1), atol=1e-4)
    np.testing.assert_array_equal(table[:, 5], [n_frames] * 3)


def test_a_star_needs_a_few_frames_to_appear() -> None:
    night = accumulator(min_frames=3)
    for index in range(5):
        rows = [1, 2] if index < 2 else [1]  # star 2 is in two frames only
        night.add(
            EVENING + index * STEP_NS,
            frame(rows, [0.0] * len(rows), [0.0] * len(rows), [10.0] * len(rows)),
        )
    record = night.flush()
    assert record is not None
    assert star_rows(record)[:, 0].tolist() == [1.0]
    assert record.n_frames == 5


def test_a_night_with_no_qualifying_star_gives_no_record() -> None:
    night = accumulator(min_frames=3)
    night.add(EVENING, frame([1], [0.0], [0.0], [10.0]))
    assert night.flush() is None
    assert night.night is None  # the night is closed anyway
    assert accumulator().flush() is None


def test_the_next_night_closes_the_last_one() -> None:
    night = accumulator(min_frames=2)
    first_night = [EVENING + i * STEP_NS for i in range(4)]
    second_night = [iso_to_utc_ns("2026-10-02T19:00:00Z") + i * STEP_NS for i in range(3)]
    closed = None
    for t in first_night:
        assert night.add(t, frame([5], [0.1], [0.2], [11.0])) is None
    assert night.night == "2026-10-01"
    assert night.frames == 4
    closed = night.add(second_night[0], frame([5], [0.4], [0.5], [11.1]))
    assert closed is not None
    assert (closed.night, closed.n_frames, closed.t_utc_ns) == ("2026-10-01", 4, first_night[0])
    assert star_rows(closed)[0, 1] == pytest.approx(0.1)  # the new frame is not in it
    assert night.night == "2026-10-02"
    assert night.frames == 1
    for t in second_night[1:]:
        assert night.add(t, frame([5], [0.4], [0.5], [11.1])) is None
    last = night.flush()
    assert last is not None
    assert (last.night, last.n_frames, last.t_utc_ns) == ("2026-10-02", 3, second_night[0])
    assert star_rows(last)[0, 1] == pytest.approx(0.4, abs=1e-6)


def test_the_midnight_of_utc_does_not_split_a_night() -> None:
    night = accumulator(min_frames=1)
    night.add(iso_to_utc_ns("2026-10-01T23:50:00Z"), frame([1], [0.0], [0.0], [10.0]))
    assert (
        night.add(iso_to_utc_ns("2026-10-02T00:10:00Z"), frame([1], [0.0], [0.0], [10.0])) is None
    )
    record = night.flush()
    assert record is not None
    assert record.night == "2026-10-01"
    assert record.n_frames == 2


def test_a_star_counts_once_per_frame_and_a_bad_row_is_ignored() -> None:
    night = accumulator(n_catalog=10, min_frames=1)
    night.add(
        EVENING,
        frame(
            [3, 3, 99, -1, 4], [1.0, 9.0, 5.0, 5.0, 2.0], [0.0] * 5, [10.0, 20.0, 1.0, 1.0, 12.0]
        ),
    )
    record = night.flush()
    assert record is not None
    table = star_rows(record)
    assert table[:, 0].tolist() == [3.0, 4.0]
    assert table[0, 1] == pytest.approx(1.0)  # the first of the two detections of row 3
    assert table[:, 5].tolist() == [1.0, 1.0]
    assert table[:, 4].tolist() == [0.0, 0.0]  # a single frame has no scatter


def test_a_frame_with_no_stars_still_counts_as_a_frame() -> None:
    night = accumulator(min_frames=1)
    night.add(EVENING, frame([1], [0.0], [0.0], [10.0]))
    night.add(EVENING + STEP_NS, FrameStars.empty())
    record = night.flush()
    assert record is not None
    assert record.n_frames == 2


def test_a_long_night_keeps_the_running_mean_exact() -> None:
    rng = np.random.default_rng(2)
    night = accumulator(n_catalog=5, min_frames=1)
    values = rng.normal(11.0, 0.05, 400)
    for index, value in enumerate(values):
        night.add(EVENING + index * 10 * NS_PER_S, frame([2], [0.0], [0.0], [float(value)]))
    record = night.flush()
    assert record is not None
    assert record.n_frames == 400
    table = star_rows(record)
    assert table[0, 3] == pytest.approx(np.float32(values).mean(), abs=2e-4)
    assert table[0, 4] == pytest.approx(np.float32(values).std(ddof=1), abs=2e-4)


def test_the_record_for_1500_stars_is_the_size_that_the_architecture_budgets() -> None:
    n = 1500
    night = accumulator(n_catalog=n, min_frames=1)
    rows = list(range(n))
    night.add(EVENING, frame(rows, [0.0] * n, [0.0] * n, [11.0] * n))
    record = night.flush()
    assert record is not None
    assert len(record.data) == n * 24  # six float32 columns
    assert len(record.data) * 365 / 1e6 == pytest.approx(13.1, abs=0.1)  # MB per year


def test_the_accumulator_refuses_nonsense() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        NightAccumulator(0, station_id="s", profile_id="p", provenance={"algo": "x"}, min_frames=1)
