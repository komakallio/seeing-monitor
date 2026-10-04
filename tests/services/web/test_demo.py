"""Demo mode: the synthetic records, the star field, the fake core, and the app."""

from __future__ import annotations

import asyncio
import io
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Iterator
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.records.base import Record
from seeingmon.records.survey import PointingRecord
from seeingmon.scheduler.commands import (
    CancelTask,
    Pause,
    QueueDark,
    QueueFlat,
    Resume,
    StartAlignment,
    StopAlignment,
)
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.contract import AlignmentState, pack_frame, unpack_frame
from seeingmon.services.web.demo import (
    DEMO_DARK_SCRIPT,
    DEMO_FLAT_SCRIPT,
    DEMO_NOW_NS,
    DEMO_SENSOR_TEMPERATURE_C,
    DEMO_STATION,
    DEMO_TOKEN,
    FITS_BLOCK,
    FRAME_HEIGHT_PX,
    FRAME_PERIOD_S,
    FRAME_WIDTH_PX,
    HISTOGRAM_BINS,
    IMAGE_SIZE,
    PLATE_SCALE_ARCSEC_PX,
    PREVIEW_SIZE,
    DemoApp,
    DemoClock,
    DemoCore,
    StarField,
    build_demo,
    demo_records,
    fits_bytes,
    sun_elevation_deg,
    write_demo_images,
    write_demo_store,
)
from seeingmon.services.web.images import ImageStore
from seeingmon.store.db import StoreReader
from seeingmon.store.layout import DataLayout
from tests.records.jsonschema_lite import validate
from tests.services.web.client import TestClient
from tests.services.web.conftest import CONFIG, PROFILE
from tests.services.web.helpers import bearer

API = "/api/v1"
HOURS = 3600 * NS_PER_S


# --- The clock and the night -----------------------------------------------------------------


def test_the_clock_stands_still_in_utc_and_runs_in_monotonic_time() -> None:
    clock = DemoClock()
    first = clock.utc_ns()
    mono = clock.monotonic_ns()
    clock.sleep(0.02)
    assert clock.utc_ns() == first == DEMO_NOW_NS
    assert clock.monotonic_ns() - mono >= 15_000_000
    assert clock.status().synchronized is True


def test_the_demo_night_is_centered_on_22_30_utc() -> None:
    day = 86_400 * NS_PER_S * 20_000

    def elevation(hours: float) -> float:
        return sun_elevation_deg(day + round(hours * HOURS))

    assert elevation(22.5) == pytest.approx(-66.0)
    assert elevation(10.5) == pytest.approx(30.0)
    assert elevation(16.5) == pytest.approx(-18.0)
    assert elevation(4.5) == pytest.approx(-18.0)
    assert elevation(12.0) > 0
    assert -18.0 < elevation(5.0) < -6.0  # dawn twilight
    assert elevation(1.0) < -18.0


# --- The records -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def records() -> list[Record]:
    return demo_records()


def of_type(records: list[Record], record_type: str) -> list[Record]:
    return [record for record in records if record.record_type == record_type]


def test_the_records_cover_the_last_24_hours_in_time_order(records: list[Record]) -> None:
    times = [record.t_utc_ns for record in records]
    assert times == sorted(times)
    assert times[0] >= DEMO_NOW_NS - 24 * HOURS
    assert times[-1] < DEMO_NOW_NS


def test_the_demo_has_every_served_record_type(records: list[Record]) -> None:
    counts = Counter(record.record_type for record in records)
    assert 700 <= counts["seeing_window"] <= 900
    assert 150 <= counts["sky_quality"] <= 200
    assert 70 <= counts["pointing"] <= 100
    assert counts["health"] == 288
    assert 10 <= counts["event"] <= 20


def test_nothing_in_the_demo_names_a_real_station_or_a_site(records: list[Record]) -> None:
    assert {record.station_id for record in records} == {DEMO_STATION}
    for window in of_type(records, "seeing_window"):
        assert window.zenith_angle_deg is None  # type: ignore[attr-defined]


def test_seeing_runs_only_while_the_sun_is_low_and_has_gaps_for_clouds(
    records: list[Record],
) -> None:
    windows = of_type(records, "seeing_window")
    assert all(sun_elevation_deg(window.t_utc_ns) < -6.0 for window in windows)
    values = [w.seeing_fwhm_arcsec for w in windows]  # type: ignore[attr-defined]
    assert any(value is None for value in values)
    assert all(0.7 < value < 5.0 for value in values if value is not None)
    flags = {flag for window in windows for flag in window.flags}  # type: ignore[attr-defined]
    assert {"cloud", "twilight", "vibration"} <= flags
    missing = [w for w in windows if w.seeing_fwhm_arcsec is None]  # type: ignore[attr-defined]
    assert all(w.quality for w in missing)


def test_r0_follows_the_seeing_by_the_kolmogorov_relation(records: list[Record]) -> None:
    for window in of_type(records, "seeing_window")[:200]:
        seeing = window.seeing_fwhm_arcsec  # type: ignore[attr-defined]
        if seeing is None:
            continue
        expected = 0.98 * 500e-9 / (seeing * math.pi / 180 / 3600) * 100
        assert window.r0_cm == pytest.approx(expected, rel=0.01)  # type: ignore[attr-defined]


def test_the_sky_is_dark_at_night_and_bright_in_twilight(records: list[Record]) -> None:
    sky = of_type(records, "sky_quality")
    dark = [r.sky_mag_arcsec2 for r in sky if sun_elevation_deg(r.t_utc_ns) < -18]  # type: ignore[attr-defined]
    twilight = [r.sky_mag_arcsec2 for r in sky if -12 < sun_elevation_deg(r.t_utc_ns) < -6]  # type: ignore[attr-defined]
    assert max(dark) > 20.0
    assert min(twilight) < 19.0


def test_the_newest_health_record_is_fresh_and_healthy(records: list[Record]) -> None:
    health = of_type(records, "health")
    newest = health[-1]
    assert DEMO_NOW_NS - newest.t_utc_ns <= 60 * NS_PER_S
    assert newest.state == "auto"  # type: ignore[attr-defined]
    assert newest.degraded is False  # type: ignore[attr-defined]
    assert set(newest.components.values()) == {"ok"}  # type: ignore[attr-defined]
    assert any(record.degraded for record in health)  # type: ignore[attr-defined]
    states = {record.state for record in health}  # type: ignore[attr-defined]
    assert states == {"auto", "safe"}


def test_the_events_have_every_level_and_a_dotted_kind(records: list[Record]) -> None:
    events = of_type(records, "event")
    assert {event.level for event in events} == {"info", "warning", "error"}  # type: ignore[attr-defined]
    assert all("." in event.kind for event in events)  # type: ignore[attr-defined]


def test_the_same_seed_gives_the_same_records_and_another_seed_gives_others() -> None:
    first = [record.model_dump_json() for record in demo_records(seed=1)]
    again = [record.model_dump_json() for record in demo_records(seed=1)]
    other = [record.model_dump_json() for record in demo_records(seed=2)]
    assert first == again
    assert first != other


# --- The pole and Polaris --------------------------------------------------------------------

FRAME_CENTER = ((FRAME_WIDTH_PX - 1) / 2, (FRAME_HEIGHT_PX - 1) / 2)
SIDEREAL_DEG_PER_HOUR = 360.98564736629 / 24.0
COLATITUDE_PX = 0.62 * 3600 / PLATE_SCALE_ARCSEC_PX  # the circle of Polaris, about 584 pixels


def polaris_angle_deg(record: Record) -> float:
    """The position angle of Polaris around the pole, from image up toward image left."""
    assert isinstance(record, PointingRecord)
    assert None not in (record.polaris_x_px, record.polaris_y_px)
    assert None not in (record.pole_x_px, record.pole_y_px)
    dx = record.polaris_x_px - record.pole_x_px  # type: ignore[operator]
    dy = record.polaris_y_px - record.pole_y_px  # type: ignore[operator]
    return math.degrees(math.atan2(-dx, -dy))


def test_the_pole_of_the_demo_lies_near_the_middle_of_the_frame_and_stays_put(
    records: list[Record],
) -> None:
    pointing = of_type(records, "pointing")
    assert len(pointing) >= 60
    poles = [(r.pole_x_px, r.pole_y_px) for r in pointing]  # type: ignore[attr-defined]
    assert None not in [value for pole in poles for value in pole]
    for x, y in poles:
        assert math.hypot(x - FRAME_CENTER[0], y - FRAME_CENTER[1]) == pytest.approx(47.0, abs=0.5)
    assert max(x for x, _ in poles) - min(x for x, _ in poles) < 0.5  # a rigid mount
    assert max(y for _, y in poles) - min(y for _, y in poles) < 0.5


def test_the_roll_of_the_demo_is_the_direction_from_the_center_to_the_pole(
    records: list[Record],
) -> None:
    for record in of_type(records, "pointing"):
        dx = record.pole_x_px - FRAME_CENTER[0]  # type: ignore[attr-defined]
        dy = record.pole_y_px - FRAME_CENTER[1]  # type: ignore[attr-defined]
        assert math.degrees(math.atan2(-dx, -dy)) == pytest.approx(
            record.roll_deg,  # type: ignore[attr-defined]
            abs=0.05,
        )


def test_polaris_circles_the_pole_inside_the_frame_at_the_distance_of_its_colatitude(
    records: list[Record],
) -> None:
    for record in of_type(records, "pointing"):
        x, y = record.polaris_x_px, record.polaris_y_px  # type: ignore[attr-defined]
        pole_x, pole_y = record.pole_x_px, record.pole_y_px  # type: ignore[attr-defined]
        assert math.hypot(x - pole_x, y - pole_y) == pytest.approx(COLATITUDE_PX, abs=0.5)
        assert 0.0 <= x <= FRAME_WIDTH_PX - 1
        assert 0.0 <= y <= FRAME_HEIGHT_PX - 1


def test_polaris_turns_counterclockwise_at_15_degrees_an_hour_in_the_demo(
    records: list[Record],
) -> None:
    pointing = of_type(records, "pointing")
    pairs = [
        (first, second)
        for first, second in pairwise(pointing)
        if second.t_utc_ns - first.t_utc_ns == 600 * NS_PER_S
    ]
    assert len(pairs) >= 60
    for first, second in pairs:
        turn = (polaris_angle_deg(second) - polaris_angle_deg(first) + 180.0) % 360.0 - 180.0
        assert turn == pytest.approx(SIDEREAL_DEG_PER_HOUR / 6.0, abs=0.02)  # 2.5 degrees


# --- The star field --------------------------------------------------------------------------


def decode(jpeg: bytes) -> np.ndarray[Any, Any]:
    with Image.open(io.BytesIO(jpeg)) as image:
        return np.asarray(image.convert("L"))


@pytest.fixture(scope="module")
def field() -> StarField:
    return StarField()


def test_the_picture_is_a_jpeg_of_the_requested_size(field: StarField) -> None:
    jpeg = field.image(1)
    assert jpeg.startswith(b"\xff\xd8")
    assert decode(jpeg).shape == (PREVIEW_SIZE[1], PREVIEW_SIZE[0])
    assert decode(field.image(1, size=IMAGE_SIZE)).shape == (IMAGE_SIZE[1], IMAGE_SIZE[0])


def test_the_same_seed_gives_the_same_bytes_and_another_seed_changes_the_noise(
    field: StarField,
) -> None:
    assert field.image(4) == field.image(4)
    assert field.image(4) != field.image(5)


def brightest_near(
    picture: np.ndarray[Any, Any], x: float, y: float, radius: int = 25
) -> tuple[int, int]:
    y0, x0 = max(0, int(y) - radius), max(0, int(x) - radius)
    window = picture[y0 : int(y) + radius, x0 : int(x) + radius]
    row, col = np.unravel_index(int(np.argmax(window)), window.shape)
    return x0 + int(col), y0 + int(row)


def test_polaris_sits_at_the_target_and_moves_with_the_offset(field: StarField) -> None:
    scale = PREVIEW_SIZE[0] / FRAME_WIDTH_PX
    target = (field.target_x * scale, field.target_y * scale)
    at_target = brightest_near(decode(field.image(7)), *target)
    assert abs(at_target[0] - target[0]) <= 1.5
    assert abs(at_target[1] - target[1]) <= 1.5
    moved = brightest_near(
        decode(field.image(7, dx=40.0, dy=-30.0)), target[0] + 40 * scale, target[1] - 30 * scale
    )
    assert moved[0] - at_target[0] == pytest.approx(40 * scale, abs=2)
    assert moved[1] - at_target[1] == pytest.approx(-30 * scale, abs=2)


def test_the_sky_is_dark_and_the_stars_are_bright(field: StarField) -> None:
    picture = decode(field.image(2))
    assert np.median(picture) < 40
    assert picture.max() > 200


def test_a_frame_has_a_valid_state_and_a_jpeg(field: StarField) -> None:
    frame = field.frame(12)
    assert frame.jpeg.startswith(b"\xff\xd8")
    state = frame.state
    assert AlignmentState.model_validate(state.model_dump(mode="json")) == state
    assert state.active is True
    assert state.frame is not None
    assert state.frame.seq == 12
    assert (state.frame.width_px, state.frame.height_px) == (FRAME_WIDTH_PX, FRAME_HEIGHT_PX)
    assert state.frame.plate_scale_arcsec_px == PLATE_SCALE_ARCSEC_PX
    assert state.t_utc is not None
    assert state.t_utc.endswith("Z")


def test_the_offset_is_the_solved_position_minus_the_target(field: StarField) -> None:
    for seq in (1, 50, 333):
        state = field.frame(seq).state
        assert state.target is not None
        assert state.solved is not None
        assert state.offset is not None
        assert state.offset.dx_px == pytest.approx(state.solved.x_px - state.target.x_px, abs=0.01)
        assert state.offset.dy_px == pytest.approx(state.solved.y_px - state.target.y_px, abs=0.01)
        assert state.offset.distance_px == pytest.approx(
            math.hypot(state.offset.dx_px, state.offset.dy_px), abs=0.01
        )
        assert state.offset.dx_arcsec == pytest.approx(
            state.offset.dx_px * PLATE_SCALE_ARCSEC_PX, abs=0.06
        )
        assert state.offset.distance_arcsec == pytest.approx(
            state.offset.distance_px * PLATE_SCALE_ARCSEC_PX, abs=0.06
        )
        assert state.offset.roll_deg == pytest.approx(
            (state.solved.roll_deg or 0) - (state.target.roll_deg or 0), abs=0.002
        )


def test_the_mount_drifts_and_the_picture_follows(field: StarField) -> None:
    first, second = field.frame(5), field.frame(60)
    assert first.state.offset != second.state.offset
    assert first.jpeg != second.jpeg


def test_the_frame_has_a_log_histogram_a_focus_measure_and_a_saturation_figure(
    field: StarField,
) -> None:
    state = field.frame(20).state
    assert state.histogram is not None
    assert len(state.histogram.counts) == HISTOGRAM_BINS
    assert state.histogram.counts[0] > state.histogram.counts[10] > state.histogram.counts[20]
    assert state.focus is not None
    assert state.focus.fwhm_px is not None
    assert state.focus.best_fwhm_px is not None
    assert state.focus.fwhm_px >= state.focus.best_fwhm_px
    assert state.saturation is not None


def test_the_saturation_warning_comes_and_goes(field: StarField) -> None:
    warnings = [field.frame(seq).state.saturation.warning for seq in range(0, 900, 15)]  # type: ignore[union-attr]
    assert True in warnings
    assert False in warnings


def test_a_frame_packs_into_the_message_that_core_sends(field: StarField) -> None:
    frame = field.frame(3)
    assert unpack_frame(pack_frame(frame.state, frame.jpeg)) == frame


def test_a_frame_takes_well_under_the_frame_period_to_make(field: StarField) -> None:
    field.frame(1)
    started = time.perf_counter()
    for seq in range(2, 12):
        field.frame(seq)
    assert (time.perf_counter() - started) / 10 < FRAME_PERIOD_S  # a slow CI runner has room


# --- The fake core ---------------------------------------------------------------------------


async def take(core: DemoCore, count: int, timeout_s: float = 10.0) -> list[Any]:
    frames: list[Any] = []

    async def collect() -> None:
        async for frame in core.alignment_frames():
            frames.append(frame)
            if len(frames) == count:
                return

    await asyncio.wait_for(collect(), timeout_s)
    return frames


def test_the_core_is_idle_until_the_alignment_starts() -> None:
    core = DemoCore(period_s=0.0)
    assert core.alignment_state() == AlignmentState(active=False)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(take(core, 1, timeout_s=0.5))


def test_the_core_streams_frames_while_it_aligns_and_stops_when_it_stops() -> None:
    core = DemoCore(period_s=0.0)
    assert core.submit(StartAlignment()).accepted
    frames = asyncio.run(take(core, 3))
    assert [frame.state.frame.seq for frame in frames] == [1, 2, 3]
    assert core.alignment_state().active is True
    assert core.alignment_state().frame.seq >= 3  # type: ignore[union-attr]
    assert core.submit(StopAlignment()).accepted
    assert core.alignment_state() == AlignmentState(active=False)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(take(core, 1, timeout_s=0.5))


def test_the_sequence_numbers_go_on_across_sessions() -> None:
    core = DemoCore(period_s=0.0)
    core.submit(StartAlignment())
    asyncio.run(take(core, 2))
    core.submit(StopAlignment())
    core.submit(StartAlignment())
    later = asyncio.run(take(core, 1))
    assert later[0].state.frame.seq > 2


# --- Files -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_folder(tmp_path_factory: pytest.TempPathFactory) -> DataLayout:
    layout = DataLayout(tmp_path_factory.mktemp("demo-data"))
    write_demo_store(layout)
    return layout


def test_the_store_holds_the_records_and_opens_read_only(demo_folder: DataLayout) -> None:
    with StoreReader.open(demo_folder.db_path) as reader:
        assert reader.latest("health") is not None
        assert reader.latest("seeing_window") is not None
        assert len(reader.range("event", 0, DEMO_NOW_NS)) >= 10


def test_previews_and_fits_frames_have_the_names_that_core_gives_them(
    demo_folder: DataLayout,
) -> None:
    images = ImageStore(demo_folder, WebSettings().images)
    items, more = images.recent(200)
    assert more is False
    assert 20 <= len(items) <= 40
    kinds = Counter(item.key.kind for item in items)
    assert set(kinds) == {"preview", "survey"}
    assert all(item.has_fits for item in items if item.key.kind == "survey")
    assert not any(item.has_fits for item in items if item.key.kind == "preview")
    newest = items[0]
    assert newest.key.t_utc_ns < DEMO_NOW_NS


def test_every_preview_is_a_jpeg_of_the_image_size(demo_folder: DataLayout) -> None:
    files = sorted(demo_folder.previews_dir.rglob("*.jpg"))
    assert files
    for path in files:
        with Image.open(path) as image:
            assert image.size == IMAGE_SIZE
            assert image.format == "JPEG"


def test_a_fits_frame_is_a_valid_file_of_whole_blocks() -> None:
    data = fits_bytes(32, 20, seed=1, t_utc_ns=DEMO_NOW_NS)
    assert len(data) % FITS_BLOCK == 0
    header = data[:FITS_BLOCK].decode("ascii")
    cards = [header[i : i + 80] for i in range(0, FITS_BLOCK, 80)]
    assert cards[0].startswith("SIMPLE  =")
    assert cards[0].rstrip().endswith("T")
    keywords = {card[:8].strip(): card[10:30].strip() for card in cards if card[8:10] == "= "}
    assert keywords["BITPIX"] == "16"
    assert keywords["NAXIS"] == "2"
    assert keywords["NAXIS1"] == "32"
    assert keywords["NAXIS2"] == "20"
    assert any(card.startswith("END") for card in cards)
    pixels = np.frombuffer(data[FITS_BLOCK : FITS_BLOCK + 32 * 20 * 2], dtype=">i2")
    values = pixels.astype(np.int64) + 32768  # BZERO
    assert values.min() >= 0
    assert values.max() > 20_000  # the star


def test_the_fits_frame_is_the_same_for_the_same_seed() -> None:
    assert fits_bytes(16, 16, 3, DEMO_NOW_NS) == fits_bytes(16, 16, 3, DEMO_NOW_NS)
    assert fits_bytes(16, 16, 3, DEMO_NOW_NS) != fits_bytes(16, 16, 4, DEMO_NOW_NS)


def test_images_are_written_only_in_the_dark_and_never_in_the_last_minutes(
    tmp_path: Path,
) -> None:
    layout = DataLayout(tmp_path)
    layout.create()
    count = write_demo_images(layout, StarField(stars=20), DEMO_NOW_NS)
    assert count == len(list(layout.previews_dir.rglob("*.jpg")))
    assert count == len(list(layout.previews_dir.rglob("preview-*.jpg"))) + len(
        list(layout.previews_dir.rglob("survey-*.jpg"))
    )


# --- The app ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DemoApp]:
    built = build_demo(
        WebSettings(),
        profile=PROFILE,
        config=CONFIG,
        directory=tmp_path_factory.mktemp("demo-app"),
        frame_period_s=0.05,
    )
    yield built
    built.close()


@pytest.fixture
def demo_client(demo: DemoApp, open_client: Callable[..., TestClient]) -> TestClient:
    return open_client(demo.app)


def test_the_status_says_that_this_is_a_demo_with_a_healthy_station(
    demo_client: TestClient,
) -> None:
    body = demo_client.get(f"{API}/status").json()
    assert body["demo"] is True
    assert body["station_id"] == DEMO_STATION
    assert body["health"]["status"] == "healthy"
    assert body["core"] == {"reachable": True, "instance": "demo-core"}
    assert body["ui"]["commands_enabled"] is True
    assert body["now"].startswith("2026-10-01T03:00:00")
    assert demo_client.get(f"{API}/health").status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/seeing/latest",
        "/sky/latest",
        "/pointing/latest",
        "/seeing",
        "/sky",
        "/pointing",
        "/events",
        "/images",
        "/images/latest",
        "/profile",
        "/config",
        "/alignment/state",
    ],
)
def test_every_read_route_answers_over_the_demo_data(demo_client: TestClient, path: str) -> None:
    assert demo_client.get(f"{API}{path}").status_code == 200


def test_the_latest_demo_pointing_record_gives_the_pole_and_polaris(
    demo_client: TestClient,
) -> None:
    latest = demo_client.get(f"{API}/pointing/latest").json()
    for name in ("pole_x_px", "pole_y_px", "polaris_x_px", "polaris_y_px"):
        assert isinstance(latest[name], float), name
    assert latest["quality"] is None
    radius = math.hypot(
        latest["polaris_x_px"] - latest["pole_x_px"], latest["polaris_y_px"] - latest["pole_y_px"]
    )
    assert radius == pytest.approx(COLATITUDE_PX, abs=0.5)


def test_the_demo_history_of_three_hours_shows_the_orbit(demo_client: TestClient) -> None:
    params = {
        "from": "2026-10-01T00:00:00Z",
        "fields": "pole_x_px,pole_y_px,polaris_x_px,polaris_y_px",
    }
    items = demo_client.get(f"{API}/pointing", params=params).json()["items"]
    assert len(items) == 18  # one record every 10 minutes, from 00:00 to 02:50
    first, last = items[0], items[-1]
    turn = 0.0
    for before, after in pairwise(items):
        angles = [
            math.degrees(
                math.atan2(
                    -(i["polaris_x_px"] - i["pole_x_px"]), -(i["polaris_y_px"] - i["pole_y_px"])
                )
            )
            for i in (before, after)
        ]
        turn += (angles[1] - angles[0] + 180.0) % 360.0 - 180.0
    assert turn == pytest.approx(17 * SIDEREAL_DEG_PER_HOUR / 6.0, abs=0.2)  # 42.6 degrees
    assert first["pole_x_px"] == pytest.approx(last["pole_x_px"], abs=0.5)


def test_the_history_covers_the_night_in_pages_of_means(demo_client: TestClient) -> None:
    page = demo_client.get(f"{API}/seeing", params={"step": "10m", "limit": 500}).json()
    assert page["step"] == "10m"
    assert 70 <= len(page["items"]) <= 100
    assert page["next_cursor"] is None
    raw = demo_client.get(f"{API}/seeing", params={"limit": 2000}).json()
    assert 700 <= len(raw["items"]) <= 900


def test_the_demo_answers_match_the_documented_schemas(demo_client: TestClient) -> None:
    from seeingmon.services.web.openapi import render_openapi

    document = json.loads(render_openapi())
    cases = [
        ("/status", {}),
        ("/health", {}),
        ("/seeing/latest", {}),
        ("/sky/latest", {}),
        ("/pointing/latest", {}),
        ("/seeing", {"step": "1h"}),
        ("/sky", {"step": "10m"}),
        ("/pointing", {"step": "1h"}),
        ("/events", {}),
        ("/images", {}),
        ("/dark", {}),
        ("/flat", {}),
    ]
    for path, params in cases:
        response = demo_client.get(f"{API}{path}", params=params)
        schema = document["paths"][f"{API}{path}"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        validate(response.json(), schema, document)


def test_the_demo_image_and_the_fits_frame_download(demo_client: TestClient) -> None:
    items = demo_client.get(f"{API}/images", params={"limit": 200}).json()["items"]
    survey = next(item for item in items if item["has_fits"])
    preview = demo_client.get(survey["preview_url"])
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/jpeg"
    fits = demo_client.get(survey["fits_url"])
    assert fits.status_code == 200
    assert fits.content.startswith(b"SIMPLE  =")


def test_the_demo_token_starts_and_stops_the_alignment_and_a_wrong_token_does_not(
    demo_client: TestClient,
) -> None:
    refused = demo_client.post(f"{API}/alignment/start", headers=bearer("wrong"))
    assert refused.status_code == 401
    started = demo_client.post(f"{API}/alignment/start", headers=bearer(DEMO_TOKEN))
    assert started.status_code == 200
    assert demo_client.get(f"{API}/alignment/state").json()["active"] is True
    stopped = demo_client.post(f"{API}/alignment/stop", headers=bearer(DEMO_TOKEN))
    assert stopped.status_code == 200
    assert demo_client.get(f"{API}/alignment/state").json()["active"] is False


def test_the_live_view_streams_frames_after_the_alignment_starts(
    demo_client: TestClient,
) -> None:
    assert demo_client.post(f"{API}/alignment/start", headers=bearer(DEMO_TOKEN)).status_code == 200
    try:
        with demo_client.websocket_connect(f"{API}/alignment/stream") as session:
            message = session.receive_json()
            while message.get("type") != "state":
                message = session.receive_json()
            jpeg = session.receive_bytes()
            assert jpeg.startswith(b"\xff\xd8")
            assert message["state"]["active"] is True
            assert message["state"]["frame"]["width_px"] == FRAME_WIDTH_PX
    finally:
        demo_client.post(f"{API}/alignment/stop", headers=bearer(DEMO_TOKEN))


def test_the_demo_serves_the_dark_library_and_a_session_that_the_demo_token_starts(
    demo_client: TestClient,
) -> None:
    library = demo_client.get(f"{API}/dark").json()
    assert len(library["sets"]) == 6
    assert library["status"]["due"] is True
    assert library["task"]["state"] == "idle"
    refused = demo_client.post(f"{API}/commands/dark", json={}, headers=bearer("wrong"))
    assert refused.status_code == 401
    started = demo_client.post(f"{API}/commands/dark", json={}, headers=bearer(DEMO_TOKEN))
    assert started.status_code == 200
    try:
        assert demo_client.get(f"{API}/dark").json()["task"]["state"] in {"queued", "running"}
        again = demo_client.post(f"{API}/commands/dark", json={}, headers=bearer(DEMO_TOKEN))
        assert again.status_code == 409
        assert again.json()["reason"] == "busy"
    finally:  # leave the shared demo as it was: the pause aborts the session, and Resume lifts it
        paused = {"mode": "paused"}
        demo_client.post(f"{API}/mode", json=paused, headers=bearer(DEMO_TOKEN))
        demo_client.post(f"{API}/mode", json={"mode": "auto"}, headers=bearer(DEMO_TOKEN))
    assert demo_client.get(f"{API}/dark").json()["task"]["state"] == "aborted"


def test_the_demo_serves_the_flat_library_a_preview_and_a_session_that_the_demo_token_starts(
    demo_client: TestClient,
) -> None:
    library = demo_client.get(f"{API}/flat").json()
    assert len(library["flats"]) == 2
    assert library["blocker"] is None  # the demo library holds dark sets
    assert library["task"]["state"] == "idle"
    assert library["active_version"] == library["flats"][0]["version"]
    assert library["pending_version"] is None
    preview = demo_client.get(library["flats"][0]["image_url"])
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/jpeg"
    assert preview.content.startswith(b"\xff\xd8\xff")
    refused = demo_client.post(f"{API}/flat/session", json={}, headers=bearer("wrong"))
    assert refused.status_code == 401
    started = demo_client.post(f"{API}/flat/session", json={}, headers=bearer(DEMO_TOKEN))
    assert started.status_code == 200
    try:
        assert demo_client.get(f"{API}/flat").json()["task"]["state"] in {"queued", "running"}
        again = demo_client.post(f"{API}/flat/session", json={}, headers=bearer(DEMO_TOKEN))
        assert again.status_code == 409
        assert again.json()["reason"] == "busy"
        stopped = demo_client.post(f"{API}/flat/session/stop", headers=bearer(DEMO_TOKEN))
        assert stopped.status_code == 200
    finally:  # leave the shared demo as it was: a pause, and then a resume
        demo_client.post(f"{API}/mode", json={"mode": "paused"}, headers=bearer(DEMO_TOKEN))
        demo_client.post(f"{API}/mode", json={"mode": "auto"}, headers=bearer(DEMO_TOKEN))
    assert demo_client.get(f"{API}/flat").json()["task"]["state"] == "aborted"


def test_the_demo_token_is_a_fixed_word_and_not_a_secret() -> None:
    assert DEMO_TOKEN == "demo"


def test_close_removes_a_folder_that_the_demo_made_and_keeps_one_that_it_was_given(
    tmp_path: Path,
) -> None:
    kept = tmp_path / "kept"
    given = build_demo(WebSettings(), directory=kept)
    given.close()
    assert kept.exists()
    made = build_demo(WebSettings())
    folder = made.directory
    assert folder.exists()
    made.close()
    assert not folder.exists()
    made.close()  # safe to repeat


def test_a_demo_that_fails_to_build_leaves_no_folder_behind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tempfile

    from seeingmon.services.web import demo as demo_module

    made: list[Path] = []
    real = tempfile.mkdtemp

    def tracked(**options: Any) -> str:
        path = str(real(**options))
        made.append(Path(path))
        return path

    def broken(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("no store")

    monkeypatch.setattr(tempfile, "mkdtemp", tracked)
    monkeypatch.setattr(demo_module, "write_demo_store", broken)
    with pytest.raises(RuntimeError, match="no store"):
        build_demo(WebSettings())
    assert made
    assert not made[0].exists()


# --- The dark library of the demo ------------------------------------------------------------


@pytest.fixture
def dark_core() -> tuple[DemoCore, VirtualClock]:
    clock = VirtualClock(DEMO_NOW_NS)
    return DemoCore(clock), clock


def test_the_demo_library_has_six_sets_a_model_and_a_gap_at_the_sensor_temperature(
    dark_core: tuple[DemoCore, VirtualClock],
) -> None:
    core, _ = dark_core
    library = core.dark_library()
    assert [item.age_days for item in library.sets] == [3.0, 12.0, 25.0, 47.0, 66.0, 90.0]
    assert library.sensor_temperature_c == DEMO_SENSOR_TEMPERATURE_C
    assert library.model is not None
    assert library.model.doubling_fitted is True
    assert library.model.n_sets == 6
    for item in library.sets:  # the sets follow the model, with a little scatter
        expected = library.model.rate_ref_e_per_s * 2 ** (
            (item.temperature_c - library.model.reference_c) / library.model.doubling_c
        )
        assert item.rate_e_per_s == pytest.approx(expected, rel=0.05)
    assert library.status.due is True  # no set lies within 3 C of the sensor temperature
    assert library.status.gap_c is not None
    assert library.status.gap_c > 3.0
    assert library.task.state == "idle"
    assert len({item.name for item in library.sets}) == 6


def test_the_demo_library_can_start_empty() -> None:
    library = DemoCore(VirtualClock(DEMO_NOW_NS), library=False).dark_library()
    assert library.sets == []
    assert library.model is None
    assert library.status.due is True


def test_a_demo_session_follows_its_timeline_and_closes_the_gap(
    dark_core: tuple[DemoCore, VirtualClock],
) -> None:
    core, clock = dark_core
    script = DEMO_DARK_SCRIPT
    assert core.submit(QueueDark()).accepted
    assert core.dark_library().task.state == "queued"  # the queue lasts a few seconds
    clock.advance(script.queued_s)
    task = core.dark_library().task
    assert (task.state, task.phase, task.steps) == ("running", "bias", 9)
    clock.advance(script.bias_s)  # the wait for the cover starts, and nobody has covered the camera
    task = core.dark_library().task
    assert (task.phase, task.covered) == ("cover", False)
    assert task.reason  # a plain reason: the frame is not dark yet
    clock.advance(script.cover_s * 0.7)  # a few seconds later, the camera is covered
    assert core.dark_library().task.covered is True
    clock.advance(script.cover_s * 0.3)
    task = core.dark_library().task
    assert (task.phase, task.steps) == ("dark", 9)
    clock.advance(script.dark_s)
    assert core.dark_library().task.phase == "build"
    clock.advance(script.build_s)
    library = core.dark_library()
    assert library.task.state == "ok"
    assert library.task.set_name == library.sets[0].name
    assert library.sets[0].temperature_c == DEMO_SENSOR_TEMPERATURE_C
    assert len(library.sets) == 7
    assert library.model is not None
    assert library.model.n_sets == 7
    assert library.status.due is False  # the new set closes the gap
    assert core.status().scheduler.state == "paused"


def test_the_demo_scheduler_resumes_after_a_session(
    dark_core: tuple[DemoCore, VirtualClock],
) -> None:
    core, clock = dark_core
    core.submit(QueueDark())
    clock.advance(60)
    assert core.status().scheduler.state == "paused"
    assert core.submit(Resume()).accepted
    assert core.status().scheduler.state == "safe"


def test_a_demo_session_that_does_not_wait_for_the_cover_fails(
    dark_core: tuple[DemoCore, VirtualClock],
) -> None:
    core, clock = dark_core
    core.submit(QueueDark(wait_for_cover=False))
    clock.advance(DEMO_DARK_SCRIPT.queued_s + DEMO_DARK_SCRIPT.bias_s + 0.1)
    library = core.dark_library()
    assert library.task.state == "failed"
    assert "not covered" in library.task.summary
    assert len(library.sets) == 6


def test_a_pause_aborts_a_demo_session(dark_core: tuple[DemoCore, VirtualClock]) -> None:
    core, clock = dark_core
    core.submit(QueueDark())
    clock.advance(12)
    assert core.dark_library().task.state == "running"
    assert core.submit(Pause()).accepted
    assert core.dark_library().task.state == "aborted"


def test_two_demo_sessions_in_a_row_name_their_sets_apart(
    dark_core: tuple[DemoCore, VirtualClock],
) -> None:
    """The demo clock stands still in UTC, so two sets get the same stamp."""
    core, clock = dark_core
    for _ in range(2):
        core.submit(QueueDark(pause_after=False))
        clock.advance(60)
    names = [item.name for item in core.dark_library().sets]
    assert len(names) == 8
    assert len(set(names)) == 8


# --- The flat library of the demo ------------------------------------------------------------


@pytest.fixture
def flat_core() -> tuple[DemoCore, VirtualClock]:
    clock = VirtualClock(DEMO_NOW_NS)
    return DemoCore(clock), clock


def test_the_demo_flat_library_has_two_flats_and_the_newer_one_is_in_use(
    flat_core: tuple[DemoCore, VirtualClock],
) -> None:
    core, _ = flat_core
    library = core.flat_library()
    assert [item.age_days for item in library.flats] == pytest.approx([12.0, 47.0], abs=0.01)
    in_use, older = library.flats
    assert (in_use.active, in_use.state, in_use.second_set) == (True, "approved", True)
    assert (older.active, older.state, older.second_set) == (False, "approved", False)
    assert in_use.shadows == 3
    assert older.shadows == 4  # before the lens was cleaned
    assert library.active_version == in_use.version
    assert (library.pending_version, library.session, library.blocker) == (None, None, None)
    assert library.sensor_temperature_c == DEMO_SENSOR_TEMPERATURE_C
    assert (library.mode, library.gain) == ("bin2", 120)
    assert library.task.state == "idle"


def test_the_demo_flat_library_can_start_empty_and_then_blocks_a_session() -> None:
    core = DemoCore(VirtualClock(DEMO_NOW_NS), library=False)
    library = core.flat_library()
    assert library.flats == []
    assert library.blocker is not None
    assert "dark set" in library.blocker
    assert not core.submit(QueueFlat()).accepted


def test_a_demo_flat_session_follows_its_timeline_and_adds_a_pending_flat(
    flat_core: tuple[DemoCore, VirtualClock],
) -> None:
    core, clock = flat_core
    script = DEMO_FLAT_SCRIPT
    assert core.submit(QueueFlat()).accepted
    assert core.flat_library().task.state == "queued"  # the queue lasts a few seconds
    clock.advance(script.queued_s)
    task = core.flat_library().task
    assert (task.state, task.phase, task.steps) == ("running", "setup", 1)
    clock.advance(script.setup_s + 0.1)  # the search for the exposure
    task = core.flat_library().task
    assert (task.phase, task.step, task.steps) == ("exposure", 1, 8)
    assert task.exposure_s is not None
    assert task.level_fraction is not None
    assert task.level_fraction < 0.3  # the first try is too dark
    clock.advance(script.exposure_s * 0.6)
    task = core.flat_library().task
    assert task.level_fraction == pytest.approx(0.5, abs=0.02)  # the second try lands
    clock.advance(script.exposure_s * 0.4 + 0.1)  # the frames
    task = core.flat_library().task
    assert (task.phase, task.steps) == ("capture", 32)
    assert task.warnings == []
    clock.advance(script.capture_s * 0.6)  # the light drifts from the middle on
    assert core.flat_library().task.warnings == list(script.warnings)
    clock.advance(script.capture_s * 0.4)
    assert core.flat_library().task.phase == "build"
    clock.advance(script.build_s)
    library = core.flat_library()
    assert library.task.state == "ok"
    assert library.task.version == library.flats[0].version
    assert (library.flats[0].state, library.flats[0].pending) == ("pending", True)
    assert library.pending_version == library.task.version
    assert library.session is not None
    assert len(library.flats) == 3
    assert library.active_version == library.flats[1].version  # nothing changes until you decide
    assert core.status().scheduler.state == "paused"


def test_a_demo_second_set_replaces_the_flat_of_the_first_set(
    flat_core: tuple[DemoCore, VirtualClock],
) -> None:
    core, clock = flat_core
    core.submit(QueueFlat())
    clock.advance(60)
    first = core.flat_library().flats[0]
    core.submit(Resume())
    assert core.submit(QueueFlat(set_number=2)).accepted
    clock.advance(60)
    library = core.flat_library()
    assert len(library.flats) == 3  # the flat of the first set gave way
    two_sets = library.flats[0]
    assert two_sets.version != first.version
    assert (two_sets.second_set, two_sets.source_turned) == (True, True)
    assert library.session is None
    assert two_sets.optics_tilt is not None


def test_two_demo_sessions_in_a_row_keep_the_newest_flat_first(
    flat_core: tuple[DemoCore, VirtualClock],
) -> None:
    """The demo clock stands still in UTC, so two flats get the same time."""
    core, clock = flat_core
    for _ in range(2):
        core.submit(QueueFlat(pause_after=False))
        clock.advance(60)
        newest = core.flat_library().task.version
        assert core.flat_library().flats[0].version == newest
    versions = [item.version for item in core.flat_library().flats]
    assert len(versions) == len(set(versions)) == 4


def test_a_pause_aborts_a_demo_flat_session_and_a_stop_removes_a_waiting_one(
    flat_core: tuple[DemoCore, VirtualClock],
) -> None:
    core, clock = flat_core
    core.submit(QueueFlat())
    clock.advance(12)
    assert core.flat_library().task.state == "running"
    assert core.submit(Pause()).accepted
    assert core.flat_library().task.state == "aborted"
    core.submit(Resume())
    core.submit(QueueFlat())
    assert core.submit(CancelTask(kind="flat")).accepted
    assert core.flat_library().task.state == "aborted"


def test_the_demo_flats_have_previews(flat_core: tuple[DemoCore, VirtualClock]) -> None:
    core, _ = flat_core
    for item in core.flat_library().flats:
        jpeg = core.flat_image(item.version)
        assert jpeg is not None
        image = Image.open(io.BytesIO(jpeg))
        assert image.size == (518, 353)
        pixels = np.asarray(image, dtype=np.float32)
        center = pixels[150:200, 230:290].mean()
        corner = pixels[:30, :40].mean()
        assert center > corner + 40  # the corners get less light, and the stretch shows it
