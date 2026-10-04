"""The demo video of Polaris, and the rolling seeing value that the demo serves."""

from __future__ import annotations

import asyncio
import io
import json
import math
import time
from collections.abc import Callable, Iterator
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from PIL import Image

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.scheduler.commands import Pause, Resume, StartAlignment, StopAlignment
from seeingmon.services.core.polaris import FrameSlot, PolarisRenderer
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.contract import (
    PNG_MAGIC,
    LiveSeeingView,
    PolarisFrame,
    pack_polaris_frame,
    unpack_polaris_frame,
)
from seeingmon.services.web.demo import (
    CAMERA_FPS,
    DEMO_NOW_NS,
    DEMO_SEEING_ARCSEC,
    FAST_PLATE_SCALE_ARCSEC_PX,
    LIVE_EVERY_S,
    LIVE_MIN_SPAN_S,
    POLARIS_PERIOD_S,
    POLARIS_ROI,
    POLARIS_SIZE,
    DemoApp,
    DemoCore,
    PolarisSky,
    build_demo,
    demo_live_seeing,
    demo_seeing_arcsec,
)
from tests.records.jsonschema_lite import validate
from tests.services.conftest import wait_until
from tests.services.web.client import TestClient
from tests.services.web.conftest import CONFIG, PROFILE

API = "/api/v1"


def frames_of(sky: PolarisSky, count: int) -> list[tuple[FrameSlot, float]]:
    return [sky.next_frame() for _ in range(count)]


def centroid(slot: FrameSlot) -> tuple[float, float]:
    """Where the demo put the star, in pixels of the image."""
    assert slot.star.x_px is not None
    assert slot.star.y_px is not None
    return slot.star.x_px - slot.roi.x, slot.star.y_px - slot.roi.y


# --- The sky ---------------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_video_and_another_seed_another() -> None:
    first = frames_of(PolarisSky(seed=3), 4)
    again = frames_of(PolarisSky(seed=3), 4)
    other = frames_of(PolarisSky(seed=4), 4)
    for (a, _), (b, _), (c, _) in zip(first, again, other, strict=True):
        assert np.array_equal(a.data, b.data)
        assert not np.array_equal(a.data, c.data)


def test_a_frame_holds_the_counts_of_a_12_bit_adc_in_a_16_bit_container() -> None:
    slot, t_s = PolarisSky().next_frame()
    assert t_s == 0.0
    assert slot.data.dtype == np.uint16
    assert slot.data.shape == (POLARIS_SIZE, POLARIS_SIZE)
    assert np.all(slot.data % 16 == 0)
    assert int(slot.data.max()) <= 65_520
    assert slot.roi == POLARIS_ROI
    assert (slot.mode, slot.exposure_us, slot.gain, slot.adc_bits) == ("bin1", 2000, 0, 12)
    assert slot.star.found is True


def test_the_star_sits_near_the_middle_and_wanders_by_about_04_pixel_rms() -> None:
    slots = [slot for slot, _ in frames_of(PolarisSky(), 600)]
    x = np.array([centroid(slot)[0] for slot in slots])
    y = np.array([centroid(slot)[1] for slot in slots])
    assert abs(x.mean() - 64.0) < 0.8
    assert abs(y.mean() - 64.0) < 0.8
    assert 0.3 < x.std() < 0.75
    assert 0.3 < y.std() < 0.75
    assert float(np.corrcoef(x[:-1], x[1:])[0, 1]) < 0.8  # the jitter forgets its past quickly


def test_the_centroid_of_the_pixels_is_where_the_state_says_the_star_is() -> None:
    for slot, _ in frames_of(PolarisSky(), 10):
        data = slot.data.astype(float) - 480.0
        yy, xx = np.mgrid[0:POLARIS_SIZE, 0:POLARIS_SIZE]
        x, y = centroid(slot)
        window = (np.abs(xx - x) < 6) & (np.abs(yy - y) < 6)
        cx = float((data * xx * window).sum() / (data * window).sum())
        cy = float((data * yy * window).sum() / (data * window).sum())
        assert cx == pytest.approx(x, abs=0.15)
        assert cy == pytest.approx(y, abs=0.15)


def test_the_brightness_flickers_by_a_few_percent() -> None:
    flux = np.array(
        [
            float((slot.data.astype(float) - 480.0)[52:76, 52:76].sum())
            for slot, _ in frames_of(PolarisSky(), 600)
        ]
    )
    relative = float(flux.std() / flux.mean())
    assert 0.02 < relative < 0.09


def test_the_sky_is_noisy_and_the_star_is_about_14_pixels_wide() -> None:
    renderer = PolarisRenderer(scale_for=lambda mode: FAST_PLATE_SCALE_ARCSEC_PX)
    widths, sky_noise = [], []
    for slot, _ in frames_of(PolarisSky(), 40):
        state = renderer.render(slot).state
        assert state.star.fwhm_arcsec is not None
        widths.append(state.star.fwhm_arcsec / FAST_PLATE_SCALE_ARCSEC_PX)
        sky_noise.append(float(slot.data[:40, :40].std()))
    assert 1.3 < float(np.mean(widths)) < 2.1  # pixels, FWHM from the second moments
    assert 30.0 < float(np.mean(sky_noise)) < 70.0  # counts: photon noise and read noise


def test_the_frames_come_from_a_camera_at_82_frames_a_second() -> None:
    slots = [slot for slot, _ in frames_of(PolarisSky(), 50)]
    for before, after in pairwise(slots):
        rate = (after.count - before.count) / ((after.t_utc_ns - before.t_utc_ns) / NS_PER_S)
        assert rate == pytest.approx(CAMERA_FPS, rel=1e-3)
        assert 0.045 < (after.t_utc_ns - before.t_utc_ns) / NS_PER_S < 0.065
    assert slots[0].t_utc_ns == DEMO_NOW_NS


def test_a_frame_takes_well_under_a_period_to_make_and_render() -> None:
    sky = PolarisSky()
    renderer = PolarisRenderer(scale_for=lambda mode: FAST_PLATE_SCALE_ARCSEC_PX)
    renderer.render(sky.next_frame()[0])
    started = time.perf_counter()
    for _ in range(10):
        renderer.render(sky.next_frame()[0])
    assert (time.perf_counter() - started) / 10 < POLARIS_PERIOD_S  # a slow runner has room


# --- The rolling seeing value ----------------------------------------------------------------


def test_the_rolling_value_comes_after_four_seconds_and_then_every_two() -> None:
    assert demo_live_seeing(0.0) is None
    assert demo_live_seeing(LIVE_MIN_SPAN_S - 0.1) is None
    first = demo_live_seeing(LIVE_MIN_SPAN_S)
    assert first is not None
    assert first.span_s == LIVE_MIN_SPAN_S
    assert demo_live_seeing(LIVE_MIN_SPAN_S + 1.9) == first  # the value holds for two seconds
    second = demo_live_seeing(LIVE_MIN_SPAN_S + LIVE_EVERY_S)
    assert second is not None
    assert second.t_utc_ns - first.t_utc_ns == int(LIVE_EVERY_S * NS_PER_S)
    assert second != first


def test_the_value_varies_slowly_around_the_seeing_of_the_demo() -> None:
    values = []
    for step in range(2, 120):  # four minutes
        live = demo_live_seeing(step * LIVE_EVERY_S)
        assert live is not None
        assert live.seeing_fwhm_arcsec is not None
        values.append(live.seeing_fwhm_arcsec)
    assert float(np.mean(values)) == pytest.approx(DEMO_SEEING_ARCSEC, rel=0.05)
    assert 1.1 < min(values) < max(values) < 2.2
    assert max(abs(b - a) for a, b in pairwise(values)) < 0.45


def test_the_value_is_complete_and_consistent() -> None:
    live = demo_live_seeing(30.0)
    assert isinstance(live, LiveSeeingView)
    assert live.span_s == 10.0
    assert live.stream_id == 1
    assert (live.readout_mode, live.exposure_us) == ("bin1", 2000)
    assert live.n_frames == pytest.approx(10 * CAMERA_FPS, abs=5)
    assert live.n_usable <= live.n_frames
    assert 0.99 < live.valid_fraction <= 1.0
    assert live.flags == []
    assert live.quality == {}
    assert live.seeing_fwhm_arcsec is not None
    assert live.r0_cm is not None
    r0 = 0.98 * 500e-9 / (live.seeing_fwhm_arcsec * math.pi / 648_000) * 100
    assert live.r0_cm == pytest.approx(r0, rel=0.01)  # the same relation as the stored records
    assert live.seeing_fwhm_structure_arcsec == pytest.approx(live.seeing_fwhm_arcsec, rel=0.2)
    assert live.image_motion_rms_x_arcsec == pytest.approx(0.48 * live.seeing_fwhm_arcsec, rel=0.15)
    assert live.width_fwhm_arcsec == pytest.approx(3.0, abs=0.2)
    assert demo_live_seeing(30.0) == live  # the same time gives the same value


def test_the_seeing_curve_of_the_night_is_the_one_that_the_records_follow() -> None:
    assert 0.9 < demo_seeing_arcsec(DEMO_NOW_NS) < 2.2


# --- The fake core ---------------------------------------------------------------------------


async def take(core: DemoCore, count: int, timeout_s: float = 10.0) -> list[PolarisFrame]:
    frames: list[PolarisFrame] = []

    async def collect() -> None:
        async for frame in core.polaris_frames():
            frames.append(frame)
            if len(frames) == count:
                return

    await asyncio.wait_for(collect(), timeout_s)
    return frames


def test_the_core_streams_the_video_in_the_auto_state() -> None:
    core = DemoCore(polaris_period_s=0.0)
    assert core.state == "auto"
    frames = asyncio.run(take(core, 5))
    assert len(frames) == 5
    states = [frame.state for frame in frames]
    assert [state.seq for state in states] == [0] * 5  # the hub numbers the frames, not core
    times = [state.t_utc_ns for state in states]
    assert times == sorted(times)
    assert states[-1].fast_fps == pytest.approx(CAMERA_FPS, rel=0.01)
    for frame in frames:
        assert frame.image.startswith(PNG_MAGIC)
        picture = Image.open(io.BytesIO(frame.image))
        assert picture.size == (frame.state.image_width, frame.state.image_height)
        assert unpack_polaris_frame(pack_polaris_frame(frame.state, frame.image)) == frame


def test_the_core_keeps_the_rate_of_the_video() -> None:
    core = DemoCore()
    started = time.perf_counter()
    asyncio.run(take(core, 21))
    elapsed = time.perf_counter() - started
    assert 0.8 < elapsed < 3.0  # 20 intervals of 50 ms, with room for a slow runner


def test_the_core_adds_the_rolling_value_to_the_state_after_four_seconds() -> None:
    clock = VirtualClock()
    core = DemoCore(clock, polaris_period_s=0.0)
    first = asyncio.run(take(core, 1))[0]
    assert first.state.live_seeing is None
    assert core.live_seeing() is None
    clock.advance(30.0)  # the fake core follows its own clock
    later = asyncio.run(take(core, 1))[0]
    assert later.state.live_seeing is not None
    assert later.state.live_seeing == core.live_seeing()
    assert "live_seeing" not in later.state.quality


def test_the_video_goes_quiet_while_the_alignment_runs_or_the_scheduler_is_paused() -> None:
    core = DemoCore(polaris_period_s=0.0)
    assert core.submit(StartAlignment()).accepted
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(take(core, 1, timeout_s=0.4))
    assert core.submit(StopAlignment()).accepted
    assert core.state == "safe"
    assert len(asyncio.run(take(core, 2))) == 2  # the video comes back
    assert core.submit(Pause()).accepted
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(take(core, 1, timeout_s=0.4))
    assert core.submit(Resume()).accepted
    assert len(asyncio.run(take(core, 2))) == 2


def test_every_stream_starts_its_own_stretch() -> None:
    core = DemoCore(polaris_period_s=0.0)
    first = asyncio.run(take(core, 2))
    second = asyncio.run(take(core, 2))
    assert first[0].state.fast_fps is None  # the rate needs two frames of the stream
    assert second[0].state.fast_fps is None
    assert first[1].state.fast_fps is not None


# --- The app ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DemoApp]:
    built = build_demo(
        WebSettings(),
        profile=PROFILE,
        config=CONFIG,
        directory=tmp_path_factory.mktemp("demo-polaris"),
        frame_period_s=0.05,
        polaris_period_s=0.02,
    )
    yield built
    built.close()


@pytest.fixture
def demo_client(demo: DemoApp, open_client: Callable[..., TestClient]) -> TestClient:
    return open_client(demo.app)


def test_the_websocket_of_the_demo_app_sends_the_video(demo_client: TestClient) -> None:
    with demo_client.websocket_connect(f"{API}/polaris/stream") as session:
        seen = []
        for _ in range(6):
            message = session.receive_json()
            assert message["type"] == "state"
            image = session.receive_bytes()
            assert image.startswith(PNG_MAGIC)
            seen.append(message["state"])
    assert [state["seq"] for state in seen] == sorted(state["seq"] for state in seen)
    assert seen[0]["seq"] >= 1  # the hub numbers what it receives, from 1
    state = seen[-1]
    assert state["image_type"] == "image/png"
    assert (state["image_width"], state["image_height"]) == (128, 128)
    assert state["roi"] == {"x": 2008, "y": 1347, "width": 128, "height": 128}
    assert state["scale_arcsec_px"] == 1.91
    assert state["star"]["found"] is True


def test_the_poll_route_of_the_demo_app_serves_the_video_with_its_state(
    demo_client: TestClient,
) -> None:
    assert wait_until(lambda: demo_client.get(f"{API}/polaris/frame").status_code == 200)
    response = demo_client.get(f"{API}/polaris/frame")
    assert response.headers["content-type"] == "image/png"
    state = json.loads(response.headers["x-frame-state"])
    assert state["seq"] == int(response.headers["x-frame-seq"])
    assert response.content.startswith(PNG_MAGIC)
    again = demo_client.get(f"{API}/polaris/frame", params={"after": state["seq"]})
    assert again.status_code in {200, 204}


def test_the_status_of_the_demo_tells_the_page_how_fast_the_video_runs(
    demo_client: TestClient,
) -> None:
    ui = demo_client.get(f"{API}/status").json()["ui"]
    assert ui["polaris_max_fps"] == 20.0
    assert ui["polaris_stall_s"] == 5.0


def test_the_rolling_seeing_route_answers_over_the_demo_core(
    make_app: Callable[..., Any], open_client: Callable[..., TestClient], seeded: Any
) -> None:
    clock = VirtualClock(DEMO_NOW_NS)
    core = DemoCore(clock)
    chosen = open_client(make_app(core=core, clock=clock))
    assert chosen.get(f"{API}/seeing/live").status_code == 404  # the first seconds have no value
    clock.advance(40.0)
    response = chosen.get(f"{API}/seeing/live")
    assert response.status_code == 200
    body = response.json()
    assert body["span_s"] == 10.0
    assert 1.0 < body["seeing_fwhm_arcsec"] < 2.4
    assert body["age_s"] <= 2.0 + 0.001  # a value is never older than the interval of the estimate
    from seeingmon.services.web.openapi import render_openapi

    document = json.loads(render_openapi())
    schema = document["paths"][f"{API}/seeing/live"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    validate(body, schema, document)
