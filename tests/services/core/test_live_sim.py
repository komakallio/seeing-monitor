"""The live video of Polaris from the simulated camera, through the real analyzer and client.

The simulator renders stars through turbulence of a known strength and keeps the truth of every
frame: where the star really sat. The frames pass through the real fast analyzer, which the
`LiveFastAnalyzer` wraps, and the stream sends them over the real connection layer to the real
client of the web process. The video must show the star where it is, flicker as the star does,
and measure the width that the analyzer measures.
"""

from __future__ import annotations

import asyncio
import io
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("seeingmon.drivers.sim", reason="the simulator needs SciPy (the fast extra)")

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.drivers.sim import SimDriver, sim_camera
from seeingmon.fastpath import FastPathAnalyzer, FastPathConfig, create_fast_analyzer
from seeingmon.frames import PixelFormat, Roi, StreamConfig
from seeingmon.profile import Profile, load_profile
from seeingmon.services.core.live import LiveFastAnalyzer, PolarisStream
from seeingmon.services.core.polaris import PolarisRenderer
from seeingmon.services.core.settings import PolarisSettings
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.server import IpcServer
from seeingmon.services.ipc.stream import StreamService, StreamWindow
from seeingmon.services.web.contract import PNG_MAGIC, POLARIS_CHANNEL, PolarisFrame
from seeingmon.services.web.core_client import RpcCoreClient

from ..conftest import native_endpoint, wait_until

KEY = ConnectionKey.from_text("a-test-key-of-more-than-32-characters")
FRAMES = 330  # four seconds of frame time at 82 frames a second
PLATE_SCALE = 1.91


@dataclass
class SimRig:
    stream: PolarisStream
    server: IpcServer
    roi: Roi
    driver: SimDriver
    analyzer: FastPathAnalyzer
    wrapper: LiveFastAnalyzer


@pytest.fixture(scope="module")
def profile() -> Profile:
    return load_profile("asi294mm-gs250")


@pytest.fixture
def rig(short_dir: Path, profile: Profile) -> Iterator[SimRig]:
    clock = VirtualClock()
    driver = sim_camera(
        clock,
        r0_m=0.08,
        wind_speed_m_s=10.0,
        outer_scale_m=20.0,
        seed=11,
        psf_mode="gaussian",
        zenith_angle_deg=30.0,
    )
    driver.open()
    stars = driver.truth.star_positions(clock.utc_ns(), "bin1")
    brightest = int(np.argmin(stars.mag))
    roi = Roi(int(stars.x[brightest]) - 64, int(stars.y[brightest]) - 64, 128, 128)
    active = driver.configure(
        StreamConfig("bin1", 2000, 0, roi=roi, pixel_format=PixelFormat.RAW16)
    )
    driver.start()
    analyzer = create_fast_analyzer(profile, FastPathConfig(), "sim")
    analyzer.begin_stream(active)
    settings = PolarisSettings()
    stream = PolarisStream(
        settings,
        clock=clock,
        renderer=PolarisRenderer(
            settings, scale_for=lambda mode: PLATE_SCALE, aperture_for=lambda slot: 16.0
        ),
    )
    stream.start()
    window = StreamWindow(16, 8 * 1024 * 1024)
    server = IpcServer(
        native_endpoint(short_dir, "core"),
        KEY,
        {POLARIS_CHANNEL: StreamService(stream.attach, max_window=window)},
        handshake_timeout_s=2.0,
    )
    server.start()
    yield SimRig(stream, server, roi, driver, analyzer, LiveFastAnalyzer(analyzer, stream))
    stream.stop()
    server.stop()


def watch_while_the_scheduler_runs(rig: SimRig, frames: int) -> list[PolarisFrame]:
    """Feed the camera frames from a thread, as the scheduler does, and collect the video."""
    client = RpcCoreClient(rig.server.endpoint, KEY, retry_interval_s=0.01, poll_s=0.05)
    received: list[PolarisFrame] = []
    done = threading.Event()

    def scheduler_thread() -> None:
        for _ in range(frames):
            rig.wrapper.push(rig.driver.read_frame(1.0))
        done.set()

    async def main() -> None:
        async def watch() -> None:
            async for frame in client.polaris_frames():
                received.append(frame)
                if done.is_set() and len(received) >= 55:
                    return

        task = asyncio.create_task(watch())
        while rig.stream.viewers == 0:  # the viewer attaches before the first frame
            await asyncio.sleep(0.01)
        worker = threading.Thread(target=scheduler_thread)
        worker.start()
        await asyncio.wait_for(task, 60.0)
        worker.join()

    try:
        asyncio.run(main())
    finally:
        client.close()
    return received


def test_the_video_shows_the_star_where_it_is_and_measures_what_the_analyzer_measures(
    rig: SimRig,
) -> None:
    frames = watch_while_the_scheduler_runs(rig, FRAMES)
    rows = rig.analyzer.drain_metrics()
    assert rows is not None
    truth = {frame.t_utc_ns: frame for frame in rig.driver.truth.frames}

    # About 20 frames a second of the video, all numbered as core numbers them: with a zero.
    assert 55 <= len(frames) <= 85
    assert all(frame.state.seq == 0 for frame in frames)
    times = [frame.state.t_utc_ns for frame in frames]
    assert times == sorted(set(times))
    assert np.diff(times).mean() / NS_PER_S == pytest.approx(0.05, rel=0.1)

    errors = []
    for frame in frames:
        state = frame.state
        assert frame.image.startswith(PNG_MAGIC)
        assert (state.image_width, state.image_height) == (128, 128)
        assert (state.roi.x, state.roi.y) == (rig.roi.x, rig.roi.y)
        assert state.mode == "bin1"
        assert state.exposure_us == 2000
        assert state.scale_arcsec_px == PLATE_SCALE
        assert state.star.found is True
        assert state.star.x is not None
        assert state.star.y is not None
        true = truth[state.t_utc_ns]
        errors.append(
            (state.star.x + rig.roi.x - true.star_x_px, state.star.y + rig.roi.y - true.star_y_px)
        )
        # The picture shows the star at the same place: the centroid of its pixels.
        picture = np.asarray(Image.open(io.BytesIO(frame.image))).astype(float)
        yy, xx = np.mgrid[0:128, 0:128]
        window = (np.abs(xx - state.star.x) < 5) & (np.abs(yy - state.star.y) < 5)
        weight = picture * window
        assert float((weight * xx).sum() / weight.sum()) == pytest.approx(state.star.x, abs=0.6)
        assert float((weight * yy).sum() / weight.sum()) == pytest.approx(state.star.y, abs=0.6)
    error = np.array(errors)
    assert np.abs(error).max() < 0.1  # pixels: the analyzer centroid against the true position
    assert np.sqrt((error**2).mean()) < 0.04

    # The frame rate of the camera, from the counts and the times of the kept frames.
    fps = [frame.state.fast_fps for frame in frames[2:]]
    assert None not in fps
    assert np.mean([value for value in fps if value is not None]) == pytest.approx(82.1, rel=0.01)

    # The width of each frame, against the analyzer's own width over the same frames.
    analyzer_fwhm = float(
        np.nanmean(0.5 * (rows["width_x_px"] + rows["width_y_px"])) * 2.3548 * PLATE_SCALE
    )
    video_fwhm = float(np.mean([frame.state.star.fwhm_arcsec for frame in frames]))
    assert video_fwhm == pytest.approx(analyzer_fwhm, rel=0.03)

    # The brightness flickers in the picture as it does in the frames: the brightest pixel of
    # the picture follows the peak that the analyzer measured.
    peaks = {int(t): float(p) for t, p in zip(rows["t_utc_ns"], rows["peak_dn"], strict=True)}
    shown = np.array([np.asarray(Image.open(io.BytesIO(f.image))).max() for f in frames[20:]])
    measured = np.array([peaks[frame.state.t_utc_ns] for frame in frames[20:]])
    assert float(np.std(measured) / np.mean(measured)) > 0.05  # the star flickers in the frames
    clipped = shown == 255  # the stretch clips the brightest flashes at white
    assert clipped.mean() < 0.3
    assert float(np.corrcoef(shown[~clipped], measured[~clipped])[0, 1]) > 0.9  # and in the picture


def test_without_a_viewer_the_stream_keeps_and_encodes_nothing(rig: SimRig) -> None:
    for _ in range(40):
        rig.wrapper.push(rig.driver.read_frame(1.0))
    assert rig.stream.frames_kept == 0
    assert rig.stream.frames_encoded == 0
    assert rig.analyzer.frames_pushed == 40
    assert wait_until(lambda: rig.stream.viewers == 0)
