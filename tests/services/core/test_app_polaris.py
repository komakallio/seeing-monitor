"""`CoreApp` and the live video of Polaris: the wiring, the channel, and the method `live_seeing`.

The rig runs the scheduler on its own thread of the test, with a virtual clock, so that a night of
fast, survey, and alignment steps takes seconds. The frames of the video must be the frames that
the fast analyzer received, and no others.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("PIL.Image", reason="the preview needs Pillow")

from seeingmon.frames import Frame
from seeingmon.scheduler import SchedulerConfig
from seeingmon.scheduler.commands import StartAlignment, StopAlignment
from seeingmon.services.core.live import LiveFastAnalyzer
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.stream import StreamKind, connect_stream
from seeingmon.services.web.contract import (
    PNG_MAGIC,
    POLARIS_CHANNEL,
    PolarisFrame,
    unpack_polaris_frame,
)
from seeingmon.services.web.core_client import RpcCoreClient

from ..conftest import native_endpoint, wait_until
from .rig import CoreRig, build_rig

KEY = ConnectionKey.from_text("a-test-key-of-more-than-32-characters")


@dataclasses.dataclass(frozen=True)
class FakeLive:
    """What `FastPathAnalyzer.live` holds: a rolling value with the fields of the contract."""

    t_utc_ns: int = 1_790_000_000_000_000_000
    span_s: float = 10.0
    n_frames: int = 820
    n_usable: int = 815
    valid_fraction: float = 0.99
    seeing_fwhm_arcsec: float | None = 1.6
    seeing_fwhm_structure_arcsec: float | None = None
    r0_cm: float | None = 6.1
    r0_structure_cm: float | None = None
    image_motion_rms_x_arcsec: float | None = 0.7
    image_motion_rms_y_arcsec: float | None = 0.6
    width_fwhm_arcsec: float | None = 2.9
    stream_id: int = 1
    readout_mode: str = "bin1"
    exposure_us: int = 2000
    flags: tuple[str, ...] = ("cloud",)
    quality: dict[str, str] = dataclasses.field(
        default_factory=lambda: {"seeing_fwhm_structure_arcsec": "too few pairs of frames"}
    )


@pytest.fixture
def served(short_dir: Path, tmp_path: Path) -> Iterator[tuple[CoreRig, RpcCoreClient]]:
    rig = build_rig(tmp_path, key=KEY, endpoint=native_endpoint(short_dir, "core"))
    rig.app.start()
    client = RpcCoreClient(rig.app.bound_endpoint, KEY, retry_interval_s=0.01, poll_s=0.05)  # type: ignore[arg-type]
    yield rig, client
    client.close()
    rig.app.stop()


class TestWiring:
    def test_the_scheduler_gets_the_wrapper_and_the_wrapper_holds_the_analyzer(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, _ = served
        scheduler_analyzer = rig.app.scheduler._fast
        assert isinstance(scheduler_analyzer, LiveFastAnalyzer)
        assert scheduler_analyzer.inner is rig.fast
        assert scheduler_analyzer is rig.app.live_fast

    def test_the_settings_of_the_video_come_from_the_configuration(self, tmp_path: Path) -> None:
        rig = build_rig(
            tmp_path,
            config_extra="[services.core.polaris]\nmax_fps = 10.0\nheadroom = 2.0\n",
        )
        try:
            assert rig.app.settings.polaris.max_fps == 10.0
            assert rig.app.polaris._interval_ns == 100_000_000
            assert rig.app.polaris._renderer.stretch.headroom == 2.0
        finally:
            rig.app.stop()

    def test_the_plate_scale_and_the_aperture_come_from_the_profile_and_the_analyzer(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, _ = served
        renderer = rig.app.polaris._renderer
        assert renderer._scale_for(rig.app.profile.fast_mode.mode) == pytest.approx(
            rig.app.profile.plate_scale_arcsec_per_px(rig.app.profile.fast_mode.mode)
        )
        assert renderer._scale_for("no-such-mode") is None  # an unknown mode has no scale

    def test_the_polaris_thread_starts_and_stops_with_the_threads_of_core(
        self, tmp_path: Path
    ) -> None:
        rig = build_rig(tmp_path, threads=True)
        try:
            rig.app.start()
            assert rig.app.polaris._thread is not None
            assert rig.app.polaris._thread.is_alive()
        finally:
            rig.app.stop()
        assert rig.app.polaris._thread is None


class TestLiveSeeingMethod:
    def test_the_method_answers_null_while_the_analyzer_has_no_value(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        _, client = served
        assert client.live_seeing() is None

    def test_the_method_answers_the_view_of_the_value_of_the_analyzer(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        rig.fast.live = FakeLive()  # type: ignore[attr-defined]
        view = client.live_seeing()
        assert view is not None
        assert view.seeing_fwhm_arcsec == 1.6
        assert view.flags == ["cloud"]
        assert view.quality == {"seeing_fwhm_structure_arcsec": "too few pairs of frames"}
        assert (view.n_frames, view.span_s, view.stream_id) == (820, 10.0, 1)
        rig.fast.live = None  # type: ignore[attr-defined]
        assert client.live_seeing() is None

    def test_a_value_is_converted_once_and_read_by_the_video_and_the_method_alike(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, _ = served
        rig.fast.live = FakeLive()  # type: ignore[attr-defined]
        first = rig.app.live_seeing()
        assert first is not None
        assert rig.app.live_seeing() is first  # the same value gives the same view
        rig.fast.live = FakeLive(seeing_fwhm_arcsec=2.2)  # type: ignore[attr-defined]
        second = rig.app.live_seeing()
        assert second is not None
        assert second is not first
        assert second.seeing_fwhm_arcsec == 2.2


def watch_polaris(rig: CoreRig, client: RpcCoreClient, frames: int) -> list[PolarisFrame]:
    """Open the video, run the scheduler, and encode the newest kept frame now and then."""
    received: list[PolarisFrame] = []

    async def watch() -> None:
        async for frame in client.polaris_frames():
            received.append(frame)
            if len(received) == frames:
                return

    async def main() -> None:
        task = asyncio.create_task(watch())
        while rig.app.polaris.viewers == 0:
            await asyncio.sleep(0.01)
        for _ in range(60):
            await asyncio.to_thread(rig.run_for, 2.0)
            slot = rig.app.polaris._slot  # no thread runs in this rig: the test does its work
            if slot is not None:
                rig.app.polaris._slot = None
                await asyncio.to_thread(rig.app.polaris.process, slot)
            if task.done():
                break
            await asyncio.sleep(0.05)
        await asyncio.wait_for(task, 20.0)

    asyncio.run(main())
    return received


class TestVideoThroughTheScheduler:
    def test_a_viewer_gets_the_frames_of_the_fast_stream_and_only_those(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, client = served
        offered: list[Frame] = []
        keep = rig.app.polaris._keep

        def spy(frame: Frame, update: Any, rapid: bool) -> None:
            offered.append(frame)
            keep(frame, update, rapid)

        rig.app.polaris._keep = spy  # type: ignore[method-assign]
        frames = watch_polaris(rig, client, 3)  # fast periods and survey steps run meanwhile
        # The alignment shows bin2 frames to its own helper, and they must stay out of the video.
        assert rig.app.scheduler.submit(StartAlignment(exposure_s=0.5)).accepted
        rig.run_for(10.0)
        assert rig.app.alignment.frames_received > 0
        assert rig.app.scheduler.submit(StopAlignment()).accepted

        fast_mode = rig.app.profile.fast_mode.mode
        assert len(frames) == 3
        for frame in frames:
            assert frame.image.startswith(PNG_MAGIC)
            assert frame.state.mode == fast_mode
            assert (
                frame.state.exposure_us
                == rig.app.config.section("scheduler", SchedulerConfig).fast.exposure_us
            )
        assert offered, "the scheduler offered no frame to the video"
        assert {frame.mode for frame in offered} == {fast_mode}
        assert len(offered) <= rig.fast.frames_pushed  # each went through the analyzer first
        modes = {config.mode for name, config in rig.camera.calls if name == "configure"}
        assert len(modes) > 1  # the camera did run other streams, and none of them got in

    def test_the_channel_is_served_next_to_the_other_two_and_a_viewer_is_counted(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, _ = served
        receiver, _reply = connect_stream(
            rig.app.bound_endpoint,  # type: ignore[arg-type]
            KEY,
            {"role": "web"},
            channel=POLARIS_CHANNEL,
        )
        try:
            assert wait_until(lambda: rig.app.polaris.viewers == 1)
            assert rig.app.polaris.active is True
            rig.run_for(3.0)
            slot = rig.app.polaris._slot
            assert slot is not None
            rig.app.polaris.process(slot)
            message = receiver.recv(10.0)
            assert message is not None
            assert message.kind is StreamKind.DATA
            decoded = unpack_polaris_frame(message.payload)
            assert decoded.image.startswith(PNG_MAGIC)
        finally:
            receiver.close()

        def gone() -> bool:
            rig.app.polaris.housekeeping()
            return rig.app.polaris.viewers == 0

        assert wait_until(gone)
        assert rig.app.polaris.active is False

    def test_without_a_viewer_the_scheduler_keeps_and_encodes_nothing(
        self, served: tuple[CoreRig, RpcCoreClient]
    ) -> None:
        rig, _ = served
        rig.run_for(10.0)
        assert rig.fast.frames_pushed > 0
        assert rig.app.polaris.frames_kept == 0
        assert rig.app.polaris.frames_encoded == 0
