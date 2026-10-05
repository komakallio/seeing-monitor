"""Create the camera driver that `acquire` owns, from its configuration.

`create_camera_driver` hands every name to `seeingmon.drivers.create_driver`, which imports
`seeingmon.drivers.<name>` and calls its `create(profile=..., clock=..., options=...)`. One name
is special: `fake` builds the scripted `FakeCameraDriver` of `seeingmon.testing`, so that the
end-to-end tests and a development machine without a camera can run `acquire` with a driver that
needs nothing. The options of the fake are `adc_bits`, `overhead_s`, `row_time_s`, and
`temperature_c`. The option `hang_in` (`open`, `configure`, `start`, or `read_frame`) makes that
call block for good, so that a test can show that a hung driver call ends the process.

The `sim` driver takes three options beyond its own: `pointing`, `polaris`, and `polaris_mag`, which
let a simulated run agree with the survey code (see `seeingmon.services.simsky`).

The `asi` driver takes one option beyond its own: `fake_sdk`. With it, the driver runs on the
`FakeAsiSdk` of `seeingmon.hardware.asi.fake`, with no vendor library and no camera, so that a test
can run the real-camera setup of `seeingmon dev --driver asi` (the profile, the clock, and the
settings) as processes. A real run never sets it.

A driver whose `create` takes an `on_event` argument (the `asi` driver does) gets the callback, so
that its hardware events reach the log of `acquire`. This module imports the fakes, so import it
only when you create a driver.
"""

from __future__ import annotations

import importlib
import inspect
import threading
from collections.abc import Callable, Mapping
from typing import Any

from seeingmon.clock import Clock
from seeingmon.drivers import CameraConfigError, CameraDriver
from seeingmon.drivers.base import CameraInfo
from seeingmon.frames import ActiveStream, Frame, StreamConfig
from seeingmon.hardware.events import HardwareEvent
from seeingmon.testing import FakeCameraDriver

FAKE_OPTIONS = {"adc_bits": int, "overhead_s": float, "row_time_s": float, "temperature_c": float}
HANGABLE = ("open", "configure", "start", "read_frame")


class HangingFakeDriver(FakeCameraDriver):
    """A fake whose `hang_in` call blocks for good, as an SDK call can after a USB fault."""

    def __init__(self, clock: Clock, *, hang_in: str, **options: Any) -> None:
        super().__init__(clock, **options)
        self._hang_in = hang_in

    def _hang(self, name: str) -> None:
        if name == self._hang_in:
            threading.Event().wait()

    def open(self) -> CameraInfo:
        self._hang("open")
        return super().open()

    def configure(self, config: StreamConfig) -> ActiveStream:
        self._hang("configure")
        return super().configure(config)

    def start(self) -> None:
        self._hang("start")
        super().start()

    def read_frame(self, timeout_s: float) -> Frame:
        self._hang("read_frame")
        return super().read_frame(timeout_s)


def accepts_events(name: str) -> bool:
    """Whether the `create` of driver `name` takes an `on_event` argument."""
    if not name.isidentifier() or name.startswith("_") or name == "base":
        return False
    try:
        module = importlib.import_module(f"seeingmon.drivers.{name}")
    except ModuleNotFoundError:
        return False
    return "on_event" in inspect.signature(module.create).parameters


def create_camera_driver(
    name: str,
    *,
    profile: Any,
    clock: Clock,
    options: Mapping[str, Any],
    on_event: Callable[[HardwareEvent], None] | None = None,
) -> CameraDriver:
    """Build the driver `name`. Raises `CameraConfigError` for an option it does not know."""
    if name == "fake":
        return _create_fake(clock, options)
    if name == "asi" and options.get("fake_sdk"):
        return _create_asi_on_fake_sdk(profile, clock, options, on_event)
    if on_event is not None and accepts_events(name):
        module = importlib.import_module(f"seeingmon.drivers.{name}")
        driver: CameraDriver = module.create(
            profile=profile, clock=clock, options=dict(options), on_event=on_event
        )
        return driver
    if name == "sim":
        return _create_sim(profile, clock, options)
    from seeingmon.drivers import create_driver

    return create_driver(name, profile=profile, clock=clock, options=dict(options))


def _create_asi_on_fake_sdk(
    profile: Any,
    clock: Clock,
    options: Mapping[str, Any],
    on_event: Callable[[HardwareEvent], None] | None,
) -> CameraDriver:
    """Build the `asi` driver on the fake SDK. The other options are those of `AsiOptions`.

    There is no USB resetter and no watchdog thread, because nothing real can hang or reset.
    """
    from seeingmon.drivers.asi.driver import AsiDriver
    from seeingmon.drivers.asi.options import AsiOptions
    from seeingmon.hardware.asi.fake import FakeAsiSdk

    rest = {key: value for key, value in options.items() if key != "fake_sdk"}
    return AsiDriver(
        api=FakeAsiSdk(clock),
        profile=profile,
        clock=clock,
        options=AsiOptions.from_mapping(rest),
        usb_resetter=None,
        watchdog=None,
        watchdog_thread=False,
        on_event=on_event,
    )


def _create_sim(profile: Any, clock: Clock, options: Mapping[str, Any]) -> CameraDriver:
    """Build the simulator. Two options of this function go beyond the simulator's own.

    `pointing` is a table with `t_ref_utc_ns` (the time at which Polaris sits at the optical axis
    plus the offset), and the optional `roll_deg`, `offset_x_arcsec`, and `offset_y_arcsec`. Without
    it the simulator takes the time of its creation as the reference time. A fixed reference time
    lets another process compute where every simulated star is. `polaris = "real"` puts Polaris
    where the survey code predicts the real one and every star at its apparent place of date, so
    the sky turns about the true pole of date, and `polaris_mag` makes Polaris fainter (see
    `seeingmon.services.simsky`).
    """
    from dataclasses import replace

    from seeingmon.drivers import create_driver
    from seeingmon.drivers.sim import SimOptions
    from seeingmon.drivers.sim.stars import Pointing

    rest = dict(options)
    table = rest.pop("pointing", None)
    polaris = rest.pop("polaris", "synthetic")
    polaris_mag = rest.pop("polaris_mag", None)
    if polaris not in ("synthetic", "real"):
        raise CameraConfigError("the sim option polaris is synthetic or real")
    if table is None and polaris == "synthetic":
        return create_driver("sim", profile=profile, clock=clock, options=rest)
    sim_options = SimOptions.from_mapping(rest)
    if table is not None:
        allowed = {"t_ref_utc_ns", "roll_deg", "offset_x_arcsec", "offset_y_arcsec"}
        if not isinstance(table, Mapping) or set(table) - allowed or "t_ref_utc_ns" not in table:
            raise CameraConfigError(
                "the sim option pointing is a table with t_ref_utc_ns, and optionally roll_deg, "
                "offset_x_arcsec, and offset_y_arcsec"
            )
        sim_options = replace(
            sim_options,
            pointing=Pointing(
                t_ref_utc_ns=int(table["t_ref_utc_ns"]),
                roll_deg=float(table.get("roll_deg", 0.0)),
                offset_arcsec=(
                    float(table.get("offset_x_arcsec", 0.0)),
                    float(table.get("offset_y_arcsec", 0.0)),
                ),
            ),
        )
    if polaris == "real":
        from seeingmon.services.simsky import apparent_places, sim_field

        magnitude = None if polaris_mag is None else float(polaris_mag)
        sim_options = replace(
            sim_options,
            stars=sim_field(sim_options.seed, polaris_mag=magnitude),
            places=apparent_places,
        )
    return create_driver("sim", profile=profile, clock=clock, options=sim_options)


def _create_fake(clock: Clock, options: Mapping[str, Any]) -> CameraDriver:
    rest = dict(options)
    hang_in = rest.pop("hang_in", None)
    unknown = set(rest) - set(FAKE_OPTIONS)
    if unknown:
        raise CameraConfigError(f"the fake driver has no option named {sorted(unknown)[0]!r}")
    kwargs: dict[str, Any] = {}
    for key, value in rest.items():
        kind = FAKE_OPTIONS[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CameraConfigError(f"the option {key!r} of the fake driver must be a number")
        kwargs[key] = kind(value)
    if hang_in is None:
        return FakeCameraDriver(clock, **kwargs)
    if hang_in not in HANGABLE:
        raise CameraConfigError(f"hang_in is one of {', '.join(HANGABLE)}")
    return HangingFakeDriver(clock, hang_in=str(hang_in), **kwargs)
