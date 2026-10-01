"""The options of the `asi` driver.

`create` builds the options from the mapping that it receives (`AsiOptions.from_mapping`). The
`acquire` process takes that mapping from `[services.acquire.driver_options]` in the
configuration, so you set `library_path` and the other keys there. Every key has a default, so an
empty mapping works on a camera with the vendor library on the system search path.

The values below are starting points that suit the reference camera. Phase 3 (commissioning)
tunes them on hardware. None of them is a measured result.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, ValidationError

from seeingmon.config import SectionModel
from seeingmon.drivers.base import CameraConfigError

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class AsiOptions(SectionModel):
    """Settings of the `asi` driver. Each field documents its unit and its effect."""

    library_path: str | None = None
    """The vendor library. `None` falls back to `SEEINGMON_ASI__LIBRARY_PATH`, then to the
    system search path."""

    camera_index: NonNegativeInt = 0
    """Which camera to open when several are attached, counted from 0."""

    discard_frames: Annotated[int, Field(ge=0, le=100)] = 1
    """Frames that `start` reads and drops after it starts video capture. The first frame after
    a mode change can be stale or partial."""

    roi_move_discard_frames: Annotated[int, Field(ge=0, le=100)] = 2
    """Frames that the next reads drop after `move_roi`, because the SDK buffer still holds
    frames from the old position."""

    geometry_check_interval: NonNegativeInt = 8
    """Check the ROI, binning, and position after every this many frames, and stop with
    `CameraConfigError` when they differ from the stream. The check costs two SDK calls. 1
    checks every frame, and 0 turns the check off."""

    temperature_interval_s: PositiveFloat = 5.0
    """How often a running video stream refreshes the sensor temperature that frames carry."""

    temperature_warmup_s: NonNegativeFloat = 0.3
    """How long after initialization a temperature of exactly 0 counts as not yet valid. The
    SDK returns 0 for about 250 ms."""

    call_timeout_s: PositiveFloat = 10.0
    """The deadline of an SDK call other than a frame read."""

    read_margin_s: PositiveFloat = 2.0
    """The time added to a frame read's own wait to form its watchdog deadline."""

    watchdog: bool = True
    """Run the call watchdog thread, which ends the process when an SDK call hangs."""

    watchdog_poll_s: PositiveFloat = 0.1
    """How often the watchdog thread checks the deadlines."""

    usb_reset: Literal["auto", "ioctl", "sysfs", "off"] = "auto"
    """How the third recovery step resets the USB device. `off` disables the step."""

    usb_vendor_id: Annotated[int, Field(ge=0, le=0xFFFF)] = 0x03C3
    """The USB vendor ID of the camera (ZWO)."""

    usb_product_id: Annotated[int, Field(ge=0, le=0xFFFF)] | None = None
    """Also match this USB product ID. `None` matches any product of the vendor."""

    usb_device: str | None = None
    """Match only the USB device with this sysfs name, such as `1-2`. Set it in the local
    configuration when more than one device of the vendor is attached."""

    usb_reenumerate_timeout_s: PositiveFloat = 15.0
    """How long the third recovery step waits for the camera to appear after the reset."""

    time_error_ms: NonNegativeFloat = 5.0
    """The uncertainty of the latency between the end of a frame and its arrival, in
    milliseconds. It adds to the clock's error bound in `Frame.t_err_ns`. Commissioning measures
    it with a light pulse."""

    status_interval_s: PositiveFloat = 5.0
    """How often the driver reads the clock status that decides the time quality of a frame."""

    @classmethod
    def from_mapping(cls, options: Mapping[str, object]) -> AsiOptions:
        """Validate a mapping of options. Raises `CameraConfigError` that names the keys.

        The message never shows a value, because a value can be a path.
        """
        try:
            return cls.model_validate(dict(options))
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in item['loc']) or 'options'}: {item['msg']}"
                for item in error.errors(include_url=False, include_input=False)
            )
            raise CameraConfigError(f"invalid asi driver options: {problems}") from None
