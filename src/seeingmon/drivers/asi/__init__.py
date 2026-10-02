"""The `asi` camera driver. `seeingmon.drivers.asi.create` builds it.

    from seeingmon.drivers import asi

    driver = asi.create(profile=config.profile, clock=clock, options={"library_path": "..."})

`create` follows the factory convention of `seeingmon.drivers.base`. It loads the vendor library
(see `seeingmon.hardware.asi.ctypes_api` for where it looks), builds the USB resetter and the call
watchdog, and returns an `AsiDriver`. To run the driver on a fake SDK, build `AsiDriver` yourself.
"""

from __future__ import annotations

from collections.abc import Mapping

from seeingmon.clock import Clock
from seeingmon.drivers.asi.driver import AsiDriver, CameraSettings, SavedControl
from seeingmon.drivers.asi.options import AsiOptions
from seeingmon.drivers.base import CameraConfigError
from seeingmon.hardware.asi.ctypes_api import load_asi_api
from seeingmon.hardware.asi.usb import LinuxUsbResetter
from seeingmon.hardware.asi.watchdog import CallWatchdog, exit_process_on_hang
from seeingmon.hardware.events import EventCallback
from seeingmon.profile.models import Profile


def create(
    *,
    profile: Profile | None,
    clock: Clock,
    options: Mapping[str, object] | None,
    on_event: EventCallback | None = None,
) -> AsiDriver:
    """Build the driver for the connected camera.

    `options` holds the keys of `AsiOptions`, such as `library_path`. An empty mapping or `None`
    uses the defaults. The profile is required, because it holds the readout modes and the
    timing. Raises `CameraConfigError` for a missing profile or a bad option, and
    `AsiLibraryError` when the vendor library is missing. The camera opens when you call `open`.
    """
    if profile is None:
        raise CameraConfigError("the asi driver needs a profile")
    opts = AsiOptions.from_mapping(options or {})
    api = load_asi_api(opts.library_path)
    resetter = (
        None
        if opts.usb_reset == "off"
        else LinuxUsbResetter(
            vendor_id=opts.usb_vendor_id,
            product_id=opts.usb_product_id,
            device=opts.usb_device,
            mode=opts.usb_reset,
            clock=clock,
        )
    )
    watchdog = (
        CallWatchdog(clock, exit_process_on_hang, poll_interval_s=opts.watchdog_poll_s)
        if opts.watchdog
        else None
    )
    return AsiDriver(
        api=api,
        profile=profile,
        clock=clock,
        options=opts,
        usb_resetter=resetter,
        watchdog=watchdog,
        watchdog_thread=opts.watchdog,
        on_event=on_event,
    )


__all__ = ["AsiDriver", "AsiOptions", "CameraSettings", "SavedControl", "create"]
