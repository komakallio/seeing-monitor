"""The USB device reset that the third recovery step uses.

A ZWO camera on a Raspberry Pi can stop delivering frames after hours or days, and a reset of its
USB device often clears the fault. `UsbResetter` is the interface that the driver calls, and two
classes implement it:

- `LinuxUsbResetter` finds the camera through sysfs and resets it. It prefers the
  `USBDEVFS_RESET` ioctl on the device node, which needs write access to the node and no
  other privilege (the camera's udev rule grants it). It falls back to toggling the `authorized`
  attribute in sysfs, which needs root. The module imports `fcntl` only when it resets a device,
  so it imports on every platform.
- `FakeUsbResetter` (in `seeingmon.hardware.asi.fake`) makes a `FakeAsiSdk` disconnect and
  return, for tests.

A reset makes the device vanish and enumerate again, so the camera starts with power-on settings.
The driver reopens it and reapplies the stream settings afterward.

The Pi cannot switch the power of one USB port. A reset does not cut power. A power cycle is a
separate step (`seeingmon.hardware.power`).
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from seeingmon.clock import Clock, SystemClock
from seeingmon.drivers.base import CameraError

ZWO_VENDOR_ID = 0x03C3  # the USB vendor ID of ZWO, a public value
USBDEVFS_RESET = 0x5514  # the Linux ioctl number _IO('U', 20)
DEFAULT_SYSFS_ROOT = Path("/sys/bus/usb/devices")
DEFAULT_DEV_ROOT = Path("/dev/bus/usb")

MODES = ("auto", "ioctl", "sysfs")


class UsbResetError(CameraError):
    """The USB reset did not work."""


class UsbResetter(Protocol):
    def reset(self) -> None:
        """Reset the camera's USB device. Raises `UsbResetError` when that fails."""
        ...


@dataclass(frozen=True, slots=True)
class UsbDevice:
    """A USB device that sysfs lists: its sysfs name, bus number, and device number."""

    name: str
    bus: int
    address: int


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _read_int(path: Path, base: int) -> int | None:
    text = _read(path)
    if text is None:
        return None
    try:
        return int(text, base)
    except ValueError:
        return None


def _describe(error: Exception) -> str:
    """Name a failure without the paths that an OS error message carries."""
    return str(error) if isinstance(error, UsbResetError) else error.__class__.__name__


def _default_ioctl(fd: int, request: int) -> None:
    try:
        fcntl = importlib.import_module("fcntl")  # Linux and macOS only
    except ImportError:
        raise UsbResetError("the ioctl reset needs Linux") from None
    fcntl.ioctl(fd, request, 0)


class LinuxUsbResetter:
    """Reset a USB device on Linux, found by vendor ID.

    Args:
        vendor_id: The USB vendor ID of the camera. The default is ZWO's.
        product_id: Also match this product ID, or `None` to match any.
        device: Match only the device with this sysfs name (such as `1-2`), or `None` for any.
            Set it when more than one device of the vendor is attached.
        mode: `ioctl`, `sysfs`, or `auto`, which tries the ioctl first and then sysfs.
        sysfs_root: The directory that lists USB devices.
        dev_root: The directory that holds the device nodes.
        ioctl: Replaces the ioctl call, as `(fd, request)`. A test passes a recorder.
        clock: The time source for the pause between deauthorizing and authorizing.
        settle_s: The pause between the two sysfs writes.
    """

    def __init__(
        self,
        *,
        vendor_id: int = ZWO_VENDOR_ID,
        product_id: int | None = None,
        device: str | None = None,
        mode: str = "auto",
        sysfs_root: Path = DEFAULT_SYSFS_ROOT,
        dev_root: Path = DEFAULT_DEV_ROOT,
        ioctl: Callable[[int, int], None] | None = None,
        clock: Clock | None = None,
        settle_s: float = 0.5,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}: {mode!r}")
        self._vendor_id = vendor_id
        self._product_id = product_id
        self._device = device
        self._mode = mode
        self._sysfs_root = sysfs_root
        self._dev_root = dev_root
        self._ioctl = ioctl or _default_ioctl
        self._clock = clock or SystemClock()
        self._settle_s = settle_s

    def find_devices(self) -> list[UsbDevice]:
        """The attached devices that match the vendor, the product, and the sysfs name."""
        found: list[UsbDevice] = []
        try:
            entries = sorted(self._sysfs_root.iterdir())
        except OSError:
            return found
        for entry in entries:
            if self._device is not None and entry.name != self._device:
                continue
            if _read_int(entry / "idVendor", 16) != self._vendor_id:
                continue  # an interface directory has no vendor ID, and other vendors differ
            if (
                self._product_id is not None
                and _read_int(entry / "idProduct", 16) != self._product_id
            ):
                continue
            bus, address = _read_int(entry / "busnum", 10), _read_int(entry / "devnum", 10)
            if bus is not None and address is not None:
                found.append(UsbDevice(entry.name, bus, address))
        return found

    def reset(self) -> None:
        devices = self.find_devices()
        if not devices:
            raise UsbResetError("no USB device matches the camera's vendor ID")
        for device in devices:
            self._reset_one(device)

    def _reset_one(self, device: UsbDevice) -> None:
        failures: list[str] = []
        if self._mode in ("auto", "ioctl"):
            try:
                self._reset_by_ioctl(device)
                return
            except (OSError, UsbResetError) as error:
                failures.append(f"ioctl: {_describe(error)}")
        if self._mode in ("auto", "sysfs"):
            try:
                self._reset_by_sysfs(device)
                return
            except OSError as error:
                failures.append(f"sysfs: {_describe(error)}")
        raise UsbResetError("the USB reset failed (" + "; ".join(failures) + ")")

    def _reset_by_ioctl(self, device: UsbDevice) -> None:
        node = self._dev_root / f"{device.bus:03d}" / f"{device.address:03d}"
        descriptor = os.open(node, os.O_WRONLY)
        try:
            self._ioctl(descriptor, USBDEVFS_RESET)
        finally:
            os.close(descriptor)

    def _reset_by_sysfs(self, device: UsbDevice) -> None:
        attribute = self._sysfs_root / device.name / "authorized"
        attribute.write_text("0", encoding="ascii")
        self._clock.sleep(self._settle_s)
        attribute.write_text("1", encoding="ascii")
