"""The USB reset: the Linux resetter on a fake sysfs tree, and the fake resetter."""

from __future__ import annotations

from pathlib import Path

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.hardware.asi.fake import FakeAsiSdk, FakeUsbResetter
from seeingmon.hardware.asi.usb import (
    USBDEVFS_RESET,
    ZWO_VENDOR_ID,
    LinuxUsbResetter,
    UsbDevice,
    UsbResetError,
    UsbResetter,
)


def add_device(
    sysfs: Path, dev: Path, name: str, vendor: str, product: str, bus: int, address: int
) -> None:
    folder = sysfs / name
    folder.mkdir(parents=True)
    (folder / "idVendor").write_text(vendor + "\n", encoding="ascii")
    (folder / "idProduct").write_text(product + "\n", encoding="ascii")
    (folder / "busnum").write_text(f"{bus}\n", encoding="ascii")
    (folder / "devnum").write_text(f"{address}\n", encoding="ascii")
    (folder / "authorized").write_text("1", encoding="ascii")
    node = dev / f"{bus:03d}" / f"{address:03d}"
    node.parent.mkdir(parents=True, exist_ok=True)
    node.write_bytes(b"")


@pytest.fixture
def tree(tmp_path: Path) -> tuple[Path, Path]:
    sysfs, dev = tmp_path / "sysfs", tmp_path / "dev"
    add_device(sysfs, dev, "1-2", "03c3", "294a", 1, 5)
    add_device(sysfs, dev, "1-3", "8087", "0029", 1, 6)
    add_device(sysfs, dev, "2-1", "03c3", "120a", 2, 3)
    (sysfs / "interface-1-2").mkdir()  # an interface directory has no vendor ID
    return sysfs, dev


class IoctlRecorder:
    def __init__(self, failure: OSError | None = None) -> None:
        self.calls: list[int] = []
        self.failure = failure

    def __call__(self, fd: int, request: int) -> None:
        assert fd >= 0
        self.calls.append(request)
        if self.failure is not None:
            raise self.failure


def resetter(tree: tuple[Path, Path], **options: object) -> LinuxUsbResetter:
    sysfs, dev = tree
    return LinuxUsbResetter(sysfs_root=sysfs, dev_root=dev, **options)  # type: ignore[arg-type]


class TestFindDevices:
    def test_finds_the_devices_of_the_vendor(self, tree: tuple[Path, Path]) -> None:
        assert resetter(tree).find_devices() == [UsbDevice("1-2", 1, 5), UsbDevice("2-1", 2, 3)]

    def test_the_product_id_narrows_the_match(self, tree: tuple[Path, Path]) -> None:
        assert resetter(tree, product_id=0x120A).find_devices() == [UsbDevice("2-1", 2, 3)]

    def test_the_sysfs_name_narrows_the_match(self, tree: tuple[Path, Path]) -> None:
        assert resetter(tree, device="1-2").find_devices() == [UsbDevice("1-2", 1, 5)]
        assert resetter(tree, device="1-3").find_devices() == []  # another vendor

    def test_another_vendor_matches_by_id(self, tree: tuple[Path, Path]) -> None:
        assert resetter(tree, vendor_id=0x8087).find_devices() == [UsbDevice("1-3", 1, 6)]

    def test_a_missing_sysfs_root_finds_nothing(self, tmp_path: Path) -> None:
        missing = LinuxUsbResetter(sysfs_root=tmp_path / "none", dev_root=tmp_path)
        assert missing.find_devices() == []

    def test_the_default_vendor_is_zwo(self) -> None:
        assert ZWO_VENDOR_ID == 0x03C3


class TestReset:
    def test_the_ioctl_resets_the_device_node(self, tree: tuple[Path, Path]) -> None:
        ioctl = IoctlRecorder()
        resetter(tree, device="1-2", ioctl=ioctl).reset()
        assert ioctl.calls == [USBDEVFS_RESET]

    def test_every_matching_device_resets(self, tree: tuple[Path, Path]) -> None:
        ioctl = IoctlRecorder()
        resetter(tree, ioctl=ioctl).reset()
        assert ioctl.calls == [USBDEVFS_RESET, USBDEVFS_RESET]

    def test_auto_falls_back_to_sysfs_when_the_ioctl_is_denied(
        self, tree: tuple[Path, Path]
    ) -> None:
        sysfs, _ = tree
        ioctl = IoctlRecorder(PermissionError("no write access"))
        clock = VirtualClock()
        started = clock.monotonic_ns()
        resetter(tree, device="1-2", ioctl=ioctl, clock=clock, settle_s=0.5).reset()
        assert ioctl.calls == [USBDEVFS_RESET]
        assert (sysfs / "1-2" / "authorized").read_text(encoding="ascii") == "1"  # authorized again
        assert clock.monotonic_ns() - started == 500_000_000

    def test_ioctl_mode_does_not_fall_back(self, tree: tuple[Path, Path]) -> None:
        ioctl = IoctlRecorder(PermissionError("no write access"))
        with pytest.raises(UsbResetError, match="ioctl"):
            resetter(tree, device="1-2", mode="ioctl", ioctl=ioctl).reset()

    def test_sysfs_mode_skips_the_ioctl(self, tree: tuple[Path, Path]) -> None:
        ioctl = IoctlRecorder()
        resetter(tree, device="1-2", mode="sysfs", ioctl=ioctl, clock=VirtualClock()).reset()
        assert ioctl.calls == []

    def test_a_failure_of_every_route_raises_without_paths(self, tree: tuple[Path, Path]) -> None:
        sysfs, dev = tree
        (dev / "001" / "005").unlink()  # the node is gone, so the ioctl cannot open it
        (sysfs / "1-2" / "authorized").unlink()
        (sysfs / "1-2" / "authorized").mkdir()  # and writing to a directory fails
        with pytest.raises(UsbResetError) as raised:
            resetter(tree, device="1-2", ioctl=IoctlRecorder(), clock=VirtualClock()).reset()
        message = str(raised.value)
        assert "ioctl" in message
        assert "sysfs" in message
        assert str(sysfs) not in message
        assert str(dev) not in message

    def test_no_matching_device_raises(self, tree: tuple[Path, Path]) -> None:
        with pytest.raises(UsbResetError, match="no USB device"):
            resetter(tree, vendor_id=0x1234).reset()

    def test_an_unknown_mode_is_refused(self, tree: tuple[Path, Path]) -> None:
        with pytest.raises(ValueError, match="mode"):
            resetter(tree, mode="power")

    def test_the_default_ioctl_reports_a_platform_without_fcntl(
        self, tree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(__import__("sys").modules, "fcntl", None)  # import fcntl fails
        with pytest.raises(UsbResetError, match="Linux"):
            resetter(tree, device="1-2", mode="ioctl").reset()


class TestFakeResetter:
    def test_satisfies_the_protocol_and_cycles_the_fake_camera(self) -> None:
        clock = VirtualClock()
        sdk = FakeAsiSdk(clock)
        fake: UsbResetter = FakeUsbResetter(sdk, reappear_after_s=3.0)
        sdk.open_camera(0)
        fake.reset()
        assert sdk.get_connected_camera_count() == 0
        clock.advance(3.0)
        assert sdk.get_connected_camera_count() == 1

    def test_a_scripted_failure_raises(self) -> None:
        fake = FakeUsbResetter()
        fake.failure = UsbResetError("no device")
        with pytest.raises(UsbResetError):
            fake.reset()
        assert fake.count == 1
