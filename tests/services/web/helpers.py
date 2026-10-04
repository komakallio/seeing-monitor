"""Helpers that the web tests share: small images, states of the live views, and a reference `core`.

`ReferenceCore` is the server side of `seeingmon.services.web.contract`, built on the real
connection layer and answering from a `FakeCoreClient`. It shows what `core` has to serve, and the
tests of `RpcCoreClient` run against it.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Mapping
from typing import Any

from PIL import Image

from seeingmon.services.ipc.codec import as_mapping
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.rpc import Handler, RpcService
from seeingmon.services.ipc.server import IpcServer
from seeingmon.services.ipc.stream import StreamSender, StreamService, StreamWindow
from seeingmon.services.web.contract import (
    ALIGNMENT_CHANNEL,
    METHOD_ALIGNMENT_RESET_FOCUS,
    METHOD_ALIGNMENT_STATE,
    METHOD_DARK_LIBRARY,
    METHOD_FLAT_ACTIVATE,
    METHOD_FLAT_DELETE,
    METHOD_FLAT_IMAGE,
    METHOD_FLAT_LIBRARY,
    METHOD_LIVE_SEEING,
    METHOD_PING,
    METHOD_STATUS,
    METHOD_SUBMIT,
    POLARIS_CHANNEL,
    RPC_CHANNEL,
    AlignmentFrameInfo,
    AlignmentState,
    DarkSetView,
    FlatActionView,
    FocusView,
    HistogramView,
    LiveSeeingView,
    OffsetView,
    PolarisStar,
    PolarisState,
    PolarisStretch,
    RoiView,
    SaturationView,
    SolvedView,
    TargetView,
    decode_command,
    encode_flat_image,
    encode_result,
    pack_frame,
    pack_polaris_frame,
)
from seeingmon.services.web.core_client import FakeCoreClient

TOKEN = "t0ken-for-the-tests-QRSTUVWXYZ-ghijklmn"


def bearer(token: str = TOKEN) -> dict[str, str]:
    """The headers of a request that carries the token."""
    return {"Authorization": f"Bearer {token}"}


def tiny_jpeg(shade: int = 0, size: tuple[int, int] = (16, 12)) -> bytes:
    """A real JPEG image of one gray level. Different shades give different bytes."""
    buffer = io.BytesIO()
    Image.new("L", size, shade).save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def tiny_png(shade: int = 0, size: tuple[int, int] = (16, 12)) -> bytes:
    """A real 8-bit grayscale PNG image of one gray level. Different shades give different bytes."""
    buffer = io.BytesIO()
    Image.new("L", size, shade).save(buffer, "PNG", compress_level=1)
    return buffer.getvalue()


def live_seeing_view(**changes: Any) -> LiveSeeingView:
    """A rolling seeing value with synthetic numbers. Keyword arguments replace fields."""
    fields: dict[str, Any] = {
        "t_utc_ns": 1_790_000_000_000_000_000,
        "span_s": 10.0,
        "n_frames": 820,
        "n_usable": 815,
        "valid_fraction": 0.994,
        "seeing_fwhm_arcsec": 1.62,
        "seeing_fwhm_structure_arcsec": 1.55,
        "r0_cm": 6.1,
        "r0_structure_cm": 6.4,
        "image_motion_rms_x_arcsec": 0.71,
        "image_motion_rms_y_arcsec": 0.66,
        "width_fwhm_arcsec": 2.9,
        "stream_id": 3,
        "readout_mode": "bin1",
        "exposure_us": 2000,
        "flags": [],
        "quality": {},
    }
    fields.update(changes)
    return LiveSeeingView(**fields)


def polaris_state(seq: int = 1, *, live: bool = True, found: bool = True) -> PolarisState:
    """A complete state of a Polaris frame with synthetic values."""
    star = (
        PolarisStar(found=True, x=64.3, y=63.8, peak_fraction=0.31, fwhm_arcsec=2.7)
        if found
        else PolarisStar(found=False)
    )
    return PolarisState(
        seq=seq,
        t_utc="2026-10-01T21:00:00.123Z",
        t_utc_ns=1_790_000_000_123_000_000,
        stream_id=3,
        mode="bin1",
        exposure_us=2000,
        gain=0,
        roi=RoiView(x=2008, y=1347, width=128, height=128),
        scale_arcsec_px=1.91,
        fast_fps=82.1,
        image_type="image/png",
        image_width=128,
        image_height=128,
        star=star,
        stretch=PolarisStretch(black_dn=480.0, white_dn=21_000.0),
        live_seeing=live_seeing_view() if live else None,
    )


def alignment_state(seq: int = 1, *, active: bool = True) -> AlignmentState:
    """A complete alignment state with synthetic values."""
    return AlignmentState(
        active=active,
        t_utc="2026-10-01T21:00:00.000000Z",
        frame=AlignmentFrameInfo(
            seq=seq,
            width_px=4144,
            height_px=2822,
            readout_mode="bin2",
            exposure_s=0.5,
            gain=120,
            plate_scale_arcsec_px=3.82,
        ),
        target=TargetView(x_px=2072.0, y_px=1411.0, roll_deg=10.0),
        solved=SolvedView(
            x_px=2080.0, y_px=1405.0, roll_deg=10.5, n_matched=40, rms_arcsec=3.1, age_s=0.4
        ),
        offset=OffsetView(
            dx_px=8.0,
            dy_px=-6.0,
            distance_px=10.0,
            dx_arcsec=30.56,
            dy_arcsec=-22.92,
            distance_arcsec=38.2,
            roll_deg=0.5,
        ),
        focus=FocusView(fwhm_px=2.4, best_fwhm_px=2.1, n_stars=35),
        histogram=HistogramView(counts=[900, 300, 90, 20, 5, 1, 0, 2], min_dn=0, max_dn=65535),
        saturation=SaturationView(fraction=0.0004, warning=False),
    )


def dark_set(name: str, temperature_c: float, age_days: float) -> DarkSetView:
    return DarkSetView(
        name=name,
        t_utc="2026-09-01T00:00:00Z",
        age_days=age_days,
        temperature_c=temperature_c,
        temperature_spread_c=0.3,
        exposure_s=30.0,
        n_frames=9,
        n_bias_frames=9,
        rate_e_per_s=0.05,
        hot_pixels=150,
    )


class ReferenceCore:
    """The server side of the web contract, on the real connection layer.

    `frames` are the frames that every new `alignment` stream receives, in order, and
    `polaris_frames` are those of every new `polaris` stream. Set `handlers` entries before `start`
    to replace a method, for example to make it fail.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        key: ConnectionKey,
        *,
        backend: FakeCoreClient | None = None,
        frames: list[tuple[AlignmentState, bytes]] | None = None,
        polaris_frames: list[tuple[PolarisState, bytes]] | None = None,
        window: StreamWindow | None = None,
    ) -> None:
        self.backend = backend or FakeCoreClient(instance="reference-core")
        self.frames = frames or []
        self.polaris_frames = polaris_frames or []
        self.handlers: dict[str, Handler] = {
            METHOD_PING: self._ping,
            METHOD_STATUS: self._status,
            METHOD_SUBMIT: self._submit,
            METHOD_ALIGNMENT_STATE: self._alignment_state,
            METHOD_ALIGNMENT_RESET_FOCUS: self._alignment_reset_focus,
            METHOD_DARK_LIBRARY: self._dark_library,
            METHOD_LIVE_SEEING: self._live_seeing,
            METHOD_FLAT_LIBRARY: self._flat_library,
            METHOD_FLAT_ACTIVATE: self._flat_activate,
            METHOD_FLAT_DELETE: self._flat_delete,
            METHOD_FLAT_IMAGE: self._flat_image,
        }
        self.raw_payloads: list[bytes] = []  # sent before the frames, to test a bad message
        self.raw_polaris_payloads: list[bytes] = []
        self.streams_started = 0
        self.polaris_streams_started = 0
        self.stream_senders: list[StreamSender] = []
        self.polaris_senders: list[StreamSender] = []
        self.sending_done = threading.Event()
        self.endpoint = endpoint  # after `start`, the address that clients use
        self._key = key
        self._window = window or StreamWindow(8, 8 * 1024 * 1024)
        self._server: IpcServer | None = None
        self._rpc: RpcService | None = None

    def _ping(self, params: Mapping[str, Any]) -> Any:
        return {"instance": "reference-core"}

    def _status(self, params: Mapping[str, Any]) -> Any:
        return self.backend.status().model_dump(mode="json")

    def _submit(self, params: Mapping[str, Any]) -> Any:
        command = decode_command(as_mapping(params, "params").get("command"))
        return encode_result(self.backend.submit(command))

    def _alignment_state(self, params: Mapping[str, Any]) -> Any:
        return self.backend.alignment_state().model_dump(mode="json")

    def _alignment_reset_focus(self, params: Mapping[str, Any]) -> Any:
        self.backend.alignment_reset_focus()
        return {"reset": True}

    def _dark_library(self, params: Mapping[str, Any]) -> Any:
        return self.backend.dark_library().model_dump(mode="json")

    def _live_seeing(self, params: Mapping[str, Any]) -> Any:
        live = self.backend.live_seeing()
        return None if live is None else live.model_dump(mode="json")

    def _flat_library(self, params: Mapping[str, Any]) -> Any:
        return self.backend.flat_library().model_dump(mode="json")

    @staticmethod
    def _version(params: Mapping[str, Any]) -> str | None:
        version = as_mapping(params, "params").get("version")
        return version if isinstance(version, str) else None

    def _flat_activate(self, params: Mapping[str, Any]) -> Any:
        version = self._version(params)
        if version is None:
            return FlatActionView(ok=False, reason="unknown", message="No such flat.").model_dump()
        return self.backend.flat_activate(version).model_dump(mode="json")

    def _flat_delete(self, params: Mapping[str, Any]) -> Any:
        version = self._version(params)
        if version is None:
            return FlatActionView(ok=False, reason="unknown", message="No such flat.").model_dump()
        return self.backend.flat_delete(version).model_dump(mode="json")

    def _flat_image(self, params: Mapping[str, Any]) -> Any:
        version = self._version(params)
        return encode_flat_image(None if version is None else self.backend.flat_image(version))

    def _on_sender(self, sender: StreamSender, params: Mapping[str, Any]) -> None:
        self.streams_started += 1
        self.stream_senders.append(sender)
        threading.Thread(target=self._send_frames, args=(sender,), daemon=True).start()

    def _on_polaris_sender(self, sender: StreamSender, params: Mapping[str, Any]) -> None:
        self.polaris_streams_started += 1
        self.polaris_senders.append(sender)
        payloads = [
            *self.raw_polaris_payloads,
            *(pack_polaris_frame(state, image) for state, image in self.polaris_frames),
        ]
        threading.Thread(target=self._send_frames, args=(sender, payloads), daemon=True).start()

    def _send_frames(self, sender: StreamSender, payloads: list[bytes] | None = None) -> None:
        try:
            if payloads is None:
                payloads = [
                    *self.raw_payloads,
                    *(pack_frame(state, jpeg) for state, jpeg in self.frames),
                ]
            for payload in payloads:
                if not sender.wait_credit(len(payload), 5.0):
                    return
                sender.send(payload)
            self.sending_done.set()
            while not sender.closed:
                sender.pump(0.05)
        except Exception:  # the client left, which ends the stream
            return
        finally:
            sender.close()

    def start(self, **server_options: Any) -> IpcServer:
        """Start the server. Returns it, so a test can stop it to simulate a crash."""
        self._rpc = RpcService(
            self.handlers, workers=2, worker_name="reference-core", max_connections=4
        )
        self._rpc.start()
        stream = StreamService(self._on_sender, max_window=self._window, name="reference-core")
        polaris = StreamService(
            self._on_polaris_sender, max_window=self._window, name="reference-core-polaris"
        )
        server = IpcServer(
            self.endpoint,
            self._key,
            {RPC_CHANNEL: self._rpc, ALIGNMENT_CHANNEL: stream, POLARIS_CHANNEL: polaris},
            handshake_timeout_s=2.0,
            **server_options,
        )
        self.endpoint = server.start()
        self._server = server
        return server

    def end_streams(self) -> None:
        """Close every live-view stream from the server side, as a stopping `core` would."""
        for sender in (*self.stream_senders, *self.polaris_senders):
            sender.close()

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop()
        if self._rpc is not None:
            self._rpc.stop()
