"""Helpers that the web tests share: small JPEG files, alignment states, and a reference `core`.

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
    METHOD_ALIGNMENT_STATE,
    METHOD_DARK_LIBRARY,
    METHOD_PING,
    METHOD_STATUS,
    METHOD_SUBMIT,
    RPC_CHANNEL,
    AlignmentFrameInfo,
    AlignmentState,
    DarkSetView,
    FocusView,
    HistogramView,
    OffsetView,
    SaturationView,
    SolvedView,
    TargetView,
    decode_command,
    encode_result,
    pack_frame,
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

    `frames` are the frames that every new `alignment` stream receives, in order. Set `handlers`
    entries before `start` to replace a method, for example to make it fail.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        key: ConnectionKey,
        *,
        backend: FakeCoreClient | None = None,
        frames: list[tuple[AlignmentState, bytes]] | None = None,
        window: StreamWindow | None = None,
    ) -> None:
        self.backend = backend or FakeCoreClient(instance="reference-core")
        self.frames = frames or []
        self.handlers: dict[str, Handler] = {
            METHOD_PING: self._ping,
            METHOD_STATUS: self._status,
            METHOD_SUBMIT: self._submit,
            METHOD_ALIGNMENT_STATE: self._alignment_state,
            METHOD_DARK_LIBRARY: self._dark_library,
        }
        self.raw_payloads: list[bytes] = []  # sent before the frames, to test a bad message
        self.streams_started = 0
        self.stream_senders: list[StreamSender] = []
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

    def _dark_library(self, params: Mapping[str, Any]) -> Any:
        return self.backend.dark_library().model_dump(mode="json")

    def _on_sender(self, sender: StreamSender, params: Mapping[str, Any]) -> None:
        self.streams_started += 1
        self.stream_senders.append(sender)
        threading.Thread(target=self._send_frames, args=(sender,), daemon=True).start()

    def _send_frames(self, sender: StreamSender) -> None:
        try:
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
        server = IpcServer(
            self.endpoint,
            self._key,
            {RPC_CHANNEL: self._rpc, ALIGNMENT_CHANNEL: stream},
            handshake_timeout_s=2.0,
            **server_options,
        )
        self.endpoint = server.start()
        self._server = server
        return server

    def end_streams(self) -> None:
        """Close every alignment stream from the server side, as a stopping `core` would."""
        for sender in self.stream_senders:
            sender.close()

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop()
        if self._rpc is not None:
            self._rpc.stop()
