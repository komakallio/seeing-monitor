"""The client of the commissioning commands: queue a task in the running `core`, and wait for it.

`seeingmon burst`, `sweep`, and `replay` connect to `core` over the local connection layer with the
role `cli`, send the command with the `submit` method, and print the answer. With `--wait` they
poll the `results` method (see `seeingmon.services.core.rpc`) until the result of their task
appears. A task runs at the next cycle boundary of the scheduler, so it may start after a short
wait, and a burst or replay may take minutes.

`cells_from_result` rebuilds the cells of a sweep from the JSON of its result, so the command prints
the same table as the scheduler's own formatter.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from seeingmon.clock import NS_PER_S, Clock, SystemClock
from seeingmon.scheduler.commands import Command, CommandResult
from seeingmon.scheduler.commission import SweepCell, SweepCellResult
from seeingmon.services.ipc.codec import CodecError, as_mapping
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.errors import IpcError, RpcError
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.rpc import RpcClient, connect_rpc
from seeingmon.services.web.contract import (
    METHOD_SUBMIT,
    RPC_CHANNEL,
    decode_result,
    encode_command,
)

_log = logging.getLogger(__name__)

METHOD_RESULTS = "results"


class CoreCommandError(Exception):
    """`core` cannot be reached, refused the key, or answered something unusable."""


@dataclass(frozen=True, slots=True)
class WaitOutcome:
    """The result of a task, as `core` reports it, or `None` when the wait ran out."""

    result: Mapping[str, Any] | None
    waited_s: float


class CoreCommandClient:
    """A connection to `core` for the commissioning commands. Close it when done."""

    def __init__(
        self,
        endpoint: Endpoint,
        key: ConnectionKey,
        *,
        connect_timeout_s: float = 5.0,
        handshake_timeout_s: float = 5.0,
        rpc_timeout_s: float = 30.0,
        clock: Clock | None = None,
    ) -> None:
        self._clock = SystemClock() if clock is None else clock
        try:
            self._rpc: RpcClient
            self._rpc, _ = connect_rpc(
                endpoint,
                key,
                {"role": "cli"},
                channel=RPC_CHANNEL,
                connect_timeout_s=connect_timeout_s,
                handshake_timeout_s=handshake_timeout_s,
                default_timeout_s=rpc_timeout_s,
                clock=self._clock,
                name="cli-rpc",
            )
        except IpcError as error:
            raise CoreCommandError(f"cannot reach core: {error}") from None

    def submit(self, command: Command) -> CommandResult:
        """Send a command. A rejection is a result, and anything else is an error."""
        try:
            answer = self._rpc.call(METHOD_SUBMIT, {"command": encode_command(command)})
            return decode_result(answer)
        except (IpcError, RpcError, CodecError, TypeError) as error:
            raise CoreCommandError(f"core did not take the command: {error}") from None

    def results(self) -> list[Mapping[str, Any]]:
        """The latest commissioning results of `core`, oldest first."""
        try:
            answer = as_mapping(self._rpc.call(METHOD_RESULTS), "results answer")
            items = answer.get("results")
            if not isinstance(items, list):
                raise CodecError("results answer has no list of results")
            return [as_mapping(item, "result") for item in items]
        except (IpcError, RpcError, CodecError) as error:
            raise CoreCommandError(f"core did not give its results: {error}") from None

    def wait_for(self, task_id: int, *, timeout_s: float, poll_s: float = 1.0) -> WaitOutcome:
        """Poll until the result of a task appears, or the time runs out."""
        started = self._clock.monotonic_ns()
        while True:
            for result in self.results():
                if result.get("task_id") == task_id:
                    waited = (self._clock.monotonic_ns() - started) / NS_PER_S
                    return WaitOutcome(result, waited)
            waited = (self._clock.monotonic_ns() - started) / NS_PER_S
            if waited >= timeout_s:
                return WaitOutcome(None, waited)
            self._clock.sleep(min(poll_s, max(timeout_s - waited, 0.0)))

    def close(self) -> None:
        self._rpc.close("the command is done")


def cells_from_result(data: Mapping[str, Any]) -> list[SweepCellResult]:
    """The cells of a sweep result, rebuilt from the JSON that `core` sent."""
    cells: list[SweepCellResult] = []
    for item in data.get("cells", []):
        cell = item["cell"]
        roi = item.get("roi_px")
        cells.append(
            SweepCellResult(
                cell=SweepCell(
                    str(cell["mode"]),
                    int(cell["exposure_us"]),
                    int(cell["gain"]),
                    float(cell["roi_arcmin"]),
                ),
                status=str(item["status"]),
                note=item.get("note"),
                roi_px=None if roi is None else (int(roi[0]), int(roi[1])),
                n_frames=int(item.get("n_frames", 0)),
                n_dropped=int(item.get("n_dropped", 0)),
                frame_rate_hz=item.get("frame_rate_hz"),
                drop_rate=item.get("drop_rate"),
                saturated_fraction=item.get("saturated_fraction"),
                peak_fraction_mean=item.get("peak_fraction_mean"),
                background_fraction=item.get("background_fraction"),
                snr_median=item.get("snr_median"),
                star_found_fraction=item.get("star_found_fraction"),
                n_windows=int(item.get("n_windows", 0)),
                centroid_noise_px=item.get("centroid_noise_px"),
                image_motion_rms_arcsec=item.get("image_motion_rms_arcsec"),
                seeing_fwhm_arcsec=item.get("seeing_fwhm_arcsec"),
            )
        )
    return cells
