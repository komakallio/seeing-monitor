"""The `[services]` configuration section.

The defaults live in `config/default.d/services.toml`. A real installation overrides a few
values in `local/config.toml` or through `SEEINGMON_SERVICES__<KEY>` environment variables, and
sets the connection key there (see `seeingmon.services.ipc.keys`).

    config = load_config()
    services = config.section("services", ServicesConfig)
    endpoint = services.endpoint("acquire")
    key = services.load_key()
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator

from seeingmon.clock import Clock, ScaledClock, SystemClock
from seeingmon.config import SectionModel
from seeingmon.services.clockprobe import make_probe
from seeingmon.services.core.settings import CoreSettings
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.keys import ConnectionKey, load_connection_key

MIB = 1024 * 1024


class ClockSettings(SectionModel):
    """The clock of a process.

    `system` is the operating system clock, and a real installation uses it. Its `probe` tells the
    clock whether it is synchronized: `adjtimex` reads the state of the kernel clock (Linux), `none`
    leaves the state unknown, and `auto` picks `adjtimex` on Linux. `scaled` runs
    faster than real time for end-to-end tests: every process that must agree on the time gets
    the same `start_utc_ns`, `origin_real_ns`, and `speed` (see `seeingmon.clock.ScaledClock`).
    """

    kind: Literal["system", "scaled"] = "system"
    probe: Literal["auto", "adjtimex", "none"] = "auto"
    speed: float = Field(1.0, gt=0)
    start_utc_ns: int | None = None
    origin_real_ns: int | None = None

    @model_validator(mode="after")
    def _scaled_needs_a_shared_origin(self) -> ClockSettings:
        if self.kind == "scaled" and (self.start_utc_ns is None or self.origin_real_ns is None):
            raise ValueError("a scaled clock needs start_utc_ns and origin_real_ns")
        return self

    def build(self) -> Clock:
        """Build the clock."""
        if self.kind == "system":
            return SystemClock(status_probe=make_probe(self.probe))
        if self.start_utc_ns is None or self.origin_real_ns is None:  # the validator rules this out
            raise ValueError("a scaled clock needs start_utc_ns and origin_real_ns")
        return ScaledClock(
            start_utc_ns=self.start_utc_ns,
            origin_real_ns=self.origin_real_ns,
            speed=self.speed,
        )


class CallTimeouts(SectionModel):
    """How long each driver call may take before the process counts it as a hang.

    A call that exceeds its timeout makes `acquire` exit, and systemd restarts it. A frame
    read gets its own timeout plus `read_grace_s`.
    """

    open_s: float = Field(60.0, gt=0)
    close_s: float = Field(30.0, gt=0)
    capabilities_s: float = Field(10.0, gt=0)
    configure_s: float = Field(60.0, gt=0)
    start_s: float = Field(30.0, gt=0)
    stop_s: float = Field(30.0, gt=0)
    move_roi_s: float = Field(10.0, gt=0)
    temperature_s: float = Field(10.0, gt=0)
    dropped_s: float = Field(10.0, gt=0)
    recover_s: float = Field(180.0, gt=0)
    read_grace_s: float = Field(5.0, gt=0)


class AcquireSettings(SectionModel):
    """Settings of the `acquire` process."""

    driver: str = Field("sim", min_length=1, max_length=64)
    driver_options: dict[str, Any] = Field(default_factory=dict)

    # The frame queue between the capture thread and the sender thread. A full queue drops its
    # oldest frame and counts it. `queue_max_bytes` also bounds the memory of large frames.
    queue_depth: int = Field(256, ge=1)
    queue_max_bytes: int = Field(96 * MIB, ge=1)

    # Batches. A receiver that asks for it gets the frames of the last `batch_delay_s` in one
    # message, up to `stream_batch_frames` frames and `batch_bytes` bytes (see
    # `seeingmon.services.acquire.service`). A delay of 0 sends each frame at once.
    batch_delay_s: float = Field(0.06, ge=0, le=1.0)
    batch_bytes: int = Field(256 * 1024, ge=1024)

    # Waits. A frame read waits `read_timeout_factor` frame periods plus `read_timeout_margin_s`.
    read_timeout_factor: float = Field(2.0, ge=1.0)
    read_timeout_margin_s: float = Field(0.5, gt=0)
    default_frame_period_s: float = Field(1.0, gt=0)
    error_backoff_s: float = Field(0.05, ge=0)

    # Timing. `auto` keeps the time of a frame that the driver stamped exactly (a simulated or
    # replayed frame), and `stamp` always fits the arrival times. `driver` never changes the time.
    time_source: Literal["auto", "stamp", "driver"] = "auto"
    latency_s: float = Field(0.0, ge=0)
    latency_sigma_s: float = Field(0.005, ge=0)
    arrival_jitter_s: float = Field(0.002, ge=0)
    unknown_clock_error_s: float = Field(0.1, ge=0)
    invalid_clock_error_s: float = Field(86_400.0, ge=0)
    fit_window: int = Field(128, ge=8)
    fit_warmup: int = Field(20, ge=4)
    outlier_sigmas: float = Field(6.0, gt=0)
    outlier_floor_s: float = Field(0.003, ge=0)
    step_frames: int = Field(3, ge=2)

    # Drops. A gap between frames longer than `gap_factor` frame periods counts as lost frames.
    gap_factor: float = Field(1.5, gt=1.0)

    raise_priority: bool = True

    # How `acquire` calls the driver from two threads. `serialize` lets one thread in at a time,
    # for a driver that is not thread-safe (the fakes, `sim`, and `replay`). `concurrent` trusts a
    # driver that is, such as `asi`. `auto` serializes unless the driver says that it is safe.
    driver_threads: Literal["auto", "serialize", "concurrent"] = "auto"

    # The watchdog thread checks the process each tick, sends a heartbeat to systemd when it runs
    # under it, and logs a health summary.
    watchdog_tick_s: float = Field(0.25, gt=0)
    heartbeat_interval_s: float = Field(5.0, gt=0)
    health_log_interval_s: float = Field(60.0, gt=0)

    call_timeouts: CallTimeouts = Field(default_factory=CallTimeouts)

    @model_validator(mode="after")
    def _check_fit(self) -> AcquireSettings:
        if self.fit_warmup > self.fit_window:
            raise ValueError("fit_warmup must not exceed fit_window")
        return self


class ServicesConfig(SectionModel):
    """The `[services]` section: connections, limits, and the settings of `acquire` and `core`.

    An empty address selects the default for the platform (see `Endpoint.default`). The
    connection key has three sources (see `seeingmon.services.ipc.keys`), and the value of
    `connection_key` stays hidden in `repr` and in the effective configuration.
    """

    acquire_address: str = ""
    core_address: str = ""

    connection_key: SecretStr | None = None
    connection_key_file: str = ""
    connection_key_credential: str = "seeingmon-connection-key"

    connect_timeout_s: float = Field(5.0, gt=0)
    rpc_timeout_s: float = Field(30.0, gt=0)
    handshake_timeout_s: float = Field(5.0, gt=0)

    # Flow control of a frame stream: how many messages and bytes the receiver accepts before
    # it has consumed the first ones.
    stream_window_messages: int = Field(64, ge=1)
    stream_window_bytes: int = Field(64 * MIB, ge=1)
    # The most frames that one message of the stream holds. `core` asks for this many, and
    # `acquire` sends up to this many. 1 turns batches off.
    stream_batch_frames: int = Field(16, ge=1, le=1024)
    max_rpc_bytes: int = Field(1 * MIB, ge=1024)
    max_frame_bytes: int = Field(128 * MIB, ge=1024)

    clock: ClockSettings = Field(default_factory=ClockSettings)
    acquire: AcquireSettings = Field(default_factory=AcquireSettings)
    core: CoreSettings = Field(default_factory=CoreSettings)

    @field_validator("connection_key", mode="before")
    @classmethod
    def _numeric_key_is_text(cls, value: object) -> object:
        # An environment variable such as 123456789012345678 parses as a number.
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value

    def endpoint(
        self,
        role: Literal["acquire", "core"],
        *,
        platform: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Endpoint:
        """The endpoint that a service listens on: the configured address, or the default."""
        text = self.acquire_address if role == "acquire" else self.core_address
        return Endpoint.from_setting(text, role, platform=platform, env=env)

    def load_key(self, env: Mapping[str, str] | None = None) -> ConnectionKey:
        """Find the connection key. Raises `IpcConfigError` when none is configured."""
        value = None if self.connection_key is None else self.connection_key.get_secret_value()
        return load_connection_key(
            value=value,
            credential=self.connection_key_credential or None,
            file=self.connection_key_file or None,
            env=os.environ if env is None else env,
        )
