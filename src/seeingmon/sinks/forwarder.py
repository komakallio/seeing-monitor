"""The sink forwarder: it sends the rows of the store to every sink, at least once and in order.

`Forwarder` keeps one cursor for each sink and each record type in the store (`sink_cursor`). A
cursor is the last row ID that the sink acknowledged. A pass reads the rows after the cursor in
batches, calls `Sink.send`, and moves the cursor only after `send` returns, so a crash between the
two repeats a batch and loses none. A sink upserts by key, so a repeated batch changes nothing. A
new sink starts at row zero and backfills the whole history, and a restart resumes from the stored
cursors.

**Independence.** Each sink has its own state. A sink that fails never blocks another. A pass
sends at most `max_batches_per_pass` batches to a sink, so a long backfill does not starve the
others.

**Failures.** A `SinkError` with `retryable=True` keeps the cursor where it is and waits before
the next attempt. The wait starts at `backoff_initial_s`, grows by `backoff_factor` with each
consecutive failure, stops at `backoff_max_s`, and varies by up to `jitter` at random. The random
source is a seeded `random.Random` that you can pass in, and the waits use the monotonic clock of
the `Clock`, so a test with a `VirtualClock` runs days in a moment. The first failure of an outage
writes a `sink.unavailable` event, and the first success after it writes `sink.recovered`. A
`SinkError` with `retryable=False` parks the sink: the forwarder writes a `sink.parked` event,
leaves the cursor, and stops sending to that sink until you call `resume` or restart the service.
Any other exception from a sink counts as a retryable failure.

**Record types.** The forwarder sends the `table` record types. The per-frame `frame` metrics live
in segment files and have no row IDs, so no sink gets them from the forwarder.

**Backlog.** `backlog` counts the rows after each cursor for each sink and record type, and
`sink_backlog` sums them for each sink, which is the `sink_backlog` field of the `health` record.
`backlog_exceeded` names the sinks over `backlog_flag_rows`, which raise the `sink_backlog` flag.

**Threads.** Run the passes from one thread (`run_loop`). `status`, `backlog`, `sink_backlog`, and
`resume` are safe to call from other threads.
"""

from __future__ import annotations

import logging
import random
import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.records.sqlite_schema import table_record_types
from seeingmon.sinks.base import Sink, SinkError
from seeingmon.store.config import ForwarderConfig
from seeingmon.store.db import Store, StoreError
from seeingmon.store.events import EventEmitter

logger = logging.getLogger(__name__)

# The states of one sink in one pass, and in `status`.
IDLE = "idle"  # nothing was waiting
SENT = "sent"  # the pass sent rows
BACKOFF = "backoff"  # the sink waits before the next attempt
PARKED = "parked"  # a permanent error stopped the sink
FAILED = "failed"  # the sink failed in this pass
STORE_ERROR = "store_error"  # the store could not be read or updated
OK = "ok"  # `status`: the sink works, or has not failed yet

_MAX_EXPONENT = 64


@dataclass(frozen=True, slots=True)
class SinkPass:
    """What one pass did for one sink. `more` is true when rows are still waiting."""

    sink: str
    state: str
    rows: int = 0
    batches: int = 0
    more: bool = False


@dataclass(frozen=True, slots=True)
class ForwardReport:
    """The result of `run_once`: one `SinkPass` for each sink, in the order of the sinks."""

    passes: tuple[SinkPass, ...]

    @property
    def rows(self) -> int:
        """The rows that every sink acknowledged in this pass."""
        return sum(item.rows for item in self.passes)

    @property
    def progressed(self) -> bool:
        return self.rows > 0

    @property
    def more_pending(self) -> bool:
        """Whether a healthy sink still has rows waiting, so the loop should not sleep."""
        return any(item.more for item in self.passes)


@dataclass(frozen=True, slots=True)
class SinkStatus:
    """A snapshot of one sink. `retry_in_s` is the wait that remains, or `None`."""

    name: str
    state: str
    failures: int
    retry_in_s: float | None
    last_error: str | None
    last_success_utc_ns: int | None


@dataclass(slots=True)
class _SinkState:
    failures: int = 0
    next_attempt_ns: int = 0
    parked: bool = False
    last_error: str | None = None
    last_success_utc_ns: int | None = None
    outage_reported: bool = False


class Forwarder:
    """Sends the rows of a `Store` to sinks. See the module documentation.

    `config` defaults to `ForwarderConfig()`. Pass `events` to write the events of failures. Pass
    `rng` to make the jitter repeatable, for example `random.Random(1)`.
    """

    def __init__(
        self,
        store: Store,
        sinks: Sequence[Sink],
        clock: Clock,
        config: ForwarderConfig | None = None,
        *,
        events: EventEmitter | None = None,
        rng: random.Random | None = None,
    ) -> None:
        names = [sink.name for sink in sinks]
        if len(set(names)) != len(names):
            raise ValueError(f"every sink needs a unique name; got {names}")
        for sink in sinks:
            if sink.max_batch_rows < 1:
                raise ValueError(f"sink {sink.name!r} needs a max_batch_rows of at least 1")
        self._store = store
        self._sinks = list(sinks)
        self._clock = clock
        self._config = config if config is not None else ForwarderConfig()
        self._events = events
        self._rng = rng if rng is not None else random.Random()
        self._types = [cls.record_type for cls in table_record_types()]
        self._lock = threading.Lock()
        self._state = {sink.name: _SinkState() for sink in self._sinks}

    # --- passes ---

    def run_once(self) -> ForwardReport:
        """Send what is waiting to every sink that is ready. Call it from one thread.

        A sink that waits after a failure, or that a permanent error parked, is skipped.
        """
        return ForwardReport(tuple(self._pump(sink) for sink in self._sinks))

    def run_loop(self, should_stop: Callable[[], bool], *, max_passes: int | None = None) -> int:
        """Call `run_once` until `should_stop()` is true. Return the number of passes.

        The loop sleeps on the `Clock` when no sink has rows to send: for `poll_interval_s`, or
        less if a sink is due to retry sooner. Pass `threading.Event().is_set` to stop from
        another thread. `max_passes` ends the loop after that many passes, which suits a test.
        """
        passes = 0
        while not should_stop():
            report = self.run_once()
            passes += 1
            if max_passes is not None and passes >= max_passes:
                break
            if not report.more_pending:
                self._clock.sleep(self._idle_seconds())
        return passes

    def _idle_seconds(self) -> float:
        wait = self._config.poll_interval_s
        now = self._clock.monotonic_ns()
        with self._lock:
            due = [
                (state.next_attempt_ns - now) / NS_PER_S
                for state in self._state.values()
                if not state.parked and state.next_attempt_ns > now
            ]
        return max(0.01, min([wait, *due]))

    def _types_for(self, sink: Sink) -> list[str]:
        return [record_type for record_type in self._types if sink.accepts(record_type)]

    def _pump(self, sink: Sink) -> SinkPass:
        state = self._state[sink.name]
        now = self._clock.monotonic_ns()
        with self._lock:
            if state.parked:
                return SinkPass(sink.name, PARKED)
            if now < state.next_attempt_ns:
                return SinkPass(sink.name, BACKOFF)
        limit = min(sink.max_batch_rows, self._config.batch_rows)
        rows_sent = batches = 0
        for record_type in self._types_for(sink):
            while True:
                if batches >= self._config.max_batches_per_pass:
                    return SinkPass(sink.name, SENT, rows_sent, batches, more=True)
                try:
                    cursor = self._store.cursor(sink.name, record_type)
                    rows = self._store.after(record_type, cursor, limit)
                except (StoreError, sqlite3.Error) as exc:
                    logger.warning("the forwarder could not read the store: %s", exc)
                    return SinkPass(sink.name, STORE_ERROR, rows_sent, batches)
                if not rows:
                    break
                try:
                    sink.send(record_type, rows)
                except SinkError as exc:
                    self._failed(sink, record_type, exc.retryable, str(exc))
                    return SinkPass(sink.name, FAILED, rows_sent, batches)
                except Exception as exc:
                    # A bug in an adapter must not stop the other sinks. Report only the class
                    # of the exception, because its message can hold a secret.
                    self._failed(sink, record_type, True, f"unexpected {type(exc).__name__}")
                    return SinkPass(sink.name, FAILED, rows_sent, batches)
                try:
                    self._store.advance_cursor(
                        sink.name, record_type, rows[-1].row_id, t_utc_ns=self._clock.utc_ns()
                    )
                except (StoreError, sqlite3.Error) as exc:
                    # The sink took the batch, so it arrives again: delivery is at least once.
                    logger.warning("the forwarder could not save a cursor: %s", exc)
                    return SinkPass(sink.name, STORE_ERROR, rows_sent, batches)
                self._succeeded(sink, record_type)
                rows_sent += len(rows)
                batches += 1
                if len(rows) < limit:
                    break
        return SinkPass(sink.name, SENT if batches else IDLE, rows_sent, batches)

    # --- state changes and events ---

    def _delay_s(self, failures: int) -> float:
        config = self._config
        base = config.backoff_initial_s * config.backoff_factor ** min(failures - 1, _MAX_EXPONENT)
        spread = config.jitter * (2 * self._rng.random() - 1)
        return max(0.0, min(config.backoff_max_s, base * (1 + spread)))

    def _failed(self, sink: Sink, record_type: str, retryable: bool, message: str) -> None:
        state = self._state[sink.name]
        delay = 0.0
        announce = True
        with self._lock:
            state.last_error = message
            if retryable:
                state.failures += 1
                delay = self._delay_s(state.failures)
                state.next_attempt_ns = self._clock.monotonic_ns() + round(delay * NS_PER_S)
                announce = not state.outage_reported  # one event for each outage
                state.outage_reported = True
            else:
                state.parked = True
            failures = state.failures
        detail = {"sink": sink.name, "record_type": record_type, "failures": failures}
        if not retryable:
            self._emit(
                "error",
                "sink.parked",
                f"Sink {sink.name} stopped, because retrying cannot help: {message}. "
                "Fix the cause and restart the service.",
                detail,
            )
        elif announce:
            self._emit(
                "warning",
                "sink.unavailable",
                f"Sink {sink.name} did not take a batch of {record_type} rows: {message}. "
                "The forwarder keeps the rows and retries with a growing wait.",
                {**detail, "retry_in_s": round(delay, 3)},
            )

    def _succeeded(self, sink: Sink, record_type: str) -> None:
        state = self._state[sink.name]
        with self._lock:
            failures = state.failures
            recovered = state.outage_reported
            state.failures = 0
            state.next_attempt_ns = 0
            state.last_error = None
            state.last_success_utc_ns = self._clock.utc_ns()
            state.outage_reported = False
        if recovered:
            self._emit(
                "info",
                "sink.recovered",
                f"Sink {sink.name} takes rows again after {failures} failed attempts.",
                {"sink": sink.name, "record_type": record_type, "failures": failures},
            )

    def _emit(self, level: str, kind: str, message: str, detail: Mapping[str, object]) -> None:
        if self._events is not None:
            self._events.emit(level, kind, message, detail)

    # --- control and reports ---

    def resume(self, sink: str) -> bool:
        """Let a parked sink try again now, and clear its backoff. Return whether it was waiting."""
        with self._lock:
            state = self._state.get(sink)
            if state is None:
                raise KeyError(f"unknown sink {sink!r}")
            waiting = state.parked or state.failures > 0
            state.parked = False
            state.failures = 0
            state.next_attempt_ns = 0
            return waiting

    def status(self) -> dict[str, SinkStatus]:
        """A snapshot of every sink: its state, its failures, and the wait that remains."""
        now = self._clock.monotonic_ns()
        result: dict[str, SinkStatus] = {}
        with self._lock:
            for name, state in self._state.items():
                if state.parked:
                    label = PARKED
                elif state.next_attempt_ns > now:
                    label = BACKOFF
                else:
                    label = OK
                wait = (state.next_attempt_ns - now) / NS_PER_S
                result[name] = SinkStatus(
                    name=name,
                    state=label,
                    failures=state.failures,
                    retry_in_s=wait if label == BACKOFF else None,
                    last_error=state.last_error,
                    last_success_utc_ns=state.last_success_utc_ns,
                )
        return result

    def backlog(self) -> dict[str, dict[str, int]]:
        """Count the rows that each sink has not acknowledged, by record type.

        The result maps each sink name to a map from a record type that the sink accepts to the
        number of rows after its cursor.
        """
        counts: dict[str, dict[str, int]] = {}
        for sink in self._sinks:
            counts[sink.name] = {
                record_type: self._store.count_after(
                    record_type, self._store.cursor(sink.name, record_type)
                )
                for record_type in self._types_for(sink)
            }
        return counts

    def sink_backlog(self) -> dict[str, int]:
        """The unsent rows of each sink, as the `sink_backlog` field of a `health` record."""
        return {name: sum(per_type.values()) for name, per_type in self.backlog().items()}

    def backlog_exceeded(self) -> list[str]:
        """The names of the sinks with more unsent rows than `backlog_flag_rows`."""
        limit = self._config.backlog_flag_rows
        return [name for name, rows in self.sink_backlog().items() if rows > limit]

    def close(self) -> None:
        """Call `close` on every sink that has one."""
        for sink in self._sinks:
            closer = getattr(sink, "close", None)
            if callable(closer):
                closer()
