"""The activity of the demo: a scripted evening on a short cycle, so that the page moves.

The fake `core` of the demo has no camera, so it plays what the real scheduler reports as its
`activity` (see `seeingmon.scheduler.status.ActivityStatus`). In `auto` it repeats a playlist:

1. a survey step that has to find the pointing (the 1 ms frame, the long frame, and the wait for
   the solution),
2. three cycles, each with a fast period of six analysis windows, the survey step, and the idle
   slack until the next slot, and
3. a camera fault with two recovery steps, in which the status stays `auto` and not yet degraded.
   The fault reads as the scheduler reports a read that timed out: the cause `timeout`, and in
   words, `no frame arrived; the camera may be disconnected`.

A real cycle takes 3 minutes. The demo's takes 45 seconds, so that you see every phase within a
few minutes, and the playlist takes about 3 minutes. The other states show their own activity: the
brightness watch of `safe` (the gate opens after `GATE_OPEN_S`), the live view of `align`, the
phases of the dark session and the flat session of `commission`, and `paused`. The words come from
`seeingmon.scheduler.activity`, the same module that the real scheduler uses.

Every time of the activity is relative to the clock of the fake `core`: `since_utc_ns` is now minus
the time that the phase has run, and `ends_utc_ns` is now plus the time that it has left. So the
demo works on a clock whose UTC time stands still, and a page that subtracts the time of the
response from these values gets the right durations.
"""

from __future__ import annotations

from dataclasses import dataclass

from seeingmon.clock import NS_PER_S
from seeingmon.drivers.base import CameraTimeoutError
from seeingmon.scheduler import activity as words
from seeingmon.scheduler.faults import FaultCause, reason_text
from seeingmon.scheduler.status import ActivityPhase
from seeingmon.services.web.contract import ActivityView, DarkTaskView, FaultView, FlatTaskView

# The cycle of the demo. The names of the real settings are in the comments.
FAST_WINDOW_S = 5.0  # [scheduler.fast] analysis_window_s
FAST_WINDOWS = 6  # the windows of a fast period, as window_s / analysis_window_s
SHORT_EXPOSURE_S = 0.001  # [scheduler.survey] short_exposure_s
LONG_EXPOSURE_S = 6.0  # [scheduler.survey] long_exposure_s, which is 30 s in the real cycle
SURVEY_SHORT_S = 2.0  # the time that the short exposure takes with its overhead
SURVEY_LONG_S = 8.0
IDLE_S = 5.0
SOLVE_WAIT_S = 6.0  # [scheduler.survey] solve_wait_s
CYCLES = 3
CYCLE_S = FAST_WINDOW_S * FAST_WINDOWS + SURVEY_SHORT_S + SURVEY_LONG_S + IDLE_S
FAULT_STEPS = (("restart_capture", 2, 12.0), ("reopen", 3, 13.0))  # step, failures, length
FAULT_ERROR = CameraTimeoutError("no frame arrived in time")
FAULT_CAUSE = FaultCause.TIMEOUT
FAULT_REASON = reason_text(FAULT_CAUSE, FAULT_ERROR)
DEGRADED_AFTER = 5  # [scheduler.faults] degraded_after
SLOW_RETRY_S = 600.0  # [scheduler.faults] slow_retry_s
WATCH_INTERVAL_S = 10.0  # [scheduler.watch] interval_s
WATCH_EXPOSURE_S = 0.001
DAYLIGHT_RESUME_DEG = -4.0
GATE_OPEN_S = 25.0  # how long the demo sky holds the gate in `safe`
ALIGN_TIMEOUT_S = 1800.0  # [scheduler.align] idle_timeout_s

# What a state means when the fake `core` holds no reason of its own for it.
STATE_REASONS = {
    "auto": "the sky is dark enough",
    "safe": "the scheduler checks the sky first",
    "align": "you started the alignment",
    "paused": "you paused the scheduler",
    "commission": "a dark session runs",
}


@dataclass(frozen=True, slots=True)
class Segment:
    """One stretch of the playlist. Times are seconds from the start of the playlist."""

    phase: ActivityPhase
    start_s: float
    end_s: float
    label: str
    next_label: str | None
    detail: str | None = None
    episode_start_s: float | None = None  # a camera fault: when the episode began
    failures: int = 0  # a camera fault: the failures so far
    next_step: str | None = None  # a camera fault: the step that comes next


def _step_label() -> str:
    return words.survey_step_label(SHORT_EXPOSURE_S, LONG_EXPOSURE_S)


def build_playlist() -> tuple[tuple[Segment, ...], float]:
    """The segments of the playlist and its length in seconds."""
    segments: list[Segment] = []
    cursor = 0.0

    def add(
        phase: ActivityPhase,
        length_s: float,
        label: str,
        next_label: str | None,
        detail: str | None = None,
        *,
        episode_start_s: float | None = None,
        failures: int = 0,
        next_step: str | None = None,
    ) -> None:
        nonlocal cursor
        segments.append(
            Segment(
                phase,
                cursor,
                cursor + length_s,
                label,
                next_label,
                detail,
                episode_start_s,
                failures,
                next_step,
            )
        )
        cursor += length_s

    short = words.survey_frame_label(SHORT_EXPOSURE_S)
    long = words.survey_frame_label(LONG_EXPOSURE_S)
    # The first survey step has to find the pointing.
    add(ActivityPhase.SURVEY_SHORT, SURVEY_SHORT_S, short, long, words.SURVEY_SHORT_DETAIL)
    add(
        ActivityPhase.SURVEY_LONG,
        SURVEY_LONG_S,
        long,
        words.SOLVE_WAIT_LABEL,
        words.SURVEY_LONG_DETAIL,
    )
    add(
        ActivityPhase.SOLVE_WAIT,
        SOLVE_WAIT_S,
        words.SOLVE_WAIT_LABEL,
        words.FAST_LABEL,
        words.solve_wait_detail(SOLVE_WAIT_S),
    )
    for _ in range(CYCLES):
        add(ActivityPhase.FAST, FAST_WINDOW_S * FAST_WINDOWS, words.FAST_LABEL, _step_label())
        add(ActivityPhase.SURVEY_SHORT, SURVEY_SHORT_S, short, long, words.SURVEY_SHORT_DETAIL)
        add(
            ActivityPhase.SURVEY_LONG,
            SURVEY_LONG_S,
            long,
            words.FAST_LABEL,
            words.SURVEY_LONG_DETAIL,
        )
        add(
            ActivityPhase.IDLE,
            IDLE_S,
            words.IDLE_LABEL,
            words.FAST_LABEL,
            words.idle_detail(IDLE_S),
        )
    episode = cursor
    for step, failures, length_s in FAULT_STEPS:
        add(
            ActivityPhase.CAMERA_FAULT,
            length_s,
            words.fault_label(FAULT_CAUSE.value, degraded=False),
            words.recovery_label(step),
            words.fault_detail(
                failures, degraded=False, degraded_after=DEGRADED_AFTER, slow_retry_s=SLOW_RETRY_S
            ),
            episode_start_s=episode,
            failures=failures,
            next_step=step,
        )
    return tuple(segments), cursor


PLAYLIST, PLAYLIST_S = build_playlist()


def _segment_at(position_s: float) -> Segment:
    for segment in PLAYLIST:
        if position_s < segment.end_s:
            return segment
    return PLAYLIST[-1]


def _later(now_utc_ns: int, seconds: float) -> int:
    return now_utc_ns + round(seconds * NS_PER_S)


def auto_activity(elapsed_s: float, now_utc_ns: int, reason: str | None) -> ActivityView:
    """The activity of `auto`, `elapsed_s` seconds after the state began."""
    position = elapsed_s % PLAYLIST_S
    segment = _segment_at(position)
    began = segment.start_s if segment.episode_start_s is None else segment.episode_start_s
    detail = segment.detail
    if segment.phase is ActivityPhase.FAST:
        closed = int((position - segment.start_s) // FAST_WINDOW_S)
        detail = words.fast_detail(FAST_WINDOW_S, closed, FAST_WINDOWS, clouds=False)
    ends = _later(now_utc_ns, segment.end_s - position)
    failing = segment.phase is ActivityPhase.CAMERA_FAULT
    return ActivityView(
        state="auto",
        phase=segment.phase.value,
        label=segment.label,
        since_utc_ns=_later(now_utc_ns, began - position),
        ends_utc_ns=None if failing else ends,
        next_label=segment.next_label,
        next_utc_ns=None if segment.phase is ActivityPhase.SOLVE_WAIT else ends,
        cadence_s=None if failing else CYCLE_S,
        detail=detail,
        reason=FAULT_REASON if failing else reason,
    )


def auto_fault(elapsed_s: float, now_utc_ns: int) -> FaultView:
    """The fault of the status while the playlist plays its camera fault, and none otherwise."""
    position = elapsed_s % PLAYLIST_S
    segment = _segment_at(position)
    if segment.phase is not ActivityPhase.CAMERA_FAULT:
        return FaultView()
    began = segment.start_s if segment.episode_start_s is None else segment.episode_start_s
    return FaultView(
        failures=segment.failures,
        good_frames=0,
        last_error=f"{type(FAULT_ERROR).__name__}: {FAULT_ERROR}",
        next_attempt_utc_ns=_later(now_utc_ns, segment.end_s - position),
        next_step=segment.next_step,
        cause=FAULT_CAUSE.value,
        reason=FAULT_REASON,
        since_utc_ns=_later(now_utc_ns, began - position),
    )


def safe_activity(elapsed_s: float, now_utc_ns: int, reason: str | None) -> ActivityView:
    """The brightness watch of `safe`: the gate holds, and a frame comes every few seconds."""
    until_frame = WATCH_INTERVAL_S - elapsed_s % WATCH_INTERVAL_S
    return ActivityView(
        state="safe",
        phase=ActivityPhase.WATCH.value,
        label=words.daylight_label(DAYLIGHT_RESUME_DEG),
        since_utc_ns=_later(now_utc_ns, -elapsed_s),
        next_label=words.WATCH_NEXT_LABEL,
        next_utc_ns=_later(now_utc_ns, until_frame),
        detail=words.watch_detail(WATCH_EXPOSURE_S, WATCH_INTERVAL_S),
        reason=reason,
    )


def align_activity(elapsed_s: float, now_utc_ns: int, reason: str | None) -> ActivityView:
    """The live view of `align`, which ends after the idle timeout."""
    ends = _later(now_utc_ns, ALIGN_TIMEOUT_S)
    return ActivityView(
        state="align",
        phase=ActivityPhase.ALIGN.value,
        label=words.ALIGN_LABEL,
        since_utc_ns=_later(now_utc_ns, -elapsed_s),
        ends_utc_ns=ends,
        next_label=words.ALIGN_NEXT_LABEL,
        next_utc_ns=ends,
        detail=words.align_detail(0.0, ALIGN_TIMEOUT_S),
        reason=reason,
    )


def commission_activity(
    elapsed_s: float,
    now_utc_ns: int,
    reason: str | None,
    task: DarkTaskView | FlatTaskView,
    *,
    kind: str = "dark",
    ends_in_s: float | None = None,
) -> ActivityView:
    """A dark or flat session in `commission`, with the phase that its simulator reports.

    `ends_in_s` is the time left of the phase, when the session knows it: the frames of a flat
    session.
    """
    return ActivityView(
        state="commission",
        phase=ActivityPhase.COMMISSION.value,
        label=words.task_label(kind, task.phase),
        since_utc_ns=_later(now_utc_ns, -elapsed_s),
        ends_utc_ns=None if ends_in_s is None else _later(now_utc_ns, ends_in_s),
        next_label=words.AFTER_TASK_PAUSED if task.pause_after else words.AFTER_TASK_SAFE,
        detail=words.task_detail(0, task.message),
        reason=reason,
    )


def paused_activity(elapsed_s: float, now_utc_ns: int, reason: str | None) -> ActivityView:
    """`paused`: nothing runs until you resume."""
    return ActivityView(
        state="paused",
        phase=ActivityPhase.PAUSED.value,
        label=words.PAUSED_LABEL,
        since_utc_ns=_later(now_utc_ns, -elapsed_s),
        next_label=words.PAUSED_NEXT_LABEL,
        detail=words.PAUSED_DETAIL,
        reason=reason,
    )
