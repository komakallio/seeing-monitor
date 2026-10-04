"""The words of the activity: labels, details, and reasons that a person at the telescope reads.

`Scheduler.status()` builds an `ActivityStatus` (see `seeingmon.scheduler.status`) from the cycle,
the alignment session, the commissioning task, and the fault episode. The structure comes from the
scheduler. This module holds the text, so that the wording lives in one place and a test can check
it without a camera. Every text is a phrase or one sentence without a final period, in the voice
of the rest of the status: second person when it speaks to you, plain words, and a number with its
unit (`30 s`, `-4 degrees`).

The functions take plain values and return strings. None of them reads a clock or a setting.
"""

from __future__ import annotations

# --- Numbers ---------------------------------------------------------------------------------


def duration_text(seconds: float) -> str:
    """A duration in the plainest form, such as `250 us`, `1 ms`, `30 s`, `2 min 20 s`, `1 h 5 min`.

    The text rounds a duration of a minute or more to whole seconds, and a shorter one to a tenth
    of a second.
    """
    if seconds <= 0:
        return "0 s"
    if seconds < 0.001:
        return f"{round(seconds * 1e6):g} us"
    if seconds < 1.0:
        return f"{round(seconds * 1e3, 1):g} ms"
    if seconds < 60.0:
        return f"{round(seconds, 1):g} s"
    minutes, rest = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes} min" if rest == 0 else f"{minutes} min {rest} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h" if minutes == 0 else f"{hours} h {minutes} min"


def count_text(count: int, singular: str, plural: str | None = None) -> str:
    """`1 task`, `2 tasks`: a count with the right form of the noun."""
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


# --- The fast stream and the survey step ----------------------------------------------------

FAST_LABEL = "Fast stream: seeing windows"
IDLE_LABEL = "Idle until the next cycle"
RETRY_LABEL = "Pausing before the next try to find the pointing"
SOLVE_WAIT_LABEL = "Waiting for a pointing solution"


def survey_step_label(short_s: float, long_s: float) -> str:
    """The name of the whole survey step: `Survey step: a 1 ms and a 30 s frame`."""
    return f"Survey step: a {duration_text(short_s)} and a {duration_text(long_s)} frame"


def survey_frame_label(exposure_s: float) -> str:
    """The name of one exposure of the survey step: `Survey step: the 30 s frame`."""
    return f"Survey step: the {duration_text(exposure_s)} frame"


SURVEY_SHORT_DETAIL = "The short exposure catches the bright stars and checks the sky brightness"
SURVEY_LONG_DETAIL = (
    "The long exposure shows the faint stars, for the pointing, the transparency, and the sky"
)


def fast_detail(window_s: float, closed: int, total: int, *, clouds: bool) -> str:
    """The progress of a fast period: `Windows of 20 s: 4 of 7 closed`."""
    text = f"Windows of {duration_text(window_s)}: {min(closed, total)} of {total} closed"
    return f"{text}; clouds shorten the period" if clouds else text


def solve_wait_detail(give_up_s: float) -> str:
    """Why the scheduler waits, and for how long at most."""
    return (
        "The analysis of the survey frames looks for the stars; "
        f"the scheduler waits up to {duration_text(give_up_s)} for the answer"
    )


def idle_detail(slack_s: float) -> str:
    """The idle slack until the next slot of the cycle."""
    return f"The camera rests for {duration_text(slack_s)} to keep the cadence"


def retry_detail(wait_s: float) -> str:
    """The wait before another survey step looks for the pointing."""
    return f"The scheduler waits {duration_text(wait_s)} before it takes another survey step"


def recovering_text(good_frames: int, needed: int) -> str:
    """The note of a camera that works again but is not yet trusted."""
    return f"the camera recovers: {good_frames} of {needed} good frames"


# --- Safe -------------------------------------------------------------------------------------


def daylight_label(resume_limit_deg: float) -> str:
    """The label while the Sun holds the daylight gate.

    The limit is the one that `auto` resumes below, which is a margin under the one that stops it,
    so the label reads `Daylight gate: the Sun is above -4 degrees` with the default settings.
    """
    return f"Daylight gate: the Sun is above {resume_limit_deg:g} degrees"


BRIGHT_SKY_LABEL = "Brightness gate: the sky is too bright"
FIRST_FRAME_LABEL = "Brightness watch: waiting for the first frame"
WATCH_LABEL = "Brightness watch: checking whether the sky is dark enough"
WATCH_NEXT_LABEL = "Brightness frame: the cycle starts when the sky is dark enough"


def watch_detail(exposure_s: float, interval_s: float) -> str:
    """How the brightness watch works."""
    return f"The camera takes a {duration_text(exposure_s)} frame every {duration_text(interval_s)}"


def bright_sky_detail(background_fraction: float | None, resume_fraction: float) -> str:
    """The measured brightness and the level that opens the gate."""
    limit = f"the cycle resumes below {resume_fraction * 100:g}% of saturation"
    if background_fraction is None:
        return limit[0].upper() + limit[1:]
    return f"The background is {background_fraction * 100:.0f}% of saturation; {limit}"


# --- Align, commission, paused ----------------------------------------------------------------

ALIGN_LABEL = "Aligning (the live view)"
ALIGN_NEXT_LABEL = "Brightness check, then the normal cycle"
PAUSED_LABEL = "Paused: nothing runs"
PAUSED_DETAIL = "The camera stays open and idle until you resume"
PAUSED_NEXT_LABEL = "After you resume: a brightness check, then the normal cycle"


def align_detail(idle_s: float, timeout_s: float) -> str:
    """The idle timer of the alignment."""
    return (
        f"The alignment ends after {duration_text(timeout_s)} without use; "
        f"it has been idle for {duration_text(idle_s)}"
    )


AFTER_TASK_AUTO = "Back to the normal cycle"
AFTER_TASK_SAFE = "Back to the brightness watch"
AFTER_TASK_PAUSED = "Paused: nothing records until you resume"

# The label of a commissioning task, by kind. A dark session names its phase (see `DARK_PHASES`).
TASK_LABELS = {
    "burst": "Burst: recording raw frames",
    "sweep": "Sweep: testing exposure, gain, and ROI",
    "replay": "Replay: running a recording through the analysis",
    "dark": "Dark session",
}
TASK_STARTING_LABEL = "Commissioning: starting the next task"
DARK_PHASES = {
    "bias": "Dark session: taking bias frames",
    "cover": "Dark session: waiting for the cover",
    "dark": "Dark session: taking dark frames",
    "build": "Dark session: building the master dark",
}


def task_label(kind: str | None, phase: str | None = None) -> str:
    """The label of a commissioning task, from its kind and, for a dark session, its phase."""
    if kind is None:
        return TASK_STARTING_LABEL
    if kind == "dark" and phase in DARK_PHASES:
        return DARK_PHASES[phase]
    return TASK_LABELS.get(kind, f"Commissioning: the {kind} task runs")


def task_detail(waiting: int, message: str | None) -> str | None:
    """The progress message of the task, and the number of tasks that wait behind it."""
    parts = []
    if message:
        parts.append(message.rstrip("."))
    if waiting:
        parts.append(f"{count_text(waiting, 'more task')} queued")
    return "; ".join(parts) or None


# --- The camera -------------------------------------------------------------------------------

STEP_TEXT = {
    "restart_capture": "restart the capture",
    "reopen": "reopen the camera",
    "usb_reset": "reset the USB device",
    "restart_acquire": "restart the acquire process",
    "reboot": "reboot the machine",
    "power_cycle": "cut and restore the power",
}


def step_text(step_name: str) -> str:
    """A ladder step in words, such as `reopen the camera`."""
    return STEP_TEXT.get(step_name, step_name.replace("_", " "))


def recovery_label(step_name: str) -> str:
    """The name of the next recovery step: `Recovery step: reopen the camera`."""
    return f"Recovery step: {step_text(step_name)}"


# --- Why a state holds ------------------------------------------------------------------------

# The reasons that the scheduler records for a change of state, in words. A reason that this table
# does not name is already a phrase, such as `the sky is dark enough`.
_STATE_REASONS = {
    "startup": "the scheduler started, and it checks the sky first",
    "daylight": "the Sun is above the daylight gate",
    "bright_sky": "the sky is too bright for the camera",
    "no_measurement": "the sky brightness is not known yet",
    "fault": "the camera failed",
    "camera fault": "the camera failed",
    "alignment started": "you started the alignment",
    "alignment stopped": "you stopped the alignment, so the scheduler checks the sky first",
    "alignment idle timeout": "the alignment was idle for too long",
    "no alignment session": "the alignment session ended",
    "pause command": "you paused the scheduler",
    "resume command": "you resumed the scheduler, so it checks the sky first",
    "a task is queued": "a commissioning task is queued",
    "a dark session starts at once": "a dark session starts at once",
    "commissioning is done": "the commissioning tasks are done",
    "state_change": "the state changed",
}


def state_reason_text(reason: str | None) -> str | None:
    """The reason that the machine recorded for its state, in words."""
    if not reason:
        return None
    return _STATE_REASONS.get(reason, reason)
