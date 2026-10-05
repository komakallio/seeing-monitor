"""The events that the scheduler writes, with what each one means.

Every event has a dotted `kind`. A UI or a script can filter the `event` table by these codes.
The scheduler writes the events through its own thread, so they arrive in the order that they
happened. The `detail` of each kind is a JSON object. `EVENT_KINDS` lists the codes, and a test
checks that the scheduler writes no code that this list leaves out.

The result of a commissioning task has the kind `scheduler.<kind>_result`, with the kind of the
task in place of `<kind>`: `scheduler.sweep_result`, `scheduler.burst_result`,
`scheduler.replay_result`, `scheduler.dark_result`, and `scheduler.flat_result`, plus one for each
kind that you register a handler for. A handler may write events of its own through
`CommissionContext.emit_event`, and the list names those too (`scheduler.dark_phase` and
`scheduler.flat_phase`). A task that `CancelTask` removes from the queue gets a result event too.

A kind that describes the sky or the camera rather than the scheduler has its own prefix, such as
`pointing.moved`, which the scheduler writes when a survey result shows that the camera moved.
`core` writes two kinds of the sky from the survey results that the scheduler polls, `sky.dark` and
`sky.clear_verdict` (`seeingmon.services.core.darkness`), and the list names them too, so that it
holds every kind that the survey results lead to.
The `polaris.*` kinds follow the visibility of Polaris: `polaris.visible` when the fast stream
switches from search to measure, and `polaris.hidden` whenever measure ends, for any reason (the
star went missing, the state left `auto`, a camera fault, the pointing solution went away, or the
shutdown). Both carry the Sun's elevation in `sun_elevation_deg`, which is `null` without a site
or with a clock that is not synchronized. `polaris.search_limit_low` warns that a probe burst
found Polaris while the Sun was above `[scheduler.search] max_sun_elevation_deg`.
"""

from __future__ import annotations

from collections.abc import Mapping

# The events that a dark session and a flat session write when they move to another phase. The
# scheduler reads their `phase` to name the activity, and `expected_s` (the seconds that the phase
# takes, when the session knows) to announce its end (see `Scheduler._note_task_event`).
DARK_PHASE_EVENT = "scheduler.dark_phase"
FLAT_PHASE_EVENT = "scheduler.flat_phase"

EVENT_KINDS: Mapping[str, str] = {
    "scheduler.start": "The scheduler started in `safe`. The detail says whether a site is set.",
    "scheduler.no_site": "No site is configured, so the Sun's elevation and `twilight` are off.",
    "scheduler.state_change": "The state changed. The detail holds `from`, `to`, and `reason`.",
    "scheduler.command": "A command arrived. The detail says whether the scheduler accepted it.",
    "scheduler.fault": "A camera error ended an activity. The detail holds `cause` and `reason`.",
    "scheduler.recovery_step": "The scheduler performed a step of the recovery ladder, or failed.",
    "scheduler.degraded": "The camera failed repeatedly or is gone, so the status is `degraded`.",
    "scheduler.recovered": "Good frames in a row cleared a fault and the `degraded` status.",
    "scheduler.solve_requested": "No pointing solution exists, so a survey step runs to solve.",
    "pointing.moved": "A solve found the camera off its reference. It comes once for each move.",
    "polaris.visible": "Search bursts found Polaris, so the fast stream measures seeing.",
    "polaris.hidden": "Measure ended. The detail holds `reason` and the Sun's elevation.",
    "polaris.search_limit_low": "A probe found Polaris above the search limit of the Sun.",
    "scheduler.roi_recentered": "The star neared the ROI edge, so the window ended early.",
    "scheduler.roi_at_limit": "The star is near the ROI edge, and the ROI cannot move closer.",
    "scheduler.cloud": "The cloud response started or ended. The detail holds `active`.",
    "scheduler.survey_skipped": "The survey analysis is behind, so a survey step was skipped.",
    "scheduler.clock_unsynchronized": "The clock lost synchronization, so the Sun is not used.",
    "scheduler.clock_synchronized": "The clock is synchronized again.",
    "scheduler.task_started": "A commissioning task started.",
    "scheduler.sweep_result": "A sweep finished. The detail holds the table of cells.",
    "scheduler.burst_result": "A burst finished. The detail names the files that it wrote.",
    "scheduler.replay_result": "A replay finished.",
    "scheduler.dark_result": "A dark session finished. The detail holds the set and the dark rate.",
    "scheduler.dark_phase": "A dark session moved to another phase: bias, cover, dark, or build.",
    "scheduler.flat_result": "A flat session finished. The detail holds the version of the flat.",
    "scheduler.flat_phase": "A flat session began a phase: setup, exposure, capture, or build.",
    "scheduler.task_error": "A commissioning handler raised an error. The task failed.",
    "scheduler.result_sink_failed": "Storing a commissioning result failed.",
    "scheduler.alignment_sink_failed": "The consumer of the alignment frames raised an error.",
    "scheduler.rapid_focus": "Rapid focus started or ended. The detail holds `phase` and `reason`.",
    "scheduler.focus_sink_failed": "The consumer of the rapid focus frames raised an error.",
    "scheduler.stop_failed": "The camera did not stop cleanly when the scheduler ended a stream.",
    "scheduler.internal_error": "The loop hit an unexpected error and went on.",
    "scheduler.stalled": "The loop did not run for a while: the machine may have been suspended.",
    "sky.dark": "The sky stopped getting darker. `core` writes it once a night.",
    "sky.clear_verdict": "The share of clear frames after `sky.dark`, from `core` once a night.",
}
