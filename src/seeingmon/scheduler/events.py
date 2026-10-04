"""The events that the scheduler writes, with what each one means.

Every event has a dotted `kind`. A UI or a script can filter the `event` table by these codes.
The scheduler writes the events through its own thread, so they arrive in the order that they
happened. The `detail` of each kind is a JSON object. `EVENT_KINDS` lists the codes, and a test
checks that the scheduler writes no code that this list leaves out.

The result of a commissioning task has the kind `scheduler.<kind>_result`, with the kind of the
task in place of `<kind>`: `scheduler.sweep_result`, `scheduler.burst_result`,
`scheduler.replay_result`, and `scheduler.dark_result`, plus one for each kind that you register a
handler for. A handler may write events of its own through `CommissionContext.emit_event`, and the
list names those too (`scheduler.dark_phase`).
"""

from __future__ import annotations

from collections.abc import Mapping

# The event that a dark session writes when it moves to another phase. The scheduler reads its
# `phase` to name the activity (see `Scheduler._note_task_event`).
DARK_PHASE_EVENT = "scheduler.dark_phase"

EVENT_KINDS: Mapping[str, str] = {
    "scheduler.start": "The scheduler started in `safe`. The detail says whether a site is set.",
    "scheduler.no_site": "No site is configured, so the Sun's elevation and `twilight` are off.",
    "scheduler.state_change": "The state changed. The detail holds `from`, `to`, and `reason`.",
    "scheduler.command": "A command arrived. The detail says whether the scheduler accepted it.",
    "scheduler.fault": "A camera error ended an activity. The detail holds `cause` and `reason`.",
    "scheduler.recovery_step": "The scheduler performed a step of the recovery ladder, or failed.",
    "scheduler.degraded": "The camera failed repeatedly or is gone, so the status is `degraded`.",
    "scheduler.recovered": "Good frames in a row cleared a fault and the `degraded` status.",
    "scheduler.solve_requested": "A survey step runs to solve the pointing again. See `reason`.",
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
    "scheduler.task_error": "A commissioning handler raised an error. The task failed.",
    "scheduler.result_sink_failed": "Storing a commissioning result failed.",
    "scheduler.alignment_sink_failed": "The consumer of the alignment frames raised an error.",
    "scheduler.stop_failed": "The camera did not stop cleanly when the scheduler ended a stream.",
    "scheduler.internal_error": "The loop hit an unexpected error and went on.",
    "scheduler.stalled": "The loop did not run for a while: the machine may have been suspended.",
}
