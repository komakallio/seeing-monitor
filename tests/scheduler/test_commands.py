"""The command classes, and the result that `submit` returns."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from seeingmon.frames import StreamConfig
from seeingmon.scheduler.commands import (
    QUEUE_COMMANDS,
    TASK_KINDS,
    Command,
    CommandResult,
    Pause,
    QueueBurst,
    QueueDark,
    QueueReplay,
    QueueSweep,
    RejectReason,
    Resume,
    StartAlignment,
    StopAlignment,
)

ALL_COMMANDS = [
    StartAlignment(),
    StopAlignment(),
    Pause(),
    Resume(),
    QueueBurst(),
    QueueSweep(),
    QueueReplay(),
    QueueDark(),
]


@pytest.mark.parametrize("command", ALL_COMMANDS, ids=lambda command: type(command).__name__)
def test_every_command_is_a_frozen_dataclass_without_a_dict(command: Any) -> None:
    with pytest.raises(TypeError):  # slots, so a typo in a field name is an error
        vars(command)
    fields = dataclasses.fields(command)
    if fields:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(command, fields[0].name, None)


def test_the_commands_are_the_ones_that_the_brief_lists() -> None:
    assert all(isinstance(command, Command) for command in ALL_COMMANDS)
    names = {type(command).__name__ for command in ALL_COMMANDS}
    assert names == {
        "StartAlignment",
        "StopAlignment",
        "Pause",
        "Resume",
        "QueueBurst",
        "QueueSweep",
        "QueueReplay",
        "QueueDark",
    }


def test_the_queue_commands_map_to_the_task_kinds() -> None:
    assert set(QUEUE_COMMANDS) == set(TASK_KINDS)
    assert TASK_KINDS[QueueBurst] == "burst"
    assert TASK_KINDS[QueueSweep] == "sweep"
    assert TASK_KINDS[QueueReplay] == "replay"
    assert TASK_KINDS[QueueDark] == "dark"


def test_the_queue_commands_share_a_priority_that_defaults_to_zero() -> None:
    assert all(
        command.priority == 0 for command in ALL_COMMANDS if isinstance(command, QUEUE_COMMANDS)
    )


def test_a_burst_takes_stream_settings() -> None:
    stream = StreamConfig(mode="bin1", exposure_us=2000, gain=0)
    assert QueueBurst(duration_s=5.0, stream=stream, label="test").stream is stream


def test_a_sweep_leaves_its_axes_empty_for_the_configured_defaults() -> None:
    sweep = QueueSweep()
    assert (sweep.exposure_us, sweep.gain, sweep.roi_arcmin, sweep.modes) == ((), (), (), ())
    assert sweep.window_s is None


def test_replay_options_are_independent_between_instances() -> None:
    first, second = QueueReplay(source="a"), QueueReplay(source="b")
    assert first.options is not second.options


def test_a_result_names_the_reason_of_a_rejection() -> None:
    accepted = CommandResult(accepted=True, message="started", state="align")
    rejected = CommandResult(
        accepted=False, message="no", state="paused", reason=RejectReason.PAUSED
    )
    assert accepted.reason is None
    assert rejected.reason is RejectReason.PAUSED
    assert str(rejected.reason) == "paused"  # a string enum, so the web layer can use the value


def test_the_rejection_reasons_are_stable_codes() -> None:
    assert {reason.value for reason in RejectReason} == {
        "paused",
        "already_paused",
        "not_paused",
        "not_aligning",
        "degraded",
        "no_handler",
        "queue_full",
        "invalid",
        "closed",
    }
