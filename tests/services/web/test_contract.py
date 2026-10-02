"""The contract between `web` and `core`: commands, results, status, and the alignment frame."""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

import pytest

from seeingmon.frames import PixelFormat, Roi, StreamConfig, StreamKind
from seeingmon.scheduler.commands import (
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
from seeingmon.scheduler.status import Counters, FaultStatus, SchedulerStatus, StreamInfo
from seeingmon.services.ipc.codec import CodecError, decode_json, encode_json
from seeingmon.services.web.contract import (
    FRAME_MAGIC,
    METHODS,
    AlignmentState,
    CoreStatus,
    decode_alignment_state,
    decode_command,
    decode_dark_library,
    decode_result,
    decode_status,
    encode_command,
    encode_result,
    encode_status,
    pack_frame,
    unpack_frame,
)
from tests.services.web.helpers import alignment_state, tiny_jpeg

COMMANDS: list[Command] = [
    StartAlignment(),
    StartAlignment(exposure_s=0.25, gain=90),
    StopAlignment(),
    Pause(),
    Resume(),
    QueueBurst(),
    QueueBurst(
        duration_s=30.0,
        stream=StreamConfig(
            mode="bin1",
            exposure_us=2000,
            gain=10,
            pixel_format=PixelFormat.RAW8,
            roi=Roi(x=8, y=8, width=128, height=128),
            kind=StreamKind.VIDEO,
            offset=5,
            bandwidth_pct=80,
            high_speed=True,
        ),
        label="focus run",
        priority=3,
    ),
    QueueSweep(),
    QueueSweep(
        exposure_us=(500, 1000),
        gain=(0, 60),
        roi_arcmin=(4.1, 8.0),
        modes=("bin1",),
        window_s=5.0,
        priority=-2,
    ),
    QueueReplay(),
    QueueReplay(source="night-1", speed=0.0, options={"loop": True, "start_s": 12.5}, priority=1),
    QueueDark(),
    QueueDark(
        exposure_s=60.0,
        frames=5,
        bias_frames=4,
        wait_for_cover=False,
        pause_after=False,
        label="winter set",
        priority=2,
    ),
]


@pytest.mark.parametrize("command", COMMANDS, ids=lambda command: repr(command)[:60])
def test_a_command_survives_the_round_trip_through_json(command: Command) -> None:
    wire = decode_json(encode_json(encode_command(command)))
    assert decode_command(wire) == command


def test_the_names_of_the_commands_are_the_documented_ones() -> None:
    names = {encode_command(command)["type"] for command in COMMANDS}
    assert names == {
        "start_alignment",
        "stop_alignment",
        "pause",
        "resume",
        "queue_burst",
        "queue_sweep",
        "queue_replay",
        "queue_dark",
    }


def test_a_command_without_optional_fields_takes_the_defaults() -> None:
    assert decode_command({"type": "queue_burst"}) == QueueBurst()
    assert decode_command({"type": "start_alignment"}) == StartAlignment()
    assert decode_command({"type": "queue_sweep"}) == QueueSweep()
    assert decode_command({"type": "queue_replay"}) == QueueReplay()
    assert decode_command({"type": "queue_dark"}) == QueueDark()
    assert QueueDark().wait_for_cover
    assert QueueDark().pause_after


def test_an_object_that_is_not_a_command_cannot_be_sent() -> None:
    with pytest.raises(TypeError, match="cannot send"):
        encode_command(Command())


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "pause",
        {},
        {"type": 5},
        {"type": "reboot"},
        {"type": "pause", "extra": 1},
        {"type": "start_alignment", "gain": 1.5},
        {"type": "start_alignment", "gain": True},
        {"type": "start_alignment", "exposure_s": "fast"},
        {"type": "start_alignment", "exposure_s": math.inf},
        {"type": "queue_burst", "duration_s": None},
        {"type": "queue_burst", "label": 5},
        {"type": "queue_burst", "priority": 1.0},
        {"type": "queue_burst", "stream": {"mode": "bin1"}},
        {"type": "queue_burst", "stream": {"mode": "", "exposure_us": 1, "gain": 0}},
        {"type": "queue_burst", "stream": {"mode": "bin1", "exposure_us": 0, "gain": 0}},
        {"type": "queue_sweep", "exposure_us": [1, 2.5]},
        {"type": "queue_sweep", "exposure_us": "1"},
        {"type": "queue_sweep", "exposure_us": list(range(300))},
        {"type": "queue_sweep", "gain": [True]},
        {"type": "queue_sweep", "roi_arcmin": ["x"]},
        {"type": "queue_sweep", "modes": [1]},
        {"type": "queue_sweep", "modes": "bin1"},
        {"type": "queue_replay", "source": 1},
        {"type": "queue_replay", "speed": "fast"},
        {"type": "queue_replay", "options": [1]},
        {"type": "queue_replay", "options": {1: 2}},
        {"type": "queue_dark", "frames": 2.5},
        {"type": "queue_dark", "frames": True},
        {"type": "queue_dark", "exposure_s": "long"},
        {"type": "queue_dark", "exposure_s": math.nan},
        {"type": "queue_dark", "wait_for_cover": "yes"},
        {"type": "queue_dark", "pause_after": 1},
        {"type": "queue_dark", "label": 7},
        {"type": "queue_dark", "priority": 1.5},
        {"type": "queue_dark", "extra": 1},
    ],
)
def test_a_malformed_command_is_refused(value: Any) -> None:
    with pytest.raises(CodecError):
        decode_command(value)


# --- The result of a command -----------------------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [
        CommandResult(accepted=True, message="alignment started", state="align"),
        CommandResult(accepted=True, message="queued", state="auto", task_id=7),
        CommandResult(accepted=False, message="paused", state="paused", reason=RejectReason.PAUSED),
    ],
)
def test_a_result_survives_the_round_trip_through_json(result: CommandResult) -> None:
    assert decode_result(decode_json(encode_json(encode_result(result)))) == result


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"accepted": "yes", "message": "m", "state": "s"},
        {"accepted": True, "message": 1, "state": "s"},
        {"accepted": True, "message": "m", "state": "s", "reason": "because"},
        {"accepted": True, "message": "m", "state": "s", "task_id": 1.5},
        {"accepted": True, "message": "m", "state": "s", "extra": 1},
    ],
)
def test_a_malformed_result_is_refused(value: Any) -> None:
    with pytest.raises(CodecError):
        decode_result(value)


def test_the_message_of_a_result_is_cut_to_a_bound() -> None:
    value = {"accepted": True, "message": "m" * 100_000, "state": "auto"}
    assert len(decode_result(value).message) == 4000


# --- The status ------------------------------------------------------------------------------


def scheduler_status(**overrides: Any) -> SchedulerStatus:
    fields: dict[str, Any] = {
        "t_utc_ns": 1_767_225_660_000_000_000,
        "state": "auto",
        "state_reason": "the sky is dark",
        "state_since_utc_ns": 1_767_225_600_000_000_000,
        "last_transition_utc_ns": 1_767_225_600_000_000_000,
        "degraded": False,
        "stream": StreamInfo(
            stream_id=4,
            purpose="fast",
            mode="bin1",
            exposure_us=2000,
            gain=0,
            roi=Roi(x=100, y=200, width=128, height=128),
        ),
        "cloud": True,
        "cloud_fraction": 0.62,
        "twilight": False,
        "sun_elevation_deg": -30.5,
        "background_fraction": 0.01,
        "sensor_temperature_c": 4.5,
        "counters": Counters(frames=1000, dropped=3, windows=12),
        "fault": FaultStatus(failures=1, good_frames=4, last_error="timeout", next_step="reopen"),
        "queued_tasks": 2,
        "survey_pending": 1,
        "alignment_idle_s": None,
    }
    fields.update(overrides)
    return SchedulerStatus(**fields)


def test_a_status_survives_the_round_trip_through_json() -> None:
    status = scheduler_status()
    wire = decode_json(encode_json(encode_status(status, "core-1")))
    decoded = decode_status(wire)
    assert decoded.instance == "core-1"
    assert decoded.scheduler.state == "auto"
    assert decoded.scheduler.stream is not None
    assert decoded.scheduler.stream.roi is not None
    assert decoded.scheduler.stream.roi.width == 128
    assert decoded.scheduler.counters["frames"] == 1000
    assert decoded.scheduler.fault.next_step == "reopen"
    assert decoded.scheduler.cloud_fraction == 0.62
    assert dataclasses.asdict(status)["queued_tasks"] == decoded.scheduler.queued_tasks


def test_a_status_without_a_stream_decodes() -> None:
    wire = decode_json(encode_json(encode_status(scheduler_status(stream=None), "core-1")))
    assert decode_status(wire).scheduler.stream is None


def test_a_newer_core_may_add_fields() -> None:
    wire = decode_json(encode_json(encode_status(scheduler_status(), "core-1")))
    wire["storage"] = {"free_gb": 12.0}
    wire["scheduler"]["new_field"] = 1
    assert decode_status(wire).instance == "core-1"


@pytest.mark.parametrize(
    "value",
    [None, [], {}, {"instance": "x"}, {"instance": "x", "scheduler": {"state": "auto"}}],
)
def test_a_malformed_status_is_refused(value: Any) -> None:
    with pytest.raises(CodecError):
        decode_status(value)


def test_the_error_of_a_malformed_status_names_the_fields_and_not_the_values() -> None:
    value = {"instance": "x", "scheduler": {"state": "secret-value", "t_utc_ns": "later"}}
    with pytest.raises(CodecError) as raised:
        decode_status(value)
    text = str(raised.value)
    assert "t_utc_ns" in text
    assert "secret-value" not in text
    assert "later" not in text


def test_core_status_is_frozen() -> None:
    status = CoreStatus.model_validate(
        decode_json(encode_json(encode_status(scheduler_status(), "c")))
    )
    with pytest.raises(ValueError, match="frozen"):
        status.instance = "other"  # type: ignore[misc]


# --- The alignment state and the frame -------------------------------------------------------


def test_an_empty_alignment_state_is_inactive() -> None:
    state = AlignmentState()
    assert state.active is False
    assert state.target is None
    assert state.quality == {}


def test_an_alignment_state_survives_the_round_trip_through_json() -> None:
    state = alignment_state(seq=9)
    assert decode_alignment_state(json.loads(state.model_dump_json())) == state


@pytest.mark.parametrize(
    "value",
    [
        {"frame": {"seq": -1, "width_px": 10, "height_px": 10}},
        {"frame": {"seq": 0, "width_px": 0, "height_px": 10}},
        {"saturation": {"fraction": 1.5}},
        {"histogram": {"counts": [1, 2], "max_dn": 0}},
        {"histogram": {"counts": list(range(300)), "max_dn": 10}},
        {"offset": {"dx_px": float("nan"), "dy_px": 0.0, "distance_px": 1.0}},
        {"quality": {"a": 1}},
        {"active": "yes"},
    ],
)
def test_a_malformed_alignment_state_is_refused(value: Any) -> None:
    with pytest.raises(CodecError):
        decode_alignment_state(value)


def test_a_frame_survives_the_round_trip() -> None:
    state = alignment_state(seq=3)
    jpeg = tiny_jpeg(shade=120)
    payload = pack_frame(state, jpeg)
    assert payload.startswith(FRAME_MAGIC)
    frame = unpack_frame(memoryview(payload))
    assert frame.state == state
    assert frame.jpeg == jpeg


def test_the_documented_layout_of_a_frame() -> None:
    state = AlignmentState(active=True)
    jpeg = tiny_jpeg()
    payload = pack_frame(state, jpeg)
    length = int.from_bytes(payload[4:8], "little")
    assert json.loads(payload[8 : 8 + length]) == json.loads(state.model_dump_json())
    assert payload[8 + length :] == jpeg


def test_pack_refuses_something_that_is_not_a_jpeg() -> None:
    with pytest.raises(ValueError, match="JPEG"):
        pack_frame(AlignmentState(), b"not an image")


def test_pack_refuses_a_state_that_is_too_large() -> None:
    state = AlignmentState(quality={f"k{index}": "v" * 3000 for index in range(25)})
    with pytest.raises(ValueError, match="too large"):
        pack_frame(state, tiny_jpeg())


def test_unpack_refuses_what_is_not_a_frame() -> None:
    good = pack_frame(alignment_state(), tiny_jpeg())
    state_length = int.from_bytes(good[4:8], "little")
    bad_length = FRAME_MAGIC + (10**9).to_bytes(4, "little") + good[8:]
    no_image = good[: 8 + state_length]
    not_jpeg = good[: 8 + state_length] + b"plain bytes here"
    bad_state = FRAME_MAGIC + (4).to_bytes(4, "little") + b"nope" + tiny_jpeg()
    for payload in [b"", b"SMAF", b"XXXX" + good[4:], bad_length, no_image, not_jpeg, bad_state]:
        with pytest.raises(CodecError):
            unpack_frame(payload)


def test_the_methods_are_the_documented_ones() -> None:
    assert METHODS == ("ping", "status", "submit", "alignment_state")


# --- The dark library ------------------------------------------------------------------------


def dark_library_example() -> dict[str, Any]:
    return {
        "mode": "bin2",
        "gain": 120,
        "exposure_s": 30.0,
        "sensor_temperature_c": 18.4,
        "status": {
            "due": True,
            "reason": "no recent set within 3.0 C of 18.4 C (the nearest is 6.2 C away)",
            "tolerance_c": 3.0,
            "max_age_days": 183.0,
            "gap_c": 6.2,
            "nearest_name": "dark-20260301T120000Z-bin2-g120.fits",
            "newest_age_days": 40.5,
        },
        "model": {
            "reference_c": 20.0,
            "rate_ref_e_per_s": 0.21,
            "doubling_c": 6.0,
            "doubling_fitted": False,
            "rms_log2": None,
            "n_sets": 1,
        },
        "sets": [
            {
                "name": "dark-20260301T120000Z-bin2-g120.fits",
                "t_utc": "2026-03-01T12:00:00Z",
                "age_days": 40.5,
                "temperature_c": 12.2,
                "temperature_spread_c": 0.4,
                "exposure_s": 30.0,
                "n_frames": 9,
                "n_bias_frames": 9,
                "rate_e_per_s": 0.09,
                "hot_pixels": 211,
            }
        ],
        "task": {
            "state": "running",
            "task_id": 7,
            "phase": "cover",
            "step": 0,
            "steps": 0,
            "message": "Cover the camera now. Waiting for a dark frame.",
            "covered": False,
            "level_dn": 3012.5,
            "reason": "the median is 2400 counts above the expected level",
            "exposure_s": 30.0,
            "frames": 9,
            "bias_frames": 9,
            "wait_for_cover": True,
            "pause_after": True,
            "started_utc": "2026-04-10T20:00:00Z",
            "finished_utc": None,
            "summary": "",
            "set_name": None,
        },
    }


def test_a_dark_library_survives_the_round_trip_through_json() -> None:
    wire = decode_json(encode_json(dark_library_example()))
    view = decode_dark_library(wire)
    assert view.status.due is True
    assert view.sets[0].temperature_c == 12.2
    assert view.task.phase == "cover"
    assert view.model is not None
    assert view.model.doubling_fitted is False
    assert json.loads(view.model_dump_json())["task"]["step"] == 0


def test_an_empty_library_decodes_with_an_idle_task() -> None:
    view = decode_dark_library(
        {
            "mode": "bin2",
            "gain": 120,
            "exposure_s": 30.0,
            "status": {
                "due": True,
                "reason": "the library holds no dark set",
                "tolerance_c": 3.0,
                "max_age_days": 183.0,
            },
        }
    )
    assert view.sets == []
    assert view.model is None
    assert view.sensor_temperature_c is None
    assert view.task.state == "idle"
    assert view.task.phase is None
    assert view.task.task_id is None


def test_a_newer_core_may_add_dark_fields() -> None:
    example = dark_library_example()
    example["something_new"] = 1
    example["task"]["something_new"] = "x"
    example["sets"][0]["something_new"] = [1]
    assert decode_dark_library(example).task.task_id == 7


def test_numbers_that_json_writes_without_a_fraction_still_decode() -> None:
    example = dark_library_example()
    example["exposure_s"] = 30
    example["sensor_temperature_c"] = 18
    example["sets"][0]["temperature_c"] = 12
    example["sets"][0]["rate_e_per_s"] = 0
    view = decode_dark_library(decode_json(encode_json(example)))
    assert view.exposure_s == 30.0
    assert view.sets[0].temperature_c == 12.0


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.pop("status"),
        lambda d: d.update(gain="120"),
        lambda d: d.update(sets="none"),
        lambda d: d.update(sets=[{"name": "x"}]),
        lambda d: d["status"].update(due="yes"),
        lambda d: d["task"].update(step="3"),
        lambda d: d["task"].update(covered="false"),
        lambda d: d["task"].update(level_dn=math.inf),
        lambda d: d.update(sets=d["sets"] * 300),
    ],
)
def test_a_malformed_dark_library_is_refused(change: Any) -> None:
    example = dark_library_example()
    change(example)
    with pytest.raises(CodecError):
        decode_dark_library(example)


def test_the_error_of_a_malformed_dark_library_names_the_fields_and_not_the_values() -> None:
    example = dark_library_example()
    example["task"]["message"] = 5
    example["gain"] = "a-secret-value"
    with pytest.raises(CodecError) as error:
        decode_dark_library(example)
    assert "gain" in str(error.value)
    assert "a-secret-value" not in str(error.value)
