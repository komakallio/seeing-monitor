"""The dark session against the simulated camera: sets, the model they give, and the wait.

Every test runs on a `VirtualClock`, so a 30 s exposure and a wait for the cover take no real
time, and the numbers do not depend on how fast the machine is.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from seeingmon.clock import VirtualClock
from seeingmon.drivers.base import CameraDriver
from seeingmon.drivers.sim import SimDriver, SimOptions
from seeingmon.drivers.sim.params import SimParams
from seeingmon.frames import Frame
from seeingmon.survey.dark import DarkError, DarkLibrary
from seeingmon.survey.dark_session import (
    DarkSessionOptions,
    DarkSessionResult,
    run_dark_session,
)
from tests.survey import simfx, synth

PROFILE = synth.cropped_profile(512, 384)
SIM_PARAMS = SimParams.from_profile(PROFILE, "bin2")
E_PER_ADU = PROFILE.e_per_adu("bin2", 120)
FAST = DarkSessionOptions(wait=False, frames=5, bias_frames=5)


def run(
    library: DarkLibrary,
    driver: CameraDriver,
    clock: VirtualClock,
    options: DarkSessionOptions = FAST,
) -> tuple[DarkSessionResult, list[str]]:
    lines: list[str] = []
    result = run_dark_session(driver, library, PROFILE, clock, options, say=lines.append)
    return result, lines


def covered(ambient_c: float, *, seed: int = 3, clock: VirtualClock | None = None) -> SimDriver:
    options = simfx.covered_options(ambient_c=ambient_c, hot_pixels_per_mpix=300.0, seed=seed)
    return simfx.make_driver(PROFILE, options, clock or VirtualClock())


def test_a_session_records_the_dark_rate_the_bias_and_the_hot_pixels(tmp_path: Path) -> None:
    clock = VirtualClock()
    library = DarkLibrary(tmp_path / "darks")
    driver = covered(14.0, clock=clock)
    result, lines = run(library, driver, clock)
    dark_set = result.dark_set
    assert dark_set.temperature_c == pytest.approx(
        18.0
    )  # the ambient 14 C plus 4 C of self-heating
    truth_e_per_s = SIM_PARAMS.dark_rate_e_per_s(18.0)
    assert dark_set.rate_dn_per_s * E_PER_ADU == pytest.approx(truth_e_per_s, rel=0.03)
    assert dark_set.bias_dn == pytest.approx(120.0, abs=0.05)  # the offset of 30, in 14-bit counts
    assert dark_set.read_noise_dn == pytest.approx(1.85 / E_PER_ADU, rel=0.1)
    assert (dark_set.n_frames, dark_set.n_bias_frames) == (5, 5)
    assert dark_set.exposure_s == 30.0
    assert (dark_set.width_px, dark_set.height_px) == (512, 384)
    assert library.sets() == (dark_set,)
    # The hot pixels that stand out are the injected ones, and they are all of the strong ones.
    x, y, rate = simfx.hot_map_truth(driver, "bin2")
    factor = SIM_PARAMS.dark_rate_e_per_s(18.0) / SIM_PARAMS.dark_rate_e_per_s(20.0)
    excess_dn = rate * factor * 30.0 / E_PER_ADU
    mask = library.hot_pixel_mask("bin2", 120, 18.0)
    assert mask is not None
    assert int(mask.sum()) == dark_set.n_hot_pixels
    strong = excess_dn > 40.0  # far above the threshold of about 12 counts
    assert np.all(mask[y[strong], x[strong]])
    weak = excess_dn < 6.0
    assert not np.any(mask[y[weak], x[weak]])
    truth = np.zeros(mask.shape, dtype=bool)
    truth[y, x] = True
    assert not np.any(mask & ~truth)  # nothing but the injected pixels stands out
    # The summary names the file and the numbers, and nothing about the machine.
    text = "\n".join(lines)
    assert dark_set.name in text
    assert "hot pixels stand out" in text
    assert str(tmp_path) not in text
    assert "library covers this temperature" in text


def test_four_sets_fit_the_dark_model_of_the_sensor(tmp_path: Path) -> None:
    clock = VirtualClock()
    library = DarkLibrary(tmp_path / "darks")
    for ambient in (-2.0, 6.0, 14.0, 21.0):
        run(library, covered(ambient, clock=clock), clock)
    model = library.model("bin2", 120)
    assert model is not None
    assert model.doubling_fitted
    assert model.n_sets == 4
    # The simulator interpolates ZWO's chart, which doubles every 5.2 C to 6.5 C.
    assert 5.4 < model.doubling_c < 6.4
    truth_dn_per_s = SIM_PARAMS.dark_rate_e_per_s(20.0) / E_PER_ADU
    assert model.rate_ref_dn_per_s == pytest.approx(truth_dn_per_s, rel=0.03)
    for temperature in (2.0, 10.0, 18.0, 25.0):
        truth = SIM_PARAMS.dark_rate_e_per_s(temperature) / E_PER_ADU
        assert model.rate_dn_per_s(temperature) == pytest.approx(truth, rel=0.12)
    assert model.rms_log2 is not None
    assert model.rms_log2 < 0.08


def test_the_session_waits_for_the_cover_without_waiting_in_real_time(tmp_path: Path) -> None:
    clock = VirtualClock()
    uncovered = simfx.make_driver(PROFILE, SimOptions(seed=5), clock)
    cover = simfx.make_driver(PROFILE, simfx.covered_options(ambient_c=15.0, seed=5), clock)
    # 5 bias frames and 3 test frames look at the sky, and then the cover goes on.
    wrapper = simfx.CoverableDriver(uncovered, cover, cover_after_reads=5 + 3)
    options = DarkSessionOptions(frames=3, bias_frames=5, poll_s=5.0, stable_polls=2)
    library = DarkLibrary(tmp_path / "darks")
    started = clock.utc_ns()
    result, lines = run(library, wrapper, clock, options)
    # Five polls (three lit, two dark) and four waits of 5 s between them, plus the exposures.
    assert 20.0 <= result.waited_s < 30.0
    assert wrapper.reads == 5 + 5 + 3
    assert lines.count("Cover the camera now. Waiting for a dark frame.") == 1
    assert "The camera is dark." in lines
    assert (clock.utc_ns() - started) / 1e9 > 3 * 30.0  # the three dark exposures happened too
    assert len(library.sets()) == 1


def test_one_lit_poll_in_a_row_resets_the_count_of_dark_polls(tmp_path: Path) -> None:
    clock = VirtualClock()
    uncovered = simfx.make_driver(PROFILE, SimOptions(seed=5), clock)
    cover = simfx.make_driver(PROFILE, simfx.covered_options(ambient_c=15.0, seed=5), clock)
    wrapper = simfx.CoverableDriver(uncovered, cover, cover_after_reads=5 + 1)
    options = DarkSessionOptions(frames=3, bias_frames=5, stable_polls=3, poll_s=1.0)
    result, _ = run(DarkLibrary(tmp_path / "darks"), wrapper, clock, options)
    assert wrapper.reads == 5 + 1 + 3 + 3  # one lit poll, three dark polls, three dark frames
    assert result.n_sets == 1


def test_the_session_gives_up_when_the_camera_stays_uncovered(tmp_path: Path) -> None:
    clock = VirtualClock()
    library = DarkLibrary(tmp_path / "darks")
    uncovered = simfx.make_driver(PROFILE, SimOptions(seed=5), clock)
    options = DarkSessionOptions(frames=3, bias_frames=5, poll_s=5.0, wait_timeout_s=60.0)
    started = clock.monotonic_ns()
    with pytest.raises(DarkError, match="not dark after 60 s") as raised:
        run(library, uncovered, clock, options)
    assert "above the bias" in str(raised.value)  # the reason of the last check
    assert library.sets() == ()
    assert 60.0 < (clock.monotonic_ns() - started) / 1e9 < 90.0  # virtual seconds


def test_no_wait_still_refuses_a_frame_with_light_on_it(tmp_path: Path) -> None:
    clock = VirtualClock()
    library = DarkLibrary(tmp_path / "darks")
    uncovered = simfx.make_driver(PROFILE, SimOptions(seed=5), clock)
    with pytest.raises(DarkError, match=r"dark frame 1 of 3 is not dark"):
        run(library, uncovered, clock, DarkSessionOptions(wait=False, frames=3, bias_frames=3))
    assert library.sets() == ()


def test_a_second_set_at_the_same_temperature_adds_to_the_library(tmp_path: Path) -> None:
    clock = VirtualClock()
    library = DarkLibrary(tmp_path / "darks")
    run(library, covered(14.0, clock=clock), clock)
    result, lines = run(library, covered(14.0, seed=4, clock=clock), clock)
    assert result.n_sets == 2
    model = library.model("bin2", 120)
    assert model is not None
    assert not model.doubling_fitted  # two sets at one temperature fix no slope
    assert "assumed" in "\n".join(lines)


class NoTemperature:
    """A driver whose camera reports no sensor temperature."""

    def __init__(self, inner: SimDriver) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def read_frame(self, timeout_s: float) -> Frame:
        return replace(self._inner.read_frame(timeout_s), temperature_c=None)

    def read_temperature_c(self) -> float | None:
        return None


def test_a_camera_without_a_temperature_cannot_make_a_set(tmp_path: Path) -> None:
    clock = VirtualClock()
    library = DarkLibrary(tmp_path / "darks")
    with pytest.raises(DarkError, match="no sensor temperature"):
        run(library, cast(CameraDriver, NoTemperature(covered(14.0, clock=clock))), clock)
    assert library.sets() == ()


def test_the_options_refuse_nonsense() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        DarkSessionOptions(frames=2)
    with pytest.raises(ValueError, match="exposures must be positive"):
        DarkSessionOptions(exposure_s=0.0)
    with pytest.raises(ValueError, match="stable_polls"):
        DarkSessionOptions(stable_polls=0)
