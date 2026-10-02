"""The flat from panel frames: sources, units, the bias, the tests of a frame, and the result.

The lens of the main tests is `flatfx.OWNER_LENS`, which carries the numbers of a real bench test
(a vignetting of 9.9% in the corners, a tilt of the optics of +0.56% and -0.34%, a source gradient
that flips when the source turns, and a 3% shadow). It sits on a sensor with a quarter of the
pixels in each direction, so the tests run in seconds. `scale_down` keeps the field and the
plate-scale-dependent numbers, and the analysis bins by 2 where the full-size one bins by 4.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from seeingmon.clock import VirtualClock
from seeingmon.profile import Profile
from seeingmon.recordings.ser import ColorId, SerWriter
from seeingmon.solvers import fitsio
from seeingmon.survey import flat_make as fm
from seeingmon.survey.sky import load_flat
from tests.survey import flatfx as fx

QUARTER = (704, 1036)
SMALL = (352, 512)
OWNER_BIAS = fm.LibraryBias(133.9, "from the dark library (test)")
# At a quarter of the pixels, the edge artifacts sit 6 px from the edge, and the margin is 12 px,
# so that the shadow 20 px from the right edge (79 px at full size) stays a shadow.
EDGE_X = 6.0
OWNER_OPTIONS = fm.MakeOptions(bin_factor=2, high_pass_px=20.0, edge_margin_px=12.0)


def geometry_for(shape: tuple[int, int]) -> fm.Geometry:
    profile: Profile = fx.scaled_profile(shape[1], shape[0])
    readout = profile.survey_readout
    return fm.Geometry(shape, profile.plate_scale_arcsec_per_px(readout), readout.adc_bits)


def scale_down(shape: tuple[int, int]) -> float:
    return fx.REFERENCE_SHAPE[1] / shape[1]


@lru_cache(maxsize=4)
def truth_of(name: str, shape: tuple[int, int]) -> fx.FloatImage:
    lens = {"owner": fx.OWNER_LENS, "simulation": fx.SIMULATION_LENS}[name]
    return fx.lens_flat(lens, shape, scale_down=scale_down(shape), edge_artifact_x=EDGE_X)


def owner_sets(
    shape: tuple[int, int] = QUARTER, *, frames: int = 24, turned: bool = True, **options: object
) -> list[fx.ArraySource]:
    truth = truth_of("owner", shape)
    gx, gy = fx.OWNER_SOURCE_GRADIENT
    second = (-gx, -gy) if turned else (gx, gy)
    return [
        fx.panel_set(truth, (gx, gy), frames=frames, seed=1, **options),  # type: ignore[arg-type]
        fx.panel_set(truth, second, frames=frames, seed=2, **options),  # type: ignore[arg-type]
    ]


def run(
    sets: Sequence[fx.ArraySource],
    *,
    shape: tuple[int, int] = QUARTER,
    bias: fm.BiasInput | None = None,
    options: fm.MakeOptions = OWNER_OPTIONS,
    turned: bool = True,
) -> fm.MakeResult:
    return fm.make_flat(
        sets,
        bias=bias or fm.BiasInput(library=OWNER_BIAS),
        geometry=geometry_for(shape),
        options=options,
        source_turned=turned,
        clock=VirtualClock(),
    )


@pytest.fixture(scope="module")
def owner_result() -> fm.MakeResult:
    """Two sets of 24 frames, the source turned by 180 degrees, and the bias from the library."""
    return run(owner_sets())


# --- The frames of the owner's lens ---------------------------------------------------------


class TestTheOwnersLens:
    def test_the_vignetting_matches_the_lens_at_the_five_radii_and_the_corners(
        self, owner_result: fm.MakeResult
    ) -> None:
        points = owner_result.summary.profile
        measured = [p.change_percent for p in points[:5]]
        expected = fx.expected_vignetting_percent(fx.OWNER_LENS, [0.5, 1.0, 1.5, 2.0, 2.5])
        assert all(m is not None for m in measured)
        np.testing.assert_allclose([m for m in measured if m is not None], expected, atol=0.15)
        # the numbers of the bench test: 0.43, 2.25, 4.05, 6.5, and 9.3% at the five radii
        np.testing.assert_allclose(
            [m for m in measured if m is not None], [-0.43, -2.25, -4.05, -6.5, -9.3], atol=0.2
        )
        corners = points[-1]
        assert corners.corner
        assert corners.change_percent == pytest.approx(-9.9, abs=0.3)

    def test_each_set_has_its_tilt_and_the_split_finds_the_source_and_the_optics(
        self, owner_result: fm.MakeResult
    ) -> None:
        first, second = (s.tilt for s in owner_result.sets)
        assert (first.width_percent, first.height_percent) == pytest.approx((-0.08, 0.34), abs=0.06)
        assert (second.width_percent, second.height_percent) == pytest.approx(
            (1.20, -1.02), abs=0.06
        )
        assert owner_result.split is not None
        stays, turns = owner_result.split.stays, owner_result.split.turns
        assert (stays.width_percent, stays.height_percent) == pytest.approx((0.56, -0.34), abs=0.05)
        assert (turns.width_percent, turns.height_percent) == pytest.approx((-0.64, 0.68), abs=0.05)

    def test_the_flat_has_the_tilt_of_the_optics_and_not_that_of_the_source(
        self, owner_result: fm.MakeResult
    ) -> None:
        tilt = owner_result.summary.tilt
        assert (tilt.width_percent, tilt.height_percent) == pytest.approx((0.56, -0.34), abs=0.05)

    def test_the_quotient_of_the_sets_shows_the_plane_of_the_source_and_the_noise(
        self, owner_result: fm.MakeResult
    ) -> None:
        (agreement,) = owner_result.agreements
        assert agreement.plane.width_percent == pytest.approx(-1.29, abs=0.07)
        assert agreement.plane.height_percent == pytest.approx(1.35, abs=0.07)
        assert agreement.smooth_rms == pytest.approx(0.54, abs=0.05)
        assert agreement.expected_fine_rms is not None
        assert agreement.fine_rms == pytest.approx(agreement.expected_fine_rms, rel=0.15)

    def test_the_three_dust_shadows_are_found_where_they_are(
        self, owner_result: fm.MakeResult
    ) -> None:
        found = {(s.x_px, s.y_px): s for s in owner_result.summary.shadows}
        assert len(found) == 3
        for spot in fx.OWNER_LENS.shadows:
            near = [
                s
                for (x, y), s in found.items()
                if abs(x - spot.x / 4.0) < 4 and abs(y - spot.y / 4.0) < 4
            ]
            assert len(near) == 1, spot
            assert near[0].depth == pytest.approx(spot.depth, abs=0.004)
            assert near[0].width_px == pytest.approx(spot.diameter / 4.0, abs=3.0)

    def test_the_two_edge_artifacts_are_listed_apart(self, owner_result: fm.MakeResult) -> None:
        artifacts = owner_result.summary.edge_artifacts
        assert len(artifacts) == 2
        for spot, artifact in zip(
            fx.OWNER_LENS.edge_artifacts, sorted(artifacts, key=lambda a: a.y_px), strict=True
        ):
            assert artifact.x_px == pytest.approx(EDGE_X, abs=3)
            assert artifact.y_px == pytest.approx(spot.y / 4.0, abs=4)
            assert 0.010 < artifact.depth < 0.016
        assert not any(s.at_edge for s in owner_result.summary.shadows)

    def test_the_flat_is_close_to_the_lens(self, owner_result: fm.MakeResult) -> None:
        flat = owner_result.flat
        truth = truth_of("owner", QUARTER)
        assert flat.dtype == np.float32
        assert flat.shape == QUARTER
        assert float(np.median(flat)) == pytest.approx(1.0, abs=1e-6)
        assert float(flat.min()) > 0.5
        error = flat / truth - 1.0
        assert fx.rms(error) < 0.0025  # the noise of the flat is 0.18% per pixel
        assert float(np.abs(error).max()) < 0.012

    def test_the_noise_and_the_levels_are_reported(self, owner_result: fm.MakeResult) -> None:
        first = owner_result.sets[0]
        assert (first.frames, first.used) == (24, 24)
        assert first.mean_level / owner_result.units.full_scale == pytest.approx(0.46, abs=0.02)
        assert first.noise_one_frame == pytest.approx(0.0122, abs=0.001)
        assert first.noise_mean == pytest.approx(first.noise_one_frame / np.sqrt(24), rel=1e-9)
        assert owner_result.noise_flat < first.noise_mean
        assert first.saturated_fraction == 0.0

    def test_a_good_run_with_the_source_turned_has_no_warning(
        self, owner_result: fm.MakeResult
    ) -> None:
        assert owner_result.warnings == ()
        assert owner_result.bias.source == "library"
        assert owner_result.elapsed_s is not None

    def test_the_report_gives_every_number_and_names_no_path(
        self, owner_result: fm.MakeResult
    ) -> None:
        text = "\n".join(fm.format_make_report(owner_result, name="flat.npy"))
        for expected in (
            "Set 1: 24 of 24 frames used",
            "Set 2: 24 of 24 frames used",
            "Bias: 133.9 native counts (535.6 in the counts of the frames), from the dark library",
            "Tilt of each set after the radial part:",
            "set 1: -0.0",
            "Tilt of the optics and the sensor (half the sum of sets 1 and 2",
            "Gradient of the light source in set 1 (half the difference",
            "Quotient of set 1 over set 2:",
            "plane of the quotient: -1.2",
            "0.5 degrees: -0.4",
            "corners (",
            "Shadows deeper than 1%: 3",
            "Edge artifacts deeper than 1% (center within 12 px of an edge): 2",
            "Wrote flat.npy. Set flat_file",
        ):
            assert expected in text, expected
        assert "Warning" not in text
        for fragment in ("\\", "/tmp", "Users"):
            assert fragment not in text


# --- The light source and the sets ----------------------------------------------------------


class TestTheLightSource:
    def test_one_set_warns_that_the_tilt_may_hold_the_gradient_of_the_source(self) -> None:
        result = run(owner_sets(frames=16)[:1], turned=False)
        (warning,) = result.warnings
        assert "gradient of your light source" in warning
        assert "up to about 1% for a phone screen" in warning
        assert result.split is None
        assert result.agreements == ()
        text = "\n".join(fm.format_make_report(result))
        assert "Set 1" not in text  # a single set has no number
        assert "A second set with the source turned by 180 degrees" in text

    def test_one_set_reads_the_tilt_of_the_source_and_the_optics_together(self) -> None:
        # the first test of the owner: one orientation read -1.07% across the width
        result = run(owner_sets(frames=16)[:1], turned=False)
        assert result.summary.tilt.width_percent == pytest.approx(-0.08, abs=0.07)

    def test_sets_that_differ_and_were_not_declared_turned_warn(self) -> None:
        result = run(owner_sets(frames=16), turned=False)
        (warning,) = result.warnings
        assert "Set 1 and set 2 differ in tilt" in warning
        assert "more than 0.3%" in warning
        assert "pass --source-turned" in warning
        text = "\n".join(fm.format_make_report(result))
        assert "Half the sum of sets 1 and 2:" in text
        assert "only if you turned the source by 180 degrees" in text
        assert "Tilt of the optics and the sensor" not in text  # the report does not claim it

    def test_sets_of_a_source_that_stayed_agree_and_give_no_warning(self) -> None:
        result = run(owner_sets(frames=16, turned=False), turned=False)
        (agreement,) = result.agreements
        assert abs(agreement.plane.width_percent) < 0.3
        assert abs(agreement.plane.height_percent) < 0.3
        assert result.warnings == ()
        text = "\n".join(fm.format_make_report(result))
        assert "only if you turned or moved the source" in text

    def test_the_flat_is_the_mean_of_the_sets_whatever_is_declared(self) -> None:
        sets_a = owner_sets(frames=16)
        flat_declared = run(sets_a, turned=True).flat
        flat_undeclared = run(owner_sets(frames=16), turned=False).flat
        np.testing.assert_array_equal(flat_declared, flat_undeclared)


# --- The simulated lens ---------------------------------------------------------------------


class TestTheSimulatedLens:
    def test_a_lens_that_loses_30_percent_in_the_corners_and_tilts_1_5_percent(self) -> None:
        truth = truth_of("simulation", SMALL)
        sets = [
            fx.panel_set(truth, (0.0, 0.0), frames=16, seed=5),
            fx.panel_set(truth, (0.0, 0.0), frames=16, seed=6),
        ]
        options = fm.MakeOptions(bin_factor=2, high_pass_px=10.0, edge_margin_px=12.0)
        result = run(sets, shape=SMALL, options=options, turned=False)
        measured = [p.change_percent for p in result.summary.profile]
        expected = fx.expected_vignetting_percent(
            fx.SIMULATION_LENS, [0.5, 1.0, 1.5, 2.0, 2.5, result.summary.profile[-1].radius_deg]
        )
        assert all(m is not None for m in measured)
        values = [m for m in measured if m is not None]
        np.testing.assert_allclose(values[:5], expected[:5], atol=0.3)
        assert values[5] == pytest.approx(expected[5], abs=0.6)  # the corner ring is 2 pixels wide
        assert result.summary.tilt.width_percent == pytest.approx(1.5, abs=0.1)
        assert fx.rms(result.flat / truth - 1.0) < 0.004
        assert len(result.summary.shadows) == 3
        # a 30% loss makes the flat 1 / 0.7 times as bright at the middle as at the corners
        assert float(result.flat[SMALL[0] // 2, SMALL[1] // 2]) / float(result.flat[0, 0]) > 1.35


# --- The units ------------------------------------------------------------------------------


class TestUnits:
    def frame(self, shift: int) -> np.ndarray:
        rng = np.random.default_rng(1)
        native = rng.integers(2000, 9000, size=(40, 50), dtype=np.uint16)
        return np.asarray(native << np.uint16(shift), dtype=np.uint16)

    def test_a_container_holds_its_counts_in_the_high_bits(self) -> None:
        units = fm.detect_units(self.frame(2), declared_bits=None, adc_bits=14)
        assert (units.full_scale, units.shift) == (65535.0, 2)
        assert units.native_to_file == 4.0

    def test_native_counts_have_the_depth_of_the_profile(self) -> None:
        units = fm.detect_units(self.frame(0), declared_bits=None, adc_bits=14)
        assert (units.full_scale, units.shift) == (16383.0, 0)

    def test_data_above_the_depth_of_the_profile_are_16_bit(self) -> None:
        frame = self.frame(0)
        frame[0, 0] = 40000
        units = fm.detect_units(frame, declared_bits=None, adc_bits=14)
        assert units.full_scale == 65535.0

    def test_a_declared_depth_wins_over_the_profile(self) -> None:
        units = fm.detect_units(self.frame(0), declared_bits=12, adc_bits=14)
        assert units.full_scale == 4095.0

    def test_the_full_scale_can_be_set_by_hand_and_8_bit_data_are_255(self) -> None:
        units = fm.detect_units(self.frame(2), declared_bits=None, adc_bits=14, full_scale=60000)
        assert (units.full_scale, units.shift) == (60000.0, 2)
        eight = (self.frame(0) >> 6).astype(np.uint8)
        assert fm.detect_units(eight, declared_bits=None, adc_bits=14).full_scale == 255.0

    def test_floating_point_data_need_the_full_scale(self) -> None:
        frame = self.frame(0).astype(np.float32)
        with pytest.raises(fm.FlatError, match="floating-point"):
            fm.detect_units(frame, declared_bits=None, adc_bits=14)
        assert fm.detect_units(frame, declared_bits=None, adc_bits=14, full_scale=1e4).shift == 0

    def test_native_frames_give_the_same_flat_and_a_bias_in_native_counts(self) -> None:
        truth = truth_of("owner", SMALL)
        gradient = fx.OWNER_SOURCE_GRADIENT
        sets = [fx.panel_set(truth, gradient, frames=12, seed=3, shift=0)]
        result = run(sets, shape=SMALL, turned=False)
        assert (result.units.full_scale, result.units.shift) == (16383.0, 0)
        assert result.bias.note.startswith("133.9 counts, from the dark library")
        assert result.sets[0].mean_level / 16383.0 == pytest.approx(0.46, abs=0.02)


# --- The frames of a set --------------------------------------------------------------------


class TestTheTestsOfAFrame:
    def stats(self, means: Sequence[float], std: float = 100.0) -> list[fm.FrameStat]:
        return [fm.FrameStat(mean, std, 0.0) for mean in means]

    UNITS = fm.FrameUnits(65535.0, 2)

    def test_frames_outside_20_to_80_percent_of_full_scale_drop(self) -> None:
        selection = fm.select_frames(
            self.stats([5000, 30000, 31000, 60000]), self.UNITS, fm.MakeOptions()
        )
        assert selection.used == (1, 2)
        assert selection.dropped == {fm.LOW: 1, fm.HIGH: 1}

    def test_a_frame_more_than_5_percent_from_the_median_flickers(self) -> None:
        selection = fm.select_frames(
            self.stats([30000, 30100, 29900, 30200, 33000, 27000]), self.UNITS, fm.MakeOptions()
        )
        assert selection.used == (0, 1, 2, 3)
        assert selection.dropped == {fm.FLICKER: 2}

    def test_an_uneven_frame_drops(self) -> None:
        stats = self.stats([30000, 30000, 30000])
        stats[1] = fm.FrameStat(30000.0, 12000.0, 0.0)  # a spread of 40% of the mean
        selection = fm.select_frames(stats, self.UNITS, fm.MakeOptions())
        assert selection.used == (0, 2)
        assert selection.dropped == {fm.UNEVEN: 1}

    def test_the_limits_are_options(self) -> None:
        stats = self.stats([5000, 30000, 33000, 60000])
        wide = fm.MakeOptions(min_level_percent=5.0, max_level_percent=95.0, flicker_percent=200.0)
        assert fm.select_frames(stats, self.UNITS, wide).used == (0, 1, 2, 3)
        tight = fm.MakeOptions(min_level_percent=5.0, max_level_percent=95.0, flicker_percent=20.0)
        selection = fm.select_frames(stats, self.UNITS, tight)
        assert selection.used == (1, 2)  # the two far frames now fail the flicker test
        assert selection.dropped == {fm.FLICKER: 2}

    def test_the_drops_read_as_text_with_the_limits(self) -> None:
        text = fm.describe_drops({fm.LOW: 2, fm.FLICKER: 1}, fm.MakeOptions())
        assert text == ("2 below 20% of full scale, 1 more than 5% from the median level (flicker)")

    def test_dropped_frames_are_reported_and_the_rest_make_the_flat(self) -> None:
        truth = truth_of("owner", SMALL)
        flicker = {2: 0.2, 5: -0.15}
        sets = [
            fx.panel_set(truth, (0.0, 0.0), frames=14, seed=8, flicker=flicker),
            fx.panel_set(truth, (0.0, 0.0), frames=14, seed=9, level=3000.0),  # 19% of full scale
        ]
        with pytest.raises(fm.FlatError, match="no frame of set 2"):
            run(sets, shape=SMALL, turned=False)
        sets[1] = fx.panel_set(truth, (0.0, 0.0), frames=14, seed=9)
        result = run(sets, shape=SMALL, turned=False)
        assert result.sets[0].used == 12
        assert result.sets[0].dropped == {fm.FLICKER: 2}
        assert "Dropped: 2 more than 5% from the median level (flicker)." in "\n".join(
            fm.format_make_report(result)
        )

    def test_a_set_with_no_frame_left_is_an_error_that_says_why(self) -> None:
        truth = truth_of("owner", SMALL)
        sets = [fx.panel_set(truth, (0.0, 0.0), frames=6, seed=1, level=1500.0)]
        with pytest.raises(fm.FlatError) as caught:
            run(sets, shape=SMALL, turned=False)
        message = str(caught.value)
        assert "6 below 20% of full scale" in message
        assert "byte order" in message

    def test_few_frames_and_saturated_pixels_warn(self) -> None:
        truth = truth_of("owner", SMALL)
        source = fx.panel_set(truth, (0.0, 0.0), frames=8, seed=4)
        for index in range(8):
            frame = np.array(source.frame(index))
            frame[:4, :80] = 65532  # 320 saturated pixels of 180224: 0.18%
            source._frames[index] = frame  # type: ignore[index]
        result = run([source], shape=SMALL, turned=False)
        text = " ".join(result.warnings)
        assert "Only 8 frames" in text
        assert "are saturated, over 0.1%" in text
        assert result.sets[0].saturated_fraction == pytest.approx(320 / (352 * 512), rel=0.05)


# --- The bias -------------------------------------------------------------------------------


class TestTheBias:
    def test_the_library_gives_the_bias_at_the_sensor_temperature(self, tmp_path: Path) -> None:
        library = fx.make_library(tmp_path)
        found = fm.library_bias(library, mode="bin2", gain=120, temperature_c=26.9, doubling_c=6.0)
        assert found is not None
        assert found.level_native == pytest.approx(128.0 + 6.2 * (26.9 - 20.2) / 9.8)
        assert "2 sets of bin2 at gain 120, interpolated to 26.9 C" in found.description

    def test_the_bias_stays_at_the_first_and_the_last_set_beyond_them(self, tmp_path: Path) -> None:
        library = fx.make_library(tmp_path)
        cold = fm.library_bias(library, mode="bin2", gain=120, temperature_c=5.0, doubling_c=6.0)
        warm = fm.library_bias(library, mode="bin2", gain=120, temperature_c=45.0, doubling_c=6.0)
        assert cold is not None
        assert warm is not None
        assert (cold.level_native, warm.level_native) == (128.0, 134.2)

    def test_without_a_temperature_the_bias_is_the_mean_of_the_sets(self, tmp_path: Path) -> None:
        library = fx.make_library(tmp_path)
        found = fm.library_bias(library, mode="bin2", gain=120, temperature_c=None, doubling_c=6.0)
        assert found is not None
        assert found.level_native == pytest.approx(131.1)
        assert "no sensor temperature" in found.description

    def test_a_library_without_a_set_of_the_gain_gives_nothing(self, tmp_path: Path) -> None:
        library = fx.make_library(tmp_path)
        assert (
            fm.library_bias(library, mode="bin2", gain=0, temperature_c=20.0, doubling_c=6.0)
            is None
        )
        assert (
            fm.library_bias(
                fx.make_library(tmp_path / "other", biases=()),
                mode="bin2",
                gain=120,
                temperature_c=20.0,
                doubling_c=6.0,
            )
            is None
        )

    def test_a_scalar_beats_the_library(self) -> None:
        bias = fm.BiasInput(level=540.0, library=OWNER_BIAS)
        choice = fm.choose_bias(bias, fm.FrameUnits(65535.0, 2), fm.MakeOptions(), SMALL)
        assert (choice.source, choice.level) == ("level", 540.0)
        assert "from --bias-level" in choice.note

    def test_the_library_serves_when_nothing_else_is_given(self) -> None:
        choice = fm.choose_bias(
            fm.BiasInput(library=OWNER_BIAS), fm.FrameUnits(65535.0, 2), fm.MakeOptions(), SMALL
        )
        assert choice.source == "library"
        assert choice.level == pytest.approx(133.9 * 4)

    def test_no_bias_at_all_is_an_error_that_says_what_to_do(self) -> None:
        with pytest.raises(fm.FlatError, match="record a dark set"):
            fm.choose_bias(fm.BiasInput(), fm.FrameUnits(65535.0, 2), fm.MakeOptions(), SMALL)

    def test_bias_frames_with_the_lens_covered_pass_and_make_a_master(self) -> None:
        frames = fx.bias_set(SMALL, frames=16, seed=2)
        choice = fm.choose_bias(
            fm.BiasInput(frames=frames, library=OWNER_BIAS),
            fm.FrameUnits(65535.0, 2),
            fm.MakeOptions(),
            SMALL,
        )
        assert choice.source == "frames"
        assert choice.master is not None
        assert choice.warnings == ()
        assert choice.level == pytest.approx(133.9 * 4, abs=0.3 * 4)
        assert "from a master of 16 bias frames" in choice.note

    def test_bias_frames_with_light_in_them_raise_the_warning_and_the_library_serves(self) -> None:
        # the owner's frames: a level of 147 against 132, noise of 5 counts, a brighter middle
        frames = fx.bias_set(SMALL, frames=16, seed=2, light=13.0, middle_extra=2.5)
        choice = fm.choose_bias(
            fm.BiasInput(frames=frames, library=OWNER_BIAS),
            fm.FrameUnits(65535.0, 2),
            fm.MakeOptions(),
            SMALL,
        )
        assert choice.source == "library"
        assert choice.master is None
        (warning,) = choice.warnings
        assert "The bias frames hold light, so the command did not use them" in warning
        assert "counts per pixel, over the limit of 3.5" in warning
        assert "brighter than the corners, over 0.5" in warning
        assert "instead of the bias frames" in choice.note

    def test_light_in_the_bias_frames_falls_back_on_the_scalar_before_the_library(self) -> None:
        frames = fx.bias_set(SMALL, frames=16, seed=2, light=13.0, middle_extra=2.5)
        choice = fm.choose_bias(
            fm.BiasInput(level=536.0, frames=frames, library=OWNER_BIAS),
            fm.FrameUnits(65535.0, 2),
            fm.MakeOptions(),
            SMALL,
        )
        assert choice.source == "level"
        assert choice.warnings

    def test_lit_bias_frames_and_nothing_else_is_an_error_that_names_the_check(self) -> None:
        frames = fx.bias_set(SMALL, frames=16, seed=2, light=13.0, middle_extra=2.5)
        with pytest.raises(fm.FlatError, match="failed their check"):
            fm.choose_bias(
                fm.BiasInput(frames=frames),
                fm.FrameUnits(65535.0, 2),
                fm.MakeOptions(),
                SMALL,
            )

    def test_few_bias_frames_warn_and_the_shape_must_match(self) -> None:
        few = fx.bias_set(SMALL, frames=4, seed=2)
        choice = fm.choose_bias(
            fm.BiasInput(frames=few), fm.FrameUnits(65535.0, 2), fm.MakeOptions(), SMALL
        )
        assert "Only 4 bias frames" in choice.warnings[0]
        with pytest.raises(fm.FlatError, match="the bias frames are 512 x 352"):
            fm.choose_bias(
                fm.BiasInput(frames=few), fm.FrameUnits(65535.0, 2), fm.MakeOptions(), (400, 600)
            )

    def test_a_lit_bias_set_still_gives_a_good_flat(self) -> None:
        truth = truth_of("owner", SMALL)
        sets = [fx.panel_set(truth, fx.OWNER_SOURCE_GRADIENT, frames=16, seed=2)]
        lit = fx.bias_set(SMALL, frames=16, seed=3, light=13.0, middle_extra=2.5)
        result = run(
            sets,
            shape=SMALL,
            bias=fm.BiasInput(frames=lit, library=OWNER_BIAS),
            turned=False,
        )
        assert result.bias.source == "library"
        assert any("bias frames hold light" in w for w in result.warnings)
        text = "\n".join(fm.format_make_report(result))
        assert "Warning: The bias frames hold light" in text
        assert "instead of the bias frames" in text
        assert fx.rms(result.flat / truth - 1.0) < 0.012  # the source's tilt stays with one set

    def test_the_bias_frames_flat_matches_the_library_flat_to_a_few_hundredths_of_a_percent(
        self,
    ) -> None:
        truth = truth_of("owner", SMALL)
        capped = fx.bias_set(SMALL, frames=16, seed=3)
        by_frames = run(
            [fx.panel_set(truth, (0.0, 0.0), frames=16, seed=2)],
            shape=SMALL,
            bias=fm.BiasInput(frames=capped),
            turned=False,
        )
        by_library = run(
            [fx.panel_set(truth, (0.0, 0.0), frames=16, seed=2)], shape=SMALL, turned=False
        )
        assert by_frames.bias.source == "frames"
        assert by_library.bias.source == "library"
        assert fx.rms(by_frames.flat / by_library.flat - 1.0) < 0.0005


# --- Inputs that are wrong ------------------------------------------------------------------


class TestWrongInputs:
    def test_frames_of_another_size_are_an_error_that_names_both_sizes(self) -> None:
        truth = truth_of("owner", SMALL)
        sets = [fx.panel_set(truth, (0.0, 0.0), frames=4, seed=1)]
        with pytest.raises(fm.FlatError) as caught:
            run(sets, shape=QUARTER, turned=False)
        message = str(caught.value)
        assert "512 x 352 pixels" in message
        assert "1036 x 704" in message
        assert "no region of interest" in message

    def test_no_sets_are_an_error(self) -> None:
        with pytest.raises(fm.FlatError, match="give --frames"):
            run([], turned=False)

    def test_bad_options_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="level limits"):
            fm.MakeOptions(min_level_percent=60.0, max_level_percent=40.0)
        with pytest.raises(ValueError, match="positive"):
            fm.MakeOptions(clip_sigma=0.0)
        with pytest.raises(ValueError, match="binning"):
            fm.MakeOptions(bin_factor=0)

    def test_a_dead_pixel_takes_a_floor_and_a_warning(self) -> None:
        truth = truth_of("owner", SMALL)
        source = fx.panel_set(truth, (0.0, 0.0), frames=16, seed=4)
        for index in range(16):
            frame = np.array(source.frame(index))
            frame[100, 200] = 0  # a pixel that never sees light
            source._frames[index] = frame  # type: ignore[index]
        result = run([source], shape=SMALL, turned=False)
        assert float(result.flat.min()) == pytest.approx(fm.MIN_FLAT)
        assert any("1 pixels read at or below zero" in w for w in result.warnings)

    def test_a_cosmic_ray_in_one_frame_does_not_reach_the_flat(self) -> None:
        truth = truth_of("owner", SMALL)
        source = fx.panel_set(truth, (0.0, 0.0), frames=16, seed=4)
        clean = run([fx.panel_set(truth, (0.0, 0.0), frames=16, seed=4)], shape=SMALL, turned=False)
        frame = np.array(source.frame(3))
        frame[150, 250] = 60000  # a hit in one frame
        source._frames[3] = frame  # type: ignore[index]
        hit = run([source], shape=SMALL, turned=False)
        difference = float(abs(hit.flat[150, 250] / clean.flat[150, 250] - 1.0))
        assert difference < 0.002  # the sigma clip drops the hit

    def test_each_frame_is_read_three_times_and_the_levels_once_more_for_the_units(self) -> None:
        sets = owner_sets(SMALL, frames=6)
        run(sets, shape=SMALL)
        # the unit check reads one frame of the first set, then the levels (1), the moments (1),
        # and the clipped sum (1) read each frame of each set
        assert sets[1].reads == 6 * 3
        assert sets[0].reads == 6 * 3 + 1


# --- Files ----------------------------------------------------------------------------------


def write_fits(path: Path, frame: np.ndarray, **header: object) -> None:
    fitsio.write_image(path, frame, header=header)  # type: ignore[arg-type]


def sharpcap_style_fits(
    frame: np.ndarray, *, extra: Sequence[str] = (), row_order: str | None = None
) -> bytes:
    """A FITS file as a capture program writes it: BZERO 32768, comments, and its own cards."""
    height, width = frame.shape
    cards = [
        "SIMPLE  =                    T / file does conform to FITS standard",
        "BITPIX  =                   16 / number of bits per data pixel",
        "NAXIS   =                    2 / number of data axes",
        f"NAXIS1  =           {width:>10d} / length of data axis 1",
        f"NAXIS2  =           {height:>10d} / length of data axis 2",
        "EXTEND  =                    T / FITS dataset may contain extensions",
        "COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy",
        "COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H",
        "BZERO   =                32768 / offset data range to that of unsigned short",
        "BSCALE  =                    1 / default scaling factor",
        "INSTRUME= 'ZWO ASI294MM'       / detector / a slash inside a comment",
        "EXPTIME =                  0.1 / [s] exposure",
        "GAIN    =                  120 / camera gain",
        "CCD-TEMP=                 28.6 / [degC] sensor temperature",
        "HISTORY made by a test",
        *extra,
    ]
    if row_order is not None:
        cards.append(f"ROWORDER= '{row_order}'")
    cards.append("END")
    header = "".join(f"{card:<80}" for card in cards).encode("ascii")
    header += b" " * (-len(header) % 2880)
    data = (frame.astype(np.int32) - 32768).astype(">i2").tobytes()
    data += b"\0" * (-len(data) % 2880)
    return header + data


class TestFitsFiles:
    def frame(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(10000, 40000, size=(32, 48), dtype=np.uint16)

    def test_a_folder_of_fits_files_is_a_set_with_its_temperature_and_gain(
        self, tmp_path: Path
    ) -> None:
        frames = [self.frame(i) for i in range(3)]
        for index, frame in enumerate(frames):
            write_fits(
                tmp_path / f"flat-{index:03d}.fits",
                frame,
                **{"CCD-TEMP": 28.0 + index, "GAIN": 120, "ADCBITS": 14},
            )
        (tmp_path / ".hidden.tmp").write_bytes(b"not fits")
        (tmp_path / "notes.txt").write_text("a note", encoding="utf-8")
        source = fm.open_frames(tmp_path)
        assert (source.count, source.shape) == (3, (32, 48))
        assert source.temperature_c == pytest.approx(29.0)
        assert source.gain == 120
        assert source.declared_bits == 14
        for index, frame in enumerate(frames):
            np.testing.assert_array_equal(source.frame(index), frame)

    def test_the_reader_takes_16_bit_fits_with_bzero_as_a_capture_program_writes_it(
        self, tmp_path: Path
    ) -> None:
        frame = self.frame(4)
        path = tmp_path / "capture.fits"
        path.write_bytes(sharpcap_style_fits(frame))
        source = fm.open_frames(tmp_path)
        assert source.temperature_c == pytest.approx(28.6)
        assert source.gain == 120
        assert source.declared_bits is None
        assert source.frame(0).dtype == np.uint16
        np.testing.assert_array_equal(source.frame(0), frame)

    def test_a_single_fits_file_is_a_set_of_one_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "one.fit"
        path.write_bytes(sharpcap_style_fits(self.frame(5)))
        assert fm.open_frames(path).count == 1

    def test_a_file_that_says_bottom_up_is_turned_to_the_row_order_of_the_survey_frames(
        self, tmp_path: Path
    ) -> None:
        frame = self.frame(6)
        (tmp_path / "up.fits").write_bytes(sharpcap_style_fits(frame, row_order="BOTTOM-UP"))
        np.testing.assert_array_equal(fm.open_frames(tmp_path).frame(0), frame[::-1])
        (tmp_path / "up.fits").write_bytes(sharpcap_style_fits(frame, row_order="TOP-DOWN"))
        np.testing.assert_array_equal(fm.open_frames(tmp_path).frame(0), frame)

    def test_files_of_different_sizes_and_a_folder_without_fits_are_errors(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(fm.FlatError, match="no FITS file"):
            fm.open_frames(tmp_path)
        write_fits(tmp_path / "a.fits", self.frame())
        write_fits(tmp_path / "b.fits", self.frame()[:30])
        with pytest.raises(fm.FlatError, match="differ in size"):
            fm.open_frames(tmp_path)

    def test_a_missing_path_and_a_bad_file_are_errors_that_name_no_path(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(fm.FlatError) as missing:
            fm.open_frames(tmp_path / "nowhere")
        assert str(tmp_path) not in str(missing.value)
        bad = tmp_path / "bad.fits"
        bad.write_bytes(b"this is not a FITS file at all")
        with pytest.raises(fm.FlatError) as broken:
            fm.open_frames(bad)
        assert str(tmp_path) not in str(broken.value)

    def test_a_3d_fits_file_is_rejected(self, tmp_path: Path) -> None:
        text = sharpcap_style_fits(self.frame()).replace(
            b"NAXIS   =                    2", b"NAXIS   =                    3", 1
        )
        (tmp_path / "color.fits").write_bytes(text)
        with pytest.raises(fm.FlatError, match="2-D image"):
            fm.open_frames(tmp_path)


class TestSerFiles:
    def write(self, path: Path, frames: Sequence[np.ndarray], **options: object) -> None:
        with SerWriter(
            path,
            width=frames[0].shape[1],
            height=frames[0].shape[0],
            timestamps=False,
            **options,  # type: ignore[arg-type]
        ) as writer:
            for frame in frames:
                writer.write_frame(frame)

    def test_a_ser_recording_is_a_set(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(1)
        frames = [rng.integers(8000, 40000, size=(32, 48), dtype=np.uint16) for _ in range(4)]
        path = tmp_path / "flat.ser"
        self.write(path, frames, pixel_depth=16)
        source = fm.open_frames(path)
        try:
            assert (source.count, source.shape) == (4, (32, 48))
            assert source.declared_bits is None  # 16 bits: the container, or the real 16 bits
            assert source.temperature_c is None
            assert source.gain is None
            for index, frame in enumerate(frames):
                np.testing.assert_array_equal(source.frame(index), frame)
        finally:
            source.close()

    def test_a_depth_below_16_bits_is_declared(self, tmp_path: Path) -> None:
        frames = [np.full((8, 8), 1000, dtype=np.uint16)]
        path = tmp_path / "flat.ser"
        self.write(path, frames, pixel_depth=14)
        source = fm.open_frames(path)
        try:
            assert source.declared_bits == 14
        finally:
            source.close()

    def test_the_byte_order_can_be_forced(self, tmp_path: Path) -> None:
        frame = np.full((8, 8), 0x1234, dtype=np.uint16)
        path = tmp_path / "flat.ser"
        self.write(path, [frame], pixel_depth=16, byte_order="big")
        wrong = fm.open_frames(path, byte_order="little")
        right = fm.open_frames(path)
        try:
            assert int(right.frame(0)[0, 0]) == 0x1234
            assert int(wrong.frame(0)[0, 0]) == 0x3412
        finally:
            wrong.close()
            right.close()

    def test_a_color_recording_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "color.ser"
        with SerWriter(
            path, width=8, height=8, pixel_depth=16, color=ColorId.RGB, timestamps=False
        ) as writer:
            writer.write_frame(np.zeros((8, 8, 3), dtype=np.uint16))
        with pytest.raises(fm.FlatError, match="color planes"):
            fm.open_frames(path)

    def test_a_file_that_is_not_ser_is_an_error_that_names_no_path(self, tmp_path: Path) -> None:
        path = tmp_path / "flat.ser"
        path.write_bytes(b"x" * 400)
        with pytest.raises(fm.FlatError) as caught:
            fm.open_frames(path)
        assert str(tmp_path) not in str(caught.value)


class TestTheWrittenFlat:
    def test_the_flat_file_loads_as_the_flat_of_the_survey_path(self, tmp_path: Path) -> None:
        from seeingmon.survey.flat_files import read_flat_image, write_flat

        truth = truth_of("owner", SMALL)
        result = run(
            [fx.panel_set(truth, fx.OWNER_SOURCE_GRADIENT, frames=8, seed=1)],
            shape=SMALL,
            turned=False,
        )
        for name in ("flat.npy", "flat.fits"):
            path = tmp_path / name
            write_flat(path, result.flat)
            loaded = load_flat(path)
            assert loaded.version.startswith("flat-")
            image = loaded.image(SMALL)
            assert image is not None
            assert image.dtype == np.float32
            np.testing.assert_allclose(image, result.flat, rtol=1e-6)
            np.testing.assert_allclose(read_flat_image(path), result.flat, rtol=1e-6)
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]  # no temporary file

    def test_a_flat_file_that_cannot_be_read_gives_a_message_without_the_path(
        self, tmp_path: Path
    ) -> None:
        from seeingmon.survey.flat_files import FlatFileError, read_flat_image

        with pytest.raises(FlatFileError) as caught:
            read_flat_image(tmp_path / "missing.npy")
        assert str(caught.value) == "cannot read the flat file: No such file or directory"
        negative = tmp_path / "negative.npy"
        np.save(negative, -np.ones((4, 4), dtype=np.float32))
        with pytest.raises(FlatFileError, match="positive"):
            read_flat_image(negative)

    def test_a_flat_file_needs_a_known_suffix(self, tmp_path: Path) -> None:
        from seeingmon.survey.flat_files import FlatFileError, write_flat

        with pytest.raises(FlatFileError, match=r"\.npy"):
            write_flat(tmp_path / "flat.txt", np.ones((4, 4), dtype=np.float32))


@pytest.mark.slow
class TestAFullSizeSensor:
    """The owner's lens on the real sensor size, with the real binning: 4144 x 2822 pixels."""

    def test_the_numbers_of_the_bench_test(self) -> None:
        shape = fx.REFERENCE_SHAPE
        truth = fx.lens_flat(fx.OWNER_LENS, shape, scale_down=1.0)
        gx, gy = fx.OWNER_SOURCE_GRADIENT
        sets = [
            fx.panel_set(truth, (gx, gy), frames=8, seed=1, on_demand=True),
            fx.panel_set(truth, (-gx, -gy), frames=8, seed=2, on_demand=True),
        ]
        profile = fx.scaled_profile(shape[1], shape[0])
        readout = profile.survey_readout
        geometry = fm.Geometry(shape, profile.plate_scale_arcsec_per_px(readout), readout.adc_bits)
        result = fm.make_flat(
            sets,
            bias=fm.BiasInput(library=OWNER_BIAS),
            geometry=geometry,
            options=fm.MakeOptions(min_frames=8),
            source_turned=True,
        )
        measured = [p.change_percent for p in result.summary.profile[:5]]
        np.testing.assert_allclose(
            [m for m in measured if m is not None], [-0.43, -2.25, -4.05, -6.5, -9.3], atol=0.15
        )
        assert result.split is not None
        assert result.split.stays.width_percent == pytest.approx(0.56, abs=0.05)
        assert result.split.turns.width_percent == pytest.approx(-0.64, abs=0.05)
        shadows = {(round(s.x_px, -1), round(s.y_px, -1)): s for s in result.summary.shadows}
        assert len(shadows) == 3
        assert any(abs(x - 2489) < 20 and abs(y - 1994) < 20 for x, y in shadows)
        assert len(result.summary.edge_artifacts) == 2
        assert result.warnings == ()
