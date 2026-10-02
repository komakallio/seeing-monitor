"""The flat from the night sky: the tests of a frame, the masks, the accumulator, and the recipe.

The night is `skyfx`: a sky that turns about the middle of the frame (a structure, stars, and a
Polaris with a halo) seen through the lens of `flatfx`, with the ground-fixed gradient of the sky
and a dark library. The frames go to FITS files as `core` writes them, and the command reads them.
The sensor has 512 x 352 pixels, and the analysis bins by 2, so that a test runs in seconds. The
errors are measured against the lens that made the frames:

- the *radial error*, the rms of the azimuthal mean of the estimated flat over the true flat, over
  the rings that the rotation averages (not the middle, where the sky does not turn);
- the *fine error*, the rms of the difference of the fine parts;
- the *total error*, the rms of the estimated flat over the true flat, which holds the tilt that the
  sky cannot give, and the *unit error*, the same for a flat of ones.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, VirtualClock, iso_to_utc_ns
from seeingmon.frames import FrameFlag
from seeingmon.profile import Profile
from seeingmon.store.layout import DataLayout
from seeingmon.survey import flat_report as fr
from seeingmon.survey import flat_sky as fs
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.dark import DarkLibrary
from seeingmon.survey.framefile import FrameMeta
from tests.survey import flatfx as fx
from tests.survey import skyfx as sf

SHAPE = (352, 512)
SCALE_DOWN = fx.REFERENCE_SHAPE[1] / SHAPE[1]
FACTOR = 2
# The real defaults, scaled to a sensor with 8.09 times fewer pixels in each direction: the mask
# of Polaris of 400 px is 49 px, the Gaussian of 40 binned pixels at 4 x 4 binning is 20 at 2 x 2,
# and the margin of 20 px for an edge artifact is 12 px.
OPTIONS = fs.BuildOptions(
    bin_factor=FACTOR, high_pass_px=20.0, polaris_mask_px=49.0, edge_margin_px=12.0
)
EDGE_X = 6.0


@dataclass(frozen=True)
class Night:
    """A synthetic night on disk: the frames, the dark library, and the truth that made them."""

    layout: DataLayout
    profile: Profile
    sky: sf.NightSky
    truth: fx.FloatImage
    library: DarkLibrary
    times: tuple[int, ...]

    @property
    def folder(self) -> Path:
        return self.layout.survey_dir


def make_night(
    root: Path,
    lens: fx.LensSpec,
    *,
    frames: int = 24,
    times: Sequence[int] | None = None,
    halo: sf.Halo | None = None,
    write: bool = True,
    seed: int = 21,
    **options: object,
) -> Night:
    truth = fx.lens_flat(lens, SHAPE, scale_down=SCALE_DOWN, edge_artifact_x=EDGE_X)
    profile = fx.scaled_profile(SHAPE[1], SHAPE[0])
    scale = profile.plate_scale_arcsec_per_px(profile.survey_readout)
    orbit = fs.polaris_separation_deg(sf.NIGHT_START_NS) * 3600.0 / scale
    sky = sf.NightSky(truth, orbit_px=orbit, halo=halo)
    layout = DataLayout(root / "data")
    layout.create()
    sf.make_night_library(root / "calibration", sky)
    chosen = tuple(times) if times is not None else tuple(sf.night_times(frames))
    if write:
        sf.write_night(layout, sky, profile, chosen, seed=seed, **options)  # type: ignore[arg-type]
    return Night(layout, profile, sky, truth, DarkLibrary(root / "calibration" / "darks"), chosen)


def build(
    night: Night,
    *,
    folder: Path | str | None = "frames",
    options: fs.BuildOptions = OPTIONS,
    accumulator: Path | None = None,
    site: object = sf.TEST_SITE,
    library: DarkLibrary | str | None = "night",
) -> fs.SkyResult:
    return fs.build_sky_flat(
        night.folder if folder == "frames" else folder,  # type: ignore[arg-type]
        profile=night.profile,
        survey=SurveyConfig(),
        library=night.library if library == "night" else library,  # type: ignore[arg-type]
        site=site,  # type: ignore[arg-type]
        accumulator_path=accumulator,
        options=options,
        clock=VirtualClock(),
    )


@dataclass(frozen=True)
class Errors:
    radial: float
    fine: float
    total: float
    unit: float


def errors_of(flat: fx.FloatImage, truth: fx.FloatImage) -> Errors:
    """The errors of an estimated flat against the true one, in percent."""
    center = ((SHAPE[1] - 1) / 2.0, (SHAPE[0] - 1) / 2.0)
    center_binned = fr.binned_position(center, FACTOR)
    estimated, true = fr.block_mean(flat, FACTOR), fr.block_mean(truth, FACTOR)
    ratio = fr.azimuthal_profile(estimated / true, fr.radius_map(estimated.shape, center_binned))
    rings = ratio.radius >= 10.0  # the middle holds the sky that does not turn
    fine_estimated = fr.decompose(estimated, center_xy=center_binned, high_pass_px=20.0).fine
    fine_true = fr.decompose(true, center_xy=center_binned, high_pass_px=20.0).fine
    return Errors(
        radial=100.0 * fx.rms(ratio.value[rings] - 1.0),
        fine=100.0 * fx.rms(fine_estimated / fine_true - 1.0),
        total=100.0 * fx.rms(flat / truth - 1.0),
        unit=100.0 * fx.rms(1.0 / truth - 1.0),
    )


@pytest.fixture(scope="module")
def owner_night(tmp_path_factory: pytest.TempPathFactory) -> Night:
    return make_night(tmp_path_factory.mktemp("owner"), fx.OWNER_LENS)


@pytest.fixture(scope="module")
def owner_run(owner_night: Night) -> fs.SkyResult:
    return build(owner_night)


@pytest.fixture(scope="module")
def simulation_night(tmp_path_factory: pytest.TempPathFactory) -> Night:
    return make_night(tmp_path_factory.mktemp("simulation"), fx.SIMULATION_LENS)


@pytest.fixture(scope="module")
def simulation_run(simulation_night: Night) -> fs.SkyResult:
    return build(simulation_night)


# --- One clear night of the owner's lens ----------------------------------------------------


class TestAClearNight:
    def test_all_24_frames_go_in_and_the_roll_covers_the_night(
        self, owner_run: fs.SkyResult
    ) -> None:
        assert owner_run.selection.found == 24
        assert owner_run.used_now == 24
        assert not owner_run.selection.rejected
        acc = owner_run.accumulator
        assert len(acc.frames) == 24
        coverage = fs.roll_coverage([r.roll_deg for r in acc.frames])
        assert coverage == pytest.approx(173.0, abs=1.0)  # 11.5 hours at 15.04 degrees an hour
        assert owner_run.warnings == ()

    def test_the_recipe_reaches_the_noise_floor_of_the_night(
        self, owner_run: fs.SkyResult, owner_night: Night
    ) -> None:
        e = errors_of(owner_run.flat, owner_night.truth)
        # The simulation of the recipe gave 0.14% for the radial profile and 0.31% for the fine
        # part after one clear night. This sky has more noise a binned pixel (0.57%).
        assert e.radial < 0.35  # 0.19% measured, and the rings of the sky set the floor
        assert e.fine < 0.65  # 0.45% measured, from the noise of a binned pixel
        assert e.total < 0.95  # 0.60% measured
        assert e.unit / e.total >= 3.5  # a lens of 10% gains a factor of 4 over no flat

    def test_the_flat_is_a_float32_image_of_the_sensor_with_a_median_of_one(
        self, owner_run: fs.SkyResult
    ) -> None:
        flat = owner_run.flat
        assert flat.dtype == np.float32
        assert flat.shape == SHAPE
        assert float(np.median(flat)) == pytest.approx(1.0, abs=1e-3)
        assert float(flat.min()) > 0.5

    def test_the_dark_that_the_frames_used_is_the_master_of_the_set(
        self, owner_run: fs.SkyResult
    ) -> None:
        assert owner_run.dark_note == "the master dark of the set of 2026-12-01 for 24 frames"

    def test_the_noise_of_a_binned_pixel_is_reported(self, owner_run: fs.SkyResult) -> None:
        assert owner_run.average.noise_mean == pytest.approx(0.0056, abs=0.0015)
        assert owner_run.average.noise_frame == pytest.approx(0.053, abs=0.008)

    def test_the_big_shadow_is_found_and_the_noise_sets_the_depth_of_the_search(
        self, owner_run: fs.SkyResult
    ) -> None:
        summary = owner_run.summary
        assert summary.shadow_depth > fr.SHADOW_MIN_DEPTH  # 5 times the noise of the fine part
        big = fx.OWNER_LENS.shadows[0]
        near = [
            s
            for s in summary.shadows
            if abs(s.x_px - big.x / SCALE_DOWN) < 6 and abs(s.y_px - big.y / SCALE_DOWN) < 6
        ]
        assert len(near) == 1
        assert near[0].depth == pytest.approx(big.depth, abs=0.012)
        assert near[0].width_px == pytest.approx(big.diameter / SCALE_DOWN, abs=3.0)

    def test_polaris_is_found_in_every_frame_and_the_orbit_comes_from_its_positions(
        self, owner_run: fs.SkyResult, owner_night: Night
    ) -> None:
        records = owner_run.accumulator.frames
        assert all(np.isfinite(r.polaris_x) for r in records)
        for record in records[::6]:
            expected = owner_night.sky.polaris_position(sf.roll_rad(record.t_utc_ns))
            assert (record.polaris_x, record.polaris_y) == pytest.approx(expected, abs=0.6)
        ring = owner_run.ring
        assert ring.measured
        assert ring.radius_px == pytest.approx(owner_night.sky.orbit_px, abs=1.5)
        assert ring.center_xy == pytest.approx(owner_night.sky.center, abs=1.0)
        assert ring.bump is not None
        assert ring.bump < 0.003  # a mask that is large enough leaves no ring

    def test_the_report_gives_every_number_and_names_no_path(self, owner_run: fs.SkyResult) -> None:
        text = "\n".join(fs.format_sky_report(owner_run, name="flat.npy"))
        for expected in (
            "Frames: 24 found in the folder, 0 already in the accumulator, 0 rejected, 24 added",
            "Accumulator: 24 frames from 2026-12-09 to 2026-12-10. Roll coverage: 173 degrees.",
            "Dark: the master dark of the set of 2026-12-01 for 24 frames.",
            "Noise: 0.5",
            "Vignetting at each radius",
            "0.5 degrees:",
            "2.0 degrees:",
            "Tilt: not determined. The sky cannot tell a tilt of the flat from a gradient",
            "Polaris orbit: radius 72 px (from the positions of Polaris), bump +0.",
            "Time: 0.0 s",
            "Wrote flat.npy. Set flat_file",
        ):
            assert expected in text, expected
        assert "Warning" not in text
        for fragment in ("\\", "/tmp", "Users", "survey/"):
            assert fragment not in text


class TestASecondNight:
    def test_a_second_night_in_the_same_accumulator_lowers_the_noise_and_the_fine_error(
        self, owner_run: fs.SkyResult, owner_night: Night, tmp_path: Path
    ) -> None:
        accumulator = tmp_path / "acc.npz"
        owner_run.accumulator.save(accumulator)  # the first night
        second = sf.night_times(24, start_ns=sf.NIGHT_START_NS + 24 * 3600 * NS_PER_S)
        night = make_night(tmp_path / "second", fx.OWNER_LENS, times=second, seed=77)
        both = build(night, accumulator=accumulator)
        assert both.used_now == 24
        assert len(both.accumulator.frames) == 48
        assert both.warnings == ()
        before = errors_of(owner_run.flat, owner_night.truth)
        after = errors_of(both.flat, owner_night.truth)
        # the simulation gave 0.31% and 0.24% for 24 and 48 frames: the noise falls by 1.4
        assert after.fine < before.fine
        assert after.fine < 0.5  # 0.37% measured
        assert after.total < 0.8  # 0.53% measured
        assert both.average.noise_mean < 0.8 * owner_run.average.noise_mean
        assert "48 frames from 2026-12-09 to 2026-12-11" in "\n".join(fs.format_sky_report(both))


class TestTheSimulationLens:
    """The lens of the simulation of the recipe: 30% radial vignetting and a 1.5% tilt."""

    def test_the_vignetting_comes_out_at_the_five_radii(self, simulation_run: fs.SkyResult) -> None:
        points = simulation_run.summary.profile
        expected = fx.expected_vignetting_percent(fx.SIMULATION_LENS, [0.5, 1.0, 1.5, 2.0, 2.5])
        measured = [p.change_percent for p in points[:5]]
        assert all(m is not None for m in measured)
        np.testing.assert_allclose([m for m in measured if m is not None], expected, atol=1.0)

    def test_the_tilt_is_left_out_because_the_sky_cannot_give_it(
        self, simulation_run: fs.SkyResult, simulation_night: Night
    ) -> None:
        center = fr.binned_position(simulation_night.sky.center, FACTOR)
        estimated = fr.block_mean(simulation_run.flat, FACTOR)
        true = fr.block_mean(simulation_night.truth, FACTOR)
        tilt_estimated = fr.tilt_after_radial(
            estimated,
            fr.decompose(estimated, center_xy=center, high_pass_px=20.0),
            center_xy=center,
        )
        tilt_true = fr.tilt_after_radial(
            true, fr.decompose(true, center_xy=center, high_pass_px=20.0), center_xy=center
        )
        assert tilt_true.width_percent == pytest.approx(1.5, abs=0.1)  # the lens has a tilt
        assert abs(tilt_estimated.width_percent) < 0.3  # and the flat from the sky has none
        assert abs(tilt_estimated.height_percent) < 0.3

    def test_the_recipe_beats_no_flat_by_a_factor_of_four_or_more(
        self, simulation_run: fs.SkyResult, simulation_night: Night
    ) -> None:
        e = errors_of(simulation_run.flat, simulation_night.truth)
        assert e.unit > 5.0  # a lens of 30% is wrong by 8% rms without a flat
        assert e.unit / e.total >= 4.0  # 11.7 measured
        assert e.radial < 0.35  # 0.18% measured
        assert e.fine < 0.65  # 0.45% measured
        # almost all of the error that is left is the tilt that the sky cannot give
        assert e.total < 1.0  # 0.71% measured, against 0.43% for a tilt of 1.5% alone

    def test_the_gradient_that_is_fixed_to_the_ground_stays_out_of_the_flat(
        self, simulation_run: fs.SkyResult
    ) -> None:
        # a gradient of 4.5% across the height sits in every frame, and the flat holds none of it
        estimated = fr.block_mean(simulation_run.flat, FACTOR)
        center = ((estimated.shape[1] - 1) / 2.0, (estimated.shape[0] - 1) / 2.0)
        tilt = fr.fit_tilt(estimated, center_xy=center)
        assert abs(tilt.height_percent) < 0.5


# --- The tests that rule a frame out --------------------------------------------------------


def tiny_file(layout: DataLayout, profile: Profile, t_utc_ns: int, **options: object) -> Path:
    """A frame of 8 x 8 pixels with the header of a survey frame, for the tests of the header."""
    pixels = np.full((8, 8), 4000, dtype=np.uint16)
    return sf.write_frame(layout, pixels, t_utc_ns, profile, **options)  # type: ignore[arg-type]


@pytest.fixture
def small(tmp_path: Path) -> Night:
    return make_night(tmp_path, fx.OWNER_LENS, write=False)


MIDNIGHT = sf.night_times(1)[0] + 6 * 3600 * NS_PER_S  # the middle of the night of the tests
FULL_MOON_NIGHT = iso_to_utc_ns("2026-12-23T00:00:00Z")  # the Moon is up and 98% lit
CRESCENT_NIGHT = iso_to_utc_ns("2026-12-13T17:30:00Z")  # the Moon is up and 18% lit
DAYTIME = iso_to_utc_ns("2026-12-09T11:00:00Z")


def screen(
    night: Night, t_utc_ns: int = MIDNIGHT, *, options: fs.BuildOptions = OPTIONS, **header: object
) -> fs.Selection:
    path = tiny_file(night.layout, night.profile, t_utc_ns, **header)
    return fs.screen_headers([path], mode="bin2", known=set(), site=sf.TEST_SITE, options=options)


class TestTheHeaderTests:
    def test_a_clear_night_frame_passes(self, small: Night) -> None:
        selection = screen(small)
        assert len(selection.candidates) == 1
        assert not selection.rejected

    @pytest.mark.parametrize(
        "reasons", ["event:cloud", "event:unsolved", "event:moved", "event:bright_sky"]
    )
    def test_event_frames_stay_out(self, small: Night, reasons: str) -> None:
        selection = screen(small, reasons=("every_10", reasons))
        assert selection.rejected == {fs.EVENT: 1}

    def test_a_frame_that_is_only_every_tenth_is_not_an_event(self, small: Night) -> None:
        assert len(screen(small, reasons=("every_10",)).candidates) == 1

    def test_the_cloud_fraction_must_stay_under_the_limit(self, small: Night) -> None:
        assert screen(small, cloud_fraction=0.09).rejected == {}
        assert screen(small, MIDNIGHT + 1_000_000_000, cloud_fraction=0.10).rejected == {
            fs.CLOUD: 1
        }

    def test_the_transparency_must_reach_the_limit(self, small: Night) -> None:
        assert screen(small, transparency=0.95).rejected == {}
        assert screen(small, MIDNIGHT + 1_000_000_000, transparency=0.94).rejected == {
            fs.TRANSPARENCY: 1
        }

    def test_the_sun_must_be_below_18_degrees(self, small: Night) -> None:
        assert screen(small, DAYTIME).rejected == {fs.SUN: 1}

    def test_the_moon_may_be_up_when_it_is_under_a_quarter_lit(self, small: Night) -> None:
        assert screen(small, FULL_MOON_NIGHT).rejected == {fs.MOON: 1}
        assert screen(small, CRESCENT_NIGHT).rejected == {}

    def test_every_threshold_is_an_option(self, small: Night) -> None:
        loose = dataclasses.replace(
            OPTIONS,
            max_cloud_fraction=0.6,
            min_transparency=0.5,
            max_sun_elevation_deg=20.0,
            max_moon_illumination=1.0,
        )
        for index, (t, header) in enumerate(
            [
                (MIDNIGHT, {"cloud_fraction": 0.5}),
                (MIDNIGHT + 1_000_000_000, {"transparency": 0.6}),
                (DAYTIME, {}),
                (FULL_MOON_NIGHT, {}),
            ]
        ):
            assert not screen(small, t + index, options=loose, **header).rejected, header
        strict = dataclasses.replace(OPTIONS, max_moon_illumination=0.1)
        assert screen(small, CRESCENT_NIGHT, options=strict).rejected == {fs.MOON: 1}
        higher = dataclasses.replace(OPTIONS, moon_min_elevation_deg=15.0)
        assert not screen(small, CRESCENT_NIGHT + 1, options=higher).rejected  # 12 degrees up

    def test_a_frame_without_the_cloud_fraction_or_the_transparency_stays_out(
        self, small: Night
    ) -> None:
        assert screen(small, cloud_fraction=None).rejected == {fs.NO_CLOUD: 1}
        assert screen(small, MIDNIGHT + 1_000_000_000, transparency=None).rejected == {
            fs.NO_TRANSPARENCY: 1
        }

    def test_accept_unchecked_lets_such_a_frame_in_and_counts_it(self, small: Night) -> None:
        options = dataclasses.replace(OPTIONS, accept_unchecked=True)
        selection = screen(small, options=options, cloud_fraction=None, transparency=None)
        assert len(selection.candidates) == 1
        assert selection.unchecked == {fs.NO_CLOUD: 1, fs.NO_TRANSPARENCY: 1}

    def test_a_frame_that_is_clouded_stays_out_even_when_unchecked_frames_may_enter(
        self, small: Night
    ) -> None:
        options = dataclasses.replace(OPTIONS, accept_unchecked=True)
        assert screen(small, options=options, cloud_fraction=0.5).rejected == {fs.CLOUD: 1}

    def test_without_a_site_the_sun_and_the_moon_cannot_be_checked(self, small: Night) -> None:
        path = tiny_file(small.layout, small.profile, MIDNIGHT)
        none = fs.screen_headers([path], mode="bin2", known=set(), site=None, options=OPTIONS)
        assert none.rejected == {fs.NO_SITE: 1}
        options = dataclasses.replace(OPTIONS, accept_unchecked=True)
        some = fs.screen_headers([path], mode="bin2", known=set(), site=None, options=options)
        assert len(some.candidates) == 1
        assert some.unchecked == {fs.NO_SITE: 1}

    def test_a_clock_that_was_not_synchronized_rules_a_frame_out(self, small: Night) -> None:
        assert screen(small, flags=FrameFlag.TIME_INVALID).rejected == {fs.TIME_INVALID: 1}

    def test_a_short_frame_and_another_readout_mode_stay_out(self, small: Night) -> None:
        assert screen(small, exposure_s=1.0).rejected == {fs.SHORT: 1}
        options = dataclasses.replace(OPTIONS, min_exposure_s=0.5)
        assert not screen(small, MIDNIGHT + 1_000_000_000, options=options, exposure_s=1.0).rejected
        path = tiny_file(small.layout, small.profile, MIDNIGHT + 2_000_000_000)
        other = fs.screen_headers(
            [path], mode="bin1", known=set(), site=sf.TEST_SITE, options=OPTIONS
        )
        assert other.rejected == {fs.WRONG_SIZE: 1}

    def test_a_frame_without_a_sensor_temperature_stays_out(self, small: Night) -> None:
        assert screen(small, temperature_c=None).rejected == {fs.NO_TEMPERATURE: 1}

    def test_a_frame_that_is_already_known_is_counted_apart(self, small: Night) -> None:
        path = tiny_file(small.layout, small.profile, MIDNIGHT)
        selection = fs.screen_headers(
            [path],
            mode="bin2",
            known={MIDNIGHT // 1_000_000 * 1_000_000},
            site=sf.TEST_SITE,
            options=OPTIONS,
        )
        assert selection.already == 1
        assert not selection.candidates
        assert not selection.rejected

    def test_a_file_that_is_not_fits_is_unreadable_and_a_hidden_file_is_not_a_frame(
        self, small: Night
    ) -> None:
        folder = small.layout.survey_dir
        (folder / "garbage.fits").write_bytes(b"this is not a FITS file")
        (folder / ".hidden.fits.tmp").write_bytes(b"temporary")
        (folder / "notes.txt").write_text("a note", encoding="utf-8")
        files = fs.list_frame_files(folder)
        assert [p.name for p in files] == ["garbage.fits"]
        selection = fs.screen_headers(
            files, mode="bin2", known=set(), site=sf.TEST_SITE, options=OPTIONS
        )
        assert selection.rejected == {fs.UNREADABLE: 1}

    def test_the_reasons_read_as_text_with_the_limits(self) -> None:
        text = fs.describe_reasons({fs.CLOUD: 2, fs.SKY_LEVEL: 1, fs.SUN: 3}, OPTIONS)
        assert text == (
            "2 frames with a cloud fraction of 0.1 or more, "
            "3 frames with the Sun above -18 degrees, "
            "1 frame with a sky level more than 10% from the median"
        )


class TestAnOpticalCenterOffTheFrame:
    def test_the_command_refuses_it_before_it_reads_a_frame(self, small: Night) -> None:
        options = dataclasses.replace(OPTIONS, center_xy=(-5000.0, -5000.0))
        with pytest.raises(fs.SkyFlatError, match=r"optical center \(-5000, -5000\) lies off"):
            build(small, options=options)
        edge = dataclasses.replace(OPTIONS, center_xy=(float(SHAPE[1]), 100.0))
        with pytest.raises(fs.SkyFlatError, match="lies off the frame of 512 x 352 pixels"):
            build(small, options=edge)


class TestNothingToUse:
    def test_a_folder_without_frames_says_so(self, small: Night) -> None:
        with pytest.raises(fs.SkyFlatError, match="holds no FITS file"):
            build(small)

    def test_a_frame_of_another_size_leaves_nothing_and_the_message_counts_it(
        self, small: Night
    ) -> None:
        pixels = np.full((100, 120), 4000, dtype=np.uint16)
        sf.write_frame(small.layout, pixels, MIDNIGHT, small.profile)
        with pytest.raises(fs.SkyFlatError) as caught:
            build(small)
        assert "no frame is usable (1 found: 1 frame of another readout mode or size)" in str(
            caught.value
        )

    def test_without_a_dark_library_no_frame_has_a_dark(self, small: Night) -> None:
        tiny_file(small.layout, small.profile, MIDNIGHT)
        with pytest.raises(fs.SkyFlatError, match="no dark set of the library covers"):
            build(small, library=None)

    def test_without_a_site_the_message_says_to_set_it_or_to_accept_unchecked(
        self, small: Night
    ) -> None:
        tiny_file(small.layout, small.profile, MIDNIGHT)
        with pytest.raises(fs.SkyFlatError, match=r"Set \[site\] in the configuration"):
            build(small, site=None)

    def test_files_from_before_the_cards_say_to_accept_unchecked(self, small: Night) -> None:
        tiny_file(small.layout, small.profile, MIDNIGHT, cloud_fraction=None, transparency=None)
        with pytest.raises(fs.SkyFlatError, match="pass --accept-unchecked"):
            build(small)


# --- The sky level --------------------------------------------------------------------------


class TestTheSkyLevel:
    def test_a_frame_with_a_brighter_sky_is_ruled_out_by_the_median(self, tmp_path: Path) -> None:
        night = make_night(tmp_path, fx.OWNER_LENS, frames=6)
        # a frame at twilight: the same time of night, with 40% more sky
        bright = night.sky.frame_pixels(MIDNIGHT, np.random.default_rng(5))
        brighter = np.rint((bright.astype(np.float32) - 534.0) * 1.4 + 534.0).astype(np.uint16)
        sf.write_frame(night.layout, brighter, MIDNIGHT, night.profile)
        selection = fs.screen_headers(
            fs.list_frame_files(night.folder),
            mode="bin2",
            known=set(),
            site=sf.TEST_SITE,
            options=OPTIONS,
        )
        assert len(selection.candidates) == 7
        darks = fs.DarkSource(night.library, doubling_c=6.0, tolerance_c=3.0)
        kept = fs.screen_sky_levels(
            selection, fs.Accumulator.empty(SHAPE, FACTOR, "bin2", {}), darks, SHAPE, OPTIONS
        )
        assert len(kept) == 6
        assert selection.rejected == {fs.SKY_LEVEL: 1}

    def test_the_median_covers_the_accumulator_too(self, tmp_path: Path) -> None:
        night = make_night(tmp_path, fx.OWNER_LENS, frames=4)
        selection = fs.screen_headers(
            fs.list_frame_files(night.folder),
            mode="bin2",
            known=set(),
            site=sf.TEST_SITE,
            options=OPTIONS,
        )
        darks = fs.DarkSource(night.library, doubling_c=6.0, tolerance_c=3.0)
        # an accumulator whose frames had a much brighter sky: the new frames lie far from it
        acc = fs.Accumulator.empty(SHAPE, FACTOR, "bin2", {})
        part = fs.Contribution(
            np.zeros_like(acc.sum), np.zeros_like(acc.sum), np.zeros_like(acc.count), 1.0, None, 0
        )
        for k in range(8):
            acc.add(part, fs.FrameRecord(k, 0.0, 50.0, float("nan"), float("nan")))
        kept = fs.screen_sky_levels(selection, acc, darks, SHAPE, OPTIONS)
        assert kept == []
        assert selection.rejected == {fs.SKY_LEVEL: 4}

    def test_the_tolerance_is_an_option(self, tmp_path: Path) -> None:
        night = make_night(tmp_path, fx.OWNER_LENS, frames=4)
        selection = fs.screen_headers(
            fs.list_frame_files(night.folder),
            mode="bin2",
            known=set(),
            site=sf.TEST_SITE,
            options=OPTIONS,
        )
        darks = fs.DarkSource(night.library, doubling_c=6.0, tolerance_c=3.0)
        tight = dataclasses.replace(OPTIONS, sky_tolerance=0.0001)
        empty = fs.Accumulator.empty(SHAPE, FACTOR, "bin2", {})
        kept = fs.screen_sky_levels(selection, empty, darks, SHAPE, tight)
        assert len(kept) <= 2  # the frames differ by noise, and a tolerance of 0.01% drops most
        assert selection.rejected[fs.SKY_LEVEL] >= 2


# --- The roll, the circle, and the masks ----------------------------------------------------


class TestTheRoll:
    def test_one_sidereal_day_turns_the_sky_by_360_degrees(self) -> None:
        t = iso_to_utc_ns("2026-12-09T20:00:00Z")
        sidereal_day_ns = round(86164.0905 * NS_PER_S)
        assert fs.roll_deg(t + sidereal_day_ns) == pytest.approx(fs.roll_deg(t), abs=0.02)
        assert (fs.roll_deg(t + 3600 * NS_PER_S) - fs.roll_deg(t)) % 360.0 == pytest.approx(
            15.041, abs=0.01
        )

    def test_the_coverage_is_360_degrees_minus_the_largest_gap(self) -> None:
        assert fs.roll_coverage([10.0, 50.0, 100.0]) == pytest.approx(90.0)
        assert fs.roll_coverage([350.0, 10.0, 30.0]) == pytest.approx(40.0)  # across 0
        assert fs.roll_coverage([0.0, 120.0, 240.0]) == pytest.approx(240.0)
        assert fs.roll_coverage([42.0]) == 0.0
        assert fs.roll_coverage([]) == 0.0

    def test_polaris_lies_about_two_thirds_of_a_degree_from_the_pole(self) -> None:
        separation = fs.polaris_separation_deg(iso_to_utc_ns("2026-10-02T00:00:00Z"))
        assert 0.62 < separation < 0.72

    def test_a_circle_is_fitted_to_points_on_it(self) -> None:
        angles = np.radians(np.linspace(0.0, 200.0, 12))
        x = 250.0 + 70.0 * np.cos(angles)
        y = 170.0 + 70.0 * np.sin(angles)
        cx, cy, radius = fs.fit_circle(x, y)
        assert (cx, cy, radius) == pytest.approx((250.0, 170.0, 70.0))


class TestTheMasks:
    def test_a_disk_has_the_area_of_a_circle(self) -> None:
        mask = np.zeros((200, 200), dtype=np.bool_)
        fs.paint_disks(mask, np.array([100.0]), np.array([100.0]), np.array([30.0]))
        assert int(mask.sum()) == pytest.approx(np.pi * 30.0**2, rel=0.02)
        assert mask[100, 100]
        assert not mask[100, 135]

    def test_a_disk_at_the_edge_is_cut_and_a_disk_off_the_frame_paints_nothing(self) -> None:
        mask = np.zeros((50, 50), dtype=np.bool_)
        fs.paint_disks(mask, np.array([0.0, 500.0]), np.array([0.0, 500.0]), np.array([10.0, 10.0]))
        assert 0 < int(mask.sum()) < np.pi * 10.0**2
        assert mask[0, 0]

    def test_the_edge_mask_and_the_growth(self) -> None:
        edge = fs.edge_mask((40, 60), 8)
        assert edge[:8].all()
        assert edge[-8:].all()
        assert not edge[20, 30]
        assert int((~edge).sum()) == (40 - 16) * (60 - 16)
        single = np.zeros((9, 9), dtype=np.bool_)
        single[4, 4] = True
        assert int(fs.grow(single, 1).sum()) == 9  # a square of 3 x 3
        assert int(fs.grow(single, 2).sum()) == 25

    def detections(self, flux: Sequence[float], saturated: Sequence[bool] = ()) -> object:
        from seeingmon.survey.detect import Detections, StarFlag

        n = len(flux)
        flags = np.zeros(n, dtype=np.uint16)
        for i, flag in enumerate(saturated):
            if flag:
                flags[i] = int(StarFlag.SATURATED)
        ones = np.ones(n)
        return Detections(
            shape=(100, 100),
            x=np.arange(n, dtype=np.float64) * 10.0,
            y=np.full(n, 50.0),
            flux=np.asarray(flux, dtype=np.float64),
            peak=ones,
            fwhm_px=np.full(n, 2.6),
            x_error_px=ones,
            y_error_px=ones,
            elongation=ones,
            trail_length_px=np.zeros(n),
            trail_angle_rad=np.zeros(n),
            flags=flags,
            snr=ones * 50,
            n_pixels=np.full(n, 40, dtype=np.int32),
            background_level=100.0,
            background_rms=3.0,
        )

    def test_a_star_masks_a_radius_that_grows_with_its_flux(self) -> None:
        radii = fs.star_radii(self.detections([1e3, 1e5, 1e7]), sky_px=100.0)  # type: ignore[arg-type]
        assert radii[0] < radii[1] < radii[2]
        # the faintest keeps about 3 sigma, and the wing adds the cube root of flux over sky
        assert radii[0] == pytest.approx(3.0 * 2.6 / 2.3548 + 10.0 ** (1 / 3) * 1.0, abs=0.2)
        assert radii[2] < 3.0 + fs.MAX_HALO_PX + 5.0

    def test_a_saturated_star_masks_at_least_the_radius_of_its_blob(self) -> None:
        radii = fs.star_radii(self.detections([1e5, 1e5], saturated=[False, True]), sky_px=100.0)  # type: ignore[arg-type]
        assert radii[1] == pytest.approx(
            radii[0] - 3.0 * 2.6 / 2.3548 + 2.0 * np.sqrt(40), abs=0.5
        ) or (radii[1] > radii[0])

    def test_polaris_is_the_brightest_star_when_it_stands_well_out(self) -> None:
        assert fs.find_polaris(self.detections([1e4, 8e6, 2e5])) == 1  # type: ignore[arg-type]
        assert fs.find_polaris(self.detections([1e5, 2e5, 1.5e5])) is None  # type: ignore[arg-type]
        assert fs.find_polaris(self.detections([5e5])) == 0  # type: ignore[arg-type]
        assert fs.find_polaris(self.detections([])) is None  # type: ignore[arg-type]


# --- The accumulator ------------------------------------------------------------------------


def sample_accumulator() -> fs.Accumulator:
    acc = fs.Accumulator.empty((16, 24), 4, "bin2", {"polaris_mask_px": 400.0, "edge_px": 8.0})
    rng = np.random.default_rng(2)
    for k in range(3):
        part = fs.Contribution(
            rng.uniform(10, 16, acc.sum.shape),
            rng.uniform(10, 18, acc.sum.shape),
            rng.integers(8, 17, acc.sum.shape).astype(np.int64),
            100.0 + k,
            None,
            5,
        )
        acc.add(part, fs.FrameRecord(1_000 * (3 - k), 30.0 * k, 3.3 + k, 10.0 + k, float("nan")))
    return acc


class TestTheAccumulatorFile:
    def test_a_saved_accumulator_loads_back_the_same(self, tmp_path: Path) -> None:
        acc = sample_accumulator()
        path = tmp_path / "acc.npz"
        acc.save(path)
        back = fs.Accumulator.load(path)
        np.testing.assert_array_equal(back.sum, acc.sum)
        np.testing.assert_array_equal(back.sumsq, acc.sumsq)
        np.testing.assert_array_equal(back.count, acc.count)
        assert (back.shape, back.bin_factor, back.mode) == ((16, 24), 4, "bin2")
        assert back.settings == {"polaris_mask_px": 400.0, "edge_px": 8.0}
        assert [r.t_utc_ns for r in back.frames] == [1_000, 2_000, 3_000]  # in time order
        assert back.times() == {1_000, 2_000, 3_000}
        assert back.frames[0].polaris_x == 12.0
        assert np.isnan(back.frames[0].polaris_y)

    def test_a_crash_while_saving_leaves_the_old_file_whole_and_no_temporary_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        acc = sample_accumulator()
        path = tmp_path / "acc.npz"
        acc.save(path)
        before = hashlib.sha256(path.read_bytes()).hexdigest()

        def crash(handle: object, **arrays: object) -> None:
            handle.write(b"half a file")  # type: ignore[attr-defined]
            raise OSError("the disk is full")

        monkeypatch.setattr(np, "savez_compressed", crash)
        acc.add(
            fs.Contribution(
                np.ones_like(acc.sum), np.ones_like(acc.sum), np.ones_like(acc.count), 1.0, None, 0
            ),
            fs.FrameRecord(9_000, 0.0, 1.0, float("nan"), float("nan")),
        )
        with pytest.raises(OSError, match="disk is full"):
            acc.save(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]
        assert len(fs.Accumulator.load(path).frames) == 3

    def test_a_run_that_cannot_save_the_accumulator_fails_with_a_message_and_keeps_the_old_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        night = make_night(tmp_path, fx.OWNER_LENS, frames=3)
        accumulator = tmp_path / "acc.npz"
        build(night, accumulator=accumulator)
        before = hashlib.sha256(accumulator.read_bytes()).hexdigest()
        more = sf.night_times(2, start_ns=sf.NIGHT_START_NS + 4 * 3600 * NS_PER_S)
        sf.write_night(night.layout, night.sky, night.profile, more)

        def full_disk(handle: object, **arrays: object) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(np, "savez_compressed", full_disk)
        with pytest.raises(fs.SkyFlatError) as caught:
            build(night, accumulator=accumulator)
        assert "cannot write the accumulator file: No space left on device" in str(caught.value)
        assert str(tmp_path) not in str(caught.value)
        assert hashlib.sha256(accumulator.read_bytes()).hexdigest() == before

    def test_a_file_that_is_not_an_accumulator_is_an_error_that_names_no_path(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.npz"
        bad.write_bytes(b"not a zip file")
        with pytest.raises(fs.SkyFlatError) as caught:
            fs.Accumulator.load(bad)
        assert str(tmp_path) not in str(caught.value)
        assert "cannot read the accumulator file" in str(caught.value)
        np.savez(tmp_path / "other.npz", values=np.arange(3))
        with pytest.raises(fs.SkyFlatError, match="cannot read the accumulator file"):
            fs.Accumulator.load(tmp_path / "other.npz")

    def test_an_unknown_format_is_refused(self, tmp_path: Path) -> None:
        acc = sample_accumulator()
        path = tmp_path / "acc.npz"
        acc.save(path)
        with np.load(path) as data:
            arrays = {name: data[name] for name in data.files}
        meta = bytes(arrays["meta"]).replace(b'"format": 1', b'"format": 9')
        arrays["meta"] = np.frombuffer(meta, dtype=np.uint8)
        np.savez(path, **arrays)
        with pytest.raises(fs.SkyFlatError, match="unknown format"):
            fs.Accumulator.load(path)

    def test_an_accumulator_of_another_binning_is_refused_by_a_run(self, tmp_path: Path) -> None:
        night = make_night(tmp_path, fx.OWNER_LENS, frames=2)
        path = tmp_path / "acc.npz"
        fs.Accumulator.empty(SHAPE, 4, "bin2", {}).save(path)
        with pytest.raises(fs.SkyFlatError) as caught:
            build(night, accumulator=path)
        assert "holds 4 x 4 binned sums" in str(caught.value)
        assert "this run needs 2 x 2 binned sums" in str(caught.value)
        assert "use another accumulator file" in str(caught.value)
        assert str(tmp_path) not in str(caught.value)


@pytest.fixture(scope="module")
def halves(
    tmp_path_factory: pytest.TempPathFactory, owner_night: Night, owner_run: fs.SkyResult
) -> dict[str, object]:
    root = tmp_path_factory.mktemp("halves")
    first_folder = DataLayout(root / "first")
    second_folder = DataLayout(root / "second")
    first_folder.create()
    second_folder.create()
    sources = sorted(owner_night.folder.rglob("*.fits"))
    paths: dict[str, list[Path]] = {"first": sources[:12], "second": sources[12:]}
    for name, layout in (("first", first_folder), ("second", second_folder)):
        for source in paths[name]:
            target = layout.survey_dir / source.relative_to(owner_night.folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    accumulator = root / "acc.npz"
    one = build(owner_night, folder=first_folder.survey_dir, accumulator=accumulator)
    two = build(owner_night, folder=second_folder.survey_dir, accumulator=accumulator)
    return {
        "root": root,
        "accumulator": accumulator,
        "first": one,
        "second": two,
        "second_layout": second_folder,
        "first_layout": first_folder,
        "whole": owner_run,
    }


class TestTwoRunsMakeOne:
    """Two runs on two folders add up to one run on both, and a repeated run adds nothing."""

    def test_the_second_run_adds_to_the_first(self, halves: dict[str, object]) -> None:
        one, two = halves["first"], halves["second"]
        assert isinstance(one, fs.SkyResult)
        assert isinstance(two, fs.SkyResult)
        assert (one.used_now, len(one.accumulator.frames)) == (12, 12)
        assert (two.used_now, len(two.accumulator.frames)) == (12, 24)

    def test_the_sums_and_the_flat_equal_those_of_one_run_on_both_folders(
        self, halves: dict[str, object]
    ) -> None:
        two, whole = halves["second"], halves["whole"]
        assert isinstance(two, fs.SkyResult)
        assert isinstance(whole, fs.SkyResult)
        np.testing.assert_allclose(two.accumulator.sum, whole.accumulator.sum, rtol=1e-12)
        np.testing.assert_allclose(two.accumulator.sumsq, whole.accumulator.sumsq, rtol=1e-12)
        np.testing.assert_array_equal(two.accumulator.count, whole.accumulator.count)
        np.testing.assert_allclose(two.flat, whole.flat, rtol=1e-6, atol=1e-6)
        assert [r.t_utc_ns for r in two.accumulator.frames] == [
            r.t_utc_ns for r in whole.accumulator.frames
        ]

    def test_a_repeated_run_adds_nothing_and_leaves_the_file_as_it_was(
        self, halves: dict[str, object], owner_night: Night
    ) -> None:
        accumulator = halves["accumulator"]
        assert isinstance(accumulator, Path)
        before = hashlib.sha256(accumulator.read_bytes()).hexdigest()
        again = build(owner_night, accumulator=accumulator)
        assert again.used_now == 0
        assert again.selection.already == 24
        assert len(again.accumulator.frames) == 24
        assert hashlib.sha256(accumulator.read_bytes()).hexdigest() == before
        text = "\n".join(fs.format_sky_report(again))
        assert (
            "24 found in the folder, 24 already in the accumulator, 0 rejected, 0 added now" in text
        )

    def test_a_frame_that_expires_from_the_folder_stays_in_the_sum(
        self, halves: dict[str, object], owner_night: Night
    ) -> None:
        first_layout = halves["first_layout"]
        accumulator = halves["accumulator"]
        whole = halves["whole"]
        assert isinstance(first_layout, DataLayout)
        assert isinstance(accumulator, Path)
        assert isinstance(whole, fs.SkyResult)
        for path in first_layout.survey_dir.rglob("*.fits"):
            path.unlink()  # retention deletes the frames of the first run
        later = build(owner_night, folder=first_layout.survey_dir, accumulator=accumulator)
        assert later.selection.found == 0
        assert len(later.accumulator.frames) == 24
        np.testing.assert_allclose(later.flat, whole.flat, rtol=1e-6, atol=1e-6)

    def test_a_run_with_no_folder_builds_from_the_accumulator_alone(
        self, halves: dict[str, object], owner_night: Night
    ) -> None:
        accumulator = halves["accumulator"]
        whole = halves["whole"]
        assert isinstance(accumulator, Path)
        assert isinstance(whole, fs.SkyResult)
        alone = build(owner_night, folder=None, accumulator=accumulator)
        assert alone.used_now == 0
        np.testing.assert_allclose(alone.flat, whole.flat, rtol=1e-6, atol=1e-6)

    def test_a_run_with_neither_a_folder_nor_an_accumulator_has_nothing(
        self, owner_night: Night
    ) -> None:
        with pytest.raises(fs.SkyFlatError, match="holds no FITS file"):
            build(owner_night, folder=None)

    def test_the_date_range_and_the_roll_cover_the_whole_accumulator(
        self, halves: dict[str, object]
    ) -> None:
        two = halves["second"]
        assert isinstance(two, fs.SkyResult)
        text = "\n".join(fs.format_sky_report(two))
        assert (
            "Accumulator: 24 frames from 2026-12-09 to 2026-12-10. Roll coverage: 173 degrees."
            in text
        )
        assert "12 added now" in text


# --- The ring around the orbit of Polaris ---------------------------------------------------


@pytest.fixture(scope="module")
def halo_night(tmp_path_factory: pytest.TempPathFactory) -> Night:
    # a bright, wide halo: 4 times the sky at the core, falling with the cube of the distance
    return make_night(
        tmp_path_factory.mktemp("halo"), fx.OWNER_LENS, frames=16, halo=sf.Halo(4.0, 8.0)
    )


class TestThePolarisRing:
    def test_a_mask_that_is_too_small_leaves_a_ring_and_the_command_says_so(
        self, halo_night: Night
    ) -> None:
        result = build(halo_night, options=dataclasses.replace(OPTIONS, polaris_mask_px=8.0))
        assert result.ring.bump is not None
        assert result.ring.bump > 0.01
        assert any("the mask around Polaris is too small" in w for w in result.warnings)
        text = "\n".join(fs.format_sky_report(result))
        assert "Warning: The mean sky has a bump of" in text
        assert "Raise --polaris-mask-px" in text

    def test_a_large_mask_leaves_no_ring_and_no_warning(self, halo_night: Night) -> None:
        result = build(halo_night, options=dataclasses.replace(OPTIONS, polaris_mask_px=49.0))
        assert result.ring.bump is not None
        assert result.ring.bump < 0.003
        assert not any("Polaris is too small" in w for w in result.warnings)

    def test_the_limit_is_an_option(self, halo_night: Night) -> None:
        options = dataclasses.replace(OPTIONS, polaris_mask_px=8.0, ring_limit=0.5)
        assert not any("too small" in w for w in build(halo_night, options=options).warnings)

    def test_without_enough_positions_the_orbit_comes_from_the_ephemeris(
        self, halo_night: Night
    ) -> None:
        result = build(halo_night, options=dataclasses.replace(OPTIONS, polaris_mask_px=49.0))
        acc = result.accumulator
        blind = dataclasses.replace(
            acc,
            frames=[
                dataclasses.replace(r, polaris_x=float("nan"), polaris_y=float("nan"))
                for r in acc.frames
            ],
        )
        scale = halo_night.profile.plate_scale_arcsec_per_px(halo_night.profile.survey_readout)
        ring = fs.ring_check(
            result.average, blind, optical_center=halo_night.sky.center, scale_arcsec_px=scale
        )
        assert not ring.measured
        assert ring.radius_px == pytest.approx(halo_night.sky.orbit_px, abs=2.0)
        assert ring.center_xy == halo_night.sky.center
        text = "\n".join(fs.format_sky_report(dataclasses.replace(result, ring=ring)))
        assert "(from the ephemeris)" in text


# --- The warnings ---------------------------------------------------------------------------


class TestTheWarnings:
    def test_few_frames_over_a_short_roll_warn_twice(self, tmp_path: Path) -> None:
        # 8 frames in 3.5 hours: 8 frames and 53 degrees of roll
        night = make_night(tmp_path, fx.OWNER_LENS, frames=8)
        result = build(night)
        messages = " ".join(result.warnings)
        assert "Only 8 frames went in, under 20" in messages
        assert "cover 53 degrees of roll, under 60" in messages

    def test_the_limits_are_options(self, tmp_path: Path) -> None:
        night = make_night(tmp_path, fx.OWNER_LENS, frames=8)
        options = dataclasses.replace(OPTIONS, min_frames=8, min_roll_deg=50.0)
        assert build(night, options=options).warnings == ()

    def test_a_scalar_dark_level_alone_leaves_the_pattern_of_the_dark_and_says_so(
        self, tmp_path: Path
    ) -> None:
        night = make_night(tmp_path, fx.OWNER_LENS, frames=4)
        # a library of the same bias and dark level whose set lies 12 C from the frames
        far = fx.make_library(tmp_path / "far", biases=((22.0, sf.BIAS_NATIVE),))
        result = build(night, library=far)
        assert any("No dark set of the library matched" in w for w in result.warnings)
        assert result.dark_note == "the level of the dark model alone for 4 frames"

    def test_unchecked_frames_are_listed_in_a_warning(self, tmp_path: Path) -> None:
        night = make_night(tmp_path, fx.OWNER_LENS, frames=4, cloud_fraction=None)
        options = dataclasses.replace(OPTIONS, accept_unchecked=True)
        result = build(night, options=options)
        assert any("went in unchecked (no_cloud_fraction)" in w for w in result.warnings)
        with pytest.raises(fs.SkyFlatError, match="pass --accept-unchecked"):
            build(night)  # without the option, the frames fail first

    def test_a_different_mask_in_a_later_run_is_noted(self, tmp_path: Path) -> None:
        night = make_night(tmp_path, fx.OWNER_LENS, frames=4)
        accumulator = tmp_path / "acc.npz"
        build(night, accumulator=accumulator)
        more = sf.night_times(2, start_ns=sf.NIGHT_START_NS + 2 * 3600 * NS_PER_S)
        sf.write_night(night.layout, night.sky, night.profile, more)
        options = dataclasses.replace(OPTIONS, polaris_mask_px=30.0)
        result = build(night, accumulator=accumulator, options=options)
        assert any(
            "mixes frames that hid Polaris with different radii" in w for w in result.warnings
        )


class TestTheFrameRecord:
    def test_a_frame_with_the_header_of_survey_frames_reads_its_roll_from_its_time(
        self, owner_run: fs.SkyResult
    ) -> None:
        first = owner_run.accumulator.frames[0]
        assert first.roll_deg == pytest.approx(fs.roll_deg(first.t_utc_ns))
        assert first.sky_rate == pytest.approx(
            sf.SKY_E_PER_PX / fx.E_PER_ADU / sf.EXPOSURE_S, rel=0.15
        )

    def test_the_meta_of_a_night_frame_has_what_the_tests_need(self, owner_night: Night) -> None:
        from seeingmon.survey.framefile import parse_frame_header, read_frame_header

        path = sorted(owner_night.folder.rglob("*.fits"))[0]
        meta = parse_frame_header(read_frame_header(path))
        assert isinstance(meta, FrameMeta)
        assert meta.cloud_fraction == 0.02
        assert meta.transparency == 0.99
        assert meta.temperature_c == sf.TEMPERATURE_C


class TestRiceFiles:
    def test_rice_compressed_files_give_the_same_sums_as_whole_files(self, tmp_path: Path) -> None:
        pytest.importorskip("astropy.io.fits")
        whole = make_night(tmp_path / "whole", fx.OWNER_LENS, frames=3, compress=False)
        rice = make_night(tmp_path / "rice", fx.OWNER_LENS, frames=3, compress=True)
        first, second = build(whole), build(rice)
        assert [p.stat().st_size for p in rice.folder.rglob("*.fits")] < [
            p.stat().st_size for p in whole.folder.rglob("*.fits")
        ]
        np.testing.assert_array_equal(first.accumulator.sum, second.accumulator.sum)
        np.testing.assert_array_equal(first.accumulator.count, second.accumulator.count)
