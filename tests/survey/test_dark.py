"""The dark library, the dark model, `dark_due`, and the check that a frame is dark."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.solvers import fitsio
from seeingmon.store.layout import DataLayout
from seeingmon.survey import dark
from seeingmon.survey.dark import DarkError, DarkLibrary, DarkSet

DAY_S = 86_400
NOW_NS = iso_to_utc_ns("2026-10-01T22:00:00Z")


def noisy_master(
    shape: tuple[int, int] = (96, 128), *, level: float = 150.0, sigma: float = 2.0, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.rint(rng.normal(level, sigma, shape)).astype(np.uint16)


def add_set(
    library: DarkLibrary,
    *,
    temperature_c: float,
    t_utc_ns: int = NOW_NS,
    master: np.ndarray | None = None,
    mode: str = "bin2",
    gain: int = 120,
    bias_dn: float = 120.0,
    dark_dn: float | None = None,
    exposure_s: float = 30.0,
) -> DarkSet:
    return library.add_set(
        noisy_master() if master is None else master,
        mode=mode,
        gain=gain,
        exposure_s=exposure_s,
        temperature_c=temperature_c,
        temperature_spread_c=0.4,
        t_utc_ns=t_utc_ns,
        n_frames=9,
        n_bias_frames=9,
        bias_dn=bias_dn,
        read_noise_dn=2.1,
        adc_bits=14,
        dark_dn=dark_dn,
    )


def synthetic_sets(
    temperatures: list[float], *, rate_ref: float, doubling: float, bias: float = 120.0
) -> list[DarkSet]:
    """Sets whose rates follow the model exactly, for the fit."""
    sets = []
    for index, temperature in enumerate(temperatures):
        rate = rate_ref * 2.0 ** ((temperature - 20.0) / doubling)
        sets.append(
            DarkSet(
                name=f"dark-20260101T00000{index}Z-bin2-g120.fits",
                mode="bin2",
                gain=120,
                exposure_s=30.0,
                temperature_c=temperature,
                temperature_spread_c=0.3,
                t_utc_ns=NOW_NS + index,
                n_frames=9,
                n_bias_frames=9,
                bias_dn=bias,
                dark_dn=bias + 30.0 * rate,
                read_noise_dn=2.1,
                adc_bits=14,
                width_px=64,
                height_px=48,
                n_hot_pixels=0,
            )
        )
    return sets


# --- The master and the hot pixels ---------------------------------------------------------


def test_the_master_is_the_per_pixel_median() -> None:
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 16000, (20, 30)).astype(np.uint16) for _ in range(5)]
    expected = np.median(np.stack(frames), axis=0)
    np.testing.assert_array_equal(dark.master_median(frames, strip_rows=7), expected)
    # An even count averages the two middle values and rounds to whole counts.
    even = frames[:4]
    np.testing.assert_array_equal(
        dark.master_median(even, strip_rows=7), np.rint(np.median(np.stack(even), axis=0))
    )
    with pytest.raises(ValueError, match="no frames"):
        dark.master_median([])
    with pytest.raises(ValueError, match="differ in size"):
        dark.master_median([frames[0], frames[1][:5]])


def test_hot_pixels_stand_above_their_neighbors_and_a_glow_does_not_count() -> None:
    rng = np.random.default_rng(2)
    shape = (150, 200)
    gradient = np.linspace(0.0, 30.0, shape[1])[None, :]  # a slow glow across the frame
    image = 150.0 + gradient + rng.normal(0.0, 1.8, shape)
    hot_x = np.array([10, 55, 120, 199, 0])
    hot_y = np.array([5, 90, 40, 149, 70])
    image[hot_y, hot_x] += np.array([60.0, 25.0, 200.0, 40.0, 30.0])
    master = np.rint(image).astype(np.uint16)
    columns, rows, excess = dark.find_hot_pixels(master)
    found = set(zip(columns.tolist(), rows.tolist(), strict=True))
    assert found == set(zip(hot_x.tolist(), hot_y.tolist(), strict=True))
    assert excess.dtype == np.float32
    assert excess.min() > 20.0


def test_a_quiet_master_has_no_hot_pixels() -> None:
    columns, rows, excess = dark.find_hot_pixels(noisy_master((200, 300), seed=3))
    assert columns.size == rows.size == excess.size == 0


# --- The library ---------------------------------------------------------------------------


def test_a_set_goes_into_one_fits_file_with_its_numbers_and_hot_pixels(tmp_path: Path) -> None:
    library = DarkLibrary(tmp_path / "calibration" / "darks")
    assert library.sets() == ()  # a folder that does not exist is an empty library
    master = noisy_master(seed=4)
    master[10, 20] += 80
    master[50, 100] += 120
    summary = add_set(library, temperature_c=18.4, master=master, dark_dn=151.25)
    assert summary.n_hot_pixels == 2
    assert summary.dark_dn == 151.25
    assert summary.rate_dn_per_s == pytest.approx((151.25 - 120.0) / 30.0)
    path = library.directory / summary.name
    assert summary.name.startswith("dark-20261001T220000Z-bin2-g120")
    # The file is plain FITS: the image first, and the hot pixels in the first extension.
    astropy_fits = pytest.importorskip("astropy.io.fits")
    with astropy_fits.open(path) as hdus:
        np.testing.assert_array_equal(hdus[0].data, master)
        assert hdus[0].header["SENSTEMP"] == pytest.approx(18.4)
        assert hdus[0].header["DATE-OBS"] == "2026-10-01T22:00:00Z"
        assert hdus[1].header["EXTNAME"] == "HOTPIX"
        assert sorted(zip(hdus[1].data["X"], hdus[1].data["Y"], strict=True)) == [
            (20, 10),
            (100, 50),
        ]
    (back,) = library.sets()
    assert back == summary
    np.testing.assert_array_equal(library.load_master(back), master)
    columns, rows, excess = library.hot_pixels(back)
    assert sorted(zip(columns.tolist(), rows.tolist(), strict=True)) == [(20, 10), (100, 50)]
    assert excess.min() > 60.0
    mask = library.hot_pixel_mask("bin2", 120, 18.0)
    assert mask is not None
    assert mask.shape == master.shape
    assert int(mask.sum()) == 2
    assert mask[10, 20]
    assert library.hot_pixel_mask("bin1", 120, 18.0) is None


def test_a_set_with_no_hot_pixels_still_reads_back(tmp_path: Path) -> None:
    library = DarkLibrary(tmp_path / "darks")
    summary = add_set(library, temperature_c=10.0)
    assert summary.n_hot_pixels == 0
    mask = library.hot_pixel_mask("bin2", 120, 10.0)
    assert mask is not None
    assert not mask.any()


def test_a_set_takes_the_clipped_mean_of_its_master_when_you_give_no_level(tmp_path: Path) -> None:
    library = DarkLibrary(tmp_path / "darks")
    master = noisy_master(level=147.3, sigma=2.5, seed=5)
    summary = add_set(library, temperature_c=10.0, master=master)
    assert summary.dark_dn == pytest.approx(147.3, abs=0.1)


def test_the_library_lists_the_sets_oldest_first_and_filters_them(tmp_path: Path) -> None:
    library = DarkLibrary(tmp_path / "darks")
    later = add_set(library, temperature_c=8.0, t_utc_ns=NOW_NS + 5 * DAY_S * NS_PER_S)
    earlier = add_set(library, temperature_c=14.0, t_utc_ns=NOW_NS)
    other_gain = add_set(library, temperature_c=14.0, t_utc_ns=NOW_NS + 1, gain=200)
    other_mode = add_set(library, temperature_c=14.0, t_utc_ns=NOW_NS + 2, mode="bin1")
    assert [item.name for item in library.sets()] == [
        earlier.name,
        other_gain.name,
        other_mode.name,
        later.name,
    ]
    assert library.sets_for("bin2", 120) == (earlier, later)
    assert library.sets_for("bin2") == (earlier, other_gain, later)
    nearest = library.nearest("bin2", 120, 9.0)
    assert nearest == later
    assert library.nearest("bin2", 120, 13.0) == earlier
    assert library.nearest("bin2", 999, 13.0) is None


def test_two_sets_with_the_same_second_get_distinct_names(tmp_path: Path) -> None:
    library = DarkLibrary(tmp_path / "darks")
    first = add_set(library, temperature_c=10.0)
    second = add_set(library, temperature_c=10.5)
    assert first.name != second.name
    assert len(library.sets()) == 2


def test_the_library_skips_files_that_are_not_sets(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    library = DarkLibrary(tmp_path / "darks")
    good = add_set(library, temperature_c=10.0)
    (library.directory / "notes.txt").write_text("not a set")
    (library.directory / "dark-20260101T000000Z-bin2-g1.fits").write_bytes(b"garbage")
    (library.directory / "dark-20260101T000001Z-bin2-g2.fits").write_bytes(
        fitsio.image_bytes(np.zeros((4, 4), np.uint16), header={"SMKIND": "flat"})
    )
    with caplog.at_level(logging.WARNING, logger="seeingmon.survey"):
        assert library.sets() == (good,)
    assert "skips dark-20260101T000000Z-bin2-g1.fits" in caplog.text
    assert "not a dark set" in caplog.text


def test_a_set_that_lacks_a_header_card_is_refused() -> None:
    header: fitsio.Header = {"SMKIND": "dark", "SMMODE": "bin2"}
    with pytest.raises(DarkError, match="lacks the header card"):
        DarkSet.from_header("x.fits", header)


def test_the_library_version_changes_with_the_sets(tmp_path: Path) -> None:
    library = DarkLibrary(tmp_path / "darks")
    assert library.version() == "darks-none"
    add_set(library, temperature_c=10.0)
    one = library.version()
    add_set(library, temperature_c=14.0, t_utc_ns=NOW_NS + 1)
    assert one != library.version() != "darks-none"
    assert library.version() == library.version()


def test_the_library_sits_in_the_calibration_folder_of_the_data_directory(tmp_path: Path) -> None:
    library = DarkLibrary.from_layout(DataLayout(tmp_path / "data"))
    assert library.directory == tmp_path / "data" / "calibration" / "darks"
    summary = add_set(library, temperature_c=10.0)
    assert (tmp_path / "data" / "calibration" / "darks" / summary.name).is_file()
    assert not list(library.directory.glob(".*.tmp"))  # the write was atomic


# --- The model -----------------------------------------------------------------------------


def test_the_fit_recovers_the_doubling_temperature_and_the_rate_at_20_c() -> None:
    sets = synthetic_sets([0.0, 6.0, 12.0, 19.0, 24.0], rate_ref=0.8, doubling=5.7)
    model = dark.fit_dark_model(sets)
    assert model.doubling_fitted
    assert model.doubling_c == pytest.approx(5.7, rel=1e-9)
    assert model.rate_ref_dn_per_s == pytest.approx(0.8, rel=1e-9)
    assert model.rms_log2 is not None
    assert model.rms_log2 < 1e-9
    assert model.rate_dn_per_s(8.3) == pytest.approx(0.8 * 2 ** ((8.3 - 20.0) / 5.7))
    assert model.level_dn(8.3, 30.0) == pytest.approx(120.0 + 30.0 * model.rate_dn_per_s(8.3))
    assert model.n_sets == 5
    assert model.version.startswith("darks-")


def test_the_fit_survives_noisy_rates() -> None:
    rng = np.random.default_rng(6)
    sets = synthetic_sets(list(np.linspace(-2.0, 25.0, 9)), rate_ref=0.7, doubling=6.2)
    noisy = [
        replace(
            item,
            dark_dn=item.bias_dn + 30.0 * item.rate_dn_per_s * float(np.exp(rng.normal(0.0, 0.03))),
        )
        for item in sets
    ]
    model = dark.fit_dark_model(noisy)
    assert model.doubling_c == pytest.approx(6.2, rel=0.05)
    assert model.rate_ref_dn_per_s == pytest.approx(0.7, rel=0.05)


def test_one_set_keeps_the_prior_doubling_temperature() -> None:
    (only,) = synthetic_sets([15.0], rate_ref=0.8, doubling=6.0)
    model = dark.fit_dark_model([only], prior_doubling_c=6.0)
    assert not model.doubling_fitted
    assert model.doubling_c == 6.0
    assert model.rms_log2 is None
    assert model.rate_dn_per_s(15.0) == pytest.approx(only.rate_dn_per_s)
    # With another prior the rate at 15 C still matches the set, and the slope follows the prior.
    other = dark.fit_dark_model([only], prior_doubling_c=5.0)
    assert other.rate_dn_per_s(15.0) == pytest.approx(only.rate_dn_per_s)
    assert other.rate_dn_per_s(20.0) == pytest.approx(only.rate_dn_per_s * 2.0)


def test_sets_within_a_few_degrees_keep_the_prior_too() -> None:
    sets = synthetic_sets([14.0, 15.0, 16.0], rate_ref=0.8, doubling=5.0)
    model = dark.fit_dark_model(sets, prior_doubling_c=6.0, min_span_c=4.0)
    assert not model.doubling_fitted
    assert model.doubling_c == 6.0


def test_a_fit_that_makes_no_physical_sense_keeps_the_prior() -> None:
    # The rate falls with the temperature: a bias error, not dark current.
    sets = synthetic_sets([0.0, 10.0, 20.0], rate_ref=0.8, doubling=-6.0)
    model = dark.fit_dark_model(sets)
    assert not model.doubling_fitted
    assert model.doubling_c == dark.DEFAULT_DOUBLING_C


def test_a_set_with_no_signal_above_the_bias_does_not_break_the_fit() -> None:
    sets = synthetic_sets([-10.0, 10.0, 22.0], rate_ref=0.8, doubling=6.0)
    sets[0] = replace(sets[0], dark_dn=sets[0].bias_dn)
    model = dark.fit_dark_model(sets)
    assert model.doubling_c == pytest.approx(6.0, rel=1e-6)  # the two warm sets decide
    assert model.rate_dn_per_s(-10.0) > 0.0


def test_the_bias_interpolates_between_the_sets_and_holds_beyond_them() -> None:
    low, high = synthetic_sets([5.0, 15.0], rate_ref=0.5, doubling=6.0)
    low = replace(low, bias_dn=118.0, dark_dn=118.0 + 30.0 * low.rate_dn_per_s)
    model = dark.fit_dark_model([low, high])
    assert model.bias_dn(5.0) == pytest.approx(118.0)
    assert model.bias_dn(10.0) == pytest.approx(119.0)
    assert model.bias_dn(15.0) == pytest.approx(120.0)
    assert model.bias_dn(-20.0) == pytest.approx(118.0)
    assert model.bias_dn(40.0) == pytest.approx(120.0)


def test_the_fit_needs_sets_of_one_mode_and_gain() -> None:
    with pytest.raises(ValueError, match="no sets"):
        dark.fit_dark_model([])
    mixed = synthetic_sets([5.0, 15.0], rate_ref=0.5, doubling=6.0)
    other = DarkSet(
        name="x",
        mode="bin1",
        gain=120,
        exposure_s=30.0,
        temperature_c=10.0,
        temperature_spread_c=0.0,
        t_utc_ns=NOW_NS,
        n_frames=3,
        n_bias_frames=3,
        bias_dn=100.0,
        dark_dn=101.0,
        read_noise_dn=2.0,
        adc_bits=12,
        width_px=8,
        height_px=8,
        n_hot_pixels=0,
    )
    with pytest.raises(ValueError, match="one readout mode and gain"):
        dark.fit_dark_model([*mixed, other])


def test_the_library_fits_the_model_of_one_mode_and_gain_and_ignores_old_sets(
    tmp_path: Path,
) -> None:
    library = DarkLibrary(tmp_path / "darks")
    assert library.model("bin2", 120) is None
    old = NOW_NS - 400 * DAY_S * NS_PER_S
    add_set(library, temperature_c=5.0, t_utc_ns=old, dark_dn=121.0)
    add_set(library, temperature_c=20.0, t_utc_ns=NOW_NS, dark_dn=130.0)
    both = library.model("bin2", 120)
    assert both is not None
    assert both.n_sets == 2
    recent = library.model("bin2", 120, now_ns=NOW_NS, max_age_s=300 * DAY_S)
    assert recent is not None
    assert recent.n_sets == 1
    # When every set is old, the model uses them all rather than none.
    stale = library.model(
        "bin2", 120, now_ns=NOW_NS + 900 * DAY_S * NS_PER_S, max_age_s=100 * DAY_S
    )
    assert stale is not None
    assert stale.n_sets == 2


# --- dark_due ------------------------------------------------------------------------------


def test_an_empty_library_is_due(tmp_path: Path) -> None:
    library = DarkLibrary(tmp_path / "darks")
    status = dark.dark_status(library, 15.0, NOW_NS)
    assert status.due
    assert "no dark set" in status.reason
    assert dark.dark_due(library, 15.0, NOW_NS)


def test_a_recent_set_within_the_tolerance_covers_the_temperature(tmp_path: Path) -> None:
    library = DarkLibrary(tmp_path / "darks")
    add_set(library, temperature_c=15.0, t_utc_ns=NOW_NS - 10 * DAY_S * NS_PER_S)
    assert not dark.dark_due(library, 15.0, NOW_NS)
    assert not dark.dark_due(library, 17.9, NOW_NS)  # within the default 3 C
    assert not dark.dark_due(library, 12.0, NOW_NS)
    assert dark.dark_due(library, 18.5, NOW_NS)
    assert dark.dark_due(library, 5.0, NOW_NS)
    status = dark.dark_status(library, 5.0, NOW_NS)
    assert status.gap_c == pytest.approx(10.0)
    assert "10.0 C away" in status.reason
    assert status.newest_age_days == pytest.approx(10.0)


def test_the_tolerance_is_configurable(tmp_path: Path) -> None:
    library = DarkLibrary(tmp_path / "darks")
    add_set(library, temperature_c=15.0)
    assert dark.dark_due(library, 17.0, NOW_NS, tolerance_c=1.5)
    assert not dark.dark_due(library, 17.0, NOW_NS, tolerance_c=2.5)
    assert not dark.dark_due(library, 20.0, NOW_NS, tolerance_c=5.0)


def test_a_set_older_than_six_months_does_not_count(tmp_path: Path) -> None:
    library = DarkLibrary(tmp_path / "darks")
    add_set(library, temperature_c=15.0, t_utc_ns=NOW_NS - 200 * DAY_S * NS_PER_S)
    status = dark.dark_status(library, 15.0, NOW_NS)
    assert status.due
    assert "200 days old" in status.reason
    # The limit is 183 days by default, and you can move it.
    assert not dark.dark_due(library, 15.0, NOW_NS, max_age_days=365.0)
    add_set(library, temperature_c=15.0, t_utc_ns=NOW_NS - 182 * DAY_S * NS_PER_S)
    assert not dark.dark_due(library, 15.0, NOW_NS)


def test_an_old_set_at_the_right_temperature_does_not_hide_a_missing_recent_one(
    tmp_path: Path,
) -> None:
    library = DarkLibrary(tmp_path / "darks")
    add_set(library, temperature_c=15.0, t_utc_ns=NOW_NS - 200 * DAY_S * NS_PER_S)
    add_set(library, temperature_c=2.0, t_utc_ns=NOW_NS - 5 * DAY_S * NS_PER_S)
    assert dark.dark_due(library, 15.0, NOW_NS)  # only the 2 C set is recent
    assert not dark.dark_due(library, 2.5, NOW_NS)


def test_dark_due_looks_at_one_mode_and_gain_when_you_ask(tmp_path: Path) -> None:
    library = DarkLibrary(tmp_path / "darks")
    add_set(library, temperature_c=15.0, mode="bin1")
    assert not dark.dark_due(library, 15.0, NOW_NS)
    assert dark.dark_due(library, 15.0, NOW_NS, mode="bin2")
    assert not dark.dark_due(library, 15.0, NOW_NS, mode="bin1", gain=120)
    assert dark.dark_due(library, 15.0, NOW_NS, mode="bin1", gain=200)


def test_a_camera_without_a_temperature_is_judged_by_age_alone(tmp_path: Path) -> None:
    library = DarkLibrary(tmp_path / "darks")
    add_set(library, temperature_c=15.0, t_utc_ns=NOW_NS - 20 * DAY_S * NS_PER_S)
    assert not dark.dark_due(library, None, NOW_NS)
    assert dark.dark_due(library, None, NOW_NS + 400 * DAY_S * NS_PER_S)


# --- Is the frame dark? --------------------------------------------------------------------

E_PER_ADU = 0.88
READ_NOISE_E = 1.85


def dark_frame(
    *,
    level: float = 120.7,
    exposure_s: float = 1.0,
    sky_e: float = 0.0,
    seed: int = 7,
    shape: tuple[int, int] = (300, 400),
) -> np.ndarray:
    """A bin2-like frame: bias 120, dark current and sky in electrons, Poisson and read noise."""
    rng = np.random.default_rng(seed)
    mean_e = max(level - 120.0, 0.0) * E_PER_ADU + sky_e
    electrons = rng.poisson(mean_e, shape) + rng.normal(0.0, READ_NOISE_E, shape)
    return np.rint(120.0 + electrons / E_PER_ADU).astype(np.uint16)


def check(data: np.ndarray, *, exposure_s: float = 1.0, rate: float = 0.7) -> dark.DarkCheck:
    return dark.check_dark_frame(
        data,
        bias_dn=120.0,
        exposure_s=exposure_s,
        e_per_adu=E_PER_ADU,
        read_noise_e=READ_NOISE_E,
        expected_rate_e_per_s=rate,
    )


def test_a_covered_frame_is_dark() -> None:
    result = check(dark_frame(level=120.8))
    assert result.ok
    assert result.reason == ""
    assert result.level_dn == pytest.approx(120.8, abs=0.1)
    assert result.excess_rate_e_per_s == pytest.approx(0.7, abs=0.15)


def test_a_frame_with_light_on_it_is_not_dark() -> None:
    lit = dark_frame(level=120.8, sky_e=4.2)  # a dark sky: 4 e- per second per pixel
    result = check(lit)
    assert not result.ok
    assert "above the bias" in result.reason


def test_a_frame_with_stars_is_not_dark_even_at_a_dark_level() -> None:
    data = dark_frame(level=120.8, seed=8)
    rng = np.random.default_rng(9)
    rows = rng.integers(0, 300, 400)
    columns = rng.integers(0, 400, 400)
    data[rows, columns] += 400  # one pixel in 300 stands far above the median
    result = check(data)
    assert not result.ok
    assert "stand far above the median" in result.reason


def test_a_frame_with_structure_has_too_much_noise() -> None:
    rng = np.random.default_rng(10)
    data = dark_frame(level=120.8, seed=11).astype(np.float64)
    data += rng.normal(0.0, 6.0, data.shape)  # a pattern that the dark current cannot explain
    result = check(np.clip(np.rint(data), 0, 65535).astype(np.uint16))
    assert not result.ok
    assert "the noise is" in result.reason


def test_a_handful_of_hot_pixels_does_not_spoil_a_dark_frame() -> None:
    data = dark_frame(level=120.8, seed=12)
    rng = np.random.default_rng(13)
    data[rng.integers(0, 300, 40), rng.integers(0, 400, 40)] += 500
    assert check(data).ok


def test_the_rate_limit_scales_with_the_expected_dark_current() -> None:
    # 3 e- per second for 30 s is 90 e-, a lot for a cold camera and ordinary for a warm one.
    level = 120.0 + 90.0 / E_PER_ADU
    flat = np.full((60, 80), round(level), dtype=np.uint16)
    cold = dark.check_dark_frame(
        flat,
        bias_dn=120.0,
        exposure_s=30.0,
        e_per_adu=E_PER_ADU,
        read_noise_e=READ_NOISE_E,
        expected_rate_e_per_s=0.05,
    )
    warm = dark.check_dark_frame(
        flat,
        bias_dn=120.0,
        exposure_s=30.0,
        e_per_adu=E_PER_ADU,
        read_noise_e=READ_NOISE_E,
        expected_rate_e_per_s=1.2,
    )
    assert not cold.ok
    assert warm.ok
