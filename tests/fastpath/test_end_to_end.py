"""The whole chain on the `sim` driver: frames, the kernel, windows, and the estimator.

The simulator renders stars through frozen-flow von Karman turbulence of a known `r0`, with photon
and read noise, scintillation, and an uncooled sensor. Its `truth` object holds the injected `r0`,
the true star position of every frame, and the scintillation factor of every frame.

**Sampling error.** A window of 6 s at 112 frames per second holds about 700 frames, and the tilt
of a 50 mm aperture decorrelates in about 25 ms, so a window has about 240 independent samples.
That gives 9% in the variance and 5% in `r0` per window. Three windows give 3%, so the 10%
tolerance is three standard errors. The seed is fixed, so the result is the same on every run,
and in practice it lands within 2%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("seeingmon.drivers.sim", reason="the simulator needs SciPy (the fast extra)")

from seeingmon.analysis import FastContext
from seeingmon.clock import VirtualClock
from seeingmon.drivers.sim import FrameTruth, SimFaults, SimTruth, sim_camera
from seeingmon.fastpath import FastPathAnalyzer, FastPathConfig, create_fast_analyzer, models
from seeingmon.frames import PixelFormat, Roi, StreamConfig
from seeingmon.profile import Profile
from seeingmon.records import SeeingWindowRecord

FloatArray = npt.NDArray[np.float64]

ZENITH_DEG = 30.0


@dataclass(slots=True)
class SimRun:
    windows: list[SeeingWindowRecord]
    rows: npt.NDArray[np.void]
    truth: SimTruth
    frames: list[FrameTruth]
    analyzer: FastPathAnalyzer


def run_sim(
    profile: Profile,
    *,
    r0_m: float,
    seed: int,
    windows: int,
    window_s: float,
    roi_size: int = 64,
    faults: SimFaults | None = None,
    min_frames: int = 0,
    exposure_us: int = 2000,
) -> SimRun:
    """Drive the simulated camera and the analyzer until `windows` windows have closed."""
    clock = VirtualClock()
    driver = sim_camera(
        clock,
        r0_m=r0_m,
        wind_speed_m_s=10.0,
        wind_direction_deg=45.0,
        outer_scale_m=20.0,
        seed=seed,
        psf_mode="gaussian",
        zenith_angle_deg=ZENITH_DEG,
        faults=SimFaults() if faults is None else faults,
    )
    driver.open()
    stars = driver.truth.star_positions(clock.utc_ns(), "bin1")
    brightest = int(np.argmin(stars.mag))
    half = roi_size // 2
    roi = Roi(int(stars.x[brightest]) - half, int(stars.y[brightest]) - half, roi_size, roi_size)
    stream = driver.configure(
        StreamConfig("bin1", exposure_us, 0, roi=roi, pixel_format=PixelFormat.RAW16)
    )
    driver.start()
    analyzer = create_fast_analyzer(
        profile, FastPathConfig(window_s=window_s, min_window_s=min(3.0, window_s)), "sim"
    )
    analyzer.begin_stream(stream)
    analyzer.set_context(FastContext(zenith_angle_deg=ZENITH_DEG))
    closed: list[SeeingWindowRecord] = []
    pushed = 0
    while len(closed) < windows or pushed < min_frames:
        closed += analyzer.push(driver.read_frame(1.0)).windows
        pushed += 1
    rows = analyzer.drain_metrics()
    assert rows is not None
    return SimRun(closed, rows, driver.truth, driver.truth.frames[: len(rows)], analyzer)


@pytest.fixture(scope="module")
def run(profile: Profile) -> SimRun:
    """Three windows of 6 s for r0 = 10 cm at a zenith angle of 30 degrees."""
    return run_sim(profile, r0_m=0.10, seed=5, windows=3, window_s=6.0)


def mean(values: list[float | None]) -> float:
    assert None not in values
    return float(np.mean([v for v in values if v is not None]))


class TestSeeing:
    def test_recovers_r0_at_the_zenith_within_ten_percent(self, run: SimRun) -> None:
        """The simulated line of sight is 30 degrees off the zenith, so r0 along it is 8% smaller.

        The analyzer converts the measured value to the zenith with the context's zenith angle.
        """
        truth = run.truth.r0_zenith_m() * 100.0
        r0 = mean([w.r0_cm for w in run.windows])
        assert r0 == pytest.approx(truth, rel=0.10)
        assert r0 == pytest.approx(truth, rel=0.06)  # in practice the error is about 2%
        assert [w.zenith_angle_deg for w in run.windows] == [ZENITH_DEG] * 3

    def test_the_seeing_is_the_kolmogorov_value_at_the_zenith(self, run: SimRun) -> None:
        truth = run.truth.kolmogorov_seeing_fwhm_arcsec(at_zenith=True)
        assert mean([w.seeing_fwhm_arcsec for w in run.windows]) == pytest.approx(truth, rel=0.08)

    def test_the_structure_function_agrees_with_the_truth_to_ten_percent(self, run: SimRun) -> None:
        """The cross-check, with the full-size screens of the simulator: it agrees to 3%."""
        truth = run.truth.r0_zenith_m() * 100.0
        assert mean([w.r0_structure_cm for w in run.windows]) == pytest.approx(truth, rel=0.10)
        r0 = mean([w.r0_cm for w in run.windows])
        sf = mean([w.r0_structure_cm for w in run.windows])
        assert sf == pytest.approx(r0, rel=0.06)

    def test_the_image_motion_is_the_injected_motion_with_the_noise_taken_out(
        self, run: SimRun
    ) -> None:
        """The record's RMS is detrended, with the noise subtracted. It reads the injected RMS of
        the instantaneous tilt (0.43 arcsec per axis at r0 = 9.2 cm along the line of sight) minus
        the loss from the exposure (1%) and the detrend (0.5%), within the sampling error."""
        rms = mean(
            [w.image_motion_rms_x_arcsec for w in run.windows]
            + [w.image_motion_rms_y_arcsec for w in run.windows]
        )
        expected = run.truth.image_motion_rms_arcsec()
        assert rms == pytest.approx(expected, rel=0.10)


class TestKernel:
    def test_the_centroids_follow_the_true_star_to_the_noise_level(self, run: SimRun) -> None:
        """The residual of the centroid against the true position, which includes the drift and
        the tilt, is 0.018 px. The modeled centroid noise is 0.017 px."""
        truth_x = np.array([f.star_x_px for f in run.frames])
        truth_y = np.array([f.star_y_px for f in run.frames])
        dx = run.rows["cx_px"].astype(np.float64) - truth_x
        dy = run.rows["cy_px"].astype(np.float64) - truth_y
        assert np.isfinite(dx).all()
        assert np.isfinite(dy).all()
        assert float(np.std(dx)) < 0.03
        assert float(np.std(dy)) < 0.03
        assert abs(float(np.mean(dx))) < 0.005
        assert abs(float(np.mean(dy))) < 0.005
        noise = mean([w.centroid_noise_px for w in run.windows])
        assert noise == pytest.approx(float(np.std(dx)), rel=0.25)

    def test_the_flux_follows_the_scintillation_of_the_simulator(self, run: SimRun) -> None:
        """The measured flux of every frame tracks the injected flux, which varies by 40%."""
        truth = np.array([f.flux_e for f in run.frames])
        measured = run.rows["flux_e"].astype(np.float64)
        assert float(np.corrcoef(truth, measured)[0, 1]) > 0.99
        assert float(np.mean(measured)) == pytest.approx(float(np.mean(truth)), rel=0.05)

    def test_the_rows_cover_every_frame_in_order(self, run: SimRun) -> None:
        assert run.rows["seq"].tolist() == list(range(len(run.rows)))
        assert len(run.rows) == len(run.frames)
        assert (np.diff(run.rows["t_utc_ns"]) > 0).all()


class TestExposure:
    def test_each_window_is_corrected_for_the_exposure_of_its_stream(
        self, profile: Profile, run: SimRun
    ) -> None:
        """The adaptive exposure of the scheduler changes the exposure between windows.

        The analyzer derives the exposure correction from the stream of each window, so a window
        of 1 ms gets the correction of 1 ms, and its `r0` agrees with the truth as at 2 ms.
        """
        short = run_sim(profile, r0_m=0.10, seed=5, windows=2, window_s=6.0, exposure_us=1000)
        spectrum = models.tilt_spectrum(profile.optics.aperture_mm * 1e-3, 20.0, 10.0)
        for window, exposure_s in [(w, 0.001) for w in short.windows] + [
            (w, 0.002) for w in run.windows
        ]:
            assert window.exposure_us == round(exposure_s * 1e6)
            assert window.exposure_correction_factor == pytest.approx(
                1.0 / spectrum.exposure_variance_ratio(exposure_s), rel=1e-9
            )
        truth = short.truth.r0_zenith_m() * 100.0
        assert mean([w.r0_cm for w in short.windows]) == pytest.approx(truth, rel=0.10)


class TestScintillation:
    def test_the_index_is_the_injected_one(self, run: SimRun) -> None:
        """The simulator's relative flux variance at 2 ms is 0.17 (an RMS of 0.41)."""
        factors = np.array([f.scintillation_factor for f in run.frames])
        realized = float(np.var(factors) / np.mean(factors) ** 2)
        assert mean([w.scintillation_index for w in run.windows]) == pytest.approx(
            realized, rel=0.12
        )


class TestRecords:
    def test_every_window_validates_and_describes_the_stream(self, run: SimRun) -> None:
        for window in run.windows:
            assert type(window).from_row(window.to_row()) == window
            assert window.readout_mode == "bin1"
            assert window.exposure_us == 2000
            assert window.n_dropped == 0
            assert window.valid_fraction == 1.0
            assert window.flags == []
            assert window.duration_s == pytest.approx(6.0, abs=0.02)
            assert window.frame_rate_hz == pytest.approx(102.3, rel=0.01)  # 9.8 ms per frame
            # Scintillation lifts Polaris to the saturation level in 0.3% of the frames.
            assert window.saturated_fraction is not None
            assert window.saturated_fraction < 0.01
            assert window.sensor_temperature_c == pytest.approx(19.0)
            assert window.exposure_correction_factor == pytest.approx(1.02, abs=0.005)
            assert window.outer_scale_correction_factor == pytest.approx(1.2609, abs=0.001)
            assert window.assumed_wind_ms == 10.0
            assert window.outer_scale_m == 20.0
            assert window.motion_psd_freq_hz is not None
            assert len(window.motion_psd_freq_hz) == len(window.motion_psd_x_arcsec2_per_hz or [])
            assert window.vibration_lines_hz == []

    def test_the_spectrum_has_the_turbulence_slope_and_the_right_units(self, run: SimRun) -> None:
        """The spectrum of a one-axis tilt is flat below 1 Hz and falls toward 0.05 arcsec^2/Hz...

        Its integral over frequency is the variance of the detrended series, 0.18 arcsec^2.
        """
        window = run.windows[1]
        assert window.motion_psd_freq_hz is not None
        assert window.motion_psd_x_arcsec2_per_hz is not None
        freq = np.array(window.motion_psd_freq_hz)
        psd = np.array(window.motion_psd_x_arcsec2_per_hz)
        assert (psd > 0).all()
        assert (np.diff(freq) > 0).all()
        assert float(np.max(freq)) <= 56.1  # the Nyquist frequency at 112 frames per second
        # The frequency resolution of a 2 s segment is 0.5 Hz, so the bins reach the variance
        # only roughly: the sum of the bins over their widths is the variance within a third.
        edges = np.sqrt(freq[1:] * freq[:-1])
        widths = np.diff(
            np.concatenate([[freq[0] ** 2 / edges[0]], edges, [freq[-1] ** 2 / edges[-1]]])
        )
        assert float(np.sum(psd * widths)) == pytest.approx(0.18, rel=0.45)


class TestDrops:
    def test_scripted_drops_are_counted_in_the_windows_that_contain_them(
        self, profile: Profile
    ) -> None:
        """Drops of 3, 3, and 30 frames in windows of 2 s: 36 frames, and `degraded` for the last
        window, whose drops are 12% of its frames."""
        faults = SimFaults(scripted_drops=((100, 3), (330, 3), (560, 30)))
        run = run_sim(
            profile,
            r0_m=0.10,
            seed=3,
            windows=3,
            window_s=2.0,
            roi_size=48,
            faults=faults,
        )
        windows = run.windows[:3]
        assert [w.n_dropped for w in windows] == [3, 3, 30]
        assert [("degraded" in w.flags) for w in windows] == [False, False, True]
        assert windows[2].valid_fraction == pytest.approx(
            windows[2].n_frames / (windows[2].n_frames + 30)
        )
        rows = run.rows
        assert int(rows["dropped_before"].sum()) == 36
        # The truth of the simulator agrees: the frames after the drops report them.
        assert sum(f.dropped_before for f in run.frames) == 36


@pytest.mark.slow
class TestOtherSeeingValues:
    """The same chain for r0 of 5 cm and 15 cm, with windows of 10 s (slow: about 40 s each)."""

    @pytest.mark.parametrize("r0_m", [0.05, 0.15])
    def test_recovers_r0_within_ten_percent(self, profile: Profile, r0_m: float) -> None:
        run = run_sim(profile, r0_m=r0_m, seed=11, windows=3, window_s=10.0)
        assert mean([w.r0_cm for w in run.windows]) == pytest.approx(r0_m * 100.0, rel=0.10)
        assert mean([w.r0_structure_cm for w in run.windows]) == pytest.approx(
            r0_m * 100.0, rel=0.12
        )
