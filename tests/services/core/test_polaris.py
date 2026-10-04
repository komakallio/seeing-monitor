"""The frame of the live video of Polaris: the stretch, the width, the PNG, and the state."""

from __future__ import annotations

import io
import json
import math
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from seeingmon.analysis.base import NO_STAR, StarState
from seeingmon.clock import NS_PER_S
from seeingmon.frames import Roi
from seeingmon.services.core.polaris import (
    Autostretch,
    FrameSlot,
    PolarisRenderer,
    encode_png,
    star_width_px,
)
from seeingmon.services.core.settings import PolarisSettings
from seeingmon.services.web.contract import MAX_STATE_BYTES, PNG_MAGIC
from tests.services.web.helpers import live_seeing_view

T0 = 1_790_000_000 * NS_PER_S
PERIOD_NS = 12_180_000  # a frame every 12.18 ms: the camera runs at 82 frames per second
BACKGROUND = 480.0
SIGMA_PX = 1.1


def star_frame(
    *,
    amplitude: float = 18_000.0,
    x: float = 64.0,
    y: float = 64.0,
    sigma: float = SIGMA_PX,
    noise: float = 24.0,
    seed: int = 1,
    size: int = 128,
) -> npt.NDArray[np.uint16]:
    """A Gaussian star on a flat sky with noise, in the counts of a 12-bit ADC in a 16-bit frame."""
    yy, xx = np.mgrid[0:size, 0:size]
    star = amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
    rng = np.random.default_rng(seed)
    data = BACKGROUND + star + rng.normal(0.0, noise, (size, size))
    counts: npt.NDArray[np.uint16] = np.clip(np.rint(data), 0, 65_535).astype(np.uint16)
    return counts


def slot_of(
    data: npt.NDArray[np.uint16] | npt.NDArray[np.uint8],
    index: int = 0,
    *,
    star: StarState | None = None,
    stream_id: int = 3,
    roi: Roi | None = None,
    mode: str = "bin1",
    count_step: int = 5,
    exposure_us: int = 2000,
    gain: int = 0,
) -> FrameSlot:
    """A kept frame. Slots are `count_step` camera frames (about 61 ms) apart."""
    chosen_roi = roi or Roi(2008, 1347, data.shape[1], data.shape[0])
    return FrameSlot(
        data=data,
        stream_id=stream_id,
        t_utc_ns=T0 + index * count_step * PERIOD_NS,
        mode=mode,
        exposure_us=exposure_us,
        gain=gain,
        adc_bits=12,
        roi=chosen_roi,
        star=star
        or StarState(
            found=True,
            x_px=chosen_roi.x + 64.0,
            y_px=chosen_roi.y + 64.0,
            peak_fraction=0.28,
            edge_distance_px=60.0,
        ),
        count=index * count_step,
    )


# --- The stretch -----------------------------------------------------------------------------


@pytest.mark.parametrize("level", [0, 1, 480, 40_000, 65_535])
def test_a_constant_frame_gives_a_black_image_without_an_error(level: int) -> None:
    stretch = Autostretch()
    result = stretch.apply(np.full((128, 128), level, dtype=np.uint16), T0)
    assert result.image.dtype == np.uint8
    assert result.image.shape == (128, 128)
    assert int(result.image.max()) == 0
    assert result.black_dn == level
    assert result.white_dn > result.black_dn


def test_an_eight_bit_frame_stretches_too() -> None:
    data = np.clip(star_frame(amplitude=200.0, noise=0.5) // 16, 0, 255).astype(np.uint8)
    assert data.dtype == np.uint8
    result = Autostretch().apply(data, T0)
    assert result.image.shape == (128, 128)
    assert int(result.image[64, 64]) > 200


def test_black_is_the_median_and_the_star_stays_below_white() -> None:
    data = star_frame()
    result = Autostretch().apply(data, T0)
    assert result.black_dn == float(np.median(data))
    peak_above = float(data.max()) - result.black_dn
    assert result.white_dn == pytest.approx(result.black_dn + 1.5 * peak_above)
    assert 200 <= int(result.image.max()) < 250  # the headroom keeps the core below white
    assert int(result.image[0, 0]) <= 5  # the sky is nearly black


def test_a_saturated_star_reaches_white_and_a_saturated_frame_does_not_fail() -> None:
    data = star_frame(amplitude=80_000.0)  # the counts clip at 65,535
    assert int(data.max()) == 65_535
    stretch = Autostretch()
    result = stretch.apply(data, T0)
    assert int(result.image[64, 64]) >= 230
    full = np.full((128, 128), 65_535, dtype=np.uint16)
    full[:, :64] = 100
    assert Autostretch().apply(full, T0).image.shape == (128, 128)


@pytest.mark.parametrize("gain", [5.0, 30.0, 200.0, 1000.0])
def test_the_table_follows_the_exact_asinh_curve_within_one_gray_level(gain: float) -> None:
    stretch = Autostretch(asinh_gain=gain, floor_sigmas=0.1)
    data = np.tile(np.arange(0, 20_000, 40, dtype=np.uint16), (4, 1))  # a ramp of 500 counts
    result = stretch.apply(data, T0)
    span = result.white_dn - result.black_dn
    x = np.clip((data.astype(np.float64) - result.black_dn) / span, 0.0, 1.0)
    exact = 255.0 * np.arcsinh(gain * x) / math.asinh(gain)
    assert np.abs(result.image.astype(np.float64) - exact).max() <= 1.0
    assert np.all(np.diff(result.image[0].astype(int)) >= 0)  # the curve never goes down


def test_the_curve_lifts_the_faint_wings() -> None:
    data = np.full((128, 128), 500, dtype=np.uint16)
    data[64, 64] = 20_500  # the peak, 20,000 above the sky
    data[64, 66] = 700  # a wing at 1% of the peak
    result = Autostretch(floor_sigmas=1.0).apply(data, T0)
    linear = 255 * 200 / (1.5 * 20_000)
    assert linear < 2
    assert int(result.image[64, 66]) >= 10  # asinh shows the wing at five times the level


def test_a_star_that_flickers_by_five_percent_still_flickers_in_the_output() -> None:
    """The white level follows the star slowly, so a brightness change survives the stretch."""
    stretch = Autostretch()
    amplitudes, levels = [], []
    for index in range(160):  # 8 s at 20 frames per second
        amplitude = 18_000.0 * (1.0 + 0.05 * math.sin(2.0 * math.pi * index / 8.0))
        data = star_frame(amplitude=amplitude, seed=index)
        out = stretch.apply(data, T0 + index * 50_000_000)
        if index >= 60:  # after the first seconds the average has settled
            amplitudes.append(float(data.max()) - BACKGROUND)
            levels.append(float(out.image.max()))
    spread = max(levels) - min(levels)
    assert spread >= 4  # gray levels of the brightest pixel between the dim and the bright frames
    assert np.corrcoef(amplitudes, levels)[0, 1] > 0.9


def test_a_stretch_that_followed_each_frame_would_lose_the_flicker() -> None:
    """The contrast case: a stretch with a time constant of one frame divides the flicker out."""
    fast = Autostretch(time_constant_s=0.001, headroom=1.0)
    levels = []
    for index in range(160):
        amplitude = 18_000.0 * (1.0 + 0.05 * math.sin(2.0 * math.pi * index / 8.0))
        out = fast.apply(star_frame(amplitude=amplitude, seed=index), T0 + index * 50_000_000)
        levels.append(float(out.image.max()))
    assert max(levels) - min(levels) <= 1  # the brightest pixel always maps to white


def test_the_white_level_follows_the_peak_with_the_time_constant() -> None:
    stretch = Autostretch(time_constant_s=3.0)
    bright = star_frame(amplitude=20_000.0, noise=0.0)
    stretch.apply(bright, T0)
    start = stretch.peak_dn
    faint = star_frame(amplitude=10_000.0, noise=0.0)
    step = start - (float(faint.max()) - BACKGROUND)  # the peak must fall by this much
    stretch.apply(faint, T0 + 300_000_000)
    assert start - stretch.peak_dn == pytest.approx(step * (1 - math.exp(-0.1)), rel=0.02)
    for index in range(2, 12):
        stretch.apply(faint, T0 + index * NS_PER_S)
    assert stretch.peak_dn == pytest.approx(float(faint.max()) - BACKGROUND, rel=0.05)


def test_the_white_level_has_a_floor_of_noise_sigmas() -> None:
    rng = np.random.default_rng(3)
    data = np.clip(np.rint(BACKGROUND + rng.normal(0.0, 24.0, (128, 128))), 0, 65535).astype(
        np.uint16
    )
    result = Autostretch(floor_sigmas=8.0).apply(data, T0)
    sigma = 1.4826 * float(np.median(np.abs(data.astype(float) - np.median(data))))
    assert result.white_dn - result.black_dn == pytest.approx(8.0 * sigma, rel=0.1)
    assert 20.0 < sigma < 28.0


def test_a_step_back_in_time_and_a_reset_restart_the_average() -> None:
    stretch = Autostretch()
    stretch.apply(star_frame(amplitude=20_000.0), T0 + 10 * NS_PER_S)
    assert stretch.peak_dn > 15_000
    stretch.apply(star_frame(amplitude=5_000.0), T0)  # the clock stepped back
    assert stretch.peak_dn == pytest.approx(5_000.0, rel=0.1)
    stretch.apply(star_frame(amplitude=20_000.0), T0 + NS_PER_S)
    stretch.reset()
    assert stretch.peak_dn == 0.0
    stretch.apply(star_frame(amplitude=5_000.0), T0 + 2 * NS_PER_S)
    assert stretch.peak_dn == pytest.approx(5_000.0, rel=0.1)


@pytest.mark.parametrize(
    "options",
    [{"time_constant_s": 0.0}, {"headroom": 0.5}, {"floor_sigmas": 0.0}, {"asinh_gain": -1.0}],
)
def test_the_stretch_refuses_settings_that_cannot_work(options: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="positive"):
        Autostretch(**options)


# --- The width -------------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.7, 1.1, 1.8])
def test_the_width_of_a_gaussian_star_is_its_sigma(sigma: float) -> None:
    data = star_frame(sigma=sigma, x=60.3, y=70.6, noise=8.0)
    widths = star_width_px(data, float(np.median(data)), 60.3, 70.6, 8.0)
    assert widths is not None
    assert widths[0] == pytest.approx(sigma, rel=0.06)
    assert widths[1] == pytest.approx(sigma, rel=0.06)


def test_the_two_axes_have_their_own_width() -> None:
    yy, xx = np.mgrid[0:128, 0:128]
    star = 18_000.0 * np.exp(-((xx - 64.0) ** 2) / (2 * 1.0**2) - ((yy - 64.0) ** 2) / (2 * 1.6**2))
    data = np.rint(BACKGROUND + star).astype(np.uint16)
    widths = star_width_px(data, BACKGROUND, 64.0, 64.0, 8.0)
    assert widths is not None
    assert widths[0] == pytest.approx(1.0, rel=0.05)
    assert widths[1] == pytest.approx(1.6, rel=0.05)


def test_a_star_at_the_edge_still_has_a_width_and_a_dark_aperture_has_none() -> None:
    data = star_frame(x=2.0, y=3.0, sigma=1.0, noise=0.0)
    widths = star_width_px(data, BACKGROUND, 2.0, 3.0, 8.0)
    assert widths is not None
    assert widths[0] == pytest.approx(1.0, rel=0.2)
    flat = np.full((128, 128), 500, dtype=np.uint16)
    assert star_width_px(flat, 500.0, 64.0, 64.0, 8.0) is None
    assert star_width_px(flat, 500.0, 500.0, 500.0, 8.0) is None  # the aperture is off the frame


# --- The image -------------------------------------------------------------------------------


def test_the_png_decodes_to_the_8_bit_image() -> None:
    result = Autostretch().apply(star_frame(), T0)
    blob = encode_png(result.image)
    assert blob.startswith(PNG_MAGIC)
    picture = Image.open(io.BytesIO(blob))
    assert picture.mode == "L"
    assert picture.size == (128, 128)
    assert np.array_equal(np.asarray(picture), result.image)  # lossless


@pytest.mark.parametrize("shape", [(128, 128), (1, 1), (7, 300), (300, 7)])
def test_the_png_writer_is_lossless_for_any_image(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(shape[0] * 1000 + shape[1])
    image = rng.integers(0, 256, shape, dtype=np.uint8)
    picture = Image.open(io.BytesIO(encode_png(image)))
    picture.verify()  # the chunks and their checksums hold together
    again = Image.open(io.BytesIO(encode_png(image)))
    assert again.mode == "L"
    assert again.size == (shape[1], shape[0])
    assert np.array_equal(np.asarray(again), image)


def test_the_png_of_a_non_contiguous_image_is_the_same_as_of_its_copy() -> None:
    image = np.arange(256 * 128, dtype=np.uint32).astype(np.uint8).reshape(256, 128)
    view = image[::2, ::2]
    assert encode_png(view) == encode_png(np.ascontiguousarray(view))


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros(16, dtype=np.uint8),
        np.zeros((4, 4), dtype=np.uint16),
        np.zeros((0, 4), dtype=np.uint8),
    ],
)
def test_the_png_writer_refuses_what_is_not_an_8_bit_gray_image(
    image: np.ndarray[Any, Any],
) -> None:
    with pytest.raises(ValueError, match="8-bit"):
        encode_png(image)


def test_a_frame_of_a_star_on_a_quiet_sky_takes_a_few_kilobytes() -> None:
    blob = encode_png(Autostretch().apply(star_frame(), T0).image)
    assert len(blob) < 6_000


def test_a_frame_of_noise_alone_stays_within_the_budget_of_the_wire() -> None:
    rng = np.random.default_rng(7)
    data = np.clip(np.rint(BACKGROUND + rng.normal(0.0, 24.0, (128, 128))), 0, 65535).astype(
        np.uint16
    )
    blob = encode_png(Autostretch().apply(data, T0).image)
    assert len(blob) < 16_000


# --- The renderer ----------------------------------------------------------------------------


def render_one(**options: object) -> tuple[FrameSlot, PolarisRenderer]:
    renderer = PolarisRenderer(scale_for=lambda mode: 1.91, aperture_for=lambda slot: 16.0)
    return slot_of(star_frame(), **options), renderer  # type: ignore[arg-type]


def test_the_state_describes_the_frame() -> None:
    slot, renderer = render_one(
        roi=Roi(2008, 1347, 128, 128),
        star=StarState(True, 2008 + 64.25, 1347 + 63.5, 0.2845, 40.0),
    )
    frame = renderer.render(slot)
    state = frame.state
    assert state.seq == 0
    assert state.t_utc_ns == slot.t_utc_ns
    assert state.t_utc.endswith("Z")
    assert (state.stream_id, state.mode, state.exposure_us, state.gain) == (3, "bin1", 2000, 0)
    assert (state.roi.x, state.roi.y, state.roi.width, state.roi.height) == (2008, 1347, 128, 128)
    assert (state.image_width, state.image_height, state.image_type) == (128, 128, "image/png")
    assert state.scale_arcsec_px == 1.91
    assert state.star.found is True
    assert (state.star.x, state.star.y) == (64.25, 63.5)  # pixels of the image, not of the sensor
    assert state.star.peak_fraction == 0.2845
    assert state.star.fwhm_arcsec == pytest.approx(2.3548 * SIGMA_PX * 1.91, rel=0.06)
    assert state.stretch.black_dn == pytest.approx(BACKGROUND, abs=2.0)
    assert state.stretch.white_dn > state.stretch.black_dn
    assert state.live_seeing is None
    assert frame.image.startswith(PNG_MAGIC)
    assert Image.open(io.BytesIO(frame.image)).size == (state.image_width, state.image_height)


def test_the_star_moves_with_the_roi_origin_and_not_with_the_sensor() -> None:
    first, renderer = render_one(roi=Roi(100, 200, 128, 128), star=StarState(True, 164.0, 264.0))
    second, other = render_one(roi=Roi(900, 800, 128, 128), star=StarState(True, 964.0, 864.0))
    assert renderer.render(first).state.star.x == 64.0
    assert other.render(second).state.star.y == 64.0


def test_the_state_is_small_and_fits_the_codec() -> None:
    slot, renderer = render_one()
    frame = renderer.render(slot, live_seeing_view())
    body = frame.state.model_dump_json()
    assert len(body.encode()) < 1500
    assert len(body.encode()) < MAX_STATE_BYTES
    assert json.loads(body)["live_seeing"]["seeing_fwhm_arcsec"] == 1.62


def test_the_rolling_seeing_value_goes_into_the_state() -> None:
    slot, renderer = render_one()
    view = live_seeing_view()
    assert renderer.render(slot, view).state.live_seeing == view
    assert "live_seeing" not in renderer.render(slot, view).state.quality


def test_a_frame_without_a_star_has_null_star_values_and_a_note() -> None:
    renderer = PolarisRenderer(scale_for=lambda mode: 1.91)
    frame = renderer.render(slot_of(star_frame(amplitude=0.0), star=NO_STAR))
    assert frame.state.star.found is False
    assert frame.state.star.model_dump() == {
        "found": False,
        "x": None,
        "y": None,
        "peak_fraction": None,
        "fwhm_arcsec": None,
    }
    assert "no star" in frame.state.quality["star"]
    assert frame.image.startswith(PNG_MAGIC)


def test_without_a_plate_scale_the_scale_and_the_width_are_null_and_say_why() -> None:
    renderer = PolarisRenderer()  # no profile: no plate scale
    frame = renderer.render(slot_of(star_frame()))
    assert frame.state.scale_arcsec_px is None
    assert frame.state.star.found is True
    assert frame.state.star.fwhm_arcsec is None
    assert "plate scale" in frame.state.quality["fwhm_arcsec"]
    assert "readout mode" in frame.state.quality["scale_arcsec_px"]


def test_the_frame_rate_comes_from_the_counts_and_the_times_of_the_kept_frames() -> None:
    renderer = PolarisRenderer(scale_for=lambda mode: 1.91)
    states = [renderer.render(slot_of(star_frame(seed=i), i)).state for i in range(4)]
    assert states[0].fast_fps is None
    assert "two frames" in states[0].quality["fast_fps"]
    assert states[1].fast_fps == pytest.approx(82.1, abs=0.1)
    assert states[3].fast_fps == pytest.approx(82.1, abs=0.1)
    assert "fast_fps" not in states[3].quality


def test_lost_frames_count_toward_the_rate_of_the_camera() -> None:
    renderer = PolarisRenderer()
    renderer.render(slot_of(star_frame(), 0, count_step=5))
    # The camera lost frames between the kept ones, and the counts include them, so the rate is the
    # rate of the camera and not the rate of the frames that arrived.
    slot = replace(slot_of(star_frame(seed=2), 1, count_step=5), count=5)
    assert renderer.render(slot).state.fast_fps == pytest.approx(82.1, abs=0.1)


def test_a_new_stream_or_a_new_exposure_resets_the_stretch_and_the_rate() -> None:
    renderer = PolarisRenderer()
    for index in range(3):
        renderer.render(slot_of(star_frame(amplitude=20_000.0), index))
    assert renderer.stretch.peak_dn > 15_000
    after = renderer.render(slot_of(star_frame(amplitude=4_000.0), 3, stream_id=4))
    assert after.state.fast_fps is None  # the rate starts again
    assert renderer.stretch.peak_dn == pytest.approx(4_000.0, rel=0.1)
    renderer.render(slot_of(star_frame(amplitude=4_000.0), 4, stream_id=4))
    changed = renderer.render(
        slot_of(star_frame(amplitude=9_000.0), 5, stream_id=4, exposure_us=8000)
    )
    assert changed.state.fast_fps is None
    assert renderer.stretch.peak_dn == pytest.approx(9_000.0, rel=0.1)


def test_the_settings_reach_the_stretch() -> None:
    renderer = PolarisRenderer(PolarisSettings(headroom=2.0, floor_sigmas=3.0, asinh_gain=5.0))
    assert renderer.stretch.headroom == 2.0
    assert renderer.stretch.floor_sigmas == 3.0
    assert renderer.stretch.asinh_gain == 5.0
    assert renderer.stretch.time_constant_s == 3.0
