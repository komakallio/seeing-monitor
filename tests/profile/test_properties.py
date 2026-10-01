"""Properties of the derived values, checked over random sensors, optics, and requests."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.profile import Limits, Optics, Profile, ReadoutMode, parse_profile
from seeingmon.profile import derived as d
from tests.profile.builders import reference_data, synthetic_data

ARCSEC_PER_RAD = 180 * 3600 / math.pi


def make_mode(**overrides: Any) -> ReadoutMode:
    """A validated readout mode: the synthetic `native` mode with some fields replaced."""
    fields = synthetic_data()["readout_modes"][0]
    fields.update(overrides)
    return ReadoutMode.model_validate(fields)


def make_limits(width_multiple: int = 8, height_multiple: int = 2) -> Limits:
    return Limits.model_validate(
        {
            "roi_width_multiple": width_multiple,
            "roi_height_multiple": height_multiple,
            "gain_range": [0, 570],
            "exposure_us_range": [32, 2_000_000_000],
        }
    )


def make_optics(focal_length_mm: float = 250.0, aperture_mm: float = 50.0) -> Optics:
    return Optics.model_validate(
        {"focal_length_mm": focal_length_mm, "aperture_mm": aperture_mm, "wavelength_nm": 600.0}
    )


# --- ROI rounding ----------------------------------------------------------------------------

steps = st.sampled_from([1, 2, 4, 8, 16, 32])
requests = st.one_of(
    st.integers(-100_000, 100_000),
    st.floats(-100_000, 100_000, allow_nan=False, allow_infinity=False),
)


@given(
    width_step=steps,
    height_step=steps,
    frame_width=st.integers(32, 20_000),
    frame_height=st.integers(32, 20_000),
    x=requests,
    y=requests,
    width=requests,
    height=requests,
)
def test_a_clamped_roi_always_follows_the_rules_and_stays_in_the_frame(
    width_step: int,
    height_step: int,
    frame_width: int,
    frame_height: int,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    mode = make_mode(width_px=frame_width, height_px=frame_height)
    limits = make_limits(width_step, height_step)
    roi = d.clamp_roi(mode, limits, x, y, width, height)
    assert roi.width % width_step == 0
    assert roi.height % height_step == 0
    assert width_step <= roi.width <= frame_width
    assert height_step <= roi.height <= frame_height
    assert 0 <= roi.x <= frame_width - roi.width
    assert 0 <= roi.y <= frame_height - roi.height
    # Clamping again changes nothing.
    assert d.clamp_roi(mode, limits, roi.x, roi.y, roi.width, roi.height) == roi


@given(
    width_step=steps,
    height_step=steps,
    frame_width=st.integers(32, 20_000),
    frame_height=st.integers(32, 20_000),
    data=st.data(),
)
def test_a_valid_roi_is_returned_unchanged(
    width_step: int, height_step: int, frame_width: int, frame_height: int, data: st.DataObject
) -> None:
    mode = make_mode(width_px=frame_width, height_px=frame_height)
    limits = make_limits(width_step, height_step)
    width = data.draw(st.integers(1, frame_width // width_step)) * width_step
    height = data.draw(st.integers(1, frame_height // height_step)) * height_step
    x = data.draw(st.integers(0, frame_width - width))
    y = data.draw(st.integers(0, frame_height - height))
    roi = d.clamp_roi(mode, limits, x, y, width, height)
    assert (roi.x, roi.y, roi.width, roi.height) == (x, y, width, height)


@given(
    step=steps,
    frame=st.integers(64, 20_000),
    requested=st.floats(0, 30_000, allow_nan=False, allow_infinity=False),
)
def test_the_size_rounds_to_the_nearest_allowed_multiple(
    step: int, frame: int, requested: float
) -> None:
    largest = frame // step * step
    roi = d.clamp_roi(
        make_mode(width_px=frame, height_px=frame),
        make_limits(step, step),
        0,
        0,
        requested,
        requested,
    )
    if requested <= step:
        assert roi.width == step
    elif requested >= largest:
        assert roi.width == largest
    else:
        assert abs(roi.width - requested) <= step / 2 + 1e-9  # a tie rounds up


@given(
    width_step=steps,
    height_step=steps,
    pixel_um=st.floats(0.5, 50, allow_nan=False),
    focal_mm=st.floats(10, 5000, allow_nan=False),
    full_width_arcmin=st.floats(0.01, 600, allow_nan=False),
)
def test_a_roi_for_an_angular_width_follows_the_rules(
    width_step: int, height_step: int, pixel_um: float, focal_mm: float, full_width_arcmin: float
) -> None:
    mode = make_mode(pixel_size_um=pixel_um)
    limits = make_limits(width_step, height_step)
    width, height = d.roi_size_px(mode, make_optics(focal_mm), limits, full_width_arcmin)
    assert width % width_step == 0
    assert height % height_step == 0
    assert width_step <= width <= mode.width_px
    assert height_step <= height <= mode.height_px


@given(
    small=st.floats(0.05, 100, allow_nan=False),
    extra=st.floats(0, 100, allow_nan=False),
)
def test_a_wider_patch_never_gives_a_smaller_roi(small: float, extra: float) -> None:
    mode, limits, optics = make_mode(), make_limits(), make_optics()
    narrow = d.roi_size_px(mode, optics, limits, small)
    wide = d.roi_size_px(mode, optics, limits, small + extra)
    assert wide[0] >= narrow[0]
    assert wide[1] >= narrow[1]


# --- Plate scale and field of view ---------------------------------------------------------------


@given(
    pixel_um=st.floats(0.5, 50, allow_nan=False),
    focal_mm=st.floats(10, 5000, allow_nan=False),
    factor=st.floats(0.1, 10, allow_nan=False),
)
def test_the_plate_scale_is_linear_in_pixel_size_and_inverse_in_focal_length(
    pixel_um: float, focal_mm: float, factor: float
) -> None:
    mode = make_mode(pixel_size_um=pixel_um)
    optics = make_optics(focal_mm)
    scale = d.plate_scale_arcsec_per_px(mode, optics)
    scaled_pixel = d.plate_scale_arcsec_per_px(make_mode(pixel_size_um=pixel_um * factor), optics)
    scaled_focus = d.plate_scale_arcsec_per_px(mode, make_optics(focal_mm / factor))
    assert scaled_pixel == pytest.approx(scale * factor, rel=1e-12)
    assert scaled_focus == pytest.approx(scale * factor, rel=1e-12)
    # The product of the scale and the focal length over the pixel size is a constant.
    assert scale * focal_mm / pixel_um == pytest.approx(ARCSEC_PER_RAD * 1e-3, rel=1e-12)


@given(
    pixel_um=st.floats(0.5, 50, allow_nan=False),
    focal_mm=st.floats(10, 5000, allow_nan=False),
    width=st.integers(8, 20_000),
    height=st.integers(8, 20_000),
)
def test_the_diagonal_field_follows_from_the_width_and_the_height(
    pixel_um: float, focal_mm: float, width: int, height: int
) -> None:
    fov = d.field_of_view(
        make_mode(pixel_size_um=pixel_um, width_px=width, height_px=height), make_optics(focal_mm)
    )

    def half_tangent(degrees: float) -> float:
        return math.tan(math.radians(degrees) / 2)

    assert half_tangent(fov.diagonal_deg) ** 2 == pytest.approx(
        half_tangent(fov.width_deg) ** 2 + half_tangent(fov.height_deg) ** 2, rel=1e-9
    )
    assert fov.diagonal_deg >= max(fov.width_deg, fov.height_deg)


@given(
    pixel_um=st.floats(0.5, 50, allow_nan=False),
    focal_mm=st.floats(500, 5000, allow_nan=False),
)
def test_the_field_of_view_is_the_pixel_count_times_the_scale_for_a_small_field(
    pixel_um: float, focal_mm: float
) -> None:
    """For a narrow field, tan(x) is x, so the field is the pixel count times the plate scale."""
    mode = make_mode(pixel_size_um=pixel_um, width_px=100, height_px=100)
    optics = make_optics(focal_mm)
    fov = d.field_of_view(mode, optics)
    expected = 100 * d.plate_scale_arcsec_per_px(mode, optics) / 3600  # degrees
    assert fov.width_deg == pytest.approx(expected, rel=1e-4)


# --- Gain tables ---------------------------------------------------------------------------------


@st.composite
def gain_tables(draw: st.DrawFn) -> list[dict[str, Any]]:
    """Random strictly increasing gain rows, with steps where the gains are adjacent."""
    gains = sorted(draw(st.lists(st.integers(1, 600), unique=True, min_size=0, max_size=6)))
    rows = []
    previous = 0
    for gain in gains:
        step = gain == previous + 1 and draw(st.booleans())
        rows.append(
            {
                "gain": gain,
                "e_per_adu": draw(st.floats(0.01, 10, allow_nan=False)),
                "read_noise_e": draw(st.floats(0.5, 10, allow_nan=False)),
                "step": step,
            }
        )
        previous = gain
    return rows


@given(table=gain_tables())
def test_the_gain_table_returns_its_own_rows(table: list[dict[str, Any]]) -> None:
    mode = make_mode(gain_points=table)
    for row in table:
        assert d.e_per_adu(mode, row["gain"]) == row["e_per_adu"]
        assert d.read_noise_e(mode, row["gain"]) == row["read_noise_e"]
    assert d.e_per_adu(mode, 0) == mode.e_per_adu_gain0
    assert d.read_noise_e(mode, 0) == mode.read_noise_gain0_e


@given(table=gain_tables(), fraction=st.floats(0, 1, exclude_max=True, allow_nan=False))
def test_values_between_two_rows_lie_between_the_rows(
    table: list[dict[str, Any]], fraction: float
) -> None:
    mode = make_mode(gain_points=table)
    rows = [(0, mode.e_per_adu_gain0, mode.read_noise_gain0_e, False)] + [
        (p["gain"], p["e_per_adu"], p["read_noise_e"], p["step"]) for p in table
    ]
    for (g0, e0, n0, _), (g1, e1, n1, step1) in pairwise(rows):
        gain = g0 + fraction * (g1 - g0)
        if gain >= g1:  # rounding put the gain on the next row
            continue
        e, n = d.e_per_adu(mode, gain), d.read_noise_e(mode, gain)
        if step1:  # the values never interpolate across a step
            assert (e, n) == (e0, n0)
            continue
        assert min(e0, e1) * (1 - 1e-12) <= e <= max(e0, e1) * (1 + 1e-12)
        assert min(n0, n1) - 1e-12 <= n <= max(n0, n1) + 1e-12
        # Electrons per ADU is log-linear in gain, and the read noise is linear in gain.
        assert math.log(e) == pytest.approx(math.log(e0) + fraction * (math.log(e1) - math.log(e0)))
        assert n == pytest.approx(n0 + fraction * (n1 - n0))


@given(table=gain_tables(), gain=st.floats(0, 700, allow_nan=False))
def test_every_gain_has_positive_finite_values(table: list[dict[str, Any]], gain: float) -> None:
    mode = make_mode(gain_points=table)
    for value in (d.e_per_adu(mode, gain), d.read_noise_e(mode, gain)):
        assert 0 < value < math.inf


@given(
    table=gain_tables(),
    gain=st.floats(0, 700, allow_nan=False),
    bits=st.integers(8, 16),
    full_well=st.floats(100, 200_000, allow_nan=False),
)
def test_saturation_is_the_smaller_of_the_well_and_the_adc(
    table: list[dict[str, Any]], gain: float, bits: int, full_well: float
) -> None:
    mode = make_mode(gain_points=table, adc_bits=bits, full_well_gain0_e=full_well)
    saturation = d.saturation(mode, gain)
    conversion = d.e_per_adu(mode, gain)
    full_scale = 2**bits - 1
    assert saturation.native_dn <= full_scale
    assert saturation.native_dn == pytest.approx(min(full_well / conversion, full_scale))
    assert saturation.container_dn == pytest.approx(saturation.native_dn * 2 ** (16 - bits))
    assert saturation.container_dn <= 65535
    assert saturation.full_well_e <= full_well * (1 + 1e-12)
    assert saturation.full_well_e == pytest.approx(d.full_well_e(mode, gain))
    assert saturation.full_well_e == pytest.approx(min(full_well, conversion * full_scale))
    assert (saturation.limited_by == "well") == (full_well / conversion < full_scale)


# --- Frame period ----------------------------------------------------------------------------


@given(
    rows=st.integers(1, 1500),
    more_rows=st.integers(0, 100),
    exposure_us=st.floats(1, 1e7, allow_nan=False),
    row_time_us=st.floats(1, 100, allow_nan=False),
    overhead_ms=st.floats(0, 10, allow_nan=False),
)
def test_the_frame_period_is_the_larger_of_the_exposure_and_the_readout(
    rows: int, more_rows: int, exposure_us: float, row_time_us: float, overhead_ms: float
) -> None:
    mode = make_mode(row_time_us=row_time_us, frame_overhead_ms=overhead_ms)
    readout = overhead_ms * 1e-3 + rows * row_time_us * 1e-6
    period = d.frame_period_s(mode, rows, exposure_us)
    assert period == pytest.approx(max(exposure_us * 1e-6, readout))
    assert period >= exposure_us * 1e-6
    assert period >= d.readout_time_s(mode, rows) * (1 - 1e-12)
    assert d.max_frame_rate_hz(mode, rows, exposure_us) * period == pytest.approx(1.0)
    # More rows never make a frame faster.
    taller = min(rows + more_rows, mode.height_px)
    assert d.frame_period_s(mode, taller, exposure_us) >= period * (1 - 1e-12)


@given(
    width=st.integers(1, 2000),
    height=st.integers(1, 1500),
    exposure_us=st.floats(1, 1e6, allow_nan=False),
)
def test_the_data_rate_is_the_frame_size_times_the_frame_rate(
    width: int, height: int, exposure_us: float
) -> None:
    mode = make_mode()
    rate = d.data_rate_bytes_per_s(mode, width, height, exposure_us)
    assert rate == pytest.approx(
        width * height * 2 * d.max_frame_rate_hz(mode, height, exposure_us)
    )


# --- The dark-current prior ------------------------------------------------------------------


@given(
    start=st.floats(-40, 0, allow_nan=False),
    gaps=st.lists(st.floats(1, 15, allow_nan=False), min_size=1, max_size=5),
    rates=st.lists(st.floats(1e-4, 10, allow_nan=False), min_size=6, max_size=6),
    fraction=st.floats(0, 1, allow_nan=False),
)
def test_the_dark_current_prior_stays_between_neighbouring_points(
    start: float, gaps: list[float], rates: list[float], fraction: float
) -> None:
    temperatures = [start]
    for gap in gaps:
        temperatures.append(temperatures[-1] + gap)
    points = [
        {"temperature_c": t, "e_per_s_per_px": r} for t, r in zip(temperatures, rates, strict=False)
    ]
    profile = _with_dark_current(points)
    for first, second in pairwise(points):
        t = first["temperature_c"] + fraction * (second["temperature_c"] - first["temperature_c"])
        value = profile.dark_current_e_per_s_per_px(t)
        low, high = sorted((first["e_per_s_per_px"], second["e_per_s_per_px"]))
        assert low * (1 - 1e-9) <= value <= high * (1 + 1e-9)
    for point in points:
        assert profile.dark_current_e_per_s_per_px(point["temperature_c"]) == pytest.approx(
            point["e_per_s_per_px"]
        )


def _with_dark_current(points: list[dict[str, float]]) -> Profile:
    data = reference_data()
    data["photometry"]["dark_current"] = points
    return parse_profile(data)
