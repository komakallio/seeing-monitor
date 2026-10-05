"""The configuration of the fast path: every constant of the estimators, with its default.

`FastPathConfig` is the pydantic model of the `[fastpath]` table. The defaults live in
`config/default.d/fastpath.toml`. Read the section with

    config = load_config()
    fast = config.section("fastpath", FastPathConfig)

Every value is provisional until commissioning (phase 3). The estimator stores the assumptions
that a reader needs to undo them (the outer scale, the wind speed, and the correction factors)
with every window. Hardware numbers (the aperture, the plate scale, the saturation level, the
conversion gain) come from the profile, so none of them appears here.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from seeingmon.config import SectionModel

Positive = Annotated[float, Field(gt=0)]
NonNegative = Annotated[float, Field(ge=0)]
Fraction = Annotated[float, Field(ge=0, le=1)]


class FastPathConfig(SectionModel):
    """Settings of the per-frame kernel, the windows, and the estimators. Lengths of time are in
    seconds of frame time."""

    # --- windows ---
    window_s: Positive = 60.0
    min_window_s: Positive = 5.0  # a shorter window gets no seeing statistics
    min_valid_fraction: Fraction = 0.5  # below this share of usable frames, no seeing statistics
    saturated_window_fraction: Fraction = (
        0.02  # the share of saturated frames that sets `saturated`
    )

    # --- the kernel ---
    aperture_diameter_px: Annotated[float, Field(ge=3, le=60)] | None = None
    aperture_min_px: Annotated[float, Field(ge=3, le=60)] = 15.0
    aperture_airy_widths: Positive = 12.0  # the derived aperture, in Airy FWHM of the profile
    recenter_iterations: Annotated[int, Field(ge=0, le=20)] = 2
    border_px: Annotated[int, Field(ge=1)] = 4
    border_step: Annotated[int, Field(ge=1)] = 2
    edge_margin_px: NonNegative = 1.0
    saturation_fraction: Annotated[float, Field(gt=0, le=1)] = 0.98
    min_star_snr: NonNegative = 6.0  # of the matched filter or the aperture
    hot_pixel_ratio: Annotated[float, Field(gt=0, lt=1)] = 0.03
    # The matched filters, in Airy FWHM: the first serves the missing-star test, and a search
    # frame keeps the best of all.
    matched_fwhm_airy_widths: Annotated[
        tuple[Annotated[float, Field(ge=0.25, le=8)], ...], Field(min_length=1, max_length=4)
    ] = (1.0, 2.0, 4.0)

    # --- the seeing estimator ---
    outer_scale_m: Positive = 20.0
    assumed_wind_ms: Positive = 10.0
    detrend_order: Annotated[int, Field(ge=0, le=4)] = 2
    outlier_sigma: NonNegative = 8.0
    g_tilt_coefficient: Positive = 0.170
    fwhm_coefficient: Positive = 0.98
    zenith_correction: bool = True
    apply_centroid_gain: bool = True
    structure_lag_min_s: Positive = 0.04
    structure_lag_max_s: Positive = 0.12
    min_samples: Annotated[int, Field(ge=10)] = 100

    # --- the spectrum ---
    welch_segment_s: Positive = 2.0
    welch_overlap: Annotated[float, Field(ge=0, lt=0.9)] = 0.5
    psd_bins: Annotated[int, Field(ge=4, le=200)] = 24
    max_interp_gap_frames: Annotated[int, Field(ge=0)] = 3
    max_interp_fraction: Fraction = 0.1
    vibration_threshold: Annotated[float, Field(gt=1)] = 5.0
    vibration_local_bins: Annotated[int, Field(ge=3)] = 15
    vibration_min_hz: NonNegative = 4.0

    # --- scintillation ---
    scintillation_trend_s: Positive = 1.0

    # --- the live estimate ---
    live_enabled: bool = True
    live_span_s: Positive = 10.0
    live_every_s: Positive = 2.0
    live_min_span_s: Positive = 4.0

    # --- buffers ---
    max_buffered_metrics: Annotated[int, Field(ge=100)] = 200_000

    @model_validator(mode="after")
    def _check_ranges(self) -> FastPathConfig:
        if self.structure_lag_max_s < self.structure_lag_min_s:
            raise ValueError("structure_lag_max_s must not be below structure_lag_min_s")
        if self.min_window_s > self.window_s:
            raise ValueError("min_window_s must not exceed window_s")
        if self.live_min_span_s > self.live_span_s:
            raise ValueError("live_min_span_s must not exceed live_span_s")
        return self
