"""What the simulator knows to be true, for tests of the estimators.

`SimDriver.truth` returns a `SimTruth`. It answers two kinds of question:

- *Properties and functions of time.* The atmosphere (`r0`, the seeing, the outer scale, the wind),
  the image motion for any exposure (`tilt_arcsec`), the sky and cloud conditions, and where each
  star sits on the sensor. Each answer is a pure function of the seed and the time, so you can
  ask in any order, with or without frames.
- *A record of each frame.* `frames` holds one `FrameTruth` per delivered frame, with the true
  image motion of the reference star, its true position, and the conditions of that frame.

**Image motion.** `tilt_x_arcsec` and `tilt_y_arcsec` are the G-tilt of the turbulence averaged
over the exposure of the reference star: the shift of the centroid of an ideal, noise-free image
from the star's catalog position, along the sensor columns (`x`) and rows (`y`). A plain
centroid of a bin1 frame follows them with a gain of 0.998 (the stamp truncates the diffraction
wings).
**Positions** use pixels of the full frame of the readout mode. The centre of pixel `(0, 0)` is
`(0, 0)`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S
from seeingmon.drivers.sim.params import ARCSEC_PER_RAD, SimParams
from seeingmon.drivers.sim.sky import (
    Clouds,
    ScintillationConfig,
    Site,
    airmass,
    sky_brightness_mag_arcsec2,
    sun_altitude_deg,
)
from seeingmon.drivers.sim.stars import SkyProjector, StarField
from seeingmon.drivers.sim.turbulence import (
    Layer,
    PupilGrid,
    TurbulenceModel,
    g_tilt_rms_arcsec,
    seeing_fwhm_arcsec,
)

FloatArray = npt.NDArray[np.float64]

# An exposure longer than this uses the long-exposure image model, and more tilt samples.
LONG_EXPOSURE_S = 0.1


@dataclass(frozen=True, slots=True)
class FrameTruth:
    """The truth about one delivered frame.

    `t_utc_ns` equals `Frame.t_utc_ns` (the middle of the exposure of the first row), and
    `t_ref_utc_ns` is the middle of the exposure of the reference star's row. The reference star
    is the brightest star whose centre falls in the ROI, or the brightest one near it. When the
    frame has no star, `star_index` is -1 and the star fields are NaN.
    """

    seq: int
    stream_id: int
    t_utc_ns: int
    t_ref_utc_ns: int
    exposure_s: float
    mode: str
    tilt_x_arcsec: float
    tilt_y_arcsec: float
    star_index: int
    star_x_px: float
    star_y_px: float
    catalog_x_px: float
    catalog_y_px: float
    flux_e: float
    scintillation_factor: float
    transparency: float
    sky_mag_arcsec2: float
    sun_altitude_deg: float
    sensor_temperature_c: float
    dropped_before: int


@dataclass(frozen=True, slots=True)
class StarPositions:
    """Stars on the sensor at one time, in pixels of the full frame of a mode."""

    index: npt.NDArray[np.intp]  # position in the field that you gave the driver
    x: FloatArray
    y: FloatArray
    mag: FloatArray


class SimTruth:
    """The injected truth of a simulated camera. See the module documentation."""

    def __init__(
        self,
        *,
        model: TurbulenceModel,
        modes: Mapping[str, SimParams],
        field: StarField,
        projector: SkyProjector,
        clouds: Clouds,
        site: Site,
        epoch_utc_ns: int,
        dark_sky_mag_arcsec2: float,
        twilight: bool,
        scintillation: ScintillationConfig,
        ambient_c: float,
        sensor_rise_c: float,
        ambient_drift_c_per_hour: float,
        keep_frames: int | None,
    ) -> None:
        self._model = model
        self._modes = dict(modes)
        self._field = field
        self._projector = projector
        self._clouds = clouds
        self._site = site
        self._epoch_utc_ns = epoch_utc_ns
        self._dark_sky = dark_sky_mag_arcsec2
        self._twilight = twilight
        self._scintillation = scintillation
        self._ambient_c = ambient_c
        self._rise_c = sensor_rise_c
        self._drift = ambient_drift_c_per_hour
        self._keep = keep_frames
        self._pupil = PupilGrid.circular(model.aperture_m, model.aperture_m / 48.0)
        self._frames: list[FrameTruth] = []
        self._first_seq_kept = 0
        self.frames_recorded = 0

    # --- the atmosphere ---

    @property
    def model(self) -> TurbulenceModel:
        """The turbulence model. Use it to read phase screens or to compute more tilt."""
        return self._model

    @property
    def aperture_m(self) -> float:
        return self._model.aperture_m

    @property
    def outer_scale_m(self) -> float:
        return self._model.config.outer_scale_m

    @property
    def layers(self) -> tuple[Layer, ...]:
        """The turbulent layers: the share of the turbulence, the wind speed, and the direction."""
        return self._model.config.layers

    @property
    def zenith_angle_deg(self) -> float:
        """The zenith angle of the line of sight, which scales `r0` by `(cos z)^(3/5)`."""
        return self._model.config.zenith_angle_deg

    def r0_zenith_m(self, t_utc_ns: int | None = None) -> float:
        """The Fried parameter at 500 nm at the zenith. Estimators report this quantity."""
        return self._model.r0_zenith_m(self._turbulence_time(t_utc_ns))

    def r0_observed_m(self, t_utc_ns: int | None = None) -> float:
        """The Fried parameter at 500 nm along the line of sight."""
        return self._model.r0_observed_m(self._turbulence_time(t_utc_ns))

    def seeing_fwhm_arcsec(self, t_utc_ns: int | None = None, *, at_zenith: bool = True) -> float:
        """The Kolmogorov seeing FWHM at 500 nm, `0.98 lambda / r0`, with the outer-scale factor.

        At the zenith by default, to match `r0_zenith_m`. Pass `at_zenith=False` for the line of
        sight.
        """
        r0 = self.r0_zenith_m(t_utc_ns) if at_zenith else self.r0_observed_m(t_utc_ns)
        return seeing_fwhm_arcsec(r0, outer_scale_m=self.outer_scale_m)

    def kolmogorov_seeing_fwhm_arcsec(
        self, t_utc_ns: int | None = None, *, at_zenith: bool = True
    ) -> float:
        """The same, without the outer-scale reduction: `0.98 lambda / r0`."""
        r0 = self.r0_zenith_m(t_utc_ns) if at_zenith else self.r0_observed_m(t_utc_ns)
        return seeing_fwhm_arcsec(r0)

    def image_motion_rms_arcsec(self, t_utc_ns: int | None = None) -> float:
        """The expected one-axis rms of the instantaneous G-tilt, with the outer scale applied."""
        return g_tilt_rms_arcsec(self.aperture_m, self.r0_observed_m(t_utc_ns), self.outer_scale_m)

    def kolmogorov_image_motion_rms_arcsec(self, t_utc_ns: int | None = None) -> float:
        """The same for an infinite outer scale: `sqrt(0.170 lambda^2 D^(-1/3) r0^(-5/3))`."""
        return g_tilt_rms_arcsec(self.aperture_m, self.r0_observed_m(t_utc_ns))

    def substeps(self, exposure_s: float) -> int:
        """The number of instants that the simulator samples inside a short exposure.

        An exposure longer than `LONG_EXPOSURE_S` uses the exact mean of the low-frequency part
        instead (`SimTruth.tilt_arcsec`), so this returns the number for the longest short one.
        """
        return self._model.suggest_substeps(min(exposure_s, LONG_EXPOSURE_S))

    def tilt_arcsec(
        self, t_start_utc_ns: int, exposure_s: float, n_sub: int | None = None
    ) -> tuple[float, float]:
        """The G-tilt averaged over an exposure, in arcseconds along the sensor `x` and `y`.

        `t_start_utc_ns` is the start of the exposure of the star's row. The result is a pure
        function of the seed, so it needs no frame.
        """
        start = self._turbulence_time(t_start_utc_ns)
        if exposure_s > LONG_EXPOSURE_S and n_sub is None:
            tilt_x, tilt_y = self._model.long_exposure_tilt_rad(start, exposure_s, self._pupil)
        else:
            steps = n_sub if n_sub is not None else self.substeps(exposure_s)
            tilt_x, tilt_y = self._model.exposure_tilt_rad(start, exposure_s, self._pupil, steps)
        return tilt_x * ARCSEC_PER_RAD, tilt_y * ARCSEC_PER_RAD

    def tilt_series_arcsec(
        self, t_start_utc_ns: npt.ArrayLike, exposure_s: float, n_sub: int | None = None
    ) -> FloatArray:
        """The exposure-averaged G-tilt for many start times, with shape `(times, 2)`."""
        starts = np.atleast_1d(np.asarray(t_start_utc_ns, dtype=np.int64))
        out = np.empty((len(starts), 2), dtype=np.float64)
        for index, start in enumerate(starts):
            out[index] = self.tilt_arcsec(int(start), exposure_s, n_sub)
        return out

    def _turbulence_time(self, t_utc_ns: int | None) -> float:
        return 0.0 if t_utc_ns is None else (t_utc_ns - self._epoch_utc_ns) / NS_PER_S

    # --- sky, clouds, and the sensor ---

    @property
    def site(self) -> Site:
        """The synthetic site that defines the sun and the airmass."""
        return self._site

    def airmass_of_pole(self) -> float:
        """The airmass of the celestial pole, where Polaris sits."""
        return airmass(90.0 - self._site.latitude_deg)

    def transparency(self, t_utc_ns: int | FloatArray) -> float | FloatArray:
        """The cloud transparency, from 0 to 1."""
        return self._clouds.transparency(t_utc_ns)

    def sun_altitude_deg(self, t_utc_ns: int | FloatArray) -> float | FloatArray:
        return sun_altitude_deg(self._site, t_utc_ns)

    def sky_mag_arcsec2(self, t_utc_ns: int | FloatArray) -> float | FloatArray:
        """The sky surface brightness in mag/arcsec^2, with twilight when it is on."""
        return sky_brightness_mag_arcsec2(
            self._dark_sky, sun_altitude_deg(self._site, t_utc_ns), twilight=self._twilight
        )

    def scintillation_rms(self, exposure_s: float) -> float:
        """The relative rms flux noise of a star at the pole's airmass, for an exposure."""
        return self._scintillation.index(exposure_s, self.airmass_of_pole())

    def sensor_temperature_c(self, t_utc_ns: int) -> float:
        """The sensor temperature: the ambient temperature plus the self-heating, to 0.1 degree."""
        hours = (t_utc_ns - self._epoch_utc_ns) / (NS_PER_S * 3600.0)
        return round(self._ambient_c + self._drift * hours + self._rise_c, 1)

    # --- stars ---

    @property
    def field(self) -> StarField:
        """The star field that the simulator renders."""
        return self._field

    def star_positions(self, t_utc_ns: int, mode: str) -> StarPositions:
        """Where the stars that can reach the sensor sit at a time, without image motion."""
        params = self._modes[mode]
        x, y = self._projector.project(t_utc_ns, params.pixel_rad, params.width, params.height)
        kept = self._projector.stars
        return StarPositions(self._projector.indices, x, y, kept.mag)

    def pole_pixel(self, t_utc_ns: int, mode: str) -> tuple[float, float]:
        """The pixel position of the celestial pole. It does not move."""
        params = self._modes[mode]
        return self._projector.pole_pixel(t_utc_ns, params.pixel_rad, params.width, params.height)

    @property
    def roll_deg(self) -> float:
        """The position angle of north (the direction to the pole) in the image."""
        return self._projector.roll_deg()

    # --- frames ---

    def record(self, frame: FrameTruth) -> None:
        """Store the truth of a delivered frame. The driver calls this."""
        self._frames.append(frame)
        self.frames_recorded += 1
        if self._keep is not None and len(self._frames) > self._keep:
            drop = len(self._frames) - self._keep
            del self._frames[:drop]
            self._first_seq_kept += drop

    @property
    def frames(self) -> list[FrameTruth]:
        """The stored per-frame truth, oldest first. The list is live: do not modify it."""
        return self._frames

    def frame_arrays(self, stream_id: int | None = None) -> dict[str, Any]:
        """The stored frames as arrays, keyed by field name. Pass a stream to select one."""
        frames = [f for f in self._frames if stream_id is None or f.stream_id == stream_id]
        names = [field.name for field in dataclasses.fields(FrameTruth)]
        return {name: np.asarray([getattr(f, name) for f in frames]) for name in names}
