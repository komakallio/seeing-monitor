"""The frame of the live video of Polaris: the stretch, the width of the star, and the PNG.

`PolarisRenderer` turns one kept frame of the fast stream (a `FrameSlot`) into the `PolarisFrame`
that goes to `web`: the lossless image, and the state beside it. It runs on the thread of the
stream service and never on the scheduler thread, because every step costs more than a frame
period can spare. The steps:

1. **Levels.** Black is the median of the frame. The noise is the robust sigma of every second pixel
   in each direction (the median absolute deviation, which a star does not move).
2. **Stretch.** `Autostretch` maps the counts to 8 bits. White is `headroom` times the peak of the
   star above black, and the peak is an exponential average over `time_constant_s` seconds of frame
   time. A stretch that followed each frame would divide the brightness of the star out, so that
   the flicker would vanish. The slow average keeps the flicker, and an excursion above the white
   level clips. White never falls below `floor_sigmas` times the noise, so a frame with no star
   shows its noise and not a stretched speck. An `asinh` curve with the gain `asinh_gain` is linear
   for the faint pixels and logarithmic for the bright ones, so the wings of the star show and the
   core does not clip. The curve is the same for every frame, so a table holds it (at least 4,096
   entries, and more for a steep curve), and the stretch of a frame is one subtraction, one
   scaling, and one lookup. The table differs from the exact curve by at most one gray level.
3. **Width.** The second moments of the star inside the aperture of the fast analysis (a soft
   circular aperture around the centroid that the analysis found) give the width of this frame,
   with the formula of the stored windows: the mean of both axes times 2.355 times the plate scale.
4. **Encode.** A small writer makes a lossless 8-bit grayscale PNG: no filter, and a zlib stream
   with Huffman coding only. The pixels of a frame are noise-like, so the string matching of the
   default strategy costs time and gains nothing. On a development machine, the frame of a star
   on a quiet sky takes about 3 KB and 0.2 ms, against 6 KB and 0.6 ms for Pillow with zlib at
   level 1, and 3 KB and 1 to 2 ms at level 6.
5. **State.** The `PolarisState` holds the time, the stream, the ROI, the plate scale, the frame
   rate of the camera, the size of the image, the star, the levels, and the rolling seeing value.

The renderer keeps the stretch and the estimate of the frame rate for one stream. A frame of
another stream (or of another exposure, gain, or mode) resets both.
"""

from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.analysis.base import StarState
from seeingmon.clock import NS_PER_S, utc_ns_to_iso
from seeingmon.frames import FrameData, Roi
from seeingmon.services.core.settings import PolarisSettings
from seeingmon.services.web.contract import (
    PNG_MAGIC,
    LiveSeeingView,
    PolarisFrame,
    PolarisStar,
    PolarisState,
    PolarisStretch,
    RoiView,
)

MAD_TO_SIGMA = 1.4826
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
MIN_RANGE_DN = 4.0
LUT_MIN_SIZE = 4096
NOISE_STRIDE = 2
DEFAULT_APERTURE_PX = 16.0
FPS_SMOOTHING = 0.3

NOT_FOUND = PolarisStar(found=False)


@dataclass(frozen=True, slots=True)
class FrameSlot:
    """What the stream keeps of one frame until the renderer takes it.

    `data` is a copy of the ROI, because the camera may reuse the buffer of the frame. `star` is the
    star as the fast analysis saw it in this frame. `count` is the number of camera frames of the
    stream up to this one, the lost frames included, so that the difference of two counts over
    the difference of two times is the frame rate of the camera.
    """

    data: FrameData
    stream_id: int
    t_utc_ns: int
    mode: str
    exposure_us: int
    gain: int
    adc_bits: int
    roi: Roi
    star: StarState
    count: int


@dataclass(frozen=True, slots=True)
class Stretched:
    """An 8-bit image and the levels that made it. The levels are in the counts of the frame."""

    image: npt.NDArray[np.uint8]
    black_dn: float
    white_dn: float


class Autostretch:
    """Map the counts of a frame to 8 bits with a stretch that adapts slowly (see above)."""

    def __init__(
        self,
        *,
        time_constant_s: float = 3.0,
        headroom: float = 1.5,
        floor_sigmas: float = 8.0,
        asinh_gain: float = 30.0,
    ) -> None:
        if time_constant_s <= 0 or headroom < 1 or floor_sigmas <= 0 or asinh_gain <= 0:
            raise ValueError("the time constant, the gain, and the floor must be positive")
        self.time_constant_s = time_constant_s
        self.headroom = headroom
        self.floor_sigmas = floor_sigmas
        self.asinh_gain = asinh_gain
        # One entry step moves the curve by at most one gray level at its steepest point, which
        # is its start: `asinh_gain / asinh(asinh_gain)` gray levels per unit of the input.
        self._size = max(LUT_MIN_SIZE, math.ceil(255.0 * asinh_gain / math.asinh(asinh_gain)) + 1)
        levels = np.arange(self._size, dtype=np.float64) / (self._size - 1)
        curve = np.rint(255.0 * np.arcsinh(asinh_gain * levels) / math.asinh(asinh_gain))
        self._lut = curve.astype(np.uint8)
        self._peak = 0.0
        self._last_ns: int | None = None

    @classmethod
    def from_settings(cls, settings: PolarisSettings) -> Autostretch:
        return cls(
            time_constant_s=settings.time_constant_s,
            headroom=settings.headroom,
            floor_sigmas=settings.floor_sigmas,
            asinh_gain=settings.asinh_gain,
        )

    @property
    def peak_dn(self) -> float:
        """The followed peak of the star above black, in counts."""
        return self._peak

    def reset(self) -> None:
        """Forget the peak, so that the next frame sets it. Call it when the stream changes."""
        self._peak = 0.0
        self._last_ns = None

    def _follow(self, peak: float, t_utc_ns: int) -> None:
        last = self._last_ns
        if last is None or t_utc_ns < last:  # the first frame, or the time stepped back
            self._peak = peak
        else:
            alpha = 1.0 - math.exp(-((t_utc_ns - last) / NS_PER_S) / self.time_constant_s)
            self._peak += alpha * (peak - self._peak)
        self._last_ns = t_utc_ns

    def apply(self, data: FrameData, t_utc_ns: int) -> Stretched:
        """Stretch a frame taken at `t_utc_ns`. A constant frame gives a black image."""
        black = float(np.median(data))
        sample = data[::NOISE_STRIDE, ::NOISE_STRIDE].astype(np.float32)
        sigma = MAD_TO_SIGMA * float(np.median(np.abs(sample - np.float32(black))))
        self._follow(max(float(data.max()) - black, 0.0), t_utc_ns)
        span = max(self.headroom * self._peak, self.floor_sigmas * sigma, MIN_RANGE_DN)
        work = data.astype(np.float32)
        work -= np.float32(black)
        work *= np.float32((self._size - 1) / span)
        work += np.float32(0.5)  # the lookup rounds to the nearest entry
        np.clip(work, 0.0, self._size - 1, out=work)
        return Stretched(self._lut[work.astype(np.intp)], black, black + span)


def star_width_px(
    data: FrameData, black: float, x: float, y: float, radius_px: float
) -> tuple[float, float] | None:
    """The widths of a star in pixels: the second-moment sigma along x and along y.

    `x` and `y` are the centroid in pixels of `data`, and `black` is the background. The aperture is
    the soft circle of the fast analysis: a pixel at the distance `r` from the centroid has the
    weight `clip(radius_px + 0.5 - r, 0, 1)`. Returns `None` when the aperture holds no light.
    """
    height, width = data.shape
    reach = math.ceil(radius_px + 1.0)
    x0, x1 = max(0, math.floor(x) - reach), min(width, math.floor(x) + reach + 2)
    y0, y1 = max(0, math.floor(y) - reach), min(height, math.floor(y) + reach + 2)
    if x0 >= x1 or y0 >= y1:
        return None
    box = data[y0:y1, x0:x1].astype(np.float64) - black
    dx = np.arange(x0, x1, dtype=np.float64) - x
    dy = np.arange(y0, y1, dtype=np.float64) - y
    weight = np.clip(radius_px + 0.5 - np.hypot(dx[None, :], dy[:, None]), 0.0, 1.0)
    light = box * weight
    total = float(light.sum())
    if not total > 0.0:
        return None
    mean_x = float((light.sum(axis=0) * dx).sum()) / total
    mean_y = float((light.sum(axis=1) * dy).sum()) / total
    var_x = float((light.sum(axis=0) * dx * dx).sum()) / total - mean_x * mean_x
    var_y = float((light.sum(axis=1) * dy * dy).sum()) / total - mean_y * mean_y
    return math.sqrt(max(var_x, 0.0)), math.sqrt(max(var_y, 0.0))


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def encode_png(image: npt.NDArray[np.uint8]) -> bytes:
    """Encode a 2-D 8-bit image as a lossless grayscale PNG. See the module text for the format."""
    if image.ndim != 2 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("the image must be a non-empty 2-D array of 8-bit pixels")
    height, width = image.shape
    rows = np.zeros((height, width + 1), dtype=np.uint8)  # the first byte of a row is the filter
    rows[:, 1:] = image
    packer = zlib.compressobj(1, zlib.DEFLATED, 15, 8, zlib.Z_HUFFMAN_ONLY)
    body = packer.compress(rows.tobytes()) + packer.flush()
    header = struct.pack(
        ">IIBBBBB", width, height, 8, 0, 0, 0, 0
    )  # 8 bits, grayscale, no interlace
    return PNG_MAGIC + _chunk(b"IHDR", header) + _chunk(b"IDAT", body) + _chunk(b"IEND", b"")


class PolarisRenderer:
    """Turn the kept frames of one stream into `PolarisFrame`s. See the module text.

    `scale_for` gives the plate scale of a readout mode in arcseconds per pixel, or `None` when the
    profile does not know the mode. `aperture_for` gives the diameter of the aperture of the fast
    analysis for a frame, in pixels. Call `render` from one thread.
    """

    def __init__(
        self,
        settings: PolarisSettings | None = None,
        *,
        scale_for: Callable[[str], float | None] | None = None,
        aperture_for: Callable[[FrameSlot], float] | None = None,
    ) -> None:
        self._stretch = Autostretch.from_settings(settings or PolarisSettings())
        self._scale_for = scale_for or (lambda mode: None)
        self._aperture_for = aperture_for or (lambda slot: DEFAULT_APERTURE_PX)
        self._key: tuple[int, str, int, int] | None = None
        self._previous: FrameSlot | None = None
        self._fps: float | None = None

    @property
    def stretch(self) -> Autostretch:
        return self._stretch

    def _follow_rate(self, slot: FrameSlot) -> float | None:
        """The frame rate of the camera, smoothed over the frames that the stream kept."""
        key = (slot.stream_id, slot.mode, slot.exposure_us, slot.gain)
        previous = self._previous
        if key != self._key:
            self._key = key
            self._fps = None
            self._stretch.reset()
        elif previous is not None and slot.t_utc_ns > previous.t_utc_ns:
            rate = (slot.count - previous.count) / ((slot.t_utc_ns - previous.t_utc_ns) / NS_PER_S)
            if rate > 0.0:
                self._fps = (
                    rate
                    if self._fps is None
                    else ((1.0 - FPS_SMOOTHING) * self._fps + FPS_SMOOTHING * rate)
                )
        self._previous = slot
        return self._fps

    def render(self, slot: FrameSlot, live: LiveSeeingView | None = None) -> PolarisFrame:
        """Stretch and encode a frame, and build its state. `live` is the rolling seeing value."""
        fps = self._follow_rate(slot)
        stretched = self._stretch.apply(slot.data, slot.t_utc_ns)
        scale = self._scale_for(slot.mode)
        roi = slot.roi
        quality: dict[str, str] = {}
        star = NOT_FOUND
        found = slot.star
        if found.found and found.x_px is not None and found.y_px is not None:
            x, y = found.x_px - roi.x, found.y_px - roi.y
            fwhm: float | None = None
            widths = star_width_px(
                slot.data, stretched.black_dn, x, y, 0.5 * self._aperture_for(slot)
            )
            if widths is None:
                quality["fwhm_arcsec"] = "the aperture of the star holds no light"
            elif scale is None:
                quality["fwhm_arcsec"] = "the profile does not give the plate scale of the mode"
            else:
                fwhm = round(FWHM_PER_SIGMA * 0.5 * (widths[0] + widths[1]) * scale, 3)
            peak = found.peak_fraction
            star = PolarisStar(
                found=True,
                x=round(x, 3),
                y=round(y, 3),
                peak_fraction=None if peak is None else round(peak, 4),
                fwhm_arcsec=fwhm,
            )
        else:
            quality["star"] = "the analysis found no star in this frame"
        if scale is None:
            quality["scale_arcsec_px"] = "the profile does not describe this readout mode"
        if fps is None:
            quality["fast_fps"] = "the frame rate needs two frames of the stream"
        if live is None:
            quality["live_seeing"] = "core has no rolling seeing value yet"
        height, width = slot.data.shape
        state = PolarisState(
            seq=0,
            t_utc=utc_ns_to_iso(slot.t_utc_ns, digits=3),
            t_utc_ns=slot.t_utc_ns,
            stream_id=slot.stream_id,
            mode=slot.mode,
            exposure_us=slot.exposure_us,
            gain=slot.gain,
            roi=RoiView(x=roi.x, y=roi.y, width=roi.width, height=roi.height),
            scale_arcsec_px=scale,
            fast_fps=None if fps is None else round(fps, 2),
            image_type="image/png",
            image_width=width,
            image_height=height,
            star=star,
            stretch=PolarisStretch(
                black_dn=round(stretched.black_dn, 1), white_dn=round(stretched.white_dn, 1)
            ),
            live_seeing=live,
            quality=quality,
        )
        return PolarisFrame(state, encode_png(stretched.image))
