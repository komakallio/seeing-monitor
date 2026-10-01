"""The nightly star summary: the `star_epoch` record.

Each survey frame measures the position and the brightness of every matched star. A long-term
analysis (proper motion checks, variable stars, an aging sensor, a drifting pointing) needs
these values over years, but the star lists keep them for one year only. So each night the
survey averages the frames of the night into one row per star:

- `cat_row`: the row of the star in the catalog (the catalog ID is in the provenance).
- `dx_arcsec` and `dy_arcsec`: the mean position offset, measured minus predicted, along the
  sensor columns and rows.
- `mag`: the mean magnitude in the Gaia G scale, from the zero point and the color term of
  each frame.
- `mag_rms`: the scatter of the magnitude over the night's frames.
- `n_frames`: the number of frames that measured the star.

Every value is a little-endian float32 (`StarEpochRecord.data`), so the row takes 24 bytes. A
station with 1,500 stars writes 13 MB a year.

**The offset** is the residual of the pointing fit: where the detector put the star minus where
the attitude of the frame predicts it. It follows what the fit leaves out (refraction, lens
distortion, a catalog error), and it averages the noise of a single frame down.

**The night** follows `seeingmon.survey.nights`: the UTC hour that ends one night and starts the
next comes from the configuration. The accumulator closes a night when a frame of the next one
arrives, or when you call `flush`. `t_utc_ns` of the record is the time of the first frame of the
summary, and the label of the night is `night`. A restart in the middle of a night leaves two
summaries with the same label, and each starts at its own first frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.records.survey import StarEpochRecord, pack_star_rows
from seeingmon.survey.geometry import FloatArray
from seeingmon.survey.nights import night_label

STAR_EPOCH_COLUMNS = ["cat_row", "dx_arcsec", "dy_arcsec", "mag", "mag_rms", "n_frames"]
_FRAME_FLOATS = 3  # dx, dy, and the magnitude
_STAR_BYTES = 4 + 4 * _FRAME_FLOATS


@dataclass(frozen=True, slots=True, eq=False)
class FrameStars:
    """What one frame contributes: its matched stars with offsets in arcseconds and magnitudes."""

    cat_row: npt.NDArray[np.int32]
    dx_arcsec: npt.NDArray[np.float32]
    dy_arcsec: npt.NDArray[np.float32]
    mag: npt.NDArray[np.float32]

    def __len__(self) -> int:
        return int(self.cat_row.size)

    @classmethod
    def empty(cls) -> FrameStars:
        return cls(
            np.zeros(0, np.int32),
            np.zeros(0, np.float32),
            np.zeros(0, np.float32),
            np.zeros(0, np.float32),
        )

    @classmethod
    def build(
        cls, cat_row: npt.ArrayLike, dx: npt.ArrayLike, dy: npt.ArrayLike, mag: npt.ArrayLike
    ) -> FrameStars:
        return cls(
            np.asarray(cat_row, dtype=np.int32),
            np.asarray(dx, dtype=np.float32),
            np.asarray(dy, dtype=np.float32),
            np.asarray(mag, dtype=np.float32),
        )

    def to_bytes(self) -> bytes:
        """The columns as one buffer, for the trip from a worker process (little-endian)."""
        rows = np.empty((len(self), _FRAME_FLOATS), dtype="<f4")
        rows[:, 0] = self.dx_arcsec
        rows[:, 1] = self.dy_arcsec
        rows[:, 2] = self.mag
        return self.cat_row.astype("<i4").tobytes() + rows.tobytes()

    @classmethod
    def from_bytes(cls, payload: bytes) -> FrameStars:
        count, remainder = divmod(len(payload), _STAR_BYTES)
        if remainder:
            raise ValueError("the star buffer has the wrong size")
        cat_row = np.frombuffer(payload[: 4 * count], dtype="<i4").astype(np.int32)
        rows = np.frombuffer(payload[4 * count :], dtype="<f4").reshape(count, _FRAME_FLOATS)
        return cls(
            cat_row,
            rows[:, 0].astype(np.float32),
            rows[:, 1].astype(np.float32),
            rows[:, 2].astype(np.float32),
        )


class NightAccumulator:
    """Averages the stars of a night's frames, one running mean for each catalog row.

    `n_catalog` is the number of rows of the catalog. `add` takes one frame. When a frame belongs
    to a later night than the frames before it, the accumulator closes the earlier night and
    returns its record, and `flush` closes the open night. A star needs at least `min_frames`
    frames to appear in the summary.
    """

    def __init__(
        self,
        n_catalog: int,
        *,
        station_id: str,
        profile_id: str,
        provenance: dict[str, str],
        split_utc_hour: float = 12.0,
        min_frames: int = 3,
    ) -> None:
        if n_catalog < 1 or min_frames < 1:
            raise ValueError("n_catalog and min_frames must be at least 1")
        self._n = n_catalog
        self._station_id = station_id
        self._profile_id = profile_id
        self._provenance = dict(provenance)
        self._split = split_utc_hour
        self._min_frames = min_frames
        self._night: str | None = None
        self._first_ns = 0
        self._frames = 0
        self._count: npt.NDArray[np.int32] = np.zeros(0, np.int32)
        self._mean: FloatArray = np.zeros((0, 3))
        self._m2: FloatArray = np.zeros(0)

    @property
    def night(self) -> str | None:
        """The label of the open night, or `None` before the first frame."""
        return self._night

    @property
    def frames(self) -> int:
        """The number of frames in the open night."""
        return self._frames

    def add(self, t_utc_ns: int, stars: FrameStars) -> StarEpochRecord | None:
        """Add a frame. Returns the record of the night that this frame closed, if any."""
        label = night_label(t_utc_ns, self._split)
        finished: StarEpochRecord | None = None
        if self._night is not None and label != self._night:
            finished = self.flush()
        if self._night is None:
            self._start(label, t_utc_ns)
        self._frames += 1
        rows = stars.cat_row.astype(np.intp)
        valid = np.flatnonzero((rows >= 0) & (rows < self._n))
        rows, first = np.unique(rows[valid], return_index=True)  # a star counts once per frame
        if rows.size == 0:
            return finished
        picked = valid[first]
        offsets = np.column_stack(
            [stars.dx_arcsec[picked], stars.dy_arcsec[picked], stars.mag[picked]]
        ).astype(np.float64)
        # Welford's update, with a mean and a sum of squares for each catalog row.
        count = self._count[rows] + 1
        delta = offsets - self._mean[rows]
        self._mean[rows] += delta / count[:, None]
        self._m2[rows] += delta[:, 2] * (offsets[:, 2] - self._mean[rows][:, 2])
        self._count[rows] = count
        return finished

    def flush(self) -> StarEpochRecord | None:
        """Close the open night and return its record, or `None` when it has no star."""
        if self._night is None:
            return None
        label, first_ns, frames = self._night, self._first_ns, self._frames
        keep = np.flatnonzero(self._count >= self._min_frames)
        record: StarEpochRecord | None = None
        if keep.size:
            count = self._count[keep].astype(np.float64)
            variance = np.where(count > 1, self._m2[keep] / np.maximum(count - 1.0, 1.0), 0.0)
            table = np.column_stack(
                [
                    keep.astype(np.float64),
                    self._mean[keep, 0],
                    self._mean[keep, 1],
                    self._mean[keep, 2],
                    np.sqrt(np.maximum(variance, 0.0)),
                    count,
                ]
            )
            record = StarEpochRecord(
                station_id=self._station_id,
                t_utc_ns=first_ns,
                profile_id=self._profile_id,
                provenance=self._provenance,
                night=label,
                n_stars=int(keep.size),
                n_frames=frames,
                columns=list(STAR_EPOCH_COLUMNS),
                data=pack_star_rows(table),
            )
        self._night = None
        self._frames = 0
        return record

    def _start(self, label: str, t_utc_ns: int) -> None:
        self._night = label
        self._first_ns = t_utc_ns
        self._frames = 0
        self._count = np.zeros(self._n, dtype=np.int32)
        self._mean = np.zeros((self._n, 3))
        self._m2 = np.zeros(self._n)
