"""Small builders that the scheduler tests share."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from seeingmon.frames import Frame, FrameFlag, Roi, TimeQuality


def make_frame(
    data: npt.NDArray[np.uint8] | npt.NDArray[np.uint16],
    *,
    mode: str = "bin2",
    gain: int = 120,
    exposure_us: int = 1000,
    adc_bits: int = 14,
    t_utc_ns: int = 0,
    seq: int = 0,
    stream_id: int = 1,
    dropped_before: int = 0,
) -> Frame:
    """Wrap a pixel array in a `Frame` with plausible metadata."""
    height, width = data.shape
    return Frame(
        data=data,
        stream_id=stream_id,
        seq=seq,
        t_arrival_ns=t_utc_ns,
        t_utc_ns=t_utc_ns,
        t_err_ns=1000,
        t_quality=TimeQuality.EXACT,
        dropped_before=dropped_before,
        exposure_us=exposure_us,
        gain=gain,
        mode=mode,
        roi=Roi(0, 0, width, height),
        adc_bits=adc_bits,
        flags=FrameFlag.SIMULATED,
    )
