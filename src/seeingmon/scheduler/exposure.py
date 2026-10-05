"""The adaptive exposure of the fast stream: the exposure that keeps the sky background at a target.

A 2 ms bin1 frame at gain 0 saturates its sky background while the Sun is still a few degrees up,
long before the sky hides Polaris (see `docs/research-notes.md`, "Polaris in a bright sky"). So the
fast stream shortens its exposure as the sky brightens. Before each fast period and each search
burst, the scheduler takes the sky background of the previous window or burst as a share of
saturation, and it picks the exposure that puts the background at
`[scheduler.fast] target_background_fraction`. The background grows in proportion to the exposure,
so the rule is one ratio, clamped between the profile's shortest exposure and
`[scheduler.fast] exposure_us`. In a dark sky the rule asks for more than the longest exposure, so
the stream runs at `exposure_us` all night.

**Between windows, never within one.** The exposure changes only where a stream starts. A fast
period holds whole analysis windows on one stream, so the scheduler adapts at the start of each
period, and every window has one exposure. The fast analyzer derives the exposure correction of
the seeing for each stream (`exposure_correction_factor` in the window record).

**The offset.** The background of a frame includes the camera's offset (the black level). The
profile does not know the offset, because the camera reports it at run time, so the scheduler
counts it as sky, as the daylight gate does (`seeingmon.scheduler.gates.read_sky`). With the
simulator's offset that is 0.7% of saturation in a bin1 frame, so the background settles that much
below the target, and the exposure about 2% short of it.
"""

from __future__ import annotations

import math


def background_fraction(
    background_dn: float, saturation_dn: float, offset_dn: float = 0.0
) -> float:
    """The sky background of a frame as a share of saturation: the level above the offset.

    `background_dn` and `offset_dn` are in the counts that the frame carries, and `saturation_dn`
    is the saturation level of its readout mode and gain in the same counts (the profile's level
    ignores the offset). A level at or below the offset gives 0.
    """
    if not saturation_dn > 0.0:
        raise ValueError("saturation_dn must be positive")
    return max(background_dn - offset_dn, 0.0) / saturation_dn


def adapted_exposure_us(
    exposure_us: float,
    fraction: float,
    *,
    target: float,
    shortest_us: int,
    longest_us: int,
) -> int:
    """The exposure that moves a background of `fraction` at `exposure_us` to `target`.

    The result lies between `shortest_us` and `longest_us`. A background of 0, which a dark sky
    above the offset gives, or a value that is not a number, gives `longest_us`.
    """
    if shortest_us > longest_us:
        raise ValueError("shortest_us must not exceed longest_us")
    if not (fraction > 0.0 and math.isfinite(fraction)):
        return longest_us
    wanted = exposure_us * target / fraction
    if not math.isfinite(wanted):
        return longest_us
    return int(min(longest_us, max(shortest_us, round(wanted))))
