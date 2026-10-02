"""Compare the night sky with a base flat, and update it: `seeingmon flat build --base-flat`.

**The base.** A flat that you took with a panel (`seeingmon flat make`) holds the tilt of the optics
and the sensor, which the night sky cannot give. The night sky holds what a panel flat cannot keep
current: the vignetting and the dust shadows of the lens as they are now. `--base-flat` combines
the two, and it keeps the tilt of the base.

**What the sky says about the base.** The mean sky of the accumulated frames holds the flat times
the mean sky. It holds the gradient of the sky that is fixed to the ground too. The mean sky over
the base flat is therefore a smooth image with a gradient across it where the flat has not changed,
and the decomposition of `seeingmon.survey.flat_report` splits it into the same three parts as
before:

- the *radial change*, the azimuthal mean about the optical center against its mean over the disk
  at the middle (within 15% of the way to the corners), which says that the vignetting is not what
  the base says;
- the *fine changes*, the dips of the fine part that are *new shadows* (dust that came), and the
  bumps that are *bright patches* (dust that left, so that the base holds a shadow that is gone).
  A bump counts only over a shadow of the base flat, because the mean of a few frames holds the
  residue of the stars that the masks missed, and those are bumps too;
- the *plane*, which adds the gradient of the sky to any change of the tilt of the flat. The sky
  cannot tell them apart, so the report gives the plane and the update never applies it.

**The update.** The new flat is the base flat times a correction that holds only the changes that
exceed their limits:

- the radial change, as the whole radial part, when it exceeds 1% at any of the five radii of the
  report (`flat_report.RADII_DEG`), and not at all otherwise, because the rings of the sky itself
  add 0.15 to 0.5% to a radial profile;
- the new shadows and the bright patches that the search for shadows finds (the search needs 1%, or
  5 times the noise when that is more), each over its own region with a soft edge, and
  not the dips at the frame edge, which the report lists apart.

Everything else stays as the base has it, down to the pixel, so the new flat keeps the tilt, the
pattern of the pixels, and the shadows that did not change. It is scaled to a median of 1.

**When the sky is not to be trusted.** The halo of Polaris leaves a ring in the radial part of the
mean sky and arcs in its fine part when the mask around Polaris is too small. The ring check of
`seeingmon.survey.flat_sky` finds it, and then the update applies nothing: the new flat equals the
base flat, and the report says why.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.survey import _scipy, flat_report

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

# The radial change that the update applies, as a fraction, at any of the five radii of the report.
RADIAL_LIMIT = 0.01
# The width of the soft edge of a region that the update corrects, in binned pixels.
FEATHER_SIGMA = 1.5
# The radial change is measured against the disk within this share of the way to the corners. The
# sky does not turn at the center, so a small disk would carry a structure of the sky that the
# rotation does not average, and would shift the change at every radius with it.
REFERENCE_FRACTION = 0.15
# A bright patch counts as dust that left only where the base flat has a shadow at least this share
# as deep as the patch is bright.
BASE_SHADOW_SHARE = 0.5


@dataclass(frozen=True, slots=True)
class BaseComparison:
    """The mean sky over the base flat, and what changed.

    `summary` describes the quotient in the terms of a flat: its `profile` is the radial change at
    each radius (against the center), its `tilt` is the plane, and its `shadows` and
    `edge_artifacts` are the new shadows. `bright` and `bright_edge` are the patches that are
    brighter than the base. `peak` is the radius of the five where the radial change is largest.
    `radial_exceeds` says that the radial change is over `radial_limit` (a fraction) at one of the
    five radii. `reference_deg` is the radius of the disk at the middle that the changes are
    measured against, and `center_value` its mean. `bright_unmatched` counts the bumps that lie
    over no shadow of the base flat, which the comparison leaves out.
    """

    summary: flat_report.FlatSummary
    bright: tuple[flat_report.Shadow, ...]
    bright_edge: tuple[flat_report.Shadow, ...]
    radial_limit: float
    peak: flat_report.ProfilePoint | None
    radial_exceeds: bool
    parts: flat_report.Decomposition
    center_value: float
    reference_deg: float
    dark_dips: flat_report.Dips
    light_dips: flat_report.Dips
    smooth_sigma: float
    bright_unmatched: int = 0


@dataclass(frozen=True, slots=True)
class Applied:
    """What an update applied (or would apply): the radial change, and the number of regions.

    `blocked` says that the update applies nothing, because the caller does not trust the mean
    sky (the halo of Polaris is in it).
    """

    radial: bool
    shadows: int
    bright: int
    blocked: bool = False

    @property
    def nothing(self) -> bool:
        return not (self.radial or self.shadows or self.bright)


def over_shadows(
    light: flat_report.Dips, base_fine: FloatArray, smooth_sigma: float
) -> tuple[flat_report.Dips, int]:
    """Keep the bright patches that lie over a shadow of the base flat, and count the others.

    A patch stays when the deepest shadow of the base within its region is at least half as deep
    as the patch is bright. `base_fine` is the fine part of the base flat.
    """
    if not light.shadows:
        return light, 0
    deficit = 1.0 - _scipy.gaussian_filter(base_fine, smooth_sigma)
    inside = light.labels > 0
    deepest = np.zeros(int(light.labels.max()) + 1)
    np.maximum.at(deepest, light.labels[inside], deficit[inside])
    kept = [
        (shadow, region)
        for shadow, region in zip(light.shadows, light.ids, strict=True)
        if deepest[region] >= BASE_SHADOW_SHARE * shadow.depth
    ]
    matched = flat_report.Dips(
        shadows=tuple(shadow for shadow, _ in kept),
        labels=light.labels,
        ids=tuple(region for _, region in kept),
    )
    return matched, len(light.shadows) - len(kept)


def compare_with_base(
    mean: FloatArray,
    base_binned: FloatArray,
    *,
    valid: BoolArray,
    noise_scale: FloatArray,
    factor: int,
    sensor_shape: tuple[int, int],
    scale_arcsec_px: float,
    center_xy: tuple[float, float],
    high_pass_px: float,
    radial_limit: float = RADIAL_LIMIT,
    edge_margin_px: float = flat_report.EDGE_MARGIN_PX,
    smooth_sigma: float = 1.0,
) -> BaseComparison:
    """Divide the mean sky by the base flat and describe the quotient.

    `mean` and `base_binned` are binned by `factor` and have the same shape. `valid` marks the
    pixels that the sky covers, `noise_scale` is the noise of each binned pixel over the typical
    one, and `center_xy` is the optical center in binned pixels.
    """
    if mean.shape != base_binned.shape:
        raise ValueError("the sky average and the base flat must have the same shape")
    usable = valid & np.isfinite(base_binned) & (base_binned > 0)
    ratio = np.where(usable, mean / np.where(usable, base_binned, 1.0), 1.0)
    parts = flat_report.decompose(
        ratio, center_xy=center_xy, high_pass_px=high_pass_px, valid=usable
    )
    depth, depth_map = flat_report.shadow_search_depth(
        parts.fine, smooth_sigma=smooth_sigma, noise_scale=noise_scale, valid=usable
    )

    def search(fine: FloatArray) -> flat_report.Dips:
        return flat_report.locate_dips(
            fine,
            factor=factor,
            sensor_shape=sensor_shape,
            min_depth=depth_map,
            edge_margin_px=edge_margin_px,
            smooth_sigma=smooth_sigma,
            valid=usable,
        )

    dark = search(parts.fine)
    light = search(2.0 - parts.fine)  # a bump of the fine part is a dip of its mirror image
    base_fine = flat_report.decompose(
        np.where(np.isfinite(base_binned) & (base_binned > 0), base_binned, 1.0),
        center_xy=center_xy,
        high_pass_px=high_pass_px,
    ).fine
    light, unmatched = over_shadows(light, base_fine, smooth_sigma)
    profile = flat_report.vignetting_profile(
        ratio,
        factor=factor,
        scale_arcsec_px=scale_arcsec_px,
        center_xy=center_xy,
        valid=usable,
        center_fraction=REFERENCE_FRACTION,
    )
    center_value, reference_bins = flat_report.center_level(
        ratio, center_xy=center_xy, valid=usable, center_fraction=REFERENCE_FRACTION
    )
    summary = flat_report.FlatSummary(
        profile=profile,
        tilt=flat_report.tilt_after_radial(ratio, parts, center_xy=center_xy, valid=usable),
        shadows=tuple(shadow for shadow in dark.shadows if not shadow.at_edge),
        edge_artifacts=tuple(shadow for shadow in dark.shadows if shadow.at_edge),
        shadow_depth=depth,
    )
    radii = [point for point in profile if not point.corner and point.change_percent is not None]
    peak = max(radii, key=lambda point: abs(point.change_percent or 0.0), default=None)
    exceeds = peak is not None and abs(peak.change_percent or 0.0) > 100.0 * radial_limit
    return BaseComparison(
        summary=summary,
        bright=tuple(shadow for shadow in light.shadows if not shadow.at_edge),
        bright_edge=tuple(shadow for shadow in light.shadows if shadow.at_edge),
        radial_limit=radial_limit,
        peak=peak,
        radial_exceeds=exceeds,
        parts=parts,
        center_value=center_value,
        reference_deg=reference_bins * factor * scale_arcsec_px / 3600.0,
        dark_dips=dark,
        light_dips=light,
        smooth_sigma=smooth_sigma,
        bright_unmatched=unmatched,
    )


def correction_image(comparison: BaseComparison, plan: Applied) -> FloatArray:
    """The correction of the update at the binned size: 1 where nothing changes enough.

    It is the radial part of the quotient over its value at the center when the plan has the
    radial change, times the lightly smoothed fine part over the regions of the new shadows and
    of the bright patches that the plan has, with a soft edge. The plane is never in it.
    """
    parts = comparison.parts
    radial: float | FloatArray = parts.radial_map / comparison.center_value if plan.radial else 1.0
    region = np.zeros(parts.fine.shape, dtype=np.bool_)
    if plan.shadows:
        region |= comparison.dark_dips.mask()
    if plan.bright:
        region |= comparison.light_dips.mask()
    fine = np.ones(parts.fine.shape, dtype=np.float64)
    if region.any():
        smooth = _scipy.gaussian_filter(parts.fine, comparison.smooth_sigma)
        weight = np.clip(_scipy.gaussian_filter(region.astype(np.float64), FEATHER_SIGMA), 0.0, 1.0)
        fine = 1.0 + (smooth - 1.0) * weight
    return np.asarray(radial * fine, dtype=np.float64)


def plan_update(comparison: BaseComparison, *, trusted: bool = True) -> Applied:
    """What an update applies: the radial change when it exceeds its limit, and every region.

    `trusted` false says that the mean sky holds something that does not belong to the flat (the
    halo of Polaris), so the update applies nothing.
    """
    if not trusted:
        return Applied(radial=False, shadows=0, bright=0, blocked=True)
    return Applied(
        radial=comparison.radial_exceeds,
        shadows=len(comparison.summary.shadows),
        bright=len(comparison.bright),
    )


def apply_update(
    base: Float32Array, comparison: BaseComparison, plan: Applied, *, factor: int
) -> Float32Array:
    """The base flat times the correction of `plan`, scaled to a median of 1."""
    correction = correction_image(comparison, plan)
    spread = flat_report.upsample_bilinear(correction, factor, (base.shape[0], base.shape[1]))
    new = np.asarray(base * spread, dtype=np.float32)
    new /= np.float32(np.median(new))
    return new


# --- Text -----------------------------------------------------------------------------------


def _join(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _count(number: int, singular: str, plural: str) -> str:
    return f"{number} {singular if number == 1 else plural}"


def comparison_lines(
    comparison: BaseComparison, *, base_name: str | None, edge_margin_px: float
) -> list[str]:
    """The lines that say what the sky shows against the base flat. They name no path."""
    summary = comparison.summary
    limit_percent = 100.0 * comparison.radial_limit
    lines: list[str] = []
    if base_name is not None:
        lines.append(f"Base flat: {base_name}. The mean sky is divided by it.")
    lines.extend(
        flat_report.profile_lines(
            summary.profile,
            title=(
                "Change of the vignetting against the base flat, at each radius from the center "
                f"(against the disk within {comparison.reference_deg:.2f} degrees of it):"
            ),
        )
    )
    peak = comparison.peak
    if peak is None or peak.change_percent is None:
        lines.append("Radial profile: no radius of the report lies inside the frame.")
    else:
        relation = "over" if comparison.radial_exceeds else "within"
        change = flat_report.format_percent(peak.change_percent)
        lines.append(
            f"Radial profile: the largest change is {change} at {peak.radius_deg:.1f} degrees, "
            f"{relation} the limit of {limit_percent:g}%."
        )
    lines.append(
        f"Plane: the mean sky over the base flat has {flat_report.tilt_text(summary.tilt)} (a "
        "positive value means that it rises toward the right edge or the bottom edge). That is "
        "the gradient of the sky plus any change of the tilt of the flat, and the sky cannot tell "
        "them apart, so the tilt comes from the base flat."
    )
    depth = summary.shadow_depth
    lines.extend(flat_report.shadow_lines(summary.shadows, depth=depth, title="New shadows"))
    lines.extend(
        flat_report.shadow_lines(
            comparison.bright,
            depth=depth,
            title="Patches brighter than the base flat",
            relation="by more than",
            measure="excess",
        )
    )
    if comparison.bright_unmatched:
        lines.append(
            f"Ignored: {_count(comparison.bright_unmatched, 'bright patch', 'bright patches')} "
            "where the base flat has no shadow (the residue of a star that the masks missed, "
            "and not dust that left)."
        )
    lines.extend(
        flat_report.edge_artifact_lines(
            summary.edge_artifacts, margin_px=edge_margin_px, depth=depth
        )
    )
    if comparison.bright_edge:
        lines.extend(
            flat_report.shadow_lines(
                comparison.bright_edge,
                depth=depth,
                title="Bright edge artifacts",
                relation="by more than",
                measure="excess",
                qualifier=f"center within {edge_margin_px:.0f} px of an edge",
            )
        )
    return lines


def update_lines(applied: Applied, *, written: bool) -> list[str]:
    """The lines that say what the update applied, or would apply when `written` is false."""
    if applied.blocked:
        reason = (
            "the ring around Polaris (see the warning) means that the mean sky holds the halo of "
            "Polaris, which would go into the flat"
        )
        if written:
            return [
                f"Update: applied nothing, because {reason}. The new flat equals the base flat."
            ]
        return [f"An update would apply nothing, because {reason}."]
    items: list[str] = []
    if applied.radial:
        items.append("the radial change")
    if applied.shadows:
        items.append(_count(applied.shadows, "new shadow", "new shadows"))
    if applied.bright:
        items.append(_count(applied.bright, "bright patch", "bright patches"))
    if written:
        if items:
            return [
                f"Update: applied {_join(items)}. The plane and every smaller change stay as the "
                "base flat has them."
            ]
        return [
            "Update: applied nothing, because no change exceeds its limit. The new flat equals "
            "the base flat."
        ]
    if items:
        return [
            f"An update would apply {_join(items)}. It would leave the plane and every smaller "
            "change as the base flat has them."
        ]
    return ["An update would change nothing: no change exceeds its limit."]
