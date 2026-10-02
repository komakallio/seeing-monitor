"""A base flat: the night sky compared with a panel flat, and the update that follows from it.

The tests come in two halves. The first half works on arrays: a mean sky made from a lens, and a
base flat that differs from it in one way at a time (a shadow, a vignetting, a tilt). Each rule of
the update shows alone there, and the tests run in milliseconds. The second half runs the whole
command on a synthetic night of the lens, against base flats that were *doctored*: the base lacks
a shadow that the sky has, or it has dust that left, or its vignetting or its tilt differs. The
true flat is known, so the update can be judged against it.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from seeingmon.clock import VirtualClock
from seeingmon.survey import flat_base as fb
from seeingmon.survey import flat_report as fr
from seeingmon.survey import flat_sky as fs
from seeingmon.survey.config import SurveyConfig
from tests.survey import flatfx as fx
from tests.survey import skyfx as sf
from tests.survey import test_flat_sky as t

SHAPE = t.SHAPE
FACTOR = t.FACTOR
SCALE = fx.REFERENCE_SCALE_ARCSEC_PX * t.SCALE_DOWN  # arcsec per pixel of the small sensor
CENTER_PX = ((SHAPE[1] - 1) / 2.0, (SHAPE[0] - 1) / 2.0)
CENTER = fr.binned_position(CENTER_PX, FACTOR)
HIGH_PASS = 20.0
RADII = (0.5, 1.0, 1.5, 2.0, 2.5)
GROUND = (0.0, 0.045)  # the gradient of the sky across the width and the height

# A shadow that the owner's lens does not have, 4.5% deep and 130 px across on the reference
# sensor (16 px on the small one).
NEW_DUST = fx.Spot(1500, 1000, 0.045, 130)
BIG_SHADOW = fx.OWNER_LENS.shadows[0]  # 3% and 71 px on the reference sensor


def lens(spec: fx.LensSpec) -> fx.FloatImage:
    return fx.lens_flat(spec, SHAPE, scale_down=t.SCALE_DOWN, edge_artifact_x=t.EDGE_X)


def with_shadows(spec: fx.LensSpec, shadows: Sequence[fx.Spot]) -> fx.LensSpec:
    return dataclasses.replace(spec, shadows=tuple(shadows))


def radius_deg_map(shape: tuple[int, int]) -> fx.FloatImage:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    return np.asarray(
        np.hypot(x - CENTER_PX[0], y - CENTER_PX[1]) * np.float32(SCALE / 3600.0), dtype=np.float32
    )


def with_radial(flat: fx.FloatImage, deltas: Sequence[float]) -> fx.FloatImage:
    """The flat times `1 + delta` at 0, 0.5, 1.0, 1.5, 2.0, 2.5, and 2.7 degrees (a polyline)."""
    knots = (0.0, *RADII, 2.7)
    values = np.interp(radius_deg_map(flat.shape), knots, (0.0, *deltas, deltas[-1]))
    return np.asarray(flat * (1.0 + values), dtype=np.float32)


def with_tilt(flat: fx.FloatImage, tilt: tuple[float, float]) -> fx.FloatImage:
    """The flat times a plane: a fraction across the width and across the height."""
    height, width = flat.shape
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    plane = 1.0 + tilt[0] * (x - CENTER_PX[0]) / width + tilt[1] * (y - CENTER_PX[1]) / height
    return np.asarray(flat * plane, dtype=np.float32)


def sky_of(
    truth: fx.FloatImage,
    *,
    gradient: tuple[float, float] = GROUND,
    noise: float = 0.0,
    seed: int = 3,
) -> fr.FloatArray:
    """The mean sky at the binned size: the lens, the gradient of the sky, and white noise."""
    mean = fr.block_mean(truth, FACTOR)
    rows, columns = mean.shape
    y, x = np.mgrid[0:rows, 0:columns].astype(np.float64)
    cx, cy = fr.binned_position(CENTER_PX, FACTOR)
    mean = mean * (1.0 + gradient[0] * (x - cx) / columns + gradient[1] * (y - cy) / rows)
    if noise:
        mean = mean * (1.0 + noise * np.random.default_rng(seed).standard_normal(mean.shape))
    return np.asarray(mean / np.median(mean), dtype=np.float64)


def compare(
    mean: fr.FloatArray,
    base: fx.FloatImage,
    *,
    valid: fr.BoolArray | None = None,
    noise_scale: fr.FloatArray | None = None,
    **options: float,
) -> fb.BaseComparison:
    return fb.compare_with_base(
        mean,
        fr.block_mean(base, FACTOR),
        valid=np.ones(mean.shape, dtype=np.bool_) if valid is None else valid,
        noise_scale=np.ones(mean.shape) if noise_scale is None else noise_scale,
        factor=FACTOR,
        sensor_shape=(base.shape[0], base.shape[1]),
        scale_arcsec_px=SCALE,
        center_xy=CENTER,
        high_pass_px=HIGH_PASS,
        edge_margin_px=12.0,
        **options,
    )


def updated(
    base: fx.FloatImage, comparison: fb.BaseComparison, *, trusted: bool = True
) -> tuple[fx.FloatImage, fb.Applied]:
    plan = fb.plan_update(comparison, trusted=trusted)
    return fb.apply_update(base, comparison, plan, factor=FACTOR), plan


def plane_of(ratio: fx.FloatImage) -> fr.Tilt:
    binned = fr.block_mean(ratio, FACTOR)
    return fr.fit_tilt(binned, center_xy=CENTER)


def near(shadow: fr.Shadow, spot: fx.Spot) -> bool:
    return (
        abs(shadow.x_px - spot.x / t.SCALE_DOWN) < 4
        and abs(shadow.y_px - spot.y / t.SCALE_DOWN) < 4
    )


OWNER = fx.OWNER_LENS
TRUTH = lens(OWNER)

# The base flats that differ from the lens in one way.
WITHOUT_BIG_SHADOW = lens(with_shadows(OWNER, OWNER.shadows[1:]))
WITH_DUST = lens(with_shadows(OWNER, (*OWNER.shadows, NEW_DUST)))
# A base that is brighter than the lens by this fraction at 0.5, 1.0, 1.5, 2.0, and 2.5 degrees: the
# vignetting of the lens is deeper than the base says.
DEEPER = (0.002, 0.005, 0.011, 0.017, 0.025)
SMALL = (0.001, 0.003, 0.005, 0.007, 0.008)  # every change under 1%


def expected_change_percent(deltas: Sequence[float]) -> list[float]:
    """The sky over a base that is brighter by `1 + delta`: `1 / (1 + delta) - 1`, in percent."""
    return [100.0 * (1.0 / (1.0 + d) - 1.0) for d in deltas]


# --- What the sky shows against the base ----------------------------------------------------


class TestWhatTheSkyShows:
    def test_a_base_that_is_still_true_shows_no_change_but_the_gradient_of_the_sky(self) -> None:
        c = compare(sky_of(TRUTH), TRUTH)
        changes = [p.change_percent for p in c.summary.profile]
        assert all(v is not None and abs(v) < 0.1 for v in changes)
        assert not c.radial_exceeds
        assert c.summary.shadows == ()
        assert c.bright == ()
        assert c.summary.edge_artifacts == ()
        assert c.summary.tilt.width_percent == pytest.approx(0.0, abs=0.1)
        assert c.summary.tilt.height_percent == pytest.approx(4.5, abs=0.1)

    def test_a_radial_change_shows_at_each_radius_against_the_middle(self) -> None:
        c = compare(sky_of(TRUTH), with_radial(TRUTH, DEEPER))
        measured = [p.change_percent for p in c.summary.profile[:5]]
        assert all(m is not None for m in measured)
        np.testing.assert_allclose(
            [m for m in measured if m is not None], expected_change_percent(DEEPER), atol=0.15
        )
        assert c.radial_exceeds
        assert c.peak is not None
        assert c.peak.radius_deg == 2.5
        assert c.peak.change_percent == pytest.approx(-2.31, abs=0.15)  # 2.44 less the middle

    def test_a_change_under_the_limit_does_not_exceed_it(self) -> None:
        c = compare(sky_of(TRUTH), with_radial(TRUTH, SMALL))
        assert c.peak is not None
        assert abs(c.peak.change_percent or 0.0) == pytest.approx(0.74, abs=0.1)
        assert not c.radial_exceeds

    def test_the_limit_is_a_setting(self) -> None:
        mean, base = sky_of(TRUTH), with_radial(TRUTH, SMALL)
        assert not compare(mean, base, radial_limit=0.01).radial_exceeds
        assert compare(mean, base, radial_limit=0.005).radial_exceeds

    def test_the_middle_that_the_changes_are_measured_against_is_a_disk_of_15_percent(self) -> None:
        c = compare(sky_of(TRUTH), TRUTH)
        assert c.reference_deg == pytest.approx(0.15 * 2.66, abs=0.02)

    def test_a_new_shadow_shows_with_its_place_depth_and_width(self) -> None:
        c = compare(sky_of(lens(with_shadows(OWNER, (*OWNER.shadows, NEW_DUST)))), TRUTH)
        assert len(c.summary.shadows) == 1
        shadow = c.summary.shadows[0]
        assert near(shadow, NEW_DUST)
        assert shadow.depth == pytest.approx(0.045, abs=0.006)
        assert shadow.width_px == pytest.approx(NEW_DUST.diameter / t.SCALE_DOWN, abs=3.0)
        assert not shadow.at_edge
        assert c.bright == ()

    def test_a_shadow_that_the_base_has_and_the_sky_has_not_shows_as_a_bright_patch(self) -> None:
        c = compare(sky_of(TRUTH), WITH_DUST)
        assert len(c.bright) == 1
        assert near(c.bright[0], NEW_DUST)
        assert c.bright[0].depth == pytest.approx(0.047, abs=0.006)  # 1 / (1 - 0.045) - 1
        assert c.summary.shadows == ()

    def test_a_bump_where_the_base_has_no_shadow_is_left_out_and_counted(self) -> None:
        # the residue of a star that the masks missed: a bump of 4% and 4 px, not dust that left
        y, x = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]].astype(np.float32)
        blob = 1.0 + 0.04 * np.exp(-((x - 300.0) ** 2 + (y - 100.0) ** 2) / (2 * 4.0**2))
        c = compare(sky_of(np.asarray(TRUTH * blob, dtype=np.float32)), TRUTH)
        assert c.bright == ()
        assert c.bright_unmatched == 1
        assert c.summary.shadows == ()
        new, plan = updated(TRUTH, c)
        assert plan.nothing
        np.testing.assert_allclose(new, TRUTH, rtol=1e-6)
        # the same bump over a shadow of the base counts: that is dust that left
        over = compare(sky_of(TRUTH), WITH_DUST)
        assert len(over.bright) == 1
        assert over.bright_unmatched == 0

    def test_a_new_dip_at_the_frame_edge_is_listed_apart(self) -> None:
        edge = fx.Spot(14, 1500, 0.04, 13)  # the edge artifacts keep their x: 6 px on this sensor
        sky = lens(dataclasses.replace(OWNER, edge_artifacts=(*OWNER.edge_artifacts, edge)))
        c = compare(sky_of(sky), TRUTH)
        assert c.summary.shadows == ()
        assert len(c.summary.edge_artifacts) == 1
        artifact = c.summary.edge_artifacts[0]
        assert artifact.at_edge
        assert artifact.y_px == pytest.approx(1500 / t.SCALE_DOWN, abs=3)

    def test_the_noise_raises_the_depth_of_the_search_and_no_shadow_comes_from_it(self) -> None:
        c = compare(sky_of(TRUTH, noise=0.03), TRUTH)
        assert c.summary.shadow_depth > 0.03  # 5 times 0.03 smoothed by a Gaussian of one pixel
        assert c.summary.shadows == ()
        assert c.bright == ()

    def test_a_pixel_that_the_sky_does_not_cover_shows_nothing(self) -> None:
        sky = lens(with_shadows(OWNER, (*OWNER.shadows, NEW_DUST)))
        mean = sky_of(sky)
        valid = np.ones(mean.shape, dtype=np.bool_)
        x, y = fr.binned_position((NEW_DUST.x / t.SCALE_DOWN, NEW_DUST.y / t.SCALE_DOWN), FACTOR)
        yy, xx = np.mgrid[0 : mean.shape[0], 0 : mean.shape[1]]
        valid[np.hypot(xx - x, yy - y) < 14.0] = False  # a mask around the dust, as Polaris has
        c = compare(mean, TRUTH, valid=valid)
        assert c.summary.shadows == ()

    def test_the_sky_and_the_base_must_have_one_shape(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            fb.compare_with_base(
                np.ones((10, 12)),
                np.ones((10, 11)),
                valid=np.ones((10, 12), dtype=np.bool_),
                noise_scale=np.ones((10, 12)),
                factor=FACTOR,
                sensor_shape=SHAPE,
                scale_arcsec_px=SCALE,
                center_xy=CENTER,
                high_pass_px=HIGH_PASS,
            )


# --- The update at the level of arrays ------------------------------------------------------


def distance_from(spot: fx.Spot, shape: tuple[int, int]) -> fx.FloatImage:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    return np.asarray(
        np.hypot(x - spot.x / t.SCALE_DOWN, y - spot.y / t.SCALE_DOWN), dtype=np.float32
    )


class TestTheUpdate:
    def test_it_takes_a_new_shadow_and_leaves_the_rest_of_the_base_as_it_was(self) -> None:
        sky = lens(with_shadows(OWNER, (*OWNER.shadows, NEW_DUST)))
        c = compare(sky_of(sky), TRUTH)
        new, plan = updated(TRUTH, c)
        assert plan == fb.Applied(radial=False, shadows=1, bright=0)
        ratio = new / TRUTH
        distance = distance_from(NEW_DUST, SHAPE)
        far = distance > 3 * NEW_DUST.diameter / t.SCALE_DOWN
        # outside the region, every pixel is the base pixel (the median scales the whole flat)
        assert float(np.abs(ratio[far] / np.median(ratio[far]) - 1.0).max()) < 1e-6
        # at the shadow, the new flat has the depth of the lens in the core. The light smoothing
        # blurs the edge: this shadow is 16 px wide and the blur is 2 px, so the new flat is
        # closer to the lens than the base is, but not exact (the real shadows are tens of pixels
        # wide at the same blur, and the error there is 0.3%)
        radius = NEW_DUST.diameter / t.SCALE_DOWN / 2.0
        core = distance < 0.4 * radius
        assert float(np.abs(new[core] / sky[core] - 1.0).mean()) < 0.006
        window = distance < 1.6 * radius
        error_new = fx.rms(new[window] / sky[window] - 1.0)
        error_base = fx.rms(TRUTH[window] / sky[window] - 1.0)
        assert error_base > 0.015  # the base has no shadow here
        assert error_new < 0.5 * error_base

    def test_it_takes_dust_that_left_out_of_the_base(self) -> None:
        c = compare(sky_of(TRUTH), WITH_DUST)
        new, plan = updated(WITH_DUST, c)
        assert plan == fb.Applied(radial=False, shadows=0, bright=1)
        radius = NEW_DUST.diameter / t.SCALE_DOWN / 2.0
        core = distance_from(NEW_DUST, SHAPE) < 0.4 * radius
        assert float(np.abs(new[core] / TRUTH[core] - 1.0).mean()) < 0.006
        assert float(np.abs(WITH_DUST[core] / TRUTH[core] - 1.0).mean()) > 0.03

    def test_it_takes_the_radial_change_as_a_whole_when_one_radius_exceeds_the_limit(self) -> None:
        base = with_radial(TRUTH, DEEPER)
        c = compare(sky_of(TRUTH), base)
        new, plan = updated(base, c)
        assert plan == fb.Applied(radial=True, shadows=0, bright=0)
        # the new flat has the vignetting of the lens: against the truth, its radial error is small
        ratio = fr.block_mean(new / TRUTH, FACTOR)
        profile = fr.azimuthal_profile(ratio, fr.radius_map(ratio.shape, CENTER))
        rings = profile.radius >= 10.0
        assert (
            float(np.abs(profile.value[rings] / np.median(profile.value[rings]) - 1.0).max())
            < 0.003
        )
        # the base was off by 2.4% at 2.5 degrees
        base_ratio = fr.block_mean(base / TRUTH, FACTOR)
        base_profile = fr.azimuthal_profile(base_ratio, fr.radius_map(base_ratio.shape, CENTER))
        assert float(base_profile.value.max() / base_profile.value.min()) > 1.02

    def test_it_leaves_a_radial_change_under_the_limit_out_even_where_one_radius_is_near_it(
        self,
    ) -> None:
        base = with_radial(TRUTH, SMALL)
        c = compare(sky_of(TRUTH), base)
        new, plan = updated(base, c)
        assert plan.nothing
        ratio = new / base
        assert float(np.abs(ratio / np.median(ratio) - 1.0).max()) < 1e-6  # the base, to the pixel

    def test_a_lower_limit_takes_the_smaller_change(self) -> None:
        base = with_radial(TRUTH, SMALL)
        c = compare(sky_of(TRUTH), base, radial_limit=0.004)
        new, plan = updated(base, c)
        assert plan.radial
        assert float(np.abs(new / base - 1.0).max()) > 0.004

    def test_an_update_that_the_caller_does_not_trust_applies_nothing_and_says_so(self) -> None:
        base = with_radial(WITHOUT_BIG_SHADOW, DEEPER)
        c = compare(sky_of(TRUTH), base)
        assert c.radial_exceeds
        assert len(c.summary.shadows) == 1
        new, plan = updated(base, c, trusted=False)
        assert plan == fb.Applied(radial=False, shadows=0, bright=0, blocked=True)
        assert plan.nothing
        ratio = new / base
        assert float(np.abs(ratio / np.median(ratio) - 1.0).max()) < 1e-6

    def test_it_never_applies_the_plane_even_with_a_change_of_the_radial_profile(self) -> None:
        # the base has a tilt that the lens has not (1% across the width), and a deeper vignetting
        base = with_tilt(with_radial(TRUTH, DEEPER), (0.01, -0.006))
        c = compare(sky_of(TRUTH), base)
        # the plane of the sky over the base: the gradient of the sky less the tilt of the base
        assert c.summary.tilt.width_percent == pytest.approx(-1.0, abs=0.15)
        assert c.summary.tilt.height_percent == pytest.approx(5.1, abs=0.15)
        new, plan = updated(base, c)
        assert plan.radial
        tilt = plane_of(new / base)
        assert abs(tilt.width_percent) < 0.05
        assert abs(tilt.height_percent) < 0.05

    def test_a_dip_at_the_frame_edge_is_not_applied(self) -> None:
        edge = fx.Spot(14, 1500, 0.04, 13)
        sky = lens(dataclasses.replace(OWNER, edge_artifacts=(*OWNER.edge_artifacts, edge)))
        c = compare(sky_of(sky), TRUTH)
        new, plan = updated(TRUTH, c)
        assert plan.nothing
        assert len(c.summary.edge_artifacts) == 1
        ratio = new / TRUTH
        assert float(np.abs(ratio / np.median(ratio) - 1.0).max()) < 1e-6

    def test_the_correction_has_a_soft_edge(self) -> None:
        sky = lens(with_shadows(OWNER, (*OWNER.shadows, NEW_DUST)))
        c = compare(sky_of(sky), TRUTH)
        correction = fb.correction_image(c, fb.plan_update(c))
        steps = max(
            float(np.abs(np.diff(correction, axis=0)).max()),
            float(np.abs(np.diff(correction, axis=1)).max()),
        )
        assert float(correction.min()) < 0.96  # the shadow is in it
        assert float(correction.max()) < 1.003
        assert steps < 0.02  # no step of the size of the shadow, 4.5%, between two binned pixels

    def test_nothing_that_exceeds_a_limit_gives_the_base_flat_again(self) -> None:
        c = compare(sky_of(TRUTH), TRUTH)
        new, plan = updated(TRUTH, c)
        assert plan.nothing
        np.testing.assert_allclose(new, TRUTH, rtol=1e-6)

    def test_the_new_flat_is_float32_positive_and_scaled_to_a_median_of_one(self) -> None:
        base = with_radial(WITH_DUST, DEEPER)
        c = compare(sky_of(TRUTH), base)
        new, _ = updated(base, c)
        assert new.dtype == np.float32
        assert new.shape == base.shape
        assert float(np.median(new)) == pytest.approx(1.0, abs=1e-6)
        assert float(new.min()) > 0.5
        assert bool(np.isfinite(new).all())

    def test_a_sensor_that_the_blocks_do_not_fill_still_works(self) -> None:
        odd = np.ascontiguousarray(WITH_DUST[:351, :511])
        mean = sky_of(np.ascontiguousarray(TRUTH[:351, :511]))
        assert mean.shape == (175, 255)
        c = fb.compare_with_base(
            mean,
            fr.block_mean(odd, FACTOR),
            valid=np.ones(mean.shape, dtype=np.bool_),
            noise_scale=np.ones(mean.shape),
            factor=FACTOR,
            sensor_shape=(351, 511),
            scale_arcsec_px=SCALE,
            center_xy=CENTER,
            high_pass_px=HIGH_PASS,
            edge_margin_px=12.0,
        )
        new = fb.apply_update(odd, c, fb.plan_update(c), factor=FACTOR)
        assert new.shape == (351, 511)
        assert len(c.bright) == 1


# --- The words of the report ----------------------------------------------------------------


class TestTheWords:
    def test_what_the_update_applies_reads_as_a_list(self) -> None:
        lines = fb.update_lines(fb.Applied(radial=True, shadows=2, bright=1), written=True)
        assert lines == [
            "Update: applied the radial change, 2 new shadows, and 1 bright patch. The plane and "
            "every smaller change stay as the base flat has them."
        ]
        lines = fb.update_lines(fb.Applied(radial=False, shadows=1, bright=2), written=False)
        assert lines == [
            "An update would apply 1 new shadow and 2 bright patches. It would leave the plane "
            "and every smaller change as the base flat has them."
        ]

    def test_an_update_that_changes_nothing_says_so(self) -> None:
        none = fb.Applied(radial=False, shadows=0, bright=0)
        assert fb.update_lines(none, written=True) == [
            "Update: applied nothing, because no change exceeds its limit. The new flat equals "
            "the base flat."
        ]
        assert fb.update_lines(none, written=False) == [
            "An update would change nothing: no change exceeds its limit."
        ]

    def test_a_blocked_update_says_why(self) -> None:
        blocked = fb.Applied(radial=False, shadows=0, bright=0, blocked=True)
        assert fb.update_lines(blocked, written=True) == [
            "Update: applied nothing, because the ring around Polaris (see the warning) means "
            "that the mean sky holds the halo of Polaris, which would go into the flat. The new "
            "flat equals the base flat."
        ]
        assert fb.update_lines(blocked, written=False) == [
            "An update would apply nothing, because the ring around Polaris (see the warning) "
            "means that the mean sky holds the halo of Polaris, which would go into the flat."
        ]

    def test_the_comparison_names_the_base_the_middle_the_limit_and_the_plane(self) -> None:
        c = compare(sky_of(TRUTH), with_radial(TRUTH, DEEPER))
        text = "\n".join(fb.comparison_lines(c, base_name="panel.npy", edge_margin_px=20.0))
        for expected in (
            "Base flat: panel.npy. The mean sky is divided by it.",
            "Change of the vignetting against the base flat, at each radius from the center "
            f"(against the disk within {c.reference_deg:.2f} degrees of it):",
            "Radial profile: the largest change is -2.3% at 2.5 degrees, over the limit of 1%.",
            "Plane: the mean sky over the base flat has 0.00% across the width, +4.50% across the "
            "height (a positive value means that it rises toward the right edge or the bottom "
            "edge).",
            "so the tilt comes from the base flat.",
            "New shadows deeper than 1%: none",
            "Patches brighter than the base flat by more than 1%: none",
            "Edge artifacts deeper than 1% (center within 20 px of an edge): none",
        ):
            assert expected in text, expected
        assert "  2.5 degrees: -2.31%" in text

    def test_a_change_within_the_limit_reads_within(self) -> None:
        c = compare(sky_of(TRUTH), with_radial(TRUTH, SMALL))
        text = "\n".join(fb.comparison_lines(c, base_name=None, edge_margin_px=20.0))
        assert "within the limit of 1%." in text
        assert "Base flat:" not in text

    def test_a_shadow_and_a_patch_are_listed_with_their_numbers(self) -> None:
        sky = lens(with_shadows(OWNER, (*OWNER.shadows, NEW_DUST)))
        text = "\n".join(
            fb.comparison_lines(compare(sky_of(sky), TRUTH), base_name=None, edge_margin_px=20.0)
        )
        assert "New shadows deeper than 1%: 1\n  x " in text
        assert re.search(r": depth \d\.\d%, width \d+ px", text)
        text = "\n".join(
            fb.comparison_lines(
                compare(sky_of(TRUTH), WITH_DUST), base_name=None, edge_margin_px=20.0
            )
        )
        assert "Patches brighter than the base flat by more than 1%: 1\n  x " in text
        assert re.search(r": excess \d\.\d%, width \d+ px", text)

    def test_the_bumps_that_lie_over_no_shadow_of_the_base_are_counted_in_the_report(self) -> None:
        y, x = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]].astype(np.float32)
        blob = 1.0 + 0.04 * np.exp(-((x - 300.0) ** 2 + (y - 100.0) ** 2) / (2 * 4.0**2))
        c = compare(sky_of(np.asarray(TRUTH * blob, dtype=np.float32)), TRUTH)
        text = "\n".join(fb.comparison_lines(c, base_name=None, edge_margin_px=20.0))
        assert "Patches brighter than the base flat by more than 1%: none" in text
        assert (
            "Ignored: 1 bright patch where the base flat has no shadow (the residue of a star "
            "that the masks missed, and not dust that left)."
        ) in text

    def test_a_bright_edge_patch_is_listed_apart(self) -> None:
        edge = fx.Spot(14, 1500, 0.04, 13)
        base = lens(dataclasses.replace(OWNER, edge_artifacts=(*OWNER.edge_artifacts, edge)))
        text = "\n".join(
            fb.comparison_lines(compare(sky_of(TRUTH), base), base_name=None, edge_margin_px=12.0)
        )
        assert "Bright edge artifacts by more than 1% (center within 12 px of an edge): 1" in text


# --- The whole command on a night -----------------------------------------------------------


@dataclass(frozen=True)
class Kept:
    """A night of the owner's lens, with the accumulator that its frames left."""

    night: t.Night
    accumulator: Path
    sky_only: fs.SkyResult | None = None


@pytest.fixture(scope="module")
def kept(tmp_path_factory: pytest.TempPathFactory) -> Kept:
    root = tmp_path_factory.mktemp("based")
    night = t.make_night(root, fx.OWNER_LENS)
    accumulator = root / "acc.npz"
    return Kept(night, accumulator, t.build(night, accumulator=accumulator))


def against(
    kept_night: Kept,
    base: fx.FloatImage,
    *,
    update: bool = False,
    options: fs.BuildOptions = t.OPTIONS,
) -> fs.SkyResult:
    """Build from the accumulator alone, against a base flat. No frame is read again."""
    night = kept_night.night
    return fs.build_sky_flat(
        None,
        profile=night.profile,
        survey=SurveyConfig(),
        library=night.library,
        site=sf.TEST_SITE,
        accumulator_path=kept_night.accumulator,
        options=options,
        clock=VirtualClock(),
        base_flat=base,
        update=update,
    )


# What the lens did since the panel flat: it lost the big shadow from the base, and its
# vignetting is deeper than the base says, and the base has a tilt that the lens has not.
CHANGED = with_tilt(with_radial(WITHOUT_BIG_SHADOW, DEEPER), (0.01, 0.0))


class TestTheCommandAgainstABaseFlat:
    def test_the_changes_since_the_panel_flat_show_in_the_report(self, kept: Kept) -> None:
        result = against(kept, CHANGED)
        assert result.base is not None
        profile = result.base.summary.profile[:5]
        np.testing.assert_allclose(
            [p.change_percent for p in profile if p.change_percent is not None],
            expected_change_percent(DEEPER),
            atol=0.7,  # the rings of the sky add 0.15 to 0.5%, and 0.5 measured at 1 degree
        )
        assert result.base.radial_exceeds
        shadows = result.base.summary.shadows
        assert len(shadows) == 1
        assert near(shadows[0], BIG_SHADOW)
        assert shadows[0].depth == pytest.approx(BIG_SHADOW.depth, abs=0.012)
        # the plane of the sky over the base: the gradient of the sky, less the tilt of the base
        tilt = result.base.summary.tilt
        assert tilt.width_percent == pytest.approx(-1.0, abs=0.35)
        assert tilt.height_percent == pytest.approx(4.5, abs=0.35)

    def test_without_update_the_command_writes_nothing_and_says_what_it_would_apply(
        self, kept: Kept
    ) -> None:
        result = against(kept, CHANGED)
        assert not result.updated
        assert result.applied is not None
        assert result.applied.radial
        assert result.applied.shadows == 1
        text = "\n".join(fs.format_sky_report(result, base_name="panel.npy"))
        assert "Flat from the night sky, compared with a base flat." in text
        assert "An update would apply the radial change and 1 new shadow." in text
        assert "Nothing written. Add --update and --out to write the new flat." in text
        assert "Wrote" not in text
        assert "Tilt: not determined" not in text  # the tilt comes from the base

    def test_the_update_applies_the_changes_and_keeps_the_tilt_of_the_base(
        self, kept: Kept
    ) -> None:
        result = against(kept, CHANGED, update=True)
        assert result.updated
        assert result.applied == fb.Applied(radial=True, shadows=1, bright=0)
        new = result.flat
        assert new.dtype == np.float32
        assert new.shape == SHAPE
        assert float(np.median(new)) == pytest.approx(1.0, abs=1e-6)
        # the plane of the new flat over the base is nil, the tilt of the base stays
        tilt = plane_of(new / CHANGED)
        assert abs(tilt.width_percent) < 0.1
        assert abs(tilt.height_percent) < 0.1
        text = "\n".join(fs.format_sky_report(result, name="new.npy", base_name="panel.npy"))
        assert "Update: applied the radial change and 1 new shadow." in text
        assert "Wrote new.npy, the base flat with these changes. Set flat_file" in text

    def test_the_update_is_closer_to_the_lens_than_the_base_and_than_the_sky_alone(
        self, kept: Kept
    ) -> None:
        truth = kept.night.truth
        # the base has the tilt of the lens, so that only the radial change and the shadow differ
        base = with_radial(WITHOUT_BIG_SHADOW, DEEPER)
        result = against(kept, base, update=True)
        on_base = t.errors_of(base, truth)
        update = t.errors_of(result.flat, truth)
        assert kept.sky_only is not None
        alone = t.errors_of(kept.sky_only.flat, truth)
        # measured: the base 1.37% radial and 1.25% total, the sky alone 0.19% radial, 0.45% fine,
        # and 0.60% total, the update 0.18% radial, 0.04% fine, and 0.16% total
        assert on_base.radial > 1.0  # the vignetting is off by up to 2.4%
        assert update.radial < 0.35
        assert update.fine < 0.15  # the base has the dust, and the sky takes the new shadow
        assert update.total < 0.3
        # the tilt that the sky cannot give comes from the base
        assert update.total < 0.5 * alone.total
        assert update.total < 0.25 * on_base.total

    def test_dust_that_left_since_the_panel_flat_is_taken_out_of_the_flat(self, kept: Kept) -> None:
        result = against(kept, WITH_DUST, update=True)
        assert result.applied == fb.Applied(radial=False, shadows=0, bright=1)
        core = distance_from(NEW_DUST, SHAPE) < 0.2 * NEW_DUST.diameter / t.SCALE_DOWN
        truth = kept.night.truth
        error_new = float(np.abs(result.flat[core] / truth[core] - 1.0).mean())
        error_base = float(np.abs(WITH_DUST[core] / truth[core] - 1.0).mean())
        assert error_base > 0.04  # the base has the dust of 4.5%
        assert error_new < 0.3 * error_base  # 0.9% measured
        text = "\n".join(fs.format_sky_report(result))
        assert "Patches brighter than the base flat by more than" in text
        assert "excess" in text

    def test_a_base_that_is_still_true_changes_nothing(self, kept: Kept) -> None:
        result = against(kept, kept.night.truth, update=True)
        assert result.base is not None
        assert not result.base.radial_exceeds  # the rings of the sky stay under 1%
        assert result.applied is not None
        assert result.applied.nothing
        np.testing.assert_allclose(result.flat, kept.night.truth, rtol=1e-5)
        text = "\n".join(fs.format_sky_report(result, name="new.npy"))
        assert "Update: applied nothing, because no change exceeds its limit." in text

    def test_the_limit_of_the_radial_change_is_an_option(self, kept: Kept) -> None:
        options = dataclasses.replace(t.OPTIONS, radial_limit=0.001)
        result = against(kept, kept.night.truth, update=True, options=options)
        assert result.applied is not None
        assert result.applied.radial
        assert not np.allclose(result.flat, kept.night.truth, rtol=1e-5)
        with pytest.raises(ValueError, match="radial limit"):
            dataclasses.replace(t.OPTIONS, radial_limit=0.0)

    def test_the_report_names_the_base_by_its_file_and_no_path(self, kept: Kept) -> None:
        text = "\n".join(fs.format_sky_report(against(kept, CHANGED), base_name="panel.npy"))
        assert "Base flat: panel.npy." in text
        for fragment in ("\\", "/tmp", "Users", ".npz"):
            assert fragment not in text

    def test_a_base_of_another_size_is_refused_before_any_frame_is_read(self, kept: Kept) -> None:
        with pytest.raises(fs.SkyFlatError, match=r"base flat has 16 x 12 pixels.*512 x 352"):
            against(kept, np.ones((12, 16), dtype=np.float32))

    def test_an_update_needs_a_base_flat(self, kept: Kept) -> None:
        night = kept.night
        with pytest.raises(fs.SkyFlatError, match="an update needs a base flat"):
            fs.build_sky_flat(
                None,
                profile=night.profile,
                survey=SurveyConfig(),
                library=night.library,
                site=sf.TEST_SITE,
                accumulator_path=kept.accumulator,
                options=t.OPTIONS,
                update=True,
            )

    def test_without_a_base_the_result_has_no_comparison(self, kept: Kept) -> None:
        alone = kept.sky_only
        assert alone is not None
        assert alone.base is None
        assert alone.applied is None
        assert not alone.updated
        text = "\n".join(fs.format_sky_report(alone, name="flat.npy"))
        assert text.startswith("Flat from the night sky.\nFrames: ")
        assert "Wrote flat.npy. Set flat_file" in text


# --- The ring of Polaris keeps the radial change out ----------------------------------------


@pytest.fixture(scope="module")
def ringed(tmp_path_factory: pytest.TempPathFactory) -> Kept:
    root = tmp_path_factory.mktemp("ringed")
    night = t.make_night(root, fx.OWNER_LENS, frames=16, halo=sf.Halo(4.0, 8.0))
    accumulator = root / "acc.npz"
    options = dataclasses.replace(t.OPTIONS, polaris_mask_px=8.0)  # too small for this halo
    return Kept(night, accumulator, t.build(night, accumulator=accumulator, options=options))


class TestARingAroundPolaris:
    def test_a_mask_that_leaves_a_ring_blocks_the_update_whole(self, ringed: Kept) -> None:
        options = dataclasses.replace(t.OPTIONS, polaris_mask_px=8.0)
        base = with_radial(WITHOUT_BIG_SHADOW, DEEPER)
        result = against(ringed, base, update=True, options=options)
        assert result.ring.bump is not None
        assert result.ring.bump > 0.003
        assert result.base is not None
        assert result.base.radial_exceeds  # the change is there
        assert result.applied == fb.Applied(radial=False, shadows=0, bright=0, blocked=True)
        assert any("the mask around Polaris is too small" in w for w in result.warnings)
        text = "\n".join(fs.format_sky_report(result, name="new.npy"))
        assert "Update: applied nothing, because the ring around Polaris" in text
        ratio = result.flat / base
        assert float(np.abs(ratio / np.median(ratio) - 1.0).max()) < 1e-5

    def test_a_mask_that_is_large_enough_lets_the_changes_in(self, tmp_path: Path) -> None:
        night = t.make_night(tmp_path, fx.OWNER_LENS, frames=16, halo=sf.Halo(4.0, 8.0))
        accumulator = tmp_path / "acc.npz"
        t.build(night, accumulator=accumulator)  # the mask of 49 px of OPTIONS
        base = with_radial(night.truth, DEEPER)
        result = against(Kept(night, accumulator), base, update=True)
        assert result.applied is not None
        assert result.applied.radial
        assert not result.applied.blocked


def test_the_doctored_bases_differ_from_the_lens_as_they_say() -> None:
    """A guard for the tests above: the radial change of the base is what `DEEPER` says."""
    binned = fr.block_mean(with_radial(TRUTH, DEEPER) / TRUTH, FACTOR)
    profile = fr.azimuthal_profile(binned, fr.radius_map(binned.shape, CENTER))
    assert float(profile.value.max()) == pytest.approx(1.027, abs=0.003)
    assert profile.value[0] == pytest.approx(1.0, abs=0.001)
