"""The measures of a flat field: binning, the radial part, the tilt, the shadows, and the text."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.survey import flat_report as fr

Image = npt.NDArray[np.float64]

FACTOR = 4
SHAPE = (176, 259)  # binned pixels, so the sensor is 704 x 1036
SENSOR_SHAPE = (SHAPE[0] * FACTOR, SHAPE[1] * FACTOR)
CENTER = ((SHAPE[1] - 1) / 2.0, (SHAPE[0] - 1) / 2.0)
SCALE = 15.28  # arcsec per sensor pixel, so that 4.4 degrees fill the width


def grid() -> tuple[Image, Image]:
    y, x = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]].astype(np.float64)
    return x, y


def dip(x: Image, y: Image, *, at: tuple[float, float], depth: float, radius: float) -> Image:
    """A dip with a soft edge, as a factor near 1."""
    distance = np.hypot(x - at[0], y - at[1])
    return np.asarray(1.0 - depth * 0.5 * (1.0 - np.tanh((distance - radius) / 0.8)), np.float64)


class TestBinning:
    def test_a_block_mean_averages_whole_blocks_and_drops_the_rest(self) -> None:
        image = np.arange(7 * 9, dtype=np.float64).reshape(7, 9)
        binned = fr.block_mean(image, 3)
        assert binned.shape == (2, 3)
        assert binned[0, 0] == pytest.approx(image[:3, :3].mean())
        assert binned[1, 2] == pytest.approx(image[3:6, 6:9].mean())

    def test_a_block_sum_of_integers_is_exact_and_does_not_overflow(self) -> None:
        image = np.full((8, 8), 65535, dtype=np.uint16)
        assert fr.block_sum(image, 4)[0, 0] == 16 * 65535

    def test_a_factor_of_zero_and_an_image_smaller_than_a_block_are_errors(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            fr.block_sum(np.ones((4, 4)), 0)
        with pytest.raises(ValueError, match="smaller than one block"):
            fr.block_sum(np.ones((3, 8)), 4)

    def test_the_binned_and_the_sensor_positions_invert_each_other(self) -> None:
        sensor = (13.5, 77.25)
        binned = fr.binned_position(sensor, 4)
        assert fr.sensor_position(binned, 4) == pytest.approx(sensor)
        assert fr.sensor_position((0.0, 0.0), 4) == pytest.approx((1.5, 1.5))  # the first block

    def test_bilinear_upsampling_keeps_a_constant_and_a_ramp(self) -> None:
        constant = np.full((6, 8), 2.5)
        assert np.all(fr.upsample_bilinear(constant, 4, (24, 32)) == np.float32(2.5))
        x = np.arange(8, dtype=np.float64)
        ramp = np.tile(x, (6, 1))
        full = fr.upsample_bilinear(ramp, 4, (24, 32))
        assert full.dtype == np.float32
        # the ramp is 1 per block, so 1 / 4 per pixel, from the center of the first block on
        columns = np.arange(32)
        expected = np.clip((columns + 0.5) / 4.0 - 0.5, 0.0, 7.0)
        np.testing.assert_allclose(full[10], expected, atol=1e-6)


class TestTheDecomposition:
    def make(self) -> Image:
        x, y = grid()
        radius = np.hypot(x - CENTER[0], y - CENTER[1])
        radial = 1.0 - 0.00001 * radius**2
        tilt = 1.0 + 0.01 * (x - CENTER[0]) / SHAPE[1] - 0.004 * (y - CENTER[1]) / SHAPE[0]
        return np.asarray(radial * tilt * dip(x, y, at=(60, 50), depth=0.03, radius=5), np.float64)

    def test_the_three_parts_multiply_back_to_the_image(self) -> None:
        image = self.make()
        parts = fr.decompose(image, center_xy=CENTER, high_pass_px=10.0)
        np.testing.assert_allclose(parts.radial_map * parts.rest * parts.fine, image, rtol=1e-12)

    def test_the_radial_part_has_no_tilt_and_the_rest_holds_it(self) -> None:
        parts = fr.decompose(self.make(), center_xy=CENTER, high_pass_px=10.0)
        tilt = fr.fit_tilt(parts.radial_map, center_xy=CENTER)
        assert abs(tilt.width_percent) < 0.02  # the radial part is symmetric about the center
        assert abs(tilt.height_percent) < 0.02
        rest = fr.fit_tilt(parts.rest, center_xy=CENTER)
        assert rest.width_percent == pytest.approx(1.0, abs=0.05)
        assert rest.height_percent == pytest.approx(-0.4, abs=0.05)

    def test_a_plane_leaves_no_slope_in_the_rings_at_the_frame_edge(self) -> None:
        # The optical center lies a quarter pixel off the middle of the frame, as it does when a
        # binned frame drops a row, and the frame has a plane of 4.5% across its height. A ring at
        # the corner holds few pixels, and when they are not symmetric about the center, a plane
        # that stays in the image leaves its slope in the mean of the ring.
        x, y = grid()
        radius = np.hypot(x - CENTER[0], y - CENTER[1])
        image = (1.0 - 0.00001 * radius**2) * (1.0 + 0.045 * (y - CENTER[1]) / SHAPE[0])
        parts = fr.decompose(image, center_xy=(CENTER[0], CENTER[1] + 0.25), high_pass_px=10.0)
        assert float(np.abs(parts.fine - 1.0).max()) < 0.001  # 1.6% when the plane stays in

    def test_invalid_pixels_do_not_count_and_their_fine_part_is_one(self) -> None:
        image = self.make()
        valid = np.ones(SHAPE, dtype=np.bool_)
        valid[:20, :20] = False
        image[:20, :20] = 1e6  # garbage in the invalid corner must not matter
        parts = fr.decompose(image, center_xy=CENTER, high_pass_px=10.0, valid=valid)
        assert np.all(parts.fine[:20, :20] == 1.0)
        clean = fr.decompose(self.make(), center_xy=CENTER, high_pass_px=10.0)
        np.testing.assert_allclose(parts.rest[100:, 100:], clean.rest[100:, 100:], atol=1e-3)

    def test_a_bad_width_and_an_empty_image_are_errors(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            fr.decompose(np.ones(SHAPE), center_xy=CENTER, high_pass_px=0.0)
        with pytest.raises(ValueError, match="no valid pixels"):
            fr.decompose(
                np.ones(SHAPE),
                center_xy=CENTER,
                high_pass_px=5.0,
                valid=np.zeros(SHAPE, dtype=np.bool_),
            )


class TestTheVignettingProfile:
    def image(self) -> Image:
        x, y = grid()
        radius_deg = np.hypot(x - CENTER[0], y - CENTER[1]) * FACTOR * SCALE / 3600.0
        return np.asarray(1.0 - 0.02 * radius_deg**2, dtype=np.float64)  # 2% a degree squared

    def test_each_radius_gives_the_loss_against_the_center(self) -> None:
        points = fr.vignetting_profile(
            self.image(), factor=FACTOR, scale_arcsec_px=SCALE, center_xy=CENTER
        )
        by_radius = {p.radius_deg: p for p in points if not p.corner}
        for degrees in (0.5, 1.0, 1.5, 2.0):
            expected = -100.0 * 0.02 * degrees**2
            point = by_radius[degrees].change_percent
            assert point is not None
            assert point == pytest.approx(expected, abs=0.04)

    def test_a_radius_beyond_the_corners_is_outside_the_frame(self) -> None:
        points = fr.vignetting_profile(
            self.image(),
            factor=FACTOR,
            scale_arcsec_px=SCALE,
            center_xy=CENTER,
            radii_deg=(0.5, 3.0),  # the corners lie at 2.66 degrees
        )
        assert [p.change_percent is None for p in points[:2]] == [False, True]

    def test_the_last_point_gives_the_corners(self) -> None:
        points = fr.vignetting_profile(
            self.image(), factor=FACTOR, scale_arcsec_px=SCALE, center_xy=CENTER
        )
        corners = points[-1]
        assert corners.corner
        half_diagonal = np.hypot(SHAPE[1] - 1, SHAPE[0] - 1) / 2.0 * FACTOR * SCALE / 3600.0
        assert corners.radius_deg == pytest.approx(half_diagonal, abs=0.01)
        assert corners.change_percent == pytest.approx(
            -2.0 * half_diagonal**2, abs=0.3
        )  # a ring of 2 pixels

    def test_the_center_is_a_disk_whose_share_of_the_field_you_can_set(self) -> None:
        image = self.image()
        level, radius = fr.center_level(image, center_xy=CENTER)
        corner = float(np.hypot(SHAPE[1] - 1, SHAPE[0] - 1)) / 2.0
        assert radius == pytest.approx(0.03 * corner)
        wide_level, wide_radius = fr.center_level(image, center_xy=CENTER, center_fraction=0.15)
        assert wide_radius == pytest.approx(0.15 * corner)
        assert wide_level < level < 1.0  # the loss grows with the radius: a wider disk is darker
        tiny_level, tiny_radius = fr.center_level(image, center_xy=CENTER, center_fraction=0.001)
        assert tiny_radius == 4.0  # 4 binned pixels at least
        assert tiny_level == pytest.approx(level, abs=1e-3)

    def test_a_wider_center_moves_the_change_at_every_radius(self) -> None:
        image = self.image()
        narrow = fr.vignetting_profile(
            image, factor=FACTOR, scale_arcsec_px=SCALE, center_xy=CENTER
        )
        wide = fr.vignetting_profile(
            image,
            factor=FACTOR,
            scale_arcsec_px=SCALE,
            center_xy=CENTER,
            center_fraction=0.15,
        )
        level, _ = fr.center_level(image, center_xy=CENTER)
        wide_level, _ = fr.center_level(image, center_xy=CENTER, center_fraction=0.15)
        shift = 100.0 * (level / wide_level - 1.0)
        for a, b in zip(narrow[:4], wide[:4], strict=True):
            assert a.change_percent is not None
            assert b.change_percent is not None
            # the disk is in the ratio, so a wider disk shifts every change by about the same
            assert b.change_percent - a.change_percent == pytest.approx(shift, abs=0.05)

    def test_a_masked_center_is_an_error(self) -> None:
        valid = np.ones(SHAPE, dtype=np.bool_)
        valid[70:105, 100:160] = False
        with pytest.raises(ValueError, match="center"):
            fr.vignetting_profile(
                self.image(), factor=FACTOR, scale_arcsec_px=SCALE, center_xy=CENTER, valid=valid
            )


class TestTheTilt:
    def test_a_plane_gives_its_change_across_the_width_and_across_the_height(self) -> None:
        x, y = grid()
        image = 1.0 + 0.012 * (x - CENTER[0]) / SHAPE[1] - 0.0035 * (y - CENTER[1]) / SHAPE[0]
        tilt = fr.fit_tilt(image, center_xy=CENTER)
        assert tilt.width_percent == pytest.approx(1.2, abs=0.01)
        assert tilt.height_percent == pytest.approx(-0.35, abs=0.01)

    def test_a_radial_profile_adds_no_tilt(self) -> None:
        x, y = grid()
        image = 1.0 - 4e-5 * ((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2)
        tilt = fr.fit_tilt(image, center_xy=CENTER)
        assert abs(tilt.width_percent) < 1e-6
        assert abs(tilt.height_percent) < 1e-6

    def test_the_tilt_after_the_radial_part_reads_a_plane_that_scales_the_vignetting(self) -> None:
        x, y = grid()
        radial = 1.0 - 4e-5 * ((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2)
        image = radial * (1.0 + 0.01 * (x - CENTER[0]) / SHAPE[1])
        parts = fr.decompose(image, center_xy=CENTER, high_pass_px=10.0)
        tilt = fr.tilt_after_radial(image, parts, center_xy=CENTER)
        assert tilt.width_percent == pytest.approx(1.0, abs=0.02)
        assert tilt.height_percent == pytest.approx(0.0, abs=0.02)

    def test_the_split_is_half_the_sum_and_half_the_difference(self) -> None:
        # the numbers of a real pair of sets: the source turned by 180 degrees between them
        first = fr.Tilt(-0.08, 0.34)
        second = fr.Tilt(1.20, -1.01)
        split = fr.split_tilts(first, second)
        assert split.stays.width_percent == pytest.approx(0.56)
        assert split.stays.height_percent == pytest.approx(-0.335)
        assert split.turns.width_percent == pytest.approx(-0.64)
        assert split.turns.height_percent == pytest.approx(0.675)


class TestTheShadows:
    def fine(self) -> Image:
        x, y = grid()
        fine = dip(x, y, at=(100, 60), depth=0.03, radius=4.5)  # an interior shadow
        fine *= dip(x, y, at=(1.5, 40), depth=0.02, radius=2.0)  # an edge artifact
        fine *= dip(x, y, at=(200, 120), depth=0.006, radius=4.0)  # too shallow to count
        return fine

    def test_a_shadow_has_a_position_a_depth_and_a_width_in_sensor_pixels(self) -> None:
        found = fr.find_shadows(self.fine(), factor=FACTOR, sensor_shape=SENSOR_SHAPE)
        (shadow,) = [s for s in found if not s.at_edge]
        assert shadow.x_px == pytest.approx(100 * FACTOR + 1.5, abs=2)
        assert shadow.y_px == pytest.approx(60 * FACTOR + 1.5, abs=2)
        assert shadow.depth == pytest.approx(0.03, rel=0.2)  # a light smoothing takes some depth
        assert shadow.width_px == pytest.approx(2 * 4.5 * FACTOR, rel=0.15)

    def test_a_dip_within_20_pixels_of_an_edge_is_an_edge_artifact(self) -> None:
        found = fr.find_shadows(self.fine(), factor=FACTOR, sensor_shape=SENSOR_SHAPE)
        (artifact,) = [s for s in found if s.at_edge]
        assert artifact.x_px < fr.EDGE_MARGIN_PX
        assert artifact.depth == pytest.approx(0.02, rel=0.3)

    def test_the_margin_is_a_parameter(self) -> None:
        found = fr.find_shadows(
            self.fine(), factor=FACTOR, sensor_shape=SENSOR_SHAPE, edge_margin_px=2.0
        )
        assert not any(s.at_edge for s in found)

    def test_a_dip_under_one_percent_is_not_a_shadow_and_the_list_is_deepest_first(self) -> None:
        found = fr.find_shadows(self.fine(), factor=FACTOR, sensor_shape=SENSOR_SHAPE)
        assert [round(s.depth, 2) for s in found] == [0.03, 0.02]  # the 0.6% dip is missing
        assert found[0].depth > found[1].depth
        assert fr.find_shadows(np.ones(SHAPE), factor=FACTOR, sensor_shape=SENSOR_SHAPE) == ()

    def test_locating_dips_gives_the_region_of_each_one(self) -> None:
        fine = self.fine()
        dips = fr.locate_dips(fine, factor=FACTOR, sensor_shape=SENSOR_SHAPE)
        assert dips.shadows == fr.find_shadows(fine, factor=FACTOR, sensor_shape=SENSOR_SHAPE)
        assert len(dips.ids) == 2
        assert dips.labels.shape == SHAPE
        mask = dips.mask()  # the dips that count: not the one at the edge
        assert mask[60, 100]
        assert not mask[40, 1]
        assert not mask[120, 200]  # the dip that is too shallow
        both = dips.mask(edge=True)
        assert both[40, 1]
        assert int(both.sum()) > int(mask.sum())
        assert (
            fr.locate_dips(np.ones(SHAPE), factor=FACTOR, sensor_shape=SENSOR_SHAPE).shadows == ()
        )

    def test_a_bump_is_a_dip_of_the_mirror_image(self) -> None:
        x, y = grid()
        bump = 2.0 - dip(x, y, at=(100, 60), depth=0.03, radius=4.5)  # 1.03 at the middle
        assert fr.find_shadows(bump, factor=FACTOR, sensor_shape=SENSOR_SHAPE) == ()
        (patch,) = fr.find_shadows(2.0 - bump, factor=FACTOR, sensor_shape=SENSOR_SHAPE)
        assert patch.depth == pytest.approx(0.03, rel=0.2)

    def test_the_search_needs_one_percent_or_five_times_the_noise(self) -> None:
        rng = np.random.default_rng(5)
        quiet = 1.0 + 0.001 * rng.standard_normal(SHAPE)
        depth, depth_map = fr.shadow_search_depth(quiet)
        assert depth == fr.SHADOW_MIN_DEPTH  # 5 times the smoothed noise is under 1%
        assert depth_map == depth
        noisy = 1.0 + 0.04 * rng.standard_normal(SHAPE)
        depth, depth_map = fr.shadow_search_depth(noisy)
        assert depth == pytest.approx(5.0 * 0.04 * 0.28, rel=0.15)  # a Gaussian of 1 px takes 0.28
        assert depth_map == depth
        scale = np.ones(SHAPE)
        scale[:, 200:] = 3.0  # fewer frames over the right side
        _, depth_map = fr.shadow_search_depth(noisy, noise_scale=scale)
        assert isinstance(depth_map, np.ndarray)
        assert depth_map[0, 0] == pytest.approx(depth)
        assert depth_map[0, 250] == pytest.approx(3.0 * depth)

    def test_the_summary_separates_the_shadows_from_the_edge_artifacts(self) -> None:
        x, y = grid()
        radius = np.hypot(x - CENTER[0], y - CENTER[1])
        image = (1.0 - 0.00001 * radius**2) * self.fine()
        summary = fr.summarize_flat(
            image,
            factor=FACTOR,
            sensor_shape=SENSOR_SHAPE,
            scale_arcsec_px=SCALE,
            center_xy=CENTER,
            high_pass_px=10.0,
        )
        assert len(summary.shadows) == 1
        assert len(summary.edge_artifacts) == 1
        assert summary.profile[0].radius_deg == 0.5
        assert summary.tilt.width_percent == pytest.approx(0.0, abs=0.1)


class TestTwoSets:
    def sets(self, source_gradient: float) -> tuple[Image, Image]:
        x, y = grid()
        radius = np.hypot(x - CENTER[0], y - CENTER[1])
        lens = (1.0 - 0.00001 * radius**2) * dip(x, y, at=(100, 60), depth=0.03, radius=4.5)
        first = lens * (1.0 + source_gradient * (x - CENTER[0]) / SHAPE[1])
        second = lens * (1.0 - source_gradient * (x - CENTER[0]) / SHAPE[1])
        return first, second

    def test_two_sets_of_a_steady_source_agree_to_the_noise(self) -> None:
        rng = np.random.default_rng(3)
        lens, _ = self.sets(0.0)
        noise = 0.002  # a fraction per sensor pixel
        first = lens * (1.0 + noise / FACTOR * rng.standard_normal(SHAPE))
        second = lens * (1.0 + noise / FACTOR * rng.standard_normal(SHAPE))
        agreement = fr.compare_sets(
            first,
            second,
            factor=FACTOR,
            center_xy=CENTER,
            high_pass_px=10.0,
            noise_first=noise,
            noise_second=noise,
        )
        assert agreement.expected_fine_rms == pytest.approx(100 * noise * np.sqrt(2) / FACTOR)
        assert agreement.fine_rms == pytest.approx(agreement.expected_fine_rms, rel=0.1)
        assert agreement.smooth_rms < 0.02
        assert abs(agreement.plane.width_percent) < 0.05

    def test_a_source_that_turned_leaves_its_gradient_in_the_plane_of_the_quotient(self) -> None:
        first, second = self.sets(0.0065)
        agreement = fr.compare_sets(
            first, second, factor=FACTOR, center_xy=CENTER, high_pass_px=10.0
        )
        assert agreement.expected_fine_rms is None  # no noise was given
        assert agreement.plane.width_percent == pytest.approx(1.3, abs=0.05)  # twice the gradient
        assert agreement.plane.height_percent == pytest.approx(0.0, abs=0.05)
        assert agreement.smooth_rms == pytest.approx(1.3 / np.sqrt(12), abs=0.03)


class TestTheText:
    def test_a_percentage_has_a_sign_and_never_shows_negative_zero(self) -> None:
        assert fr.format_percent(-3.24) == "-3.2%"
        assert fr.format_percent(0.44, digits=2) == "+0.44%"
        assert fr.format_percent(-0.004, digits=2) == "0.00%"

    def test_the_profile_lines_name_the_radii_the_corners_and_a_radius_off_the_frame(self) -> None:
        lines = fr.profile_lines(
            [
                fr.ProfilePoint(0.5, -0.43),
                fr.ProfilePoint(2.5, None),
                fr.ProfilePoint(2.66, -9.9, corner=True),
            ]
        )
        assert lines[1] == "  0.5 degrees: -0.43%"
        assert lines[2] == "  2.5 degrees: outside the frame"
        assert lines[3] == "  corners (2.66 degrees): -9.90%"

    def test_the_shadow_lines_list_the_shadows_and_count_those_that_do_not_fit(self) -> None:
        shadows = [fr.Shadow(100 + i, 50, 0.02, 30.0) for i in range(15)]
        lines = fr.shadow_lines(shadows)
        assert lines[0] == "Shadows deeper than 1%: 15"
        assert lines[1] == "  x 100, y 50: depth 2.0%, width 30 px"
        assert len(lines) == 1 + fr.MAX_LISTED_SHADOWS + 1
        assert lines[-1] == "  and 3 more"
        assert fr.shadow_lines([]) == ["Shadows deeper than 1%: none"]

    def test_the_heading_of_the_shadows_takes_the_words_of_the_caller(self) -> None:
        patch = [fr.Shadow(10, 20, 0.031, 18.0)]
        lines = fr.shadow_lines(
            patch,
            title="Patches brighter than the base flat",
            relation="by more than",
            measure="excess",
        )
        assert lines == [
            "Patches brighter than the base flat by more than 1%: 1",
            "  x 10, y 20: excess 3.1%, width 18 px",
        ]
        assert fr.shadow_lines([], title="New shadows") == ["New shadows deeper than 1%: none"]

    def test_a_note_about_the_edge_and_a_note_about_the_noise_share_one_pair_of_parentheses(
        self,
    ) -> None:
        edge = fr.edge_artifact_lines([], margin_px=20.0)
        assert edge == ["Edge artifacts deeper than 1% (center within 20 px of an edge): none"]
        noisy = fr.edge_artifact_lines([], margin_px=12.0, depth=0.0151)
        assert noisy == [
            "Edge artifacts deeper than 1.51% (center within 12 px of an edge; the noise raises "
            "the search above 1%): none"
        ]
        assert fr.shadow_lines([], depth=0.02) == [
            "Shadows deeper than 2% (the noise raises the search above 1%): none"
        ]

    def test_the_tilt_text_gives_the_two_directions_and_the_sign_rule(self) -> None:
        assert (
            fr.tilt_text(fr.Tilt(0.56, -0.34))
            == "+0.56% across the width, -0.34% across the height"
        )
        assert "rises toward the right edge or the bottom edge" in fr.tilt_line(fr.Tilt(0.1, 0.1))
