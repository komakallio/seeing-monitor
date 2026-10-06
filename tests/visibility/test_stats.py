"""The statistics of the visibility summaries: groups by month and by transparency."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from seeingmon.visibility import stats


@dataclass(frozen=True)
class Night:
    """The fields of a summary that the statistics read."""

    night: str
    first_visible_sun_deg: float | None = None
    last_visible_sun_deg: float | None = None
    first_censored: bool = False
    last_censored: bool = False
    transparency_median: float | None = None
    first_visible_utc_ns: int | None = None
    last_visible_utc_ns: int | None = None

    def __post_init__(self) -> None:
        # A night with a Sun elevation has a time too.
        if self.first_visible_sun_deg is not None and self.first_visible_utc_ns is None:
            object.__setattr__(self, "first_visible_utc_ns", 1)
        if self.last_visible_sun_deg is not None and self.last_visible_utc_ns is None:
            object.__setattr__(self, "last_visible_utc_ns", 2)


NIGHTS = [
    Night("2026-01-01", -6.0, -7.0, transparency_median=0.95),
    Night("2026-01-02", -4.0, -5.0, transparency_median=0.85),
    Night("2026-01-03", -3.0, -6.0, first_censored=True, transparency_median=0.92),
    Night("2026-01-04"),  # watched, never seen: clouds
    Night("2026-02-01", -2.0, -3.0, last_censored=True),
    Night("2026-02-02", first_censored=True, last_censored=True),  # down all night
]


class TestASide:
    def test_censored_values_stay_apart_from_the_measured_ones(self) -> None:
        side = stats.side_of(NIGHTS, "first")
        assert side.measured == (-6.0, -4.0, -2.0)
        assert side.censored == (-3.0,)
        assert side.censored_without_value == 1
        assert side.n_censored == 2
        assert side.median == -4.0

    def test_the_last_detection_has_its_own_censoring(self) -> None:
        side = stats.side_of(NIGHTS, "last")
        assert side.measured == (-7.0, -6.0, -5.0)
        assert side.censored == (-3.0,)
        assert side.censored_without_value == 1

    def test_a_detection_without_the_suns_elevation_counts_apart(self) -> None:
        night = Night("2026-03-01", first_visible_utc_ns=5)
        side = stats.side_of([night], "first")
        assert (side.measured, side.without_sun) == ((), 1)
        assert side.median is None


class TestTheGroups:
    def test_the_months_come_in_time_order_with_the_unseen_nights(self) -> None:
        groups = stats.by_month(reversed(NIGHTS))
        assert [g.key for g in groups] == ["2026-01", "2026-02"]
        january, february = groups
        assert (january.nights, january.unseen) == (4, 1)
        assert january.first.measured == (-6.0, -4.0)
        assert january.first.censored == (-3.0,)
        assert (february.nights, february.unseen) == (2, 0)
        assert february.last.n_censored == 2

    def test_the_transparency_bins_go_from_the_most_opaque_to_unknown(self) -> None:
        groups = stats.by_transparency(NIGHTS, (0.6, 0.8, 0.9))
        assert [g.key for g in groups] == ["0.8 to 0.9", ">= 0.9", "unknown"]
        assert [g.nights for g in groups] == [1, 2, 3]

    @pytest.mark.parametrize(
        ("value", "label"),
        [
            (0.3, "< 0.6"),
            (0.6, "0.6 to 0.8"),
            (0.85, "0.8 to 0.9"),
            (0.9, ">= 0.9"),
            (None, "unknown"),
        ],
    )
    def test_a_value_on_an_edge_falls_in_the_bin_above(
        self, value: float | None, label: str
    ) -> None:
        assert stats.transparency_bin(value, (0.6, 0.8, 0.9)) == label

    def test_edges_that_do_not_rise_are_refused(self) -> None:
        with pytest.raises(ValueError, match="rise"):
            stats.by_transparency(NIGHTS, (0.8, 0.6))

    def test_no_edges_make_one_bin(self) -> None:
        assert [g.key for g in stats.by_transparency(NIGHTS, ())] == ["all", "unknown"]
