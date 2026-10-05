"""The typed SciPy helpers of the survey path, where another test does not already cover them."""

from __future__ import annotations

import numpy as np

from seeingmon.survey import _scipy


def test_the_pair_indices_are_the_pairs_of_the_lists() -> None:
    rng = np.random.default_rng(4)
    first = rng.uniform(0.0, 100.0, (60, 2))
    second = np.vstack([first[:10], rng.uniform(0.0, 100.0, (80, 2))])  # ten points repeat
    i, j = _scipy.pair_indices(first, second, 9.0)
    assert i.dtype == np.intp
    assert j.dtype == np.intp
    expected = {
        (a, b) for a, row in enumerate(_scipy.pairs_within(first, second, 9.0)) for b in row
    }
    assert {(int(a), int(b)) for a, b in zip(i, j, strict=True)} == expected
    assert len(expected) > 100
    assert (np.hypot(*(first[i] - second[j]).T) <= 9.0 + 1e-9).all()
    assert {(k, k) for k in range(10)} <= expected  # a point pairs with its own copy, at 0


def test_the_pair_indices_of_nothing_are_empty() -> None:
    none = np.zeros((0, 2))
    some = np.array([[1.0, 2.0], [3.0, 4.0]])
    for first, second in ((none, some), (some, none), (none, none)):
        i, j = _scipy.pair_indices(first, second, 5.0)
        assert i.size == 0
        assert j.size == 0
