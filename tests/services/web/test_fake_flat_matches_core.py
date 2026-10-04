"""The sentences and the limits of the flat fake against the ones of the real code.

The web process does not import the survey code, so the fake keeps its own copies of the sentences
that `core` gives. These tests fail when one side changes and the other does not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from seeingmon.scheduler.commands import (
    MAX_FLAT_FRAMES,
    MAX_FLAT_TARGET,
    MIN_FLAT_FRAMES,
    MIN_FLAT_TARGET,
)
from seeingmon.services.core.commissioning import flat as real
from seeingmon.services.web import fake_flat as fake
from seeingmon.services.web.models import FLAT_VERSION_PATTERN
from seeingmon.survey import flat_session
from seeingmon.survey.flat_library import (
    FLAT_IN_USE,
    UNKNOWN_FLAT,
    VERSION_PATTERN,
    FlatLibrary,
    FlatLibraryError,
)


def test_the_limits_are_the_ones_of_the_scheduler() -> None:
    assert (fake.MIN_FRAMES, fake.MAX_FRAMES) == (MIN_FLAT_FRAMES, MAX_FLAT_FRAMES)
    assert (fake.MIN_TARGET, fake.MAX_TARGET) == (MIN_FLAT_TARGET, MAX_FLAT_TARGET)


def test_the_sentences_are_the_ones_of_the_real_code(tmp_path: Path) -> None:
    assert fake.DARK_FIRST == flat_session.DARK_FIRST
    assert fake.NO_FIRST_SET == flat_session.NO_FIRST_SET
    assert fake.BUSY_MESSAGE == real.BUSY_MESSAGE
    assert fake.ABORTED_SUMMARY == real.ABORTED_SUMMARY
    assert fake.CANCELLED_SUMMARY == real.CANCELLED_SUMMARY
    assert fake.HOLD_MESSAGES == real.WAITING_MESSAGES
    assert fake.WAIT_MESSAGE == real.WAITING_MESSAGE
    assert fake.UNKNOWN_MESSAGE == UNKNOWN_FLAT
    assert fake.ACTIVE_MESSAGE == FLAT_IN_USE
    with pytest.raises(FlatLibraryError) as error:
        FlatLibrary(tmp_path).activate("flat-00000000", now_utc_ns=0, expect_shape=None)
    assert error.value.message == fake.UNKNOWN_MESSAGE


@pytest.mark.parametrize("seconds", [32e-6, 0.00255, 0.02, 0.039, 0.5, 1.0, 2.5, 120.0])
def test_the_exposures_read_as_the_real_session_writes_them(seconds: float) -> None:
    assert fake.exposure_text(seconds) == flat_session.format_exposure(seconds)


def test_the_advice_of_a_failure_is_the_advice_of_the_real_session() -> None:
    source = Path(flat_session.__file__).read_text(encoding="utf-8")
    joined = re.sub(r'"\s*\n\s*f?"', "", source)  # the sentences continue across string literals
    dim = "Use a brighter source, or hold it closer to the lens."
    bright = "Dim the source, or put a layer of cloth between it and the lens."
    assert dim in joined
    assert bright in joined
    assert dim in fake.DIM_SUMMARY
    assert bright in fake.BRIGHT_SUMMARY


def test_the_pattern_of_a_version_in_the_api_is_the_pattern_of_the_library() -> None:
    assert VERSION_PATTERN.pattern == FLAT_VERSION_PATTERN
