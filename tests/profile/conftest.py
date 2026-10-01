from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from seeingmon.profile import Profile, parse_profile
from tests.profile.builders import reference_data, synthetic_data


@pytest.fixture(scope="session")
def reference() -> Profile:
    """The reference profile, loaded from the repository's file."""
    return parse_profile(reference_data(), source="the reference profile")


@pytest.fixture
def data() -> dict[str, Any]:
    """A fresh copy of the reference profile's data that a test can change."""
    return reference_data()


@pytest.fixture(scope="session")
def synthetic() -> Profile:
    """A second profile that shares no value with the reference profile."""
    return parse_profile(synthetic_data(), source="the synthetic profile")


@pytest.fixture(scope="session")
def profiles_dir(repo_root: Path) -> Path:
    """The repository's `profiles/` folder."""
    return repo_root / "profiles"
