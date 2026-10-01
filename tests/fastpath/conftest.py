"""Fixtures for the fast-path tests."""

from __future__ import annotations

import pytest

from seeingmon.profile import Profile, load_profile


@pytest.fixture(scope="session")
def profile() -> Profile:
    """The reference profile: the ASI294MM behind a GS-250 (50 mm, f/5)."""
    return load_profile("asi294mm-gs250")
