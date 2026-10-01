"""Fixtures of the web tests: a seeded store and a clock."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.services.web.auth import ScryptParams, hash_token
from seeingmon.services.web.config import WebSettings
from seeingmon.store.db import Store, StoreReader
from seeingmon.store.layout import DataLayout
from tests.services.web.helpers import TOKEN
from tests.services.web.seed import NOW_NS, seed_records


@pytest.fixture(scope="session")
def token() -> str:
    return TOKEN


@pytest.fixture(scope="session")
def token_hash() -> str:
    """The hash of the test token, at a cost that a test can afford."""
    return hash_token(TOKEN, params=ScryptParams(ln=10, r=8, p=1))


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock(NOW_NS)


@pytest.fixture
def layout(tmp_path: Path) -> DataLayout:
    layout = DataLayout(tmp_path / "data")
    layout.create()
    return layout


@pytest.fixture
def writer(layout: DataLayout) -> Iterator[Store]:
    """The writer of the store. In production only `core` has one. The tests seed through it."""
    with Store.open(layout.db_path) as store:
        yield store


@pytest.fixture
def seeded(writer: Store) -> Store:
    seed_records(writer)
    return writer


@pytest.fixture
def reader(layout: DataLayout, writer: Store) -> Iterator[StoreReader]:
    """The read-only view that the web process gets."""
    with StoreReader.open(layout.db_path) as opened:
        yield opened


@pytest.fixture
def settings() -> WebSettings:
    return WebSettings.model_validate(
        {
            "live": {"max_fps": 30.0, "stall_s": 0.3, "idle_s": 5.0, "max_clients": 2},
            "rate_limit": {"commands_per_window": 5, "window_s": 60.0},
        }
    )
