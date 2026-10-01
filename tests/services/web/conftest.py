"""Fixtures of the web tests: a seeded store, a fake `core`, the app, and a client."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI

from seeingmon.clock import VirtualClock
from seeingmon.services.web.app import build_context, create_app
from seeingmon.services.web.auth import ScryptParams, hash_token
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.core_client import FakeCoreClient
from seeingmon.services.web.images import ImageStore
from seeingmon.store.db import Store, StoreReader
from seeingmon.store.layout import DataLayout
from tests.services.web.client import TestClient
from tests.services.web.helpers import TOKEN
from tests.services.web.seed import NOW_NS, STATION, seed_records

PROFILE: dict[str, Any] = {
    "id": "test-profile",
    "sensor": {"name": "Test camera"},
    "readout_modes": [{"name": "bin1", "derived": {"plate_scale_arcsec_per_px": 1.9}}],
}

CONFIG: dict[str, Any] = {
    "profile": "test-profile",
    "station_id": STATION,
    "scheduler": {"fast": {"window_s": 120.0}},
    "paths": {"data_dir": "<redacted>"},
}


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


@pytest.fixture
def core(clock: VirtualClock) -> FakeCoreClient:
    return FakeCoreClient(clock=clock)


@pytest.fixture(scope="session")
def shared_app(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FastAPI]:
    """One app for the whole run. Building the routes of an app costs about 200 ms.

    The routes read the context from `app.state.ctx` on every request, so each test puts its own
    context in the app (see `make_app`).
    """
    folder = tmp_path_factory.mktemp("shared-app")
    layout = DataLayout(folder / "data")
    layout.create()
    settings = WebSettings()
    with Store.open(layout.db_path), StoreReader.open(layout.db_path) as reader:
        yield create_app(settings, reader, ImageStore(layout, settings.images), FakeCoreClient())


@pytest.fixture
def make_app(
    shared_app: FastAPI,
    settings: WebSettings,
    reader: StoreReader,
    layout: DataLayout,
    core: FakeCoreClient,
    clock: VirtualClock,
    token_hash: str,
) -> Callable[..., FastAPI]:
    """Give the shared app a context over the test store. Keyword arguments change what it gets."""

    def build(**overrides: Any) -> FastAPI:
        options: dict[str, Any] = {
            "clock": clock,
            "token_hash": token_hash,
            "profile": PROFILE,
            "config": CONFIG,
            "station_id": STATION,
        }
        chosen_settings = overrides.pop("settings", settings)
        chosen_core = overrides.pop("core", core)
        options.update(overrides)
        shared_app.state.ctx = build_context(
            chosen_settings,
            reader,
            ImageStore(layout, chosen_settings.images),
            chosen_core,
            **options,
        )
        return shared_app

    return build


@pytest.fixture
def app(make_app: Callable[..., FastAPI]) -> FastAPI:
    return make_app()


@pytest.fixture
def open_client() -> Iterator[Callable[..., TestClient]]:
    """Open clients with the lifespan running. Every client closes when the test ends."""
    with ExitStack() as stack:

        def open_one(app: FastAPI, **options: Any) -> TestClient:
            return stack.enter_context(TestClient(app, **options))

        yield open_one


@pytest.fixture
def client(app: FastAPI, seeded: Store, open_client: Callable[..., TestClient]) -> TestClient:
    """A client of the app over a seeded store."""
    return open_client(app)


@pytest.fixture
def empty_client(app: FastAPI, writer: Store, open_client: Callable[..., TestClient]) -> TestClient:
    """A client of the app over a store with no record."""
    return open_client(app)
