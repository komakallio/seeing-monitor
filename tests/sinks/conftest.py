from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.store.db import Store
from seeingmon.store.events import EventEmitter
from tests.store.builders import STATION, T0


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock(T0)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "results.sqlite") as opened:
        yield opened


@pytest.fixture
def emitter(store: Store, clock: VirtualClock) -> EventEmitter:
    """Writes events into the store, as production does, so that sinks receive them too."""
    return EventEmitter(
        store.write,
        clock,
        station_id=STATION,
        profile_id="profile-1",
        provenance={"software": "test"},
    )
