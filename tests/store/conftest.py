from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from seeingmon.store.db import Store


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "db" / "results.sqlite"


@pytest.fixture
def store(db_path: Path) -> Iterator[Store]:
    with Store.open(db_path) as opened:
        yield opened
