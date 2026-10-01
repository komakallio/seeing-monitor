"""A crash loses only the transaction that was open, and never corrupts the database."""

from __future__ import annotations

import contextlib
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from seeingmon.records.sqlite_schema import insert_record
from seeingmon.store.db import Store
from tests.store.builders import T0, make_health

KILLED_WRITER = """
import os
import sys

from seeingmon.records.samples import sample_record
from seeingmon.records.sqlite_schema import insert_record
from seeingmon.records.system import HealthRecord
from seeingmon.store.db import Store

T0 = 1_767_225_600_000_000_000
store = Store.open(sys.argv[1])
for n in range(5):
    store.write(sample_record(HealthRecord, t_utc_ns=T0 + n))
connection = store._writer
connection.execute("BEGIN IMMEDIATE")
for n in range(5, 9):
    insert_record(connection, sample_record(HealthRecord, t_utc_ns=T0 + n))
os._exit(7)  # no commit, no close, no clean-up: the process dies here
"""


def raw(path: Path) -> contextlib.closing[sqlite3.Connection]:
    return contextlib.closing(sqlite3.connect(path))


def test_a_process_that_dies_in_a_transaction_loses_only_the_open_transaction(
    db_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, "-c", KILLED_WRITER, str(db_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 7, result.stderr
    with Store.open(db_path) as store:
        assert store.count("health") == 5
        assert [row.row_id for row in store.after("health", 0)] == [1, 2, 3, 4, 5]
        assert store.write(make_health(T0 + 100)) == 6  # the rolled-back rows left no hole
    with raw(db_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_closing_a_store_in_the_middle_of_a_transaction_rolls_it_back(db_path: Path) -> None:
    store = Store.open(db_path)
    store.write(make_health(T0))
    connection = store._writer
    assert connection is not None
    connection.execute("BEGIN IMMEDIATE")
    insert_record(connection, make_health(T0 + 1))
    store.close()  # no final commit
    with Store.open(db_path) as again:
        assert again.count("health") == 1


def test_a_torn_write_ahead_log_recovers_to_a_prefix_of_the_commits(
    store: Store, db_path: Path, tmp_path: Path
) -> None:
    """Cut the log in the middle of a frame, as a power loss can, and open the copy."""
    store.checkpoint()  # the schema is in the database file, and the log starts empty
    total = 40
    for n in range(total):
        store.write(make_health(T0 + n))
    copy = tmp_path / "copy" / "results.sqlite"
    copy.parent.mkdir()
    shutil.copyfile(db_path, copy)
    log = Path(f"{db_path}-wal").read_bytes()
    assert len(log) > 4096 * 10
    cut = len(log) * 6 // 10 + 123  # inside a frame
    Path(f"{copy}-wal").write_bytes(log[:cut])
    with raw(copy) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        ids = [row[0] for row in connection.execute('SELECT row_id FROM "health" ORDER BY row_id')]
    assert 0 < len(ids) < total
    assert ids == list(range(1, len(ids) + 1))  # a prefix: no commit after the tear survives
    with Store.open(copy) as recovered:  # and the store opens it and goes on writing
        assert recovered.write(make_health(T0 + 1000)) == len(ids) + 1
