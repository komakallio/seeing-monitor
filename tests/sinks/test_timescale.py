"""The PostgreSQL and TimescaleDB sink: generated SQL, batches, failures, and the driver."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.config import ConfigError
from seeingmon.records.base import RECORD_TYPES
from seeingmon.records.samples import sample_record
from seeingmon.records.sink_mapping import sink_mapping
from seeingmon.records.system import EventRecord
from seeingmon.sinks.base import Sink, SinkError, StoredRow
from seeingmon.sinks.config import TimescaleSinkConfig
from seeingmon.sinks.forwarder import Forwarder
from seeingmon.sinks.timescale import (
    EXTENSION_QUERY,
    NS_PER_DAY,
    TimescaleSink,
    hypertable_sql,
    is_permanent,
    make_psycopg_connect,
    row_parameters,
    schema_statements,
    upsert_sql,
)
from seeingmon.store.config import ForwarderConfig
from seeingmon.store.db import Store
from tests.sinks import fake_db
from tests.sinks.fake_db import FakeDatabase
from tests.store.builders import NS_PER_S, T0, make_event, make_health

TABLE_TYPES = [cls for cls in RECORD_TYPES.values() if cls.storage == "table"]
CODE = "example-code-value"

EVENT_CREATE = """\
CREATE TABLE IF NOT EXISTS "event" (
    "station_id" TEXT NOT NULL,
    "t_utc_ns" BIGINT NOT NULL,
    "revision" BIGINT NOT NULL,
    "profile_id" TEXT NOT NULL,
    "provenance" JSONB NOT NULL,
    "quality" JSONB,
    "level" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "detail" JSONB
)"""

EVENT_UPSERT = (
    'INSERT INTO "event" ("station_id", "t_utc_ns", "revision", "profile_id", "provenance", '
    '"quality", "level", "kind", "message", "detail") VALUES (%s, %s, %s, %s, %s::jsonb, '
    '%s::jsonb, %s, %s, %s, %s::jsonb) ON CONFLICT ("station_id", "t_utc_ns", "revision") '
    'DO UPDATE SET "profile_id" = EXCLUDED."profile_id", "provenance" = EXCLUDED."provenance", '
    '"quality" = EXCLUDED."quality", "level" = EXCLUDED."level", "kind" = EXCLUDED."kind", '
    '"message" = EXCLUDED."message", "detail" = EXCLUDED."detail"'
)


def rows_of(*records: Any) -> list[StoredRow]:
    return [StoredRow(n + 1, record.to_row()) for n, record in enumerate(records)]


def make_sink(database: FakeDatabase, **overrides: Any) -> TimescaleSink:
    return TimescaleSink("archive", database.connect, **overrides)


class TestGeneratedSql:
    def test_the_event_table_is_created_with_the_mapped_types(self) -> None:
        statements = schema_statements("event", hypertable=False)
        assert statements[0] == EVENT_CREATE

    def test_every_column_is_added_if_it_is_missing_so_an_old_table_gains_new_fields(self) -> None:
        statements = schema_statements("event", hypertable=False)
        alters = [s for s in statements if s.startswith("ALTER TABLE")]
        assert alters == [
            'ALTER TABLE "event" ADD COLUMN IF NOT EXISTS "profile_id" TEXT',
            'ALTER TABLE "event" ADD COLUMN IF NOT EXISTS "provenance" JSONB',
            'ALTER TABLE "event" ADD COLUMN IF NOT EXISTS "quality" JSONB',
            'ALTER TABLE "event" ADD COLUMN IF NOT EXISTS "level" TEXT',
            'ALTER TABLE "event" ADD COLUMN IF NOT EXISTS "kind" TEXT',
            'ALTER TABLE "event" ADD COLUMN IF NOT EXISTS "message" TEXT',
            'ALTER TABLE "event" ADD COLUMN IF NOT EXISTS "detail" JSONB',
        ]

    def test_the_key_has_a_unique_index_that_includes_the_time_column(self) -> None:
        statements = schema_statements("event", hypertable=False)
        assert statements[-1] == (
            'CREATE UNIQUE INDEX IF NOT EXISTS "event_key" ON "event" '
            '("station_id", "t_utc_ns", "revision")'
        )

    def test_a_hypertable_is_made_on_the_time_column_with_a_chunk_interval_in_nanoseconds(
        self,
    ) -> None:
        statements = schema_statements("event", hypertable=True)
        assert statements[-1] == (
            "SELECT create_hypertable('\"event\"', 't_utc_ns', chunk_time_interval => "
            "604800000000000, if_not_exists => TRUE, migrate_data => TRUE)"
        )
        assert NS_PER_DAY * 7 == 604_800_000_000_000
        again = schema_statements("event", hypertable=True, chunk_days=14)
        assert "chunk_time_interval => 1209600000000000" in again[-1]
        mapping = sink_mapping("event").timescale
        assert hypertable_sql(mapping, 1).count(str(NS_PER_DAY)) == 1

    def test_the_hypertable_comes_after_the_index_and_only_when_asked(self) -> None:
        plain = schema_statements("event", hypertable=False)
        hyper = schema_statements("event", hypertable=True)
        assert hyper[:-1] == plain
        assert not any("create_hypertable" in s for s in plain)

    def test_the_upsert_replaces_every_column_except_the_key(self) -> None:
        assert upsert_sql("event") == EVENT_UPSERT

    @pytest.mark.parametrize("cls", TABLE_TYPES, ids=lambda cls: cls.record_type)
    def test_every_table_record_type_has_matching_columns_placeholders_and_parameters(
        self, cls: Any
    ) -> None:
        mapping = sink_mapping(cls.record_type).timescale
        sql = upsert_sql(cls.record_type)
        assert sql.count("%s") == len(mapping.columns)
        record = sample_record(cls)
        assert len(row_parameters(cls.record_type, record.to_row())) == len(mapping.columns)
        for column in mapping.columns:
            if column.kind == "json":
                assert "%s::jsonb" in sql
        statements = schema_statements(cls.record_type, hypertable=True)
        assert statements[0].startswith(f'CREATE TABLE IF NOT EXISTS "{cls.record_type}"')
        assert 'ON CONFLICT ("station_id", "t_utc_ns", "revision")' in sql

    def test_the_frame_type_has_no_table_in_the_forwarder_but_a_mapping_exists(self) -> None:
        assert sink_mapping("frame").timescale.table == "frame"  # a sink may still accept it

    def test_the_pole_columns_exist_and_an_old_pointing_table_gains_them(self) -> None:
        statements = schema_statements("pointing", hypertable=False)
        create = statements[0]
        alters = [statement for statement in statements if statement.startswith("ALTER TABLE")]
        for name in ("pole_x_px", "pole_y_px"):
            assert f'"{name}" DOUBLE PRECISION' in create
            assert f'"{name}" DOUBLE PRECISION NOT NULL' not in create  # the pole is optional
            assert (
                f'ALTER TABLE "pointing" ADD COLUMN IF NOT EXISTS "{name}" DOUBLE PRECISION'
                in alters
            )
        assert '"pole_x_px" = EXCLUDED."pole_x_px"' in upsert_sql("pointing")


class TestRowParameters:
    def test_the_values_take_the_driver_types(self) -> None:
        record = sample_record(
            "star_list", n_stars=1, columns=["x"], data=b"\x00\x00\x80\x3f", catalog="gaia-dr3"
        )
        parameters = dict(
            zip(
                [c.name for c in sink_mapping("star_list").timescale.columns],
                row_parameters("star_list", record.to_row()),
                strict=True,
            )
        )
        assert parameters["data"] == b"\x00\x00\x80\x3f"  # base64 text becomes BYTEA bytes
        assert json.loads(parameters["columns"]) == ["x"]  # a list becomes JSON text
        assert parameters["n_stars"] == 1
        assert parameters["catalog"] == "gaia-dr3"
        assert parameters["quality"] is None
        assert parameters["revision"] == 0

    def test_booleans_floats_and_missing_values(self) -> None:
        record = make_health(T0, degraded=True, heater_duty=0.25, sink_backlog={"a": 3})
        names = [c.name for c in sink_mapping("health").timescale.columns]
        parameters = dict(zip(names, row_parameters("health", record.to_row()), strict=True))
        assert parameters["degraded"] is True
        assert parameters["dark_due"] is False
        assert parameters["heater_duty"] == 0.25
        assert parameters["free_space_gb"] is None
        assert json.loads(parameters["sink_backlog"]) == {"a": 3}

    def test_a_row_from_older_software_with_missing_columns_still_builds(self) -> None:
        row = {k: v for k, v in make_event(T0).to_row().items() if k != "detail"}
        assert row_parameters("event", row)[-1] is None

    def test_a_nul_character_leaves_text_and_json(self) -> None:
        record = make_event(T0, message="a\x00b", detail={"k": "x\x00y"})
        names = [c.name for c in sink_mapping("event").timescale.columns]
        parameters = dict(zip(names, row_parameters("event", record.to_row()), strict=True))
        assert parameters["message"] == "ab"
        assert json.loads(parameters["detail"]) == {"k": "xy"}

    def test_non_ascii_text_stays_readable_in_json(self) -> None:
        record = make_event(T0, detail={"name": "Pohjantähti"})
        names = [c.name for c in sink_mapping("event").timescale.columns]
        parameters = dict(zip(names, row_parameters("event", record.to_row()), strict=True))
        assert parameters["detail"] == '{"name":"Pohjantähti"}'

    def test_the_pole_pixel_is_a_float_parameter_and_a_missing_pole_is_null(self) -> None:
        names = [c.name for c in sink_mapping("pointing").timescale.columns]
        pole = sample_record("pointing", t_utc_ns=T0, pole_x_px=2100.5, pole_y_px=-35.0)
        parameters = dict(zip(names, row_parameters("pointing", pole.to_row()), strict=True))
        assert (parameters["pole_x_px"], parameters["pole_y_px"]) == (2100.5, -35.0)
        bare = sample_record("pointing", t_utc_ns=T0)
        parameters = dict(zip(names, row_parameters("pointing", bare.to_row()), strict=True))
        assert (parameters["pole_x_px"], parameters["pole_y_px"]) == (None, None)

    def test_a_non_finite_float_becomes_null(self) -> None:
        row = {**make_health(T0).to_row(), "heater_duty": float("nan")}
        names = [c.name for c in sink_mapping("health").timescale.columns]
        parameters = dict(zip(names, row_parameters("health", row), strict=True))
        assert parameters["heater_duty"] is None


class TestSending:
    def test_the_first_batch_connects_prepares_the_table_and_commits_once_more(self) -> None:
        database = FakeDatabase()
        sink = make_sink(database)
        rows = rows_of(make_event(T0), make_event(T0 + 5))
        sink.send("event", rows)
        kinds = [entry[0] for entry in database.log]
        assert kinds == (
            ["connect", "execute", "commit"]  # detect the extension
            + ["execute"] * len(schema_statements("event", hypertable=True))
            + ["commit", "executemany", "commit"]  # the table is ready, then the batch
        )
        assert database.statements()[0] == EXTENSION_QUERY
        assert database.statements()[1:] == schema_statements("event", hypertable=True)
        ((_, sql, parameters),) = database.calls("executemany")
        assert sql == EVENT_UPSERT
        assert parameters == [row_parameters("event", row.values) for row in rows]
        assert all(cursor.closed for cursor in database.connections[0].cursors)

    def test_later_batches_reuse_the_connection_and_skip_the_schema(self) -> None:
        database = FakeDatabase()
        sink = make_sink(database)
        sink.send("event", rows_of(make_event(T0)))
        before = len(database.log)
        sink.send("event", rows_of(make_event(T0 + 1)))
        assert [entry[0] for entry in database.log[before:]] == ["executemany", "commit"]
        assert len(database.connections) == 1

    def test_a_new_record_type_prepares_its_own_table(self) -> None:
        database = FakeDatabase()
        sink = make_sink(database)
        sink.send("event", rows_of(make_event(T0)))
        sink.send("health", rows_of(make_health(T0)))
        creates = [s for s in database.statements() if s.startswith("CREATE TABLE")]
        assert [s.split('"')[1] for s in creates] == ["event", "health"]
        assert len(database.connections) == 1

    def test_a_server_without_timescaledb_gets_plain_tables(self) -> None:
        database = FakeDatabase(timescale=False)
        make_sink(database).send("event", rows_of(make_event(T0)))
        assert not any("create_hypertable" in s for s in database.statements())
        assert EXTENSION_QUERY in database.statements()

    def test_hypertables_can_be_forced_off_without_asking_the_server(self) -> None:
        database = FakeDatabase()
        make_sink(database, hypertables=False).send("event", rows_of(make_event(T0)))
        assert EXTENSION_QUERY not in database.statements()
        assert not any("create_hypertable" in s for s in database.statements())

    def test_hypertables_can_be_forced_on_and_the_chunk_interval_follows_the_setting(self) -> None:
        database = FakeDatabase(timescale=False)
        make_sink(database, hypertables=True, chunk_days=3).send("event", rows_of(make_event(T0)))
        assert EXTENSION_QUERY not in database.statements()
        assert f"chunk_time_interval => {3 * NS_PER_DAY}" in database.statements()[-1]

    def test_a_repeated_batch_leaves_the_same_rows(self) -> None:
        database = FakeDatabase()
        sink = make_sink(database)
        rows = rows_of(make_event(T0), make_event(T0 + 1))
        sink.send("event", rows)
        first = dict(database.rows["event"])
        sink.send("event", rows)
        assert database.rows["event"] == first
        assert len(first) == 2

    def test_a_corrected_row_replaces_the_old_one_by_key(self) -> None:
        database = FakeDatabase()
        sink = make_sink(database)
        sink.send("health", rows_of(make_health(T0, state="auto")))
        sink.send("health", rows_of(make_health(T0, state="auto", heater_duty=0.5)))
        assert len(database.rows["health"]) == 1

    def test_the_sink_reports_its_name_batch_size_and_filter(self) -> None:
        sink = make_sink(FakeDatabase(), max_batch_rows=50, record_types=["event"])
        assert (sink.name, sink.max_batch_rows) == ("archive", 50)
        assert sink.accepts("event")
        assert not sink.accepts("health")
        assert isinstance(sink, Sink)
        everything = make_sink(FakeDatabase())
        assert everything.accepts("health")
        assert not everything.accepts("frame")
        assert not everything.accepts("no_such_type")

    def test_a_sink_builds_from_its_configuration(self) -> None:
        config = TimescaleSinkConfig(
            kind="timescale",
            host="timescale.example.org",
            database="seeing",
            user="u",
            record_types=["event"],
            max_batch_rows=7,
            hypertables=False,
            chunk_days=2,
        )
        sink = TimescaleSink.from_config("archive", config, FakeDatabase().connect)
        assert (sink.max_batch_rows, sink.accepts("health")) == (7, False)

    def test_close_closes_the_connection_and_the_next_batch_connects_again(self) -> None:
        database = FakeDatabase()
        sink = make_sink(database)
        sink.send("event", rows_of(make_event(T0)))
        sink.close()
        sink.close()  # closing twice is fine
        assert database.connections[0].closed
        sink.send("event", rows_of(make_event(T0 + 1)))
        assert len(database.connections) == 2
        assert database.statements().count(EVENT_CREATE) == 2  # the schema runs again


class TestFailures:
    @pytest.mark.parametrize(
        "error",
        [
            fake_db.OperationalError("lost the connection"),
            fake_db.AdminShutdown("the server is restarting"),
            fake_db.InterfaceError("the connection is closed"),
            fake_db.InternalError("an internal error"),
            ConnectionResetError("reset"),
            TimeoutError("timed out"),
            RuntimeError("something unexpected"),
        ],
        ids=lambda e: type(e).__name__,
    )
    def test_an_operational_problem_is_retryable(self, error: BaseException) -> None:
        database = FakeDatabase()
        database.batch_failures.append(error)
        with pytest.raises(SinkError) as caught:
            make_sink(database).send("event", rows_of(make_event(T0)))
        assert caught.value.retryable is True
        assert not is_permanent(error)

    @pytest.mark.parametrize(
        "error",
        [
            fake_db.ProgrammingError("syntax error"),
            fake_db.UndefinedTable("relation event does not exist", "relation does not exist"),
            fake_db.IntegrityError("a constraint failed"),
            fake_db.DataError("a value is out of range"),
            fake_db.NotSupportedError("not supported"),
        ],
        ids=lambda e: type(e).__name__,
    )
    def test_a_problem_that_retrying_cannot_fix_is_permanent(self, error: BaseException) -> None:
        database = FakeDatabase()
        database.batch_failures.append(error)
        with pytest.raises(SinkError) as caught:
            make_sink(database).send("event", rows_of(make_event(T0)))
        assert caught.value.retryable is False
        assert is_permanent(error)

    def test_a_connection_error_drops_the_connection_and_the_next_batch_connects_again(
        self,
    ) -> None:
        database = FakeDatabase()
        database.batch_failures.append(fake_db.AdminShutdown("restart"))
        sink = make_sink(database)
        with pytest.raises(SinkError):
            sink.send("event", rows_of(make_event(T0)))
        assert database.connections[0].closed
        assert database.calls("rollback") == []  # a broken connection cannot roll back
        sink.send("event", rows_of(make_event(T0)))
        assert len(database.connections) == 2
        assert len(database.rows["event"]) == 1

    def test_a_refused_batch_rolls_back_and_keeps_the_connection(self) -> None:
        database = FakeDatabase()
        database.batch_failures.append(fake_db.DataError("out of range"))
        sink = make_sink(database)
        with pytest.raises(SinkError):
            sink.send("event", rows_of(make_event(T0)))
        assert len(database.calls("rollback")) == 1
        assert not database.connections[0].closed
        sink.send("event", rows_of(make_event(T0)))
        assert len(database.connections) == 1

    def test_a_failure_while_preparing_the_table_rolls_back_and_the_next_batch_tries_again(
        self,
    ) -> None:
        database = FakeDatabase()
        database.execute_failure = ("CREATE TABLE", fake_db.ProgrammingError("permission denied"))
        sink = make_sink(database)
        with pytest.raises(SinkError) as caught:
            sink.send("event", rows_of(make_event(T0)))
        assert caught.value.retryable is False
        assert database.calls("rollback")
        database.execute_failure = None
        sink.send("event", rows_of(make_event(T0)))  # the table was not marked ready
        assert len(database.rows["event"]) == 1

    def test_a_failure_while_detecting_the_extension_does_not_leave_a_half_ready_connection(
        self,
    ) -> None:
        database = FakeDatabase()
        database.execute_failure = ("pg_extension", fake_db.OperationalError("timeout"))
        sink = make_sink(database)
        with pytest.raises(SinkError):
            sink.send("event", rows_of(make_event(T0)))
        assert database.connections[0].closed
        database.execute_failure = None
        sink.send("event", rows_of(make_event(T0)))
        assert any("create_hypertable" in s for s in database.statements())

    def test_a_failed_connection_is_retryable_and_nothing_is_kept(self) -> None:
        database = FakeDatabase()
        database.connect_failures.append(fake_db.OperationalError("could not connect"))
        sink = make_sink(database)
        with pytest.raises(SinkError) as caught:
            sink.send("event", rows_of(make_event(T0)))
        assert caught.value.retryable is True
        sink.send("event", rows_of(make_event(T0)))
        assert len(database.rows["event"]) == 1

    def test_a_row_that_does_not_fit_its_declaration_is_permanent_and_never_reaches_the_database(
        self,
    ) -> None:
        database = FakeDatabase()
        bad = StoredRow(1, {**make_event(T0).to_row(), "t_utc_ns": "not a number"})
        broken = StoredRow(2, {**make_health(T0).to_row(), "heater_duty": "abc"})
        for record_type, row in (("event", bad), ("health", broken)):
            with pytest.raises(SinkError) as caught:
                make_sink(database).send(record_type, [row])
            assert caught.value.retryable is False
        undecodable = StoredRow(1, {**sample_record("star_list").to_row(), "data": "not base64!"})
        with pytest.raises(SinkError) as caught:
            make_sink(database).send("star_list", [undecodable])
        assert caught.value.retryable is False
        assert database.log == []

    def test_a_message_names_the_error_class_and_the_state_and_nothing_the_server_said(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        database = FakeDatabase()
        error = fake_db.UndefinedTable(
            'relation "event" does not exist; the query was INSERT INTO event ... '
            "Failing row contains (PRIVATE-DETAIL)",
            'role "private-user" has no table',
        )
        database.batch_failures.append(error)
        with caplog.at_level("DEBUG"), pytest.raises(SinkError) as caught:
            make_sink(database).send("event", rows_of(make_event(T0)))
        text = str(caught.value)
        assert "UndefinedTable" in text
        assert "SQLSTATE 42P01" in text
        for private in ("PRIVATE-DETAIL", "private-user", "INSERT", "has no table"):
            assert private not in text
        # The server's message goes to the debug log, and the query and the row values never do.
        assert "private-user" in caplog.text
        assert "PRIVATE-DETAIL" not in caplog.text
        assert "INSERT" not in caplog.text

    def test_a_connection_error_message_holds_no_address_or_credential(self) -> None:
        database = FakeDatabase()
        database.connect_failures.append(
            fake_db.OperationalError(
                'connection to server at "db.example.org" (203.0.113.7), port 5432 failed: '
                "PRIVATE-DETAIL"
            )
        )
        with pytest.raises(SinkError) as caught:
            make_sink(database).send("event", rows_of(make_event(T0)))
        text = str(caught.value)
        assert "PRIVATE-DETAIL" not in text
        assert "db.example.org" not in text
        assert "203.0.113.7" not in text

    def test_close_survives_a_connection_that_fails_to_close(self) -> None:
        database = FakeDatabase()
        sink = make_sink(database)
        sink.send("event", rows_of(make_event(T0)))

        def explode() -> None:
            raise fake_db.OperationalError("already gone")

        database.connections[0].close = explode  # type: ignore[method-assign]
        sink.close()
        sink.send("event", rows_of(make_event(T0 + 1)))
        assert len(database.connections) == 2


class TestWithTheForwarder:
    def test_an_outage_and_a_return_leave_every_row_in_the_database_once(
        self, tmp_path: Any
    ) -> None:
        clock = VirtualClock(T0)
        database = FakeDatabase()
        with Store.open(tmp_path / "results.sqlite") as store:
            store.write_many([make_event(T0 + n) for n in range(30)])
            store.write_many([make_health(T0 + n * NS_PER_S) for n in range(120)])
            sink = TimescaleSink("archive", database.connect, max_batch_rows=25)
            forwarder = Forwarder(
                store,
                [sink],
                clock,
                ForwarderConfig(jitter=0, backoff_initial_s=1),
                rng=random.Random(1),
            )
            database.batch_failures.extend(
                [fake_db.AdminShutdown("restart"), fake_db.OperationalError("lost"), None, None]
            )
            database.batch_failures.extend([fake_db.AdminShutdown("again")])
            for _ in range(200):
                forwarder.run_once()
                if not any(forwarder.sink_backlog().values()):
                    break
                clock.advance(2)
            assert forwarder.sink_backlog() == {"archive": 0}
        assert len(database.rows["event"]) == 30
        assert len(database.rows["health"]) == 120
        expected_times = {T0 + n * NS_PER_S for n in range(120)}
        assert {key[1] for key in database.rows["health"]} == expected_times


class TestDriver:
    CONFIG = TimescaleSinkConfig(
        kind="timescale",
        host="timescale.example.org",
        port=6543,
        database="seeing",
        user="example-user",
        sslmode="require",
        connect_timeout_s=2.5,
    )

    def test_the_connect_function_passes_the_settings_to_psycopg(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_connect(**kwargs: Any) -> str:
            calls.append(kwargs)
            return "connection"

        module = ModuleType("psycopg")
        module.__dict__["connect"] = fake_connect
        connect = make_psycopg_connect(self.CONFIG, CODE, import_module=lambda name: module)
        assert calls == []  # building the function does not connect
        assert connect() == "connection"
        assert calls == [
            {
                "host": "timescale.example.org",
                "port": 6543,
                "dbname": "seeing",
                "user": "example-user",
                "password": CODE,
                "sslmode": "require",
                "connect_timeout": 3,  # rounded up to whole seconds
            }
        ]

    def test_a_missing_driver_is_a_configuration_error_that_names_the_extra(self) -> None:
        def missing(name: str) -> Any:
            raise ModuleNotFoundError(name)

        with pytest.raises(ConfigError, match="timescale"):
            make_psycopg_connect(self.CONFIG, None, import_module=missing)

    def test_the_real_driver_imports_and_builds_a_connect_function(self) -> None:
        pytest.importorskip("psycopg")
        connect = make_psycopg_connect(self.CONFIG, None)  # nothing connects until you call it
        assert callable(connect)

    def test_the_sink_module_imports_without_the_driver(self) -> None:
        code = "import sys; sys.modules['psycopg'] = None; import seeingmon.sinks.timescale"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


def test_the_sample_event_round_trips_through_the_parameters() -> None:
    record = make_event(T0, detail={"a": 1})
    names = [c.name for c in sink_mapping("event").timescale.columns]
    parameters = dict(zip(names, row_parameters("event", record.to_row()), strict=True))
    rebuilt = EventRecord.from_row(
        {
            **parameters,
            "provenance": json.loads(parameters["provenance"]),
            "detail": json.loads(parameters["detail"]),
        }
    )
    assert rebuilt == record
