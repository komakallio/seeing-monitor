"""The SQM-LE reader: the parser, the TCP client on a fake unit, and the polling reader."""

from __future__ import annotations

import socket
import time
from collections.abc import Iterator

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from seeingmon.clock import VirtualClock
from seeingmon.hardware.events import HardwareEvent
from seeingmon.hardware.sqm import (
    SqmCalibration,
    SqmConfig,
    SqmConnectionError,
    SqmInfo,
    SqmLeClient,
    SqmLeReader,
    SqmParseError,
    SqmReading,
    SqmTimeoutError,
    parse_calibration,
    parse_info,
    parse_reading,
)
from seeingmon.records.reference import ReferenceRecord
from tests.hardware.sqm_server import SERIAL, FakeSqmServer

LINE = "r, 06.70m,0000022921Hz,0000000020c,0000000.000s, 039.4C\r\n"


class TestParseReading:
    def test_the_documented_line(self) -> None:
        assert parse_reading(LINE) == SqmReading(
            kind="r",
            magnitude=6.70,
            frequency_hz=22921.0,
            period_count=20.0,
            period_s=0.0,
            temperature_c=39.4,
        )

    def test_the_unaveraged_reading_has_the_letter_u(self) -> None:
        assert parse_reading(LINE.replace("r,", "u,", 1)).kind == "u"

    @pytest.mark.parametrize(
        ("text", "magnitude", "temperature"),
        [
            ("r,-1.23m,0000000100Hz,0000000020c,0000000.000s,-005.5C", -1.23, -5.5),
            ("r,+21.37m,1Hz,2c,3.5s,+10.0C", 21.37, 10.0),
            ("  r , 21.37 m , 22921 Hz , 20 c , 0.000 s , 3.5 C  \r\n", 21.37, 3.5),
            ("R, 21.37m,0000022921hz,0000000020c,0000000.000s, 003.5C", 21.37, 3.5),
            ("\r\nr, 21.37m,0000022921Hz,0000000020c,0000000.000s, 003.5C\r\n\r\n", 21.37, 3.5),
        ],
    )
    def test_tolerates_signs_spaces_case_and_blank_lines(
        self, text: str, magnitude: float, temperature: float
    ) -> None:
        reading = parse_reading(text)
        assert reading.magnitude == pytest.approx(magnitude)
        assert reading.temperature_c == pytest.approx(temperature)

    def test_a_response_with_other_lines_uses_the_last_reading_line(self) -> None:
        text = "r, 10.00m,1Hz,1c,0.1s, 001.0C\r\nstale\r\nr, 21.37m,2Hz,2c,0.2s, 002.0C\r\n"
        assert parse_reading(text).magnitude == pytest.approx(21.37)

    def test_a_line_without_units_is_read_by_position(self) -> None:
        reading = parse_reading("r, 21.37, 22921, 20, 0.000, 3.5")
        assert reading.magnitude == pytest.approx(21.37)
        assert reading.frequency_hz == 22921.0
        assert reading.temperature_c == pytest.approx(3.5)

    def test_a_missing_quantity_is_none_and_the_rest_survives(self) -> None:
        reading = parse_reading("r, 21.37m,0000022921Hz")
        assert reading.magnitude == pytest.approx(21.37)
        assert reading.temperature_c is None
        assert reading.period_s is None

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "\r\n",
            "garbage",
            "\x00\xff not a reading \x01",
            "r,not,a,reading",
            "r,",
            "i,00000002,00000003,00000001,00000413",  # an information line is not a reading
            "r, 99.99m,0000022921Hz,0000000020c,0000000.000s, 039.4C",  # out of range
            "r,-9.50m,0000022921Hz,0000000020c,0000000.000s, 039.4C",
            "r,0000022921Hz,0000000020c",  # units, but no magnitude
            "x, 21.37m",
        ],
    )
    def test_garbage_is_refused(self, text: str) -> None:
        with pytest.raises(SqmParseError):
            parse_reading(text)

    @given(
        magnitude=st.floats(min_value=-4.99, max_value=29.99, allow_nan=False),
        frequency=st.integers(min_value=0, max_value=9_999_999_999),
        counts=st.integers(min_value=0, max_value=9_999_999_999),
        period=st.floats(min_value=0, max_value=9_999_999, allow_nan=False),
        temperature=st.floats(min_value=-99.9, max_value=99.9, allow_nan=False),
    )
    def test_any_line_in_the_documented_format_round_trips(
        self, magnitude: float, frequency: int, counts: int, period: float, temperature: float
    ) -> None:
        sign = "-" if magnitude < 0 else " "
        t_sign = "-" if temperature < 0 else " "
        text = (
            f"r,{sign}{abs(magnitude):05.2f}m,{frequency:010d}Hz,{counts:010d}c,"
            f"{period:011.3f}s,{t_sign}{abs(temperature):05.1f}C\r\n"
        )
        reading = parse_reading(text)
        assert reading.magnitude == pytest.approx(round(magnitude, 2), abs=1e-9)
        assert reading.frequency_hz == frequency
        assert reading.period_count == counts
        assert reading.temperature_c == pytest.approx(round(temperature, 1), abs=1e-9)


class TestParseInfoAndCalibration:
    def test_information_drops_the_serial_number(self) -> None:
        info = parse_info(f"i,00000002,00000003,00000001,{SERIAL}\r\n")
        assert info == SqmInfo(protocol=2, model=3, feature=1)
        assert SERIAL not in repr(info)

    def test_information_needs_three_numbers(self) -> None:
        with pytest.raises(SqmParseError):
            parse_info("i,00000002")
        with pytest.raises(SqmParseError):
            parse_info("r, 21.37m")

    def test_calibration_groups_the_numbers_by_unit_in_arrival_order(self) -> None:
        calibration = parse_calibration(
            "c,00000017.60m,0000000.000s, 039.4C,00000008.71m, 039.4C\r\n"
        )
        assert calibration == SqmCalibration(
            magnitudes=(17.6, 8.71), periods_s=(0.0,), temperatures_c=(39.4, 39.4)
        )

    def test_calibration_without_numbers_is_refused(self) -> None:
        with pytest.raises(SqmParseError):
            parse_calibration("c,nothing here")


@pytest.fixture
def server() -> Iterator[FakeSqmServer]:
    fake = FakeSqmServer()
    try:
        yield fake
    finally:
        fake.close()


def client(server: FakeSqmServer, **options: object) -> SqmLeClient:
    settings: dict[str, object] = {"connect_timeout_s": 2.0, "read_timeout_s": 0.4}
    settings.update(options)
    return SqmLeClient("127.0.0.1", server.port, **settings)  # type: ignore[arg-type]


class TestClient:
    def test_reads_the_information_the_reading_and_the_calibration(
        self, server: FakeSqmServer
    ) -> None:
        unit = client(server)
        assert unit.read_info() == SqmInfo(2, 3, 1)
        reading = unit.read_reading()
        assert (reading.kind, reading.magnitude, reading.temperature_c) == ("r", 21.37, 3.5)
        assert unit.read_reading("ux").kind == "u"
        assert unit.read_calibration().magnitudes == (17.6, 8.71)
        assert server.commands == [b"ix", b"rx", b"ux", b"cx"]  # two letters, no terminator

    def test_a_connection_for_each_request_unless_it_is_persistent(
        self, server: FakeSqmServer
    ) -> None:
        unit = client(server)
        for _ in range(3):
            unit.read_reading()
        assert server.connections == 3
        persistent = client(server, persistent=True)
        for _ in range(3):
            persistent.read_reading()
        persistent.close()
        persistent.close()
        assert server.connections == 4

    def test_a_unit_that_does_not_answer_times_out(self, server: FakeSqmServer) -> None:
        server.script.append("stall")
        started = time.monotonic()
        with pytest.raises(SqmTimeoutError):
            client(server, read_timeout_s=0.3).read_reading()
        assert 0.25 < time.monotonic() - started < 10

    def test_a_refused_connection_is_an_error_without_the_address(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        # Linux refuses at once, and Windows retries until the connect timeout runs out.
        with pytest.raises((SqmConnectionError, SqmTimeoutError)) as raised:
            SqmLeClient("127.0.0.1", port, connect_timeout_s=2.0).read_reading()
        assert str(port) not in str(raised.value)
        assert "127.0.0.1" not in str(raised.value)

    @pytest.mark.parametrize("behavior", ["drop", "silent_close", "partial"])
    def test_a_connection_that_closes_before_the_line_ends_is_an_error(
        self, server: FakeSqmServer, behavior: str
    ) -> None:
        server.script.append(behavior)
        with pytest.raises(SqmConnectionError):
            client(server).read_reading()

    @pytest.mark.parametrize("behavior", ["garbage", "wrong_line", "long"])
    def test_garbage_is_a_parse_error(self, server: FakeSqmServer, behavior: str) -> None:
        server.script.append(behavior)
        with pytest.raises(SqmParseError):
            client(server).read_reading()

    def test_a_failure_closes_the_connection_and_the_next_request_opens_a_new_one(
        self, server: FakeSqmServer
    ) -> None:
        server.script.extend(["garbage", "respond"])
        unit = client(server, persistent=True)
        with pytest.raises(SqmParseError):
            unit.read_reading()
        assert unit.read_reading().magnitude == 21.37
        assert server.connections == 2
        unit.close()

    def test_a_persistent_connection_that_the_unit_closed_recovers_after_one_failure(
        self, server: FakeSqmServer
    ) -> None:
        server.script.append("once")
        unit = client(server, persistent=True)
        assert unit.read_reading().magnitude == 21.37
        with pytest.raises(SqmConnectionError):  # the unit closed the connection after the answer
            unit.read_reading()
        assert unit.read_reading().magnitude == 21.37
        unit.close()


class FakeClient:
    """A stand-in for `SqmLeClient` that returns readings or raises errors in order."""

    def __init__(self, *outcomes: SqmReading | SqmParseError | SqmTimeoutError) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[str] = []
        self.closed = 0
        self.info: SqmInfo | SqmConnectionError = SqmInfo(2, 3, 1)

    def read_reading(self, command: str = "rx") -> SqmReading:
        self.requests.append(command)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, SqmReading):
            return outcome
        raise outcome

    def read_info(self) -> SqmInfo:
        if isinstance(self.info, SqmConnectionError):
            raise self.info
        return self.info

    def close(self) -> None:
        self.closed += 1


GOOD = SqmReading("r", 21.37, 22921.0, 20.0, 0.0, 3.5)


def make_reader(
    client: object, *, events: list[HardwareEvent] | None = None, **overrides: object
) -> tuple[SqmLeReader, VirtualClock]:
    clock = VirtualClock()
    values: dict[str, object] = {"enabled": True, "host": "unit.example.com", **overrides}
    reader = SqmLeReader(
        SqmConfig(**values),
        clock=clock,
        station_id="station-1",
        profile_id="profile-1",
        client=client,  # type: ignore[arg-type]
        on_event=None if events is None else events.append,
    )
    return reader, clock


class TestReader:
    def test_a_reading_becomes_a_reference_record(self) -> None:
        reader, clock = make_reader(FakeClient(GOOD), altitude_deg=45.0, azimuth_deg=0.0)
        record = reader.poll()
        assert isinstance(record, ReferenceRecord)
        assert record.instrument == "sqm_le"
        assert record.source == "fixed"
        assert record.value_mag_arcsec2 == 21.37
        assert record.temperature_c == 3.5
        assert (record.altitude_deg, record.azimuth_deg) == (45.0, 0.0)
        assert (record.station_id, record.profile_id) == ("station-1", "profile-1")
        assert record.t_utc_ns == clock.utc_ns()
        assert record.provenance == {
            "reader": "sqm-le-1",
            "protocol": "2",
            "model": "3",
            "feature": "1",
        }

    def test_the_pointing_and_the_temperature_are_optional(self) -> None:
        reader, _ = make_reader(FakeClient(SqmReading("r", 20.0)))
        record = reader.poll()
        assert record is not None
        assert (record.temperature_c, record.altitude_deg, record.azimuth_deg) == (None, None, None)

    def test_the_request_and_the_instrument_name_follow_the_configuration(self) -> None:
        client_ = FakeClient(GOOD)
        reader, _ = make_reader(client_, request="ux", instrument="sqm-le")
        record = reader.poll()
        assert client_.requests == ["ux"]
        assert record is not None
        assert record.instrument == "sqm-le"

    def test_the_information_request_is_made_once_and_may_fail(self) -> None:
        client_ = FakeClient(GOOD, GOOD)
        client_.info = SqmConnectionError("no answer")
        reader, _ = make_reader(client_)
        first, second = reader.poll(), reader.poll()
        assert first is not None
        assert second is not None
        assert first.provenance == {"reader": "sqm-le-1"}

    def test_no_record_event_or_repr_carries_the_serial_number_or_the_address(self) -> None:
        events: list[HardwareEvent] = []
        reader, _ = make_reader(FakeClient(SqmParseError("bad"), GOOD), events=events)
        reader.poll()
        record = reader.poll()
        text = repr(record) + repr(events)
        assert SERIAL not in text
        assert "unit.example.com" not in text

    def test_the_poll_interval_sets_the_delay_until_a_read_fails(self) -> None:
        reader, _ = make_reader(FakeClient(GOOD), poll_interval_s=45.0)
        reader.poll()
        assert reader.delay_s == 45.0

    def test_failures_back_off_exponentially_up_to_the_cap_and_a_success_resets(self) -> None:
        errors = [SqmTimeoutError("t")] * 9
        reader, _ = make_reader(
            FakeClient(*errors, GOOD), backoff_initial_s=5.0, backoff_max_s=300.0
        )
        delays = []
        for _ in range(9):
            assert reader.poll() is None
            delays.append(reader.delay_s)
        assert delays == [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 300.0, 300.0, 300.0]
        assert reader.failures == 9
        assert reader.poll() is not None
        assert reader.delay_s == 60.0
        assert reader.failures == 0

    def test_a_long_outage_never_overflows_the_backoff(self) -> None:
        reader, _ = make_reader(FakeClient(*[SqmTimeoutError("t")] * 1500))
        for _ in range(1500):
            reader.poll()
        assert reader.delay_s == 300.0

    def test_a_failure_closes_the_connection_and_reports_once_for_the_streak(self) -> None:
        events: list[HardwareEvent] = []
        client_ = FakeClient(SqmTimeoutError("t"), SqmParseError("p"), SqmTimeoutError("t"), GOOD)
        reader, _ = make_reader(client_, events=events)
        for _ in range(4):
            reader.poll()
        assert client_.closed == 3
        assert [event.kind for event in events] == ["sqm.read_failed", "sqm.recovered"]
        assert events[0].detail == {"cause": "SqmTimeoutError"}
        assert events[1].detail == {"failures": 3}

    def test_run_polls_at_the_interval_and_sleeps_in_slices_of_a_second(self) -> None:
        sleeps: list[float] = []
        reader, clock = make_reader(FakeClient(GOOD, GOOD, GOOD), poll_interval_s=2.5)
        original = clock.sleep

        def recording(seconds: float) -> None:
            sleeps.append(seconds)
            original(seconds)

        clock.sleep = recording  # type: ignore[method-assign]
        records: list[ReferenceRecord] = []
        started = clock.monotonic_ns()
        reader.run(lambda: len(records) >= 3, records.append)
        assert len(records) == 3
        assert sleeps[:3] == [1.0, 1.0, 0.5]
        assert (clock.monotonic_ns() - started) / 1e9 == pytest.approx(5.0)  # two full intervals

    def test_run_stops_within_a_second_of_the_request_and_closes(self) -> None:
        client_ = FakeClient(GOOD)
        reader, clock = make_reader(client_, poll_interval_s=60.0)
        records: list[ReferenceRecord] = []
        calls = []

        def should_stop() -> bool:
            calls.append(1)
            return len(calls) > 6  # a few checks, then stop

        started = clock.monotonic_ns()
        reader.run(should_stop, records.append)
        assert len(records) == 1
        assert (clock.monotonic_ns() - started) / 1e9 <= 5.0
        assert client_.closed >= 1


class TestReaderOnAFakeUnit:
    """The reader on its own TCP client against a unit that drops, stalls, and garbles."""

    def test_the_reader_recovers_after_each_kind_of_failure(self, server: FakeSqmServer) -> None:
        server.script.extend(["drop", "stall", "garbage", "respond", "respond"])
        events: list[HardwareEvent] = []
        clock = VirtualClock()
        config = SqmConfig(
            enabled=True,
            host="127.0.0.1",
            port=server.port,
            connect_timeout_s=2.0,
            read_timeout_s=0.3,
            backoff_initial_s=5.0,
        )
        reader = SqmLeReader(
            config, clock=clock, station_id="s", profile_id="p", on_event=events.append
        )
        started = clock.monotonic_ns()
        records: list[ReferenceRecord] = []
        delays: list[float] = []
        while len(records) < 1:
            record = reader.poll()
            delays.append(reader.delay_s)
            if record is not None:
                records.append(record)
            clock.sleep(reader.delay_s)
        assert delays == [5.0, 10.0, 20.0, 60.0]
        assert records[0].value_mag_arcsec2 == 21.37
        assert records[0].provenance["protocol"] == "2"
        assert [event.kind for event in events] == ["sqm.read_failed", "sqm.recovered"]
        assert (clock.monotonic_ns() - started) / 1e9 == pytest.approx(5 + 10 + 20 + 60)
        reader.close()

    def test_the_default_client_uses_the_configured_address_port_and_timeouts(
        self, server: FakeSqmServer
    ) -> None:
        config = SqmConfig(enabled=True, host="127.0.0.1", port=server.port)
        reader = SqmLeReader(config, clock=VirtualClock(), station_id="s", profile_id="p")
        record = reader.poll()
        assert record is not None
        assert record.value_mag_arcsec2 == 21.37
        assert server.commands == [b"rx", b"ix"]


class TestConfig:
    def test_the_defaults_describe_a_disabled_reader(self) -> None:
        config = SqmConfig()
        assert (config.enabled, config.port, config.instrument) == (False, 10001, "sqm_le")
        assert config.poll_interval_s == 60.0
        assert (config.request, config.persistent) == ("rx", False)

    @pytest.mark.parametrize(
        "values",
        [
            {"enabled": True},  # no host
            {"port": 0},
            {"port": 70000},
            {"poll_interval_s": 0.5},
            {"backoff_initial_s": 10.0, "backoff_max_s": 5.0},
            {"request": "cx"},
            {"altitude_deg": 91.0},
            {"azimuth_deg": 361.0},
            {"surprise": 1},
        ],
    )
    def test_invalid_values_are_refused(self, values: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            SqmConfig(**values)
