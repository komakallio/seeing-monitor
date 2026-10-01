"""The shared key of the local connections."""

from __future__ import annotations

from pathlib import Path

import pytest

from seeingmon.services.ipc.errors import IpcConfigError
from seeingmon.services.ipc.keys import ConnectionKey, load_connection_key

KEY_A = "a-test-key-with-32-characters-long"
KEY_B = "another-test-key-of-sufficient-len"


class TestConnectionKey:
    def test_whitespace_around_the_text_is_not_part_of_the_key(self) -> None:
        assert ConnectionKey.from_text(f"  {KEY_A}\n") == ConnectionKey.from_text(KEY_A)

    def test_a_short_key_is_refused_without_echoing_it(self) -> None:
        with pytest.raises(IpcConfigError) as raised:
            ConnectionKey.from_text("tooshort")
        assert "tooshort" not in str(raised.value)
        assert "16" in str(raised.value)

    def test_a_long_key_is_refused(self) -> None:
        with pytest.raises(IpcConfigError):
            ConnectionKey.from_text("k" * 2000)

    def test_the_value_stays_out_of_repr_and_str(self) -> None:
        key = ConnectionKey.from_text(KEY_A)
        assert KEY_A not in repr(key)
        assert KEY_A not in str(key)
        assert KEY_A not in repr([key])

    def test_signatures_depend_on_the_key_and_the_parts(self) -> None:
        a = ConnectionKey.from_text(KEY_A)
        b = ConnectionKey.from_text(KEY_B)
        signature = a.sign(b"label", b"nonce")
        assert len(signature) == 32
        assert a.verify(signature, b"label", b"nonce")
        assert not b.verify(signature, b"label", b"nonce")
        assert not a.verify(signature, b"label", b"other")

    def test_parts_cannot_shift_between_each_other(self) -> None:
        key = ConnectionKey.from_text(KEY_A)
        assert key.sign(b"ab", b"c") != key.sign(b"a", b"bc")

    def test_keys_compare_and_hash_by_value(self) -> None:
        assert ConnectionKey.from_text(KEY_A) == ConnectionKey.from_text(KEY_A)
        assert ConnectionKey.from_text(KEY_A) != ConnectionKey.from_text(KEY_B)
        assert len({ConnectionKey.from_text(KEY_A), ConnectionKey.from_text(KEY_A)}) == 1


class TestLoadConnectionKey:
    def test_the_configured_value_comes_first(self, tmp_path: Path) -> None:
        file = tmp_path / "key"
        file.write_text(KEY_B)
        key = load_connection_key(value=KEY_A, file=str(file), env={})
        assert key == ConnectionKey.from_text(KEY_A)

    def test_a_systemd_credential_is_next(self, tmp_path: Path) -> None:
        (tmp_path / "seeingmon-key").write_text(KEY_B + "\n")
        key = load_connection_key(
            credential="seeingmon-key",
            file=None,
            env={"CREDENTIALS_DIRECTORY": str(tmp_path)},
        )
        assert key == ConnectionKey.from_text(KEY_B)

    def test_a_key_file_is_the_last_source(self, tmp_path: Path) -> None:
        file = tmp_path / "key.txt"
        file.write_text(f"{KEY_A}\r\n")
        assert load_connection_key(file=str(file), env={}) == ConnectionKey.from_text(KEY_A)

    def test_a_missing_credential_falls_through_to_the_file(self, tmp_path: Path) -> None:
        file = tmp_path / "key.txt"
        file.write_text(KEY_A)
        key = load_connection_key(
            credential="absent", file=str(file), env={"CREDENTIALS_DIRECTORY": str(tmp_path)}
        )
        assert key == ConnectionKey.from_text(KEY_A)

    def test_no_source_names_the_ways_to_set_one(self) -> None:
        with pytest.raises(IpcConfigError, match="no connection key") as raised:
            load_connection_key(value="  ", credential="x", file="", env={})
        assert "SEEINGMON_SERVICES__CONNECTION_KEY" in str(raised.value)

    def test_an_unreadable_file_is_a_config_error_that_hides_the_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        with pytest.raises(IpcConfigError) as raised:
            load_connection_key(file=str(missing), env={})
        assert str(missing) not in str(raised.value)

    def test_a_short_key_in_a_file_is_refused(self, tmp_path: Path) -> None:
        file = tmp_path / "short"
        file.write_text("abc")
        with pytest.raises(IpcConfigError, match="16"):
            load_connection_key(file=str(file), env={})

    def test_a_file_that_is_not_text_is_refused(self, tmp_path: Path) -> None:
        file = tmp_path / "binary"
        file.write_bytes(b"\xff\xfe" * 20)
        with pytest.raises(IpcConfigError, match="UTF-8"):
            load_connection_key(file=str(file), env={})
