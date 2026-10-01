"""Addresses of the local connections."""

from __future__ import annotations

import pytest

from seeingmon.services.ipc.endpoint import (
    FAMILY_INET,
    FAMILY_PIPE,
    FAMILY_UNIX,
    PIPE_PREFIX,
    Endpoint,
)
from seeingmon.services.ipc.errors import IpcConfigError, IpcError


class TestParse:
    def test_a_linux_address_is_a_unix_socket_path(self) -> None:
        endpoint = Endpoint.parse("/run/example/acquire.sock", platform="linux")
        assert endpoint == Endpoint("/run/example/acquire.sock", FAMILY_UNIX)
        assert endpoint.path == "/run/example/acquire.sock"
        assert str(endpoint) == "/run/example/acquire.sock"

    def test_a_windows_name_gets_the_local_pipe_prefix(self) -> None:
        endpoint = Endpoint.parse("seeingmon-acquire", platform="win32")
        assert endpoint == Endpoint(PIPE_PREFIX + "seeingmon-acquire", FAMILY_PIPE)
        assert endpoint.path is None

    def test_a_windows_pipe_path_is_kept(self) -> None:
        text = PIPE_PREFIX + "seeingmon-acquire"
        assert Endpoint.parse(text, platform="win32").address == text

    def test_a_remote_pipe_is_refused(self) -> None:
        with pytest.raises(IpcConfigError, match="pipe address"):
            Endpoint.parse("\\\\other\\pipe\\name", platform="win32")

    def test_a_pipe_address_is_refused_on_linux(self) -> None:
        with pytest.raises(IpcConfigError, match="only on Windows"):
            Endpoint.parse(PIPE_PREFIX + "name", platform="linux")

    @pytest.mark.parametrize("text", ["", "   "])
    def test_an_empty_address_is_refused(self, text: str) -> None:
        with pytest.raises(IpcConfigError, match="empty"):
            Endpoint.parse(text, platform="linux")

    def test_a_long_socket_path_is_refused_with_its_limit(self) -> None:
        with pytest.raises(IpcConfigError, match="107 bytes"):
            Endpoint.parse("/" + "a" * 200, platform="linux")

    def test_a_pipe_name_has_a_safe_alphabet(self) -> None:
        with pytest.raises(IpcConfigError):
            Endpoint.parse("bad name\\with\\slashes", platform="win32")

    def test_errors_are_connection_layer_errors(self) -> None:
        with pytest.raises(IpcError):
            Endpoint("a\0b", FAMILY_UNIX)
        with pytest.raises(ValueError, match="family"):
            Endpoint("name", "AF_BOGUS")


class TestDefaults:
    def test_windows_default_is_a_pipe_named_for_the_role(self) -> None:
        endpoint = Endpoint.default("acquire", platform="win32", env={})
        assert endpoint.address == PIPE_PREFIX + "seeingmon-acquire"

    def test_systemd_runtime_directory_comes_first(self) -> None:
        env = {"RUNTIME_DIRECTORY": "/run/svc:/run/other", "XDG_RUNTIME_DIR": "/run/user/1"}
        endpoint = Endpoint.default("core", platform="linux", env=env)
        assert endpoint.address == "/run/svc/core.sock"

    def test_the_user_runtime_directory_is_next(self) -> None:
        endpoint = Endpoint.default("core", platform="linux", env={"XDG_RUNTIME_DIR": "/run/u/1"})
        assert endpoint.address == "/run/u/1/seeingmon/core.sock"

    def test_without_environment_the_system_directory_is_used(self) -> None:
        endpoint = Endpoint.default("acquire", platform="linux", env={})
        assert endpoint.address == "/run/seeingmon/acquire.sock"

    def test_a_trailing_slash_does_not_double(self) -> None:
        endpoint = Endpoint.default("acquire", platform="linux", env={"RUNTIME_DIRECTORY": "/r/"})
        assert endpoint.address == "/r/acquire.sock"

    def test_a_role_must_be_a_plain_name(self) -> None:
        with pytest.raises(IpcConfigError):
            Endpoint.default("../etc", platform="linux", env={})

    def test_a_setting_overrides_the_default_and_empty_selects_it(self) -> None:
        env: dict[str, str] = {}
        configured = Endpoint.from_setting("/x/y.sock", "acquire", platform="linux", env=env)
        assert configured.address == "/x/y.sock"
        default = Endpoint.from_setting("", "acquire", platform="linux", env=env)
        assert default.address == "/run/seeingmon/acquire.sock"


class TestLoopback:
    def test_a_loopback_endpoint_is_a_host_and_port(self) -> None:
        endpoint = Endpoint.loopback(0)
        assert endpoint.family == FAMILY_INET
        assert endpoint.address == ("127.0.0.1", 0)
        assert str(Endpoint.loopback(5000)) == "127.0.0.1:5000"

    def test_a_tcp_endpoint_needs_a_pair(self) -> None:
        with pytest.raises(IpcConfigError, match="pair"):
            Endpoint("127.0.0.1", FAMILY_INET)
