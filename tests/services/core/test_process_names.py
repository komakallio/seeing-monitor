"""The names that the two worker processes of `core` give themselves."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from seeingmon.services.core.process_names import (
    ALIGNMENT_WORKER_NAME,
    MAX_NAME_BYTES,
    PR_SET_NAME,
    SURVEY_WORKER_NAME,
    name_process,
    prctl_name,
)


class StandInLibc:
    """The one function of the C library that the name needs, with a result to choose."""

    def __init__(self, result: int = 0) -> None:
        self.result = result
        self.calls: list[tuple[int, bytes, int, int, int]] = []

    def prctl(self, option: int, name: bytes, a: int, b: int, c: int) -> int:
        self.calls.append((option, name, a, b, c))
        return self.result


def test_the_two_workers_have_different_names_that_the_kernel_keeps_whole() -> None:
    assert SURVEY_WORKER_NAME != ALIGNMENT_WORKER_NAME
    for name in (SURVEY_WORKER_NAME, ALIGNMENT_WORKER_NAME):
        assert name.isascii()
        assert len(name.encode("ascii")) <= MAX_NAME_BYTES
    assert PR_SET_NAME == 15  # the number in `linux/prctl.h`, which no header supplies here


def test_the_calling_thread_gets_the_name_through_prctl() -> None:
    libc = StandInLibc()
    assert prctl_name(libc, ALIGNMENT_WORKER_NAME) == f"process name {ALIGNMENT_WORKER_NAME}"
    assert libc.calls == [(PR_SET_NAME, ALIGNMENT_WORKER_NAME.encode("ascii"), 0, 0, 0)]


def test_a_name_that_is_too_long_is_cut_where_the_kernel_cuts_it() -> None:
    libc = StandInLibc()
    message = prctl_name(libc, "a-name-of-more-than-fifteen-bytes")
    assert libc.calls[0][1] == b"a-name-of-more-"
    assert message == "process name a-name-of-more-"


def test_a_call_that_the_system_refuses_says_so_and_never_raises() -> None:
    message = prctl_name(StandInLibc(result=-1), SURVEY_WORKER_NAME)
    assert message == "the call failed, so the process keeps its name"


@pytest.mark.skipif(sys.platform == "linux", reason="Linux names a process")
def test_another_platform_keeps_the_name_of_the_interpreter() -> None:
    assert name_process(SURVEY_WORKER_NAME) == "not supported on this platform"


@pytest.mark.skipif(sys.platform != "linux", reason="the name belongs to Linux")
class TestOnLinux:
    def comm_after_naming(self, name: str) -> list[str]:
        """Name a child process, and read what the system shows for it. The test keeps its own."""
        code = (
            "from seeingmon.services.core.process_names import name_process;"
            f"print(name_process({name!r}));"
            "print(open('/proc/thread-self/comm').read().strip())"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True
        )
        assert "Traceback" not in result.stderr
        return result.stdout.split("\n")

    def test_the_system_shows_the_name_of_a_worker(self) -> None:
        said, shown, *_ = self.comm_after_naming(ALIGNMENT_WORKER_NAME)
        assert said == f"process name {ALIGNMENT_WORKER_NAME}"
        assert shown == ALIGNMENT_WORKER_NAME

    def test_the_system_cuts_a_long_name(self) -> None:
        _, shown, *_ = self.comm_after_naming("a-name-of-more-than-fifteen-bytes")
        assert shown == "a-name-of-more-"

    def test_a_library_that_does_not_load_leaves_the_name_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ctypes

        def broken(*args: Any, **kwargs: Any) -> None:
            raise OSError("no library")

        monkeypatch.setattr(ctypes, "CDLL", broken)
        assert (
            name_process(SURVEY_WORKER_NAME)
            == "the call raised OSError, so the process keeps its name"
        )


def test_the_module_imports_nothing_heavy() -> None:
    """The performance tooling and the worker initializers import it before the survey stack."""
    code = (
        "import sys;"
        "import seeingmon.services.core.process_names;"
        "print(sorted(m for m in ('numpy', 'scipy', 'sep', 'astropy') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True
    )
    assert result.stdout.strip() == "[]"
