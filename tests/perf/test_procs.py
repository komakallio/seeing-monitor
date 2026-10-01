"""The memory and CPU time of other processes, read from outside."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from seeingmon.perf import procs

# Holds 60 MB that it touched, burns 0.8 s of CPU in a second thread, and waits for a line.
HOLD_AND_BURN = """
import sys, threading, time

data = bytearray(60 * 1024 * 1024)
for index in range(0, len(data), 4096):
    data[index] = 1  # touch every page, so that the memory is resident

burned = threading.Event()
release = threading.Event()


def work():
    end = time.process_time() + 0.8
    while time.process_time() < end:
        pass
    burned.set()
    release.wait()


worker = threading.Thread(target=work)
worker.start()
burned.wait()
print("done", flush=True)
sys.stdin.readline()
release.set()
worker.join()
"""

# Starts a child that waits for its input to close, prints its ID, and waits for a line.
SPAWN_A_CHILD = """
import subprocess, sys

child = subprocess.Popen(
    [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
)
print(child.pid, flush=True)
sys.stdin.readline()
child.stdin.close()
child.wait()
"""


@contextmanager
def running(code: str) -> Iterator[subprocess.Popen[str]]:
    """A child process that runs `code`. It ends when the context ends."""
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdin is not None
    assert child.stdout is not None
    try:
        yield child
    finally:
        try:
            child.stdin.write("\n")
            child.stdin.flush()
        except OSError:
            pass
        child.stdin.close()
        child.wait(timeout=30)
        child.stdout.close()


class TestParseStat:
    LINE = "1234 (python3 (worker)) S 77 1234 1234 0 -1 4194560 100 0 0 0 250 50 0 0 20 0 3 0 99"

    def test_it_reads_the_parent_and_the_cpu_ticks(self) -> None:
        assert procs.parse_stat(self.LINE) == (77, 250, 50)

    def test_a_name_with_parentheses_and_spaces_does_not_shift_the_fields(self) -> None:
        line = "9 (a ) b (c)) R 5 9 9 0 -1 0 0 0 0 0 11 22 0 0 20 0 1 0 1"
        assert procs.parse_stat(line) == (5, 11, 22)

    @pytest.mark.parametrize("text", ["", "no parentheses here", "1 (x) S 1 2 3", "1 (x) S a b c"])
    def test_text_without_the_fields_gives_none(self, text: str) -> None:
        assert procs.parse_stat(text) is None

    def test_a_field_that_is_not_a_number_gives_none(self) -> None:
        line = "9 (a) R x 9 9 0 -1 0 0 0 0 0 11 22 0 0 20 0 1 0 1"
        assert procs.parse_stat(line) is None


class TestReadProcess:
    def test_it_reads_the_peak_memory_and_the_cpu_time_of_a_child(self) -> None:
        with running(HOLD_AND_BURN) as child:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "done"
            real = procs.real_process(child.pid, procs.parent_map())
            reading = procs.read_process(real)
        assert reading is not None
        assert reading.pid == real
        assert reading.peak_rss_bytes is not None
        assert reading.peak_rss_bytes >= 55_000_000  # the 60 MB that the child touched
        assert reading.cpu_ns is not None
        assert reading.cpu_ns >= 0.5e9  # it burned 0.8 s

    def test_the_peak_never_falls_below_the_current_size(self) -> None:
        with running(HOLD_AND_BURN) as child:
            assert child.stdout is not None
            child.stdout.readline()
            reading = procs.read_process(procs.real_process(child.pid, procs.parent_map()))
        assert reading is not None
        assert reading.rss_bytes is not None
        assert reading.peak_rss_bytes is not None
        assert reading.peak_rss_bytes >= reading.rss_bytes

    def test_a_process_that_does_not_exist_reads_as_none(self) -> None:
        assert procs.read_process(2_147_483_000) is None

    def test_this_process_reads_too(self) -> None:
        reading = procs.read_process(os.getpid())
        assert reading is not None
        assert reading.peak_rss_bytes is not None
        assert reading.peak_rss_bytes > 10_000_000
        assert reading.cpu_ns is not None
        assert reading.cpu_ns > 0

    def test_the_cpu_time_of_a_process_never_falls(self) -> None:
        first = procs.read_process(os.getpid())
        sum(index * index for index in range(200_000))
        second = procs.read_process(os.getpid())
        assert first is not None
        assert second is not None
        assert first.cpu_ns is not None
        assert second.cpu_ns is not None
        assert second.cpu_ns >= first.cpu_ns


class TestThreads:
    @pytest.mark.skipif(sys.platform != "linux", reason="only Linux gives a CPU time by thread")
    def test_it_gives_the_cpu_time_of_each_thread(self) -> None:
        with running(HOLD_AND_BURN) as child:
            assert child.stdout is not None
            child.stdout.readline()
            by_thread = procs.thread_cpu_ns(child.pid)
        assert len(by_thread) >= 2  # the main thread and the one that burned
        assert max(by_thread.values()) >= 0.5e9

    @pytest.mark.skipif(sys.platform == "linux", reason="Linux gives a CPU time by thread")
    def test_other_systems_give_nothing(self) -> None:
        assert procs.thread_cpu_ns(os.getpid()) == {}

    def test_a_process_that_does_not_exist_has_no_threads(self) -> None:
        assert procs.thread_cpu_ns(2_147_483_000) == {}


class TestDescription:
    @pytest.mark.skipif(sys.platform != "linux", reason="only Linux gives a command line")
    def test_the_command_line_has_the_words_with_a_space_between(self) -> None:
        with running("import sys; sys.stdin.readline()") as child:
            line = procs.command_line(child.pid)
            deadline = time.monotonic() + 5.0
            while not line and time.monotonic() < deadline:
                time.sleep(0.01)  # the line reads as empty for a moment after the program starts
                line = procs.command_line(child.pid)
        assert line is not None
        assert line.endswith("import sys; sys.stdin.readline()")

    @pytest.mark.skipif(sys.platform == "linux", reason="Linux gives a command line")
    def test_other_systems_give_no_command_line(self) -> None:
        assert procs.command_line(os.getpid()) is None

    @pytest.mark.skipif(sys.platform not in ("linux", "win32"), reason="no reader on this system")
    def test_the_image_of_this_process_is_a_python_program(self) -> None:
        image = procs.image_path(os.getpid())
        assert image is not None
        assert "python" in image.replace("\\", "/").rsplit("/", 1)[-1].lower()

    def test_a_process_that_does_not_exist_has_no_description(self) -> None:
        assert procs.command_line(2_147_483_000) is None
        assert procs.image_path(2_147_483_000) is None


class TestTree:
    def test_the_parent_map_knows_this_process_and_its_parent(self) -> None:
        parents = procs.parent_map()
        assert os.getpid() in parents
        assert parents[os.getpid()] == os.getppid()

    def test_a_grandchild_is_below_the_child_and_the_child_of_the_real_process(self) -> None:
        with running(SPAWN_A_CHILD) as child:
            assert child.stdout is not None
            grandchild = int(child.stdout.readline())
            parents = procs.parent_map()
            real = procs.real_process(child.pid, parents)
            assert grandchild in procs.children_of(real, parents)
            assert grandchild in procs.descendants_of(child.pid, parents)
            assert os.getpid() not in procs.descendants_of(child.pid, parents)

    def test_children_and_descendants_of_a_made_up_map(self) -> None:
        parents = {2: 1, 3: 1, 4: 2, 5: 4, 9: 8}
        assert procs.children_of(1, parents) == [2, 3]
        assert procs.descendants_of(1, parents) == [2, 3, 4, 5]
        assert procs.descendants_of(9, parents) == []

    def test_a_process_that_is_its_own_parent_does_not_loop(self) -> None:
        parents = {0: 0, 4: 0}  # the system process of Windows has itself as parent
        assert procs.children_of(0, parents) == [4]
        assert procs.descendants_of(0, parents) == [4]


class TestVenvLauncher:
    """`real_process` follows a chain of launchers to the process that does the work."""

    @pytest.fixture
    def launchers(self, monkeypatch: pytest.MonkeyPatch) -> set[int]:
        found: set[int] = set()
        monkeypatch.setattr(procs, "is_venv_launcher", lambda pid: pid in found)
        return found

    def test_a_process_that_is_not_a_launcher_is_its_own_real_process(
        self, launchers: set[int]
    ) -> None:
        assert procs.real_process(10, {10: 1, 11: 10}) == 10

    def test_a_launcher_gives_its_only_child(self, launchers: set[int]) -> None:
        launchers.add(10)
        assert procs.real_process(10, {10: 1, 11: 10}) == 11

    def test_a_chain_of_launchers_gives_the_last_child(self, launchers: set[int]) -> None:
        launchers.update({10, 11})
        assert procs.real_process(10, {10: 1, 11: 10, 12: 11, 13: 12}) == 12

    def test_a_launcher_without_one_child_stays_where_it_is(self, launchers: set[int]) -> None:
        launchers.add(10)
        assert procs.real_process(10, {10: 1}) == 10
        assert procs.real_process(10, {10: 1, 11: 10, 12: 10}) == 10

    def test_a_process_that_does_not_exist_is_no_launcher(self) -> None:
        assert not procs.is_venv_launcher(2_147_483_000)

    def test_this_process_is_its_own_real_process(self) -> None:
        assert procs.real_process(os.getpid(), procs.parent_map()) == os.getpid()

    @pytest.mark.skipif(sys.platform != "win32", reason="only Windows has a launcher")
    @pytest.mark.skipif(sys.prefix == sys.base_prefix, reason="the tests run outside a venv")
    def test_the_real_process_of_a_venv_python_is_it_or_below_it(self) -> None:
        with running(HOLD_AND_BURN) as child:
            assert child.stdout is not None
            child.stdout.readline()
            parents = procs.parent_map()
            real = procs.real_process(child.pid, parents)
            assert real == child.pid or real in procs.descendants_of(child.pid, parents)
            assert not procs.is_venv_launcher(real) or not procs.children_of(real, parents)
