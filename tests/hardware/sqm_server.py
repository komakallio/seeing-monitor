"""A TCP server that imitates an SQM-LE, with scripted faults.

`FakeSqmServer` listens on the loopback interface in a thread. Each connection takes the next
behavior from `script` (or `respond` when the script is empty), so a test makes the first
connection drop, the second send garbage, and the third answer. The server follows the response
formats that `seeingmon.hardware.sqm` documents, which are unverified against a real unit.
"""

from __future__ import annotations

import contextlib
import socket
import threading
from collections import deque

SERIAL = "00000413"  # a made-up serial number, to check that the reader never stores it


def reading_line(kind: str, magnitude: float, temperature_c: float) -> bytes:
    sign = "-" if magnitude < 0 else " "
    t_sign = "-" if temperature_c < 0 else " "
    return (
        f"{kind},{sign}{abs(magnitude):05.2f}m,0000022921Hz,0000000020c,0000000.000s,"
        f"{t_sign}{abs(temperature_c):05.1f}C\r\n"
    ).encode("ascii")


class FakeSqmServer:
    """Behaviors for a connection:

    - `respond`: answer every command until the client closes.
    - `drop`: close the connection at once.
    - `silent_close`: read a command, then close without an answer.
    - `stall`: read nothing back and wait, so the client times out.
    - `garbage`: answer every command with binary junk and a line end.
    - `wrong_line`: answer with a line that has the right letter and no reading.
    - `partial`: send half of a reading line, then close.
    - `long`: send 2,000 bytes with no line end.
    - `once`: answer one command, then close.
    """

    def __init__(self, *, magnitude: float = 21.37, temperature_c: float = 3.5) -> None:
        self.magnitude = magnitude
        self.temperature_c = temperature_c
        self.script: deque[str] = deque()
        self.commands: list[bytes] = []
        self.connections = 0
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sockets: list[socket.socket] = []
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self._listener.settimeout(0.05)
        self.port: int = self._listener.getsockname()[1]
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _answer(self, command: bytes) -> bytes:
        if command == b"rx":
            return reading_line("r", self.magnitude, self.temperature_c)
        if command == b"ux":
            return reading_line("u", self.magnitude, self.temperature_c)
        if command == b"ix":
            return f"i,00000002,00000003,00000001,{SERIAL}\r\n".encode("ascii")
        if command == b"cx":
            return b"c,00000017.60m,0000000.000s, 039.4C,00000008.71m, 039.4C\r\n"
        return b"?\r\n"

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.connections += 1
            behavior = self.script.popleft() if self.script else "respond"
            self._sockets.append(connection)
            thread = threading.Thread(target=self._serve, args=(connection, behavior), daemon=True)
            self._threads.append(thread)
            thread.start()

    def _serve(self, connection: socket.socket, behavior: str) -> None:
        connection.settimeout(0.05)
        try:
            if behavior == "drop":
                return
            while not self._stop.is_set():
                try:
                    command = connection.recv(16)
                except TimeoutError:
                    continue
                if not command:
                    return  # the client closed
                self.commands.append(command)
                if behavior == "respond":
                    connection.sendall(self._answer(command))
                elif behavior == "once":
                    connection.sendall(self._answer(command))
                    return
                elif behavior == "silent_close":
                    return
                elif behavior == "garbage":
                    connection.sendall(b"\x00\xff\xfe not a reading \x01\r\n")
                elif behavior == "wrong_line":
                    connection.sendall(b"r,not,a,reading\r\n")
                elif behavior == "partial":
                    connection.sendall(b"r, 21.")
                    return
                elif behavior == "long":
                    connection.sendall(b"r" * 2000)
                    return
                # `stall`: keep waiting
        except OSError:
            return
        finally:
            with contextlib.suppress(OSError):
                connection.close()

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(OSError):
            self._listener.close()
        self._thread.join(5.0)
        for thread in self._threads:
            thread.join(5.0)
        for connection in self._sockets:
            with contextlib.suppress(OSError):
                connection.close()
