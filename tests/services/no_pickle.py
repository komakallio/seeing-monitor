"""Make any use of pickle fail, and record it.

`multiprocessing.connection.Connection.send` and `recv` pickle their argument, and so do
`pickle` and `multiprocessing.reduction.ForkingPickler`. `install` replaces all of them with
functions that record the call and raise `AssertionError`, so a test sees a pickle even when a
thread swallows the exception. The subprocess tests call `install` in the child process too,
through `python -c`.
"""

from __future__ import annotations

import multiprocessing.connection as connection
import multiprocessing.reduction as reduction
import pickle
from collections.abc import Callable
from typing import Any

CALLS: list[str] = []


def _forbidden(name: str) -> Callable[..., Any]:
    def replacement(*args: Any, **kwargs: Any) -> Any:
        CALLS.append(name)
        raise AssertionError(f"{name} ran, but pickle must not cross a process boundary")

    return replacement


def install() -> Callable[[], None]:
    """Forbid pickle in this process. Returns a function that restores the originals."""
    base = connection.Connection.__mro__[1]  # the class that defines `send` and `recv`
    targets: list[tuple[Any, str]] = [
        (base, "send"),
        (base, "recv"),
        (pickle, "dumps"),
        (pickle, "loads"),
        (pickle, "dump"),
        (pickle, "load"),
        (pickle, "Pickler"),
        (pickle, "Unpickler"),
        (reduction.ForkingPickler, "dumps"),
        (reduction.ForkingPickler, "loads"),
    ]
    originals = [(owner, name, owner.__dict__[name]) for owner, name in targets]
    for owner, name in targets:
        setattr(owner, name, _forbidden(f"{getattr(owner, '__name__', owner)}.{name}"))

    def restore() -> None:
        for owner, name, original in originals:
            setattr(owner, name, original)

    return restore
