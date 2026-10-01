"""Result sinks. The interface is in `seeingmon.sinks.base`."""

from __future__ import annotations

from seeingmon.sinks.base import Sink, SinkError, StoredRow

__all__ = ["Sink", "SinkError", "StoredRow"]
