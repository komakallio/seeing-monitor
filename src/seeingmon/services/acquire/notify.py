"""The systemd notifier moved to `seeingmon.services.notify`. This module keeps the old path."""

from __future__ import annotations

from seeingmon.services.notify import SystemdNotifier, _open_unix

__all__ = ["SystemdNotifier", "_open_unix"]
