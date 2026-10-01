"""The connection layer: authenticated local connections, RPC, and frame streams.

Built on `multiprocessing.connection`, and only on `send_bytes`, `recv_bytes`, and `poll`.
Nothing here calls `send` or `recv`, which pickle. A test fails if any pickle runs.
"""

from __future__ import annotations
