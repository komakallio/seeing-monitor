"""The visibility of Polaris: the nightly summary and its statistics.

`summary` turns the events, the health records, and the seeing windows of one night into the
values of a `visibility_summary` record. `core` reads the store and writes the record
(`seeingmon.services.core.visibility`). `stats` groups the stored summaries by month and by
transparency, and `seeingmon visibility stats` (`cli`) prints them.

The package `__init__` imports nothing, so the command line stays fast.
"""

from __future__ import annotations
