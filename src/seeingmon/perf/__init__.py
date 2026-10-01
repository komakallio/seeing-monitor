"""The performance harness: measures what the architecture's performance gate needs.

Run it with `seeingmon perf run`, and read a saved report with `seeingmon perf report`. The page
`docs/performance.md` explains what the harness measures, the budgets, and how to run it on a
Raspberry Pi 4.

The package has four layers, and importing it loads none of the code under test:

- **Measurement core.** `seeingmon.perf.timing` (a timer that reports min, median, p95, and max),
  `seeingmon.perf.memory` (peak memory and CPU time of the current process), and
  `seeingmon.perf.load` (how busy the machine is).
- **Cases and reports.** `seeingmon.perf.registry` holds the case registry and the case context,
  `seeingmon.perf.report` the report types and the environment facts, and `seeingmon.perf.cases`
  the cases themselves. Each case imports the code that it measures when it runs, so the harness
  works at every commit and skips a case whose component is not on `main` yet.
- **Runner.** `seeingmon.perf.runner` runs every case in a fresh child process
  (`python -m seeingmon.perf._child`), so that the peak memory of one case never includes another,
  and it pins the math libraries to one thread, so that every figure is per core.
- **Verdicts.** `seeingmon.perf.scaling` holds the assumed Raspberry Pi 4 scaling in one table,
  and `seeingmon.perf.budgets` turns a report into `pass`, `marginal`, or `fail` for each budget.
"""

from __future__ import annotations
