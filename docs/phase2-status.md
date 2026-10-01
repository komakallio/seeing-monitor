# Seeing monitor: phase 2 status

Updated: October 1, 2026, after step 0 and the first part of step 2. The lead updates this file at every step boundary, and the lanes do not edit it. The instructions are in `docs/phase2-kickoff.md`.

## Steps

States are not started, in progress, blocked, and done.

| Step | Lane | State | Notes |
|---|---|---|---|
| 0 | Foundation | Done | CI is green on Windows, Linux x64, and Linux arm64 with Python 3.11 and 3.13, plus a lock check (commit `a5fc068`). |
| 1 | Foundation | In progress | The profile schema, derived values, reference profile, and configuration layers. A background lane is working on it. |
| 2 | Foundation | In progress | Done: the `Clock` (system, virtual, and scaled), frame types with the wire format, the driver, sink, and solver interfaces, and scripted fakes (commit `8c1512e`). In progress: the record declarations and their generators. Next: the analysis and scheduler-facing interfaces, after the records land. |
| 3 | Simulation and fast path | Not started | After steps 1 and 2 |
| 4 | Simulation and fast path | Not started | After step 3 |
| 5 | Storage and sinks | Not started | After step 2 |
| 6 | Scheduler | Not started | After step 2 |
| 7 | Survey path | Not started | After steps 1 and 2 |
| 8 | Recordings | Not started | After step 2. The recordings are available. |
| 9 | Services | Not started | After steps 2 and 5 |
| 10 | Hardware-facing | Not started | Fakes only until hardware exists |
| 11 | Hardware-facing | Not started | Dev-machine harness only |
| 12 | Hardware-facing | Not started | Linted scripts only |

## Blockers

| ID | Needs | From | Blocks | Interim work |
|---|---|---|---|---|
| B1 | A camera on the dev machine | Owner (deferred) | The real-hardware checks in step 10 | A fake SDK and fake driver tests |
| B2 | A bench Raspberry Pi 4 | Owner (deferred) | The Pi 4 measurement in step 11, the install check in step 12, and the RAM and Rust-kernel decisions | The dev-machine harness and linted scripts |
| B3 | The heater HAT's pin map, sensors, and failsafe | Owner (deferred) | The final heater adapter in step 10 | A fake GPIO behind the `Io` interface |
| B4 | The remote power-cycle route | Owner (deferred) | The power-cycle hook in step 10 | A configurable command or URL |
| B5 | A sample SQM-LE reading and its protocol notes | Owner | The SQM-LE reader in step 10 | A fake TCP server |
| B6 | The InfluxDB version and field names | Owner (deferred) | Nothing | A configurable adapter |
| B7 | The web access rule | Owner (deferred) | Nothing | The default rule in the architecture |
| B8 | Permission to read the recordings' sidecar text files (the permission classifier denied a read, because the sidecars hold the camera serial number) | Owner | Checking the sidecar parser against the real files in step 8 | A synthetic sidecar fixture built from the format in the research notes. The SER files themselves are not affected. |

## Next unblocked work

Finish steps 1 and 2. Then start the simulation and fast path, storage and sinks, scheduler, survey path, and recordings lanes in parallel.
