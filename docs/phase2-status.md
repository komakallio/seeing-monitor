# Seeing monitor: phase 2 status

Updated: October 1, 2026, after steps 0 and 1. Steps 2, 3, 5, 6, 7, and 8 are in progress in parallel lanes. The lead updates this file at every step boundary, and the lanes do not edit it. The instructions are in `docs/phase2-kickoff.md`.

## Steps

States are not started, in progress, blocked, and done.

| Step | Lane | State | Notes |
|---|---|---|---|
| 0 | Foundation | Done | CI is green on Windows, Linux x64, and Linux arm64 with Python 3.11 and 3.13, plus a lock check (commit `a5fc068`). |
| 1 | Foundation | Done | The profile schema, the derived values (1.910 and 3.820 arcsec/pixel, 4.395 × 2.994 degrees), the reference profile, and the configuration layers, with `seeingmon profile` commands. The wheel carries the profiles and defaults. |
| 2 | Foundation | In progress | Done: the `Clock` (system, virtual, and scaled), frame types with the wire format, the driver, sink, solver, and analysis interfaces, scripted fakes, and the record declarations with the SQLite, API, quantity-reference, segment, and sink-mapping generators. The records lane is finishing its last polish. |
| 3 | Simulation and fast path | In progress | A background lane builds the `sim` driver (turbulence, stars, detector, faults, truth). |
| 4 | Simulation and fast path | Not started | After step 3, in the same lane |
| 5 | Storage and sinks | In progress | A background lane builds the store, segment files, retention, the forwarder, and the InfluxDB and TimescaleDB adapters. |
| 6 | Scheduler | In progress | A background lane builds the state machine against the interfaces and fakes. |
| 7 | Survey path | In progress | First half: catalog tool, detection, solver adapters, apparent places, the pointing fit, and the tracker. The second half (sky quality, transparency, dark model, `seeingmon dark`) follows in the same lane. |
| 8 | Recordings | In progress | Done: the SER reader and writer, the SharpCap and JSON sidecars, the `replay` driver, and `seeingmon recordings info`. The 13 tests against your real recordings pass. Open: validate the fast path on the recordings, which needs the step 4 estimators. |
| 9 | Services | Not started | After step 5 |
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

## Next unblocked work

Finish the running lanes. Start the services lane (step 9) when the store lands, the fast path (step 4) when the simulator lands, and the second half of the survey path after the first half. The hardware-facing lane (steps 10 to 12) can start now against fakes.
