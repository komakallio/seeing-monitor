# Seeing monitor: phase 2 status

Updated: October 1, 2026, before the first phase 2 session. The lead updates this file at every step boundary, and the lanes do not edit it. The instructions are in `docs/phase2-kickoff.md`.

## Steps

States are not started, in progress, blocked, and done.

| Step | Lane | State | Notes |
|---|---|---|---|
| 0 | Foundation | Not started | The next unblocked step |
| 1 | Foundation | Not started | After step 0, in parallel with step 2 |
| 2 | Foundation | Not started | After step 0. Defines the contracts. |
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

## Next unblocked work

Step 0.
