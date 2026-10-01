# Seeing monitor: phase 2 status

Updated: October 1, 2026, at commit `db8371d`. Steps 0, 1, 2, 5, 6, and 10 are done. Steps 3, 4, 7 to 9, and 12 are in progress in parallel lanes. The lead updates this file at every step boundary and whenever a lane lands a large piece, and the lanes do not edit it. The instructions are in `docs/phase2-kickoff.md`.

Main health: the lock goes stale whenever a lane adds a dependency (`sep`, then `psycopg`), and the lead regenerates it each time (`9e6edd6`, `db8371d`). Two timing tests fail on Windows with Python 3.11 (a fast-path benchmark and a services drop test), and their lanes are fixing them. Earlier Linux-only failures in two store tests were fixed by the storage lane.

## Steps

States are not started, in progress, blocked, and done.

| Step | Lane | State | Notes |
|---|---|---|---|
| 0 | Foundation | Done | CI runs on Windows, Linux x64, and Linux arm64 with Python 3.11 and 3.13, plus a lock check. |
| 1 | Foundation | Done | The profile schema, the derived values (1.910 and 3.820 arcsec/pixel, 4.395 × 2.994 degrees), the reference profile, and the configuration layers, with `seeingmon profile` commands. The wheel carries the profiles and defaults. |
| 2 | Foundation | Done | The `Clock` (system, virtual, and scaled), frame types with the wire format, the driver, sink, solver, and analysis interfaces, scripted fakes, and the record declarations with their generators (`docs/quantities.md`). |
| 3 | Simulation and fast path | In progress | On `main`: the `sim` driver with turbulence, wave-optics stars, sky, sensor, faults, the pointing offset, the truth object, `create_driver`, and a benchmark. The lane is finishing its validation tests. |
| 4 | Simulation and fast path | In progress | A second lane builds the kernel, windows, estimator, scintillation, and spectrum, then the validation on your recordings. It has not pushed yet. |
| 5 | Storage and sinks | Done | The SQLite store with sink cursors and snapshots, the segment writer and reader, the data layout, retention with the capture gate, the forwarder, the InfluxDB sink (versions 1 and 2, configurable), the PostgreSQL and TimescaleDB sink, `open_storage`, and `seeingmon store info`. The outage-and-resume and retention-quota tests pass. The adapters run against fakes. A test against a real PostgreSQL server runs when `SEEINGMON_TEST_POSTGRES` is set. |
| 6 | Scheduler | Done | The configuration, the ephemeris, the state machine with commands and gates, the scheduler, and the commissioning queue with the sweep. Tests cover faults, alignment, pause, shutdown, and a simulated evening. Twelve hours of virtual time run in 9 to 14 seconds. The services lane writes the burst and replay handlers and the `seeingmon sweep`, `burst`, and `replay` commands. |
| 7 | Survey path | In progress | First half done: the geometry and apparent places, the cap catalog with its build command (`seeingmon catalog build`), star detection with SEP and trail models, the TAN pointing fit, the pointing tracker, the astrometry.net and ASTAP adapters, and the survey analyzer. Synthetic frames recover the pointing to 0.0005 pixel RMS, against a 0.1 pixel target, and a bin2 frame takes about 3 s on the dev machine (not measured on a Pi). The solver programs never ran for real, only against script shims. A real cap build needs about an hour against the Gaia archive. Second half in progress: photometry, zero point, sky brightness, transparency, the dark model, and `seeingmon dark`. |
| 8 | Recordings | In progress | Done: the SER reader and writer, the SharpCap and JSON sidecars, the `replay` driver, and `seeingmon recordings info`. The 13 tests against your real recordings pass. Open: validate the fast path on the recordings, which needs the step 4 estimators. |
| 9 | Services | In progress | Two lanes. The services lane builds `acquire` and the connection layer. On `main`: the configuration, the authenticated connection layer (codecs, request-response, and a stream channel with flow control, with a test that no pickle runs), the time stamper, and drop accounting. Next: the `acquire` process, the remote driver, and the kill-and-restart tests. A web lane builds the REST API v1, the UI with a red night mode, and a demo mode that runs on synthetic data. The `core` composition follows when the fast and survey paths are done. |
| 10 | Hardware-facing | Done | The ASI SDK binding with a fake SDK, the call watchdog and USB reset, the ASI driver with the recovery ladder, the dew-heater controller with GPIO and sensor interfaces, the power-cycle hook, and the SQM-LE reader. The 10 `hardware` checks skip cleanly, and `docs/hardware-checks.md` explains how to run them. Unverified on real hardware: the SDK structure layouts and control numbers, the libgpiod calls, the USB reset, and the heater constants. The SQM-LE protocol comes from documentation (B5). |
| 11 | Hardware-facing | Not started | Dev-machine harness only. After the fast and survey paths exist. |
| 12 | Hardware-facing | In progress | A background lane builds the install and rollback scripts, the systemd units, the udev rule, and the runbook, all linted. The install on a fresh Pi stays blocked (B2). |

## Blockers

| ID | Needs | From | Blocks | Interim work |
|---|---|---|---|---|
| B1 | A camera on the dev machine | Owner (deferred) | The real-hardware checks in step 10 | A fake SDK and fake driver tests |
| B2 | A bench Raspberry Pi 4 | Owner (deferred) | The Pi 4 measurement in step 11, the install check in step 12, and the RAM and Rust-kernel decisions | The dev-machine harness and linted scripts |
| B3 | The heater HAT's pin map (chip, line, and polarity), the temperature and humidity sensors (files or an adapter), and the failsafe (a keepalive or a fault line) | Owner (deferred) | The final heater adapter in step 10 | A fake GPIO behind the `Io` interface. The heater constants are placeholders. |
| B4 | The remote power-cycle route | Owner (deferred) | The power-cycle hook in step 10 | A configurable command or URL |
| B5 | A sample SQM-LE reading and its protocol notes | Owner | The SQM-LE reader in step 10 | A fake TCP server |
| B6 | The InfluxDB version and field names | Owner (deferred) | Nothing | A configurable adapter |
| B7 | The web access rule | Owner (deferred) | Nothing | The default rule in the architecture |

## Not a step

`docs/demo/seeing-simulator.html` is a page of rendered simulator output (frames, centroid statistics, the motion spectrum, and the exposure bias). Open it in a browser. It is a snapshot from commit `829f48c`.

## Next unblocked work

Finish the running lanes. Then start the second half of the survey path, the `core` composition of step 9 (when the fast and survey paths are done), and the performance harness (step 11) when the fast and survey paths exist.
