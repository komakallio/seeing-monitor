# Seeing monitor: phase 2 status

Updated: October 1, 2026, at commit `11169d1`. Steps 0 to 6, 8, and 10 are done. Steps 7, 9, 11, and 12 are in progress in parallel lanes. The lead updates this file at every step boundary and whenever a lane lands a large piece, and the lanes do not edit it. The instructions are in `docs/phase2-kickoff.md`.

Main health: CI on `main` is green at the last completed run. One run failed on a typing error in a FITS test (`6dfa28d`), and the lane fixed it in the next commit (`66c2134`). The lock goes stale whenever a lane adds a dependency (`sep`, then `psycopg`), and the lead regenerates it each time (`9e6edd6`, `db8371d`). The Windows timing tests that failed on Python 3.11 now read intervals from `perf_counter_ns` or a virtual clock, and the Linux-only store failures are fixed.

## Steps

States are not started, in progress, blocked, and done.

| Step | Lane | State | Notes |
|---|---|---|---|
| 0 | Foundation | Done | CI runs on Windows, Linux x64, and Linux arm64 with Python 3.11 and 3.13, plus a lock check. |
| 1 | Foundation | Done | The profile schema, the derived values (1.910 and 3.820 arcsec/pixel, 4.395 × 2.994 degrees), the reference profile, and the configuration layers, with `seeingmon profile` commands. The wheel carries the profiles and defaults. |
| 2 | Foundation | Done | The `Clock` (system, virtual, and scaled), frame types with the wire format, the driver, sink, solver, and analysis interfaces, scripted fakes, and the record declarations with their generators (`docs/quantities.md`). |
| 3 | Simulation and fast path | Done | The `sim` driver with frozen-flow turbulence through a 50 mm aperture, wave-optics or Gaussian stars, sky, sensor noise, faults, the pointing offset, the injected truth, `create_driver`, and a benchmark. It reproduces the theory within 5% (image-motion variance, outer scale, exposure averaging). A 128 × 128 bin1 frame renders at 98 frames per second in wave mode and 173 with Gaussian stars, and a 64 × 64 bin2 frame at 164 and 272. |
| 4 | Simulation and fast path | Done | The kernel (about 100 µs for a 128 × 128 frame on the dev machine), the windows, the estimator with the outer-scale, exposure-averaging, and centroid-gain factors, scintillation, and the motion spectrum with vibration lines. The estimator recovers an injected r0 of 5, 10, and 15 cm within 1% of the simulator's truth, against a 10% target. `create_fast_analyzer` builds it from the profile and the configuration. |
| 5 | Storage and sinks | Done | The SQLite store with sink cursors and snapshots, the segment writer and reader, the data layout, retention with the capture gate, the forwarder, the InfluxDB sink (versions 1 and 2, configurable), the PostgreSQL and TimescaleDB sink, `open_storage`, and `seeingmon store info`. The outage-and-resume and retention-quota tests pass. The adapters run against fakes. A test against a real PostgreSQL server runs when `SEEINGMON_TEST_POSTGRES` is set. |
| 6 | Scheduler | Done | The configuration, the ephemeris, the state machine with commands and gates, the scheduler, and the commissioning queue with the sweep. Tests cover faults, alignment, pause, shutdown, and a simulated evening. Twelve hours of virtual time run in 9 to 14 seconds. The services lane writes the burst and replay handlers and the `seeingmon sweep`, `burst`, and `replay` commands. |
| 7 | Survey path | In progress | First half done: the geometry and apparent places, the cap catalog with its build command (`seeingmon catalog build`), star detection with SEP and trail models, the TAN pointing fit, the pointing tracker, the astrometry.net and ASTAP adapters, and the survey analyzer. Synthetic frames recover the pointing to 0.0005 pixel RMS, against a 0.1 pixel target, and a bin2 frame takes about 3 s on the dev machine (not measured on a Pi). The solver programs never ran for real, only against script shims. A real cap build needs about an hour against the Gaia archive. Second half: the dark library, the dark model, `dark_due`, and the dark session are on `main`. Photometry, the zero point, sky brightness, and transparency are in progress. |
| 8 | Recordings | Done | The SER reader and writer, the SharpCap and JSON sidecars, the `replay` driver, `seeingmon recordings info`, and the validation on your recordings (`docs/recordings-validation.md`). The star appears in all 35,252 frames, and the sidereal drift confirms the plate scale within 4%. The 60 s windows give an r0 of 8 to 12 cm, which is not calibrated. The two estimators differ by 17 to 34% at the default wind and agree at about 2 m/s, the x variance is 1.3 to 1.6 times the y variance, and the 8-bit noise model reads r0 3.5% high. |
| 9 | Services | In progress | The services lane built `acquire` (the process that owns the camera), the authenticated connection layer (a test shows that no pickle runs), the time stamper, drop accounting, and the remote camera driver, which passes the driver conformance tests. The web lane builds the REST API v1, the UI with a red night mode, and a demo mode that runs on synthetic data. Its contract for `core` (the RPC methods and the preview stream) is on `main`. The services lane now builds the `core` process, the alignment helper, `seeingmon burst`, `sweep`, and `replay`, a `seeingmon dev` launcher, and the end-to-end tests (kill and restart, sink outage, full disk, clock jump). |
| 10 | Hardware-facing | Done | The ASI SDK binding with a fake SDK, the call watchdog and USB reset, the ASI driver with the recovery ladder, the dew-heater controller with GPIO and sensor interfaces, the power-cycle hook, and the SQM-LE reader. The 10 `hardware` checks skip cleanly, and `docs/hardware-checks.md` explains how to run them. Unverified on real hardware: the SDK structure layouts and control numbers, the libgpiod calls, the USB reset, and the heater constants. The SQM-LE protocol comes from documentation (B5). |
| 11 | Hardware-facing | In progress | A lane builds `seeingmon perf`: calibration workloads, the kernel, the fast path, the link between `acquire` and `core`, a survey frame, the store, and memory peaks, each in its own child process. It prints a verdict against each budget with the Pi 4 figures stated as an estimate, writes `docs/performance.md`, and adds a smoke test to the normal CI run. The `core` case waits for the `core` process. The Pi 4 measurement stays blocked (B2). |
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

Finish the running lanes: the second half of the survey path, the web lane, the `core` composition and the commands of step 9, the performance harness, and the install scripts. Then run `seeingmon dev` end to end, review the web demo in a browser at phone and desktop width, check the definition of done, and send the owner one message that lists the remaining blockers.
