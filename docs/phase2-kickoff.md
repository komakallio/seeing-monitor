# Seeing monitor: phase 2 kickoff prompt

Status: draft. Use it only after the owner approves `docs/architecture.md`, and record the approval date in that file's status line.

You are the lead engineer for the seeing monitor, and phase 2 (implementation) starts now. Read `CLAUDE.md`, `docs/kickoff-prompt.md`, `docs/architecture.md`, and `docs/research-notes.md` before you write code. The architecture is the design of record. The research notes hold the numbers that your tests must reproduce.

## Ground rules

- Follow `CLAUDE.md` in every step: the repository is public, so commit no secrets, deployment values, or machine-specific values, push every commit, and add no co-authors. Review `git diff --staged` for leaks before each commit.
- Build in small, tested steps. Write the tests with the code, and test estimators against simulated or analytic truth.
- Keep `docs/architecture.md` accurate. When you decide something or deviate from it, edit the decisions table or the section, and say why in the commit message.
- Keep real values under `local/` (gitignored) and commit `*.example` templates. Never write the location of the owner's recordings into the repository.
- When you ask the owner a question, explain any networking, security, or electronics term in one plain sentence.
- Do not tune on the sky or choose final exposure, gain, ROI, and cadence values. That is phase 3.

## What you need from the owner

| Step | What you need |
|---|---|
| 8 | The folder and file format of the recorded 10 ms video (a few gigabytes). Ask when you reach the analysis code. |
| 10 | The camera and the ZWO SDK archive on a test machine, the heater HAT's pin map, sensors, and failsafe, and the remote power-cycle route. |
| 11 | A bench Raspberry Pi 4 with 2 GB of RAM, an SD card, and 64-bit Raspberry Pi OS Lite. |

The InfluxDB version, the field names, and the web access rule are deferred. Use the defaults in the architecture, and keep the InfluxDB version configurable.

## Tooling

Use Python 3.11 or later (3.13 recommended) with `pyproject.toml`, pytest, hypothesis, ruff, and a type checker. Pin dependencies with hashes, and record the lock tool you choose in the decisions table. Add a GitHub Actions workflow that runs the linter, the type checker, the tests, a secret scan, and a repository check on Windows, Linux x64, and Linux arm64 with Python 3.11 and 3.13. The repository check fails on absolute paths, IP addresses, and hostnames in tracked files.

## Steps

Each step is a few commits, and each commit is pushed. Steps 0 to 7 and 9 need no hardware.

| Step | Deliverable | Done when |
|---|---|---|
| 0 | Package skeleton `seeingmon`, a command-line stub, `.gitignore` additions for virtual environments and caches, a short README, the CI workflow, and the repository check | CI is green on all three runners |
| 1 | The profile schema (pydantic), the reference profile `profiles/asi294mm-gs250.toml` with hardware values only, and the derived values: plate scale, field of view, ROI rounding, sampling, and saturation level | Tests reproduce the research-notes values (1.910 and 3.820 arcsec/pixel, 4.395 × 2.994 degrees, 128 pixels for 4.1 arcmin in bin1) |
| 2 | The `Clock` interface with virtual time, and the record declarations that generate the SQLite schema, the API schema, and the quantity reference | A simulated night advances in seconds, and a test checks the generated schemas |
| 3 | The `CameraDriver` protocol and the `sim` driver: catalog stars, frozen-flow turbulence, photon and read noise, rolling shutter, sky, clouds, drops, and timeouts | The simulated image-motion variance matches 0.170 λ² D^(−1/3) r0^(−5/3) within 5% |
| 4 | The fast path: centroid, windows, the seeing estimator with the outer-scale and exposure corrections, scintillation, and the spectrum | The estimator recovers an injected r0 within 10% for r0 of 5 to 15 cm, and a benchmark script reports microseconds per frame |
| 5 | The SQLite store, segment files, retention, sink cursors, the InfluxDB line-protocol adapter, and the TimescaleDB adapter, tested against fakes | Outage-and-resume tests and retention-quota tests pass |
| 6 | The scheduler: state machine, ROI following, daylight and cloud logic, and the commissioning queue | A virtual-time night produces the expected records and flags |
| 7 | The survey path: the offline catalog tool, SEP detection, the solver adapters (astrometry.net and a fake), the WCS fit with apparent places, sky quality, transparency, the dark model, and `seeingmon dark` | Synthetic frames recover the pointing within 0.1 pixel and the zero point within 0.03 mag |
| 8 | The `replay` driver and readers for the owner's format, and validation of the fast path on the real recordings | The recordings run through production code, and the findings go into `docs/` without any path |
| 9 | The `acquire`, `core`, and `web` processes over `multiprocessing.connection`, the REST API v1 with OpenAPI, the static UI with a red night mode, the live view, and the alignment helper | Kill-and-restart tests and API contract tests pass, and the UI works at phone width |
| 10 | The ASI SDK driver with the recovery ladder, the GPIO heater adapter, the power-cycle hook, and the SQM-LE reader | Hardware tests (marked `hardware`) pass on the owner's equipment |
| 11 | The performance gate on a Pi 4: kernel time, survey analysis time, and memory peaks | The result decides Python or Rust for the kernel, and 2 GB or 4 GB of RAM |
| 12 | A generic install script, the systemd units, the udev rule, and a runbook | A fresh Raspberry Pi OS Lite install works from the runbook |

## Definition of done

Every step ends with passing CI on all three runners, an up-to-date architecture document, an imperative commit message, a push, and a two-sentence report to the owner. Stop and ask the owner when the architecture looks wrong or blocked, when you need hardware or data, or when an action would need a secret.
