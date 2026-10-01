# Seeing monitor: phase 2 kickoff prompt

Status: ready. The owner approved `docs/architecture.md` on October 1, 2026.

You are the lead engineer for the seeing monitor, and phase 2 (implementation) starts now. Read `CLAUDE.md`, `docs/kickoff-prompt.md`, `docs/architecture.md`, `docs/research-notes.md`, and `docs/phase2-status.md` before you write code. The architecture is the design of record, the research notes hold the numbers that your tests must reproduce, and the status file is the live state of the work.

## How to work

- **Work independently.** Decide within the architecture, record each decision (the decisions table or the commit message), and ask the owner only for the items under "Stop and ask".
- **Never idle on a blocker.** When a step, or part of a step, needs something you do not have, record it in `docs/phase2-status.md` (what is blocked, which input unblocks it, who supplies it), and tell the owner once, in one plain-language message that batches all current blockers. Then continue with other work. Build everything that does not need the missing input first: interfaces, fakes, fixtures, and harnesses. Return to the blocked work when the input arrives, and do not wait for a reply.
- **Give up on a loop, not on a step.** If a step fails its done criteria after three focused attempts, record the evidence in the status file, mark the step blocked, and move on.
- **Run independent lanes in parallel** (see "Lanes"). Use background subagents, and use workflows only when the owner has opted in. Otherwise take the next unblocked step yourself.
- Follow `CLAUDE.md` in every step: the repository is public, so commit no secrets, deployment values, or machine-specific values, push every commit, and add no co-authors. Review `git diff --staged` for leaks before each commit.
- Write tests with the code, and test estimators against simulated or analytic truth. Never leave `main` red: fix forward or revert at once.
- Keep `docs/architecture.md` accurate. When you decide something or deviate from it, edit the decisions table or the section and say why in the commit message.
- Keep real values under `local/` (gitignored) and commit `*.example` templates. The location of the owner's recordings and any camera serial number never enter the repository.
- When you ask the owner a question, explain any networking, security, or electronics term in one plain sentence.
- Do not tune on the sky or choose final exposure, gain, ROI, and cadence values. That is phase 3.

## What exists and what does not

- **No real camera on the dev machine and no Raspberry Pi, for now.** Build the hardware-facing code against fakes and harnesses, write the real-hardware checks (marker `hardware`) so that they skip cleanly, and list them as blocked in the status file.
- **Recordings exist.** The owner's recordings are on the dev machine, outside the repository. Your project memory holds the location, and if you cannot find it, ask once. They are two SharpCap SER files with per-frame PC-clock timestamps and a text sidecar for each (decimal commas, UTC and local-time stamps). The camera is the ASI294MM in its 11 MP read mode (the SDK's bin2), the ROI is 320 × 240, the pixels are 8-bit, the gain is 100, the exposure is 10 ms, and the rate is 97.9 fps. The recordings last 60 s and 300 s (5,874 and 29,378 frames). They differ from the planned fast mode (bin1, 2 ms, 16-bit), so use them to test the estimators, the exposure correction, and the bin2 centroid effects, and not as a template for the final settings.
- **Deferred by the owner.** The InfluxDB version and field names, the web access rule, the remote power-cycle route, and the choice of Pi and HAT. Use the defaults in the architecture, and keep each of them configurable.

## Lanes

Steps 0 to 2 define the contracts: the package layout, the profile, the `Clock`, the record declarations, and the typed interfaces for frames, drivers, sinks, and solvers. Do them first and keep them stable. After that, the lanes run independently.

| Lane | Steps | Starts after | Blocked by | Proceeds without it |
|---|---|---|---|---|
| Foundation | 0, then 1 and 2 in parallel | Nothing | Nothing | Everything |
| Simulation and fast path | 3, 4 | 1, 2 | Nothing | Everything |
| Recordings | 8 | 2 | Nothing | Everything. Real-data validation starts when step 4 has estimators. |
| Storage and sinks | 5 | 2 | InfluxDB details (deferred) | Fakes. The version stays configurable. |
| Survey path | 7 | 1, 2 | The astrometry.net binary, for an optional integration test | Synthetic frames and a fake solver |
| Scheduler | 6 | 2 | Nothing | A fake driver and fake analysis |
| Services | 9 | 2, 5 | The web access rule (deferred) | The default rule |
| Hardware-facing | 10, 11, 12 | 3, 9 | A camera on the dev machine, a bench Pi 4, the HAT documents, the power-cycle route, an SQM-LE sample | A fake SDK, a fake GPIO, a benchmark harness that runs on the dev machine, and linted scripts |

## Parallel work

- Each lane works in its own clone of the repository, in a directory outside the repository and outside any cloud-synced folder, with `main` checked out. After every commit, run `git pull --rebase origin main`, run the lane's tests (and the full suite when the rebase brought in other lanes' files), and push. If a push is rejected, pull and rebase again. Never force-push.
- Create the lane clones inside the directory that your project memory names for them, one subfolder per lane. The owner's untracked `.claude/settings.local.json` lists that directory under `permissions.additionalDirectories`, so file access there needs no approval. Do not record the path in any tracked file.
- Run git in a lane's clone as `git -C <clone> <subcommand>`, and never `cd` into the clone and then run git. A lane's agent starts in the lead's working directory, so each command names its clone. The owner's permission rules pre-approve `add`, `commit`, `pull --rebase origin main`, and `push origin main` in this form, so give the pull and push commands no extra options. A `cd` followed by `git` prompts for approval (git can run hooks in the new directory), and the prompt stalls the lane.
- A lane edits only its own package, its own tests, and its own sections of `docs/architecture.md`. It adds dependencies only to its own extra in `pyproject.toml`. The lead regenerates the lock file after merges and is the only one who edits `docs/phase2-status.md`.
- The lead watches CI on `main` (`gh run list`) and acts on any failure at once.

## Progress reporting

- After every commit, post one line in the transcript: lane, step, what changed, and the test result.
- At every step boundary, post two sentences (what works, what comes next) and update `docs/phase2-status.md`.
- During a task that runs longer than 20 minutes, post a one-line heartbeat every 20 minutes.
- On a new blocker, message the owner, send a push notification if the tool is available, and add the blocker to the status file. Send a push notification when a lane finishes and when CI fails on `main`.

## Stop and ask

Stop and ask the owner only when the architecture looks wrong or contradictory (propose a fix, and continue with other lanes meanwhile), when you need an input that no fake can stand in for, when an action would need a secret, or when an action is destructive or irreversible. If a command needs an approval that you do not have, record a blocker and continue.

## Tooling

Use Python 3.11 or later (3.13 recommended) with `pyproject.toml`, pytest, hypothesis, ruff, and a type checker. Create virtual environments outside any cloud-synced folder. Pin dependencies with hashes, and record the lock tool you choose in the decisions table. Add a GitHub Actions workflow that runs the linter, the type checker, the tests, a secret scan, and a repository check on Windows, Linux x64, and Linux arm64 with Python 3.11 and 3.13. The repository check fails on absolute paths, IP addresses, hostnames, and serial numbers in tracked files.

## Steps

Each step is a few commits, and each commit is pushed. Steps 0 to 9 need no hardware.

| Step | Deliverable | Done when |
|---|---|---|
| 0 | Package skeleton `seeingmon`, a command-line stub, `.gitignore` additions for virtual environments and caches, a short README, the CI workflow, and the repository check | CI is green on all three runners |
| 1 | The profile schema (pydantic), the reference profile `profiles/asi294mm-gs250.toml` with hardware values only, and the derived values: plate scale, field of view, ROI rounding, sampling, and saturation level | Tests reproduce the research-notes values (1.910 and 3.820 arcsec/pixel, 4.395 × 2.994 degrees, 128 pixels for 4.1 arcmin in bin1) |
| 2 | The `Clock` interface with virtual time, the record declarations that generate the SQLite schema, the API schema, and the quantity reference, and the typed interfaces for frames, drivers, sinks, and solvers | A simulated night advances in seconds, and tests check the generated schemas |
| 3 | The `CameraDriver` protocol and the `sim` driver: catalog stars, frozen-flow turbulence, photon and read noise, 8-bit and 16-bit output, rolling shutter, sky, clouds, drops, and timeouts | The simulated image-motion variance matches 0.170 λ² D^(−1/3) r0^(−5/3) within 5% |
| 4 | The fast path: centroid, windows, the seeing estimator with the outer-scale and exposure corrections, scintillation, and the spectrum | The estimator recovers an injected r0 within 10% for r0 of 5 to 15 cm, and a benchmark script reports microseconds per frame |
| 5 | The SQLite store, segment files, retention, sink cursors, the InfluxDB line-protocol adapter, and the TimescaleDB adapter, tested against fakes | Outage-and-resume tests and retention-quota tests pass |
| 6 | The scheduler: state machine, ROI following, daylight and cloud logic, and the commissioning queue | A virtual-time night produces the expected records and flags |
| 7 | The survey path: the offline catalog tool, SEP detection, the solver adapters (astrometry.net and a fake), the WCS fit with apparent places, sky quality, transparency, the dark model, and `seeingmon dark` | Synthetic frames recover the pointing within 0.1 pixel and the zero point within 0.03 mag |
| 8 | The `replay` driver and the SER reader (timestamps, 8-bit and 16-bit, the sidecar), then validation of the fast path on the owner's recordings | The recordings run through production code, and the findings go into `docs/` without any path or serial number |
| 9 | The `acquire`, `core`, and `web` processes over `multiprocessing.connection`, the REST API v1 with OpenAPI, the static UI with a red night mode, the live view, and the alignment helper | Kill-and-restart tests and API contract tests pass, and the UI works at phone width |
| 10 | The ASI SDK binding and driver with the recovery ladder, the GPIO heater adapter, the power-cycle hook, and the SQM-LE reader, all tested against fakes | Fake-backed tests pass, and the `hardware` checks are written and skip cleanly |
| 11 | The performance harness: kernel time, survey analysis time, and memory peaks, run on the dev machine with the scaling to a Pi 4 stated as an estimate | The harness runs in CI as a smoke test, and the Pi 4 measurement stays blocked |
| 12 | A generic install script, the systemd units, the udev rule, and a runbook, all linted | The scripts pass linting, and the install on a fresh Pi stays blocked |

## Definition of done

Every step ends with passing CI on all three runners, an up-to-date architecture document, an imperative commit message, a push, and a two-sentence report to the owner. Phase 2 is done when steps 0 to 9 are done, steps 10 to 12 are done against fakes and harnesses, and the status file lists every remaining item as blocked with the input that unblocks it.
