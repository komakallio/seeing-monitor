# Development guide

This guide covers how to set up a clone, run the checks, and commit. It applies to everyone who changes the code, including the phase 2 lanes (see [phase2-kickoff.md](phase2-kickoff.md)). The repository rules in `CLAUDE.md` come first.

Below, `<clone>` is the root of your clone and `<py>` is its virtual-environment interpreter (`<clone>/.venv/Scripts/python.exe` on Windows, `<clone>/.venv/bin/python` on Linux). Commands name the clone explicitly, so they work from any directory.

## Set up a clone

Create the virtual environment outside any cloud-synced folder. Then install the locked dependencies with hashes:

```bash
py -3.13 -m venv <clone>/.venv                 # Windows. On Linux: python3.13 -m venv <clone>/.venv
<py> -m pip install uv
<clone>/.venv/Scripts/uv sync --project <clone> --all-extras --python <py>
```

`uv sync` installs the exact versions in `uv.lock` and the package itself in editable mode. Pass `--python <py>` to every `uv` command that takes it, and set `UV_PYTHON_DOWNLOADS=never`, so `uv` never downloads an interpreter into your user account.

## Run the checks

```bash
<py> -m pytest <clone>/tests                   # all tests. Pass a subfolder or file for a lane.
<py> -m ruff format <clone>                    # format. CI fails when a file is not formatted.
<py> -m ruff check <clone>                     # lint
<py> -m mypy --config-file <clone>/pyproject.toml <clone>/src <clone>/tests   # strict type check
<py> <clone>/tools/check_repo.py --repo <clone>             # private or machine-specific values
<py> <clone>/tools/scan_secrets.py --repo <clone>           # the secret scan that CI runs
```

The secret scan reads every tracked file, and one false positive turns `main` red on every runner. Add `--staged` to scan only the files in the index. Mark a false positive with `# pragma: allowlist secret` on the same line.

CI runs every check on Windows, Linux x64, and Linux arm64 with Python 3.11 and 3.13, so write code that runs on 3.11. Do not use syntax or library features that appeared in 3.12 or later. Python 3.11 is not installed on the dev machine, so CI is the first place a 3.11 problem shows.

On `main`, one CI run goes at a time, and at most one waits behind it. A newer push replaces the waiting run, so a burst of pushes tests the newest commit, and the commits in between get no run of their own. A nightly workflow (`.github/workflows/nightly.yml`) runs the slow tests (`pytest --slow -m slow`) on Windows, Linux x64, and Linux arm64. Start it by hand with `gh workflow run nightly.yml`. The JavaScript scenarios of the web UI (`tests/services/web/js/`) run under Node when it is installed and skip otherwise, so a machine without Node finds a failure of them in CI.

## Commit routine

Run these steps for every commit.

1. Format, lint, and type check, and run your lane's tests.
2. Stage named files: `git -C <clone> add <path> ...`. Do not stage everything with `-A`.
3. Run `<py> <clone>/tools/check_repo.py --repo <clone> --staged` and `<py> <clone>/tools/scan_secrets.py --repo <clone> --staged`, and review `git -C <clone> diff --staged` for leaks.
4. Commit with an imperative message: `git -C <clone> commit -m "Add the SER reader"`. Add no trailers and no co-authors.
5. Run `git -C <clone> pull --rebase origin main`.
6. Run your lane's tests again, and the full suite when the rebase brought in other lanes' files.
7. Run `git -C <clone> push origin main`. If the push is rejected, go back to step 5. Never force-push.

Always run git as `git -C <clone> <subcommand>`, and never `cd` into the clone first.

## Layout and ownership

Each lane edits only its own package, its own tests, and its own sections of `docs/architecture.md`.

| Path | Content | Owner |
|---|---|---|
| `src/seeingmon/cli.py` | Entry point. It discovers commands by convention. | Foundation |
| `src/seeingmon/profile/`, `config/` (package), `paths.py`, `profiles/`, `config/` (data) | Profile schema, derived values, configuration layers, data directories | Foundation (step 1) |
| `src/seeingmon/clock.py`, `frames.py`, `records/`, `drivers/base.py`, `sinks/base.py`, `solvers/base.py`, `analysis/base.py` | The contracts: `Clock`, frame types and wire format, records, and the driver, sink, solver, and analysis interfaces | Foundation (step 2) |
| `src/seeingmon/testing/` | Scripted fakes of the interfaces (`FakeCameraDriver`, `FakeSink`, `FakeSolver`, `FakeFastAnalyzer`, `FakeSurveyAnalyzer`, `FakePointingProvider`, `ListRecordWriter`) for tests | Foundation (step 2) |
| `src/seeingmon/drivers/sim/` and `fastpath/` | Simulator and fast path | Simulation and fast path (steps 3 and 4) |
| `src/seeingmon/store/` and `sinks/` | SQLite store, retention, sink forwarder, adapters | Storage and sinks (step 5) |
| `src/seeingmon/scheduler/` | Scheduler | Scheduler (step 6) |
| `src/seeingmon/survey/` | Survey path | Survey path (step 7) |
| `src/seeingmon/recordings/`, `drivers/replay.py` | SER reader, replay driver, validation | Recordings (step 8) |
| `src/seeingmon/services/` | `acquire`, `core`, `web`, REST API, UI | Services (step 9) |
| `src/seeingmon/hardware/`, `src/seeingmon/perf/`, `deploy/`, `tools/lint_deploy.py` | ASI binding, GPIO, power cycle, SQM-LE, benchmarks, install scripts and their linter | Hardware-facing (steps 10 to 12) |

The contracts are stable after step 2. You may make an additive change to a contract that your lane owns, such as a new optional field with a default, and you must say so in the commit message and in your report. Any change that could break another lane goes to the lead: describe the problem in your report, work around it inside your own package, and let the lead decide.

Use the fakes in `seeingmon.testing` for code that depends on a driver, a sink, or a solver. They run on a `Clock`, so with a `VirtualClock` a test reads frames without waiting.

## Conventions

- Start every module with `from __future__ import annotations`.
- Never read the time or sleep directly in library code (`time.time`, `time.sleep`, `datetime.now`). Take a `Clock`, so tests can run a night in seconds.
- Never let a pause decide how many frames, drops, or events a test sees. On Windows, a wait with a timeout (`Event.wait`, `Condition.wait`) ends on a timer tick of 15.6 ms (a wait of 1 ms took 14 ms on the dev machine), and before Python 3.13 `time.monotonic()` and `time.time()` tick at the same period. Code that paces itself on them runs slower there than on Linux. Use the virtual clock, or wait until the condition holds, for example until the queue has dropped 20 frames.
- Locate repository files in tests through the `repo_root` fixture, never through the working directory. Tests must pass from any directory.
- Put the tests of a package in `tests/<package>/` with an `__init__.py`.
- Add a command by defining `register(subparsers)` in `seeingmon/<package>/cli.py` (see `seeingmon/cli.py`). The entry point finds it, so no shared file changes.
- Test estimators against simulated or analytic truth, and write the tolerance next to the assertion.
- Use the markers `hardware` (skipped unless `--hardware`), `recordings` (skipped unless the owner's recordings are configured), and `slow` (skipped unless `--slow`).
- Write documentation and comments in the Google developer documentation style: second person, active voice, present tense, sentence-case headings.

## Configuration for lane authors

Keep your lane's defaults in `config/default.d/<lane>.toml`, with every key under one table named for your lane, such as `[scheduler]`. Do not edit another lane's file or `config/default.toml`. The test in `tests/config/test_repo_config.py` fails when two default files define the same key. Declare a pydantic model for your table, and read it through the configuration:

```python
from seeingmon.config import SectionModel, load_config


class SchedulerConfig(SectionModel):  # a frozen model that rejects unknown keys
    window_s: float = 120.0


config = load_config()
scheduler = config.section("scheduler", SchedulerConfig)
```

`section` validates the merged table and applies the model's defaults. Later layers override earlier ones: `config/default.toml`, then `config/default.d/*.toml` in file-name order (so use zero-padded prefixes if the order matters), then `local/config.toml`, then `SEEINGMON_<SECTION>__<KEY>` environment variables. For example, `SEEINGMON_SCHEDULER__WINDOW_S=60` sets `window_s` in `[scheduler]`. In a test, call `load_config(local_file=<a path that does not exist>, env={})`, so your own local file and environment stay out. Read hardware values from `config.profile` and the functions in `seeingmon.profile.derived`, and never hard-code a pixel size, a focal length, or a bit depth.

## Records

Declare each record type once, in the module of the lane that owns it (`records/seeing.py`, `survey.py`, `reference.py`, or `system.py`). To add a field, append it to your class with `quantity(...)`, and make it optional or give it a default, so an existing database migrates by adding a column. Then regenerate the quantity reference and the API description, which carries the record schemas, and commit both with your change:

```bash
<py> -m seeingmon records reference --output <clone>/docs/quantities.md
<py> -m seeingmon web openapi --output <clone>/docs/openapi.json
```

A test fails when `docs/quantities.md` or `docs/openapi.json` is stale. After a rebase conflict in either file, regenerate it instead of merging it by hand. A new unit needs a line in `UNIT_SUFFIXES` in `tests/records/test_declarations.py`. Two events can share a timestamp, so the store moves a colliding event by one nanosecond. Producers do not handle that themselves.

## Dependencies and the lock

The core install needs only `pydantic` and NumPy. Add each new dependency to your lane's extra in `pyproject.toml` (`fast`, `survey`, `web`, `timescale`, or a new extra), with a lower bound and no upper bound unless a release is known to break. Do not edit or commit `uv.lock`. The lead regenerates it after merges (`uv lock`), and the lock job in CI reports when it is stale. Until then, install a new dependency into your environment with `uv pip install --python <py> <package>`.

The lock tool is `uv`. Its universal lock covers Windows x64, Linux x64, and Linux arm64.

## Local files

Real values live under `local/`, which Git ignores. Commit only `*.example` templates with placeholders.

- `local/config.toml`: site and deployment configuration. For recordings, set `recordings_dir` in a `[replay]` table, or set the environment variable `SEEINGMON_REPLAY__RECORDINGS_DIR` (`SEEINGMON_RECORDINGS_DIR` also works). Tests marked `recordings` skip when neither exists.
- `local/repo-check-deny.txt`: private literals that `tools/check_repo.py` must never find in tracked files or commit messages, such as a user name or a camera serial number. One entry per line: plain text matches case-insensitively, and `re:` starts a regular expression.

Never write the location of the recordings, a camera serial number, a host name, an address, or a site coordinate into a tracked file, a commit message, or test output. `tools/check_repo.py` catches the common forms, including host names under private domains such as `.local` and `ts.net` (the names of a tailnet), and your review of `git diff --staged` catches the rest. It does not detect site coordinates.
