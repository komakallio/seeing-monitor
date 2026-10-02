# Seeing monitor

A fixed camera points at Polaris, and a Raspberry Pi measures atmospheric seeing and sky quality from it, unattended, for months. One camera serves four modes: fast seeing, sky quality, pointing, and an alignment helper.

**Status:** phase 2, implementation. The design is approved, and the software runs against simulated and replayed data. No hardware is needed to develop or test it.

## Documentation

- [Project brief](docs/kickoff-prompt.md): the setup, the modes, the constraints, and the three phases.
- [Architecture](docs/architecture.md): the design of record.
- [Research notes](docs/research-notes.md): the sources and calculations behind the design.
- [Development guide](docs/development.md): set up a clone, run the checks, and commit.
- [Runbook](docs/runbook.md): install on a Raspberry Pi, check health, reach the UI through a VPN, and recover.
- [Performance](docs/performance.md): the budgets, the results on a development machine, and the Pi 4 estimates.
- [Recordings validation](docs/recordings-validation.md): the fast path on recorded 10 ms video.
- [Quantities](docs/quantities.md): every record field, generated from the declarations.
- [REST API](docs/openapi.json): the OpenAPI description of API v1.
- [Hardware checks](docs/hardware-checks.md): the checks that need a camera or a Pi.
- [Phase 2 kickoff](docs/phase2-kickoff.md): the instructions and the steps of the implementation phase.
- [Phase 2 status](docs/phase2-status.md): the state of each implementation step, the blockers, and your next steps.
- [Simulator demo](docs/demo/seeing-simulator.html): a page of rendered simulator output that you open in a browser.

## Try it

You need no camera. Both commands print the address of the web UI.

```bash
uv sync --all-extras
uv run seeingmon web --demo    # the UI on synthetic data, with a fake core
uv run seeingmon dev           # acquire, core, and web on a simulated sky
```

`seeingmon dev` prints a random API token for each run, unless the `[auth]` table of `local/config.toml` holds a token hash. Both commands read the `[web]` table of that file, so you can set the address and the allowed host names of your own network there. That file stays out of the repository.

With a camera on the dev machine, `seeingmon dev --driver asi --real-sky --data-dir <folder>` runs the whole system on the real sky, with your catalog, plate solver, and site. The runbook has the steps: [First light on the dev machine](docs/runbook.md#first-light-on-the-dev-machine).

## Develop

You need Python 3.11 or later (3.13 recommended) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras      # create .venv and install the locked dependencies
uv run pytest             # run the tests
uv run ruff check .       # lint
uv run ruff format .      # format
uv run mypy               # type check
uv run python tools/check_repo.py   # check for private or machine-specific values
uv run python tools/scan_secrets.py # run the secret scan that CI runs
```

Read the [development guide](docs/development.md) before you commit. The repository is public, so it must hold no secrets, no machine-specific values, and no captured data.

## License

[MIT](LICENSE)
