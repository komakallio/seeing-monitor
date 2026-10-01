# Seeing monitor

A fixed camera points at Polaris, and a Raspberry Pi measures atmospheric seeing and sky quality from it, unattended, for months. One camera serves four modes: fast seeing, sky quality, pointing, and an alignment helper.

**Status:** phase 2, implementation. The design is approved, and the software runs against simulated and replayed data. No hardware is needed to develop or test it.

## Documentation

- [Architecture](docs/architecture.md): the design of record.
- [Research notes](docs/research-notes.md): the sources and calculations behind the design.
- [Development guide](docs/development.md): set up a clone, run the checks, and commit.
- [Phase 2 status](docs/phase2-status.md): the state of each implementation step.

## Develop

You need Python 3.11 or later (3.13 recommended) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras      # create .venv and install the locked dependencies
uv run pytest             # run the tests
uv run ruff check .       # lint
uv run ruff format .      # format
uv run mypy               # type check
uv run python tools/check_repo.py   # check for private or machine-specific values
```

Read the [development guide](docs/development.md) before you commit. The repository is public, so it must hold no secrets, no machine-specific values, and no captured data.

## License

[MIT](LICENSE)
