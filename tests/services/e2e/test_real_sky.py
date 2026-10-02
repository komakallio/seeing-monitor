"""The plan of `seeingmon dev --driver asi --real-sky` on real processes, with the fake SDK.

`acquire` runs the `asi` driver on `FakeAsiSdk`, and `core` starts from a plan with no seed
solution, the catalog and the solver of a person, and the site of that person (made-up values). The
test checks what a person sees at first light: `core` says in its log that it has no pointing and
which solvers it will try, the scheduler asks for a survey step to solve, and the log of `core`
reports the first survey frame in a line that a person can read. The solver is a program that finds
no solution, so no real solver runs, and no test touches a camera or the network.

The module carries the `slow` marker, because the processes need a minute, and the clock runs in
real time. Run it with `--slow`.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("scipy", reason="the fast path needs the fast extra")

from seeingmon.survey.catalog import write_catalog
from tests.services.test_dev_real_sky import SITE, toml
from tests.survey import synth

from .system import System

pytestmark = pytest.mark.slow

# A program that exits with the code 1, which is how ASTAP says that it found no solution.
NO_SOLUTION_PROGRAM = f'"{Path(sys.executable).as_posix()}" -c "import sys; sys.exit(1)"'


@pytest.fixture(scope="module")
def system(tmp_path_factory: pytest.TempPathFactory) -> Iterator[System]:
    directory = tmp_path_factory.mktemp("real-sky")
    catalog = directory / "catalog.example"
    write_catalog(catalog, synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=0.05))
    owner = toml(
        {
            "site": dict(SITE),
            "survey": {
                "catalog_path": str(catalog),
                "solvers": ["astap"],
                "astap_command": NO_SOLUTION_PROGRAM,
            },
        }
    )
    built = System(
        directory,
        with_web=False,
        owner_settings=owner,
        acquire_driver="asi",
        real_sky=True,
        data_dir=directory / "data",
        log_level="info",  # what `seeingmon dev --real-sky` chooses
        acquire_overrides={"services": {"acquire": {"driver_options": {"fake_sdk": True}}}},
        core_overrides={
            # The Sun at the made-up site must not keep the scheduler in `safe` at noon.
            "scheduler": {"daylight": {"sun_elevation_limit_deg": 90.0}},
        },
    )
    built.start_all()
    yield built
    built.stop_all()


def core_log(system: System) -> str:
    return system.children["core"].spec.log.read_text(encoding="utf-8", errors="replace")


class TestTheFirstLight:
    def test_the_logs_of_the_children_are_in_the_data_folder(self, system: System) -> None:
        plan = system.plan
        assert plan.log_dir is not None
        assert plan.log_dir.parent == system.data_dir / "logs"
        assert {path.name for path in plan.log_dir.iterdir()} == {"acquire.log", "core.log"}
        assert not list(plan.directory.rglob("*.log"))  # the run folder keeps none
        assert not (plan.directory / "seed.json").exists()
        assert not (plan.directory / "catalog.bin").exists()

    def test_core_says_that_it_has_no_pointing_and_names_the_solvers(self, system: System) -> None:
        system.wait_for(lambda: "no pointing solution yet" in core_log(system), "the first line")
        line = next(
            line for line in core_log(system).splitlines() if "no pointing solution yet" in line
        )
        assert " INFO " in line
        assert "the plate solvers astap, in this order, until one solves" in line

    def test_the_scheduler_asks_for_a_survey_step_to_solve_and_core_reports_the_frame(
        self, system: System
    ) -> None:
        def solve_requested() -> bool:
            events = system.events("scheduler.solve_requested")
            return any((e.detail or {}).get("reason") == "no_solution" for e in events)

        system.wait_for(solve_requested, "the request for a survey step", timeout_s=180)
        frame = re.compile(
            r"survey frame \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ: [\d.]+ s bin2: .*not solved, "
            r"analysis took \d+\.\d s"
        )
        system.wait_for(lambda: bool(frame.search(core_log(system))), "a frame line", timeout_s=240)
        lines = [line for line in core_log(system).splitlines() if "survey frame" in line]
        assert lines, "no line of the survey analysis"
        for line in lines:
            assert not re.search(r"\d+\.\d{3,}\s*(deg|°)", line)  # no coordinate in the log
        assert system.alive() == {"acquire": True, "core": True}
