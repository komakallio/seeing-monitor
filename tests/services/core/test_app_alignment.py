"""How `core` builds the quick solve of the alignment helper: where it runs, and what it detects."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.services.core.alignment.solve import QuickSolver
from seeingmon.services.core.alignment.worker import ProcessQuickSolver
from seeingmon.survey.analyzer import InlineExecutor
from seeingmon.survey.catalog import write_catalog
from tests.survey import synth

from .rig import CoreRig, build_rig


def rig(tmp_path: Path, config_extra: str = "", *, catalog: bool = True) -> CoreRig:
    """A core with a small catalog, the real survey configuration, and no seed solution."""
    path = tmp_path / "catalog.bin"
    if catalog:
        write_catalog(path, synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=0.05))
    else:
        path.write_bytes(b"this is not a catalog")
    return build_rig(
        tmp_path,
        config_extra=f'[survey]\ncatalog_path = "{path.as_posix()}"\nsolvers = []\n' + config_extra,
        parts={"survey": None, "pointing": None, "survey_executor": InlineExecutor()},
    )


def quick_solver(core: CoreRig) -> object:
    return core.app.alignment._solver


class TestTheQuickSolver:
    def test_the_default_is_a_solver_in_a_worker_process_that_has_not_started(
        self, tmp_path: Path
    ) -> None:
        core = rig(tmp_path)
        try:
            solver = quick_solver(core)
            assert isinstance(solver, ProcessQuickSolver)
            assert not solver.running  # the worker starts at the first solve
            assert solver.tracker is core.app.tracker  # the tracker stays in `core`
        finally:
            core.app.stop()

    def test_the_worker_detects_only_the_bright_stars(self, tmp_path: Path) -> None:
        core = rig(tmp_path)
        try:
            solver = quick_solver(core)
            assert isinstance(solver, ProcessQuickSolver)
            assert solver.spec is not None
            detect = solver.spec.config["detect"]
            assert (detect["threshold_sigma"], detect["max_stars"]) == (8.0, 300)
            survey_detect = core.app.survey_config.detect  # the survey path keeps its own
            assert (survey_detect.threshold_sigma, survey_detect.max_stars) == (5.0, 3000)
        finally:
            core.app.stop()

    def test_the_options_of_the_alignment_section_reach_the_worker(self, tmp_path: Path) -> None:
        core = rig(
            tmp_path,
            "[alignment]\ndetect_threshold_sigma = 6.5\ndetect_max_stars = 120\n"
            "solve_timeout_s = 30.0\n",
        )
        try:
            solver = quick_solver(core)
            assert isinstance(solver, ProcessQuickSolver)
            assert solver.spec is not None
            detect = solver.spec.config["detect"]
            assert (detect["threshold_sigma"], detect["max_stars"]) == (6.5, 120)
            assert solver._timeout_s == 30.0
        finally:
            core.app.stop()

    def test_the_thread_mode_builds_the_pipeline_in_this_process(self, tmp_path: Path) -> None:
        core = rig(tmp_path, '[alignment]\nsolver_mode = "thread"\n')
        try:
            solver = quick_solver(core)
            assert isinstance(solver, QuickSolver)
            options = solver._pipeline._detect_options  # type: ignore[attr-defined]
            assert (options.threshold_sigma, options.max_stars) == (8.0, 300)
        finally:
            core.app.stop()

    @pytest.mark.parametrize("mode", ["thread", "inline"])
    def test_a_survey_worker_that_is_not_a_process_gives_a_thread_too(
        self, tmp_path: Path, mode: str
    ) -> None:
        core = rig(tmp_path, f'[services.core.survey_worker]\nmode = "{mode}"\n')
        try:
            assert isinstance(quick_solver(core), QuickSolver)
        finally:
            core.app.stop()

    def test_a_catalog_that_does_not_load_leaves_the_helper_without_a_solver(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR, logger="seeingmon.services.core.app"):
            core = rig(tmp_path, catalog=False)
        try:
            assert quick_solver(core) is None
            assert "the catalog does not load" in caplog.text
        finally:
            core.app.stop()

    def test_without_a_catalog_there_is_no_solver(self, tmp_path: Path) -> None:
        core = build_rig(tmp_path)
        try:
            assert quick_solver(core) is None
        finally:
            core.app.stop()

    def test_stopping_core_closes_the_solver(self, tmp_path: Path) -> None:
        core = rig(tmp_path)
        solver = quick_solver(core)
        assert isinstance(solver, ProcessQuickSolver)
        core.app.start()
        core.app.stop()
        assert solver._closed  # no later solve starts a worker
