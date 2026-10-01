"""The survey worker: priority, the out-of-memory score, and the choice of executor."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.services.core import survey_worker
from seeingmon.services.core.settings import SurveyWorkerSettings
from seeingmon.survey.analyzer import InlineExecutor
from seeingmon.survey.pipeline import PipelineSpec

SPEC = PipelineSpec(
    station_id="test", profile={}, config={}, catalog_path="none", solvers=(), hot_pixel_file=""
)


class TestOomScore:
    def test_the_value_is_written_to_the_file(self, tmp_path: Path) -> None:
        (tmp_path / "oom_score_adj").write_text("0")
        assert survey_worker.raise_oom_score(750, tmp_path) == "oom_score_adj 750"
        assert (tmp_path / "oom_score_adj").read_text() == "750"

    def test_a_system_without_the_file_says_so(self, tmp_path: Path) -> None:
        assert (
            survey_worker.raise_oom_score(500, tmp_path / "missing")
            == "not supported on this platform"
        )


def test_the_priority_of_a_process_goes_down_without_an_error() -> None:
    """Run it in a child process, so that the test process keeps its own priority."""
    code = (
        "from seeingmon.services.core.survey_worker import lower_process_priority;"
        "print(lower_process_priority(10))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True
    )
    assert result.stdout.strip()  # a sentence, whatever the platform could do
    assert "Traceback" not in result.stderr


class TestInitializer:
    def test_the_priority_comes_before_the_pipeline_is_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[Any] = []

        def fake_nice(nice: int) -> str:
            calls.append(("nice", nice))
            return "x"

        def fake_oom(value: int) -> str:
            calls.append(("oom", value))
            return "y"

        monkeypatch.setattr(survey_worker, "lower_process_priority", fake_nice)
        monkeypatch.setattr(survey_worker, "raise_oom_score", fake_oom)
        monkeypatch.setattr(survey_worker, "init_worker", lambda spec: calls.append(("init", spec)))
        survey_worker.init_survey_worker(SPEC, 12, 600)
        assert calls == [("nice", 12), ("oom", 600), ("init", SPEC)]


class TestExecutors:
    def test_the_modes_make_the_executors_they_name(self) -> None:
        inline = survey_worker.make_survey_executor(SPEC, SurveyWorkerSettings(mode="inline"))
        assert isinstance(inline, InlineExecutor)
        thread = survey_worker.make_survey_executor(SPEC, SurveyWorkerSettings(mode="thread"))
        assert isinstance(thread, ThreadPoolExecutor)
        thread.shutdown()
        process = survey_worker.make_survey_executor(SPEC, SurveyWorkerSettings(mode="process"))
        try:
            assert isinstance(process, ProcessPoolExecutor)
        finally:
            process.shutdown()  # no worker started yet, because no job ran
