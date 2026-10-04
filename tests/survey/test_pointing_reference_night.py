"""A reference from the records of the real survey pipeline, and the next run that uses it.

The first run solves two frames of a synthetic night with no reference, and its `pointing` records
go into a store. `seeingmon pointing set-reference` makes the reference file from that store. A
second analyzer, which `[survey.pointing] reference_file` points at the file, then analyzes two more
frames, and the mount slips on the last. The records of the second run carry the ID of the
reference and the offset: about 0 before the slip, and the slip after it.

The rebuild checks itself against the position of Polaris in the record, so this test also shows
that the frame size, the principal point, and the parity that the command assumes are the ones that
the pipeline used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from seeingmon.cli import main
from seeingmon.clock import NS_PER_S
from seeingmon.frames import Frame
from seeingmon.profile import Profile
from seeingmon.records.survey import PointingRecord
from seeingmon.store.db import Store
from seeingmon.store.layout import DataLayout
from seeingmon.survey import pointing as pt
from seeingmon.survey import pointing_cli
from seeingmon.survey.analyzer import InlineExecutor, SurveyPipelineAnalyzer
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.geometry import ARCSEC_PER_RAD, exp_so3
from seeingmon.survey.pipeline import SurveyPipeline
from tests.survey import synth

STEP_NS = 180 * NS_PER_S
SLIP_PX = 3.0
T0 = synth.NIGHT_UTC_NS


@pytest.fixture(scope="module")
def profile() -> Profile:
    return synth.cropped_profile(1200, 800)


@pytest.fixture(scope="module")
def catalog() -> CapCatalog:
    return synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=3.0, seed=1)


@pytest.fixture(scope="module")
def frames(profile: Profile, catalog: CapCatalog) -> list[tuple[Frame, synth.SynthTruth]]:
    """Frames 0 and 1 and frames 2 and 3 come from one mount. Frame 3 comes after a slip."""
    rotation = synth.make_attitude(0.9, 40.0, 25.0)
    slipped = exp_so3([0.0, SLIP_PX * 3.82 / ARCSEC_PER_RAD, 0.0]) @ rotation
    return [
        synth.render_frame(
            catalog,
            profile,
            rotation_tirs=slipped if index == 3 else rotation,
            t_utc_ns=T0 + index * STEP_NS,
            exposure_s=30.0,
            seed=60 + index,
        )
        for index in range(4)
    ]


def analyze(
    profile: Profile,
    catalog: CapCatalog,
    items: list[tuple[Frame, synth.SynthTruth]],
    config: SurveyConfig | None = None,
) -> list[PointingRecord]:
    """The `pointing` records of a new analyzer for the frames. The first one needs a solver."""
    solver = synth.QueueSolver([synth.truth_solve_result(items[0][1], catalog)])
    pipeline = SurveyPipeline(station_id="test", profile=profile, catalog=catalog, solvers=[solver])
    analyzer = SurveyPipelineAnalyzer(
        profile=profile,
        station_id="test",
        pipeline=pipeline,
        executor=InlineExecutor(),
        config=config,
    )
    records: list[PointingRecord] = []
    for frame, _ in items:
        analyzer.submit(frame)
        for output in analyzer.poll():
            records.extend(r for r in output.records if isinstance(r, PointingRecord))
    return records


def test_the_pipeline_uses_the_reference_that_the_command_made_from_its_own_records(
    profile: Profile,
    catalog: CapCatalog,
    frames: list[tuple[Frame, synth.SynthTruth]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("SEEINGMON_PATHS__DATA_DIR", "SEEINGMON_SURVEY__CALIBRATION_DIR"):
        monkeypatch.delenv(name, raising=False)
    first = analyze(profile, catalog, frames[:2])
    assert [record.reference_id for record in first] == [None, None]
    assert all(record.solver != "none" for record in first)

    layout = DataLayout(tmp_path / "data")
    layout.create()
    with Store.open(layout.db_path) as store:
        for record in first:
            store.write(record)
    real = pointing_cli._load_context

    def small_profile(args: argparse.Namespace) -> pointing_cli.PointingContext:
        context = real(args)  # the data directory and the survey table come from the arguments
        return pointing_cli.PointingContext(
            profile, context.survey, context.data_dir, context.calibration_dir
        )

    monkeypatch.setattr(pointing_cli, "_load_context", small_profile)
    monkeypatch.setattr(pointing_cli, "_now_utc_ns", lambda: T0 + 20 * 60 * NS_PER_S)
    code = main(
        ["pointing", "set-reference", "--data-dir", str(layout.root), "--min-matched", "50"]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    reference_file = layout.root / "calibration" / "pointing-reference.json"
    saved = pt.load_reference(reference_file)
    assert saved is not None
    reference_id = saved.reference_id
    assert reference_id == "reference-20261001T220300Z"  # the newest of the two frames

    config = SurveyConfig.model_validate({"pointing": {"reference_file": str(reference_file)}})
    second = analyze(profile, catalog, frames[2:], config)
    assert [record.reference_id for record in second] == [reference_id, reference_id]
    steady, slipped = second
    assert steady.offset_arcmin == pytest.approx(0.0, abs=0.03)
    assert steady.quality is None  # no "no reference solution" note
    assert slipped.offset_arcmin == pytest.approx(SLIP_PX * 3.82 / 60.0, abs=0.03)
    assert slipped.flags == []  # 0.19 arcmin is under the 5 arcmin of moved_arcmin
    # Without the file, the same frames carry no offset and say why.
    plain = analyze(profile, catalog, frames[2:])
    assert [record.offset_arcmin for record in plain] == [None, None]
    assert plain[0].quality == {"offset_arcmin": "no reference solution"}
