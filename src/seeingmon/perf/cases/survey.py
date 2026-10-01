"""The `survey` case: one synthetic bin2 survey frame through the production analyzer.

**The frame.** The simulator renders the full bin2 readout of the reference profile (4144 x 2822
pixels, 30 s at gain 120) of a made-up star field: stars at the density of the real catalog, with
trails, sky, noise, and hot pixels. The catalog of the pipeline holds the same stars. The simulator
places each star at its apparent place (precession, nutation, and aberration included), so the
pipeline finds the geometry that it expects, and the frame solves.

**The path.** `create_survey_analyzer` builds the analyzer, and `make_process_executor` gives it
a worker process, as `core` does. The tracker holds a solution of the pointing, so each frame takes
the steady-state route (detect, track, match, and records). The first frame after a cold start
needs a plate solver, which runs outside Python and which the dev machine does not have, so the
case does not measure a solve by a solver (the architecture's solver table estimates it).

**The figures.** The case reports the wall time of a frame from `submit` to the result, the CPU time
of the worker for it, the time of each stage (the pipeline reports them, and the analyzer does not
pass them on, so the case submits the same worker function itself for them), the cost of the
process boundary (the job minus its stages), the start of the worker, and the peak memory of the
worker, which is the figure to compare with the 550 MB of the budget. The peak of this process
includes the simulator, so the case reports it in the case summary only.
"""

from __future__ import annotations

import math
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seeingmon.perf.memory import current_rss_bytes, peak_rss_bytes, process_cpu_ns
from seeingmon.perf.registry import REGISTRY, CaseContext
from seeingmon.perf.report import Measurement
from seeingmon.perf.timing import TimingStats

if TYPE_CHECKING:
    from seeingmon.frames import Frame
    from seeingmon.profile import Profile
    from seeingmon.survey.analyzer import SurveyPipelineAnalyzer
    from seeingmon.survey.pointing import PointingSolution

REFERENCE_PROFILE = "asi294mm-gs250"
EXPOSURE_S = 30.0
GAIN = 120
_RESULT_TIMEOUT_S = 900.0
_STAGES = ("detect", "solve", "match", "records")


@dataclass(slots=True)
class Scene:
    """What the case needs: the profile, the frame, the catalog file, and a prior solution."""

    profile: Profile
    frame: Frame
    catalog_path: str
    solution: PointingSolution
    stars: int


def cropped_profile(profile: Profile, width: int, height: int) -> Profile:
    """The profile with a smaller sensor: `width` x `height` in bin2, twice that in bin1.

    The pixel size and the focal length stay, so the plate scale is the same and the field shrinks.
    """
    from seeingmon.profile import Profile as ProfileModel

    data = profile.model_dump(mode="python")
    for mode in data["readout_modes"]:
        factor = 2 // mode["sdk_bin"]
        mode["width_px"] = width * factor
        mode["height_px"] = height * factor
    return ProfileModel.model_validate(data)


def build_scene(folder: Path, *, smoke: bool) -> Scene:
    """Render the frame, write the catalog, and build the prior solution."""
    import numpy as np

    from seeingmon.clock import DEFAULT_START_UTC_NS, NS_PER_S, VirtualClock
    from seeingmon.drivers.sim import Pointing, SimOptions, StarField, create, make_polar_field
    from seeingmon.frames import StreamConfig, StreamKind
    from seeingmon.profile import load_profile
    from seeingmon.survey import apparent
    from seeingmon.survey.catalog import CapCatalog, write_catalog
    from seeingmon.survey.geometry import ARCSEC_PER_RAD, vector_to_radec
    from seeingmon.survey.pointing import PointingSolution
    from seeingmon.survey.wcs_fit import CameraAttitude

    profile = load_profile(REFERENCE_PROFILE)
    cap_radius_deg = 2.0 if smoke else 15.0
    if smoke:
        profile = cropped_profile(profile, 600, 400)
    field = make_polar_field(
        seed=3, cap_radius_deg=cap_radius_deg, mag_limit=13.5, include_polaris_b=False
    )
    stars = len(field)
    catalog = CapCatalog.from_columns(
        source_id=np.arange(1, stars + 1, dtype=np.int64),
        ra_deg=field.ra_deg,
        dec_deg=field.dec_deg,
        g_mag=field.mag,
        cap_radius_deg=cap_radius_deg,
        gaia_mag_limit=13.5,
    )
    catalog_path = str(folder / "catalog.smcat")
    write_catalog(catalog_path, catalog)

    # The simulator places a star where it is told, so tell it the apparent place at the time of
    # the frame, which is what the pipeline predicts from the catalog.
    t_frame = DEFAULT_START_UTC_NS + round(EXPOSURE_S / 2 * NS_PER_S)
    epoch = apparent.epoch_from_utc_ns(t_frame, 0.0)
    vectors = apparent.apparent_vectors(
        catalog.ra_deg,
        catalog.dec_deg,
        catalog.pm_ra_mas_yr,
        catalog.pm_dec_mas_yr,
        catalog.parallax_mas,
        epoch,
        catalog_epoch_jyear=catalog.epoch_jyear,
    )
    ra_apparent, dec_apparent = vector_to_radec(vectors)
    sky = StarField.from_arrays(ra_apparent, dec_apparent, catalog.g_mag)
    brightest = int(np.argmin(catalog.g_mag))  # Polaris
    pointing = Pointing(
        ra_deg=float(ra_apparent[brightest]),
        dec_deg=float(dec_apparent[brightest]),
        roll_deg=0.0,
        t_ref_utc_ns=t_frame,
    )
    driver = create(
        profile=profile,
        clock=VirtualClock(),
        options=SimOptions(seed=5, stars=sky, pointing=pointing, twilight=False),
    )
    driver.open()
    driver.configure(StreamConfig("bin2", round(EXPOSURE_S * 1e6), GAIN, kind=StreamKind.SNAPSHOT))
    driver.start()
    frame = driver.read_frame(timeout_s=EXPOSURE_S + 60.0)
    driver.stop()
    driver.close()

    # The attitude that the simulator used: x to the right, y down, z along the boresight.
    ra, dec = math.radians(pointing.ra_deg), math.radians(pointing.dec_deg)
    axis = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
    north = np.array([-math.sin(dec) * math.cos(ra), -math.sin(dec) * math.sin(ra), math.cos(dec)])
    west = np.array([math.sin(ra), -math.cos(ra), 0.0])
    rotation = np.stack([west, -north, axis])
    readout = profile.mode("bin2")
    attitude = CameraAttitude(
        rotation,
        profile.plate_scale_arcsec_per_px("bin2") / ARCSEC_PER_RAD,
        1,
        ((readout.width_px - 1) / 2.0, (readout.height_px - 1) / 2.0),
    )
    solution = PointingSolution.from_attitude(
        attitude,
        epoch,
        mode="bin2",
        width_px=readout.width_px,
        height_px=readout.height_px,
        n_matched=100,
        rms_arcsec=0.5,
        solver="synthetic",
    )
    return Scene(profile, frame, catalog_path, solution, stars)


def analyze_once(analyzer: SurveyPipelineAnalyzer, frame: Frame) -> bool:
    """Submit the frame, wait for the result, and return whether the frame solved."""
    analyzer.submit(frame)
    deadline = time.monotonic() + _RESULT_TIMEOUT_S
    while time.monotonic() < deadline:
        outputs = analyzer.poll()
        if outputs:
            return bool(outputs[0].solved)
        time.sleep(0.002)
    raise TimeoutError(f"the analyzer returned no result in {_RESULT_TIMEOUT_S:.0f} s")


@REGISTRY.case("survey", summary="One synthetic bin2 survey frame through the process worker")
def survey(ctx: CaseContext) -> list[Measurement]:
    from seeingmon.frames import encode_frame
    from seeingmon.survey.analyzer import (
        analyzer_spec,
        create_survey_analyzer,
        make_process_executor,
        run_job,
    )
    from seeingmon.survey.config import SurveyConfig

    ctx.mark_baseline()
    repeats = ctx.pick(5, 1)
    stage_repeats = ctx.pick(3, 1)
    with tempfile.TemporaryDirectory(prefix="smon-perf-", ignore_cleanup_errors=True) as name:
        scene = build_scene(Path(name), smoke=ctx.smoke)
        config = SurveyConfig(catalog_path=scene.catalog_path, solvers=())
        spec = analyzer_spec(profile=scene.profile, station_id="perf", config=config)
        executor = make_process_executor(spec)
        analyzer = create_survey_analyzer(
            profile=scene.profile, station_id="perf", config=config, executor=executor
        )
        try:
            started = time.perf_counter()
            baseline = executor.submit(current_rss_bytes).result(timeout=_RESULT_TIMEOUT_S)
            startup_s = time.perf_counter() - started
            analyzer.tracker.update(scene.solution)
            if not analyze_once(analyzer, scene.frame):  # the first frame also warms the worker
                raise RuntimeError("the synthetic frame did not solve")
            wall: list[float] = []
            worker_cpu: list[float] = []
            parent_cpu: list[float] = []
            for _ in range(repeats):
                before = executor.submit(process_cpu_ns).result(timeout=_RESULT_TIMEOUT_S)
                parent_before = process_cpu_ns()
                begin = time.perf_counter()
                solved = analyze_once(analyzer, scene.frame)
                wall.append(time.perf_counter() - begin)
                parent_cpu.append((process_cpu_ns() - parent_before) / 1e9)
                after = executor.submit(process_cpu_ns).result(timeout=_RESULT_TIMEOUT_S)
                worker_cpu.append((after - before) / 1e9)
                if not solved:
                    raise RuntimeError("the synthetic frame did not solve")
            previous = analyzer.tracker.solution
            frame_bytes = encode_frame(scene.frame)
            stages: dict[str, list[float]] = {stage: [] for stage in _STAGES}
            boundaries: list[float] = []
            last: dict[str, Any] = {}
            for index in range(stage_repeats):
                begin = time.perf_counter()
                last = executor.submit(
                    run_job,
                    frame_bytes,
                    None if previous is None else previous.to_dict(),
                    None,
                    index,
                ).result(timeout=_RESULT_TIMEOUT_S)
                job_s = time.perf_counter() - begin
                for stage in _STAGES:
                    stages[stage].append(float(last["timings"].get(stage, 0.0)))
                # What the job took beyond its stages: the frame crosses the boundary twice.
                boundaries.append(job_s - sum(float(v) for v in last["timings"].values()))
            worker_peak = executor.submit(peak_rss_bytes).result(timeout=_RESULT_TIMEOUT_S)
        finally:
            analyzer.close()
            executor.shutdown(wait=True)

    total = TimingStats.from_samples(wall)
    stage_medians = {stage: statistics.median(values) for stage, values in stages.items()}
    detected = next(
        (
            int(item["row"].get("n_detected", 0))
            for item in last.get("records", [])
            if item["record_type"] == "survey_frame"
        ),
        0,
    )
    solution = last.get("solution") or {}
    height, width = scene.frame.data.shape
    detail: dict[str, float | int | str] = {
        "width_px": width,
        "height_px": height,
        "exposure_s": EXPOSURE_S,
        "catalog_stars": scene.stars,
        "detections": detected,
        "matched": int(solution.get("n_matched", 0)),
        "repeats": repeats,
        "route": "tracker, with a prior solution",
    }
    boundary = max(statistics.median(boundaries), 1e-6)
    measurements = [
        Measurement("frame.total", "s", total.median, total, "numpy", detail),
        Measurement(
            "frame.worker_cpu", "s", max(statistics.median(worker_cpu), 1e-6), None, "numpy", {}
        ),
        Measurement(
            "frame.parent_cpu", "s", max(statistics.median(parent_cpu), 1e-6), None, "numpy", {}
        ),
        *[
            Measurement(f"stage.{stage}", "s", max(value, 1e-6), None, "numpy", {})
            for stage, value in stage_medians.items()
        ],
        Measurement("frame.boundary", "s", boundary, None, "numpy", {}),
        Measurement("worker.startup", "s", startup_s, None, "interpreter", {}),
        Measurement(
            "worker.baseline_rss", "bytes", float(baseline or 1), None, "memory", {"after": "init"}
        ),
        Measurement(
            "worker.peak_rss",
            "bytes",
            float(worker_peak or 1),
            None,
            "memory",
            {"process": "worker"},
        ),
    ]
    ctx.note(
        "The frame is rendered by the simulator, and the catalog holds the same stars. The worker "
        "is a separate process, started with make_process_executor. The stage times come from "
        "the worker function that the analyzer submits. No plate solver ran."
    )
    return measurements
