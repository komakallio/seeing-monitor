"""`CoreApp`: the composition root of `core`.

`CoreApp` reads the configuration and the profile, opens the storage, and builds the scheduler with
everything that it needs. It then runs the parts on their own threads and shuts them down in the
right order. The parts, and where they come from:

- **Storage:** `open_storage(config, clock, sinks=build_sinks(config))` opens the store, the segment
  files, the forwarder, and retention.
- **Camera:** `RemoteCameraDriver.from_config`, a driver that talks to `acquire`.
- **Fast analysis:** `create_fast_analyzer(profile, [fastpath], station_id)`. The scheduler gets it
  through a `LiveFastAnalyzer`, which hands every frame to the live video of Polaris after the
  analysis (`seeingmon.services.core.live`).
- **Survey analysis:** the survey analyzer over an executor from `make_survey_executor`, which is a
  worker process at a low priority. The `PointingTracker` of the analyzer is the pointing provider.
- **Survey frames:** `SurveyFrames` wraps the survey analyzer. It keeps the newest frames in RAM,
  and it writes the previews and the FITS files (`seeingmon.services.core.survey_frames`).
- **Preview calibration:** one `PreviewCalibrator` takes the dark level, the vignetting, and the
  dust shadows out of the previews of the survey frames and out of the live view of the alignment
  helper (`seeingmon.services.core.alignment.calibration`). It reads `[survey] flat_file` and the
  dark library of the survey analysis.
- **Scheduler:** `build_scheduler(...)`, with the store as the record writer and the segment writer
  as the metrics writer.
- **Heater, SQM-LE, power:** the sections `[heater]`, `[sqm]`, and `[power]`. Each stays off until
  you configure it. `[sqm] source` picks the SQM-LE reader: `tcp` polls the unit over the LAN, and
  `influx` reads the readings that another program wrote to InfluxDB (`create_sqm_reader`).
- **Alignment helper:** the live view and the quick solve (`seeingmon.services.core.alignment`).
- **RPC and streams:** an `IpcServer` at the core address with the channels `rpc`, `alignment`,
  and `polaris`.

**Threads.** The scheduler thread runs the loop, and it is the only thread that touches the camera,
the analyzers, and the writers of the scheduler. The housekeeping thread forwards rows to the sinks,
closes idle segments, and runs retention. The heater and SQM-LE threads run when those parts are
configured. The frame writer thread writes the previews and the FITS files of the survey frames.
The supervisor thread runs the periodic jobs: the `health` record, the collection of the events of
`acquire`, and the heartbeat to systemd. The alignment helper has two threads, the video of
Polaris has one (it encodes frames only while a client watches), and the IPC server has its own.
All of them stop in this order: new commands, the scheduler (which closes the camera), the
alignment helper, the video of Polaris, the heater and the SQM-LE reader, the supervisor, the
survey worker, the frame writer (which writes the files that wait), and last the housekeeping and
the storage, so that the final rows reach the disk.

**A run that a test drives.** With `threads=False`, `start` starts no thread. The test steps the
scheduler and calls `tick`, which does the periodic work and one pass of the housekeeping, on the
calling thread. With a `VirtualClock`, a simulated night then runs in seconds and deterministically.
Pass a `storage_clock` whose `sleep` does nothing, so that the housekeeping never moves virtual
time.

**Records.** At start the app writes a `run` record (once the scheduler has opened the camera, or
after `run_record_wait_s`), and every `health_interval_s` a `health` record. The hardware events
of the heater, the SQM-LE reader, the power-cycle hook, and the driver in `acquire` become `event`
records, and the SQM-LE readings become `reference` records.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from seeingmon.analysis import FastAnalyzer, PointingProvider, SurveyAnalyzer
from seeingmon.clock import NS_PER_S, Clock
from seeingmon.config import Config, ConfigError
from seeingmon.drivers.base import CameraDriver, CameraInfo
from seeingmon.fastpath import FastPathConfig, create_fast_analyzer
from seeingmon.hardware.heater import HeaterConfig, HeaterController, create_heater
from seeingmon.hardware.power import CommandRunner, PowerConfig, PowerCycle
from seeingmon.hardware.sqm import SqmConfig, SqmReader
from seeingmon.hardware.sqm_factory import create_sqm_reader
from seeingmon.profile import ProfileError
from seeingmon.records import ReferenceRecord
from seeingmon.scheduler import (
    Command,
    CommandResult,
    CommissionResult,
    QueueDark,
    Scheduler,
    SchedulerConfig,
    State,
    build_scheduler,
    load_site,
)
from seeingmon.scheduler.status import SchedulerStatus
from seeingmon.services.config import ServicesConfig
from seeingmon.services.core.alignment.calibration import PreviewCalibrator
from seeingmon.services.core.alignment.helper import AlignmentHelper, Solver
from seeingmon.services.core.alignment.solve import QuickSolver
from seeingmon.services.core.alignment.worker import ProcessQuickSolver
from seeingmon.services.core.commissioning.burst import BurstHandler
from seeingmon.services.core.commissioning.replay import (
    ReplayHandler,
    clean_options,
    resolve_source,
)
from seeingmon.services.core.context import ContextProvider
from seeingmon.services.core.driver_proxy import InfoDriver
from seeingmon.services.core.escalation import Escalator
from seeingmon.services.core.events import EventPump, EventSource, EventWriter
from seeingmon.services.core.health import HealthReporter, build_run_record
from seeingmon.services.core.history import StoreZeroPointHistory
from seeingmon.services.core.live import LiveFastAnalyzer, PolarisStream, live_view
from seeingmon.services.core.liveness import BeatClock, Liveness
from seeingmon.services.core.nightly import NightlySummary
from seeingmon.services.core.periodic import PeriodicTasks
from seeingmon.services.core.polaris import (
    DEFAULT_APERTURE_PX,
    FrameSlot,
    PolarisRenderer,
)
from seeingmon.services.core.rpc import CoreRpc
from seeingmon.services.core.settings import AlignmentSettings, ReplaySettings
from seeingmon.services.core.skyflags import SkyFlagWriter
from seeingmon.services.core.survey_frames import SurveyFrames
from seeingmon.services.core.survey_worker import make_survey_executor
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.server import IpcServer
from seeingmon.services.ipc.stream import StreamWindow
from seeingmon.services.notify import SystemdNotifier
from seeingmon.services.remote import RemoteCameraDriver
from seeingmon.services.web.contract import (
    ALIGNMENT_CHANNEL,
    POLARIS_CHANNEL,
    RPC_CHANNEL,
    LiveSeeingView,
)
from seeingmon.sinks.base import Sink
from seeingmon.store.db import DuplicateRecordError
from seeingmon.store.retention import DiskProbe
from seeingmon.store.wiring import Storage, open_storage

_log = logging.getLogger(__name__)

EXIT_THREAD_DIED = 71
INSTANCE_BYTES = 8
SUPERVISOR_SLICE_S = 1.0
JOIN_SLICE_S = 5.0
STATUS_INTERVAL_S = 1.0
NIGHTLY_INTERVAL_S = 30.0


class Hardware(EventSource, Protocol):
    """What `core` needs from the camera side beyond `CameraDriver`: `RemoteCameraDriver` fits."""

    def health(self) -> dict[str, Any]: ...

    def request_restart(self, reason: str = ...) -> None: ...

    @property
    def connected(self) -> bool: ...


@dataclass(slots=True)
class CoreParts:
    """Parts that replace what `CoreApp` builds from the configuration. Leave a part `None` for
    the default. A test passes fakes here; `seeingmon core` passes none."""

    driver: CameraDriver | None = None
    remote: Hardware | None = None
    fast: FastAnalyzer | None = None
    survey: SurveyAnalyzer | None = None
    tracker: Any = None
    pointing: PointingProvider | None = None
    quick_solver: Solver | None = None
    sinks: Sequence[Sink] | None = None
    disk_usage: DiskProbe | None = None
    storage_clock: Clock | None = None
    heater: HeaterController | None = None
    sqm: SqmReader | None = None
    power: PowerCycle | None = None
    notifier: SystemdNotifier | None = None
    runner: CommandRunner | None = None
    survey_executor: Executor | None = None
    extra_versions: dict[str, str] = field(default_factory=dict)


class CoreApp:
    """The composition root of `core`. Build it, `start` it, and `stop` it, or call `run`."""

    def __init__(
        self,
        config: Config,
        services: ServicesConfig,
        clock: Clock,
        key: ConnectionKey,
        *,
        parts: CoreParts | None = None,
        endpoint: Endpoint | None = None,
        threads: bool = True,
        alignment_settings: AlignmentSettings | None = None,
    ) -> None:
        self.config = config
        self.services = services
        self.clock = clock
        self.parts = parts or CoreParts()
        self.threads = threads
        self.instance = secrets.token_hex(INSTANCE_BYTES)
        self.profile = config.profile
        self.station_id = config.station_id
        self.endpoint = endpoint or services.endpoint("core")
        self.settings = services.core
        self.alignment_settings = alignment_settings or config.section(
            "alignment", AlignmentSettings
        )
        self._key = key
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._started = False
        self._stopped = False
        self._exit_reason = "stopped"
        self._fatal: str | None = None
        self._threads: dict[str, threading.Thread] = {}
        self._camera: CameraInfo | None = None
        self._run_record_written = False
        self._health_key: tuple[Any, ...] | None = None  # what the newest health record said
        self._started_mono = clock.monotonic_ns()
        self.bound_endpoint: Endpoint | None = None
        self.storage: Storage | None = None
        try:
            self._build()
        except BaseException:
            self._release_partial()
            raise

    # --- Building --------------------------------------------------------------------------

    def _build(self) -> None:
        parts = self.parts
        config, services, clock = self.config, self.services, self.clock
        scheduler_config = config.section("scheduler", SchedulerConfig)
        fast_config = config.section("fastpath", FastPathConfig)
        if scheduler_config.fast.analysis_window_s != fast_config.window_s:
            raise ConfigError(
                "scheduler.fast.analysis_window_s must equal fastpath.window_s, "
                f"but they are {scheduler_config.fast.analysis_window_s:g} s "
                f"and {fast_config.window_s:g} s"
            )
        self.fast_config = fast_config

        self.storage = open_storage(
            config,
            parts.storage_clock or clock,
            sinks=parts.sinks,
            **({} if parts.disk_usage is None else {"disk_usage": parts.disk_usage}),
        )
        storage = self.storage
        self.events = EventWriter(
            storage.store.write,
            station_id=self.station_id,
            profile_id=self.profile.id,
            clock=clock,
            source="local",
        )
        self.acquire_events = EventWriter(
            storage.store.write,
            station_id=self.station_id,
            profile_id=self.profile.id,
            clock=clock,
            source="acquire",
        )

        # The scheduler thread proves that it lives through the clock it reads and the camera calls
        # it makes. The heartbeat to systemd depends on that proof.
        self.liveness = Liveness(clock, self.settings.scheduler_stall_s)
        self.beat_clock = BeatClock(clock, self.liveness)

        # The camera, as the scheduler sees it, and as the escalation and the health see it.
        self.remote: Hardware | None
        raw: CameraDriver
        if parts.driver is not None:
            raw, self.remote = parts.driver, parts.remote
        else:
            remote = RemoteCameraDriver.from_config(services, clock=clock)
            raw, self.remote = remote, remote
        self.driver = InfoDriver(
            raw,
            self._on_camera_opened,
            liveness=self.liveness,
            call_limit_s=self.settings.driver_call_limit_s,
        )

        self.fast = parts.fast or create_fast_analyzer(self.profile, fast_config, self.station_id)
        self._build_polaris()
        self._build_survey()
        self._build_hardware()
        self._build_alignment()
        site = load_site(config)
        self.escalator = Escalator(
            writer=self.events,
            clock=self.beat_clock,
            settings=self.settings.escalation,
            restart_acquire=None if self.remote is None else self.remote.request_restart,
            acquire_connected=None if self.remote is None else self._acquire_connected,
            power=self.power,
            runner=parts.runner,
        )
        self.scheduler: Scheduler = build_scheduler(
            config,
            driver=self.driver,
            fast=self.live_fast,
            survey=self.survey,
            pointing=self.pointing,
            records=SkyFlagWriter(
                storage.store.as_record_writer(),
                site=site,
                heater=self.heater,
                settings=self.settings.sky_flags,
            ),
            metrics=storage.segments,
            clock=self.beat_clock,
            escalate=self.escalator,
            context_provider=ContextProvider(
                clock=clock,
                site=site,
                window_s=fast_config.window_s,
                heater=self.heater,
            ),
            alignment_sink=self.alignment.sink,
            result_sink=self._on_result,
        )
        self._register_handlers()
        self._build_reporting()
        self._build_server()

    def _build_polaris(self) -> None:
        """The live video of Polaris, and the analyzer that feeds it.

        `live_fast` wraps the fast analyzer, and the scheduler gets the wrapper. The stream asks
        the profile for the plate scale of a mode, and the analyzer for the aperture of a stream,
        so that the width of the star in the video follows the stored windows.
        """
        settings = self.settings.polaris
        scales: dict[str, float | None] = {}
        apertures: dict[tuple[str, int, int, int, int], float] = {}

        def scale_for(mode: str) -> float | None:
            if mode not in scales:
                try:
                    scales[mode] = self.profile.plate_scale_arcsec_per_px(mode)
                except ProfileError:
                    scales[mode] = None
            return scales[mode]

        def aperture_for(slot: FrameSlot) -> float:
            bits = 8 * slot.data.dtype.itemsize
            key = (slot.mode, slot.gain, slot.exposure_us, slot.adc_bits, bits)
            if key not in apertures:
                setup = getattr(self.fast, "kernel_setup", None)
                apertures[key] = (
                    DEFAULT_APERTURE_PX if setup is None else setup(*key)[0].aperture_diameter_px
                )
            return apertures[key]

        self.polaris = PolarisStream(
            settings,
            clock=self.clock,
            renderer=PolarisRenderer(settings, scale_for=scale_for, aperture_for=aperture_for),
            live=self.live_seeing,
        )
        self.live_fast = LiveFastAnalyzer(self.fast, self.polaris)
        self._live_cache: tuple[Any, LiveSeeingView | None] = (None, None)

    def live_seeing(self) -> LiveSeeingView | None:
        """The rolling seeing value of the fast analyzer as the contract describes it, or `None`.

        The video and the `live_seeing` method both call it, so it converts a new value once.
        """
        current = self.live_fast.live
        cached, view = self._live_cache
        if current is not cached:
            view = live_view(current)
            self._live_cache = (current, view)
        return view

    def _build_survey(self) -> None:
        parts, config = self.parts, self.config
        assert self.storage is not None
        from seeingmon.survey.analyzer import (
            analyzer_spec,
            create_survey_analyzer,
            with_calibration,
        )
        from seeingmon.survey.config import SurveyConfig
        from seeingmon.survey.quality import QualityOptions
        from seeingmon.survey.tracker import PointingTracker

        # The dark library is the one in the data directory, unless the configuration names another.
        self.survey_config = with_calibration(
            config.section("survey", SurveyConfig), self.storage.layout
        )
        self._build_dark()
        # The previews of the survey frames and the live view share one calibrator, so that they
        # share the flat and the dark model that it caches.
        self.preview_calibrator = PreviewCalibrator(
            self.survey_config, self.profile, library=self.dark_library, clock=self.clock
        )
        transparency = QualityOptions.from_config(self.survey_config).transparency
        self.tracker: PointingTracker | None = parts.tracker
        self.survey: SurveyAnalyzer
        self.history: StoreZeroPointHistory | None = None
        self._executor: Executor | None = None
        analyzer: Any
        if parts.survey is not None:
            analyzer = parts.survey
        else:
            if not self.survey_config.catalog_path:
                raise ConfigError(
                    "[survey] catalog_path is not set: core needs the cap catalog to follow "
                    "Polaris. Build it with `seeingmon catalog build`."
                )
            spec = analyzer_spec(
                profile=self.profile, station_id=self.station_id, config=self.survey_config
            )
            self._executor = parts.survey_executor or make_survey_executor(
                spec, self.settings.survey_worker
            )
            # The reference zero point comes from the store, so a restart of core keeps it. The
            # worker process never reads the history: the analyzer hands it the reference.
            self.history = StoreZeroPointHistory(
                self.storage.store, clock=self.clock, window_days=transparency.window_days
            )
            analyzer = create_survey_analyzer(
                profile=self.profile,
                station_id=self.station_id,
                config=self.survey_config,
                executor=self._executor,
                layout=self.storage.layout,
                history=self.history,
                clock=self.clock,
            )
            self.tracker = analyzer.tracker
            self._load_seed(self.tracker)
            self._seed_from_store(self.tracker)
            self._log_pointing_start(self.tracker, [solver.kind for solver in spec.solvers])
        # The frames of the survey path: a ring in RAM, the previews, and the FITS files.
        self.frames: SurveyFrames | None = None
        frame_settings = self.settings.survey_frames
        if frame_settings.enabled:
            self.frames = SurveyFrames(
                analyzer,
                layout=self.storage.layout,
                profile=self.profile,
                station_id=self.station_id,
                clock=self.clock,
                settings=frame_settings,
                long_min_exposure_s=self.survey_config.sky.min_exposure_s,
                capture_allowed=self.storage.capture_allowed,
                on_event=self.events.emit,
                calibrator=self.preview_calibrator,
            )
            analyzer = self.frames
        # An analyzer with a nightly summary gets its nights closed on time and at shutdown.
        self.nightly: NightlySummary | None = None
        if callable(getattr(analyzer, "flush_night", None)):
            self.nightly = NightlySummary(
                analyzer,
                write=self.storage.store.write,
                clock=self.clock,
                split_utc_hour=transparency.night_split_utc_hour,
            )
            analyzer = self.nightly
        self.survey = analyzer
        self.pointing: PointingProvider = parts.pointing or (
            self.tracker if self.tracker is not None else _NoPointing()
        )

    def _build_dark(self) -> None:
        """The dark library of the survey analysis, and the state of the dark task.

        The health record, the dark task, and the RPC share one library, the one that the survey
        analysis reads (`calibration_dir`), so that a set that the task adds reaches the next frame.
        """
        from seeingmon.services.core.commissioning.dark import DarkTaskState
        from seeingmon.survey.dark import DARKS_DIRNAME, DarkLibrary

        self.dark_library = DarkLibrary(Path(self.survey_config.calibration_dir) / DARKS_DIRNAME)
        self.dark_state = DarkTaskState(self.clock, self.survey_config.dark)

    def _load_seed(self, tracker: Any) -> None:
        """Start the tracker with the solution of `seed_solution_file`, when one is configured."""
        name = self.settings.seed_solution_file
        if not name:
            return
        from seeingmon.survey.pointing import PointingSolution

        try:
            data = json.loads(Path(name).read_text(encoding="utf-8"))
            tracker.update(PointingSolution.from_dict(data))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ConfigError(
                f"cannot read the seed solution: {type(error).__name__}: {error}"
            ) from None
        _log.info("the pointing tracker starts with the seed solution")

    def _seed_from_store(self, tracker: Any) -> None:
        """Start the tracker with the newest usable solution of the store, after a restart.

        A tracker that a seed file started keeps its solution. A solution that is too old, too
        thin, or from another profile, and a clock that does not know the time, leave the tracker
        empty, and the first survey frame goes to the plate solvers. Nothing here stops `core`.
        """
        if not self.settings.seed_from_store or tracker.solution is not None:
            return
        assert self.storage is not None
        if self.clock.status().synchronized is False:
            _log.info("the pointing tracker starts empty: the clock is not synchronized")
            return
        from seeingmon.survey.reference import (
            PointingReferenceError,
            format_age,
            format_time,
            seed_solution,
        )

        pointing = self.survey_config.pointing
        now_ns = self.clock.utc_ns()
        try:
            solution = seed_solution(
                self.storage.store,
                self.profile,
                now_utc_ns=now_ns,
                max_age_s=pointing.validity_s,
                min_matched=pointing.tracker_min_stars,
                max_rms_px=pointing.tracker_max_rms_px,
                dut1_s=self.survey_config.dut1_s,
            )
        except PointingReferenceError as error:
            _log.info("the pointing tracker has no stored solution to start with: %s", error)
            return
        except Exception:
            _log.exception("could not read a stored pointing solution, so the tracker starts empty")
            return
        tracker.update(solution)
        _log.info(
            "the pointing tracker starts with the stored solution of %s (%s old, solved by %s "
            "with %d matched stars)",
            format_time(solution.t_utc_ns),
            format_age((now_ns - solution.t_utc_ns) / NS_PER_S),
            solution.solver or "a solver",
            solution.n_matched,
        )

    @staticmethod
    def _log_pointing_start(tracker: Any, solvers: Sequence[str]) -> None:
        """Say in the log how a start without a pointing solution finds its first one.

        The scheduler then takes a survey frame, and the plate solvers of `[survey] solvers` run in
        the order of the list until one solves it. A person who follows the log sees that line
        first, and then one line for each solver run (`seeingmon.survey.analyzer`).
        """
        if tracker.solution is not None:
            return
        if solvers:
            _log.info(
                "no pointing solution yet: the survey frames go to the plate solvers %s, in this "
                "order, until one solves",
                ", ".join(solvers),
            )
        else:
            _log.warning(
                "no pointing solution yet, and [survey] solvers names no plate solver, so the "
                "scheduler cannot find Polaris"
            )

    def _build_hardware(self) -> None:
        parts, config, clock = self.parts, self.config, self.clock
        self.heater = parts.heater or create_heater(
            config.section("heater", HeaterConfig), clock=clock, on_event=self.events
        )
        sqm_config = config.section("sqm", SqmConfig)
        self.sqm: SqmReader | None = parts.sqm
        if self.sqm is None and sqm_config.enabled:
            # `[sqm] source` picks the class: the TCP reader or the reader of InfluxDB. A secret
            # that the Influx source takes from an environment variable is read here, so a variable
            # that is not set stops the start with a message that names it.
            self.sqm = create_sqm_reader(
                sqm_config,
                clock=clock,
                station_id=self.station_id,
                profile_id=self.profile.id,
                on_event=self.events,
            )
        self.power = parts.power or PowerCycle(
            config.section("power", PowerConfig), clock=clock, on_event=self.events
        )

    def _build_alignment(self) -> None:
        quick = self.parts.quick_solver
        if quick is None and self.tracker is not None and self.survey_config.catalog_path:
            quick = self._make_quick_solver()
        self.alignment = AlignmentHelper(
            settings=self.alignment_settings,
            profile=self.profile,
            clock=self.clock,
            is_active=lambda: self.scheduler.state is State.ALIGN,
            solver=quick,
            tracker=self.tracker,
            touch=lambda: self.scheduler.touch_alignment(),
            site=load_site(self.config),
            calibrator=self.preview_calibrator,
        )

    def _make_quick_solver(self) -> Solver | None:
        """The quick solve of the alignment helper, or `None` when the catalog does not load.

        It follows the survey worker: in a worker process when both `[alignment] solver_mode` and
        `[services.core.survey_worker] mode` say `process`, and in a thread of `core` otherwise.
        The quick solve detects only the bright stars and searches a binned copy of the frame
        (`[alignment] detect_*`), because the live view needs a few dozen stars and a fast answer.
        """
        from seeingmon.survey.analyzer import analyzer_spec
        from seeingmon.survey.catalog import read_info
        from seeingmon.survey.pipeline import build_pipeline

        assert self.tracker is not None
        settings = self.alignment_settings
        detect = self.survey_config.detect.model_copy(
            update={
                "threshold_sigma": settings.detect_threshold_sigma,
                "max_stars": settings.detect_max_stars,
                "coarse_bin": settings.detect_coarse_bin,
                "refine_stars": settings.detect_refine_stars,
            }
        )
        config = self.survey_config.model_copy(update={"detect": detect})
        pointing = config.pointing
        in_process = (
            settings.solver_mode == "thread" or self.settings.survey_worker.mode != "process"
        )
        try:
            spec = analyzer_spec(profile=self.profile, station_id=self.station_id, config=config)
            if in_process:
                return QuickSolver(
                    build_pipeline(spec, self.clock),
                    self.tracker,
                    self.clock,
                    min_stars=pointing.tracker_min_stars,
                    max_rms_px=pointing.tracker_max_rms_px,
                )
            read_info(spec.catalog_path)  # a bad catalog fails here, not in the worker
        except Exception:
            _log.exception("the catalog does not load, so the alignment has no quick solve")
            return None
        return ProcessQuickSolver(
            self.tracker,
            self.clock,
            spec=spec,
            min_stars=pointing.tracker_min_stars,
            max_rms_px=pointing.tracker_max_rms_px,
            nice=self.settings.survey_worker.nice,
            oom_score_adj=self.settings.survey_worker.oom_score_adj,
            timeout_s=settings.solve_timeout_s,
        )

    def _register_handlers(self) -> None:
        assert self.storage is not None
        from seeingmon.store.config import StoreConfig

        commissioning = self.settings.commissioning
        replay = self.config.section("replay", ReplaySettings)
        self.recordings_dir = Path(replay.recordings_dir) if replay.recordings_dir else None
        self.scheduler.register_handler(
            "burst",
            BurstHandler(
                layout=self.storage.layout,
                profile=self.profile,
                clock=self.beat_clock,
                capture_allowed=self.storage.capture_allowed,
                max_duration_s=commissioning.burst_max_duration_s,
            ),
        )
        from seeingmon.services.core.commissioning.dark import DarkHandler

        self.scheduler.register_handler(
            "dark",
            DarkHandler(
                library=self.dark_library,
                layout=self.storage.layout,
                profile=self.profile,
                clock=self.beat_clock,
                config=self.survey_config.dark,
                state=self.dark_state,
            ),
        )
        self.scheduler.register_handler(
            "replay",
            ReplayHandler(
                layout=self.storage.layout,
                profile=self.profile,
                station_id=self.station_id,
                clock=self.clock,
                fast_config=self.fast_config,
                store_config=self.config.section("store", StoreConfig),
                recordings_dir=self.recordings_dir,
                replays_dir=commissioning.replays_dir,
                beat=self.liveness.beat,
            ),
        )

    def _build_reporting(self) -> None:
        assert self.storage is not None
        from seeingmon.services.core.commissioning.dark import DarkLibraryReader
        from seeingmon.survey.dark import dark_due

        # The library of the survey analysis (`calibration_dir`), so that the health record and
        # the `dark_due` flag of the sky quality agree.
        library = self.dark_library
        survey_mode = self.profile.survey_mode.mode
        dark = self.survey_config.dark

        def dark_is_due(temperature_c: float | None, now_ns: int) -> bool:
            return dark_due(
                library,
                temperature_c,
                now_ns,
                mode=survey_mode,
                tolerance_c=dark.temperature_tolerance_c,
                max_age_days=dark.max_age_days,
            )

        self.dark_reader = DarkLibraryReader(
            library=library,
            profile=self.profile,
            clock=self.clock,
            config=dark,
            state=self.dark_state,
            temperature_c=lambda: self.scheduler.status().sensor_temperature_c,
        )
        self.rpc = CoreRpc(
            instance=self.instance,
            scheduler=self.scheduler,
            alignment=self.alignment,
            check_replay=self._check_replay,
            writer=self.events,
            dark_library=self.dark_reader.view,
            on_accepted=self._on_command_accepted,
            live_seeing=self.live_seeing,
            polaris=self.polaris,
        )
        self.health = HealthReporter(
            clock=self.clock,
            station_id=self.station_id,
            profile_id=self.profile.id,
            scheduler=self.scheduler,
            storage=self.storage,
            interval_s=self.settings.health_interval_s,
            acquire=self.remote,
            heater=self.heater,
            sqm=self.sqm,
            dark_due=dark_is_due,
            web_connected=lambda: self.rpc.web_connected,
            startup_grace_s=self.settings.run_record_wait_s,
        )
        self.pump = EventPump(self.remote, self.acquire_events) if self.remote else None
        self.notifier = self.parts.notifier or SystemdNotifier()
        self.tasks = PeriodicTasks(self.clock)
        self.tasks.add("health", self.settings.health_interval_s, self._write_health)
        if self.pump is not None:
            self.tasks.add("events", self.settings.events_interval_s, self._poll_events)
        self.tasks.add("run_record", 1.0, self._run_record_fallback)
        self.tasks.add("status", STATUS_INTERVAL_S, self._update_status)
        if self.nightly is not None:
            self.tasks.add("nightly", NIGHTLY_INTERVAL_S, self._flush_night_if_due, immediate=False)
        if self.notifier.watchdog_interval_s is not None:  # systemd asked for a heartbeat
            self.tasks.add("heartbeat", self.notifier.watchdog_interval_s, self._heartbeat)

    def _build_server(self) -> None:
        services = self.services
        rpc_service = self.rpc.rpc_service(
            max_connections=self.settings.max_rpc_connections,
            max_message_bytes=services.max_rpc_bytes,
        )
        stream_service = self.rpc.stream_service(StreamWindow(messages=8, bytes=64 * 1024 * 1024))
        polaris_service = self.rpc.polaris_service(StreamWindow(messages=16, bytes=8 * 1024 * 1024))
        self._rpc_service = rpc_service
        self.server = IpcServer(
            self.endpoint,
            self._key,
            {
                RPC_CHANNEL: rpc_service,
                ALIGNMENT_CHANNEL: stream_service,
                POLARIS_CHANNEL: polaris_service,
            },
            handshake_timeout_s=services.handshake_timeout_s,
            name="core",
        )

    def _acquire_connected(self) -> bool:
        return self.remote is not None and self.remote.connected

    def _check_replay(self, command: Any) -> str | None:
        assert self.storage is not None
        _, reason = clean_options(command.options)
        if reason:
            return reason
        path, reason = resolve_source(command.source, self.recordings_dir, self.storage.layout)
        return None if path is not None else reason

    # --- Callbacks -------------------------------------------------------------------------

    def _on_camera_opened(self, info: CameraInfo) -> None:
        self._camera = info
        self._write_run_record()
        self.tasks.trigger("health")  # the first record had no camera to report on

    def _on_command_accepted(self, command: Command, result: CommandResult) -> None:
        """The scheduler accepted a command that came through the RPC."""
        if isinstance(command, QueueDark) and result.task_id is not None:
            self.dark_state.queued(result.task_id, command, result.state)

    def _on_result(self, result: CommissionResult) -> None:
        _log.info("%s %d finished: %s", result.kind, result.task_id, result.summary)

    def _write_run_record(self) -> None:
        with self._lock:
            if self._run_record_written:
                return
            self._run_record_written = True
        assert self.storage is not None
        record = build_run_record(
            self.config,
            self.profile,
            self.clock,
            camera=self._camera,
            extra_versions=self.parts.extra_versions,
        )
        try:
            self.storage.store.write(record)
        except Exception:
            _log.exception("could not store the run record")

    def _run_record_fallback(self) -> None:
        waited_s = (self.clock.monotonic_ns() - self._started_mono) / NS_PER_S
        if not self._run_record_written and waited_s >= self.settings.run_record_wait_s:
            self._write_run_record()

    @staticmethod
    def _scheduler_key(status: SchedulerStatus) -> tuple[Any, ...]:
        """The parts of the status that a `health` record shows, and that must not wait a minute."""
        return (status.state, status.degraded, status.camera_component, status.fault.cause)

    def _write_health(self) -> None:
        assert self.storage is not None
        key = self._scheduler_key(self.scheduler.status())
        record = self.health.build()
        self.storage.store.write(record)
        self._health_key = key

    def _flush_night_if_due(self) -> None:
        if self.nightly is not None:
            self.nightly.flush_due()

    def _poll_events(self) -> None:
        if self.pump is not None:
            self.pump.poll()

    def status_text(self, status: SchedulerStatus | None = None) -> str:
        """The one line that `systemctl status` shows: the state, and what is wrong."""
        status = self.scheduler.status() if status is None else status
        parts = [status.state]
        if status.degraded:
            parts.append("the camera has failed")
        elif status.fault.failures:
            parts.append("the camera recovers")
        if self.remote is not None and not self.remote.connected:
            parts.append("acquire is not connected")
        if self.clock.status().synchronized is False:
            parts.append("the clock is not synchronized")
        return ", ".join(parts)

    def _update_status(self) -> None:
        """Send the status to systemd when it changes, and write a `health` record when it matters.

        The text for systemd has no counters. A change of the state, of `degraded`, or of the camera
        component writes a `health` record at once, because a lost camera must show within seconds
        and not at the next minute.
        """
        status = self.scheduler.status()
        self.notifier.status_changed(self.status_text(status))
        if self._health_key is not None and self._scheduler_key(status) != self._health_key:
            self.tasks.trigger("health")

    def scheduler_alive(self) -> bool:
        """Whether the scheduler makes progress. The watchdog of systemd depends on it."""
        thread = self._threads.get("core-scheduler")
        if thread is not None and not thread.is_alive():
            return False
        return self.liveness.alive()

    def _heartbeat(self) -> None:
        """Send `WATCHDOG=1`, but only while the scheduler lives. A stuck scheduler gets no
        heartbeat, and systemd restarts `core` when the interval passes."""
        if self._fatal or not self.scheduler_alive():
            return
        self.notifier.watchdog()

    # --- Running ---------------------------------------------------------------------------

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    @property
    def exit_reason(self) -> str:
        return self._fatal or self._exit_reason

    def request_stop(self, reason: str = "stopped") -> None:
        """Ask `core` to stop. Safe to call from a signal handler or any thread."""
        self._exit_reason = reason
        self._stop_event.set()

    def start(self) -> Endpoint:
        """Bind the address and start the threads. Returns the address that clients use."""
        assert self.storage is not None
        with self._lock:
            if self._started:
                raise RuntimeError("core already started")
            self._started = True
        self.events.emit("info", "core.started", "Core started.", {"instance": self.instance})
        self._rpc_service.start()
        self.bound_endpoint = self.server.start()
        if self.threads:
            self.alignment.start()
            self.polaris.start()
            self._spawn("core-scheduler", self._run_scheduler, fatal=True)
            self._spawn("core-housekeeping", self._run_housekeeping, fatal=True)
            if self.heater is not None:
                self._spawn("core-heater", lambda: self.heater.run(self._stop_event.is_set))
            if self.sqm is not None:
                self._spawn("core-sqm", self._run_sqm)
            if self.frames is not None:
                self._spawn("core-frames", self.frames.run)
            self._spawn("core-supervisor", self._run_supervisor, fatal=True)
        else:
            self._add_stepped_tasks()
        self.notifier.ready(f"listening at the core address, driver {self.driver.name}")
        _log.info("core %s listens at %s", self.instance, self.bound_endpoint)
        return self.bound_endpoint

    def _add_stepped_tasks(self) -> None:
        """The work that the threads do, as periodic tasks for a run that a test drives."""
        assert self.storage is not None
        storage = self.storage

        def housekeeping() -> None:
            storage.run_housekeeping(lambda: False, max_passes=1)

        self.tasks.add("housekeeping", storage.config.forwarder.poll_interval_s, housekeeping)
        if self.heater.status().enabled:
            self.heater.start()
            self.tasks.add("heater", 1.0, self.heater.step)
        if self.sqm is not None:
            sqm = self.sqm

            def poll_sqm() -> float:
                record = sqm.poll()
                if record is not None:
                    self._store_reading(record)
                return sqm.delay_s

            self.tasks.add("sqm", sqm.delay_s, poll_sqm)
        if self.frames is not None:
            frames = self.frames

            def write_frames() -> None:
                frames.drain()

            self.tasks.add("frames", 0.5, write_frames)

    def tick(self) -> int:
        """Do the periodic work that the supervisor thread does. Returns how many tasks ran."""
        with self.liveness.quiet():  # the supervisor proves nothing about the scheduler
            return self.tasks.run_due()

    def _spawn(self, name: str, target: Callable[[], None], *, fatal: bool = False) -> None:
        def main() -> None:
            try:
                target()
            except Exception as error:
                _log.exception("the thread %s died", name)
                self.events.emit(
                    "error",
                    "core.thread_died",
                    f"The thread {name} died: {type(error).__name__}.",
                    {"thread": name},
                )
                if fatal:
                    self._fatal = f"the thread {name} died: {type(error).__name__}"
                    self._stop_event.set()
            else:
                if fatal and not self._stop_event.is_set():
                    self._fatal = f"the thread {name} ended"
                    self._stop_event.set()

        thread = threading.Thread(target=main, name=name, daemon=True)
        self._threads[name] = thread
        thread.start()

    def _run_scheduler(self) -> None:
        self.liveness.bind_thread()  # the watchdog counts the progress of this thread
        self.scheduler.run(self._stop_event)

    def _run_housekeeping(self) -> None:
        assert self.storage is not None
        self.storage.run_housekeeping(self._stop_event.is_set)

    def _store_reading(self, record: ReferenceRecord) -> None:
        """Store a reading of the SQM-LE.

        A reading of the Influx source has the time stamp of its point. After a restart of `core`,
        the first poll can find the point that the store already holds, and the store refuses a
        second record with the same key. The reading is there then, so the refusal is no error.
        """
        assert self.storage is not None
        try:
            self.storage.store.write(record)
        except DuplicateRecordError:
            _log.debug("the store already holds the reading of this time stamp")

    def _run_sqm(self) -> None:
        assert self.sqm is not None

        def keep(record: ReferenceRecord) -> None:
            try:
                self._store_reading(record)
            except Exception:
                _log.exception("could not store a reference reading")

        self.sqm.run(self._stop_event.is_set, keep)

    def _run_supervisor(self) -> None:
        while not self._stop_event.is_set():
            self.tasks.run_due()
            wait_s = min(max(self.tasks.seconds_until_next(), 0.05), SUPERVISOR_SLICE_S)
            self.clock.sleep(wait_s)

    def run(self) -> int:
        """Start, wait for a stop request or a dead thread, and stop. Returns the exit code."""
        try:
            self.start()
            while not self._stop_event.wait(0.5):
                pass
        finally:
            self.stop()
        return EXIT_THREAD_DIED if self._fatal else 0

    # --- Stopping --------------------------------------------------------------------------

    def stop(self, reason: str | None = None) -> None:
        """Stop everything in the right order. Safe to call twice."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        if reason is not None:
            self._exit_reason = reason
        self._stop_event.set()
        timeout_s = self.settings.shutdown_timeout_s
        self.notifier.stopping()
        self.server.stop()  # no new commands, and the live view ends
        self._rpc_service.stop()
        scheduler_thread = self._threads.get("core-scheduler")
        if scheduler_thread is not None:
            scheduler_thread.join(timeout_s)  # the loop closes the scheduler and the camera
            if scheduler_thread.is_alive():
                _log.error("the scheduler did not stop within %g s", timeout_s)
        elif self._started:
            self.scheduler.close()
        self.alignment.stop()
        self.polaris.stop()
        self._close_night()
        for name in ("core-heater", "core-sqm", "core-supervisor"):
            thread = self._threads.get(name)
            if thread is not None:
                thread.join(timeout_s)
        if not self.threads and self.heater is not None:
            self.heater.stop()
        if self.sqm is not None:
            self.sqm.close()
        if self.heater is not None:
            self.heater.close()
        self._shutdown_survey()
        self._finish_frames(timeout_s)
        if self._started:
            self.events.emit(
                "info",
                "core.stopped",
                f"Core stopped: {self.exit_reason}.",
                {"instance": self.instance},
            )
        housekeeping = self._threads.get("core-housekeeping")
        if housekeeping is not None:
            housekeeping.join(timeout_s)
        self._release_storage()
        self.notifier.close()

    def _close_night(self) -> None:
        """Write the star summary of the open night. The scheduler has stopped by now."""
        if self.nightly is not None and self._started:
            try:
                self.nightly.flush()
            except Exception:
                _log.exception("could not close the night at shutdown")

    def _finish_frames(self, timeout_s: float) -> None:
        """Write the files that wait for the writer. The scheduler has stopped by now."""
        frames = getattr(self, "frames", None)
        if frames is None:
            return
        frames.finish()
        thread = self._threads.get("core-frames")
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                _log.error("the frame writer did not stop within %g s", timeout_s)
        elif self._started:
            try:
                frames.drain()
            except Exception:
                _log.exception("could not write the last survey frames at shutdown")

    def _shutdown_survey(self) -> None:
        analyzer = getattr(self, "survey", None)
        close = getattr(analyzer, "close", None)
        if callable(close):
            close()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _release_storage(self) -> None:
        storage = self.storage
        if storage is not None:
            self.storage = None
            storage.close()

    def _release_partial(self) -> None:
        """Free what a failed build opened."""
        self._release_storage()
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


class _NoPointing:
    """A pointing provider that never has a solution. The scheduler then asks for survey frames."""

    def polaris_position(self, t_utc_ns: int, mode: str) -> tuple[float, float] | None:
        return None
