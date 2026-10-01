"""The `[store]` configuration section: segment files, retention, and the sink forwarder.

The defaults live in `config/default.d/store.toml`, and every key has the same default in the
models below. Read the section through the configuration:

    from seeingmon.config import load_config
    from seeingmon.store.config import StoreConfig

    store_config = load_config().section("store", StoreConfig)

Override a key in `local/config.toml` under `[store]`, or with an environment variable such as
`SEEINGMON_STORE__RETENTION__MIN_FREE_GB=2`. The sink endpoints and credentials are a separate
section, `[sinks.<name>]`, which `seeingmon.sinks.config` describes.

Sizes are in decimal gigabytes (`GB` bytes), as disk vendors count them.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from seeingmon.config import SectionModel

GB = 1_000_000_000


class SegmentsConfig(SectionModel):
    """How the segment files of per-frame metrics are written."""

    segment_s: int = Field(
        default=600,
        ge=1,
        le=86_400,
        description="The length of one segment, in seconds of frame time.",
    )
    fsync_interval_s: float = Field(
        default=60.0,
        ge=0,
        description=(
            "How often the writer forces the open segment to disk, in seconds. The writer always "
            "flushes after each call, so a crash of the process loses nothing. This interval "
            "bounds the loss at a power cut. 0 forces the disk after every call."
        ),
    )
    idle_close_s: float = Field(
        default=120.0,
        gt=0,
        description=(
            "How long a segment may sit idle before `tick` closes it, so that a stopped "
            "capture does not leave an open file."
        ),
    )


class RetentionConfig(SectionModel):
    """The policies of the hourly retention task. See `seeingmon.store.retention`."""

    interval_s: float = Field(
        default=3600.0, gt=0, description="How often the services lane runs the task, in seconds."
    )
    quota_fraction: float = Field(
        default=0.25,
        gt=0,
        le=1,
        description="The share of the data partition that the data directory may use.",
    )
    min_free_gb: float = Field(
        default=1.0,
        ge=0,
        description="Below this free space, raw capture stops and retention frees space.",
    )
    resume_margin_gb: float = Field(
        default=0.5,
        ge=0,
        description="Capture resumes when the free space rises this far above `min_free_gb`.",
    )
    protect_recent_s: float = Field(
        default=900.0,
        ge=0,
        description="Retention never deletes a file that changed within this many seconds.",
    )
    metrics_days: float = Field(
        default=7.0, gt=0, description="How long per-frame metric segments stay, in days."
    )
    metrics_max_gb: float = Field(
        default=2.0, gt=0, description="The size cap of the per-frame metric segments."
    )
    previews_days: float = Field(
        default=7.0, gt=0, description="How long preview images stay, in days."
    )
    survey_full_days: float = Field(
        default=7.0, gt=0, description="How long every survey frame stays, in days."
    )
    survey_thinned_days: float = Field(
        default=60.0,
        ge=0,
        description=(
            "After the full days, how long one survey frame for each night stays, in days. "
            "A night keeps the frame nearest to the middle of the night."
        ),
    )
    bursts_max_gb: float = Field(
        default=2.0,
        ge=0,
        description="The size quota of the raw bursts. Pinned bursts do not count and stay.",
    )
    night_boundary_utc_hour: float = Field(
        default=12.0,
        ge=0,
        lt=24,
        description=(
            "The UTC hour at which one observing night ends and the next begins. Use the hour "
            "of local noon at your site, so that a night never splits."
        ),
    )
    temp_max_age_s: float = Field(
        default=3600.0,
        gt=0,
        description="A temporary file older than this is a leftover of a crash, and retention "
        "deletes it.",
    )


class ForwarderConfig(SectionModel):
    """How the sink forwarder batches rows and retries."""

    batch_rows: int = Field(
        default=5000,
        ge=1,
        description="The most rows in one batch. A sink may ask for a smaller batch.",
    )
    max_batches_per_pass: int = Field(
        default=20,
        ge=1,
        description="The most batches that one pass sends to one sink, so that a backfill of "
        "one sink does not starve the others.",
    )
    poll_interval_s: float = Field(
        default=1.0, gt=0, description="How long the loop sleeps when no sink has work, in seconds."
    )
    backoff_initial_s: float = Field(
        default=5.0, gt=0, description="The wait after the first failure of a sink, in seconds."
    )
    backoff_factor: float = Field(
        default=2.0, ge=1, description="Each further failure multiplies the wait by this factor."
    )
    backoff_max_s: float = Field(
        default=600.0, gt=0, description="The longest wait between two attempts, in seconds."
    )
    jitter: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description="Each wait varies by up to this fraction, up or down, at random.",
    )
    backlog_flag_rows: int = Field(
        default=20_000,
        ge=1,
        description="A sink with more unsent rows than this raises the `sink_backlog` flag.",
    )

    @model_validator(mode="after")
    def _check_backoff(self) -> Self:
        if self.backoff_max_s < self.backoff_initial_s:
            raise ValueError("backoff_max_s must not be smaller than backoff_initial_s")
        return self


class StoreConfig(SectionModel):
    """The `[store]` section."""

    busy_timeout_ms: int = Field(
        default=5000,
        ge=0,
        description="How long a writer waits for another process to release the database lock.",
    )
    reader_connections: int = Field(
        default=4, ge=1, le=64, description="The size of the pool of read-only connections."
    )
    segments: SegmentsConfig = Field(default_factory=SegmentsConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    forwarder: ForwarderConfig = Field(default_factory=ForwarderConfig)
