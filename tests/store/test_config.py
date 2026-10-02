"""The `[store]` section: the defaults file and the models agree with the architecture."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from seeingmon.config import ConfigError, load_config
from seeingmon.records.seeing import FrameRecord
from seeingmon.records.survey import StarListRecord
from seeingmon.store.config import GB, ForwarderConfig, RetentionConfig, StoreConfig


@pytest.fixture
def defaults_file(repo_root: Path) -> Path:
    return repo_root / "config" / "default.d" / "store.toml"


def test_the_defaults_file_and_the_models_declare_the_same_values(defaults_file: Path) -> None:
    with defaults_file.open("rb") as handle:
        table = tomllib.load(handle)
    assert set(table) == {"store"}
    assert StoreConfig.model_validate(table["store"]) == StoreConfig()


def test_the_defaults_file_names_every_key(defaults_file: Path) -> None:
    with defaults_file.open("rb") as handle:
        table = tomllib.load(handle)["store"]
    for name, model in (
        ("segments", StoreConfig().segments),
        ("retention", StoreConfig().retention),
        ("forwarder", StoreConfig().forwarder),
    ):
        assert set(table[name]) == set(type(model).model_fields), name
    top = set(StoreConfig.model_fields) - {"segments", "retention", "forwarder"}
    assert top <= set(table)


def test_the_configuration_loads_the_defaults(tmp_path: Path) -> None:
    config = load_config(local_file=tmp_path / "none.toml", env={})
    assert config.section("store", StoreConfig) == StoreConfig()


def test_a_local_file_and_the_environment_override_a_default(tmp_path: Path) -> None:
    local = tmp_path / "config.toml"
    local.write_text("[store.retention]\nmetrics_days = 3\nmin_free_gb = 2.5\n", encoding="utf-8")
    config = load_config(local_file=local, env={"SEEINGMON_STORE__RETENTION__MIN_FREE_GB": "4"})
    retention = config.section("store", StoreConfig).retention
    assert retention.metrics_days == 3
    assert retention.min_free_gb == 4


def test_a_misspelled_key_fails_loudly(tmp_path: Path) -> None:
    local = tmp_path / "config.toml"
    local.write_text("[store.retention]\nmetric_days = 3\n", encoding="utf-8")
    config = load_config(local_file=local, env={})
    with pytest.raises(ConfigError, match="metric_days"):
        config.section("store", StoreConfig)


def test_the_retention_defaults_follow_the_architecture() -> None:
    retention = RetentionConfig()
    assert retention.quota_fraction == 0.25  # 25% of the data partition
    assert retention.metrics_days == FrameRecord.retention_days == 7
    assert retention.metrics_max_gb * GB == 2 * GB
    assert retention.bursts_max_gb * GB == 2 * GB
    assert retention.previews_days == 7
    assert retention.previews_max_gb * GB == 1 * GB  # a week of previews takes 0.1 to 0.25 GB
    assert retention.survey_max_gb * GB == 4 * GB  # and a week of survey frames 1.8 to 3.5 GB
    assert (retention.survey_full_days, retention.survey_thinned_days) == (7, 60)
    assert retention.min_free_gb * GB == 1 * GB  # below 1 GB, raw capture stops
    assert StarListRecord.retention_days == 365  # star lists follow the record declaration


def test_the_longest_backoff_may_not_be_shorter_than_the_first() -> None:
    with pytest.raises(ValidationError, match="backoff_max_s"):
        ForwarderConfig(backoff_initial_s=30, backoff_max_s=10)


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (RetentionConfig, {"quota_fraction": 0}),
        (RetentionConfig, {"quota_fraction": 1.5}),
        (RetentionConfig, {"night_boundary_utc_hour": 24}),
        (RetentionConfig, {"survey_max_gb": 0}),
        (RetentionConfig, {"previews_max_gb": -1}),
        (ForwarderConfig, {"jitter": 2}),
        (ForwarderConfig, {"batch_rows": 0}),
    ],
)
def test_out_of_range_values_are_rejected(model: type, values: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        model(**values)
