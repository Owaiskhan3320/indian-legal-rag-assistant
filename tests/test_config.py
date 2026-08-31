from pathlib import Path

import pytest

from legal_ai.config import PROJECT_ROOT, Settings


DATA_PATH_ENV_VARS = (
    "TRAIN_DATASET_PATH",
    "DEV_DATASET_PATH",
    "TEST_DATASET_PATH",
    "SAMPLE_DATASET_PATH",
    "REFERENCE_LAW_SOURCE_DIR",
)


def clear_data_path_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in DATA_PATH_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)


def test_dataset_defaults_are_relative_to_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_data_path_environment(monkeypatch)
    settings = Settings(_env_file=None)
    expected_paths = {
        "train_dataset_path": "data/case_law/train.csv",
        "dev_dataset_path": "data/case_law/dev.csv",
        "test_dataset_path": "data/case_law/test.csv",
        "sample_dataset_path": "data/case_law/sample_10k.csv",
        "reference_law_source_dir": "data/reference_law",
    }

    for field, expected in expected_paths.items():
        configured_path = getattr(settings, field)
        assert configured_path == expected
        assert settings.resolve_path(configured_path) == PROJECT_ROOT / expected


def test_dataset_path_can_be_overridden_by_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clear_data_path_environment(monkeypatch)
    custom_path = tmp_path / "case-law.csv"
    monkeypatch.setenv("TRAIN_DATASET_PATH", str(custom_path))

    settings = Settings(_env_file=None)

    assert settings.train_dataset_path == str(custom_path)
    assert settings.resolve_path(settings.train_dataset_path) == custom_path
