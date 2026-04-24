from alpha_lab.utils.paths import (
    CONFIGS_DIR,
    DATA_DIR,
    PRIVATE_DIR,
    PROJECT_ROOT,
    RAW_DIR,
)


def test_project_root_has_pyproject():
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_data_subdirs_under_data_dir():
    assert RAW_DIR.parent == DATA_DIR
    assert PRIVATE_DIR.parent == DATA_DIR


def test_configs_dir_exists():
    assert CONFIGS_DIR.is_dir()
    assert (CONFIGS_DIR / "default.yaml").exists()
