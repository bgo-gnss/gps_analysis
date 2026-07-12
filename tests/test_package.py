"""Package-level scaffold tests: version and the stable module map."""

import importlib

import gps_analysis

EXPECTED_MODULES = [
    "baseline",
    "deformation",
    "fitting",
    "models",
    "noise",
    "preprocess",
    "transient",
    "velocity",
]


def test_version() -> None:
    assert gps_analysis.__version__ == "0.1.0"


def test_module_map_is_importable_and_documented() -> None:
    for name in EXPECTED_MODULES:
        module = importlib.import_module(f"gps_analysis.{name}")
        assert module.__doc__, f"gps_analysis.{name} must document its planned surface"
