"""Leaf-dependency guard (plan risk R6) — CI enforcement.

gps_analysis must stay a leaf math package: an import of any higher-tier
gpslibrary package would give the Tier-1 dependency graph a cycle. This test
fails on any such import anywhere under src/, statically (AST scan) and at
runtime (sys.modules after importing everything).
"""

import ast
import importlib
import sys
from pathlib import Path

from test_package import EXPECTED_MODULES

FORBIDDEN = {
    "geo_dataread",
    "gps_parser",
    "gps_plot",
    "gps_api",
    "receivers",
    "tostools",
}

SRC = Path(__file__).resolve().parents[1] / "src"


def _imported_packages(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0:
                found.add(node.module.split(".")[0])
    return found


def test_src_has_no_forbidden_imports() -> None:
    offenders: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        hits = _imported_packages(path) & FORBIDDEN
        if hits:
            offenders[str(path.relative_to(SRC))] = hits
    assert not offenders, f"leaf rule violated (plan R6): {offenders}"


def test_importing_the_package_loads_no_forbidden_packages() -> None:
    for name in EXPECTED_MODULES:
        importlib.import_module(f"gps_analysis.{name}")
    loaded = {module_name.split(".")[0] for module_name in sys.modules}
    assert not (loaded & FORBIDDEN)
