"""Unimplemented research commands must fail instead of producing fake artifacts."""

import copy
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("script", "ticket"),
    [("final_eval.py", "T3.4"), ("reproduce.py", "T7.2")],
)
def test_pending_commands_fail_without_creating_artifacts(
    script: str, ticket: str, tmp_path: Path
) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert ticket in result.stderr
    assert not result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field", ["data_design", "normalization", "units", "threshold"])
def test_data_command_rejects_protocol_drift(field: str) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/make_data.py"))
    config = yaml.safe_load((ROOT / "config/data.yaml").read_text())
    protocol = yaml.safe_load((ROOT / "config/eval_protocol.yaml").read_text())
    namespace["validate_design"](config, protocol)
    changed = copy.deepcopy(config)
    if field == "data_design":
        changed["data_design"]["id_splits"]["train"]["count"] -= 1
    elif field == "normalization":
        changed["normalization"]["fit_split"] = "validation"
    elif field == "units":
        changed["units"]["rate"] += 0.01
    else:
        changed["validation"]["max_invalid_fraction"] = 1.0
    with pytest.raises(ValueError):
        namespace["validate_design"](changed, protocol)


@pytest.mark.parametrize("failed_check", ["fixed", "adaptive"])
def test_numerical_gate_fails_before_any_dataset_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_check: str
) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/make_data.py"))
    calls = []

    def fixed(argv: list[str]) -> int:
        calls.append("fixed")
        return int(failed_check == "fixed")

    def adaptive(argv: list[str]) -> int:
        calls.append("adaptive")
        return int(failed_check == "adaptive")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("generation reached after failed numerical validation")

    fixed_module = ModuleType("validate_data_numerics")
    adaptive_module = ModuleType("validate_adaptive_labels")
    fixed_module.main = fixed  # type: ignore[attr-defined]
    adaptive_module.main = adaptive  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "validate_data_numerics", fixed_module)
    monkeypatch.setitem(sys.modules, "validate_adaptive_labels", adaptive_module)
    monkeypatch.setitem(namespace["main"].__globals__, "generate_dataset", forbidden)
    monkeypatch.setattr(sys, "argv", ["make_data.py", "--output", str(tmp_path / "forbidden")])
    with pytest.raises(RuntimeError, match="validation failed; no labels generated"):
        namespace["main"]()
    assert calls == (["fixed"] if failed_check == "fixed" else ["fixed", "adaptive"])
    assert list(tmp_path.iterdir()) == []
