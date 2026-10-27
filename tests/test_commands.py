"""Unimplemented research commands must fail instead of producing fake artifacts."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("script", "ticket"),
    [("make_data.py", "T1.4"), ("final_eval.py", "T3.4"), ("reproduce.py", "T7.2")],
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
