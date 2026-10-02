"""Keep generated shell configuration aligned with its Python source."""

import subprocess
from pathlib import Path

from examples.common.launcher_constants import LAUNCHER_DEFAULTS, render_shell_constants

ROOT = Path(__file__).resolve().parents[1]


def test_generated_launcher_constants_are_current_and_roundtrip_through_bash():
    """Reject stale generated settings and verify their actual shell values."""
    path = ROOT / "scripts/della/runtime_constants.sh"
    assert path.read_text() == render_shell_constants()
    names = list(LAUNCHER_DEFAULTS)
    result = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            'source "$1"; shift; for name; do printf "%s\\0" "${!name}"; done',
            "bash",
            str(path),
            *names,
        ],
        check=True,
        capture_output=True,
    )
    values = result.stdout.decode().split("\0")[:-1]
    assert values == [str(LAUNCHER_DEFAULTS[name]) for name in names]
    assert dict(zip(names, values, strict=True))["FOREST_CONTEXT_TOKENS"] == "262144"
