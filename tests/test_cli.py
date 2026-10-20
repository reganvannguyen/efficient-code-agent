import subprocess
import sys


def test_module_entry_point_runs_without_loading_a_model() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "harness"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == (
        "Harness initialized; model execution is not implemented yet.\n"
    )
