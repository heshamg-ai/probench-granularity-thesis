import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Set

from probench.utils.io_utils import make_json_safe, save_json


class CorrectnessRunner:
    """
    Executes a predefined set of pytest nodeids and returns correctness metrics.

    Assumptions
    ----------
    - `core_tests` are valid pytest nodeids relative to repo_root.
    - Baseline is expected to pass all tests (runner does NOT compare versions).

    Outputs (under output_dir)
    ---------------------------
    - correctness_report.json (pytest-json-report output)
    - correctness_report_summary.json (compact structured result)

    Status logic
    ------------
    - exit_code == 0  -> status = "passed"
    - exit_code == 1  -> status = "test_failed"
    - else            -> status = "run_failed"
    """

    def __init__(self, repo_root: Path, output_dir: Path, core_tests: List[str]):
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.core_tests = sorted({str(t).strip() for t in (core_tests or []) if str(t).strip()})
        if not self.core_tests:
            raise RuntimeError("[CorrectnessRunner] core_tests is empty.")

    def run(self) -> Dict[str, Any]:
        report_path = self.output_dir / "correctness_report.json"

        # Convert repo-relative nodeids to absolute paths
        pytest_args: List[str] = []
        for nodeid in self.core_tests:
            file_part, _, test_part = nodeid.partition("::")
            full_path = (self.repo_root / file_part).resolve()
            pytest_args.append(f"{full_path}::{test_part}" if test_part else str(full_path))

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            f"--rootdir={self.repo_root}",
            "-o", "addopts=",
            "-o", "filterwarnings=",
            "-W", "ignore::DeprecationWarning",
            "--disable-warnings",
            *pytest_args,
            "--json-report",
            f"--json-report-file={report_path}",
        ]

        print(f"[CorrectnessRunner] Running pytest with {len(self.core_tests)} tests")

        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            cwd=self.repo_root,
        )

        # Determine status
        if result.returncode == 0:
            status = "passed"
        elif result.returncode == 1:
            status = "test_failed"
        else:
            status = "run_failed"
        
        print(f"[CorrectnessRunner] Pytest finished with exit code {result.returncode} ({status})")

        passed_nodeids: Set[str] = set()
        failed_nodeids: Set[str] = set()

        # Parse json report if available
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                for t in report.get("tests", []):
                    nodeid = t.get("nodeid")
                    outcome = t.get("outcome")

                    if not nodeid:
                        continue

                    if outcome == "passed":
                        passed_nodeids.add(nodeid)
                    elif outcome in {"failed", "error"}:
                        failed_nodeids.add(nodeid)
            except Exception:
                # If parsing fails, rely on exit code only
                print("[CorrectnessRunner] Warning: Could not parse JSON report.")

        total_selected = len(self.core_tests)
        failed_count = len(failed_nodeids)
        passed_count = len(passed_nodeids)

        is_run_ok = status != "run_failed"
        is_correct = is_run_ok and (failed_count == 0)

        executed = passed_count + failed_count
        correctness_rate = (
            passed_count / executed
            if is_run_ok and executed > 0
            else None
        )

        summary: Dict[str, Any] = {
            "status": status,
            "exit_code": result.returncode,
            "selected_nodeids_count": total_selected,
            "selected_nodeids": self.core_tests,
            "passed_nodeids": sorted(passed_nodeids),
            "failed_nodeids": sorted(failed_nodeids),
            "passed_count": passed_count,
            "failed_count": failed_count,
            "correctness_rate": correctness_rate,
            "is_correct": is_correct,
        }

        save_json(self.output_dir / "correctness_report_summary.json", make_json_safe(summary))
        return summary