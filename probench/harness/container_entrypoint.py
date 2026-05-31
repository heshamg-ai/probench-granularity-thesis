# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import argparse
import json, subprocess
import traceback, shutil
from pathlib import Path
from typing import Any, Dict, Optional, List

from probench.scenarios.scenario_base import PerfTestFailedError
from probench.utils.harness_utils import id_to_scenario_class
from probench.prep.param_schema import ParamGrid
from probench.harness.evaluation.correctness_test_runner import CorrectnessRunner
from probench.utils.io_utils import save_json
from probench.harness.patch.patch_applier import apply_patch
from probench.harness.patch.patch_sanitizer import ApplyStatus

def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

def run_container_entrypoint(argv: Optional[List[str]] = None):
    """
    Entrypoint for running a single scenario inside the container.

    Called by:
      python -m probench.harness.container_entrypoint --config /app_results/_inner.json
    """
    parser = argparse.ArgumentParser(description="Run scenario inside the container.")
    parser.add_argument("--config", required=True, help="Path to inner JSON config")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Scenario config file not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))

    scenario_id: str = str(config.get("scenario_id")).strip()
    version: str = str(config.get("version") or "").strip()  # baseline | reference | llm-variant
    pr_number: Optional[int] = config.get("pr_number")
    seed: int = int(config.get("seed"))
    measure: Dict[str, Any] = config.get("measure")
    perf_mode = str(config.get("perf_mode")).strip()

    results_root = Path(config.get("results_root") or "/app_results").resolve()
    params_json = config.get("params")
    core_tests: list = config.get("core_tests")

    lib_metadata = config.get("lib_metadata")
    repo_root = lib_metadata.get("repo_root")
    mounted_calibration_path = config.get("mounted_calibration_path")
    crashed = False
    run_correctness = (version == "llm")

    try:

        ScenarioCls = id_to_scenario_class(scenario_id)
        params = ParamGrid.from_json(params_json)

        scenario = ScenarioCls(
            version=version,
            pr_number=pr_number,
            scenario_id=scenario_id,
            param_space=params,
            measure=measure,
            results_root=results_root,
            seed=seed,
        )

        # CALIBRATION MODE
        if perf_mode == "calibrate":
            print("[INFO] calibration mode -> generating shared calibration")
            scenario.calibrate_all_perf_tests()
            return 0

        if perf_mode != "measure":
            raise ValueError(f"Unknown perf_mode: {perf_mode}")

        if mounted_calibration_path:
            src = Path(mounted_calibration_path)
            if not src.exists():
                raise FileNotFoundError(f"Mounted calibration file not found: {src}")

            dst = results_root / "calibration.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            print(f"[INFO] copied shared calibration to {dst}")
        else:
            raise ValueError("measure mode requires mounted_calibration_path")

        if version == "llm":
            llm_root = config.get("llm_inference_root") 
            if not llm_root:
                raise ValueError("version='llm' but config missing llm_inference_root")

            llm_root = Path(llm_root).resolve()
            completion_path = llm_root / "completion.txt"
            meta_path = llm_root / "meta.json"

            if not completion_path.exists():
                raise FileNotFoundError(f"LLM completion.txt not found: {completion_path}")
            if not meta_path.exists():
                raise FileNotFoundError(f"LLM meta.json not found: {meta_path}")

            completion_text = completion_path.read_text(encoding="utf-8", errors="replace")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            print(f"[INFO] Applying LLM to repo_root={repo_root}")

            repo_dir = Path(repo_root)

            # (optional) capture the starting commit for traceability#todo:delete
            _, head_before, _ = _run(["git", "rev-parse", "HEAD"], cwd=repo_dir)#todo?
            head_before = head_before.strip()

            r = apply_patch(repo_dir=Path(repo_root), llm_output=completion_text, dry_run=False)

            # Patch result for debugging / reporting
            save_json(results_root / "llm_patch_apply.json", r.to_dict())

            rc, diff_text, diff_err = _run(["git", "diff", "--patch"], cwd=repo_dir)
            if rc != 0:
                diff_text = ""
                print(f"[WARN] git diff failed: {diff_err.strip()}")

            (results_root / "llm_patch.diff").write_text(diff_text, encoding="utf-8")

            if r.apply_status == ApplyStatus.FAILED:
                print(f"[INFO] LLM patch failed to apply: {r.error}")
                return 0

            if r.apply_status == ApplyStatus.PARTIALLY_APPLIED:
                print(f"[WARN] LLM patch partially applied: {r.warnings_count} warning(s)")

            if not diff_text.strip():
                print("[WARN] patch applied but git diff is empty — no real changes made")
                return 0

            print("[INFO] LLM patch applied.")

        if run_correctness:
            # 1) Correctness
            ok = True
            summary = None
            try:
                runner = CorrectnessRunner(
                    repo_root=Path(repo_root),
                    output_dir=results_root,
                    core_tests=core_tests,
                )
                summary = runner.run()
            except Exception as e:
                print(f"[WARN] CorrectnessRunner failed for {scenario_id}: {e}")
                traceback.print_exc()

            is_correct = bool(summary and summary.get("is_correct") is True)
            if not is_correct:
                print("[INFO] correctness failed -> skipping performance evaluation")
                return 0

            print("[INFO] correctness passed -> running performance evaluation")

        # 2) Micro benchmarks 
        scenario.run_all_perf_tests()
        return 0
    
    except PerfTestFailedError as e:
        print(f"[INFO] perf failed: {e}")

        if run_correctness:
            correctness_report_path = results_root / "correctness_report_summary.json"
            try:
                report = json.loads(correctness_report_path.read_text(encoding="utf-8"))
            except Exception:
                report = {}

            report["is_correct"] = False
            report["correctness_source"] = "perf_test_failed"
            report["correctness_message"] = (
                "correctness passed unit tests but failed during performance sweep. "
                "Code is broken on edge cases"
            )
            report["perf_failure"] = {
                "type": "PerfTestFailedError",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }

            correctness_report_path.write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )
            print("[INFO] correctness report updated with perf failure.")
        else:
            print("[INFO] correctness skipped for non-llm run; no correctness report written.")

        return 0
    
    except Exception as e:
        # unexpected crash
        crashed = True
        err = traceback.format_exc() 
        print(f"[ERROR] container crashed: {e}")
        traceback.print_exc()
        return 1

    finally:
        if not crashed:
            (results_root / "DONE").write_text("ok", encoding="utf-8")
            print("[INFO] DONE marker written.")
        else:
            (results_root / "CRASHED").write_text(err, encoding="utf-8")
            print("[INFO] CRASHED marker written.")


if __name__ == "__main__":
    raise SystemExit(run_container_entrypoint())
