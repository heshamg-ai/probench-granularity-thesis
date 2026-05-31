# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

from __future__ import annotations
import traceback
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import copy
import gc
import json
import numpy as np
import warnings

from probench.utils.measurement_utils import (
    measure_adaptive, calibrate_memory_runtime
)

from probench.utils.io_utils import save_json, load_json
from probench.prep.param_schema import ParamGrid

#warnings.filterwarnings("ignore")
class PerfTestFailedError(RuntimeError):
    """Raised when one or more micro-scenarios fail during perf testing.
    Signals that correctness tests were insufficient. The LLM optimized code is broken
    on edge cases not covered by unit tests.
    """
    pass

class PerfScenario:
    def __init__(
        self,
        *,
        version: str,                  # "baseline", "reference", "variant"
        pr_number: int,
        scenario_id: str,         
        param_space: ParamGrid,            # full grid of micro-scenarios
        measure: Dict[str, Any],
        results_root: Path,
        seed: int = 123,
    ):
        assert version in {"baseline", "reference", "llm"}

        self.version = version
        self.pr_number = pr_number
        self.scenario_id = scenario_id
        self.param_space = param_space
        self.measure = measure
        self.results_root = results_root
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.micro_scenario_params: Optional[Dict[str, Any]] = None
        self.args: Dict[str, Any] = {}
        self._args_template: Optional[Dict[str, Any]] = None

    @abstractmethod
    def setup(self):
        """
        ALWAYS called. Build and initialize scenario-specific data.
        Do parsing, data construction, etc.
        NOT measured.
        """
        pass

    def warmup(self):
        """
        OPTIONAL. Pre-run work (e.g., cache priming).
        """
        pass

    @abstractmethod
    def run(self):
        """
        The operation under test. This IS measured.
        Must not redo setup work.
        """
        pass

    def micro_scenario_key(self, params: Dict[str, Any]) -> str:
        return json.dumps(params, sort_keys=True, separators=(",", ":"))

    def _reset_rng(self) -> None:
        """Reset rng to initial state. Call before start of setup() for determinism."""
        self.rng = np.random.default_rng(self.seed)

    def capture_run_state(self):
        self._args_template = copy.deepcopy(self.args)

    def needs_warmup(self) -> bool:
        return False

    def restore_run_state(self):
        if self._args_template is None:
            raise RuntimeError("restore_run_state() called before capture_run_state()")
        self.args = copy.deepcopy(self._args_template)
        if self.needs_warmup():
            self.warmup()

    def _prepare_args(self):
        self.args = {}
        gc.collect()
        self._reset_rng()
        self.setup()
        self.warmup()
        self.capture_run_state()

    def calibrate_all_perf_tests(self):
        self.results_root.mkdir(parents=True, exist_ok=True)
 
        target_sample_time_s = float(self.measure.get("target_sample_time_s"))
        safety_factor        = float(self.measure.get("safety_factor"))
        min_runtime_invocations   = int(self.measure.get("min_runtime_invocations"))
        max_runtime_invocations   = int(self.measure.get("max_runtime_invocations"))
        calibrate_warmup_runs     = int(self.measure.get("calibrate_warmup_runs"))
        min_probe_time_s     = float(self.measure.get("min_probe_time_s"))
        max_probe_runs = int(self.measure.get("max_probe_runs"))
        min_probe_runs = int(self.measure.get("min_probe_runs"))

        micro_scenarios = {}
 
        for micro_scenario_params in self.param_space.pairs():
            self.args   = {}
            self.params = copy.deepcopy(micro_scenario_params)
 
            cal = calibrate_memory_runtime(
                function=lambda: self.run(),
                measure_cfg=self.measure,
                pre_hook=self._prepare_args,
                restore_hook=self.restore_run_state
            )
 
            key = self.micro_scenario_key(self.params)
            micro_scenarios[key] = {
                "micro_scenario_params": copy.deepcopy(self.params),
                "runtime_invocations":   cal["runtime_invocations"],
                "tm_sample_interval_s":  cal["tm_sample_interval_s"],
                "calibration":           cal,
                "calibration_source":    "baseline",
            }
 
        payload = {
            "scenario_id":    self.scenario_id,
            "version":        self.version,
            "pr":             self.pr_number,
            "seed":           self.seed,
            "measure_config": {
                "target_sample_time_s": target_sample_time_s,
                "safety_factor":        safety_factor,
                "min_runtime_invocations":    min_runtime_invocations,
                "max_runtime_invocations":    max_runtime_invocations,
                "min_probe_time_s":     min_probe_time_s,
                "calibrate_warmup_runs":       calibrate_warmup_runs,
                "min_probe_runs":    min_probe_runs,
                "max_probe_runs":    max_probe_runs
            },
            "micro_scenarios": micro_scenarios,
        }
 
        save_json(self.results_root / "calibration.json", payload)
        return payload
 
 
    # ---- main entry used by the runner inside the container ------------------
    def run_all_perf_tests(self):
        """
        Run all micro-scenarios (ParamGrid points) for this macro scenario
        using shared calibration.json.
 
        Writes <results_root>/performance_results.json and returns the payload.
        """
        self.results_root.mkdir(parents=True, exist_ok=True)
 
        calibration_path = self.results_root / "calibration.json"
        if not calibration_path.exists():
            raise FileNotFoundError(f"Missing calibration file: {calibration_path}")
 
        calibration = load_json(str(calibration_path))
        if not calibration:
            raise ValueError(f"Calibration file is empty or invalid: {calibration_path}")
 
        records:    List[Dict[str, Any]] = []
        last_stats: Optional[Dict[str, Any]] = None
 
        try:
            for micro_scenario_params in self.param_space.pairs():
                self.args = {}
                gc.collect()
 
                print("run: " + str(micro_scenario_params))
 
                self.params  = copy.deepcopy(micro_scenario_params)
                key          = self.micro_scenario_key(self.params)
                cal_entry    = (calibration.get("micro_scenarios") or {}).get(key)
 
                if not cal_entry:
                    raise KeyError(f"Missing calibration entry for micro-scenario key={key}")
 
                runtime_invocations        = int(cal_entry["runtime_invocations"])
                tm_sample_interval_s = float(cal_entry["tm_sample_interval_s"])
 
                if runtime_invocations < 1:
                    raise ValueError(f"Invalid runtime_invocations={runtime_invocations} for key={key}")
 
                try:
                    stats  = measure_adaptive(
                        function=lambda: self.run(),
                        measure_cfg=self.measure,
                        runtime_invocations=runtime_invocations,
                        tm_sample_interval_s=tm_sample_interval_s,
                        pre_measure_hook=self._prepare_args,
                        per_iter_prepare_hook=self.restore_run_state,
                    )
                    last_stats = stats
                    records.append({
                        "micro_scenario_params": copy.deepcopy(self.params),
                        "measurement_control": {
                            "runtime_invocations":  runtime_invocations,
                            "tm_sample_interval_s": tm_sample_interval_s,
                            "calibration_source":   cal_entry.get("calibration_source")
                        },
                        "metrics": stats,
                        "status":  "ok",
                    })
 
                except Exception as e:
                    print(f"[WARNING] Failed micro_scenario {micro_scenario_params}: {e}")
                    traceback.print_exc()
                    raise PerfTestFailedError(
                        f"micro_scenario {micro_scenario_params} failed: {e}"
                    ) from e
 
        finally:
            results: Dict[str, Any] = {
                "scenario_id":        self.scenario_id,
                "version":            self.version,
                "pr":                 self.pr_number,
                "seed":               self.seed,
                "results_dir":        str(self.results_root),
                "measure_config":     self.measure,
                "platform":           (last_stats.get("platform") if last_stats else None),
                "measurement_records": records,
            }
 
            out_path = self.results_root / "performance_results.json"
            print(f"[INFO] writing performance results to {out_path}")
            save_json(str(out_path), results)
            print("[INFO] performance results written")
 
        return results