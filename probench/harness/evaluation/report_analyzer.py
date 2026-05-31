from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from probench.utils import global_config as GC
from probench.utils.io_utils import load_json, save_json


# ── Workload-level status constants ───────────────────────────────────────────
WORKLOAD_UNSTABLE            = "unstable"
WORKLOAD_INSUFFICIENT_SIGNAL = "insufficient_signal"
WORKLOAD_IMPROVED            = "improved"
WORKLOAD_REGRESSED           = "regressed"

# ── Metric-level decision constants ───────────────────────────────────────────
METRIC_IMPROVED       = "improved"
METRIC_REGRESSED      = "regressed"
METRIC_NO_SIGNAL      = "no_signal"
METRIC_CONFLICTING    = "conflicting"
METRIC_NOT_MEASURABLE = "not_measurable"

# ── Comparison-level status constants ─────────────────────────────────────────
EVALUATED      = "evaluated"
PATCH_FAILED   = "patch_failed"
TESTS_FAILED   = "tests_failed"
NOT_MEASURABLE = "not_measurable"

DEFAULT_CI_Z         = 1.96
_MIN_NOISE_FLOOR_PCT = 1e-6  # epsilon floor: prevents division by zero when noise is truly 0


def _load_analysis_config() -> Dict[str, Any]:
    analysis_cfg  = GC.analysis if isinstance(GC.analysis, dict) else {}
    decisions_cfg = analysis_cfg.get("decisions")

    if not isinstance(decisions_cfg, dict):
        raise KeyError("config missing required 'analysis.decisions' block")

    def _get_config_value(node: Dict[str, Any], path: List[str]) -> Any:
        current = node
        for key in path:
            if not isinstance(current, dict):
                raise KeyError(
                    f"config path '{'.'.join(path)}' not found (missing at '{key}')"
                )
            current = current.get(key)
            if current is None:
                raise KeyError(
                    f"config path '{'.'.join(path)}' is missing or null"
                )
        return current

    enabled_metrics: List[Dict[str, Any]] = []
    for metric_name, metric_cfg in (_get_config_value(analysis_cfg, ["metrics"]) or {}).items():
        if not metric_cfg.get("enabled", True):
            continue
        enabled_metrics.append({
            "name":      metric_name,
            "raw_key":   _get_config_value(metric_cfg, ["raw_key"]),
            "direction": _get_config_value(metric_cfg, ["direction"]),
            "unit":      metric_cfg.get("unit", ""),
        })

    return {
        "min_required_samples":          int(_get_config_value(decisions_cfg,   ["workload", "min_required_samples"])),
        "max_total_system_noise_pct":    float(_get_config_value(decisions_cfg, ["workload", "max_total_system_noise_pct"])),
        "snr_threshold":                 float(_get_config_value(decisions_cfg, ["workload", "snr_threshold"])),
        "min_practical_effect_pct":      float(_get_config_value(decisions_cfg, ["workload", "min_practical_effect_pct"])),
        "min_quality_coverage_pct":      float(_get_config_value(decisions_cfg, ["metric",   "min_quality_coverage_pct"])),
        "min_directional_dominance_pct": float(_get_config_value(decisions_cfg, ["metric",   "min_directional_dominance_pct"])),
        "min_conflict_presence_pct":     float(_get_config_value(decisions_cfg, ["metric",   "min_conflict_presence_pct"])),
        "rciw_cleaning_enabled":         bool(_get_config_value(analysis_cfg,   ["rciw_cleaning", "enabled"])),
        "rciw_cleaning_k":               float(_get_config_value(analysis_cfg,  ["rciw_cleaning", "k"])),
        "rciw_cleaning_min_keep":        int(_get_config_value(analysis_cfg,    ["rciw_cleaning", "min_keep"])),
        "metrics": enabled_metrics,
    }

def _to_finite_float_list(values: Any) -> List[float]:
    if not isinstance(values, list):
        return []
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def _median(values: List[float]) -> Optional[float]:
    finite_values = [v for v in values if math.isfinite(v)]
    return float(statistics.median(finite_values)) if finite_values else None


def _arithmetic_mean(values: List[float]) -> Optional[float]:
    finite_values = [v for v in values if math.isfinite(v)]
    return float(sum(finite_values) / len(finite_values)) if finite_values else None


def _sample_std(values: List[float]) -> Optional[float]:
    finite_values = [v for v in values if math.isfinite(v)]
    return float(statistics.stdev(finite_values)) if len(finite_values) >= 2 else None


def _harmonic_mean(values: List[float]) -> Optional[float]:
    """
    Harmonic mean of positive values.
    Correct aggregator for rate ratios (baseline_time / optimized_time).
    Equivalent to ratio of arithmetic means of times (Jacob & Mudge 1995).
    One regression workload (ratio < 1.0) appropriately reduces the result.
    """
    finite_positive_values = [
        float(v) for v in values
        if v is not None and math.isfinite(v) and v > 0
    ]
    if not finite_positive_values:
        return None
    return float(
        len(finite_positive_values) / sum(1.0 / v for v in finite_positive_values)
    )


def _mad(values: List[float]) -> Optional[float]:
    finite_values = [v for v in values if math.isfinite(v)]
    if not finite_values:
        return None
    center = statistics.median(finite_values)
    return float(statistics.median([abs(v - center) for v in finite_values]))


def _mad_trim(values: List[float], k: float, min_keep: int) -> List[float]:
    """
    MAD-based outlier trimming. Removes values more than k*MAD_scaled from median.
    Falls back to original list if trimming would leave fewer than min_keep values.
    """
    finite_values = [v for v in values if math.isfinite(v)]
    if len(finite_values) < min_keep:
        return finite_values
    center    = statistics.median(finite_values)
    mad_value = _mad(finite_values)
    if mad_value is None:
        return finite_values
    if mad_value <= 0:
        trimmed = [v for v in finite_values if v == center]
        return trimmed if len(trimmed) >= min_keep else finite_values
    mad_scaled = 1.4826 * mad_value
    trimmed = [
        v for v in finite_values
        if (center - k * mad_scaled) <= v <= (center + k * mad_scaled)
    ]
    return trimmed if len(trimmed) >= min_keep else finite_values


def _clean_outliers(values: List[float], config: Dict[str, Any]) -> List[float]:
    """Filter non-finite values, then MAD-trim if cleaning is enabled."""
    finite_values = [v for v in values if math.isfinite(v)]
    if not config["rciw_cleaning_enabled"]:
        return finite_values
    return _mad_trim(
        finite_values,
        k=config["rciw_cleaning_k"],
        min_keep=config["rciw_cleaning_min_keep"],
    )


def _rciw_pct(values: List[float], z: float = DEFAULT_CI_Z) -> Optional[float]:
    """Relative CI width: CI width / mean, as a percentage."""
    number_of_values = len(values)
    if number_of_values < 2:
        return None
    mean_value = _arithmetic_mean(values)
    std_value  = _sample_std(values)
    if mean_value is None or std_value is None or mean_value == 0:
        return None
    standard_error = std_value / math.sqrt(number_of_values)
    return float(2.0 * z * standard_error / abs(mean_value) * 100.0)


def _rciw_within_pct_with_cleaning(
    series: List[List[float]],
    config: Dict[str, Any],
    z: float = DEFAULT_CI_Z,
) -> Optional[float]:
    """
    Within-run RCIW with cleaning at the profiling-sample level.
    For each container run: MAD-trim the profiling samples, compute RCIW.
    Returns the MAX across runs — worst-case within-container noise.
    Cleaning target: single bad profiling samples within one run
    (JIT warmup spikes, measurement artifacts within a single execution).
    """
    per_run_rciws = [
        rciw for rciw in (
            _rciw_pct(_clean_outliers(run_profiling_samples, config), z=z)
            for run_profiling_samples in series
        )
        if rciw is not None and math.isfinite(rciw)
    ]
    return max(per_run_rciws) if per_run_rciws else None


def _rciw_between_pct_with_cleaning(
    run_medians: List[float],
    config: Dict[str, Any],
    z: float = DEFAULT_CI_Z,
) -> Optional[float]:
    """
    Between-run RCIW with cleaning at the container-run level.
    MAD-trims the list of per-run medians, then computes RCIW.
    Cleaning target: anomalous container runs where the entire run was bad
    (bad container startup, OS scheduler interference across a full run).
    """
    run_medians_cleaned = _clean_outliers(run_medians, config)
    return _rciw_pct(run_medians_cleaned, z=z)


def _compute_change_abs(
    baseline_value: Optional[float],
    target_value: Optional[float],
    direction: str,
) -> Optional[float]:
    if baseline_value is None or target_value is None:
        return None
    if not math.isfinite(baseline_value) or not math.isfinite(target_value):
        return None
    if direction == "lower_better":
        return float(baseline_value - target_value)
    if direction == "higher_better":
        return float(target_value - baseline_value)
    raise ValueError(f"Unsupported direction: {direction}")


def _compute_change_pct(
    baseline_value: Optional[float],
    target_value: Optional[float],
    direction: str,
) -> Optional[float]:
    """
    Percentage improvement: (baseline - optimized) / baseline * 100.
    Used ONLY for SNR gating. Not aggregated or reported at experiment level.
    """
    if baseline_value is None or target_value is None or baseline_value == 0:
        return None
    change_abs = _compute_change_abs(baseline_value, target_value, direction)
    if change_abs is None:
        return None
    return float(change_abs / baseline_value * 100.0)

def _check_patch_results(target_root: Path, n_repeats: int) -> List[Dict[str, Any]]:
    patch_results = []
    for repeat_index in range(n_repeats):
        path = Path(target_root) / f"run_{repeat_index:03d}" / "llm_patch_apply.json"
        if not path.exists():
            patch_results.append({
                "repeat_index": repeat_index,
                "present":      False,
                "ok":           False,
                "apply_status": None,
                "error":        "missing llm_patch_apply.json",
                "warnings":     [],
            })
            continue
        patch_data = load_json(path)
        patch_results.append({
            "repeat_index": repeat_index,
            "present":      True,
            "ok":           bool(patch_data.get("ok", False)),
            "apply_status": patch_data.get("apply_status"),
            "error":        patch_data.get("error"),
            "warnings":     list(patch_data.get("warnings") or []),
        })
    return patch_results


def _check_correctness_results(target_root: Path, n_repeats: int) -> List[Dict[str, Any]]:
    correctness_results = []
    for repeat_index in range(n_repeats):
        path = Path(target_root) / f"run_{repeat_index:03d}" / "correctness_report_summary.json"
        if not path.exists():
            correctness_results.append({
                "repeat_index": repeat_index,
                "is_correct":   False,
                "status":       "missing",
                "exit_code":    None,
                "failed_tests": [],
            })
            continue
        correctness_data = load_json(path)
        failed_tests = list(
            correctness_data.get(
                "failed_nodeids",
                correctness_data.get("failed_tests", []),
            ) or []
        )
        is_correct = (
            bool(correctness_data["is_correct"]) if "is_correct" in correctness_data
            else (
                correctness_data.get("status") not in {"run_failed", "missing", None}
                and not failed_tests
            )
        )
        correctness_results.append({
            "repeat_index": repeat_index,
            "is_correct":   is_correct,
            "status":       correctness_data.get("status", "missing"),
            "exit_code":    correctness_data.get("exit_code"),
            "failed_tests": failed_tests,
        })
    return correctness_results


def _load_repeated_performance_results(
    root: Path,
    n_repeats: int,
) -> List[Dict[str, Any]]:
    all_repeated_results = []
    for repeat_index in range(n_repeats):
        path = Path(root) / f"run_{repeat_index:03d}" / "performance_results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing performance_results.json: {path}")
        raw_data = json.loads(path.read_text(encoding="utf-8"))
        all_repeated_results.append({
            json.dumps(record["micro_scenario_params"], sort_keys=True): record
            for record in raw_data.get("measurement_records", [])
            if record.get("status") == "ok"
        })
    return all_repeated_results


def _extract_workload_metric_series(
    workload_records: List[Dict[str, Any]],
    raw_key: str,
) -> List[List[float]]:
    return [
        _to_finite_float_list(record.get("metrics", {}).get(raw_key))
        for record in workload_records
    ]


# ══════════════════════════════════════════════════════════════════════════════
# per-workload per-metric evaluation (7-gate pipeline)
#
# Gate 1  sufficient samples after filtering           → EMPTY_OR_INCOMPLETE
# Gate 2  per-run medians computable                   → EMPTY_OR_INCOMPLETE
# Gate 3  change_pct computable (for SNR)              → EMPTY_OR_INCOMPLETE
# Gate 4  within-RCIW <= noise ceiling (both sides)   → unstable
# Gate 5  SNR >= snr_threshold                         → insufficient_signal
# Gate 6  |change_pct| >= min practical effect         → insufficient_signal
# Gate 7  change_pct and change_abs both zero          → insufficient_signal
#         → improved / regressed
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_metric_for_workload(
    metric_config: Dict[str, Any],
    baseline_records: List[Dict[str, Any]],
    target_records: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:

    direction = metric_config["direction"]
    raw_key   = metric_config["raw_key"]

    _empty_workload_result = {
        "status":                     "EMPTY_OR_INCOMPLETE",
        "is_stable":                  False,
        "has_reliable_effect":        False,
        "baseline_representative":    None,
        "target_representative":      None,
        "ratio":                      None,
        "change_abs":                 None,
        "change_pct":                 None,
        "signal_to_noise_ratio":      None,
        "baseline_rep_is_stable":   None,
        "target_rep_is_stable":     None,
        "rciw_within_baseline_pct":   None,
        "rciw_within_target_pct":     None,
        "rciw_between_baseline_pct":  None,
        "rciw_between_target_pct":    None,
        "max_total_system_noise_pct": None,
    }

    # ── Gate 1: sufficient samples ─────────────────────────────────────────
    min_required_samples   = config["min_required_samples"]
    baseline_metric_series = [
        v for v in _extract_workload_metric_series(baseline_records, raw_key)
        if len(v) >= min_required_samples
    ]
    target_metric_series = [
        v for v in _extract_workload_metric_series(target_records, raw_key)
        if len(v) >= min_required_samples
    ]
    if not baseline_metric_series or not target_metric_series:
        return {**_empty_workload_result, "reason": "insufficient samples after filtering"}

    # ── Gate 2: per-run medians computable ────────────────────────────────
    baseline_run_medians = [
        v for v in (_median(run_values) for run_values in baseline_metric_series)
        if v is not None
    ]
    target_run_medians = [
        v for v in (_median(run_values) for run_values in target_metric_series)
        if v is not None
    ]
    if not baseline_run_medians or not target_run_medians:
        return {**_empty_workload_result, "reason": "missing execution medians"}

    baseline_representative = _median(baseline_run_medians)
    target_representative   = _median(target_run_medians)

    ratio: Optional[float] = None
    if (
        baseline_representative is not None
        and target_representative is not None
        and math.isfinite(baseline_representative)
        and math.isfinite(target_representative)
        and target_representative > 0
    ):
        ratio = float(baseline_representative / target_representative)

    change_abs = _compute_change_abs(baseline_representative, target_representative, direction)
    change_pct = _compute_change_pct(baseline_representative, target_representative, direction)

    # ── Gate 3: change_pct computable ─────────────────────────────────────
    if change_pct is None:
        return {
            **_empty_workload_result,
            "reason":                  "change_pct uncomputable (baseline zero or missing)",
            "baseline_representative": baseline_representative,
            "target_representative":   target_representative,
            "ratio":                   ratio,
            "change_abs":              change_abs,
        }

    rciw_within_baseline_pct  = _rciw_within_pct_with_cleaning(baseline_metric_series, config)
    rciw_within_target_pct    = _rciw_within_pct_with_cleaning(target_metric_series,   config)
    rciw_between_baseline_pct = _rciw_between_pct_with_cleaning(baseline_run_medians,  config)
    rciw_between_target_pct   = _rciw_between_pct_with_cleaning(target_run_medians,    config)

    # ── Gate 4: within-noise reliability ──────────────────────────────────
    noise_ceiling = config["max_total_system_noise_pct"]

    baseline_is_noisy = (
        rciw_within_baseline_pct is not None
        and rciw_within_baseline_pct > noise_ceiling
    )
    target_is_noisy = (
        rciw_within_target_pct is not None
        and rciw_within_target_pct > noise_ceiling
    )

    if baseline_is_noisy or target_is_noisy:
        failed_containers = []
        if baseline_is_noisy:
            failed_containers.append(
                f"baseline: within_noise {rciw_within_baseline_pct:.4f}% "
                f"> ceiling {noise_ceiling:.1f}%"
            )
        if target_is_noisy:
            failed_containers.append(
                f"target: within_noise {rciw_within_target_pct:.4f}% "
                f"> ceiling {noise_ceiling:.1f}%"
            )
        return {
            "status":                     WORKLOAD_UNSTABLE,
            "is_stable":                  False,
            "has_reliable_effect":        False,
            "reason":                     "container representative unreliable — " + "; ".join(failed_containers),
            "baseline_representative":    baseline_representative,
            "target_representative":      target_representative,
            "ratio":                      ratio,
            "change_abs":                 change_abs,
            "change_pct":                 change_pct,
            "signal_to_noise_ratio":      None,
            "baseline_rep_is_stable":   not baseline_is_noisy,
            "target_rep_is_stable":     not target_is_noisy,
            "rciw_within_baseline_pct":   rciw_within_baseline_pct,
            "rciw_within_target_pct":     rciw_within_target_pct,
            "rciw_between_baseline_pct":  rciw_between_baseline_pct,
            "rciw_between_target_pct":    rciw_between_target_pct,
            "max_total_system_noise_pct": None,
        }

    noise_component_values = [
        v for v in [
            rciw_within_baseline_pct,
            rciw_within_target_pct,
            rciw_between_baseline_pct,
            rciw_between_target_pct,
        ]
        if v is not None and math.isfinite(v)
    ]
    max_total_system_noise_pct = max(noise_component_values) if noise_component_values else 0.0

    # ── SNR — epsilon floor prevents division by zero ──────────────────────
    effective_noise       = max_total_system_noise_pct if max_total_system_noise_pct > 0.0 else _MIN_NOISE_FLOOR_PCT
    signal_to_noise_ratio = float(abs(change_pct) / effective_noise)

    _stable_workload_base = {
        "is_stable":                  True,
        "baseline_representative":    baseline_representative,
        "target_representative":      target_representative,
        "ratio":                      ratio,
        "change_abs":                 change_abs,
        "change_pct":                 change_pct,
        "signal_to_noise_ratio":      signal_to_noise_ratio,
        "baseline_rep_is_stable":   not baseline_is_noisy,
        "target_rep_is_stable":     not target_is_noisy,
        "rciw_within_baseline_pct":   rciw_within_baseline_pct,
        "rciw_within_target_pct":     rciw_within_target_pct,
        "rciw_between_baseline_pct":  rciw_between_baseline_pct,
        "rciw_between_target_pct":    rciw_between_target_pct,
        "max_total_system_noise_pct": max_total_system_noise_pct,
    }

    # ── Gate 5: SNR threshold ──────────────────────────────────────────────
    snr_threshold = config["snr_threshold"]
    if signal_to_noise_ratio < snr_threshold:
        return {
            **_stable_workload_base,
            "status":              WORKLOAD_INSUFFICIENT_SIGNAL,
            "has_reliable_effect": False,
            "reason":              (
                f"SNR {signal_to_noise_ratio:.4f} < {snr_threshold:.1f} "
                "(effect indistinguishable from noise)"
            ),
        }

    # ── Gate 6: practical effect ───────────────────────────────────────────
    min_practical_effect_pct = config["min_practical_effect_pct"]
    if abs(change_pct) < min_practical_effect_pct:
        return {
            **_stable_workload_base,
            "status":              WORKLOAD_INSUFFICIENT_SIGNAL,
            "has_reliable_effect": False,
            "reason":              (
                f"effect too small: change_pct {change_pct:.4f}% "
                f"< {min_practical_effect_pct:.1f}%"
            ),
        }

    # ── Gate 7: both zero ─────────────────────────────────────────────────
    if change_pct == 0 and (change_abs is None or change_abs == 0.0):
        return {
            **_stable_workload_base,
            "status":              WORKLOAD_INSUFFICIENT_SIGNAL,
            "has_reliable_effect": False,
            "reason":              "change_pct and change_abs are both zero — no effect detected",
        }

    # ── Direction decision ─────────────────────────────────────────────────
    workload_status = WORKLOAD_IMPROVED if change_pct > 0 else WORKLOAD_REGRESSED
    workload_reason = (
        "zero measured noise — signal is unambiguous"
        if max_total_system_noise_pct == 0.0
        else "stable containers with reliable directional effect"
    )
    return {
        **_stable_workload_base,
        "status":              workload_status,
        "has_reliable_effect": True,
        "reason":              workload_reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# evaluate all workloads
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_all_workload_cases(
    baseline_repeated_results: List[Dict[str, Any]],
    target_repeated_results: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not baseline_repeated_results or not target_repeated_results:
        return []

    common_workload_keys = set.intersection(
        *[set(repeated.keys()) for repeated in baseline_repeated_results],
        *[set(repeated.keys()) for repeated in target_repeated_results],
    )

    all_workload_results = []
    for workload_key in sorted(common_workload_keys):
        baseline_workload_records = [r[workload_key] for r in baseline_repeated_results]
        target_workload_records   = [r[workload_key] for r in target_repeated_results]

        all_workload_results.append({
            "workload_params": baseline_repeated_results[0][workload_key]["micro_scenario_params"],
            "workload_key":    workload_key,
            "metric_results": {
                metric_config["name"]: _evaluate_metric_for_workload(
                    metric_config=metric_config,
                    baseline_records=baseline_workload_records,
                    target_records=target_workload_records,
                    config=config,
                )
                for metric_config in config["metrics"]
            },
        })
    return all_workload_results


def _aggregate_metric_decision(
    metric_config: Dict[str, Any],
    all_workload_results: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    metric_name    = metric_config["name"]
    metric_results = [
        workload["metric_results"][metric_name]
        for workload in all_workload_results
        if metric_name in workload["metric_results"]
    ]

    empty_metric_decision = {
        "decision":                       METRIC_NOT_MEASURABLE,
        "is_measurable":                  False,
        "has_valid_summary":              False,
        "n_total_workloads":              0,
        "n_stable":                       0,
        "n_reliable":                     0,
        "n_unstable":                     0,
        "n_improved":                     0,
        "n_regressed":                    0,
        "n_insufficient_signal":          0,
        "stable_workload_coverage_pct":   0.0,
        "reliable_effect_coverage_pct":   0.0,
        "harmonic_mean_ratio_reliable":   None,
        "mean_ratio_reliable":            None,
        "std_ratio_reliable":             None,
        "harmonic_mean_ratio_stable":     None,
        "mean_ratio_stable":              None,
        "std_ratio_stable":               None,
        "harmonic_mean_snr_reliable":     None,
        "harmonic_mean_snr_stable":       None,
        "workload_snr_table":             [],
        "unit":                           metric_config["unit"],
        "reason":                         "no workload results",
    }
    if not metric_results:
        return empty_metric_decision

    n_total_workloads     = len(metric_results)
    stable_results        = [r for r in metric_results if r.get("is_stable", False)]
    n_stable              = len(stable_results)
    n_unstable            = sum(1 for r in metric_results if r.get("status") == WORKLOAD_UNSTABLE)
    n_improved            = sum(1 for r in stable_results if r.get("status") == WORKLOAD_IMPROVED)
    n_regressed           = sum(1 for r in stable_results if r.get("status") == WORKLOAD_REGRESSED)
    n_insufficient_signal = sum(1 for r in stable_results if r.get("status") == WORKLOAD_INSUFFICIENT_SIGNAL)

    stable_workload_coverage_pct = (n_stable / n_total_workloads * 100.0) if n_total_workloads else 0.0
    reliable_effect_coverage_pct = ((n_improved + n_regressed) / n_total_workloads * 100.0) if n_total_workloads else 0.0

    # ── Ratio: reliable pool (improved + regressed only) ───────────────────
    reliable_workload_ratio_values = [
        r["ratio"] for r in stable_results
        if r.get("has_reliable_effect")
        and r.get("ratio") is not None
        and math.isfinite(r["ratio"])
        and r["ratio"] > 0
    ]
    harmonic_mean_ratio_reliable = _harmonic_mean(reliable_workload_ratio_values)
    mean_ratio_reliable          = _arithmetic_mean(reliable_workload_ratio_values)
    std_ratio_reliable           = _sample_std(reliable_workload_ratio_values)

    # ── Ratio: stable pool (all stable workloads) ──────────────────────────
    stable_workload_ratio_values = [
        r["ratio"] for r in stable_results
        if r.get("ratio") is not None
        and math.isfinite(r["ratio"])
        and r["ratio"] > 0
    ]
    harmonic_mean_ratio_stable = _harmonic_mean(stable_workload_ratio_values)
    mean_ratio_stable          = _arithmetic_mean(stable_workload_ratio_values)
    std_ratio_stable           = _sample_std(stable_workload_ratio_values)

    # ── SNR: reliable pool — diagnostic only ──────────────────────────────
    reliable_workload_snr_values = [
        r["signal_to_noise_ratio"]
        for r in stable_results
        if r.get("has_reliable_effect")
        and r.get("signal_to_noise_ratio") is not None
        and math.isfinite(r["signal_to_noise_ratio"])
        and r["signal_to_noise_ratio"] > 0
    ]
    harmonic_mean_snr_reliable = _harmonic_mean(reliable_workload_snr_values)

    # ── SNR: stable pool (all stable workloads) ────────────────────────────
    stable_workload_snr_values = [
        r["signal_to_noise_ratio"]
        for r in stable_results
        if r.get("signal_to_noise_ratio") is not None
        and math.isfinite(r["signal_to_noise_ratio"])
        and r["signal_to_noise_ratio"] > 0
    ]
    harmonic_mean_snr_stable = _harmonic_mean(stable_workload_snr_values)

    # ── Per-workload SNR table ─────────────────────────────────────────────
    workload_snr_table = [
        {
            "workload_key":               workload["workload_key"],
            "workload_params":            workload["workload_params"],
            "status":                     workload["metric_results"][metric_name].get("status"),
            "signal_to_noise_ratio":      workload["metric_results"][metric_name].get("signal_to_noise_ratio"),
            "ratio":                      workload["metric_results"][metric_name].get("ratio"),
            "max_total_system_noise_pct": workload["metric_results"][metric_name].get("max_total_system_noise_pct"),
            "has_reliable_effect":        workload["metric_results"][metric_name].get("has_reliable_effect"),
            "is_stable":                  workload["metric_results"][metric_name].get("is_stable"),
        }
        for workload in all_workload_results
        if metric_name in workload["metric_results"]
        and workload["metric_results"][metric_name].get("is_stable")
        and "workload_key" in workload
    ]

    # ── Direction decision ─────────────────────────────────────────────────
    min_directional_dominance_pct = config["min_directional_dominance_pct"]
    min_conflict_presence_pct     = config["min_conflict_presence_pct"]

    improved_pct_of_stable  = (n_improved  / n_stable * 100.0) if n_stable else 0.0
    regressed_pct_of_stable = (n_regressed / n_stable * 100.0) if n_stable else 0.0

    if stable_workload_coverage_pct < config["min_quality_coverage_pct"]:
        decision = METRIC_NOT_MEASURABLE
        reason   = (
            f"stable coverage {stable_workload_coverage_pct:.1f}% "
            f"< required {config['min_quality_coverage_pct']:.1f}%"
        )
    elif n_improved == 0 and n_regressed == 0:
        decision = METRIC_NO_SIGNAL
        reason   = "no stable workload passed the SNR gate"
    elif improved_pct_of_stable >= min_directional_dominance_pct and regressed_pct_of_stable == 0:
        decision = METRIC_IMPROVED
        reason   = (
            f"improvements dominate: {improved_pct_of_stable:.1f}% of stable workloads, "
            f"harmonic_snr_reliable={harmonic_mean_snr_reliable}"
        )
    elif regressed_pct_of_stable >= min_directional_dominance_pct and improved_pct_of_stable == 0:
        decision = METRIC_REGRESSED
        reason   = (
            f"regressions dominate: {regressed_pct_of_stable:.1f}% of stable workloads, "
            f"harmonic_snr_reliable={harmonic_mean_snr_reliable}"
        )
    elif improved_pct_of_stable >= min_conflict_presence_pct and regressed_pct_of_stable >= min_conflict_presence_pct:
        decision = METRIC_CONFLICTING
        reason   = (
            f"meaningful effects in both directions: "
            f"{improved_pct_of_stable:.1f}% improved, {regressed_pct_of_stable:.1f}% regressed"
        )
    else:
        decision = METRIC_NO_SIGNAL
        reason   = "directional evidence exists but is not decisive"

    return {
        "decision":                       decision,
        "is_measurable":                  decision != METRIC_NOT_MEASURABLE,
        "has_valid_stable_ratio":         n_stable > 0 and harmonic_mean_ratio_stable is not None,
        "n_total_workloads":              n_total_workloads,
        "n_stable":                       n_stable,
        "n_reliable":                     n_improved + n_regressed,
        "n_unstable":                     n_unstable,
        "n_improved":                     n_improved,
        "n_regressed":                    n_regressed,
        "n_insufficient_signal":          n_insufficient_signal,
        "stable_workload_coverage_pct":   round(stable_workload_coverage_pct, 4),
        "reliable_effect_coverage_pct":   round(reliable_effect_coverage_pct, 4),
        "harmonic_mean_ratio_reliable":   None if harmonic_mean_ratio_reliable is None else round(harmonic_mean_ratio_reliable, 4),
        "mean_ratio_reliable":            None if mean_ratio_reliable           is None else round(mean_ratio_reliable,           4),
        "std_ratio_reliable":             None if std_ratio_reliable            is None else round(std_ratio_reliable,            4),
        "harmonic_mean_ratio_stable":     None if harmonic_mean_ratio_stable is None else round(harmonic_mean_ratio_stable, 4),
        "mean_ratio_stable":              None if mean_ratio_stable           is None else round(mean_ratio_stable,           4),
        "std_ratio_stable":               None if std_ratio_stable            is None else round(std_ratio_stable,            4),
        "harmonic_mean_snr_reliable":     None if harmonic_mean_snr_reliable is None else round(harmonic_mean_snr_reliable, 4),
        "harmonic_mean_snr_stable":       None if harmonic_mean_snr_stable   is None else round(harmonic_mean_snr_stable,   4),
        "workload_snr_table":             workload_snr_table,
        "unit":                           metric_config["unit"],
        "reason":                         reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Level 1 public — analyze_comparison
# ══════════════════════════════════════════════════════════════════════════════

def analyze_comparison(
    baseline_root: Path,
    target_root: Path,
    output_dir: Path,
    *,
    label_target: str,
    is_llm_run: bool = True,
    n_repeats: int = 3,
) -> Dict[str, Any]:
    """
    Compare two versions for one scenario and return a report dict.

    is_llm_run=True  → baseline vs LLM patch  (runs patch + correctness gates)
    is_llm_run=False → baseline vs reference   (skips those gates)

    Saves output to output_dir/report.json and returns the report dict.
    """
    baseline_root = Path(baseline_root)
    target_root   = Path(target_root)
    output_dir    = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_analysis_config()

    report: Dict[str, Any] = {
        "label_target":       label_target,
        "is_llm_run":         is_llm_run,
        "n_repeats":          n_repeats,
        "patch_ok":           True,
        "tests_ok":           True,
        "measurement_status": EVALUATED,
        "metric_decisions":   {},
        "workloads":          [],
    }

    if is_llm_run:
        patch_results        = _check_patch_results(target_root, n_repeats)
        failed_patch_repeats = [r["repeat_index"] for r in patch_results if not r["ok"]]
        report["gates"]      = {
            "patch": [
                {
                    "repeat_index": r["repeat_index"],
                    "ok":           r["ok"],
                    "apply_status": r["apply_status"],
                    "error":        r["error"],
                }
                for r in patch_results
            ]
        }
        report["patch_ok"] = len(failed_patch_repeats) == 0
        if failed_patch_repeats:
            report["measurement_status"] = PATCH_FAILED
            report["reason"] = f"Patch failed in repeated executions: {failed_patch_repeats}"
            save_json(str(output_dir / "report.json"), report)
            return report

        correctness_results = _check_correctness_results(target_root, n_repeats)
        all_failed_tests    = sorted({t for r in correctness_results for t in r["failed_tests"]})
        all_correct         = all(r["is_correct"] for r in correctness_results)
        report["gates"]["correctness"] = {
            "is_correct":     all_correct,
            "failed_repeats": [r["repeat_index"] for r in correctness_results if not r["is_correct"]],
            "failed_tests":   all_failed_tests,
        }
        report["tests_ok"] = all_correct
        if not all_correct:
            report["measurement_status"] = TESTS_FAILED
            report["reason"] = (
                f"Correctness failed in repeated executions "
                f"{report['gates']['correctness']['failed_repeats']}: {all_failed_tests}"
            )
            save_json(str(output_dir / "report.json"), report)
            return report
    else:
        report["patch_ok"] = True
        report["tests_ok"] = True

    baseline_repeated_results = _load_repeated_performance_results(baseline_root, n_repeats)
    target_repeated_results   = _load_repeated_performance_results(target_root,   n_repeats)

    all_workload_results       = _evaluate_all_workload_cases(
        baseline_repeated_results, target_repeated_results, config
    )
    report["workloads"]        = all_workload_results
    report["metric_decisions"] = {
        metric_config["name"]: _aggregate_metric_decision(
            metric_config, all_workload_results, config
        )
        for metric_config in config["metrics"]
    }

    any_metric_measurable = any(
        metric_decision.get("decision") != METRIC_NOT_MEASURABLE
        for metric_decision in report["metric_decisions"].values()
    )
    report["measurement_status"] = EVALUATED if any_metric_measurable else NOT_MEASURABLE

    for metric_decision in report["metric_decisions"].values():
        relative_ready = bool(
            report["patch_ok"]
            and report["tests_ok"]
            and metric_decision.get("is_measurable", False)
            and metric_decision.get("has_valid_stable_ratio", False)
        )
        metric_decision["relative_ready"]  = relative_ready
        metric_decision["relative_reason"] = (
            "ok" if relative_ready
            else "patch_failed" if not report["patch_ok"]
            else "tests_failed" if not report["tests_ok"]
            else "not_measurable" if not metric_decision.get("is_measurable", False)
            else "no_valid_stable_ratio"
        )

    report["relative_ready"]  = any(
        metric_decision.get("relative_ready", False)
        for metric_decision in report["metric_decisions"].values()
    )
    report["relative_reason"] = "ok" if report["relative_ready"] else "not_measurable"
    save_json(str(output_dir / "report.json"), report)
    return report

def _round_or_none(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    return None if value is None else round(value, ndigits)

def compute_rel_scores(
    llm_report: Dict[str, Any],
    anchor_report: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Compute per-metric RelScore for one scenario.

    RelScore = human/reference target representative / LLM target representative

    Workload-level rel_scores are stored as structured records:
      {
        workload_key,
        rel_score,
        llm_target_representative,
        reference_target_representative,
        ...
      }
    """

    patch_ok = bool(llm_report.get("patch_ok", True))
    tests_ok = bool(llm_report.get("tests_ok", True))

    if not patch_ok or not tests_ok:
        return None

    llm_workloads_by_key = {
        workload["workload_key"]: workload
        for workload in (llm_report.get("workloads") or [])
        if "workload_key" in workload
    }

    reference_workloads_by_key = {
        workload["workload_key"]: workload
        for workload in (anchor_report.get("workloads") or [])
        if "workload_key" in workload
    }

    common_workload_keys = sorted(
        set(llm_workloads_by_key) & set(reference_workloads_by_key)
    )

    llm_metric_decisions = llm_report.get("metric_decisions") or {}
    reference_metric_decisions = anchor_report.get("metric_decisions") or {}

    all_metric_names = sorted(
        set(llm_metric_decisions) | set(reference_metric_decisions)
    )

    per_metric_results: Dict[str, Any] = {}
    any_rel_score_computable = False

    def extract_rel_scores(entries: List[Dict[str, Any]]) -> List[float]:
        return [entry["rel_score"] for entry in entries]

    def rel_score_signal(values: List[float]) -> str:
        if not values:
            return "no_signal"

        n_llm_better_or_equal = sum(1 for value in values if value >= 1.0)
        n_llm_worse = sum(1 for value in values if value < 1.0)

        if n_llm_better_or_equal > 0 and n_llm_worse == 0:
            return "llm_consistently_better_or_equal"

        if n_llm_worse > 0 and n_llm_better_or_equal == 0:
            return "llm_consistently_worse"

        return "conflicting"

    for metric_name in all_metric_names:
        llm_metric_decision = llm_metric_decisions.get(metric_name, {})
        reference_metric_decision = reference_metric_decisions.get(metric_name, {})

        llm_decision = llm_metric_decision.get("decision")
        reference_decision = reference_metric_decision.get("decision")

        stable_workload_rel_score_entries: List[Dict[str, Any]] = []
        llm_reliable_reference_stable_entries: List[Dict[str, Any]] = []
        reference_reliable_llm_stable_entries: List[Dict[str, Any]] = []

        n_workloads_llm_reliable_reference_not_stable = 0
        n_workloads_reference_reliable_llm_not_stable = 0

        for workload_key in common_workload_keys:
            llm_workload_metric = (
                llm_workloads_by_key[workload_key]
                .get("metric_results", {})
                .get(metric_name, {})
            )

            reference_workload_metric = (
                reference_workloads_by_key[workload_key]
                .get("metric_results", {})
                .get(metric_name, {})
            )

            llm_is_stable = bool(llm_workload_metric.get("is_stable", False))
            reference_is_stable = bool(reference_workload_metric.get("is_stable", False))

            llm_reliable = bool(llm_workload_metric.get("has_reliable_effect", False))
            reference_reliable = bool(
                reference_workload_metric.get("has_reliable_effect", False)
            )

            if llm_reliable and not reference_is_stable:
                n_workloads_llm_reliable_reference_not_stable += 1

            if reference_reliable and not llm_is_stable:
                n_workloads_reference_reliable_llm_not_stable += 1

            llm_target_representative = llm_workload_metric.get("target_representative")
            reference_target_representative = reference_workload_metric.get(
                "target_representative"
            )

            if (
                llm_target_representative is None
                or reference_target_representative is None
                or not math.isfinite(llm_target_representative)
                or not math.isfinite(reference_target_representative)
                or llm_target_representative <= 0
            ):
                continue

            workload_rel_score = float(
                reference_target_representative / llm_target_representative
            )

            entry = {
                "workload_key": workload_key,
                "rel_score": round(workload_rel_score, 4),
                "llm_target_representative": llm_target_representative,
                "reference_target_representative": reference_target_representative,
                "llm_is_stable": llm_is_stable,
                "reference_is_stable": reference_is_stable,
                "llm_has_reliable_effect": llm_reliable,
                "reference_has_reliable_effect": reference_reliable,
                "metric_name": metric_name,
                "unit": llm_metric_decision.get("unit", ""),
            }

            if llm_is_stable and reference_is_stable:
                stable_workload_rel_score_entries.append(entry)

            if llm_reliable and reference_is_stable:
                llm_reliable_reference_stable_entries.append(entry)

            if reference_reliable and llm_is_stable:
                reference_reliable_llm_stable_entries.append(entry)

        stable_values = extract_rel_scores(stable_workload_rel_score_entries)
        llm_reliable_reference_stable_values = extract_rel_scores(
            llm_reliable_reference_stable_entries
        )
        reference_reliable_llm_stable_values = extract_rel_scores(
            reference_reliable_llm_stable_entries
        )

        stable_signal = rel_score_signal(stable_values)
        llm_reliable_reference_stable_signal = rel_score_signal(
            llm_reliable_reference_stable_values
        )
        reference_reliable_llm_stable_signal = rel_score_signal(
            reference_reliable_llm_stable_values
        )

        has_stable_rel_score = bool(stable_values)
        has_llm_reliable_reference_stable_rel_score = bool(
            llm_reliable_reference_stable_values
        )
        has_reference_reliable_llm_stable_rel_score = bool(
            reference_reliable_llm_stable_values
        )

        if has_stable_rel_score:
            any_rel_score_computable = True

        quality_fields = {
            "n_total_workloads": llm_metric_decision.get("n_total_workloads"),
            "n_stable": llm_metric_decision.get("n_stable"),
            "n_reliable": llm_metric_decision.get("n_reliable"),
            "n_improved": llm_metric_decision.get("n_improved"),
            "n_regressed": llm_metric_decision.get("n_regressed"),
            "n_insufficient_signal": llm_metric_decision.get("n_insufficient_signal"),
            "stable_workload_coverage_pct": llm_metric_decision.get(
                "stable_workload_coverage_pct"
            ),
            "reliable_effect_coverage_pct": llm_metric_decision.get(
                "reliable_effect_coverage_pct"
            ),
            "harmonic_mean_snr_reliable": llm_metric_decision.get(
                "harmonic_mean_snr_reliable"
            ),
            "harmonic_mean_snr_stable": llm_metric_decision.get(
                "harmonic_mean_snr_stable"
            ),
            "harmonic_mean_ratio_reliable": llm_metric_decision.get(
                "harmonic_mean_ratio_reliable"
            ),
            "harmonic_mean_ratio_stable": llm_metric_decision.get(
                "harmonic_mean_ratio_stable"
            ),
            "mean_ratio_reliable": llm_metric_decision.get("mean_ratio_reliable"),
            "std_ratio_reliable": llm_metric_decision.get("std_ratio_reliable"),
            "mean_ratio_stable": llm_metric_decision.get("mean_ratio_stable"),
            "std_ratio_stable": llm_metric_decision.get("std_ratio_stable"),
            "unit": llm_metric_decision.get("unit", ""),
            "decision": llm_decision,
            "reference_decision": reference_decision,
        }

        per_metric_results[metric_name] = {
            "workload_rel_scores_stable": stable_workload_rel_score_entries,
            "workload_rel_scores_llm_reliable_reference_stable": (
                llm_reliable_reference_stable_entries
            ),
            "workload_rel_scores_reference_reliable_llm_stable": (
                reference_reliable_llm_stable_entries
            ),

            "harmonic_mean_rel_score_stable": _round_or_none(
                _harmonic_mean(stable_values)
            ),
            "mean_rel_score_stable": _round_or_none(
                _arithmetic_mean(stable_values)
            ),
            "std_rel_score_stable": _round_or_none(
                _sample_std(stable_values)
            ),
            "n_workloads_scored_stable": len(stable_values),
            "rel_score_signal_stable": stable_signal,
            "rel_score_direction_consistent_stable": stable_signal != "conflicting",

            "harmonic_mean_rel_score_llm_reliable_reference_stable": _round_or_none(
                _harmonic_mean(llm_reliable_reference_stable_values)
            ),
            "mean_rel_score_llm_reliable_reference_stable": _round_or_none(
                _arithmetic_mean(llm_reliable_reference_stable_values)
            ),
            "std_rel_score_llm_reliable_reference_stable": _round_or_none(
                _sample_std(llm_reliable_reference_stable_values)
            ),
            "n_workloads_scored_llm_reliable_reference_stable": len(
                llm_reliable_reference_stable_values
            ),
            "rel_score_signal_llm_reliable_reference_stable": (
                llm_reliable_reference_stable_signal
            ),
            "rel_score_direction_consistent_llm_reliable_reference_stable": (
                llm_reliable_reference_stable_signal != "conflicting"
            ),

            "harmonic_mean_rel_score_reference_reliable_llm_stable": _round_or_none(
                _harmonic_mean(reference_reliable_llm_stable_values)
            ),
            "mean_rel_score_reference_reliable_llm_stable": _round_or_none(
                _arithmetic_mean(reference_reliable_llm_stable_values)
            ),
            "std_rel_score_reference_reliable_llm_stable": _round_or_none(
                _sample_std(reference_reliable_llm_stable_values)
            ),
            "n_workloads_scored_reference_reliable_llm_stable": len(
                reference_reliable_llm_stable_values
            ),
            "rel_score_signal_reference_reliable_llm_stable": (
                reference_reliable_llm_stable_signal
            ),
            "rel_score_direction_consistent_reference_reliable_llm_stable": (
                reference_reliable_llm_stable_signal != "conflicting"
            ),

            "n_workloads_llm_reliable_reference_not_stable": (
                n_workloads_llm_reliable_reference_not_stable
            ),
            "n_workloads_reference_reliable_llm_not_stable": (
                n_workloads_reference_reliable_llm_not_stable
            ),

            "has_stable_rel_score": has_stable_rel_score,
            "has_llm_reliable_reference_stable_rel_score": (
                has_llm_reliable_reference_stable_rel_score
            ),
            "has_reference_reliable_llm_stable_rel_score": (
                has_reference_reliable_llm_stable_rel_score
            ),

            "reason": "ok" if has_stable_rel_score else "no_common_stable_workloads",
            **quality_fields,
        }

    if not any_rel_score_computable:
        return None

    return {
        "patch_ok": patch_ok,
        "tests_ok": tests_ok,
        "any_rel_score_computable": any_rel_score_computable,
        "metrics": per_metric_results,
    }