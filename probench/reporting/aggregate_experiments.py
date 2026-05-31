from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from probench.utils.io_utils import load_json, save_json
from probench.harness.evaluation.report_analyzer import (
    METRIC_IMPROVED,
    METRIC_REGRESSED,
    METRIC_NO_SIGNAL,
    METRIC_CONFLICTING,
    METRIC_NOT_MEASURABLE,
)


N_BENCHMARK_TOTAL = 102

ALL_DECISIONS = [
    METRIC_IMPROVED,
    METRIC_REGRESSED,
    METRIC_CONFLICTING,
    METRIC_NO_SIGNAL,
    METRIC_NOT_MEASURABLE,
]

RUNTIME_FRAGMENT = "runtime"
PEAK_FRAGMENT    = "peak"
TWMU_FRAGMENT    = "twmu"

def _mean(values: List[float]) -> Optional[float]:
    finite_values = [
        float(v) for v in values
        if isinstance(v, (int, float)) and math.isfinite(v)
    ]
    return float(sum(finite_values) / len(finite_values)) if finite_values else None


def _round(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    return None if value is None else round(value, ndigits)


def _pct(numerator: int, denominator: int, ndigits: int = 1) -> Optional[float]:
    return _round(numerator / denominator * 100.0, ndigits) if denominator > 0 else None


def _append_finite(source: Dict[str, Any], key: str, target: List[float]) -> None:
    value = source.get(key)
    if isinstance(value, (int, float)) and math.isfinite(value):
        target.append(float(value))


def _format_optional(value: Optional[float]) -> str:
    return "N/A" if value is None else str(value)


def _pool_stats(
    ratio_reliable_vals: List[float],
    n_total_prs:         int,
) -> Dict[str, Any]:
    return {
        "mean_improvement_factor": _round(_mean(ratio_reliable_vals)),
        "n_valid_prs":             len(ratio_reliable_vals),
        "valid_pr_rate_pct":       _pct(len(ratio_reliable_vals), n_total_prs),
        "rate_over_102_pct":       _pct(len(ratio_reliable_vals), N_BENCHMARK_TOTAL),
    }


def _conflicting_stats(n_conflicting: int, n_total_prs: int) -> Dict[str, Any]:
    return {
        "n_valid_prs":       n_conflicting,
        "valid_pr_rate_pct": _pct(n_conflicting, n_total_prs),
        "rate_over_102_pct": _pct(n_conflicting, N_BENCHMARK_TOTAL),
    }

def _load_scenario_data(experiment_dir: Path) -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []
    instances_dir = experiment_dir / "instances"

    if not instances_dir.exists():
        raise FileNotFoundError(f"instances/ not found under {experiment_dir}")

    for instance_dir in sorted(path for path in instances_dir.iterdir() if path.is_dir()):
        scenario_id     = instance_dir.name
        report_path     = instance_dir / "comparisons" / "baseline_vs_llm" / "report.json"
        rel_scores_path = instance_dir / "rel_scores.json"

        report     = load_json(report_path)     if report_path.exists()     else None
        rel_scores = load_json(rel_scores_path) if rel_scores_path.exists() else None

        if report is None:
            print(f"Warning: missing report.json for {scenario_id} at {report_path}")

        scenarios.append({"scenario_id": scenario_id, "report": report, "rel_scores": rel_scores})

    return scenarios


def _discover_metric_names(scenarios: List[Dict[str, Any]]) -> List[str]:
    for scenario in scenarios:
        report = scenario.get("report")
        if report and report.get("metric_decisions"):
            return sorted(report["metric_decisions"].keys())
    return []


def _find_metric(metric_names: List[str], fragment: str) -> Optional[str]:
    return next((m for m in metric_names if fragment.lower() in m.lower()), None)


def _ordered_metric_names(
    metric_names: List[str],
    runtime_metric: Optional[str],
    peak_metric: Optional[str],
    twmu_metric: Optional[str],
) -> List[str]:
    ordered = []
    for m in [runtime_metric, peak_metric, twmu_metric]:
        if m and m in metric_names:
            ordered.append(m)
    for m in metric_names:
        if m not in ordered:
            ordered.append(m)
    return ordered

def _compute_gate_funnel(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total      = len(scenarios)
    n_patch_ok   = 0
    n_tests_ok   = 0
    n_measurable = 0
    n_rel_ready  = 0

    patch_failed_ids:   List[str] = []
    tests_failed_ids:   List[str] = []
    not_measurable_ids: List[str] = []
    no_report_ids:      List[str] = []

    patch_errors: Dict[str, List[str]] = {}
    failed_tests: Dict[str, List[str]] = {}

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        report      = scenario["report"]

        if report is None:
            no_report_ids.append(scenario_id)
            patch_failed_ids.append(scenario_id)
            continue

        measurement_status = report.get("measurement_status", "")
        patch_ok           = bool(report.get("patch_ok", False))
        tests_ok           = bool(report.get("tests_ok", False))
        relative_ready     = bool(report.get("relative_ready", False))
        gates              = report.get("gates") or {}

        if not patch_ok or measurement_status == "patch_failed":
            patch_failed_ids.append(scenario_id)
            errors = [
                r.get("error") for r in gates.get("patch", [])
                if not r.get("ok") and r.get("error")
            ]
            if errors:
                patch_errors[scenario_id] = errors
            continue

        n_patch_ok += 1

        if not tests_ok or measurement_status == "tests_failed":
            tests_failed_ids.append(scenario_id)
            failed = (gates.get("correctness") or {}).get("failed_tests") or []
            if failed:
                failed_tests[scenario_id] = failed
            continue

        n_tests_ok += 1

        if measurement_status == "not_measurable":
            not_measurable_ids.append(scenario_id)
            continue

        n_measurable += 1

        if relative_ready and scenario["rel_scores"] is not None:
            n_rel_ready += 1

    return {
        "n_total":                  n_total,
        "n_patch_ok":               n_patch_ok,
        "n_patch_failed":           len(patch_failed_ids),
        "n_no_report":              len(no_report_ids),
        "patch_success_rate_pct":   _pct(n_patch_ok, n_total),
        "patch_failed_ids":         patch_failed_ids,
        "patch_errors":             patch_errors,
        "n_tests_ok":               n_tests_ok,
        "n_tests_failed":           len(tests_failed_ids),
        "test_success_rate_pct":    _pct(n_tests_ok, n_total),
        "tests_failed_ids":         tests_failed_ids,
        "failed_tests_by_scenario": failed_tests,
        "n_measurable":             n_measurable,
        "n_not_measurable":         len(not_measurable_ids),
        "measurable_rate_pct":      _pct(n_measurable, n_total),
        "not_measurable_ids":       not_measurable_ids,
        "n_rel_score_ready":        n_rel_ready,
        "rel_score_ready_rate_pct": _pct(n_rel_ready, n_total),
    }

def _compute_per_metric(
    scenarios: List[Dict[str, Any]],
    metric_names: List[str],
) -> Dict[str, Any]:
    per_metric: Dict[str, Any] = {}

    for metric_name in metric_names:
        n_total_prs = 0

        overall_ratio_reliable: List[float] = []
        imp_ratio_reliable:     List[float] = []
        reg_ratio_reliable:     List[float] = []

        decision_counts: Dict[str, int]       = {d: 0 for d in ALL_DECISIONS}
        decision_counts["other"]              = 0
        decision_ids:    Dict[str, List[str]] = {d: [] for d in ALL_DECISIONS}
        decision_ids["other"]                 = []

        for scenario in scenarios:
            report = scenario["report"]
            if report is None:
                continue

            metric_decision = (report.get("metric_decisions") or {}).get(metric_name)
            if metric_decision is None:
                continue

            n_total_prs          += 1
            scenario_id           = scenario["scenario_id"]
            metric_decision_value = metric_decision.get("decision")

            if metric_decision_value in decision_counts:
                decision_counts[metric_decision_value] += 1
                decision_ids[metric_decision_value].append(scenario_id)
            elif metric_decision_value:
                decision_counts["other"] += 1
                decision_ids["other"].append(scenario_id)

            if metric_decision_value in {METRIC_IMPROVED, METRIC_REGRESSED}:
                _append_finite(metric_decision, "harmonic_mean_ratio_reliable", overall_ratio_reliable)

            if metric_decision_value == METRIC_IMPROVED:
                _append_finite(metric_decision, "harmonic_mean_ratio_reliable", imp_ratio_reliable)

            elif metric_decision_value == METRIC_REGRESSED:
                _append_finite(metric_decision, "harmonic_mean_ratio_reliable", reg_ratio_reliable)

        n_improved    = decision_counts.get(METRIC_IMPROVED,    0)
        n_regressed   = decision_counts.get(METRIC_REGRESSED,   0)
        n_conflicting = decision_counts.get(METRIC_CONFLICTING, 0)

        per_metric[metric_name] = {
            "n_total_prs":                     n_total_prs,
            "n_improved":                      n_improved,
            "n_regressed":                     n_regressed,
            "n_conflicting":                   n_conflicting,
            "n_improved_rate_over_102_pct":    _pct(n_improved,    N_BENCHMARK_TOTAL),
            "n_regressed_rate_over_102_pct":   _pct(n_regressed,   N_BENCHMARK_TOTAL),
            "n_conflicting_rate_over_102_pct": _pct(n_conflicting, N_BENCHMARK_TOTAL),
            "overall":     _pool_stats(overall_ratio_reliable, n_total_prs),
            "improved":    _pool_stats(imp_ratio_reliable,     n_total_prs),
            "regressed":   _pool_stats(reg_ratio_reliable,     n_total_prs),
            "conflicting": _conflicting_stats(n_conflicting,   n_total_prs),
            "decision_distribution": {
                decision: {
                    "n":            decision_counts[decision],
                    "scenario_ids": decision_ids[decision],
                }
                for decision in ALL_DECISIONS + ["other"]
                if decision_counts.get(decision, 0) > 0
            },
        }

    return per_metric

def _compute_overall(per_metric: Dict[str, Any]) -> Dict[str, Any]:
    overall_ratio: List[float] = []
    overall_rates: List[float] = []
    imp_ratio:     List[float] = []
    imp_rates:     List[float] = []
    reg_ratio:     List[float] = []
    reg_rates:     List[float] = []
    conf_rates:    List[float] = []

    for metric_summary in per_metric.values():
        ov   = metric_summary["overall"]
        imp  = metric_summary["improved"]
        reg  = metric_summary["regressed"]
        conf = metric_summary["conflicting"]
        _append_finite(ov,   "mean_improvement_factor", overall_ratio)
        _append_finite(ov,   "rate_over_102_pct",        overall_rates)
        _append_finite(imp,  "mean_improvement_factor",  imp_ratio)
        _append_finite(imp,  "rate_over_102_pct",         imp_rates)
        _append_finite(reg,  "mean_improvement_factor",  reg_ratio)
        _append_finite(reg,  "rate_over_102_pct",         reg_rates)
        _append_finite(conf, "rate_over_102_pct",         conf_rates)

    return {
        "mean_improvement_factor_overall":       _round(_mean(overall_ratio)),
        "mean_overall_rate_over_102":            _round(_mean(overall_rates), 1),
        "mean_improvement_factor_improved":      _round(_mean(imp_ratio)),
        "mean_improved_rate_over_102":           _round(_mean(imp_rates),     1),
        "mean_improvement_factor_regressed":     _round(_mean(reg_ratio)),
        "mean_regressed_rate_over_102":          _round(_mean(reg_rates),     1),
        "mean_conflicting_rate_over_102":        _round(_mean(conf_rates),    1),
    }

def aggregate_experiment(
    experiment_dir: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    experiment_dir = Path(experiment_dir)
    scenarios      = _load_scenario_data(experiment_dir)

    if not scenarios:
        raise ValueError(f"No scenario instances found under {experiment_dir}/instances/")

    metric_names   = _discover_metric_names(scenarios)
    runtime_metric = _find_metric(metric_names, RUNTIME_FRAGMENT)
    peak_metric    = _find_metric(metric_names, PEAK_FRAGMENT)
    twmu_metric    = _find_metric(metric_names, TWMU_FRAGMENT)
    ordered_names  = _ordered_metric_names(metric_names, runtime_metric, peak_metric, twmu_metric)

    per_metric = _compute_per_metric(scenarios=scenarios, metric_names=ordered_names)

    summary = {
        "experiment_dir":    str(experiment_dir),
        "n_scenarios_total": len(scenarios),
        "n_benchmark_total": N_BENCHMARK_TOTAL,
        "metric_roles": {
            "runtime": runtime_metric,
            "peak":    peak_metric,
            "twmu":    twmu_metric,
        },
        "gate_funnel": _compute_gate_funnel(scenarios),
        "per_metric":  per_metric,
        "overall":     _compute_overall(per_metric),
    }

    if output_path:
        save_json(str(output_path), summary)

    return summary

def _fmt_pool(label: str, pool: Dict[str, Any], total_prs: int) -> str:
    return (
        f"    {label:<12}"
        f"  Improvement Factor={_format_optional(pool.get('mean_improvement_factor')):<12}"
        f"  valid_prs={pool.get('n_valid_prs')}/{total_prs}"
        f" ({pool.get('valid_pr_rate_pct')}%)"
        f"  end-to-end/102={pool.get('rate_over_102_pct')}%"
    )


def format_report(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    metric_roles = summary.get("metric_roles", {})

    lines.extend([
        "=" * 80,
        f"  {Path(summary.get('experiment_dir', '?')).name}",
        f"  Metrics: runtime={metric_roles.get('runtime')}  peak={metric_roles.get('peak')}  twmu={metric_roles.get('twmu')}",
        "=" * 80,
    ])

    gate_funnel = summary.get("gate_funnel", {})
    lines.extend([
        "",
        "── Gate Funnel ─────────────────────────────────────────────────────────────",
        f"  Total:              {gate_funnel.get('n_total')}",
        f"  Patch applied:      {gate_funnel.get('n_patch_ok')} / {gate_funnel.get('n_total')}   ({gate_funnel.get('patch_success_rate_pct')}%)",
        f"  Correctness passed: {gate_funnel.get('n_tests_ok')} / {gate_funnel.get('n_total')}   ({gate_funnel.get('test_success_rate_pct')}%)",
        f"  Measurable:         {gate_funnel.get('n_measurable')} / {gate_funnel.get('n_total')}   ({gate_funnel.get('measurable_rate_pct')}%)",
        "",
        "── Per-Metric ──────────────────────────────────────────────────────────────",
    ])

    for metric_name, metric_summary in summary.get("per_metric", {}).items():
        total_prs = metric_summary.get("n_total_prs")
        conf      = metric_summary["conflicting"]
        dd        = metric_summary["decision_distribution"]

        lines.append(
            f"\n  {metric_name}  "
            f"[n_improved={metric_summary.get('n_improved')} ({metric_summary.get('n_improved_rate_over_102_pct')}%)  "
            f"n_regressed={metric_summary.get('n_regressed')} ({metric_summary.get('n_regressed_rate_over_102_pct')}%)  "
            f"n_conflicting={metric_summary.get('n_conflicting')} ({metric_summary.get('n_conflicting_rate_over_102_pct')}%)]"
        )
        lines.append("    ---")
        lines.append(_fmt_pool("Overall",   metric_summary["overall"],   total_prs))
        lines.append(_fmt_pool("Improved",  metric_summary["improved"],  total_prs))
        lines.append(_fmt_pool("Regressed", metric_summary["regressed"], total_prs))
        lines.append(
            f"    {'Conflicting':<12}"
            f"  {'':30}"
            f"  valid_prs={conf.get('n_valid_prs')}/{total_prs}"
            f" ({conf.get('valid_pr_rate_pct')}%)"
            f"  end-to-end/102={conf.get('rate_over_102_pct')}%"
        )

        decision_parts: List[str] = []
        for decision in ALL_DECISIONS + ["other"]:
            entry = dd.get(decision)
            if not entry or entry["n"] == 0:
                continue
            if decision == METRIC_NO_SIGNAL:
                decision_parts.append(f"{decision}={entry['n']}")
            else:
                ids = ", ".join(entry["scenario_ids"])
                decision_parts.append(f"{decision}={entry['n']} [{ids}]")
        lines.append("    ---")
        lines.append(f"    Decisions: {'  '.join(decision_parts)}")

    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        default="results_experiments_valid/evals/oracle_full_file__claude-sonnet-4-6@default__tNone__topNone__pd_128b608333b1",
    )
    parser.add_argument("--output_path",  type=Path, default=None)
    parser.add_argument("--print_report", action="store_true", default=True)

    args        = parser.parse_args()
    output_path = args.output_path or args.experiment_dir / "experiment_summary.json"

    summary = aggregate_experiment(
        experiment_dir=args.experiment_dir,
        output_path=output_path,
    )

    if args.print_report:
        print(format_report(summary))

    print(f"\nSaved → {output_path}")