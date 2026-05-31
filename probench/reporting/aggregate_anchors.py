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


def _load_anchor_data(anchors_dir: Path) -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []

    if not anchors_dir.exists():
        raise FileNotFoundError(f"anchors_dir not found: {anchors_dir}")

    for scenario_dir in sorted(p for p in anchors_dir.iterdir() if p.is_dir()):
        scenario_id = scenario_dir.name
        candidate_paths = [
            scenario_dir / "comparisons" / "baseline_vs_reference" / "report.json",
            scenario_dir / "baseline_vs_reference" / "report.json",
        ]
        report_path = next((p for p in candidate_paths if p.exists()), None)
        report      = load_json(report_path) if report_path is not None else None

        if report is None:
            print(f"Warning: missing anchor report for {scenario_id}. Checked: {candidate_paths}")

        scenarios.append({"scenario_id": scenario_id, "report": report})

    return scenarios


def _discover_metric_names(scenarios: List[Dict[str, Any]]) -> List[str]:
    for scenario in scenarios:
        report = scenario.get("report")
        if report and report.get("metric_decisions"):
            return sorted(report["metric_decisions"].keys())
    return []

def _compute_anchor_gate_funnel(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total         = len(scenarios)
    n_report_exists = 0
    n_measurable    = 0
    missing_ids:        List[str] = []
    not_measurable_ids: List[str] = []

    for scenario in scenarios:
        report = scenario["report"]
        if report is None:
            missing_ids.append(scenario["scenario_id"])
            continue
        n_report_exists += 1
        if report.get("measurement_status") != "evaluated":
            not_measurable_ids.append(scenario["scenario_id"])
            continue
        n_measurable += 1

    return {
        "n_total":             n_total,
        "n_report_exists":     n_report_exists,
        "n_missing_report":    len(missing_ids),
        "missing_ids":         missing_ids,
        "n_measurable":        n_measurable,
        "n_not_measurable":    len(not_measurable_ids),
        "measurable_rate_pct": _pct(n_measurable, n_report_exists),
        "not_measurable_ids":  not_measurable_ids,
    }

def _compute_anchor_per_metric(
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

            n_total_prs += 1
            scenario_id  = scenario["scenario_id"]
            decision     = metric_decision.get("decision")

            if decision in decision_counts:
                decision_counts[decision] += 1
                decision_ids[decision].append(scenario_id)
            elif decision:
                decision_counts["other"] += 1
                decision_ids["other"].append(scenario_id)

            if decision in {METRIC_IMPROVED, METRIC_REGRESSED}:
                _append_finite(metric_decision, "harmonic_mean_ratio_reliable", overall_ratio_reliable)

            if decision == METRIC_IMPROVED:
                _append_finite(metric_decision, "harmonic_mean_ratio_reliable", imp_ratio_reliable)

            elif decision == METRIC_REGRESSED:
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
                decision: {"n": decision_counts[decision], "scenario_ids": decision_ids[decision]}
                for decision in ALL_DECISIONS + ["other"]
                if decision_counts.get(decision, 0) > 0
            },
        }

    return per_metric

def _compute_anchor_overall(per_metric: Dict[str, Any]) -> Dict[str, Any]:
    overall_ratio:  List[float] = []
    overall_rates:  List[float] = []
    imp_ratio:      List[float] = []
    imp_rates:      List[float] = []
    reg_ratio:      List[float] = []
    reg_rates:      List[float] = []
    conf_rates:     List[float] = []

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
        "mean_improvement_factor_overall":          _round(_mean(overall_ratio)),
        "mean_overall_rate_over_102":               _round(_mean(overall_rates), 1),
        "mean_improvement_factor_improved":         _round(_mean(imp_ratio)),
        "mean_improved_rate_over_102":              _round(_mean(imp_rates),     1),
        "mean_improvement_factor_regressed":        _round(_mean(reg_ratio)),
        "mean_regressed_rate_over_102":             _round(_mean(reg_rates),     1),
        "mean_conflicting_rate_over_102":           _round(_mean(conf_rates),    1),
    }

def aggregate_anchors(
    anchors_dir: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    anchors_dir  = Path(anchors_dir)
    scenarios    = _load_anchor_data(anchors_dir)

    if not scenarios:
        raise ValueError(f"No scenario directories found under {anchors_dir}")

    metric_names   = _discover_metric_names(scenarios)
    runtime_metric = _find_metric(metric_names, RUNTIME_FRAGMENT)
    peak_metric    = _find_metric(metric_names, PEAK_FRAGMENT)
    twmu_metric    = _find_metric(metric_names, TWMU_FRAGMENT)
    ordered_names  = _ordered_metric_names(metric_names, runtime_metric, peak_metric, twmu_metric)

    per_metric = _compute_anchor_per_metric(scenarios=scenarios, metric_names=ordered_names)

    summary = {
        "anchors_dir":       str(anchors_dir),
        "n_scenarios_total": len(scenarios),
        "n_benchmark_total": N_BENCHMARK_TOTAL,
        "metric_roles": {
            "runtime": runtime_metric,
            "peak":    peak_metric,
            "twmu":    twmu_metric,
        },
        "gate_funnel": _compute_anchor_gate_funnel(scenarios),
        "per_metric":  per_metric,
        "overall":     _compute_anchor_overall(per_metric),
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


def format_anchor_report(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    metric_roles = summary.get("metric_roles", {})

    lines.extend([
        "=" * 80,
        f"  Anchor Summary: {Path(summary.get('anchors_dir', '?')).name}",
        f"  Metrics: runtime={metric_roles.get('runtime')}  peak={metric_roles.get('peak')}  twmu={metric_roles.get('twmu')}",
        "=" * 80,
        "",
        "── Gate Funnel ─────────────────────────────────────────────────────────────",
    ])

    gate_funnel = summary.get("gate_funnel", {})
    lines.extend([
        f"  Total:          {gate_funnel.get('n_total')}",
        f"  Report exists:  {gate_funnel.get('n_report_exists')} / {gate_funnel.get('n_total')}",
        f"  Measurable:     {gate_funnel.get('n_measurable')} / {gate_funnel.get('n_report_exists')}   ({gate_funnel.get('measurable_rate_pct')}%)",
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

    overall = summary.get("overall", {})
    lines.extend([
        "",
        "── Overall ─────────────────────────────────────────────────────────────────",
        f"  mean_improvement_factor_overall:      {_format_optional(overall.get('mean_improvement_factor_overall'))}",
        f"  mean_overall_rate/102:                {_format_optional(overall.get('mean_overall_rate_over_102'))}%",
        f"  mean_improvement_factor_improved:     {_format_optional(overall.get('mean_improvement_factor_improved'))}",
        f"  mean_improved_rate/102:               {_format_optional(overall.get('mean_improved_rate_over_102'))}%",
        f"  mean_improvement_factor_regressed:    {_format_optional(overall.get('mean_improvement_factor_regressed'))}",
        f"  mean_regressed_rate/102:              {_format_optional(overall.get('mean_regressed_rate_over_102'))}%",
        f"  mean_conflicting_rate/102:            {_format_optional(overall.get('mean_conflicting_rate_over_102'))}%",
        f"\n{'=' * 80}",
    ])

    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors_dir",  type=Path, default="results_experiments_valid/anchors")
    parser.add_argument("--output_path",  type=Path, default="anchor_summary/anchor_summary.json")
    parser.add_argument("--print_report", action="store_true", default=True)

    args        = parser.parse_args()
    output_path = args.output_path or args.anchors_dir / "anchor_summary.json"

    summary = aggregate_anchors(anchors_dir=args.anchors_dir, output_path=output_path)

    if args.print_report:
        print(format_anchor_report(summary))

    print(f"\nSaved → {output_path}")