"""
run_all_aggregations.py
=======================
Runs aggregate_experiment over all eval directories and
aggregate_anchors over the anchors directory.

Usage:
    python run_all_aggregations.py --results_dir results_experiments_valid
    python run_all_aggregations.py  # uses default path above
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from probench.reporting.aggregate_experiments import aggregate_experiment, format_report
from probench.reporting.aggregate_anchors import aggregate_anchors, format_anchor_report


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _separator(label: str = "") -> None:
    if label:
        print(f"\n{'─' * 40} {label} {'─' * 40}")
    else:
        print("─" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_all(
    results_dir: Path,
    print_reports: bool = True,
) -> None:
    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    # ── 1. Anchors ────────────────────────────────────────────────────────────
    anchors_dir = results_dir / "anchors"
    if anchors_dir.exists():
        _separator("ANCHORS")
        output_path = anchors_dir / "anchor_summary.json"
        try:
            summary = aggregate_anchors(
                anchors_dir=anchors_dir,
                output_path=output_path,
            )
            if print_reports:
                print(format_anchor_report(summary))
            print(f"✓ Anchors saved → {output_path}")
        except Exception as e:
            print(f"✗ Anchors FAILED: {e}")
            traceback.print_exc()
    else:
        print(f"⚠ anchors/ not found under {results_dir} — skipping")

    # ── 2. Evals ──────────────────────────────────────────────────────────────
    evals_dir = results_dir / "evals"
    if not evals_dir.exists():
        print(f"⚠ evals/ not found under {results_dir} — skipping")
        return

    eval_dirs = sorted(p for p in evals_dir.iterdir() if p.is_dir())

    if not eval_dirs:
        print(f"⚠ No experiment directories found under {evals_dir}")
        return

    print(f"\nFound {len(eval_dirs)} experiment(s) under {evals_dir}\n")

    n_ok    = 0
    n_fail  = 0
    failed  = []

    for eval_dir in eval_dirs:
        _separator(eval_dir.name)
        output_path = eval_dir / "experiment_summary.json"
        try:
            summary = aggregate_experiment(
                experiment_dir=eval_dir,
                output_path=output_path,
            )
            if print_reports:
                print(format_report(summary))
            print(f"✓ Saved → {output_path}")
            n_ok += 1
        except Exception as e:
            print(f"✗ FAILED: {eval_dir.name}: {e}")
            traceback.print_exc()
            n_fail += 1
            failed.append(eval_dir.name)

    # ── Summary ───────────────────────────────────────────────────────────────
    _separator("SUMMARY")
    print(f"  Total experiments: {len(eval_dirs)}")
    print(f"  Success:           {n_ok}")
    print(f"  Failed:            {n_fail}")
    if failed:
        print("  Failed dirs:")
        for name in failed:
            print(f"    - {name}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all aggregations over results_experiments_valid"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results_experiments_valid"),
        help="Root results directory containing anchors/ and evals/",
    )
    parser.add_argument(
        "--no_print",
        action="store_true",
        default=False,
        help="Skip printing reports to stdout (just save JSON)",
    )

    args = parser.parse_args()

    run_all(
        results_dir=args.results_dir,
        print_reports=not args.no_print,
    )