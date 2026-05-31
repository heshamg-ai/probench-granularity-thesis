"""
ProBench harness runner

Inference layout
---------------------------
This runner expects inference outputs under:

    <inference_root>/<scenario_id>/
        completion.txt
        meta.json (optional)
        DONE (optional)

Core behavior
-------------
For each dataset row:

1) Builds/reuses Docker images:
   - baseline image from baseline_sha
   - reference image from reference_sha (optional, if --with-gold-solution)

2) Runs containers:
   - baseline (skippable if already done)
   - reference (skippable if already done)
   - llm: reuses baseline image, mounts inference dir as /llm_inference
          and patching happens inside the container entrypoint.

3) Evaluates via ReportAnalyzer:
   - correctness comparison (baseline vs llm)
   - benchmark comparison (baseline vs llm), optionally correctness-gated
   - optional baseline vs reference evaluation

4) Exports:
   - comparisons/<experiment_id>/report.json + csv/xlsx
   - SUMMARY__<experiment_id>.json per scenario
"""

from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional

from probench.prep.data_loader import DatasetLoader
from probench.utils.docker_utils import build_or_pull_image, execute_variant_container, remove_image
from probench.utils import global_config as GC
from probench.utils.io_utils import load_yaml, save_json, load_json
from probench.utils.harness_utils import load_experiment_meta
from probench.harness.evaluation.report_analyzer import analyze_comparison, compute_rel_scores

logger = logging.getLogger("probench.harness")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

CORRECTNESS_FILE = "correctness_report_summary.json"
BENCHMARK_FILE = "performance_results.json"
DONE_FILE = "DONE"

def safe_id(s: str) -> str:
    return str(s).replace("/", "__").replace(":", "_").strip()

def artifact_status(out_dir: Path) -> Dict[str, Any]:
    done_marker = out_dir / DONE_FILE

    return {
        "done": done_marker.exists(),
    }


def repeated_run_dir(root: Path, run_idx: int) -> Path:
    return root / f"run_{run_idx:03d}"


def _repeated_artifact_status(root: Path, n_repeats: int) -> Dict[str, Any]:
    runs = []
    done_count = 0

    for i in range(n_repeats):
        d = repeated_run_dir(root, i)
        st = artifact_status(d)

        runs.append({
            "run_idx": i,
            "path": str(d),
            **st,
        })

        if st["done"]:
            done_count += 1

    return {
        "n_repeats": n_repeats,
        "done_count": done_count,
        "any_done": done_count > 0,
        "all_done": done_count == n_repeats,
        "runs": runs,
    }


def _check_eval_requirements(
    label: str,
    baseline_dir: Path,
    target_dir: Path,
    n_repeats: int,
) -> Dict[str, Any]:
    baseline_status = _repeated_artifact_status(baseline_dir, n_repeats)
    target_status   = _repeated_artifact_status(target_dir,   n_repeats)
    all_done        = baseline_status["all_done"] and target_status["all_done"]
    return {
        "label":   label,
        "ok":      all_done,
        "missing": {
            "baseline_missing_done_file_runs": [r["run_idx"] for r in baseline_status["runs"] if not r["done"]],
            "target_missing_done_file_runs":   [r["run_idx"] for r in target_status["runs"]   if not r["done"]],
        },
        "baseline": baseline_status,
        "target":   target_status,
    }

def main() -> None:
    parser = ArgumentParser(description="ProBench harness runner")
    parser.add_argument(
        "--scenario_id",
        type=str,
        #default="xarray-10554",
        help="Run a single scenario_id only (default: run all)",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("config/config.yaml"), #fixed config path for now, since we need the library metadata for docker builds
        help="Path to config.yaml",
    )
    
    parser.add_argument(
        "--inference_run_dir",
        type=Path,
        #required=True,
        default="inference/additions/oracle_full_file__glm-5.1__tNone__topNone__pd_425f209a1355__full-file",
        help="Path to one inference run folder: inference/<run_id>/"
    )
    
    # mode
    parser.add_argument(
        "--with-gold-solution",
        action="store_true",
        default=True,
        help="Also run the gold/oracle solution",
    )

    # outputs
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("results_experiments_valid"),
        help="Root folder to store run outputs",
    )

    # run control
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        default=False,
        help="Force Docker image rebuild even if image exists",
    )

    parser.add_argument(
        "--skip-baseline-if-done",
        dest="skip_baseline_if_done",
        action="store_true",
        default=True,
        help="Skip baseline run if results already exist (default: enabled)",
    )
    parser.add_argument(
        "--skip-reference-if-done",
        dest="skip_reference_if_done",
        action="store_true",
        default=True,
        help="Skip reference run if results already exist (default: enabled)",
    )
    parser.add_argument(
        "--remove-image",
        action="store_true",
        default=True,
        help="Remove built Docker images after run (useful for saving disk space, but images will need to be rebuilt/load for future runs)",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=3,
        help="Number of independent container runs per variant",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Override the dataset path from the inference run metadata",
    )

    args = parser.parse_args()

    results_root = Path(args.results_root)
    inference_run_dir = Path(args.inference_run_dir)
    experiment_meta = load_experiment_meta(inference_run_dir)
    experiment_id = safe_id(experiment_meta["run_id"])
    dataset_path = Path(args.dataset_path) if args.dataset_path else Path(experiment_meta["prompt_dataset_meta"]["dataset_path"])

    config = load_yaml(args.config_path)
    GC.initialize(config)

    # Run-level inference meta
    logger.info(f"Loaded experiment meta: {experiment_meta}")
    results_root.mkdir(parents=True, exist_ok=True)

    evals_run_root = results_root / "evals" / experiment_id
    evals_run_root.mkdir(parents=True, exist_ok=True)
    save_json(evals_run_root / "RUN_META.json", experiment_meta)

    loader = DatasetLoader(str(dataset_path), mode="parametrized")#todo, we can get dataset directly from meta
    processed = 0
    for item in loader:
        if isinstance(item, tuple):
            row, paramgrid = item
        else:
            row, paramgrid = item, None

        scenario_id = row.get("scenario_id")
        if not scenario_id:
            logger.warning("Row without scenario_id encountered, skipping")
            continue
        scenario_id = str(scenario_id).strip()

        if args.scenario_id is not None and scenario_id != str(args.scenario_id).strip():
            continue

        # dataset fields
        repo = row.get("repo")
        pr_number = row.get("pr_number") or 0
        baseline_sha = row.get("baseline_sha")
        reference_sha = row.get("reference_sha")
        core_tests = row.get("core_tests") or []

        if not repo or not baseline_sha:
            logger.warning(f"{scenario_id}: missing required fields (repo/baseline_sha), skipping")
            continue

        # library metadata from config
        lib_metadata = (GC.libraries.get(repo) or {}).copy()
        repo_url = lib_metadata.get("repo_url")
        module_name = lib_metadata.get("module_name")
        dockerfile = lib_metadata.get("dockerfile")

        if not repo_url or not module_name or not dockerfile:
            logger.warning(
                f"{scenario_id}: missing library metadata in config for repo='{repo}' "
                f"(repo_url/module_name/dockerfile), skipping"
            )
            continue

        # outputs
        anchors_root = results_root / "anchors" / safe_id(scenario_id)
        anchors_root.mkdir(parents=True, exist_ok=True)

        out_baseline_root = anchors_root / f"baseline__{str(baseline_sha)[:12]}"
        out_reference_root = anchors_root / f"reference__{str(reference_sha)[:12]}" if reference_sha else None
        anchor_compare_dir  = anchors_root / "baseline_vs_reference"

        # 2) Evals are per run_id (experiment_id)
        eval_root = evals_run_root / "instances" / safe_id(scenario_id)
        eval_root.mkdir(parents=True, exist_ok=True)

        out_llm_root = eval_root / "llm"
        out_llm_root.mkdir(parents=True, exist_ok=True)

        compare_root        = eval_root / "comparisons"
        compare_bl_dir      = compare_root / "baseline_vs_llm"
        rel_scores_path     = eval_root / "rel_scores.json"
        summary_path        = eval_root / f"SUMMARY__{experiment_id}.json"

        calibration_root = anchors_root / "calibration"
        calibration_root.mkdir(parents=True, exist_ok=True)
        calibration_path = calibration_root / "calibration.json"

        # skip if all artifacts and comparisons done
        reference_enabled = bool(args.with_gold_solution and reference_sha)

        baseline_rep_status = _repeated_artifact_status(out_baseline_root, args.n_repeats)
        reference_rep_status = (
            _repeated_artifact_status(out_reference_root, args.n_repeats)
            if reference_enabled and out_reference_root else None
        )
        llm_rep_status = _repeated_artifact_status(out_llm_root, args.n_repeats)

        _baseline_done = baseline_rep_status["all_done"]
        _reference_done = (not reference_enabled) or reference_rep_status["all_done"]
        _anchor_comparison_done = (not reference_enabled) or (anchor_compare_dir / "report.json").exists()
        _llm_done = llm_rep_status["all_done"]
        _summary_done = summary_path.exists()
        _calibration_done = calibration_path.exists()

        if (
            False
            and _calibration_done
            and _baseline_done
            and _llm_done
            and _reference_done
            and _anchor_comparison_done
            and _summary_done
            and not args.force_rebuild
        ):
            logger.info(f"{scenario_id}: all artifacts and comparisons done -> skipping entirely")
            processed += 1
            continue

        # locate inference output
        inf_dir = inference_run_dir / "instances" / safe_id(scenario_id)
        completion_path = inf_dir / "completion.txt"
        has_inference = completion_path.exists()
        if not has_inference:
            logger.warning(f"{scenario_id}: missing inference completion at {completion_path}, will run anchors only")

        # docker tags
        docker_meta = row.get("docker", {}) or {}
        b_tag = ((docker_meta.get("baseline") or {}).get("tag")) or f"probench:{safe_id(scenario_id)}-baseline"
        r_tag = ((docker_meta.get("reference") or {}).get("tag")) or f"probench:{safe_id(scenario_id)}-reference"

        b_existing = None if args.force_rebuild else (docker_meta.get("baseline") or {}).get("digest")
        r_existing = None if args.force_rebuild else (docker_meta.get("reference") or {}).get("digest")

        # build/reuse baseline image
        logger.info(f"{scenario_id}: build/reuse baseline image...")
        b_tag, _b_digest = build_or_pull_image(
            docker_file=dockerfile,
            tag=b_tag,
            build_context=GC.root,
            build_args={
                "REPO_URL": repo_url,
                "SHA": baseline_sha,
                "VARIANT": "baseline",
                "PR_NUMBER": str(pr_number or ""),
            },
            existing_digest=b_existing,
            force_rebuild=args.force_rebuild
        )

        if not calibration_path.exists() or args.force_rebuild:
            logger.info(f"{scenario_id}: generating shared calibration from baseline")
            try:
                execute_variant_container(
                    image_tag=b_tag,
                    output_dir=calibration_root,
                    scenario_id=scenario_id,
                    version="baseline",
                    lib_metadata=lib_metadata,
                    pr_number=int(pr_number or 0),
                    core_tests=core_tests,
                    params=paramgrid,
                    perf_mode="calibrate",
                )
            except Exception as e:
                logger.error(f"{scenario_id}: calibration execution failed: {e}, skipping to next scenario")
                processed += 1
                continue
        else:
            logger.info(f"{scenario_id}: reusing calibration at {calibration_path}")

        for run_idx in range(args.n_repeats):
            out_baseline = repeated_run_dir(out_baseline_root, run_idx)
            out_baseline.mkdir(parents=True, exist_ok=True)
            baseline_status = artifact_status(out_baseline)

            if args.skip_baseline_if_done and baseline_status["done"]:
                logger.info(f"{scenario_id}: baseline run_{run_idx:03d} already done -> skipping")
                continue

            logger.info(f"{scenario_id}: execute baseline run_{run_idx:03d}")
            try:
                execute_variant_container(
                    image_tag=b_tag,
                    output_dir=out_baseline,
                    scenario_id=scenario_id,
                    version="baseline",
                    lib_metadata=lib_metadata,
                    pr_number=int(pr_number or 0),
                    core_tests=core_tests,
                    params=paramgrid,
                    perf_mode="measure",
                    calibration_path=calibration_path,
                )
            except Exception as e:
                logger.exception(
                    f"{scenario_id}: baseline run_{run_idx:03d} execution failed: {e}"
                )
                continue
        
        r_tag_final: Optional[str] = None
        if reference_enabled:
            for run_idx in range(args.n_repeats):
                out_reference = repeated_run_dir(out_reference_root, run_idx)
                out_reference.mkdir(parents=True, exist_ok=True)
                reference_status = artifact_status(out_reference)

                if args.skip_reference_if_done and reference_status["done"]:
                    logger.info(f"{scenario_id}: reference run_{run_idx:03d} already done -> skipping")
                    continue

                logger.info(f"{scenario_id}: execute reference run_{run_idx:03d}")
                try:
                    logger.info(f"{scenario_id}: build/reuse reference image...")
                    r_tag_final, _r_digest = build_or_pull_image(
                        docker_file=dockerfile,
                        tag=r_tag,
                        build_context=GC.root,
                        build_args={
                            "REPO_URL": repo_url,
                            "SHA": reference_sha,
                            "VARIANT": "reference",
                            "PR_NUMBER": str(pr_number or ""),
                        },
                        existing_digest=r_existing,
                        force_rebuild=args.force_rebuild
                    )

                    execute_variant_container(
                        image_tag=r_tag_final,
                        output_dir=out_reference,
                        scenario_id=scenario_id,
                        version="reference",
                        lib_metadata=lib_metadata,
                        pr_number=int(pr_number or 0),
                        core_tests=core_tests,
                        params=paramgrid,
                        perf_mode="measure",
                        calibration_path=calibration_path,
                    )
                except Exception as e:
                    logger.exception(
                        f"{scenario_id}: reference run_{run_idx:03d} execution failed: {e}"
                    )
                    continue

    
        # execute llm
        if not has_inference:
            logger.warning(f"{scenario_id}: skipping llm runs (no inference completion)")
        for run_idx in range(args.n_repeats) if has_inference else []:
            out_llm = repeated_run_dir(out_llm_root, run_idx)
            out_llm.mkdir(parents=True, exist_ok=True)
            llm_status = artifact_status(out_llm)

            if llm_status["done"]:
                logger.info(f"{scenario_id}: llm run_{run_idx:03d} already done -> skipping")
                continue

            logger.info(f"{scenario_id}: execute llm run_{run_idx:03d} (variant={experiment_id}) using baseline image")
            try:
                execute_variant_container(
                    image_tag=b_tag,
                    output_dir=out_llm,
                    scenario_id=scenario_id,
                    version="llm",
                    lib_metadata=lib_metadata,
                    pr_number=int(pr_number or 0),
                    core_tests=core_tests,
                    params=paramgrid,
                    llm_inference_dir=inf_dir,
                    perf_mode="measure",
                    calibration_path=calibration_path,
                )
            except Exception as e:
                logger.exception(f"{scenario_id}: llm run_{run_idx:03d} execution failed: {e}")
                continue
        
        # evaluation
        compare_root.mkdir(parents=True, exist_ok=True)

        # baseline vs llm
        llm_comparison_report        = None
        baseline_vs_llm_requirements = _check_eval_requirements(
            "baseline_vs_llm",
            out_baseline_root,
            out_llm_root,
            args.n_repeats,
        )
        if baseline_vs_llm_requirements["ok"]:
            logger.info(f"{scenario_id}: analyze_comparison baseline vs llm")
            llm_comparison_report = analyze_comparison(
                baseline_root=out_baseline_root,
                target_root=out_llm_root,
                output_dir=compare_bl_dir,
                label_target=f"llm:{experiment_id}",
                is_llm_run=True,
                n_repeats=args.n_repeats,
            )
        else:
            logger.warning(
                f"{scenario_id}: skipping baseline_vs_llm; "
                f"missing={baseline_vs_llm_requirements['missing']}"
            )

        # baseline vs reference (anchor)
        baseline_vs_reference_requirements = None
        if reference_enabled and out_reference_root is not None:
            baseline_vs_reference_requirements = _check_eval_requirements(
                "baseline_vs_reference",
                out_baseline_root,
                out_reference_root,
                args.n_repeats,
            )
            if False: #anchor_report_path.exists() and not args.force_rebuild:#todo
                print(f"{scenario_id}: existing baseline vs reference report found at {anchor_report_path} → skipping") 
                #logger.info(f"{scenario_id}: anchor comparison already computed → skipping")
            elif baseline_vs_reference_requirements["ok"]:
                logger.info(f"{scenario_id}: analyze_comparison baseline vs reference")
                analyze_comparison(
                    baseline_root=out_baseline_root,
                    target_root=out_reference_root,
                    output_dir=anchor_compare_dir,
                    label_target="reference",
                    is_llm_run=False,
                    n_repeats=args.n_repeats,
                )
            else:
                logger.warning(
                    f"{scenario_id}: skipping baseline_vs_reference; "
                    f"missing={baseline_vs_reference_requirements['missing']}"
                )

        # compute_rel_scores
        scenario_rel_scores = None
        if (llm_comparison_report is not None
                and reference_enabled
                and (anchor_compare_dir / "report.json").exists()
            ):

            logger.info(f"{scenario_id}: compute_rel_scores")
            anchor_report_data  = load_json(anchor_compare_dir / "report.json")
            scenario_rel_scores = compute_rel_scores(llm_comparison_report, anchor_report_data)
            if scenario_rel_scores is not None:
                save_json(rel_scores_path, scenario_rel_scores)
                logger.info(
                    f"{scenario_id}: rel_scores saved "
                    f"(metrics: {list(scenario_rel_scores['metrics'].keys())})"
                )
            else:
                logger.warning(
                    f"{scenario_id}: compute_rel_scores returned None "
                    "(no common stable workloads or patch/test failure)"
                )

        elif llm_comparison_report is not None and not reference_enabled:
            # No reference — save absolute-only results so quality metrics still flow downstream
            logger.info(f"{scenario_id}: no anchor available, saving absolute-only rel_scores")
            llm_metric_decisions = llm_comparison_report.get("metric_decisions") or {}
            scenario_rel_scores  = {
                "patch_ok":                 llm_comparison_report.get("patch_ok", True),
                "tests_ok":                 llm_comparison_report.get("tests_ok", True),
                "any_rel_score_computable": False,
                "metrics": {
                    metric_name: {
                        "harmonic_mean_rel_score_reliable": None,
                        "mean_rel_score_reliable":          None,
                        "std_rel_score_reliable":           None,
                        "n_workloads_scored_reliable":      0,
                        "harmonic_mean_rel_score_stable":   None,
                        "mean_rel_score_stable":            None,
                        "std_rel_score_stable":             None,
                        "n_workloads_scored_stable":        0,
                        "reason":                           "no_anchor",
                        "n_total_workloads":                metric_decision.get("n_total_workloads"),
                        "n_stable":                         metric_decision.get("n_stable"),
                        "n_improved":                       metric_decision.get("n_improved"),
                        "n_regressed":                      metric_decision.get("n_regressed"),
                        "n_insufficient_signal":            metric_decision.get("n_insufficient_signal"),
                        "stable_workload_coverage_pct":     metric_decision.get("stable_workload_coverage_pct"),
                        "harmonic_mean_snr_reliable":       metric_decision.get("harmonic_mean_snr_reliable"),
                        "harmonic_mean_snr_stable":         metric_decision.get("harmonic_mean_snr_stable"),
                        "harmonic_mean_ratio_reliable":     metric_decision.get("harmonic_mean_ratio_reliable"),
                        "mean_ratio_reliable":              metric_decision.get("mean_ratio_reliable"),
                        "std_ratio_reliable":               metric_decision.get("std_ratio_reliable"),
                        "harmonic_mean_ratio_stable":       metric_decision.get("harmonic_mean_ratio_stable"),
                        "mean_ratio_stable":                metric_decision.get("mean_ratio_stable"),
                        "std_ratio_stable":                 metric_decision.get("std_ratio_stable"),
                        "unit":                             metric_decision.get("unit", ""),
                        "decision":                         metric_decision.get("decision"),
                    }
                    for metric_name, metric_decision in llm_metric_decisions.items()
                },
            }
            save_json(rel_scores_path, scenario_rel_scores)

        # ── Summary ───────────────────────────────────────────────────────
        scenario_summary: Dict[str, Any] = {
            "scenario_id":            scenario_id,
            "repo":                   repo,
            "pr_number":              int(pr_number or 0),
            "baseline_sha":           baseline_sha,
            "reference_sha":          reference_sha if reference_enabled else None,
            "run_id":                 experiment_id,
            "inference_run_dir":      str(inference_run_dir),
            "inference_instance_dir": str(inf_dir),          # ← was inference_instance_dir
            "paths": {
                "baseline_anchor":        str(out_baseline_root),           # ← fixed
                "reference_anchor":       str(out_reference_root) if out_reference_root else None,  # ← fixed
                "llm_eval":               str(out_llm_root),                # ← fixed
                "comparisons_root":       str(compare_root),                # ← fixed
                "eval_root":              str(eval_root),
                "baseline_vs_llm_dir":    str(compare_bl_dir),              # ← fixed
                "anchor_comparison_dir":  str(anchor_compare_dir),          # ← fixed
                "rel_scores":             str(rel_scores_path) if rel_scores_path.exists() else None,
            },
            "status": {
                "baseline":  _repeated_artifact_status(out_baseline_root, args.n_repeats),   # ← fixed
                "reference": (
                    _repeated_artifact_status(out_reference_root, args.n_repeats)
                    if reference_enabled and out_reference_root else None                      # ← fixed
                ),
                "llm": _repeated_artifact_status(out_llm_root, args.n_repeats),              # ← fixed
            },
            "evaluation": {
                "baseline_vs_reference": None if not reference_enabled else {
                    "attempted":   bool(baseline_vs_reference_requirements and baseline_vs_reference_requirements["ok"]),
                    "report_path": str(anchor_compare_dir / "report.json") if (anchor_compare_dir / "report.json").exists() else None,  # ← fixed
                    "missing":     None if not baseline_vs_reference_requirements else baseline_vs_reference_requirements["missing"],
                },
                "baseline_vs_llm": {
                    "attempted":   bool(baseline_vs_llm_requirements["ok"]),
                    "report_path": str(compare_bl_dir / "report.json") if llm_comparison_report else None,  # ← fixed
                    "missing":     baseline_vs_llm_requirements["missing"],
                },
                "rel_scores": {
                    "computed":                 scenario_rel_scores is not None,
                    "path":                     str(rel_scores_path) if rel_scores_path.exists() else None,
                    "any_rel_score_computable": scenario_rel_scores["any_rel_score_computable"] if scenario_rel_scores else None,
                },
            },
        }

        processed += 1
        save_json(summary_path, scenario_summary)
        logger.info(f"{scenario_id}: done (processed={processed})")

        #if args.remove_image:
            #remove_image(b_tag)
            #if r_tag_final:
                #remove_image(r_tag_final)


if __name__ == "__main__":
    main()