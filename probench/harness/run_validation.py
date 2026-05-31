"""
ProBench harness runner

Inference layout
----------------
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
   - baseline (repeated independent runs)
   - reference (repeated independent runs)
   - llm: reuses baseline image, mounts inference dir as /llm_inference
          and patching happens inside the container entrypoint

3) Evaluates via ReportAnalyzer:
   - repeated benchmark comparison for baseline vs reference
   - correctness comparison is handled inside analyzer

4) Exports:
   - comparisons/<experiment_id>/report.json + csv/xlsx
   - SUMMARY__<experiment_id>.json per scenario
"""

from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

from probench.prep.data_loader import DatasetLoader
from probench.utils.docker_utils import (
    build_or_pull_image,
    execute_variant_container,
    remove_image,
)
from probench.utils import global_config as GC
from probench.utils.io_utils import load_yaml, save_json
from probench.utils.harness_utils import load_experiment_meta
from probench.harness.evaluation.report_analyzer import analyze_pr

logger = logging.getLogger("probench.harness")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

CORRECTNESS_FILE = "correctness_report.json"
BENCHMARK_FILE = "performance_results.json"
DONE_FILE = "DONE"


def safe_id(s: str) -> str:
    return str(s).replace("/", "__").replace(":", "_").strip()


def artifact_status(out_dir: Path) -> Dict[str, Any]:
    c = out_dir / CORRECTNESS_FILE
    b = out_dir / BENCHMARK_FILE
    done_marker = out_dir / DONE_FILE
    status = {
        "correctness_exists": c.exists(),
        "benchmark_exists": b.exists(),
        "container_done": done_marker.exists(),
        "missing": [],
    }
    if not status["correctness_exists"]:
        status["missing"].append(CORRECTNESS_FILE)
    if not status["benchmark_exists"]:
        status["missing"].append(BENCHMARK_FILE)
    status["done_strict"] = status["correctness_exists"] and status["benchmark_exists"]
    status["done_any"] = status["correctness_exists"] or status["benchmark_exists"]
    return status


def repeated_run_dir(root: Path, run_idx: int) -> Path:
    return root / f"run_{run_idx:03d}"


def repeated_artifact_status(root: Path, n_repeats: int) -> Dict[str, Any]:
    runs = []
    done_strict_count = 0
    done_perf_count = 0

    for i in range(n_repeats):
        d = repeated_run_dir(root, i)
        st = artifact_status(d)
        runs.append({
            "run_idx": i,
            "path": str(d),
            **st,
        })
        if st["done_strict"]:
            done_strict_count += 1
        if st["benchmark_exists"]:
            done_perf_count += 1

    return {
        "n_repeats": n_repeats,
        "done_strict_count": done_strict_count,
        "all_done_strict": done_strict_count == n_repeats,
        "done_perf_count": done_perf_count,
        "all_done_perf": done_perf_count == n_repeats,
        "runs": runs,
    }


def eval_requirements(label: str, base_dir: Path, tgt_dir: Path, n_repeats: int) -> Dict[str, Any]:
    base = repeated_artifact_status(base_dir, n_repeats)
    tgt = repeated_artifact_status(tgt_dir, n_repeats)

    # Correctness reports are only generated for LLM runs, not baseline/reference.
    # Use benchmark_exists (performance_results.json) as the completion signal here.
    ok = base["all_done_perf"] and tgt["all_done_perf"]
    missing = {
        "baseline_missing_runs": [
            r["run_idx"] for r in base["runs"] if not r["benchmark_exists"]
        ],
        "target_missing_runs": [
            r["run_idx"] for r in tgt["runs"] if not r["benchmark_exists"]
        ],
    }
    return {"label": label, "ok": ok, "missing": missing}


def main() -> None:
    parser = ArgumentParser(description="ProBench harness runner (repeated-run ReportAnalyzer-based)")

    parser.add_argument(
        "--scenario_id",
        type=str,
        #default="pandas-51635",
        help="Run a single scenario_id only (default: run all)",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--inference_run_dir",
        type=Path,
        default=None,
        help="Path to one inference run folder: inference/<run_id>/ (optional for validation-only mode)",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("data/dataset_base_13_03.json"),
        help="Path to dataset JSON (required if --inference_run_dir is not provided)",
    )
    parser.add_argument(
        "--with-gold-solution",
        action="store_true",
        help="Also run the gold/oracle solution",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("results_important"),
        help="Root folder to store run outputs",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
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
        help="Remove built Docker images after run",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=3,
        help="Number of independent container runs per variant",
    )

    args = parser.parse_args()
    args.force_rebuild = False
    args.remove_image = False

    results_root = Path(args.results_root)
    inference_run_dir = args.inference_run_dir

    if inference_run_dir is not None:
        inference_run_dir = Path(inference_run_dir)
        experiment_meta = load_experiment_meta(inference_run_dir)
        experiment_id = safe_id(experiment_meta["run_id"])
        dataset_path = Path(experiment_meta["prompt_dataset_path"])
    else:
        if args.dataset_path is None:
            parser.error("--dataset_path is required when --inference_run_dir is not provided")
        experiment_meta = {"run_id": "validation", "prompt_dataset_path": str(args.dataset_path)}
        experiment_id = "validation"
        dataset_path = Path(args.dataset_path)

    config = load_yaml(args.config_path)
    GC.initialize(config)

    logger.info(f"Loaded experiment meta: {experiment_meta}")
    results_root.mkdir(parents=True, exist_ok=True)

    evals_run_root = results_root / "evals" / experiment_id
    evals_run_root.mkdir(parents=True, exist_ok=True)
    save_json(evals_run_root / "RUN_META.json", experiment_meta)

    loader = DatasetLoader(str(dataset_path), mode="parametrized")

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

        repo = row.get("repo")
        pr_number = row.get("pr_number") or 0
        baseline_sha = row.get("baseline_sha")
        reference_sha = row.get("reference_sha")
        core_tests = row.get("core_tests") or []

        if not repo or not baseline_sha:
            logger.warning(f"{scenario_id}: missing required fields (repo/baseline_sha), skipping")
            continue

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

        anchors_root = results_root / "anchors" / safe_id(scenario_id)
        anchors_root.mkdir(parents=True, exist_ok=True)

        out_baseline_root = anchors_root / f"baseline__{str(baseline_sha)[:12]}"
        out_reference_root = anchors_root / f"reference__{str(reference_sha)[:12]}" if reference_sha else None
        anchor_compare_dir = anchors_root / "baseline_vs_reference"

        eval_root = evals_run_root / "instances" / safe_id(scenario_id)
        eval_root.mkdir(parents=True, exist_ok=True)

        out_llm = eval_root / "llm"
        out_llm.mkdir(parents=True, exist_ok=True)

        compare_root = eval_root / "comparisons"
        compare_bl_dir = compare_root / "baseline_vs_llm"
        summary_path = eval_root / f"SUMMARY__{experiment_id}.json"

        calibration_root = anchors_root / "calibration"
        calibration_root.mkdir(parents=True, exist_ok=True)
        calibration_path = calibration_root / "calibration.json"

        baseline_rep_status = repeated_artifact_status(out_baseline_root, args.n_repeats)
        reference_rep_status = (
            repeated_artifact_status(out_reference_root, args.n_repeats)
            if out_reference_root else None
        )

        _baseline_done = baseline_rep_status["all_done_perf"]
        _reference_done = (not out_reference_root) or reference_rep_status["all_done_perf"]
        _anchor_comparison_done = (not out_reference_root) or (anchor_compare_dir / "report.json").exists()
        _summary_done = summary_path.exists()
        _calibration_done = calibration_path.exists()

        if (
            _calibration_done
            and _baseline_done
            and _reference_done
            and _anchor_comparison_done
            and _summary_done
            and not args.force_rebuild
        ):
            logger.info(f"{scenario_id}: all artifacts and comparisons done -> skipping entirely")
            processed += 1
            continue

        inf_dir = None
        if inference_run_dir is not None:
            inf_dir = inference_run_dir / "instances" / safe_id(scenario_id)
            completion_path = inf_dir / "completion.txt"
            if not completion_path.exists():
                logger.warning(f"{scenario_id}: missing inference completion at {completion_path}, skipping")
                continue

        docker_meta = row.get("docker", {}) or {}
        b_tag = ((docker_meta.get("baseline") or {}).get("tag")) or f"probench:{safe_id(scenario_id)}-baseline"
        r_tag = ((docker_meta.get("reference") or {}).get("tag")) or f"probench:{safe_id(scenario_id)}-reference"

        b_existing = None if args.force_rebuild else (docker_meta.get("baseline") or {}).get("digest")
        r_existing = None if args.force_rebuild else (docker_meta.get("reference") or {}).get("digest")

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
            force_rebuild=args.force_rebuild,
        )
        
        args.with_gold_solution = True 
        r_tag_final: Optional[str] = None
        if args.with_gold_solution and reference_sha:
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
                logger.exception(f"{scenario_id}: calibration execution failed: {e}, skipping to next scenario")
                processed += 1
                continue
        else:
            logger.info(f"{scenario_id}: reusing calibration at {calibration_path}")

        if out_reference_root:
            for run_idx in range(args.n_repeats):
                out_reference = repeated_run_dir(out_reference_root, run_idx)
                out_reference.mkdir(parents=True, exist_ok=True)
                reference_status = artifact_status(out_reference)

                if args.skip_reference_if_done and reference_status["benchmark_exists"]:
                    logger.info(f"{scenario_id}: reference run_{run_idx:03d} already done -> skipping")
                    continue

                logger.info(f"{scenario_id}: execute reference run_{run_idx:03d}")
                try:
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

        for run_idx in range(args.n_repeats):
            out_baseline = repeated_run_dir(out_baseline_root, run_idx)
            out_baseline.mkdir(parents=True, exist_ok=True)
            baseline_status = artifact_status(out_baseline)

            if args.skip_baseline_if_done and baseline_status["benchmark_exists"]:
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

        report_br = None
        req_br = None
        if b_tag and out_reference_root:
            req_br = eval_requirements(
                "baseline_vs_reference",
                out_baseline_root,
                out_reference_root,
                args.n_repeats,
            )

            if (anchor_compare_dir / "report.json").exists() and not args.force_rebuild:
                logger.info(f"{scenario_id}: baseline_vs_reference anchor already computed -> skipping")
                report_br = True
            elif req_br["ok"]:
                logger.info(f"{scenario_id}: analyze baseline vs reference (anchor-level)")
                report_br = analyze_pr(
                    baseline_root=out_baseline_root,
                    target_root=out_reference_root,
                    output_dir=anchor_compare_dir,
                    label_target="reference",
                    is_llm_run=False,
                    n_repeats=args.n_repeats
                )
            else:
                logger.warning(f"{scenario_id}: skipping baseline_vs_reference; missing={req_br['missing']}")

        summary: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "repo": repo,
            "pr_number": int(pr_number or 0),
            "baseline_sha": baseline_sha,
            "reference_sha": reference_sha if (args.with_gold_solution and reference_sha) else None,
            "run_id": experiment_id,
            "inference_run_dir": str(inference_run_dir) if inference_run_dir else None,
            "inference_instance_dir": str(inf_dir) if inf_dir else None,
            "paths": {
                "baseline_anchor": str(out_baseline_root),
                "reference_anchor": str(out_reference_root) if out_reference_root else None,
                "llm_eval": str(out_llm),
                "comparisons_root": str(compare_root),
                "eval_root": str(eval_root),
                "baseline_vs_llm_root": str(compare_bl_dir),
            },
            "status": {
                "baseline": repeated_artifact_status(out_baseline_root, args.n_repeats),
                "reference": repeated_artifact_status(out_reference_root, args.n_repeats) if out_reference_root else None,
                "llm": artifact_status(out_llm),
            },
            "evaluation": {
                "baseline_vs_reference": None
                if not (b_tag and out_reference_root)
                else {
                    "attempted": bool(req_br and req_br["ok"]),
                    "report_path": str(anchor_compare_dir / "report.json") if report_br else None,
                    "missing": None if not req_br else req_br["missing"],
                },
            },
        }

        save_json(summary_path, summary)
        processed += 1

        if args.remove_image:
            try:
                remove_image(r_tag)
            except Exception as e:
                logger.warning(f"{scenario_id}: failed to remove image {r_tag}: {e}")

    logger.info(f"Finished. Processed scenarios: {processed}")


if __name__ == "__main__":
    main()