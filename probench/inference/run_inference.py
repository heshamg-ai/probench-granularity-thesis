from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from argparse import ArgumentParser

from probench.prep.data_loader import DatasetLoader
from probench.inference.llm_client import get_llm_client

logger = logging.getLogger("probench.inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def safe_id(s: str) -> str:
    return str(s).replace("/", "__").replace(":", "_").strip()

def get_scenario_id(row: Dict[str, Any]) -> str:
    v = row.get("scenario_id")
    return str(v).strip() if v and str(v).strip() else "unknown_instance"

def get_repo(row: Dict[str, Any]) -> str:
    v = row.get("repo")
    return str(v).strip() if v else "unknown_repo"

def compute_run_id(prompt_meta: Dict[str, Any], model: str, temperature: float, top_p: float, tag: str) -> str:
    builder = str(prompt_meta.get("builder") or "unknown_builder")
    granularity = str(prompt_meta.get("granularity") or "unknown_granularity")
    prompt_dataset_id = str(prompt_meta.get("prompt_dataset_id") or "unknown_prompt_dataset_id")
    #patch_kind = str(prompt_meta.get("patch_kind") or "unknown_patch_kind")

    rid = (
        f"{builder}_{granularity}"
        f"__{model}"
        f"__t{temperature}"
        f"__top{top_p}"
        f"__{prompt_dataset_id}"
        #f"__{patch_kind}"
    )
    if tag:
        rid += f"__{tag}"
    return safe_id(rid)


def main() -> None:
    p = ArgumentParser("ProBench inference runner")

    p.add_argument("--dataset_path", 
    default="prompt_bm25_gpt52.json",
    help="Prompt dataset JSON (with 'meta' and 'samples').")

    # Single output argument
    p.add_argument("--out_root", default="inference/inference_gpt5.2_bm25", help="Root output folder for inference runs.")

    # run naming
    p.add_argument("--run_id", default=None, help="Optional override. If not set, computed from dataset meta + model params.")
    p.add_argument("--tag", default=None, help="Optional label appended to computed run_id (e.g., attempt1).")

    # Model
    p.add_argument("--provider", default="openai_chat", choices=["openai_chat", "openai_responses", "ollama", "hf", "gemini", "anthropic", "nvidia_nim", "zhipu", "minimax", "qwen_nim"])
    p.add_argument("--model", 
                   default="gpt-5-2025-08-07")  
                   #default="claude-sonnet-4-6@default")
                   #default="gemini-3.1-flash-lite-preview")
    
    p.add_argument("--endpoint_url",
        #default="https://llm.sdc.siemens.cloud")
        default = "https://openai-aiattack-msa-001333-swedencentral-01-freeexperiment.openai.azure.com/openai/v1")
    
    # Generation
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--api_version", type=str, default="v1beta")

    # Control
    p.add_argument("--retries", type=int, default=10)
    p.add_argument("--sleep_s", type=float, default=0)
    p.add_argument("--max_instances", type=int, default=None)
    p.add_argument("--save_prompt", action="store_true")

    args = p.parse_args()

    loader = DatasetLoader(args.dataset_path, mode="raw")
    prompt_meta = getattr(loader, "meta", {}) or {}

    run_id = args.run_id or compute_run_id(
        prompt_meta=prompt_meta,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        tag=args.tag,
    )

    run_dir = Path(args.out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # run-level meta
    (run_dir / "INFERENCE_META.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "provider": args.provider,
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "prompt_dataset_path": str(args.dataset_path),
                "prompt_dataset_meta": prompt_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    llm_kwargs = {
    "provider": args.provider,
    "model": args.model,
    "endpoint_url": args.endpoint_url,
    "temperature": args.temperature,
    "top_p": args.top_p,
    "max_tokens": args.max_tokens,
    }

    if args.provider == "gemini":
        llm_kwargs["api_version"] = args.api_version

    llm = get_llm_client(**llm_kwargs)

    processed = 0

    for row in loader:
        scenario_id = get_scenario_id(row)
        repo = get_repo(row)

        prompt = (row.get("text") or "")
        if not prompt.strip():
            logger.warning(f"{scenario_id}: missing row['text'], skipping")
            continue

        inst_dir = run_dir / "instances" / safe_id(scenario_id)
        inst_dir.mkdir(parents=True, exist_ok=True)

        done = inst_dir / "DONE"
        if done.exists():
            continue

        (inst_dir / "row.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.save_prompt:
            (inst_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        last_err: Optional[str] = None
        for attempt in range(1, args.retries + 1):
            try:
                t0 = time.time()
                completion = llm.generate(prompt)
                dt = time.time() - t0

                (inst_dir / "completion.txt").write_text((completion or "").rstrip() + "\n", encoding="utf-8")

                (inst_dir / "meta.json").write_text(
                    json.dumps(
                        {
                            "scenario_id": scenario_id,
                            "repo": repo,
                            "seconds": dt,
                            "provider": args.provider,
                            "model": args.model,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "prompt_dataset_path": str(args.dataset_path),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                done.write_text("ok\n", encoding="utf-8")
                logger.info(f"{scenario_id}: saved -> {inst_dir}")
                break

            except Exception as e:
                last_err = str(e)
                is_rate_limit = "429" in str(e) or "too many requests" in str(e).lower()

                if is_rate_limit:
                    wait = min(30 * (2 ** (attempt - 1)), 300)
                    logger.warning(f"{scenario_id}: rate limited, waiting {wait}s (attempt {attempt}/{args.retries})")
                else:
                    wait = min(2 ** attempt, 30)
                    logger.warning(f"{scenario_id}: failed, waiting {wait}s (attempt {attempt}/{args.retries}): {e}")

                time.sleep(wait)

        if not done.exists():
            (inst_dir / "ERROR.txt").write_text((last_err or "unknown error") + "\n", encoding="utf-8")
            logger.error(f"{scenario_id}: gave up after {args.retries} attempts")


        processed += 1
        if args.max_instances is not None and processed >= args.max_instances:
            break

        if args.sleep_s > 0:
            time.sleep(args.sleep_s)


if __name__ == "__main__":
    main()
