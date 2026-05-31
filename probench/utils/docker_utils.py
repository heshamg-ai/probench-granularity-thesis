# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

from __future__ import annotations
import json, subprocess
import platform
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import os, re
from dataclasses import asdict, is_dataclass

from probench.utils import global_config as GC
from probench.utils.io_utils import save_json

DOCKER_PLATFORM = "linux/amd64"

def _run(cmd: List[str], *, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """Run and capture (no streaming)."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(cmd, text=True, capture_output=True, env=merged_env)

def _run_stream(cmd: List[str], *, env: Optional[Dict[str, str]] = None, log_prefix: str = "") -> subprocess.CompletedProcess:
    """
    Run a command, streaming stdout/stderr live to the console,
    and also collecting output for error reporting.
    """
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    # Stream + capture
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=merged_env,
    )
    captured = []
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            captured.append(line)
            # stream to console
            if log_prefix:
                print(f"{log_prefix} {line}", end="")
            else:
                print(line, end="")
    finally:
        proc.wait()

    return subprocess.CompletedProcess(cmd, proc.returncode, "".join(captured), "")

def remove_image(tag: str) -> None:
    """Remove a Docker image by tag."""
    proc = _run(["docker", "image", "rm", tag])
    if proc.returncode == 0:
        print(f"[docker] removed image {tag}")
    else:
        print(f"[docker] could not remove image {tag}: {proc.stderr.strip()}")

def create_container_tag(*, repo: str, variant: str, llm_name: str = "", base_tag: str = "") -> str:
    """
    Create a consistent Docker image tag
    - If `base_tag` is provided (e.g. probench/pandas-61116:baseline),
      it derives from that.
    - Otherwise builds from repo + variant info.
    """
    if base_tag:
        prefix = base_tag.split(":")[0]
        suffix = f"{variant}-{llm_name}" if llm_name else variant
        return f"{prefix}:{suffix}".lower()

    safe_repo = re.sub(r"[^a-zA-Z0-9_.-]", "-", repo).strip("-._").lower()
    suffix = f"{variant}-{llm_name}" if llm_name else variant
    return f"probench/{safe_repo}:{suffix}"

def build_or_pull_image(
    *,
    tag: str,
    existing_digest: Optional[str] = None,
    force_rebuild: bool = False,
    docker_file: Optional[str] = None,
    build_context: Optional[Path] = None,
    build_args: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """
    pull from registry by digest (immutable) or tag.
    build locally if docker_file provided and no registry image found.
    """
    if not force_rebuild:
        # 1) check if already local by digest
        if existing_digest:
            insp = _run(["docker", "image", "inspect", existing_digest])
            if insp.returncode == 0:
                print(f"[docker] image {existing_digest} already local -> reusing")
                return tag, existing_digest

        # 2) check if already local by tag
        else:
            insp = _run(["docker", "image", "inspect", tag])
            if insp.returncode == 0:
                print(f"[docker] image {tag} already local -> reusing")
                return tag, ""

        # 3) pull by digest — immutable, exact
        pull_ref = f"{tag}@{existing_digest}" if existing_digest else tag
        print(f"[docker] pulling {pull_ref}...")
        proc = _run_stream(
            ["docker", "pull", "--platform", DOCKER_PLATFORM, tag],
            log_prefix="[docker pull]",
        )
        if proc.returncode == 0:
            # retag if pulled by digest so we can reference by tag later
            if existing_digest:
                _run(["docker", "tag", existing_digest, tag])
            return tag, existing_digest or ""

        print(f"[docker] pull failed, falling back to local build...")

    # 4) build locally as fallback
    if not docker_file or not build_context:
        raise RuntimeError(
            f"[docker] could not pull {tag} and no dockerfile provided for local build"
        )

    docker_file_path = (build_context / docker_file).resolve()
    if not docker_file_path.exists():
        raise FileNotFoundError(f"Dockerfile not found: {docker_file_path}")

    cmd = ["docker", "build"]
    if force_rebuild:
        cmd += ["--no-cache", "--pull"]
    cmd += ["-f", str(docker_file_path), "-t", tag]
    for k, v in (build_args or {}).items():
        cmd += ["--build-arg", f"{k}={v}"]
    cmd += [str(build_context)]

    proc = _run_stream(
        cmd,
        env={"DOCKER_BUILDKIT": "1", "BUILDKIT_PROGRESS": "plain"},
        log_prefix="[docker build]",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"[docker build] failed:\n{proc.stdout}")

    insp = _run(["docker", "image", "inspect", tag, "--format", "{{json .Id}}"])
    digest = json.loads(insp.stdout.strip())
    return tag, digest

def launch_docker_container(
    *,
    image: str,
    argv: List[str],
    mounts: Dict[Path, str],
    #cpus: Optional[str] = None,
    cpuset: Optional[str] = None,
    cpuset_mems: Optional[str] = None,
    memory: Optional[str] = None,
    workdir: Optional[str] = "/app",
    env: Dict[str, str] = None,
    interactive: bool = False,
) -> int:
    env = env or {}
    cmd = ["docker", "run", "--rm"]

    if interactive:
        cmd.append("-it")

    for host_path, container_path in mounts.items():
        cmd += ["-v", f"{str(host_path.resolve())}:{container_path}"]

    #if cpus:
        #cmd += ["--cpus", str(cpus)]
    if cpuset:
        cmd += ["--cpuset-cpus", str(cpuset)]
    if memory:
        cmd += ["--memory", str(memory)]

    if workdir:
        cmd += ["-w", workdir]

    full_env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/app",
        "PYTHONHASHSEED": "0",
    }
    full_env.update(env)
    
    for k, v in full_env.items():
        cmd += ["-e", f"{k}={v}"]

    cmd += [image] + argv

    print("[docker run]", " ".join(cmd))
    return subprocess.call(cmd)

def execute_variant_container(
    *,
    image_tag: str,
    output_dir: Path,
    scenario_id: str,
    version: str,  # "baseline" | "reference" | "llm"
    lib_metadata: Dict[str, Any],
    pr_number: int,
    params: Any,                    
    core_tests: List[str],
    llm_inference_dir: Optional[Path] = None,  
    perf_mode: str = "measure",   # "calibrate" | "measure"
    calibration_path: Optional[Path] = None,
) -> None:
    """
    Runs a single scenario version inside a Docker container.

    - Always mounts:
        GC.root -> /app
        output_dir -> /app_results
    - For LLM runs, additionally mounts:
        llm_inference_dir -> /llm_inference
    """
    if perf_mode not in {"calibrate", "measure"}:
        raise ValueError(f"Invalid perf_mode={perf_mode!r}, expected 'calibrate' or 'measure'")

    output_dir.mkdir(parents=True, exist_ok=True)

    if version == "llm" and not llm_inference_dir:
        raise ValueError("version='llm' requires llm_inference_dir")

    if perf_mode == "measure" and calibration_path is None:
        raise ValueError("perf_mode='measure' requires calibration_path")

    # --- params serialization (dict OR dataclass) ---
    if params is None:
        params_json = {}
    elif isinstance(params, dict):
        params_json = params
    elif is_dataclass(params):
        params_json = asdict(params)
    else:
        raise TypeError(f"`params` must be a dict or dataclass, got: {type(params)}")

    scenario_config = {
        "scenario_id": scenario_id,
        "version": version,
        "pr_number": pr_number,
        "seed": GC.seed,
        "measure": GC.measure,
        "results_root": "/app_results",
        "params": params_json,
        "core_tests": core_tests,
        "lib_metadata": lib_metadata,
        "llm_inference_root": "/llm_inference" if llm_inference_dir else None,
        "perf_mode": perf_mode,
        "mounted_calibration_path": (
            "/app_calibration/calibration.json"
            if (perf_mode == "measure" and calibration_path is not None)
            else None
        ),
    }

    # Write config into output dir
    config_path = output_dir / "_inner.json"
    save_json(config_path, scenario_config)

    mounts: Dict[Path, str] = {
        GC.root: "/app",
        output_dir: "/app_results",
    }

    # LLM-only mount
    if version == "llm":
        mounts[llm_inference_dir] = "/llm_inference"

    # Shared calibration mount for measure mode
    
    if perf_mode == "measure" and calibration_path is not None:
        calibration_path = Path(calibration_path)
        if not calibration_path.exists():
            raise FileNotFoundError(f"Missing calibration file: {calibration_path}")
        mounts[calibration_path] = "/app_calibration/calibration.json"
    
    env = dict(GC.docker.get("env", {}))
    env["PROBENCH_PERF_MODE"] = perf_mode
    if perf_mode == "measure" and calibration_path is not None:
        env["PROBENCH_CALIBRATION_PATH"] = "/app_calibration/calibration.json"

    print(f"[DEBUG] execute_variant_container version={version}")
    print(f"[DEBUG] output_dir={output_dir}")
    print(f"[DEBUG] mounts={mounts}")

    rc = launch_docker_container(
        image=image_tag,
        argv=[
            "python", "-m", "probench.harness.container_entrypoint",
            "--config", "/app_results/_inner.json",
        ],
        mounts=mounts,
        # cpus=GC.docker.get("cpus"),
        cpuset=GC.docker.get("cpuset"),
        cpuset_mems=GC.docker.get("cpuset_mems"),
        memory=GC.docker.get("memory"),
        workdir="/app",
        env=env,
    )

    if rc != 0:
        raise RuntimeError(f"{scenario_id}:{version}:{perf_mode} container run failed (rc={rc})")

