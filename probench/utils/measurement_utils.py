# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2025, Siemens AG

from __future__ import annotations

import gc
import math
import os
import platform
import statistics
import threading
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Static info ───────────────────────────────────────────────────────────────

_PLATFORM_INFO: Dict[str, Any] = {
    "system":    platform.system(),
    "release":   platform.release(),
    "python":    platform.python_version(),
    "cpu_count": os.cpu_count() or 1,
}

# ── Aggregation helpers ───────────────────────────────────────────────────────

def _extract(per: List[Dict[str, Any]], key: str) -> List[float]:
    return [float(x[key]) for x in per if isinstance(x.get(key), (int, float))]


def _average(vals: List[float]) -> Optional[float]:
    return (sum(vals) / len(vals)) if vals else None


def _median(vals: List[float]) -> Optional[float]:
    return statistics.median(vals) if vals else None


def _std(vals: List[float]) -> Optional[float]:
    return statistics.stdev(vals) if len(vals) >= 2 else None


def _cv(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    mean_v = statistics.mean(vals)
    return (statistics.stdev(vals) / mean_v) if mean_v != 0 else None


def _rciw(vals: List[float]) -> Optional[float]:
    """
    Relative Confidence Interval Width (RCIW) for the mean.

        RCIW = (2 * 1.96 * std / sqrt(n)) / mean

    Z = 1.96 for 95% confidence.
    Result is a decimal fraction, e.g. 0.10 means 10%.
    Stoppage criterion: RCIW <= rciw_threshold (e.g. 0.10 for 10%).
    """
    if len(vals) < 2:
        return None
    mean_v = statistics.mean(vals)
    if mean_v == 0:
        return None
    return (2 * 1.96 * _std(vals) / math.sqrt(len(vals))) / mean_v


# ── Tracemalloc helpers ───────────────────────────────────────────────────────

def _tracemalloc_sample_worker(
    stop: threading.Event,
    samples: List[Tuple[float, float, float]],  # (elapsed_s, current_mib, peak_mib)
    interval_s: float,
    t_start: float,
) -> None:
    """
    Background sampler that reads tracemalloc current+peak at interval_s.
    tracemalloc must already be started by the caller before spawning this thread.
    Appends (elapsed_s, current_mib, peak_mib) tuples.
    """
    while not stop.wait(interval_s):
        try:
            current_b, peak_b = tracemalloc.get_traced_memory()
            elapsed = time.perf_counter() - t_start
            samples.append((elapsed, current_b / (1024**2), peak_b / (1024**2)))
        except Exception:
            pass


def _tracemalloc_auc_normalized(
    samples: List[Tuple[float, float, float]],
    baseline_current_mib: float,
) -> Tuple[float, float]:
    """
    Compute Time-Weighted Memory Usage (TWMU) from tracemalloc samples.

    Returns
    -------
    auc_mib_s : float
        Raw area under the current-memory curve (MiB·s), baseline-subtracted.
    auc_normalized_mib : float
        auc_mib_s divided by total elapsed time — the mean memory above baseline
        over the call duration. This is the canonical memory metric (TWMU).
    """
    if len(samples) < 2:
        return 0.0, 0.0

    pts = sorted(
        (float(t), max(0.0, float(cur) - baseline_current_mib))
        for t, cur, _ in samples
        if float(t) >= 0.0
    )

    auc = 0.0
    for (t0, y0), (t1, y1) in zip(pts, pts[1:]):
        dt = t1 - t0
        if dt > 0.0:
            auc += 0.5 * (y0 + y1) * dt

    total_time = pts[-1][0] - pts[0][0]
    normalized = auc / total_time if total_time > 0.0 else 0.0

    return auc, normalized


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_memory_runtime(
    function: Callable[[], None],
    measure_cfg: Dict[str, Any],
    pre_hook: Callable[[], None],
    restore_hook: Optional[Callable[[], None]],
) -> Dict[str, Any]:
    """
    Calibrates runtime_invocations and tm_sample_interval_s for a given workload.

    Two goals:
      1. runtime_invocations: how many fn() calls per sample so each sample
         takes approximately target_sample_time_s seconds.
      2. tm_sample_interval_s: tracemalloc polling interval so we get
         approximately tm_target_samples memory snapshots per fn() call.

    Called once per workload (per param combination) before measure_adaptive().
    """
    min_probe_time_s        = float(measure_cfg["min_probe_time_s"])
    max_runtime_invocations = int(measure_cfg["max_runtime_invocations"])
    min_runtime_invocations = int(measure_cfg["min_runtime_invocations"])
    target_s                = float(measure_cfg["target_sample_time_s"])
    safety_factor           = float(measure_cfg["safety_factor"])
    tm_target_samples       = int(measure_cfg.get("tm_target_samples"))
    tm_default_interval_s = float(measure_cfg.get("tm_default_interval_s"))
    tm_min_interval_s       = float(measure_cfg.get("tm_min_interval_s"))
    tm_max_interval_s       = float(measure_cfg.get("tm_max_interval_s"))
    calibrate_warmup_runs   = int(measure_cfg.get("calibrate_warmup_runs", 3))
    min_probe_runs          = int(measure_cfg.get("min_probe_runs", 3))
    max_probe_runs          = int(measure_cfg.get("max_probe_runs", 10))

    # ── warmup — discard results, just heat up caches ─────────────────────────
    pre_hook()
    for _ in range(max(1, calibrate_warmup_runs)):
        try:
            if restore_hook is not None:
                restore_hook()
            function()
        except Exception:
            pass

    # ── probe — measure until min_probe_runs AND min_probe_time_s both met ────
    probe_runs    = 0
    total_elapsed = 0.0
    single_runs:  List[float] = []

    while probe_runs < min_probe_runs or total_elapsed < min_probe_time_s:
        if probe_runs >= max_probe_runs:
            break
        try:
            if restore_hook is not None:
                restore_hook()
            t0 = time.perf_counter()
            function()
            dt = time.perf_counter() - t0
            total_elapsed += dt
            single_runs.append(dt)
            probe_runs += 1
        except Exception:
            pass

    if not single_runs:
        single_runs   = [1e-9]
        total_elapsed = 1e-9
        probe_runs    = 1

    representative_run_s = _median(single_runs) or 1e-9

    # ── compute runtime_invocations ───────────────────────────────────────────
    runtime_invocations = math.ceil((target_s * safety_factor) / representative_run_s)
    runtime_invocations = max(min_runtime_invocations,
                          min(runtime_invocations, max_runtime_invocations))

    at_max = runtime_invocations == max_runtime_invocations
    if at_max:
        print(
            f"[WARNING] calibrate: runtime_invocations capped at {max_runtime_invocations}. "
            f"representative_run_s={representative_run_s:.2e}s — "
            f"fn() may be too fast to measure reliably."
        )

    # ── compute tm_sample_interval_s ────────────────────────────────────────
    micro_interval_s = representative_run_s / max(tm_target_samples, 1)
    calibrated_tm_interval_s = min(
                                tm_default_interval_s,
                                micro_interval_s)

    calibrated_tm_interval_s = max(
                                tm_min_interval_s,
                                calibrated_tm_interval_s)

    return {
        "runtime_invocations":            runtime_invocations,
        "tm_sample_interval_s":           calibrated_tm_interval_s,
        "representative_run_s":           representative_run_s,
        "runtime_list":                   single_runs,
        "probe_runs_used":                probe_runs,
        "probe_elapsed_s":                total_elapsed,
        "runtime_invocations_at_maximum": at_max,
        "status":                         "ok",
    }


# ── Warmup ────────────────────────────────────────────────────────────────────

def _warmup(
    fn: Callable[[], Any],
    *,
    measure_cfg: Dict[str, Any],
    per_iter_prepare_hook: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """
    Blind per-call convergence loop using CV as stoppage criterion.
    Returns diagnostics dict.

    CV is intentionally used here (not RCIW) because warmup results are
    discarded — we only need a cheap proxy for thermal steady state.
    RCIW is reserved for the actual measurement loop in measure_adaptive().
    """
    window    = int(measure_cfg.get("warmup_window"))
    min_calls = int(measure_cfg.get("warmup_min_calls"))
    max_calls = int(measure_cfg.get("warmup_max_calls"))
    cv_target = float(measure_cfg.get("warmup_cv_threshold"))

    call_times: List[float] = []
    converged = False

    for call_idx in range(max_calls):
        if per_iter_prepare_hook is not None:
            per_iter_prepare_hook()
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as exc:
            print(f"[WARN] warmup call {call_idx} failed: {type(exc).__name__}: {exc}")
            continue
        call_times.append(time.perf_counter() - t0)

        enough_calls      = len(call_times) >= min_calls
        enough_for_window = len(call_times) >= window
        if enough_calls and enough_for_window:
            cv = _cv(call_times[-window:])  # always check last `window` samples
            if cv is not None and cv < cv_target:
                converged = True
                break

    return {
        "converged":     converged,
        "calls_run":     len(call_times),
        "final_cv":      _cv(call_times[-window:]) if len(call_times) >= window else None,
        "mean_call_s":   _average(call_times),
        "median_call_s": _median(call_times),
        "call_times_s":  call_times,
    }


# ── Single measurement ────────────────────────────────────────────────────────
#
# Two-phase design:
#
# Phase 1 — Runtime
#   runtime_invocations × [ hook (untimed) | t0 | fn() | t1 ]
#   GC disabled for the whole batch for clean wall-clock numbers.
#
# Phase 2 — tracemalloc (single call + background sampler thread)
#   One isolated fn() call. tracemalloc adds 5–15% overhead so it is
#   kept strictly separate from Phase 1.
#   Canonical memory metrics:
#     - memory_twmu_mib             — time-weighted mean MiB above baseline (TWMU)
#     - memory_final_delta_peak_mib — peak MiB above baseline

def _measure_once(
    fn: Callable[[], Any],
    *,
    runtime_invocations: int,
    per_iter_prepare_hook: Optional[Callable[[], None]] = None,
    tm_sample_interval_s: float,
) -> Dict[str, Any]:

    gc_was_enabled = gc.isenabled()

    # ── Phase 1: Runtime ──────────────────────────────────────────────────────
    runtime_invocations_s: List[float] = []
    runtime_iteration_s:   float = 0.0

    try:
        gc.collect()
        gc.disable()

        for _ in range(runtime_invocations):
            if per_iter_prepare_hook is not None:
                per_iter_prepare_hook()
            t0 = time.perf_counter()
            fn()
            runtime_invocations_s.append(time.perf_counter() - t0)

        runtime_iteration_s = sum(runtime_invocations_s)
    finally:
        if gc_was_enabled:
            gc.enable()

    runtime_invocation_median_s = statistics.median(runtime_invocations_s)

    # ── Phase 2: tracemalloc ──────────────────────────────────────────────────
    if per_iter_prepare_hook is not None:
        per_iter_prepare_hook()

    stop_tm_event       = threading.Event()
    tracemalloc_samples: List[Tuple[float, float, float]] = []

    tracemalloc.start()
    tracemalloc.clear_traces()
    tracemalloc.reset_peak()

    t0_tm = time.perf_counter()
    start_cur_b, start_peak_b = tracemalloc.get_traced_memory()
    baseline_cur_mib = start_cur_b / (1024 ** 2)
    tracemalloc_samples.append((0.0, baseline_cur_mib, start_peak_b / (1024 ** 2)))

    tm_thread = threading.Thread(
        target=_tracemalloc_sample_worker,
        args=(stop_tm_event, tracemalloc_samples, tm_sample_interval_s, t0_tm),
        daemon=True,
    )
    tm_thread.start()

    try:
        fn()
    finally:
        phase_duration_s = time.perf_counter() - t0_tm
        stop_tm_event.set()
        tm_thread.join(timeout=2.0)

        try:
            after_b, peak_b = tracemalloc.get_traced_memory()
            after_cur_mib   = after_b / (1024 ** 2)
            after_peak_mib  = peak_b  / (1024 ** 2)
        except Exception:
            after_b = peak_b = 0
            after_cur_mib = after_peak_mib = 0.0

        tracemalloc_samples.append((phase_duration_s, after_cur_mib, after_peak_mib))
        tracemalloc.stop()
        gc.collect()

    auc_mib_s, twmu_mib = _tracemalloc_auc_normalized(tracemalloc_samples, baseline_cur_mib)

    return {
        # ── runtime ───────────────────────────────────────────────────────────
        "runtime_iteration_s":           runtime_iteration_s,          # total batch time
        "runtime_invocation_median_s":   runtime_invocation_median_s,  # median single-call time
        "runtime_invocations_s":         runtime_invocations_s,        # all per-invocation times

        # ── memory — canonical metrics (what we report) ───────────────────────
        "memory_twmu_mib":               twmu_mib,
        "memory_final_delta_peak_mib":   max(0.0, after_peak_mib - baseline_cur_mib),
        "memory_final_delta_current_mib": max(0.0, after_cur_mib - baseline_cur_mib),
        "memory_auc_mib_s":              auc_mib_s,

        # ── tracemalloc — implementation internals ────────────────────────────
        "tracemalloc_duration_s":           phase_duration_s,
        "tracemalloc_baseline_current_mib": baseline_cur_mib,
        "tracemalloc_final_current_mib":    after_cur_mib,
        "tracemalloc_final_peak_mib":       after_peak_mib,
        "tracemalloc_samples":              tracemalloc_samples,
    }


def measure_adaptive(
    function: Callable[[], Any],
    measure_cfg: Dict[str, Any],
    runtime_invocations: int,
    tm_sample_interval_s: float,
    pre_measure_hook: Optional[Callable[[], None]],
    per_iter_prepare_hook: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """
    Adaptive sampling loop.

    Stoppage criterion: RCIW on all three metrics must converge:
        - runtime : RCIW(runtime_invocation_median_s) <= rciw_threshold
        - TWMU    : RCIW(memory_twmu_mib)             <= rciw_threshold
        - peak    : RCIW(memory_final_delta_peak_mib) <= rciw_threshold

    Returns raw series only — all statistics (mean, median, std, etc.)
    are computed downstream in the report analyser.
    Config key: rciw_threshold (decimal, e.g. 0.10 = 10%)
    """

    if pre_measure_hook is not None:
        pre_measure_hook()

    warmup_diag = _warmup(
        function,
        measure_cfg=measure_cfg,
        per_iter_prepare_hook=per_iter_prepare_hook,
    )

    min_samples      = int(measure_cfg["min_samples"])
    max_samples      = int(measure_cfg["max_samples"])
    max_total_time_s = float(measure_cfg["max_total_time_s"])
    rciw_threshold   = float(measure_cfg["rciw_threshold"])

    samples: List[Dict[str, Any]] = []
    t_start     = time.perf_counter()
    stop_reason: Optional[str] = None

    while True:
        sample = _measure_once(
            function,
            runtime_invocations=runtime_invocations,
            per_iter_prepare_hook=per_iter_prepare_hook,
            tm_sample_interval_s=tm_sample_interval_s,
        )
        samples.append(sample)

        n       = len(samples)
        elapsed = time.perf_counter() - t_start

        if n >= max_samples:
            stop_reason = "max_samples"
            break

        if n >= min_samples:
            rciw_runtime = _rciw(_extract(samples, "runtime_invocation_median_s"))
            rciw_twmu    = _rciw(_extract(samples, "memory_twmu_mib"))
            rciw_peak    = _rciw(_extract(samples, "memory_final_delta_peak_mib"))

            if (rciw_runtime is not None and rciw_runtime <= rciw_threshold and
                rciw_twmu    is not None and rciw_twmu    <= rciw_threshold and
                rciw_peak    is not None and rciw_peak    <= rciw_threshold):
                stop_reason = "rciw_converged"
                break

        if n >= min_samples and elapsed >= max_total_time_s:
            stop_reason = "max_total_time_s"
            break

    return {
        # ── runtime series ────────────────────────────────────────────────────
        "runtime_iteration_s":           _extract(samples, "runtime_iteration_s"),
        "runtime_invocation_median_s":   _extract(samples, "runtime_invocation_median_s"),
        "runtime_invocations_s":         [s["runtime_invocations_s"] for s in samples],

        # ── memory — canonical metrics (what we report) ───────────────────────
        "memory_twmu_mib":               _extract(samples, "memory_twmu_mib"),
        "memory_final_delta_peak_mib":   _extract(samples, "memory_final_delta_peak_mib"),
        "memory_final_delta_current_mib": _extract(samples, "memory_final_delta_current_mib"),
        "memory_auc_mib_s":              _extract(samples, "memory_auc_mib_s"),

        # ── tracemalloc — implementation internals ────────────────────────────
        "tracemalloc_duration_s":           _extract(samples, "tracemalloc_duration_s"),
        "tracemalloc_baseline_current_mib": _extract(samples, "tracemalloc_baseline_current_mib"),
        "tracemalloc_final_current_mib":    _extract(samples, "tracemalloc_final_current_mib"),
        "tracemalloc_final_peak_mib":       _extract(samples, "tracemalloc_final_peak_mib"),
        "tracemalloc_samples":              [s["tracemalloc_samples"] for s in samples],

        # ── metadata ─────────────────────────────────────────────────────────
        "sample_count":        len(samples),
        "runtime_invocations": runtime_invocations,
        "warmup":              warmup_diag,
        "sampling": {
            "stop_reason":        stop_reason,
            "elapsed_s":          time.perf_counter() - t_start,
            "min_samples":        min_samples,
            "max_samples":        max_samples,
            "max_total_time_s":   max_total_time_s,
            "rciw_threshold":     rciw_threshold,
            "final_rciw_runtime": _rciw(_extract(samples, "runtime_invocation_median_s")),
            "final_rciw_twmu":    _rciw(_extract(samples, "memory_twmu_mib")),
            "final_rciw_peak":    _rciw(_extract(samples, "memory_final_delta_peak_mib")),
        },
        "meta": {
            "runtime_invocations":  runtime_invocations,
            "tm_sample_interval_s": tm_sample_interval_s,
            "min_samples":          min_samples,
            "max_samples":          max_samples,
            "max_total_time_s":     max_total_time_s,
            "rciw_threshold":       rciw_threshold,
        },
        "platform": _PLATFORM_INFO,
    }