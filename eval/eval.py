#!/usr/bin/env python
"""Frozen evaluation harness for the autoresearch architecture-optimization loop.

This script is the GROUND TRUTH metric (the analog of a sealed evaluator). The
autoresearch agent must NOT edit this file. It evaluates one architecture YAML
against the fixed workload set -- the mamba2 decode cascade swept over a set of
batch sizes -- and reports, per batch size, the energy/latency/EDP, plus the
aggregate headline metric (geomean of per-batch EDP) and the total chip area.

Usage:
    uv run python scripts/eval_arch.py examples/arches/autoresearch_<tag>.yaml
    # optional: override the batch-size sweep
    uv run python scripts/eval_arch.py <arch.yaml> --batches 1,8,32,128

Definitions (frozen here so the agent cannot redefine the metric):
  * EDP(B)        = energy(B) * latency(B)            # Joule-seconds, lower is better
  * agg_edp       = geomean over the batch set of EDP(B)   # the headline metric
  * total_area    = spec.arch.total_area (m^2), reported in mm^2 (soft constraint)
"""

import argparse
import math
import os
import shutil
import statistics
import sys
import tempfile
import traceback
import warnings
import time
from pathlib import Path

import accelforge as af

# Optional per-phase parallelism split to avoid join-phase OOM, WITHOUT changing
# the metric. Set AF_JOIN_JOBS=N (e.g. 1 or 4) to run make_pmappings at full
# parallelism (fast) but the memory-heavy join + detail-eval at N workers (less
# per-worker dataframe duplication => lower peak RSS). The handoff is the pmapping
# cache: make_pmappings is run once to populate a temp cache, then the full
# map_workload_to_arch reuses it as a cache hit (so its internal make does no
# work) and only the join/eval run at the reduced worker count. eval_in_detail is
# left fully intact, so energy()/latency()/area are bit-for-bit identical to the
# default path. AF_JOIN_JOBS unset (or 0) => original behavior, untouched.
_AF_JOIN_JOBS = int(os.environ.get("AF_JOIN_JOBS", "0"))

# The FFM mapper floods stderr with pandas "DataFrame is highly fragmented"
# PerformanceWarnings; silence them so run.log stays readable.
warnings.simplefilter("ignore")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # scripts/ -> repo root

# --- FROZEN experiment definition -------------------------------------------
# The workload is the BATCHED mamba2 decode cascade; batch size B is swept.
# We use the *fusedconv* variant: it expresses the causal conv as
# X_CONV_PRIOR (reduce over the W-1 prior slots) + X_CONV_CUR + add. It is
# MATH-IDENTICAL to the faithful conv-state formulation in the plain
# mamba2_decode_batched.yaml (cross-checked), which does NOT run here -- that one
# has a rank-`w` bounds clash AND hits a copy-einsum `_backing` framework bug on
# the conv concats. Cost difference is <~0.5% (decode is HBM-weight-read-bound;
# the conv tensors are W=4-wide and tiny), so fusedconv is an equivalent
# stand-in, not a cheaper design.
# WORKLOAD = REPO_ROOT / "examples" / "workloads" / "mamba2_decode.yaml"
WORKLOAD = REPO_ROOT / "examples" / "workloads" / "tropical_gemm.yaml"
# WORKLOAD = REPO_ROOT / "examples" / "workloads" / "gemm.yaml"
# WORKLOAD = REPO_ROOT / "examples" / "workloads" / "matvecs.yaml"

DEFAULT_BATCHES = [1, 8, 32, 64]

# ----------------------------------------------------------------------------


def evaluate_one(arch_path: Path, batch: int) -> tuple[float, float, float]:
    """Map the workload at batch size `batch` onto `arch_path`; return
    (energy_J, latency_s, area_m2). Raises on any failure."""
    spec = af.Spec.from_yaml(str(arch_path), str(WORKLOAD), B=batch)
    # spec.mapper.metrics = af.Metrics.ENERGY | af.Metrics.LATENCY
    # # FROZEN map-space reduction -- identical for every arch and every batch, so
    # # comparisons are fair. Three pieces:
    # #   * explore_loop_orders = False, max_fused_loops = 0 -- structural freeze.
    # #     Fusion off keeps per-einsum costs additive (a correctness requirement of
    # #     the mixed-reuse-tax decomposition); loop-order exploration is a no-op for
    # #     this decode anyway (size-1 loops have no partial reuse).
    # #   * max_pmapping_templates_per_einsum = 64 -- tractability backstop. The
    # #     default FFM search is unbounded and never finishes on this ~40-einsum
    # #     cascade (instantiating all ~15.8k templates takes >13 min); the cap
    # #     bounds how many structural templates get *instantiated*. 64 (vs the old
    # #     16) gives ~4x more dataflow coverage, so the cut rarely binds when the
    # #     mapping actually matters, while still mapping in ~tens of seconds/batch.
    # #   * objective_tolerance = 0.01 -- epsilon-Pareto. This is the principled
    # #     quality lever (it replaces the cap's former role as a blind first-N
    # #     cut): the returned mapping is guaranteed within <=1% of optimal, while
    # #     near-duplicate Pareto points are thinned (which also shrinks the join).
    # #     Deterministic and identical across arches/batches, so comparisons stay
    # #     fair. NOTE: epsilon prunes the surviving Pareto population, NOT how many
    # #     templates are instantiated -- hence the cap is still needed for tractability.
    # spec.mapper.explore_loop_orders = False
    # spec.mapper.max_fused_loops = 0
    # spec.mapper.max_pmapping_templates_per_einsum = 64
    # spec.mapper.objective_tolerance = 0.01

    # Area is a property of the architecture (fanout-aware), independent of the
    # chosen mapping. Compute it from a populated copy of the spec.
    area_m2 = float(spec.calculate_component_costs().arch.total_area)

    # if _AF_JOIN_JOBS > 0:
    #     # Split parallelism via the pmapping cache (see header note). make at full
    #     # parallelism, join + eval_in_detail at the reduced worker count.
    #     from accelforge.mapper.FFM.main import (
    #         make_pmappings,
    #         map_workload_to_arch as _ffm_map,
    #     )

    #     _cache = tempfile.mkdtemp(prefix="af_pmap_")
    #     try:
    #         af.set_n_parallel_jobs(os.cpu_count())
    #         make_pmappings(
    #             spec,
    #             cache_dir=_cache,
    #             print_progress=False,
    #             print_number_of_pmappings=True,
    #             one_pbar_only=True,
    #         )
    #         af.set_n_parallel_jobs(_AF_JOIN_JOBS)
    #         mappings = _ffm_map(
    #             spec,
    #             cache_dir=_cache,
    #             print_progress=False,
    #             print_number_of_pmappings=True,
    #             one_pbar_only=True,
    #         )
    #     finally:
    #         af.set_n_parallel_jobs(os.cpu_count())  # restore default
    #         shutil.rmtree(_cache, ignore_errors=True)
    # else:
    #     mappings = spec.map_workload_to_arch(print_progress=False, print_number_of_pmappings=True, one_pbar_only=True)
    # if not mappings:
    #     raise RuntimeError(f"no valid mapping found for B={batch}")

    mappings = spec.map_workload_to_arch(print_progress=False, print_number_of_pmappings=True, one_pbar_only=True)
    m = mappings[0]  # Pareto-optimal under ENERGY|LATENCY
    return float(m.energy()), float(m.latency()), area_m2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arch", type=Path, help="path to the architecture YAML to evaluate")
    ap.add_argument(
        "--batches",
        type=lambda s: [int(x) for x in s.split(",")],
        default=DEFAULT_BATCHES,
        help="comma-separated batch sizes (default: %s)" % ",".join(map(str, DEFAULT_BATCHES)),
    )
    args = ap.parse_args()

    arch_path = args.arch.resolve()
    if not arch_path.is_file():
        print(f"EVAL FAILED: arch file not found: {arch_path}", file=sys.stderr)
        return 2

    edps: list[float] = []
    total_energy = 0.0
    total_latency = 0.0
    area_m2 = math.nan

    try:
        for b in args.batches:
            time_start = time.time()
            energy, latency, area_m2 = evaluate_one(arch_path, b)
            time_end = time.time()
            edp = energy * latency
            edps.append(edp)
            total_energy += energy
            total_latency += latency
            time_duration = time_end - time_start
            print("---")
            print(f"workload:        {WORKLOAD.stem} (B={b})")
            print(f"energy_J:        {energy:.6e}")
            print(f"latency_s:       {latency:.6e}")
            print(f"edp:             {edp:.6e}")
            print(f"time_duration:   {time_duration:.6e} s")
    except Exception:  # noqa: BLE001 -- any failure => crash for the loop
        # Print the traceback but DO NOT print an `agg_edp:` line, so that
        # `grep "^agg_edp:" run.log` is empty and the loop records a crash.
        print("EVAL FAILED:", file=sys.stderr)
        traceback.print_exc()
        return 1

    agg_edp = statistics.geometric_mean(edps)
    area_mm2 = area_m2 * 1e6
    # throughput per area: total sequences processed (sum of batch sizes) per
    # second of summed latency, divided by chip area. A benefit/cost ratio
    # (perf you GET vs area you SPEND) -- higher is better, and unlike an
    # EDP/area metric it actually exposes the perf-vs-area tradeoff.
    throughput = sum(args.batches) / total_latency  # sequences / second
    # perf_per_area = throughput / area_mm2  # sequences / second / mm^2
    print("---")
    # print(f"agg_edp:         {agg_edp:.6e}")  # geomean of per-batch EDP -- headline metric
    # print(f"total_energy_J:  {total_energy:.6e}")  # summed across the batch set
    # print(f"total_latency_s: {total_latency:.6e}")  # summed across the batch set
    print(f"total_area_mm2:  {area_mm2:.4f}")  # m^2 -> mm^2 (soft constraint)
    # print(f"perf_per_area:   {perf_per_area:.6e}")  # throughput/area = sum(B)/(total_latency_s*area_mm2) -- higher is better
    return 0


if __name__ == "__main__":
    raise SystemExit(main())