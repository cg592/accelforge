#!/usr/bin/env python

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
REPO_ROOT = HERE.parent 

# ----------------------------------------------------------------------------

def evaluate_one(arch_path: Path, workload_path: Path, batch: int) -> tuple[float, float, float]:
    """Map the workload at batch size `batch` onto `arch_path`; return
    (energy_J, latency_s, area_m2). Raises on any failure."""
    spec = af.Spec.from_yaml(str(arch_path), str(workload_path), B=batch)
    area_m2 = float(spec.calculate_component_costs().arch.total_area)
    mappings = spec.map_workload_to_arch(print_progress=False, print_number_of_pmappings=True, one_pbar_only=True)
    m = mappings[0] 
    return float(m.energy()), float(m.latency()), area_m2

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", type=Path, required=True, help="path to the architecture YAML to evaluate")
    ap.add_argument("--workload", type=Path, required=True, help="path to the workload YAML")
    ap.add_argument(
        "--batches",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1], #[1, 8, 32, 64]
        help="comma-separated batch sizes (default: 1)",
    )
    args = ap.parse_args()

    arch_path = args.arch.resolve()
    if not arch_path.is_file():
        print(f"EVAL FAILED: arch file not found: {arch_path}", file=sys.stderr)
        return 2

    workload_path = args.workload.resolve()
    if not workload_path.is_file():
        print(f"EVAL FAILED: workload file not found: {workload_path}", file=sys.stderr)
        return 2

    edps: list[float] = []
    total_energy = 0.0
    total_latency = 0.0
    area_m2 = math.nan

    try:
        for b in args.batches:
            time_start = time.time()
            energy, latency, area_m2 = evaluate_one(arch_path, workload_path, b)
            time_end = time.time()
            edp = energy * latency
            edps.append(edp)
            total_energy += energy
            total_latency += latency
            time_duration = time_end - time_start
            print("---")
            print(f"workload:        {workload_path.stem} (B={b})")
            print(f"energy_J:        {energy:.6e}")
            print(f"latency_s:       {latency:.6e}")
            print(f"edp:             {edp:.6e}")
            print(f"time_duration:   {time_duration:.6e} s")
    except Exception:  
        print("EVAL FAILED:", file=sys.stderr)
        traceback.print_exc()
        return 1

    agg_edp = statistics.geometric_mean(edps)
    area_mm2 = area_m2 * 1e6
    throughput = sum(args.batches) / total_latency  
    print("---")
    print(f"total_area_mm2:  {area_mm2:.4f}") 
    return 0

if __name__ == "__main__":
    raise SystemExit(main())