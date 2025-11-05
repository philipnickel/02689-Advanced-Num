#!/usr/bin/env python3
"""
Profiling Analysis: Bottleneck Shift with Grid Size
====================================================

Profiles the KdV solver at small, moderate, and large N to show how
computational bottlenecks shift as N increases.

Expected Observations:
- Small N (256):    Python overhead + O(1) constants dominate
- Moderate N (2048): Mix of FFT and element-wise operations
- Large N (16384):   FFT operations become dominant (N log N)

This demonstrates why overall scaling exponent α improves with N.
"""

import cProfile
import pstats
import io
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, soliton, RK4

# Configuration
DATA_DIR = Path("data/A2/ex_g")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Test at three representative grid sizes
N_VALUES = {
    "Small": 256,      # Small N: overhead dominates
    "Moderate": 2048,  # Moderate N: mixed regime
    "Large": 16384,    # Large N: FFT dominates
}

# Test parameters
L = 30.0
c = 0.5
x0 = 0.0
SAFETY_FACTOR = 0.1
N_STEPS = 1000  # Fixed number of steps for fair comparison


def estimate_stable_dt(N: int, L: float, c: float) -> float:
    """Estimate stable time step."""
    solver = KdVSolver(N, L)
    u = soliton(solver.x, 0.0, c, x0)
    u_max = float(np.max(np.abs(u)))
    dt_est = KdVSolver.stable_dt(N, L, u_max, integrator_name="rk4")
    return SAFETY_FACTOR * dt_est if np.isfinite(dt_est) else 1e-3


def profile_at_N(N: int, label: str) -> dict:
    """
    Profile solver at given N and extract key statistics.

    Parameters
    ----------
    N : int
        Grid size
    label : str
        Label for this test case ('Small', 'Moderate', 'Large')

    Returns
    -------
    dict
        Profiling statistics
    """
    print(f"\n{label} N = {N}")
    print("-" * 70)

    # Setup
    solver = KdVSolver(N, L, dealias=False)
    integrator = RK4()
    x = solver.x
    u0 = soliton(x, 0.0, c, x0)
    dt = estimate_stable_dt(N, L, c)

    # Warm up (JIT compilation)
    print("  Warming up (JIT compilation)...")
    for _ in range(100):
        _ = integrator.step(solver.rhs, u0, 0.0, dt)

    # Profile
    print(f"  Profiling {N_STEPS} steps...")
    profiler = cProfile.Profile()
    profiler.enable()

    u = u0.copy()
    t = 0.0
    for _ in range(N_STEPS):
        u = integrator.step(solver.rhs, u, t, dt)
        t += dt

    profiler.disable()

    # Extract statistics
    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    # Get total time
    total_time = 0.0
    for func_stats in stats.stats.values():
        total_time += func_stats[3]  # cumtime

    # Find key functions
    function_times = {}

    for func_name, func_stats in stats.stats.items():
        name = func_name[2]  # Function name
        cumtime = func_stats[3]
        ncalls = func_stats[0]

        # Track important functions
        if "execute" in name and "fft" in str(func_name):
            function_times["FFT_execute"] = cumtime
        elif name == "step" and "tdp.py" in str(func_name):
            function_times["RK4_step"] = cumtime
        elif name == "rhs" and "tdp.py" in str(func_name):
            function_times["KdV_rhs"] = cumtime
        elif "_raw_fft" in name:
            function_times["FFT_wrapper"] = cumtime
        elif name == "fft":
            function_times["fft_call"] = function_times.get("fft_call", 0) + cumtime
        elif name == "ifft":
            function_times["ifft_call"] = function_times.get("ifft_call", 0) + cumtime

    # Calculate percentages
    percentages = {k: 100 * v / total_time for k, v in function_times.items()}

    print(f"  Total profiled time: {total_time:.4f}s")
    print(f"  Time per step: {total_time / N_STEPS * 1000:.4f}ms")

    return {
        "N": N,
        "label": label,
        "total_time": total_time,
        "time_per_step": total_time / N_STEPS,
        **function_times,
        **{f"{k}_pct": v for k, v in percentages.items()},
    }


def main():
    """Run profiling analysis."""
    print("=" * 70)
    print("Profiling Analysis: Bottleneck Shift with N")
    print("=" * 70)
    print("\nThis analysis shows how computational bottlenecks change")
    print("as grid size N increases, explaining the scaling behavior.")
    print("=" * 70)

    results = []

    for label, N in N_VALUES.items():
        result = profile_at_N(N, label)
        results.append(result)

    # Save results
    df = pd.DataFrame(results)
    output_file = DATA_DIR / "profiling_results.parquet"
    df.to_parquet(output_file)

    print("\n" + "=" * 70)
    print("Summary: Time Breakdown by N")
    print("=" * 70)

    for _, row in df.iterrows():
        print(f"\n{row['label']:>10} (N={row['N']:>5}):")
        print(f"  Total time:     {row['total_time']:>8.4f}s")
        print(f"  Time per step:  {row['time_per_step']*1000:>8.4f}ms")

        if "FFT_execute" in row and not pd.isna(row["FFT_execute"]):
            print(f"  FFT execute:    {row['FFT_execute']:>8.4f}s  ({row['FFT_execute_pct']:>5.1f}%)")
        if "KdV_rhs" in row and not pd.isna(row["KdV_rhs"]):
            print(f"  KdV RHS:        {row['KdV_rhs']:>8.4f}s  ({row['KdV_rhs_pct']:>5.1f}%)")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_file}")
    print("=" * 70)

    print("\nKey Observation:")
    print("  As N increases, FFT time grows as O(N log N) and becomes")
    print("  the dominant term, while overhead becomes negligible.")


if __name__ == "__main__":
    main()
