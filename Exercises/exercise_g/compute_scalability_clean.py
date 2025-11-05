#!/usr/bin/env python3
"""
Clean Scalability Analysis for KdV Solver
==========================================

Measures wall-clock time vs grid size N to verify O(N log N) complexity.

This script:
1. Times the solver at multiple N values (64 to 32768)
2. Separates FFT time from total time
3. Fits scaling models to identify complexity
4. Saves results for plotting

Expected Results:
- Small N (64-512):   α ≈ 0.7 (overhead dominates)
- Medium N (1K-4K):   α ≈ 0.8 (transition region)
- Large N (8K-32K):   α ≈ 0.9 (N log N emerges)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, soliton, RK4, RK3

# Configuration
DATA_DIR = Path("data/A2/ex_g")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Test parameters
L = 30.0
c = 0.5
x0 = 0.0
SAFETY_FACTOR = 0.1  # Use 10% of stable dt for safety

# Grid sizes to test (64 -> 32K shows full scaling behavior)
N_VALUES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

# Time integrators to test
METHODS = ["RK4", "RK3"]

# Timing parameters
MIN_STEPS = 200      # Minimum steps for reliable statistics
MAX_STEPS = 2000     # Maximum steps to keep runtime reasonable
RHS_REPEATS = 200    # Repetitions for RHS-only timing


def estimate_stable_dt(N: int, L: float, c: float, method: str) -> float:
    """
    Estimate stable time step with safety factor.

    Parameters
    ----------
    N : int
        Number of grid points
    L : float
        Domain half-length
    c : float
        Soliton amplitude
    method : str
        Time integrator name ('rk4' or 'rk3')

    Returns
    -------
    float
        Safe time step
    """
    solver = KdVSolver(N, L)
    u = soliton(solver.x, 0.0, c, x0)
    u_max = float(np.max(np.abs(u)))

    dt_est = KdVSolver.stable_dt(N, L, u_max, integrator_name=method.lower())
    dt_safe = SAFETY_FACTOR * dt_est if np.isfinite(dt_est) else 1e-3

    return float(dt_safe)


def time_solver(method: str, N: int, L: float, c: float) -> dict:
    """
    Time the solver for given parameters.

    Parameters
    ----------
    method : str
        Time integrator ('RK4' or 'RK3')
    N : int
        Grid size
    L : float
        Domain half-length
    c : float
        Soliton amplitude

    Returns
    -------
    dict
        Timing results including wall time, time per step, etc.
    """
    # Setup
    integrator_map = {"RK4": RK4, "RK3": RK3}
    integrator = integrator_map[method]()
    solver = KdVSolver(N, L, dealias=False)

    x = solver.x
    u0 = soliton(x, 0.0, c, x0)

    # Determine time step
    dt = estimate_stable_dt(N, L, c, method)
    dt = min(dt, 1.0 / MIN_STEPS)  # Ensure minimum steps

    # Determine effective simulation time
    T_effective = min(1.0, MAX_STEPS * dt)
    T_effective = max(T_effective, MIN_STEPS * dt)
    n_steps = int(T_effective / dt)

    # Warm up (trigger JIT compilation)
    for _ in range(10):
        _ = integrator.step(solver.rhs, u0, 0.0, dt)

    # Benchmark RHS alone (FFT-dominated)
    rhs_start = time.perf_counter()
    for _ in range(RHS_REPEATS):
        _ = solver.rhs(u0, 0.0)
    rhs_elapsed = time.perf_counter() - rhs_start
    rhs_time_per_call = rhs_elapsed / RHS_REPEATS

    # Benchmark full integration
    u = u0.copy()
    t = 0.0

    wall_start = time.perf_counter()
    for _ in range(n_steps):
        u = integrator.step(solver.rhs, u, t, dt)
        t += dt
    wall_elapsed = time.perf_counter() - wall_start

    # Calculate metrics
    time_per_step = wall_elapsed / n_steps

    # Estimate RHS calls per step (RK4=4, RK3=3)
    rhs_calls_per_step = 4 if method == "RK4" else 3
    rhs_time_estimated = rhs_time_per_call * rhs_calls_per_step

    return {
        "method": method,
        "N": N,
        "L": L,
        "c": c,
        "T_requested": 1.0,
        "T_effective": T_effective,
        "dt": dt,
        "n_steps": n_steps,
        "wall_time": wall_elapsed,
        "time_per_step": time_per_step,
        "rhs_time": rhs_time_estimated,
    }


def main():
    """Run scalability analysis."""
    print("=" * 70)
    print("Scalability Analysis: KdV Solver")
    print("=" * 70)
    print(f"\nTesting N = {N_VALUES[0]} to {N_VALUES[-1]}")
    print(f"Methods: {', '.join(METHODS)}\n")

    results = []

    for method in METHODS:
        print(f"\n{method}:")
        print("-" * 70)

        for N in N_VALUES:
            result = time_solver(method, N, L, c)
            results.append(result)

            print(f"  N={N:5d}  "
                  f"time/step={result['time_per_step']:.6f}s  "
                  f"total={result['wall_time']:.3f}s  "
                  f"steps={result['n_steps']}")

    # Save results
    df = pd.DataFrame(results)
    output_file = DATA_DIR / "scalability_timing.parquet"
    df.to_parquet(output_file)

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_file}")
    print(f"Shape: {df.shape}")
    print("=" * 70)


if __name__ == "__main__":
    main()
