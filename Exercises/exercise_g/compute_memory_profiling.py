#!/usr/bin/env python3
"""
Memory Profiling: RK3 vs RK4 Memory Usage
==========================================

Measures peak memory usage as a function of grid size N for both
RK3 and RK4 time integrators.

Analysis:
- Both methods should scale as O(N) (linear with grid size)
- RK4 uses slightly more memory (4 stages vs 3 for RK3)
- Measures peak memory during time integration
"""

import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, soliton, RK3, RK4

# Configuration
DATA_DIR = Path("data/A2/ex_g")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Test parameters
L = 40.0          # Domain half-length
T_FINAL = 0.1     # Short simulation time (just measure memory)
SOLITON_SPEED = 0.5
SOLITON_X0 = 0.0

# Grid sizes to test
N_VALUES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

# Methods to test
METHODS = {"RK3": RK3, "RK4": RK4}


def measure_memory(method_name: str, method_class, N: int) -> dict:
    """
    Measure peak memory usage for a simulation.

    Parameters
    ----------
    method_name : str
        Name of integrator ('RK3' or 'RK4')
    method_class : class
        Integrator class
    N : int
        Number of grid points

    Returns
    -------
    dict
        Memory statistics including peak usage
    """
    # Start memory tracking
    tracemalloc.start()

    # Setup solver
    solver = KdVSolver(N, L, dealias=False)
    integrator = method_class()
    x = solver.x

    # Initial condition
    u0 = soliton(x, 0.0, SOLITON_SPEED, SOLITON_X0)

    # Estimate timestep
    u_max = float(np.max(np.abs(u0)))
    dt = KdVSolver.stable_dt(N, L, u_max,
                              integrator_name=method_name.lower(),
                              dealiased=False)

    if not np.isfinite(dt) or dt <= 0.0:
        dt = 1e-3

    # Time integration (short simulation to measure memory)
    n_steps = int(np.ceil(T_FINAL / dt))
    t = 0.0
    u = u0.copy()

    for step in range(n_steps):
        u = integrator.step(solver.rhs, u, t, dt)
        t += dt

    # Get peak memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Convert bytes to MB
    peak_mb = peak / (1024 * 1024)
    current_mb = current / (1024 * 1024)

    return {
        "method": method_name,
        "N": N,
        "peak_memory_mb": peak_mb,
        "current_memory_mb": current_mb,
        "n_steps": n_steps,
    }


def main():
    """Run memory profiling analysis."""
    print("=" * 70)
    print("Memory Profiling: RK3 vs RK4")
    print("=" * 70)
    print(f"\nDomain: x ∈ [{-L}, {L}]")
    print(f"Test duration: T = {T_FINAL}")
    print(f"Grid sizes: N = {N_VALUES[0]} to {N_VALUES[-1]}")
    print("=" * 70)

    results = []

    for method_name, method_class in METHODS.items():
        print(f"\n{method_name}:")
        print("-" * 70)

        for N in N_VALUES:
            print(f"  N = {N:5d}", end="  ", flush=True)

            result = measure_memory(method_name, method_class, N)
            results.append(result)

            print(f"Peak memory: {result['peak_memory_mb']:6.2f} MB  "
                  f"({result['n_steps']} steps)")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Analyze scaling
    print("\n" + "=" * 70)
    print("Memory Scaling Analysis")
    print("=" * 70)

    for method in METHODS.keys():
        method_data = df[df["method"] == method].sort_values("N")

        N_vals = method_data["N"].values
        mem_vals = method_data["peak_memory_mb"].values

        # Fit log(memory) = α * log(N) + c
        log_N = np.log(N_vals)
        log_mem = np.log(mem_vals)
        coeffs = np.polyfit(log_N, log_mem, 1)
        alpha = coeffs[0]

        print(f"\n{method}:")
        print(f"  Scaling exponent: α = {alpha:.2f}")
        print(f"  Expected: α = 1.00 (linear O(N))")

        if abs(alpha - 1.0) < 0.2:
            print(f"  ✓ Excellent O(N) scaling!")
        elif abs(alpha - 1.0) < 0.3:
            print(f"  ✓ Good O(N) scaling")
        else:
            print(f"  ⚠ Deviates from O(N)")

    # Compare RK3 vs RK4
    print("\n" + "=" * 70)
    print("RK3 vs RK4 Memory Comparison")
    print("=" * 70)

    rk3_data = df[df["method"] == "RK3"].sort_values("N")
    rk4_data = df[df["method"] == "RK4"].sort_values("N")

    print("\n  N      RK3 (MB)  RK4 (MB)  Ratio")
    print("  " + "-" * 40)
    for N in N_VALUES:
        rk3_mem = rk3_data[rk3_data["N"] == N]["peak_memory_mb"].values[0]
        rk4_mem = rk4_data[rk4_data["N"] == N]["peak_memory_mb"].values[0]
        ratio = rk4_mem / rk3_mem

        print(f"  {N:5d}  {rk3_mem:8.2f}  {rk4_mem:8.2f}  {ratio:5.2f}×")

    avg_ratio = (rk4_data["peak_memory_mb"] / rk3_data["peak_memory_mb"]).mean()
    print(f"\n  Average ratio: RK4 uses {avg_ratio:.2f}× memory of RK3")

    # Save results
    output_file = DATA_DIR / "memory_profiling.parquet"
    df.to_parquet(output_file, index=False)

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_file}")
    print(f"Shape: {df.shape}")
    print("=" * 70)
    print("\n✓ Memory profiling complete!")


if __name__ == "__main__":
    main()
