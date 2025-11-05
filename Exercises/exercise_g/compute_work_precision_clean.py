#!/usr/bin/env python3
"""
Clean Work-Precision Analysis for KdV Solver
=============================================

Quantifies the trade-off between accuracy and computational cost for
different time integrators using manufactured solutions.

Analysis:
1. Tests RK3 (3rd order) and RK4 (4th order) integrators
2. Varies timestep from coarse to fine (fraction of stable dt)
3. Measures errors (L2 and L∞ norms) against exact manufactured solution
4. Measures computational work (wall time, function evaluations)
5. Verifies theoretical convergence rates

Method:
- Manufactured solution: u(x,t) = A * sin(k*x) * sin(omega*t)
- Modified KdV with source: u_t + 6*u*u_x + u_xxx = f(x,t)
- Source term computed to make u_exact satisfy the equation
- Errors measured against known exact solution

Expected Results:
- RK3: Error ~ dt^3 (3rd order convergence)
- RK4: Error ~ dt^4 (4th order convergence)
- RK4 should be more efficient for tight tolerances
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, ManufacturedSolution, RK3, RK4

# Configuration
DATA_DIR = Path("data/A2/ex_g")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Physical parameters
L = 40.0          # Domain half-length [-L, L]
N = 256           # Spatial grid points
T_FINAL = 3.0     # Longer simulation to accumulate more temporal error

# Manufactured solution parameters
AMPLITUDE = 5.0              # Large amplitude for strong nonlinearity
WAVENUMBER = 2.0 * np.pi / (2 * L)  # Fundamental mode k = 2π/(2L) for periodicity
FREQUENCY = 3.0              # High frequency for strong temporal variation

# Timestep scales (fractions of stable dt)
# Focus on coarser timesteps where temporal error dominates spatial error
DT_SCALES = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2])

# Methods to test
METHODS = {"RK3": RK3, "RK4": RK4}


def estimate_stable_dt(method: str, manufactured: ManufacturedSolution) -> float:
    """
    Estimate stable timestep for given integrator.

    Parameters
    ----------
    method : str
        Integrator name ('RK3' or 'RK4')
    manufactured : ManufacturedSolution
        Manufactured solution instance

    Returns
    -------
    float
        Stable timestep estimate
    """
    solver = KdVSolver(N, L, dealias=False)
    u0 = manufactured.u_exact(solver.x, 0.0)
    u_max = float(np.max(np.abs(u0)))

    dt_est = KdVSolver.stable_dt(
        N, L, u_max,
        integrator_name=method.lower(),
        dealiased=False
    )

    if not np.isfinite(dt_est) or dt_est <= 0.0:
        return 1e-3

    return float(dt_est)


def compute_errors(u_numerical: np.ndarray, u_exact: np.ndarray, dx: float) -> tuple:
    """
    Compute L2 and L∞ error norms.

    Parameters
    ----------
    u_numerical : np.ndarray
        Numerical solution
    u_exact : np.ndarray
        Exact solution
    dx : float
        Grid spacing

    Returns
    -------
    tuple
        (L2 error, L∞ error)
    """
    diff = u_numerical - u_exact

    # L2 error: sqrt(integral of |error|^2 dx)
    error_l2 = float(np.sqrt(np.sum(diff**2) * dx))

    # L∞ error: maximum pointwise error
    error_linf = float(np.max(np.abs(diff)))

    return error_l2, error_linf


def run_simulation(
    method_name: str,
    method_class,
    dt: float,
    manufactured: ManufacturedSolution
) -> dict:
    """
    Run simulation and measure accuracy vs work.

    Parameters
    ----------
    method_name : str
        Name of integrator ('RK3' or 'RK4')
    method_class : class
        Integrator class
    dt : float
        Timestep
    manufactured : ManufacturedSolution
        Manufactured solution instance

    Returns
    -------
    dict
        Results including errors, work metrics, convergence
    """
    # Setup
    solver = KdVSolver(N, L, dealias=False)
    integrator = method_class()
    x = solver.x
    dx = solver.dx

    # Initial condition from manufactured solution
    u0 = manufactured.u_exact(x, 0.0)

    # Create RHS wrapper that includes source term
    def rhs_with_source(u: np.ndarray, t: float) -> np.ndarray:
        return solver.rhs(u, t, source_term=manufactured.source)

    # Time integration
    n_steps = int(np.round(T_FINAL / dt))
    t = 0.0
    u = u0.copy()

    # Measure wall time
    start_time = time.perf_counter()

    for step in range(n_steps):
        u = integrator.step(rhs_with_source, u, t, dt)
        t += dt

    wall_time = time.perf_counter() - start_time

    # Compute exact solution at final time
    t_final = n_steps * dt
    u_exact = manufactured.u_exact(x, t_final)

    # Compute errors
    error_l2, error_linf = compute_errors(u, u_exact, dx)

    # RHS evaluations (RK3=3, RK4=4 per step)
    rhs_evals_per_step = 3 if method_name == "RK3" else 4
    total_rhs_evals = n_steps * rhs_evals_per_step

    return {
        "method": method_name,
        "N": N,
        "L": L,
        "T_final": T_FINAL,
        "dt": dt,
        "n_steps": n_steps,
        "t_final": t_final,
        "wall_time": wall_time,
        "rhs_evaluations": total_rhs_evals,
        "error_l2": error_l2,
        "error_linf": error_linf,
    }


def estimate_convergence_rate(dt_values: np.ndarray, errors: np.ndarray) -> float:
    """
    Estimate convergence rate from error vs dt.

    Fits error ~ dt^p in log space.

    Parameters
    ----------
    dt_values : np.ndarray
        Timestep values
    errors : np.ndarray
        Corresponding errors

    Returns
    -------
    float
        Estimated convergence rate p
    """
    # Fit log(error) = p * log(dt) + c
    log_dt = np.log(dt_values)
    log_err = np.log(errors)

    # Linear regression in log space
    coeffs = np.polyfit(log_dt, log_err, 1)
    rate = coeffs[0]

    return float(rate)


def main():
    """Run work-precision analysis."""
    print("=" * 70)
    print("Work-Precision Analysis: KdV Solver with Manufactured Solutions")
    print("=" * 70)
    print(f"\nDomain: x ∈ [{-L}, {L}], N = {N}")
    print(f"Simulation time: T = {T_FINAL}")
    print(f"Manufactured solution: u(x,t) = {AMPLITUDE}*sin({WAVENUMBER:.4f}*x)*sin({FREQUENCY}*t)")
    print(f"\nTesting timesteps from {DT_SCALES[0]:.2f}× to {DT_SCALES[-1]:.3f}× stable dt")
    print("=" * 70)

    # Create manufactured solution
    manufactured = ManufacturedSolution(
        amplitude=AMPLITUDE,
        wavenumber=WAVENUMBER,
        frequency=FREQUENCY
    )

    results = []

    for method_name, method_class in METHODS.items():
        # Estimate stable timestep
        dt_stable = estimate_stable_dt(method_name, manufactured)

        print(f"\n{method_name} (Order {3 if method_name == 'RK3' else 4}):")
        print(f"  Stable dt estimate: {dt_stable:.3e}")
        print("-" * 70)

        for scale in DT_SCALES:
            dt = scale * dt_stable

            print(f"  dt = {dt:.3e} ({scale:5.2f}× stable)", end="  ", flush=True)

            result = run_simulation(method_name, method_class, dt, manufactured)
            result["dt_scale"] = scale
            result["dt_stable"] = dt_stable
            results.append(result)

            print(f"L2 err = {result['error_l2']:.3e}  "
                  f"L∞ err = {result['error_linf']:.3e}  "
                  f"time = {result['wall_time']:.3f}s")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Estimate convergence rates for each method
    print("\n" + "=" * 70)
    print("Convergence Rate Analysis")
    print("=" * 70)

    for method in METHODS.keys():
        method_data = df[df["method"] == method].sort_values("dt")
        dt_vals = method_data["dt"].values
        l2_errors = method_data["error_l2"].values

        rate_l2 = estimate_convergence_rate(dt_vals, l2_errors)

        print(f"\n{method}:")
        print(f"  Observed L2 convergence rate: {rate_l2:.2f}")
        print(f"  Expected rate:                {3 if method == 'RK3' else 4:.0f}.00")

        if abs(rate_l2 - (3 if method == "RK3" else 4)) < 0.5:
            print(f"  ✓ Convergence rate matches theory!")
        else:
            print(f"  ⚠ Convergence rate differs from theory")

    # Save results
    output_file = DATA_DIR / "work_precision.parquet"
    df.to_parquet(output_file, index=False)

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_file}")
    print(f"Shape: {df.shape}")
    print("=" * 70)
    print("\n✓ Work-precision analysis complete!")


if __name__ == "__main__":
    main()
