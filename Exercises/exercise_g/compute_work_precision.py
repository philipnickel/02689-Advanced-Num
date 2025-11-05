"""
Work-Precision Analysis: RK3 vs RK4 with Manufactured Solutions
================================================================

Measures error vs computational cost for different time integrators
using manufactured solutions to verify convergence rates.
"""

# %% Imports and setup -------------------------------------------------------
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, ManufacturedSolution, RK3, RK4

# %% Configuration -----------------------------------------------------------
DATA_DIR = Path("data/A2/ex_g")
DATA_DIR.mkdir(parents=True, exist_ok=True)

L = 40.0
N = 256
T_FINAL = 3.0
AMPLITUDE = 5.0
WAVENUMBER = 2.0 * np.pi / (2 * L)
FREQUENCY = 3.0
DT_SCALES = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2])
METHODS = {"RK3": RK3, "RK4": RK4}
N_TRIALS = 5  # Number of timing trials for confidence intervals

print("=" * 70)
print("Work-Precision Analysis: KdV Solver with Manufactured Solutions")
print("=" * 70)
print(f"Domain: x ∈ [{-L}, {L}], N = {N}")
print(f"Simulation time: T = {T_FINAL}")
print(f"Manufactured solution: u(x,t) = {AMPLITUDE}*sin({WAVENUMBER:.4f}*x)*sin({FREQUENCY}*t)")
print(f"Testing timesteps from {DT_SCALES[0]:.2f}× to {DT_SCALES[-1]:.3f}× stable dt")
print("=" * 70)

# %% Create manufactured solution --------------------------------------------
manufactured = ManufacturedSolution(
    amplitude=AMPLITUDE,
    wavenumber=WAVENUMBER,
    frequency=FREQUENCY
)

# %% Run work-precision experiments ------------------------------------------
results = []

for method_name, method_class in METHODS.items():
    # Setup solver and integrator
    solver = KdVSolver(N, L, dealias=False)
    integrator = method_class()
    x = solver.x
    dx = solver.dx

    # Initial condition
    u0 = manufactured.u_exact(x, 0.0)

    # Estimate stable timestep
    u_max = float(np.max(np.abs(u0)))
    dt_stable = KdVSolver.stable_dt(N, L, u_max, integrator_name=method_name.lower(), dealiased=False)
    if not np.isfinite(dt_stable) or dt_stable <= 0.0:
        dt_stable = 1e-3

    print(f"\n{method_name} (Order {3 if method_name == 'RK3' else 4}):")
    print(f"  Stable dt estimate: {dt_stable:.3e}")
    print("-" * 70)

    for scale in DT_SCALES:
        dt = scale * dt_stable

        # RHS wrapper with source term
        def rhs_with_source(u, t):
            return solver.rhs(u, t, source_term=manufactured.source)

        n_steps = int(np.round(T_FINAL / dt))

        # Run multiple trials for timing statistics
        timing_trials = []
        for trial in range(N_TRIALS):
            t = 0.0
            u = u0.copy()

            start_time = time.perf_counter()
            for step in range(n_steps):
                u = integrator.step(rhs_with_source, u, t, dt)
                t += dt
            wall_time = time.perf_counter() - start_time
            timing_trials.append(wall_time)

        # Compute errors (use last trial's result)
        t_final = n_steps * dt
        u_exact = manufactured.u_exact(x, t_final)
        diff = u - u_exact
        error_l2 = float(np.sqrt(np.sum(diff**2) * dx))
        error_linf = float(np.max(np.abs(diff)))

        # RHS evaluations
        rhs_evals_per_step = 3 if method_name == "RK3" else 4
        total_rhs_evals = n_steps * rhs_evals_per_step

        # Store one result per trial (for seaborn to compute CI)
        for trial_idx, wall_time in enumerate(timing_trials):
            results.append({
                "method": method_name,
                "dt": dt,
                "dt_scale": scale,
                "trial": trial_idx,
                "n_steps": n_steps,
                "wall_time": wall_time,
                "rhs_evaluations": total_rhs_evals,
                "error_l2": error_l2,
                "error_linf": error_linf,
            })

        # Print with mean timing
        mean_time = np.mean(timing_trials)
        std_time = np.std(timing_trials)
        print(f"  dt = {dt:.3e} ({scale:5.2f}× stable)  "
              f"L2 err = {error_l2:.3e}  L∞ err = {error_linf:.3e}  "
              f"time = {mean_time:.3f}±{std_time:.3f}s")

# %% Estimate convergence rates ----------------------------------------------
df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("Convergence Rate Analysis")
print("=" * 70)

for method in METHODS.keys():
    method_data = df[df["method"] == method].sort_values("dt")
    dt_vals = method_data["dt"].values
    l2_errors = method_data["error_l2"].values

    # Fit error ~ dt^p
    log_dt = np.log(dt_vals)
    log_err = np.log(l2_errors)
    coeffs = np.polyfit(log_dt, log_err, 1)
    rate_l2 = coeffs[0]

    print(f"\n{method}:")
    print(f"  Observed L2 convergence rate: {rate_l2:.2f}")
    print(f"  Expected rate:                {3 if method == 'RK3' else 4:.0f}.00")

    if abs(rate_l2 - (3 if method == "RK3" else 4)) < 0.5:
        print(f"  ✓ Convergence rate matches theory!")
    else:
        print(f"  ⚠ Convergence rate differs from theory")

# %% Save results ------------------------------------------------------------
output_file = DATA_DIR / "work_precision.parquet"
df.to_parquet(output_file, index=False)

print("\n" + "=" * 70)
print(f"Results saved to: {output_file}")
print(f"Shape: {df.shape}")
print("=" * 70)
print("\n✓ Work-precision analysis complete!")
