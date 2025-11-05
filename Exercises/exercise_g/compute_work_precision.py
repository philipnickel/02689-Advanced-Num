"""
Work-Precision Analysis: RK3 vs RK4 with Soliton Solutions
===========================================================

Measures error vs computational cost for different time integrators
using KdV soliton solutions (same problem as exercise_c).
"""

# %% Imports and setup -------------------------------------------------------
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, soliton, RK3, RK4

# %% Configuration -----------------------------------------------------------
DATA_DIR = Path("data/A2/ex_g")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Use same parameters as exercise_c
L = 40.0
X0 = 0.0
WAVE_SPEED = 1.0

N_VALUES = np.array([200])  # Fixed N like exercise_c temporal convergence
T_FINAL_VALUES = np.array([0.5, 1.0, 2.0])  # Vary simulation time for error bands
# Use absolute dt values (not scales!) - stay within stability limit (~3.8e-3 for RK3)
DT_VALUES = np.logspace(-5, -3, num=10)
METHODS = {"RK3": RK3, "RK4": RK4}
N_TRIALS = 3  # Reduced trials since we're varying T_FINAL

print("=" * 70)
print("Work-Precision Analysis: KdV Solver with Soliton Solutions")
print("=" * 70)
print(f"Domain: x ∈ [{-L}, {L}]")
print(f"Grid size: N = {N_VALUES[0]}")
print(f"Simulation times: T = {T_FINAL_VALUES[0]} to {T_FINAL_VALUES[-1]}")
print(f"Soliton: c = {WAVE_SPEED}, x0 = {X0}")
print(f"Testing dt from {DT_VALUES[0]:.3e} to {DT_VALUES[-1]:.3e}")
print(f"Number of trials per T_FINAL: {N_TRIALS}")
print("=" * 70)

# %% Run work-precision experiments ------------------------------------------
results = []

for method_name, method_class in METHODS.items():
    print(f"\n{method_name} (Order {3 if method_name == 'RK3' else 4}):")
    print("-" * 70)

    for N in N_VALUES:
        # Setup solver and integrator for this N
        solver = KdVSolver(N, L, dealias=True)  # Use dealiasing like exercise_c
        x = solver.x
        dx = solver.dx

        # Initial condition (soliton at t=0)
        u0 = soliton(x, 0.0, WAVE_SPEED, X0)

        print(f"\n  N = {N}")

        for dt in DT_VALUES:
            # Run multiple T_FINAL values and trials for visible error bands
            timing_trials = []
            for T_FINAL in T_FINAL_VALUES:
                n_steps = int(np.round(T_FINAL / dt))

                for trial in range(N_TRIALS):
                    t = 0.0
                    u = u0.copy()
                    integrator = method_class()  # Fresh integrator for each trial

                    start_time = time.perf_counter()
                    for step in range(n_steps):
                        u = integrator.step(solver.rhs, u, t, dt)
                        t += dt
                    wall_time = time.perf_counter() - start_time
                    timing_trials.append((T_FINAL, wall_time))

            # Compute error using middle T_FINAL value
            T_FINAL_ref = T_FINAL_VALUES[len(T_FINAL_VALUES) // 2]
            n_steps_ref = int(np.round(T_FINAL_ref / dt))
            t_final = n_steps_ref * dt
            u = u0.copy()
            t = 0.0
            integrator = method_class()
            for step in range(n_steps_ref):
                u = integrator.step(solver.rhs, u, t, dt)
                t += dt

            u_exact = soliton(x, t_final, WAVE_SPEED, X0)
            diff = u - u_exact
            error_l2 = float(np.sqrt(np.sum(diff**2) * dx))
            error_linf = float(np.max(np.abs(diff)))

            # RHS evaluations
            rhs_evals_per_step = 3 if method_name == "RK3" else 4

            # Store one result per T_FINAL and trial (for aggregation)
            for T_FINAL, wall_time in timing_trials:
                n_steps = int(np.round(T_FINAL / dt))
                total_rhs_evals = n_steps * rhs_evals_per_step

                results.append({
                    "method": method_name,
                    "N": N,
                    "dt": dt,
                    "T_FINAL": T_FINAL,
                    "n_steps": n_steps,
                    "wall_time": wall_time,
                    "rhs_evaluations": total_rhs_evals,
                    "error_l2": error_l2,
                    "error_linf": error_linf,
                })

            # Print with mean timing
            times = [t[1] for t in timing_trials]
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"    dt = {dt:.3e} ({n_steps_ref:4d} steps)  "
                  f"L2 err = {error_l2:.3e}  L∞ err = {error_linf:.3e}  "
                  f"time = {mean_time:.3f}±{std_time:.3f}s")

# %% Save results ------------------------------------------------------------
df = pd.DataFrame(results)
output_file = DATA_DIR / "work_precision.parquet"
df.to_parquet(output_file, index=False)

print("\n" + "=" * 70)
print(f"✓ Results saved to: {output_file}")
print(f"  Shape: {df.shape}")
print("=" * 70)
