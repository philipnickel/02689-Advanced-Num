"""
Spatial Convergence for KdV with Aliasing Comparison
=====================================================

Compares aliased vs dealiased spatial convergence using a single soliton
with the exact analytical solution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, soliton, get_time_integrator

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATA_DIR = Path("data/A2/ex_e")
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CONVERGENCE = DATA_DIR / "convergence.parquet"

L = 40.0  # half domain length
T_FINAL = 1.0  # short time to show spatial convergence cleanly
SAFETY = 0.35
INTEGRATOR_NAME = "rk3"

# Single soliton parameters
C = 1.0  # wave speed
X0 = 0.0  # initial position

# Test resolutions - focus on stable range
N_VALUES = [32, 48, 64, 96, 128, 192, 256]
DEALIAS_OPTIONS = [False, True]


# --------------------------------------------------------------------------- #
# Convergence study
# --------------------------------------------------------------------------- #

convergence_rows: list[dict] = []

for use_dealias in DEALIAS_OPTIONS:
    treatment = "dealiased (3/2-rule)" if use_dealias else "aliased"
    print(f"\n{'='*60}")
    print(f"{treatment.upper()}")
    print(f"{'='*60}")

    for N in N_VALUES:
        print(f"  N = {N:3d}", end="", flush=True)

        solver = KdVSolver(N, L, dealias=use_dealias)
        x = solver.x
        u0 = soliton(x, 0.0, C, X0)

        # Use stability-based dt
        dt = KdVSolver.stable_dt(
            N, L, float(np.max(np.abs(u0))),
            integrator_name=INTEGRATOR_NAME,
            dealiased=False  # Use aliased formula for both (more conservative)
        )
        dt *= SAFETY

        integrator = get_time_integrator(INTEGRATOR_NAME)

        try:
            t_saved, u_saved = solver.solve(
                u0.copy(),
                T_FINAL,
                dt,
                save_every=int(T_FINAL / dt),
                integrator=integrator,
            )

            u_final = u_saved[-1]
            t_final = t_saved[-1]

            # Compute exact analytical solution at final time
            u_exact = soliton(x, t_final, C, X0)

            # Compute error
            diff = u_final - u_exact
            dx = solver.dx
            l2_error = float(np.sqrt(np.sum(diff**2) * dx))
            linf_error = float(np.max(np.abs(diff)))

            convergence_rows.append({
                "N": N,
                "dealias": use_dealias,
                "Treatment": treatment,
                "L2_error": l2_error,
                "Linf_error": linf_error,
                "dt": dt,
                "T": T_FINAL,
                "L": L,
                "C": C,
            })

            print(f"  →  L2 = {l2_error:.3e}, L∞ = {linf_error:.3e}")

        except Exception as e:
            print(f"  →  FAILED: {e}")
            continue

# Save results
df_convergence = pd.DataFrame(convergence_rows)
df_convergence.to_parquet(OUTPUT_CONVERGENCE, index=False)

print(f"\n{'='*60}")
print(f"Convergence data saved → {OUTPUT_CONVERGENCE}")
print(f"{'='*60}")
