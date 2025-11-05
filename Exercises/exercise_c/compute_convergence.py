"""
Spatial and Temporal Convergence for Fourier KdV Solver
========================================================

The script generates two Parquet tables:

* ``kdv_spatial_convergence.parquet`` – error vs. number of modes (N) for
  aliased/dealiased runs.
* ``kdv_temporal_convergence.parquet`` – error vs. timestep (dt) for the time
  integrators used in the assignment.
"""

# TODO: have a look at L6 - slides 28 and 29
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, soliton, RK4, RK3


# //----------------------------------------------------------------------- #
# Configuration
# //----------------------------------------------------------------------- #

DATA_DIR = Path("data/A2/ex_c")
DATA_DIR.mkdir(parents=True, exist_ok=True)

L_SPATIAL = 40.0
L_TEMPORAL = 40.0  # Larger domain => larger dx => allows larger dt
X0 = 0.0

DEALIAS_OPTIONS = [False, True]
INTEGRATORS = [RK4, RK3]

WAVE_SPEED = 1.0

T_SPATIAL = 0.01#2.0e-2
DT_SPATIAL = 1.0e-6  # sufficiently small to suppress temporal error
# Logarithmic spacing with 20 values from 16 to 256, ensuring even numbers
#N_VALUES_SPATIAL = 2 * np.logspace(1, 2, num=20, dtype=int)#(np.geomspace(10, 350, num=20, dtype=int) // 2) * 2

# Logarithmically spaced floats from 16 to 250
N_VALUES_SPATIAL = 2 * np.logspace(np.log10(5), np.log10(175), num=20, dtype=int)
# Round and force evenness


N_TEMPORAL = 350  # From spatial study: error ~1e-11, well below temporal errors
T_TEMPORAL = 0.01  # Very short time to avoid accumulated nonlinear effects
# Logarithmic spacing for timesteps - extend to larger dt to see convergence
DT_VALUES = np.logspace(-8, -5, num=5)




# //----------------------------------------------------------------------- #
# Helpers
# //----------------------------------------------------------------------- #


def _reset_multistep_state(integrator: object) -> None:
    """Clear history buffers for multi-step integrators."""
    if hasattr(integrator, "u_history"):
        integrator.u_history = []
    if hasattr(integrator, "f_history"):
        integrator.f_history = []


def _solve_case(
    N: int,
    *,
    dt: float,
    T: float,
    integrator_class: type,
    wave_speed: float,
    dealias: bool,
    half_length: float,
) -> tuple[np.ndarray, float, np.ndarray, float, int]:
    """Integrate soliton and return (grid, dx, final solution, t_end, steps)."""
    solver = KdVSolver(N, half_length, dealias=dealias)
    x = solver.x
    dx = solver.dx

    u0 = soliton(x, 0.0, wave_speed, X0)
    integrator = integrator_class()
    _reset_multistep_state(integrator)

    save_every = max(1, int(np.ceil(T / dt)))

    t_saved, u_hist = solver.solve(
        u0.copy(),
        T,
        dt,
        integrator=integrator,
        save_every=save_every,
    )

    if len(u_hist) == 0:
        raise RuntimeError("Solver returned no states.")

    u_final = u_hist[-1]
    t_end = float(t_saved[-1])
    steps_taken = len(u_hist) - 1
    return x, dx, u_final, t_end, steps_taken


# //----------------------------------------------------------------------- #
# Spatial convergence: exponential drop with number of modes
# //----------------------------------------------------------------------- #

spatial_rows: list[dict[str, object]] = []

for dealias in DEALIAS_OPTIONS:
    dealias_label = "De-aliased" if dealias else "Aliased"
    print(f"\n--- {dealias_label} ---")

    for integrator_class in INTEGRATORS:
        method_name = integrator_class.__name__
        print(f"  Method: {method_name}")

        for N in N_VALUES_SPATIAL:
            current_half_length = L_SPATIAL
            x, dx, u_num, t_end, steps_taken = _solve_case(
                N,
                dt=DT_SPATIAL,
                T=T_SPATIAL,
                integrator_class=integrator_class,
                wave_speed=WAVE_SPEED,
                dealias=dealias,
                half_length=current_half_length,
            )
            u_exact = soliton(x, t_end, WAVE_SPEED, X0)
            diff = u_num - u_exact
            l2 = float(np.sqrt(np.sum(diff**2) * dx))
            linf = float(np.max(np.abs(diff)))

            spatial_rows.append(
                {
                    "N": N,
                    "dt": DT_SPATIAL,
                    "T": T_SPATIAL,
                    "t_end": t_end,
                    "n_steps": steps_taken,
                    "method": method_name,
                    "dealias": dealias_label,
                    "L": current_half_length,
                    "Error": l2,
                }
            )

            print(f"    N={N:3d}: L2={l2:.3e}, L∞={linf:.3e}")

df_spatial = pd.DataFrame(spatial_rows)
df_spatial["method"] = df_spatial["method"].astype("category")

spatial_path = DATA_DIR / "kdv_spatial_convergence.parquet"
df_spatial.to_parquet(spatial_path, index=False)

print(f"\nSaved spatial convergence data")


# //----------------------------------------------------------------------- #
# Temporal convergence: dt error for explicit/implicit integrators
# //----------------------------------------------------------------------- #

temporal_rows: list[dict[str, object]] = []

print("\n--- Temporal Convergence (De-aliased) ---")
print(f"Testing {len(INTEGRATORS)} integrators × {len(DT_VALUES)} timesteps = {len(INTEGRATORS) * len(DT_VALUES)} cases\n")

for integrator_idx, integrator_class in enumerate(INTEGRATORS, 1):
    method_name = integrator_class.__name__
    print(f"[{integrator_idx}/{len(INTEGRATORS)}] Method: {method_name}")

    successful_runs = 0
    for dt_idx, dt in enumerate(DT_VALUES, 1):
        try:
            current_half_length = L_TEMPORAL
            x, dx, u_num, t_end, steps_taken = _solve_case(
                N_TEMPORAL,
                dt=float(dt),
                T=T_TEMPORAL,
                integrator_class=integrator_class,
                wave_speed=WAVE_SPEED,
                dealias=True,  # Use dealiasing for stability
                half_length=current_half_length,
            )
            u_exact = soliton(x, T_TEMPORAL, WAVE_SPEED, X0)
            diff = u_num - u_exact
            l2 = float(np.sqrt(np.sum(diff**2) * dx))
            linf = float(np.max(np.abs(diff)))

            # Skip if we got NaN or inf
            if not (np.isfinite(l2) and np.isfinite(linf)):
                print(f"  [{dt_idx:2d}/{len(DT_VALUES)}] dt={dt:.3e}: SKIPPED (unstable)")
                continue

        except Exception as exc:  # pragma: no cover - diagnostic output
            print(f"  [{dt_idx:2d}/{len(DT_VALUES)}] dt={dt:.3e}: FAILED ({exc})")
            continue

        n_timesteps = int(np.round(T_TEMPORAL / dt))
        temporal_rows.append(
            {
                "dt": float(dt),
                "N": N_TEMPORAL,
                "T": T_TEMPORAL,
                "t_end": t_end,
                "n_steps": n_timesteps,
                "method": method_name,
                "dealias": "De-aliased",
                "L": current_half_length,
                "Error": l2,
            }
        )

        successful_runs += 1
        print(f"  [{dt_idx:2d}/{len(DT_VALUES)}] dt={dt:.3e} ({n_timesteps:4d} steps): L2={l2:.6e}, L∞={linf:.6e}")

    print(f"  → Completed {successful_runs}/{len(DT_VALUES)} runs for {method_name}\n")



df_temporal = pd.DataFrame(temporal_rows)
df_temporal["method"] = df_temporal["method"].astype("category")

temporal_path = DATA_DIR / "kdv_temporal_convergence.parquet"
df_temporal.to_parquet(temporal_path, index=False)

print(f"\nSaved temporal convergence data")
print("\nConvergence studies completed.")
