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
L_TEMPORAL = 40.0
X0 = 0.0

DEALIAS_OPTIONS = [False, True]

INTEGRATOR_FACTORIES: dict[str, Callable[[], object]] = {
    "RK4": RK4,
    "RK3": RK3,
}
TEMPORAL_METHODS: tuple[str, ...] = ("RK4", "RK3")

WAVE_SPEED = 1.0

T_SPATIAL = 0.01#2.0e-2
DT_SPATIAL = 1.0e-6  # sufficiently small to suppress temporal error
# Logarithmic spacing with 20 values from 16 to 256, ensuring even numbers
N_VALUES_SPATIAL = (np.geomspace(10, 300, num=20, dtype=int) // 2) * 2

N_TEMPORAL = 100
# Logarithmic spacing using arange with powers of 0.5 (halving each step)
DT_SCALES = 0.4 * (0.5**np.arange(1, 3))




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
    method_name: str,
    wave_speed: float,
    dealias: bool,
    half_length: float,
) -> tuple[np.ndarray, float, np.ndarray, float, int]:
    """Integrate soliton and return (grid, dx, final solution, t_end, steps)."""
    solver = KdVSolver(N, half_length, dealias=dealias)
    x = solver.x
    dx = solver.dx

    u0 = soliton(x, 0.0, wave_speed, X0)
    integrator = INTEGRATOR_FACTORIES[method_name]()
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


def _stability_limited_dt(
    N: int,
    wave_speed: float,
    *,
    method_name: str,
    dealias: bool,
    half_length: float,
) -> float:
    """Return a conservative stable timestep for (N, method, wave)."""
    solver = KdVSolver(N, half_length, dealias=dealias)
    u0 = soliton(solver.x, 0.0, wave_speed, X0)
    u_max = float(np.max(np.abs(u0)))
    dt_est = KdVSolver.stable_dt(
        N,
        half_length,
        u_max,
        integrator_name=method_name.lower(),
        dealiased=dealias,
    )
    if not np.isfinite(dt_est) or dt_est <= 0.0:
        return 1e-3
    return float(dt_est)


# //----------------------------------------------------------------------- #
# Spatial convergence: exponential drop with number of modes
# //----------------------------------------------------------------------- #

spatial_rows: list[dict[str, object]] = []

for dealias in DEALIAS_OPTIONS:
    dealias_label = "De-aliased" if dealias else "Aliased"
    print(f"\n--- {dealias_label} ---")

    for method_name in INTEGRATOR_FACTORIES:
        print(f"  Method: {method_name}")

        for N in N_VALUES_SPATIAL:
            current_half_length = L_SPATIAL
            x, dx, u_num, t_end, steps_taken = _solve_case(
                N,
                dt=DT_SPATIAL,
                T=T_SPATIAL,
                method_name=method_name,
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

for method_name in TEMPORAL_METHODS:
    dt_stable = _stability_limited_dt(
        N_TEMPORAL,
        WAVE_SPEED,
        method_name=method_name,
        dealias=DEALIAS_OPTIONS[1],  # dealiased
        half_length=L_TEMPORAL,
    )

    dt_values = np.array(dt_stable * DT_SCALES, dtype=float)
    dt_values = dt_values[(dt_values > 0.0) & np.isfinite(dt_values)]

    for dt in dt_values:
        target_T = float(dt)

        try:
            current_half_length = L_TEMPORAL
            x, dx, u_num, t_end, steps_taken = _solve_case(
                N_TEMPORAL,
                dt=float(dt),
                T=target_T,
                method_name=method_name,
                wave_speed=WAVE_SPEED,
                dealias=DEALIAS_OPTIONS[1],  # dealiased
                half_length=current_half_length,
            )
            u_exact = soliton(x, target_T, WAVE_SPEED, X0)
            diff = u_num - u_exact
            l2 = float(np.sqrt(np.sum(diff**2) * dx))
            linf = float(np.max(np.abs(diff)))
        except Exception as exc:  # pragma: no cover - diagnostic output
            print(f"    dt={dt:.3e}: FAILED ({exc})")
            continue


        temporal_rows.append(
            {
                "dt": float(dt),
                "N": N_TEMPORAL,
                "T": target_T,
                "t_end": target_T,
                "n_steps": steps_taken,
                "method": method_name,
                "dealias": "De-aliased",
                "L": current_half_length,
                "Error": l2,
            }
        )



df_temporal = pd.DataFrame(temporal_rows)
df_temporal["method"] = df_temporal["method"].astype("category")

temporal_path = DATA_DIR / "kdv_temporal_convergence.parquet"
df_temporal.to_parquet(temporal_path, index=False)

print(f"\nSaved temporal convergence data")
print("\nConvergence studies completed.")
