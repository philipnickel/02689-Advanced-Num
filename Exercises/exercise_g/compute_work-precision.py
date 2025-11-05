"""
Work-precision comparison
=========================

Quantify accuracy vs computational work for RK3 and RK4 on a single-soliton KdV case.

Outputs
-------
* ``data/A2/ex_g/work_precision.parquet`` – error metrics vs timestep and work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, soliton, RK3, RK4

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATA_DIR = Path("data/A2/ex_g")
DATA_DIR.mkdir(parents=True, exist_ok=True)

L = 40.0  # half-domain [-L, L]
N = 256  # grid points
T_FINAL = 1.0
SOLITON_SPEED = 0.5
SOLITON_X0 = 0.0
DT_SCALES = np.array([0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05])

INTEGRATORS: dict[str, Callable[[], object]] = {
    "RK4": RK4,
    "RK3": RK3,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _stable_dt(method: str) -> float:
    solver = KdVSolver(N, L, dealias=False)
    u0 = soliton(solver.x, 0.0, SOLITON_SPEED, SOLITON_X0)
    u_max = float(np.max(np.abs(u0)))
    dt_est = KdVSolver.stable_dt(
        N,
        L,
        u_max,
        integrator_name=method.lower(),
        dealiased=solver.dealias,
    )
    if not np.isfinite(dt_est) or dt_est <= 0.0:
        return 1e-3
    return float(dt_est)


def _run_case(method: str, dt: float, factory: Callable[[], object]) -> dict:
    solver = KdVSolver(N, L, dealias=False)
    x = solver.x
    dx = solver.dx
    u0 = soliton(x, 0.0, SOLITON_SPEED, SOLITON_X0)
    integrator = factory()

    save_every = max(1, int(np.ceil(T_FINAL / dt)))

    t_saved, u_hist, performance = solver.solve(
        u0.copy(),
        T_FINAL,
        dt,
        save_every=save_every,
        integrator=integrator,
        measure_performance=True,
    )

    if len(u_hist) == 0:
        raise RuntimeError("Solver returned no snapshots.")

    u_num = u_hist[-1]
    t_end = float(t_saved[-1])
    u_exact = soliton(x, t_end, SOLITON_SPEED, SOLITON_X0)
    diff = u_num - u_exact

    error_l2 = float(np.sqrt(np.sum(diff**2) * dx))
    error_linf = float(np.max(np.abs(diff)))

    return {
        "method": method,
        "N": N,
        "L": L,
        "T": T_FINAL,
        "dt": dt,
        "n_steps": performance["nsteps"],
        "wall_time_s": performance["wall_time_s"],
        "mean_step_time_s": performance["mean_step_time_ms"] / 1000.0,
        "error_l2": error_l2,
        "error_linf": error_linf,
        "t_end": t_end,
    }


# --------------------------------------------------------------------------- #
# Main execution
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=" * 70)
    print("Exercise G – Work vs Precision (single soliton)")
    print("=" * 70)
    print(f"Domain: x ∈ [{-L}, {L}], N = {N}, T = {T_FINAL}")

    rows: list[dict] = []

    for method, factory in INTEGRATORS.items():
        dt_stable = _stable_dt(method)
        print(f"\nMethod: {method}, stability dt ≈ {dt_stable:.3e}")

        for scale in DT_SCALES:
            dt = scale * dt_stable
            print(f"  dt scale {scale:.2f} → dt = {dt:.3e}", end=" ... ", flush=True)
            result = _run_case(method, dt, factory)
            result["dt_scale"] = scale
            result["dt_stable"] = dt_stable
            rows.append(result)
            print(
                f"error L2 = {result['error_l2']:.3e}, "
                f"wall_time = {result['wall_time_s']:.3f}s"
            )

    df = pd.DataFrame(rows)
    df.sort_values(["method", "dt"], inplace=True)
    output_path = DATA_DIR / "work_precision.parquet"
    try:
        df.to_parquet(output_path, index=False)
    except ImportError:
        output_path = output_path.with_suffix(".csv")
        df.to_csv(output_path, index=False)

    print("\nSaved work-precision data →", output_path)
    print("Done.")
