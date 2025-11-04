"""
Profiling computations
======================

Benchmark RK3 and RK4 time integrators on a single-soliton KdV case.

The script runs the solver under ``cProfile`` to capture hotspots while also
recording step timing statistics returned by the solver. Results are written to:

* ``data/A2/ex_g/profiling_summary.parquet`` – one row per method with timing stats
* ``data/A2/ex_g/profiling_functions.parquet`` – top functions by cumulative time
* ``data/A2/ex_g/profiles/profile_<method>.prof`` – raw ``cProfile`` dumps
"""

from __future__ import annotations

import cProfile
from pathlib import Path
import pstats
from typing import Callable

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, soliton, RK3, RK4

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATA_DIR = Path("data/A2/ex_g")
PROFILE_DIR = DATA_DIR / "profiles"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

L = 40.0  # half-domain [-L, L]
N = 256  # spatial grid points
T_FINAL = 1.0  # final simulation time
SOLITON_SPEED = 0.5
SOLITON_X0 = 0.0
SAFETY = 0.3  # fraction of stability-limited dt

INTEGRATORS: dict[str, Callable[[], object]] = {
    "RK4": RK4,
    "RK3": RK3,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _reset_integrator(integrator: object) -> None:
    """Clear multi-step history buffers when present."""
    if hasattr(integrator, "u_history"):
        integrator.u_history = []
    if hasattr(integrator, "f_history"):
        integrator.f_history = []


def _stable_dt(method: str) -> float:
    """Return a safety-limited timestep for the selected method."""
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
        dt_est = 1e-3
    return SAFETY * float(dt_est)


def _profile_method(method: str, factory: Callable[[], object]) -> tuple[dict, pd.DataFrame]:
    """
    Run the solver under ``cProfile`` and return summary metrics.

    Returns
    -------
    summary : dict
        Row compatible with profiling summary dataframe.
    top_functions : pd.DataFrame
        Top functions sorted by cumulative time.
    """
    dt = _stable_dt(method)
    solver = KdVSolver(N, L, dealias=False)
    x = solver.x
    u0 = soliton(x, 0.0, SOLITON_SPEED, SOLITON_X0)
    integrator = factory()
    _reset_integrator(integrator)

    save_every = max(1, int(np.ceil(T_FINAL / dt)))

    def _solve():
        return solver.solve(
            u0.copy(),
            T_FINAL,
            dt,
            save_every=save_every,
            integrator=integrator,
            measure_performance=True,
        )

    profiler = cProfile.Profile()
    t_saved, u_hist, performance = profiler.runcall(_solve)

    # Persist raw profile for further inspection
    profile_path = PROFILE_DIR / f"profile_{method.lower()}.prof"
    profiler.dump_stats(profile_path)

    stats = pstats.Stats(profiler)
    stats.strip_dirs()

    # Build dataframe of top functions by cumulative time
    items = []
    for func, stat in stats.stats.items():
        ccalls, ncalls, tottime, cumtime, _ = stat
        func_name = f"{func[0]}:{func[1]}({func[2]})"
        items.append(
            {
                "method": method,
                "function": func_name,
                "primitive_calls": ccalls,
                "calls": ncalls,
                "tottime": tottime,
                "cumtime": cumtime,
                "per_call": tottime / ncalls if ncalls else np.nan,
            }
        )

    df_functions = (
        pd.DataFrame(items)
        .sort_values(["cumtime", "tottime"], ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    # Summarise solver timings
    n_steps = performance["nsteps"]
    summary = {
        "method": method,
        "N": N,
        "L": L,
        "T": T_FINAL,
        "dt": dt,
        "n_steps": n_steps,
        "save_every": save_every,
        "wall_time_s": performance["wall_time_s"],
        "mean_step_time_s": performance["mean_step_time_ms"] / 1000.0,
        "std_step_time_s": performance["std_step_time_ms"] / 1000.0,
        "profile_path": str(profile_path),
    }

    return summary, df_functions


# --------------------------------------------------------------------------- #
# Main execution
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=" * 70)
    print("Exercise G – Profiling RK3 vs RK4 (single soliton)")
    print("=" * 70)
    print(f"Domain: x ∈ [{-L}, {L}], N = {N}, T = {T_FINAL}")

    summaries = []
    function_tables: list[pd.DataFrame] = []

    for method, factory in INTEGRATORS.items():
        print(f"\nProfiling {method}...")
        summary, df_functions = _profile_method(method, factory)
        summaries.append(summary)
        function_tables.append(df_functions)
        print(
            f"  dt = {summary['dt']:.3e}, steps = {summary['n_steps']}, "
            f"wall_time = {summary['wall_time_s']:.3f}s, "
            f"mean_step = {summary['mean_step_time_s']:.3e}s"
        )
        print("  Profile saved to:", summary["profile_path"])

    df_summary = pd.DataFrame(summaries)
    df_summary.to_parquet(DATA_DIR / "profiling_summary.parquet", index=False)

    df_functions = pd.concat(function_tables, ignore_index=True)
    df_functions.to_parquet(DATA_DIR / "profiling_functions.parquet", index=False)

    print("\nSaved profiling summary →", DATA_DIR / "profiling_summary.parquet")
    print("Saved top functions →", DATA_DIR / "profiling_functions.parquet")
    print("\nDone.")
