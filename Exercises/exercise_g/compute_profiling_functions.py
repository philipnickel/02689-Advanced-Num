"""
Function-level profiling using cProfile.

Usage
-----
    uv run python Exercises/exercise_g/compute_profiling_functions.py
"""

# %% Imports
from __future__ import annotations

import cProfile
import pstats
from pathlib import Path
import numpy as np
import pandas as pd
from spectral.tdp import KdVSolver, RK3, RK4, soliton

# %% Configuration
L = 40.0
SOLITON_SPEED = 0.5
SOLITON_X0 = 0.0
SAFETY = 0.3
T_FINAL = 0.5
N_VALUES = [128, 192, 256]

METHODS = {
    "RK4": RK4,
    "RK3": RK3,
}

output_dir = Path("data/A2/ex_g")
output_dir.mkdir(parents=True, exist_ok=True)


# %% Helper functions
def estimate_stable_dt(method: str, N: int) -> float:
    """Estimate stable time step for given method and grid size."""
    solver = KdVSolver(N, L, dealias=False)
    u0 = soliton(solver.x, 0.0, SOLITON_SPEED, SOLITON_X0)
    u_max = float(np.max(np.abs(u0)))
    dt_est = KdVSolver.stable_dt(
        N, L, u_max, integrator_name=method.lower(), dealiased=solver.dealias
    )
    if not np.isfinite(dt_est) or dt_est <= 0.0:
        dt_est = 1e-3
    return SAFETY * float(dt_est)


def profile_case(method: str, N: int, integrator_cls):
    """Profile a single case and return DataFrame."""
    dt = estimate_stable_dt(method, N)
    solver = KdVSolver(N, L, dealias=False)
    x = solver.x
    u0 = soliton(x, 0.0, SOLITON_SPEED, SOLITON_X0)
    integrator = integrator_cls()
    save_every = max(1, int(np.ceil(T_FINAL / dt)))

    print(f"Profiling {method} @ N={N}: dt={dt:.3e}")

    # Profile the solve
    profiler = cProfile.Profile()
    profiler.enable()
    solver.solve(u0.copy(), T_FINAL, dt, save_every=save_every, integrator=integrator)
    profiler.disable()

    # Convert to pstats
    stats = pstats.Stats(profiler)

    # Extract data into DataFrame
    rows = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, lineno, funcname = func
        rows.append({
            'method': method,
            'N': N,
            'filename': Path(filename).name,
            'filepath': filename,
            'line': lineno,
            'function': funcname,
            'ncalls': nc,
            'ncalls_prim': cc,  # primitive calls (non-recursive)
            'tottime': tt,      # time in this function (excluding subcalls)
            'cumtime': ct,      # time in this function + subcalls
            'percall_tot': tt / nc if nc > 0 else 0,
            'percall_cum': ct / nc if nc > 0 else 0,
        })

    return pd.DataFrame(rows)


# %% Run profiling for all methods and N values
all_dfs = []

for method, integrator_cls in METHODS.items():
    for N in N_VALUES:
        df = profile_case(method, N, integrator_cls)
        all_dfs.append(df)

# Combine all results
df_all = pd.concat(all_dfs, ignore_index=True)

# %% Filter to relevant functions
df_filtered = df_all[
    df_all['filepath'].str.contains('spectral') |
    df_all['function'].str.contains('fft') |
    df_all['function'].str.contains('step') |
    df_all['function'].str.contains('rhs')
].copy()

# %% Save results
df_filtered.to_parquet(output_dir / "cprofile_functions.parquet", index=False)

# %% Display summary
print("\n" + "="*80)
print("FUNCTION-LEVEL PROFILING RESULTS")
print("="*80)

for method in METHODS.keys():
    print(f"\n{method}:")
    method_data = df_filtered[df_filtered['method'] == method]

    # Group by function and sum times across N values
    func_summary = method_data.groupby('function').agg({
        'tottime': 'sum',
        'cumtime': 'sum',
        'ncalls': 'sum'
    }).sort_values('tottime', ascending=False).head(15)

    print(f"  {'Function':40s} {'Tot Time':>12s} {'Cum Time':>12s} {'Calls':>10s}")
    print("  " + "-"*78)
    for func, row in func_summary.iterrows():
        print(f"  {func[:40]:40s} {row['tottime']*1000:10.2f} ms "
              f"{row['cumtime']*1000:10.2f} ms {row['ncalls']:10.0f}")

print("\n" + "="*80)
print(f"✓ Saved to {output_dir}/cprofile_functions.parquet")

# %%
