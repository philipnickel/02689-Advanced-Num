"""
Line-by-line profiling using line_profiler.

Make sure the functions in spectral.tdp are decorated with @profile:
- KdVSolver.solve
- KdVSolver.rhs
- RK3.step
- RK4.step

Usage
-----
    uv run kernprof -l Exercises/exercise_g/compute_profiling_lines.py
    uv run python Exercises/exercise_g/compute_profiling_lines.py --parse
"""

# %% Imports
from __future__ import annotations

import sys
import pickle
import linecache
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


def categorize(source: str) -> str:
    """Categorize source line by operation type."""
    if 'np.fft.fft(' in source and 'ifft' not in source:
        return 'FFT'
    elif 'np.fft.ifft(' in source:
        return 'IFFT'
    elif any(kw in source for kw in ['= rhs(', 'integrator.step(', 'solver.solve(']):
        return 'Function Call'
    elif 'self.ik' in source:
        return 'Wavenumber'
    elif 'return' in source and any(op in source for op in ['-', '*', '+']):
        return 'Return'
    else:
        return 'Other'


# %% Run profiling (called by kernprof)
if '--parse' not in sys.argv:
    for method, integrator_cls in METHODS.items():
        for N in N_VALUES:
            dt = estimate_stable_dt(method, N)
            solver = KdVSolver(N, L, dealias=False)
            x = solver.x
            u0 = soliton(x, 0.0, SOLITON_SPEED, SOLITON_X0)
            integrator = integrator_cls()
            save_every = max(1, int(np.ceil(T_FINAL / dt)))

            print(f"Profiling {method} @ N={N}: dt={dt:.3e}")

            # This will be profiled by kernprof
            solver.solve(u0.copy(), T_FINAL, dt, save_every=save_every, integrator=integrator)

# %% Parse .lprof file (run with --parse flag)
if '--parse' in sys.argv:
    lprof_file = "compute_profiling_lines.py.lprof"
    print(f"Parsing {lprof_file}...")

    with open(lprof_file, 'rb') as f:
        stats = pickle.load(f)

    unit = getattr(stats, 'unit', 1e-6)

    rows = []
    for (filename, start_line, func_name), timings in stats.timings.items():
        for lineno, nhits, time_raw in timings:
            source = linecache.getline(filename, lineno).strip()
            time_s = time_raw * unit

            rows.append({
                'function': func_name,
                'filename': Path(filename).name,
                'line': lineno,
                'hits': nhits,
                'time_s': time_s,
                'time_ms': time_s * 1e3,
                'time_us': time_s * 1e6,
                'source': source,
            })

    df = pd.DataFrame(rows)
    df['category'] = df['source'].apply(categorize)

    # Save
    output_dir = Path("data/A2/ex_g")
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_dir / "line_profiler_data.parquet", index=False)

    # Display summary
    print("\n" + "="*80)
    print("LINE-BY-LINE PROFILING RESULTS")
    print("="*80)

    # Filter out function calls for clean breakdown
    df_work = df[df['category'] != 'Function Call']

    # Category breakdown
    print("\nTime by Category (excluding function call overhead):")
    cat_time = df_work.groupby('category')['time_s'].sum()
    total = cat_time.sum()

    print(f"\n  {'Category':20s} {'Time':>12s} {'Percent':>10s}")
    print("  " + "-"*44)
    for cat, time_s in cat_time.sort_values(ascending=False).items():
        pct = (time_s / total) * 100
        print(f"  {cat:20s} {time_s*1000:10.2f} ms {pct:9.1f}%")
    print("  " + "-"*44)
    print(f"  {'TOTAL':20s} {total*1000:10.2f} ms {100.0:9.1f}%")

    # Top 15 lines
    print("\n" + "="*80)
    print("Top 15 Slowest Lines")
    print("="*80)

    top = df_work.nlargest(15, 'time_s')
    print(f"\n  {'Time':>10s} {'Hits':>8s} {'Category':>12s} {'Source':50s}")
    print("  " + "-"*82)
    for _, row in top.iterrows():
        print(f"  {row['time_ms']:9.1f} ms {row['hits']:8.0f} "
              f"{row['category']:12s} {row['source'][:50]}")

    print("\n" + "="*80)
    print(f"✓ Saved to {output_dir}/line_profiler_data.parquet")

# %%
