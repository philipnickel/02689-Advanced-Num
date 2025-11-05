from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, RK4


# Configuration tuned to excite aliasing while keeping runtime modest
L = np.pi  # Half-domain length -> domain [-L, L)
T_FINAL = 0.5
MODES = np.array([1, 3, 7, 15, 23], dtype=int)
AMPLITUDES = np.array([1.0, 0.9, 0.6, 0.5, 0.4])
PHASE_SHIFT = 0.7
N_VALUES = np.array([32, 48, 64, 96, 128, 160, 200])
N_REF = 200  # Highest resolution also used as reference
DT_SAFETY = 0.25  # Relative to RK4 stability bound


def broadband_initial(x: np.ndarray) -> np.ndarray:
    """Construct a broadband periodic initial condition."""
    result = np.zeros_like(x)
    for m, amp in zip(MODES, AMPLITUDES):
        result += amp * np.sin(m * x + PHASE_SHIFT * m)
    return result


def periodic_interp(x_src: np.ndarray, u_src: np.ndarray, x_target: np.ndarray, period: float) -> np.ndarray:
    """Periodic linear interpolation from source grid to target."""
    x_mod = np.mod(x_src, period)
    idx = np.argsort(x_mod)
    x_sorted = x_mod[idx]
    u_sorted = u_src[idx]

    # Extend one period to handle wraparound
    x_ext = np.concatenate([x_sorted, x_sorted + period])
    u_ext = np.concatenate([u_sorted, u_sorted])

    x_target_mod = np.mod(x_target, period)
    return np.interp(x_target_mod, x_ext, u_ext)


@dataclass
class RunResult:
    N: int
    dealias: bool
    dt: float
    l2_error: float
    linf_error: float


def solve_case(N: int, dt: float, dealias: bool, save_every: int) -> tuple[np.ndarray, np.ndarray]:
    solver = KdVSolver(N, L, dealias=dealias)
    x = solver.x
    u0 = broadband_initial(x)
    integrator = RK4()

    t_saved, u_hist = solver.solve(
        u0.copy(),
        T_FINAL,
        dt,
        integrator=integrator,
        save_every=save_every,
    )

    if len(u_hist) == 0:
        raise RuntimeError("Solver produced no output.")

    return x, u_hist[-1]


def main() -> None:
    period = 2.0 * L

    # Determine common stable timestep (use smallest across all N plus reference)
    dt_candidates = []
    for N in np.concatenate([N_VALUES, [N_REF]]):
        x = np.linspace(-L, L, N, endpoint=False)
        u0 = broadband_initial(x)
        u_max = float(np.max(np.abs(u0)))
        dt_candidate = KdVSolver.stable_dt(N, L, u_max, integrator_name="rk4")
        dt_candidates.append(dt_candidate)
    dt_base = DT_SAFETY * float(np.min(dt_candidates))
    dt_reference = 0.5 * dt_base
    save_every = max(1, int(np.ceil(T_FINAL / dt_base)))
    save_every_ref = max(1, int(np.ceil(T_FINAL / dt_reference)))

    print(f"dt (working): {dt_base:.3e}, dt_ref: {dt_reference:.3e}, steps: {T_FINAL / dt_base:.1f}")

    # High-resolution reference (dealiased)
    x_ref, u_ref = solve_case(N_REF, dt_reference, dealias=True, save_every=save_every_ref)

    results: list[RunResult] = []

    for N in N_VALUES:
        for use_dealias in (False, True):
            if use_dealias and N == N_REF:
                x, u_num = x_ref, u_ref.copy()
            else:
                x, u_num = solve_case(N, dt_base, dealias=use_dealias, save_every=save_every)

            if N == N_REF:
                u_exact = u_ref
            else:
                u_exact = periodic_interp(x_ref, u_ref, x, period)
            dx = 2 * L / N
            diff = u_num - u_exact
            l2 = float(np.sqrt(np.sum(diff**2) * dx))
            linf = float(np.max(np.abs(diff)))
            results.append(
                RunResult(
                    N=N,
                    dealias=use_dealias,
                    dt=dt_base,
                    l2_error=l2,
                    linf_error=linf,
                )
            )
            label = "de-aliased" if use_dealias else "aliased"
            print(f"N={N:3d} ({label:10s}) -> L2={l2:.3e}, L∞={linf:.3e}")

    df = pd.DataFrame(results)
    pivot = df.pivot(index="N", columns="dealias", values="l2_error").rename(
        columns={False: "error_aliased", True: "error_dealiased"}
    )
    pivot["ratio"] = pivot["error_aliased"] / np.maximum(pivot["error_dealiased"], 1e-15)
    print("\nError comparison (L2 norm):")
    print(pivot.to_string(float_format=lambda v: f"{v:.3e}"))

    out_path = Path("data/A2/ex_c/aliasing_comparison.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved raw results to {out_path}")


if __name__ == "__main__":
    main()
