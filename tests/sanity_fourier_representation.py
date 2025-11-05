from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df = df.copy()
    df["dealias"] = df["dealias"].astype(str)
    df["method"] = df["method"].astype(str)
    return df


def compare_representation_errors(
    base_dir: Path,
    integrator: str = "RK4",
    dealias: str = "De-aliased",
) -> pd.DataFrame:
    real_path = base_dir / "kdv_spatial_convergence.parquet"
    complex_path = base_dir / "kdv_spatial_convergence_complex.parquet"

    df_real = _load_table(real_path)
    df_complex = _load_table(complex_path)

    sel_real = df_real[(df_real["method"] == integrator) & (df_real["dealias"] == dealias)]
    sel_complex = df_complex[(df_complex["method"] == integrator) & (df_complex["dealias"] == dealias)]

    merged = pd.merge(
        sel_real[["N", "Error"]].rename(columns={"Error": "Error_real"}),
        sel_complex[["N", "Error"]].rename(columns={"Error": "Error_complex"}),
        on="N",
        how="inner",
    ).sort_values("N")

    merged["ratio_complex_over_real"] = merged["Error_complex"] / merged["Error_real"]
    return merged


def main() -> None:
    base_dir = Path("data/A2/ex_c")
    df = compare_representation_errors(base_dir)

    print("N  |  Error(real)        Error(complex)    ratio")
    print("-- | ------------------  ----------------  ---------")
    for row in df.itertuples():
        print(
            f"{int(row.N):2d} | "
            f"{row.Error_real: .6e}  "
            f"{row.Error_complex: .6e}  "
            f"{row.ratio_complex_over_real: .4f}"
        )

    max_diff = np.max(np.abs(df["Error_real"] - df["Error_complex"]))
    print(f"\nMax absolute error difference: {max_diff:.3e}")


if __name__ == "__main__":
    main()
