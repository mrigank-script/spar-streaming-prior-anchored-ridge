"""Moving-target decomposition of the fixed-protocol grid.

The 27-configuration main grid is 3 spectra x 3 drift geometries x 3 prior
qualities.  On the 18 moving-target configurations (abrupt and both) the
unanchored information-form leaky method (`leaky_batch`) wins 12/18 against
SPAR-Pull.  This script decomposes that count by spectrum, prior, and drift
geometry and reports the per-configuration margins, so the aggregate claim is
a characterization rather than a bare win count.

Robustness of the wins:
  * decay3 (ill-conditioned): SPAR-Pull wins all 6, by a factor ~2.
  * isotropic:                 leaky wins all 6, by ~2%.
  * rank-deficient:            leaky wins all 6, but by <1% (a tie).
On the moving-target mean SPAR's best arm is ~10% below the leaky method, so
the 12/18 count reflects small-margin wins concentrated on two spectra.

Writes results/moving_targets.csv (one row per moving-target configuration).
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

FAMILIES = ["leaky_batch[", "leaky[", "spar_pull", "spar_eg_plus",
            "spar_kf", "spar_diag", "spar_eg"]
SPAR_FAMS = ["spar_pull", "spar_eg_plus", "spar_kf", "spar_diag", "spar_eg"]
KEY = ["spectrum", "drift", "prior", "noise", "seed"]


def load_grid() -> pd.DataFrame:
    frames = [pd.read_csv(RES / f"factorial_{t}.csv")
              for t in ("d_abrupt", "d_covrot", "d_both", "extras",
                        "pu_refresh")]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=KEY + ["method"], keep="last")
    return df[~df.method.str.startswith("wiseftgrid")]


def family_best(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    sub = df[df.method.str.startswith(prefix)]
    idx = sub.groupby(KEY)["mse_overall"].idxmin()
    return sub.loc[idx, KEY + ["mse_overall"]].rename(
        columns={"mse_overall": prefix.rstrip("[")})


def main() -> None:
    df = load_grid()
    tab = None
    for fam in FAMILIES:
        fb = family_best(df, fam)
        tab = fb if tab is None else tab.merge(fb, on=KEY)
    cfg = tab.groupby(["spectrum", "drift", "prior", "noise"]).mean(
        numeric_only=True).reset_index()
    cfg["spar_best"] = cfg[SPAR_FAMS].min(axis=1)

    main_grid = cfg[(cfg.noise == "gauss")
                    & cfg.spectrum.isin(["iso", "decay3", "rankdef"])
                    & cfg.prior.isin(["oracle", "warm", "adversarial"])
                    & cfg.drift.isin(["abrupt", "covrot", "both"])]
    mt = main_grid[main_grid.drift.isin(["abrupt", "both"])].copy()
    cov = main_grid[main_grid.drift == "covrot"]
    mt["leaky_wins"] = mt["leaky_batch"] < mt.spar_pull

    print("=== moving targets: 18 configs (abrupt, both) ===")
    print(f"leaky_batch (info form) beats spar_pull: "
          f"{int(mt.leaky_wins.sum())}/{len(mt)}")
    print(f"leaky_batch beats spar_best:            "
          f"{int((mt['leaky_batch'] < mt.spar_best).sum())}/{len(mt)}")
    print("\nper-spectrum means and margins:")
    g = mt.groupby("spectrum")[["leaky_batch", "spar_pull", "spar_eg_plus",
                                "spar_kf", "spar_best"]].mean()
    for sp, r in g.iterrows():
        print(f"  {sp:<8} leaky={r.leaky_batch:.4f} spar_best={r.spar_best:.4f} "
              f"ratio={r.leaky_batch / r.spar_best:.3f}")
    print(f"\naggregate moving mean: leaky={mt['leaky_batch'].mean():.4f} "
          f"spar_best={mt.spar_best.mean():.4f}")
    print("\nstatic covrot: spar_pull wins "
          f"{int((cov.spar_pull < cov['leaky_batch']).sum())}/{len(cov)} "
          f"({cov.spar_pull.mean():.4f} vs {cov['leaky_batch'].mean():.4f})")

    out = mt[["spectrum", "drift", "prior", "leaky_batch", "spar_pull",
              "spar_eg_plus", "spar_kf", "spar_best", "leaky_wins"]]
    out.to_csv(RES / "moving_targets.csv", index=False)
    print("\nsaved results/moving_targets.csv")


if __name__ == "__main__":
    main()
