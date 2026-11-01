"""Summarize the revision factorial grid (prediction-unit SPAR arms).

Reads the four factorial CSVs, drops duplicate (config, seed, method)
rows keeping the last write, and reports:

  * per-method mean / s.d. / worst over the 27 main configurations x 5 seeds
  * the global best single hyperparameter per baseline family
  * marginal means by spectrum, drift geometry, and prior quality
  * covariance-rotation subset and paired tests for SPAR-EG+
  * divergence counts

Usage: py -X utf8 scripts/summarize_revision.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

KEY = ["spectrum", "drift", "prior", "noise", "seed"]
MAIN_DRIFTS = ("abrupt", "covrot", "both")
MAIN_SPECTRA = ("iso", "decay3", "rankdef")
MAIN_PRIORS = ("oracle", "warm", "adversarial")


def load() -> pd.DataFrame:
    frames = []
    for tag in ("d_abrupt", "d_covrot", "d_both", "extras", "pu_refresh"):
        p = RES / f"factorial_{tag}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=KEY + ["method"], keep="last")
    return df


def stats_block(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for m in methods:
        g = df[df.method == m]
        if g.empty:
            continue
        rows.append(dict(method=m, n=len(g), mean=g.mse_overall.mean(),
                         sd=g.mse_overall.std(),
                         geo=float(np.exp(np.log(
                             g.mse_overall.clip(lower=1e-12)).mean())),
                         worst=g.mse_overall.max(),
                         pre=g.mse_pre.mean(), dur=g.mse_dur.mean(),
                         post=g.mse_post.mean(),
                         div=int(g.diverged.sum())))
    return pd.DataFrame(rows)


def family_best(df: pd.DataFrame, prefix: str):
    sub = df[df.method.str.startswith(prefix)]
    if sub.empty:
        return None
    means = sub.groupby("method").mse_overall.mean()
    member = means.idxmin()
    g = sub[sub.method == member]
    return dict(method=member, mean=g.mse_overall.mean(),
                sd=g.mse_overall.std(), worst=g.mse_overall.max(), n=len(g))


def main() -> None:
    df = load()
    main = df[df.noise.eq("gauss") & df.drift.isin(MAIN_DRIFTS)
              & df.spectrum.isin(MAIN_SPECTRA) & df.prior.isin(MAIN_PRIORS)]
    print(f"loaded {len(df)} unique rows; main grid {len(main)} rows, "
          f"{main.groupby(['spectrum', 'drift', 'prior', 'seed']).ngroups} "
          f"config-seeds")

    base_methods = [
        "leaky[0.999]", "leaky[0.997]", "leaky[0.995]", "leaky[0.99]",
        "leaky[0.98]", "leaky[0.95]", "leaky[0.9]",
        "df[0.999]", "df[0.997]", "df[0.995]", "df[0.99]",
        "df[0.98]", "df[0.95]", "df[0.9]",
        "arow[0.01]", "arow[0.05]", "arow[0.1]", "arow[0.3]", "arow[0.5]",
        "adam[0.0003]", "adam[0.001]", "adam[0.003]", "adam[0.01]",
        "sgd[0.003]", "sgd[0.01]",
        "kf[q=1e-06]", "kf[q=1e-05]", "kf[q=0.0001]",
        "wiseft[a=1.0]", "wiseft[best]",
        "spar_kf", "spar_kf2", "spar_scalar", "spar_diag",
        "spar_pull", "spar_eg", "spar_eg_plus",
    ]
    s = stats_block(main, base_methods)
    print("\n=== per-method main-grid stats (sorted by mean) ===")
    print(s.sort_values("mean").to_string(index=False,
                                          float_format=lambda x: f"{x:.4f}"))

    print("\n=== global best single hyperparameter per family ===")
    fams = {
        "leaky RLS (covariance)": "leaky[",
        "leaky/information form": "leaky_batch[",
        "directional forgetting": "df[",
        "AROW": "arow[",
        "Kalman (true noise)": "kf[",
        "SGD/Adam": "(sgd|adam)\\[",
        "WiSE-FT": "wiseft[",
        "VFF covariance": "vff[",
        "VFF information": "vff_info[",
    }
    for name, pat in fams.items():
        if pat.startswith("("):
            sub = main[main.method.str.match(pat)]
        else:
            sub = main[main.method.str.startswith(pat)]
        if sub.empty:
            continue
        means = sub.groupby("method").mse_overall.mean()
        member = means.idxmin()
        g = sub[sub.method == member]
        print(f"{name:<26} {member:<24} mean={g.mse_overall.mean():.4f} "
              f"sd={g.mse_overall.std():.4f} worst={g.mse_overall.max():.4f} "
              f"n={len(g)}")

    sel = ["spar_pull", "spar_eg", "spar_eg_plus", "spar_kf"]
    print("\n=== SPAR arms: marginal means ===")
    for axis, vals in (("spectrum", MAIN_SPECTRA), ("drift", MAIN_DRIFTS),
                       ("prior", MAIN_PRIORS)):
        print(f"-- {axis} --")
        for m in sel:
            g = main[main.method == m]
            if g.empty:
                continue
            line = " ".join(f"{v}={g[g[axis] == v].mse_overall.mean():.4f}"
                            for v in vals)
            print(f"{m:<14} {line}")

    print("\n=== covariance-rotation subset ===")
    cov = main[main.drift == "covrot"]
    for m in sel:
        g = cov[cov.method == m]
        if not g.empty:
            print(f"{m:<14} mean={g.mse_overall.mean():.4f} "
                  f"worst={g.mse_overall.max():.4f}")

    print("\n=== paired EG+ vs Pull, per-configuration means ===")
    piv = (main[main.method.isin(["spar_pull", "spar_eg_plus"])]
           .groupby(["spectrum", "drift", "prior", "method"])
           .mse_overall.mean().unstack())
    if {"spar_pull", "spar_eg_plus"} <= set(piv.columns):
        diff = piv["spar_eg_plus"] - piv["spar_pull"]
        t = stats.ttest_rel(piv["spar_eg_plus"], piv["spar_pull"])
        w = stats.wilcoxon(piv["spar_eg_plus"], piv["spar_pull"])
        print(f"EG+ wins {int((diff < 0).sum())}/{len(diff)}; "
              f"mean delta={diff.mean():+.4f}; t p={t.pvalue:.4g}; "
              f"Wilcoxon p={w.pvalue:.4g}")

    print("\n=== divergence counts (main grid) ===")
    for pat, name in (("leaky[", "leaky"), ("df[", "df"),
                      ("arow[", "arow"), ("kf[", "kf"),
                      ("(sgd|adam)\\[", "sgd/adam"), ("spar_", "spar")):
        if pat.startswith("("):
            sub = main[main.method.str.match(pat)]
        else:
            sub = main[main.method.str.startswith(pat)]
        if sub.empty:
            continue
        per = sub.groupby("method").diverged.agg(["sum", "count"])
        print(f"{name:<10} worst {int(per['sum'].max())}/{int(per['count'].max())}")

    print("\n=== LaTeX table rows (mean / geo / worst) ===")
    for m in ["leaky[0.997]", "leaky_batch[0.99]", "sgd[0.01]", "adam[0.01]",
              "spar_kf", "spar_pull", "spar_eg", "spar_eg_plus"]:
        g = main[main.method == m]
        if not g.empty:
            geo = float(np.exp(np.log(g.mse_overall.clip(lower=1e-12)).mean()))
            print(f"{m:<20} {g.mse_overall.mean():.4f} & "
                  f"{geo:.4f} & {g.mse_overall.max():.4f} \\\\")


if __name__ == "__main__":
    main()
