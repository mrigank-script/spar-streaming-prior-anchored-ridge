"""Aggregate the factorial: baseline families vs SPAR arms.

Produces:
  1. Per-config winner table (best-of-sweep per family, per-family best-of-sweep tuning)
  2. Divergence counts (the windup story)
  3. Prior-axis story: oracle / warm / adversarial
  4. Adversarial safety: SPAR excess vs no_anchor
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RES = ROOT / "results"


def load_all():
    frames = []
    for tag in ("d_abrupt", "d_covrot", "d_both", "extras",
                "pu_refresh"):
        f = RES / f"factorial_{tag}.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(
        subset=["spectrum", "drift", "prior", "noise", "seed", "method"],
        keep="last")


def family_best(df, prefix):
    """Best member of a sweep family per (config, seed), by mse_overall."""
    sub = df[df.method.str.startswith(prefix)]
    if sub.empty:
        return None
    idx = sub.groupby(["spectrum", "drift", "prior", "noise", "seed"])["mse_overall"].idxmin()
    return sub.loc[idx]


def main() -> None:
    df = load_all()
    df = df[~df.method.str.startswith("wiseftgrid")]  # keep the chosen-alpha arm
    print(f"loaded {len(df)} rows, {df.groupby(['spectrum','drift','prior','noise','seed']).ngroups} (config,seed) combos")

    key = ["spectrum", "drift", "prior", "noise", "seed"]

    fams = {
        "best_leaky": "leaky\\[", "best_df": "df\\[", "best_arow": "arow\\[",
        "best_kf": "kf\\[", "best_grad": "(sgd|adam)\\[", "best_wiseft": "wiseft\\[",
    }
    best = pd.DataFrame()
    for name, pat in fams.items():
        sub = df[df.method.str.match(pat)]
        if sub.empty:
            continue
        idx = sub.groupby(key)["mse_overall"].idxmin()
        b = sub.loc[idx, key + ["method", "mse_pre", "mse_dur", "mse_post",
                                "mse_overall", "recovery", "diverged"]]
        b["family"] = name
        best = pd.concat([best, b.rename(columns={"method": "family_member"})],
                         ignore_index=True)

    singles = df[df.method.isin(
        ["warm", "no_anchor", "spar_scalar", "spar_diag", "spar_scalar_a1",
         "spar_kf", "spar_pull", "spar_eg", "spar_eg_plus"])]
    singles = singles[key + ["method", "mse_pre", "mse_dur", "mse_post",
                            "mse_overall", "recovery", "diverged"]]
    singles["family"] = singles["method"]
    singles["family_member"] = singles["method"]
    best = pd.concat([best, singles], ignore_index=True)

    # per-config means over seeds
    agg = best.groupby(["family", "spectrum", "drift", "prior", "noise"], dropna=False)[
        ["mse_pre", "mse_dur", "mse_post", "mse_overall"]].mean().reset_index()

    # winners by config (overall)
    print("\n=== Winners per config (mean over 5 seeds, mse_overall) ===")
    for (spectrum, drift, prior, noise), g in agg.groupby(
            ["spectrum", "drift", "prior", "noise"], dropna=False):
        g2 = g.sort_values("mse_overall")
        top = ", ".join(f"{r.family}={r.mse_overall:.4f}" for r in g2.itertuples())
        print(f"[{drift}/{spectrum}/{prior}/{noise}] {top}")

    # headline table: mean over all main-grid configs per family
    print("\n=== Main-grid grand means (mean of mse_overall over configs x seeds) ===")
    main = agg[(agg.noise == "gauss")
               & agg.drift.isin(["abrupt", "covrot", "both"])
               & agg.spectrum.isin(["iso", "decay3", "rankdef"])
               & agg.prior.isin(["oracle", "warm", "adversarial"])]
    fam_order = ["warm", "no_anchor", "best_leaky", "best_df", "best_arow",
                 "best_kf", "best_grad", "best_wiseft",
                 "spar_scalar", "spar_diag", "spar_kf", "spar_scalar_a1"]
    for fam in fam_order:
        g = main[main.family == fam]
        if g.empty:
            continue
        print(f"{fam:<14} overall={g.mse_overall.mean():.4f}  pre={g.mse_pre.mean():.4f}  "
              f"dur={g.mse_dur.mean():.4f}  post={g.mse_post.mean():.4f}")

    # divergence counts
    print("\n=== Divergence counts (any row with diverged=True, raw sweep rows) ===")
    for fam, pat in (("leaky", "leaky\\[",), ("df", "df\\["), ("arow", "arow\\["),
                     ("kf", "kf\\["), ("sgd/adam", "(sgd|adam)\\["),
                     ("spar_*", "spar_")):
        sub = df[df.method.str.match(pat)]
        if sub.empty:
            continue
        per_member = sub.groupby("method")["diverged"].agg(["sum", "count"])
        worst = per_member["sum"].max()
        print(f"{fam:<10} worst member: {int(worst)}/{int(per_member['count'].max())} "
              f"runs diverged")

    # adversarial safety: SPAR vs no_anchor excess
    print("\n=== Adversarial prior safety (per config) ===")
    for (spectrum, drift), g in agg[(agg.prior == "adversarial") & (agg.noise == "gauss")].groupby(
            ["spectrum", "drift"]):
        na = g[g.family == "no_anchor"].mse_overall.mean()
        sc = g[g.family == "spar_scalar"].mse_overall.mean()
        dg = g[g.family == "spar_diag"].mse_overall.mean()
        print(f"[{drift}/{spectrum}] no_anchor={na:.4f} spar_scalar={sc:.4f} "
              f"(excess {sc - na:+.4f}) spar_diag={dg:.4f} (excess {dg - na:+.4f})")

    # oracle prior: diag vs scalar
    print("\n=== Oracle prior: per-direction (diag) vs scalar ===")
    for (spectrum, drift), g in agg[(agg.prior == "oracle") & (agg.noise == "gauss")].groupby(
            ["spectrum", "drift"]):
        sc = g[g.family == "spar_scalar"]
        dg = g[g.family == "spar_diag"]
        print(f"[{drift}/{spectrum}] pre: scalar={sc.mse_pre.mean():.4f} "
              f"diag={dg.mse_pre.mean():.4f} | overall: "
              f"scalar={sc.mse_overall.mean():.4f} diag={dg.mse_overall.mean():.4f}")

    # SPAR vs best-of-all-baselines head-to-head (per config)
    print("\n=== SPAR vs best baseline family (per config, overall) ===")
    base_fams = ["best_leaky", "best_df", "best_arow", "best_kf",
                 "best_grad", "best_wiseft", "no_anchor", "warm"]
    wins_s = 0
    wins_d = 0
    wins_k = 0
    n_cfg = 0
    for (spectrum, drift, prior, noise), g in agg.groupby(
            ["spectrum", "drift", "prior", "noise"], dropna=False):
        gb = g[g.family.isin(base_fams)]
        if gb.empty:
            continue
        bb = gb.mse_overall.min()
        sc = g[g.family == "spar_scalar"].mse_overall.mean()
        dg = g[g.family == "spar_diag"].mse_overall.mean()
        kfam = g[g.family == "spar_kf"].mse_overall.mean()
        n_cfg += 1
        wins_s += int(sc < bb)
        wins_d += int(dg < bb)
        wins_k += int(kfam < bb)
        print(f"[{drift}/{spectrum}/{prior}] best_base={bb:.4f} "
              f"spar_scalar={sc:.4f} {'WIN' if sc < bb else 'loss'} | "
              f"spar_diag={dg:.4f} {'WIN' if dg < bb else 'loss'} | "
              f"spar_kf={kfam:.4f} {'WIN' if kfam < bb else 'loss'}")
    print(f"\nSPAR_scalar beats best-of-family baselines in {wins_s}/{n_cfg} configs")
    print(f"SPAR_diag    beats best-of-family baselines in {wins_d}/{n_cfg} configs")
    print(f"SPAR_kf      beats best-of-family baselines in {wins_k}/{n_cfg} configs")

    best.to_csv(RES / "factorial_best.csv", index=False)
    print(f"\nsaved results/factorial_best.csv")


if __name__ == "__main__":
    main()
