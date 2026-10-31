"""Paper figures (7 total) from the committed result CSVs.

F1 system schematic and F2 pull-identity geometry are didactic figures
built by make_schematic.py / make_geometry.py; F3-F7 are data figures
regenerated from results/*.csv. All output is vector PDF + 300-dpi PNG
preview via scripts/figstyle.py.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import figstyle  # noqa: E402
from make_schematic import build as build_schematic  # noqa: E402
from make_geometry import build as build_geometry  # noqa: E402

OI = figstyle.OI
RES = ROOT / "results"
FIG = RES / "figures"


def load_factorial():
    frames = [pd.read_csv(RES / f"factorial_{t}.csv")
              for t in ("d_abrupt", "d_covrot", "d_both", "extras")]
    df = pd.concat(frames, ignore_index=True)
    return df[(df.noise == "gauss") & (df.drift.isin(["abrupt", "covrot", "both"]))]


def persist_dial_trajectory():
    """Run the two-estimator demo stream once and persist the trajectories."""
    out = RES / "estimator_trajectory.csv"
    if out.exists():
        return out
    from rls_baselines import make_stream_fac, run_spar_kf
    x, y, w_true, t1, t2, Q0, lam0 = make_stream_fac(
        seed=0, D=128, T=12000, spectrum="decay3", drift="abrupt")
    Tw = 2000
    _, rhos, ws = run_spar_kf(x, y, Tw, refresh_every=10, resid_fast=30,
                              rho_drift_floor=0.02, prediction_units=True,
                              return_diag=True)
    event = t1 - Tw
    rows = pd.DataFrame({
        "step": np.arange(len(ws)),
        "rho": np.asarray(rhos)[:len(ws)],
        "w": np.asarray(ws),
    })
    rows["jump"] = ((rows.step >= event) & (rows.step < event + 500)).astype(int)
    rows.to_csv(out, index=False)
    print(f"saved {out}")
    return out


def _lbl(ax, x, y, text, color, ha="left", va="center", fs=6):
    return ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va,
                   bbox=dict(facecolor="white", edgecolor="none",
                             alpha=0.85, pad=1.2))


def fig_cliff(main):
    lam = [0.999, 0.997, 0.995, 0.99]
    win = np.array([1.0 / (1.0 - l) for l in lam])
    m = np.array([main[main.method == f"leaky[{l}]"].mse_overall.mean() for l in lam])
    sd = np.array([main[main.method == f"leaky[{l}]"]
                   .groupby(["spectrum", "drift", "prior"]).mse_overall.mean().std()
                   for l in lam])

    # One-sided upper whisker only: the cross-configuration s.d. can exceed
    # the mean, so a symmetric lower bar would be negative on a log scale.
    yerr_leaky = [np.zeros_like(m), sd]

    fig, ax = plt.subplots(figsize=figstyle.SINGLE_COL)
    ax.errorbar(win, m, yerr=yerr_leaky, fmt="o-", color=OI["danger"], lw=1.2,
                ms=5.0, capsize=2.5, label="leaky RLS")

    lf = [0.98, 0.95, 0.90]
    winf = np.array([1.0 / (1.0 - l) for l in lf])
    mf = np.array([main[main.method == f"df[{l}]"].mse_overall.mean() for l in lf])
    sdf = np.array([main[main.method == f"df[{l}]"]
                    .groupby(["spectrum", "drift", "prior"]).mse_overall.mean().std()
                    for l in lf])
    yerr_df = [np.zeros_like(mf), sdf]
    ax.errorbar(winf, mf, yerr=yerr_df, fmt="s--", color=OI["warn"], lw=1.2,
                ms=5.0, capsize=2.5, label="directional forgetting")

    sk = main[main.method == "spar_pull"].mse_overall.mean()
    sc = main[main.method == "spar_scalar"].mse_overall.mean()
    kf = main[main.method.isin([f"kf[q={q}]" for q in ("1e-06", "1e-05", "0.0001")])
              ].groupby("method").mse_overall.mean().min()

    ax.axhline(sk, color=OI["ours"], ls="-", lw=1.2, label="SPAR-Pull (ours)")
    ax.axhline(sc, color=OI["sky"], ls=":", lw=1.2, label="SPAR-Pull-S")
    ax.axhline(kf, color=OI["base"], ls="-.", lw=1.0, label="Kalman (tuned $q$, true $\\sigma^2$)")

    # Shaded windup regime
    ax.axvspan(5, 128, color=OI["danger"], alpha=0.06)
    ax.text(22, 3.2e6, "windup regime\n(short windows)", fontsize=7.2, color="#666666",
            ha="center", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))

    # The one-step jump annotation
    ax.annotate("one $\\lambda$ step:\n$182 \\to 1.2\\times10^6$",
                xy=(100, 1.2e6), xytext=(180, 1.5e5), fontsize=7.2,
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=0.8),
                bbox=dict(facecolor="white", edgecolor="#cccccc", lw=0.5,
                          boxstyle="round,pad=0.3", alpha=0.9))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(8, 2200)
    ax.set_ylim(8e-2, 6e6)
    ax.set_xlabel("nominal window $1/(1-\\lambda)$ (log)")
    ax.set_ylabel("mean MSE over 27 configs (log)")
    ax.grid(alpha=0.25, which="both")

    # Clean legend placed in open mid-left quadrant (above yellow points, below windup label)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.75), fontsize=6.8,
              framealpha=0.92, borderpad=0.3, handletextpad=0.4)

    figstyle.save(fig, "F3_lambda_cliff")


def fig_dials():
    df = pd.read_csv(RES / "estimator_trajectory.csv")
    ev = df[df.jump == 1]
    xmax = df.step.iloc[-1] + 1200
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.45), sharex=True)

    # Top panel: rho_t
    axes[0].plot(df.step, df.rho, lw=1.0, color=OI["ours"])
    axes[0].text(df.step.iloc[-1] + 150, df.rho.iloc[-1], "$\\rho_t$",
                 fontsize=8.5, color=OI["ours"], va="center", fontweight="bold")
    axes[0].axhline(0.02, color="#888888", ls=":", lw=0.8)
    axes[0].text(5200, 0.007, "floor $\\rho_{\\min} = 0.02$", fontsize=7.2, color="#555555",
                 ha="center", va="top",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8))
    axes[0].set_ylabel("anchor strength $\\rho_t$")

    # Bottom panel: omega_t
    axes[1].plot(df.step, df["w"], lw=1.0, color=OI["anchor"])
    axes[1].text(df.step.iloc[-1] + 150, df["w"].iloc[-1], "$\\omega_t$",
                 fontsize=8.5, color=OI["anchor"], va="center", fontweight="bold")
    axes[1].axhline(0.05, color="#888888", ls=":", lw=0.8)
    axes[1].text(6000, 0.055, "cap $\\omega_{\\max} = 0.05$", fontsize=7.2, color="#555555",
                 ha="center", va="bottom",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8))
    axes[1].set_yscale("log")
    axes[1].set_ylabel("window $\\omega_t$")
    axes[1].set_xlabel("stream step")

    for a in axes:
        a.set_xlim(0, xmax)
        a.axvspan(ev.step.min(), ev.step.max(), color=OI["danger"], alpha=0.12)
        a.grid(alpha=0.25)

    # Event label placed cleanly centered over the shaded jump band
    axes[0].text((ev.step.min() + ev.step.max()) / 2, 0.22, "abrupt jump", fontsize=7.2,
                 color=OI["danger"], ha="center", va="bottom", fontweight="bold",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8))

    figstyle.save(fig, "F4_two_dials")


def fig_noise_floor(main):
    specs = ["iso", "decay3", "rankdef"]
    labels = ["isotropic", "ill-cond.\n($\\kappa = 10^3$)", "rank\ndeficient"]
    xs = np.arange(3)
    fig, ax = plt.subplots(figsize=(2.5, 2.1))

    for off, arm, color, name in ((-0.10, "spar_pull", OI["ours"], "SPAR-Pull"),
                                  (0.10, None, OI["base"], "best Kalman")):
        mu, sd = [], []
        for s in specs:
            if arm:
                sub = main[(main.method == arm) & (main.spectrum == s)].mse_pre
            else:
                per = (main[main.method.str.startswith("kf") & (main.spectrum == s)]
                       .groupby("method").mse_pre.mean())
                sub = main[(main.method == per.idxmin()) & (main.spectrum == s)].mse_pre
            mu.append(sub.mean())
            sd.append(sub.std())
        ax.errorbar(xs + off, mu, yerr=sd, fmt="o", ms=5.0, capsize=2.5,
                    color=color, label=name)

    # Reference noise floor line
    ax.axhline(0.01, color=OI["ref"], ls="--", lw=0.9, label="$\\sigma^2 = 0.01$ (noise floor)")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel("pre-drift MSE")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.0070, 0.0205)

    # Clean legend placed above the plot frame with all three entries
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              fontsize=6.5, frameon=False, borderpad=0, handletextpad=0.3, columnspacing=0.7)
    ax.grid(axis="y", alpha=0.25, ls=":")
    figstyle.save(fig, "F5_noise_floor")


def fig_prior_immunity(main):
    priors = ["oracle", "warm", "adversarial"]
    labels = ["oracle\nprior", "warm\nprior", "adversarial\nprior"]
    xs = np.arange(3)
    fig, ax = plt.subplots(figsize=(2.5, 2.1))

    sk_m = [main[(main.method == "spar_pull") & (main.prior == p)].mse_overall.mean()
            for p in priors]
    sk_sd = [main[(main.method == "spar_pull") & (main.prior == p)].mse_overall.std()
             for p in priors]
    wf_m = []
    for p in priors:
        per = (main[(main.method.str.startswith("wiseft")) & (main.prior == p)]
               .groupby("method").mse_overall.mean())
        wf_m.append(main[(main.method == per.idxmin()) & (main.prior == p)]
                    .mse_overall.mean())

    ax.errorbar(xs, sk_m, yerr=sk_sd, fmt="o", ms=5.0, capsize=2.5,
                color=OI["ours"], label="SPAR-Pull")
    ax.plot(xs, wf_m, "s", ms=5.0, color=OI["warn"], label="WiSE-FT (best $\\alpha$)")

    # Spread bracket lifted above SPAR-Pull points without colliding with axes
    top = 0.75
    ax.annotate("", xy=(0, top), xytext=(2, top),
                arrowprops=dict(arrowstyle="|-|", color="#555555", lw=0.8))
    spread = (max(sk_m) / min(sk_m) - 1) * 100
    ax.text(1.0, top * 1.35, f"{spread:.1f}% spread", fontsize=7.5,
            ha="center", color="#333333", fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel("overall MSE (log)")
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(0.005, 50)

    # Clean legend above the plot frame
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 1.02), ncol=2,
              fontsize=7.2, frameon=False, borderpad=0, handletextpad=0.3, columnspacing=0.8)
    ax.grid(axis="y", alpha=0.25, which="both", ls=":")
    figstyle.save(fig, "F6_prior_immunity")


def fig_regret():
    rs = pd.read_csv(RES / "regret_scaling.csv")
    fig, ax = plt.subplots(figsize=(2.6, 2.2))

    for decay, fmt, name, color in (
        (0.0, "o-", "isotropic", OI["ours"]),
        (3.0, "s--", "ill-cond. ($\\kappa{=}10^3$)", OI["base"]),
    ):
        sub = rs[rs.spectrum_decay == decay]
        slope = np.polyfit(np.log(sub["T"]), np.log(sub.per_step_regret), 1)[0]
        label = f"{name} (slope {slope:.2f})"
        ax.loglog(sub["T"], sub.per_step_regret, fmt, color=color,
                  ms=4.5, lw=1.1, label=label)
        ax.errorbar(sub["T"], sub.per_step_regret, yerr=sub.regret_std / sub["T"],
                    fmt="none", ecolor=color, capsize=2.5, lw=0.8, alpha=0.7)

    t0 = rs[rs.spectrum_decay == 0.0]
    y0 = t0.per_step_regret.iloc[0]
    T0 = t0["T"].iloc[0]
    tt = np.array([T0, t0["T"].iloc[-1] * 1.1], dtype=float)

    # Theoretical O(sqrt(T)) total regret => O(T^{-1/2}) per-step bound
    ax.loglog(tt, y0 * np.sqrt(T0 / tt), color="#555555", ls="--", lw=1.0)
    # Mechanical amortization limit of a fixed-length drift phase
    ax.loglog(tt, y0 * (T0 / tt), "k:", lw=1.0)

    # Callout labels clearly placed without intersecting any lines
    _lbl(ax, 28000, 0.012, "$T^{-1/2}$ per-step bound", "#555555", fs=6.5)
    _lbl(ax, 20000, 3.0e-4, "$T^{-1}$ amortization", "#222222", fs=6.5)

    ax.set_xlabel("horizon $T$")
    ax.set_ylabel("per-step regret vs best fixed $s$")
    ax.set_xlim(1400, 220000)
    ax.set_ylim(2e-5, 6e-2)
    ax.grid(alpha=0.25, which="both")

    # Clean legend in bottom left where space is completely open
    ax.legend(loc="lower left", fontsize=7.2, framealpha=0.9,
              borderpad=0.3, handletextpad=0.4)

    figstyle.save(fig, "F7_regret_scaling")


def main():
    build_schematic()
    build_geometry()
    persist_dial_trajectory()
    main_df = load_factorial()
    fig_cliff(main_df)
    fig_dials()
    fig_noise_floor(main_df)
    fig_prior_immunity(main_df)
    fig_regret()
    for old in ("F1_lambda_cliff.png", "F2_two_dials.png", "F3_noise_floor.png",
                "F4_prior_immunity.png", "F5_regret_scaling.png"):
        p = FIG / old
        if p.exists():
            p.unlink()
    stray = RES / "regret_scaling.png"
    if stray.exists():
        stray.unlink()
    print("7 figures written to", FIG)


if __name__ == "__main__":
    main()
