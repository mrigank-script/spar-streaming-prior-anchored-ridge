"""Finite-D concentration around the deterministic equivalent (Theorem 4).

Theorem 4 is an asymptotic (D -> infinity, zeta = D*omega fixed) statement.
This script measures the finite-D discrepancy |R_MC - R_DE| at fixed zeta as
the dimension D grows, using the *prediction* (population-covariance) risk
that the theorem is stated for.

Two conventions matter.  The committed RMT grid (`rmt_mc_fixed`) scores the
risk with the in-sample empirical covariance, `dW' Sig_hat dW`, which adds an
O(D/T) in-sample/out-of-sample term on top of the deterministic-equivalent
error; here we score with the population covariance `dW' Sigma dW` so the
residual is the finite-D error alone.  The bias is then within the Monte
Carlo standard error at every D, i.e. below the resolution of this study.

Writes results/finite_d_conc.csv.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rls_baselines import make_stream_fac  # noqa: E402

SIGMA2 = 0.01
DS = (64, 128, 256, 512)
ZETAS = (1.0, 2.0)
RHOS = (0.01, 0.1)


def spectrum(kind, D):
    if kind == "iso":
        return np.ones(D)
    if kind == "decay3":
        return np.logspace(0, -3, D)
    raise ValueError(kind)


def de_risk(lam, w, rho):
    """Deterministic-equivalent prediction risk (Theorem 4(i))."""
    D = lam.size
    c = D * w
    lo, hi = 1e-12, 1.0
    for _ in range(300):
        q = 0.5 * (lo + hi)
        x = c * q
        grow = np.expm1(x) / x if x > 1e-12 else 1.0
        if np.mean(lam / (lam + rho * grow)) - q > 0:
            lo = q
        else:
            hi = q
    q = 0.5 * (lo + hi)
    x = c * q
    rho_eff = rho * (np.expm1(x) / x if x > 1e-12 else 1.0)
    Jt = float(np.mean((lam / (lam + rho_eff)) ** 2))
    Ups = (Jt / q) * (1 - (1 - np.exp(-x)) / x) if x > 1e-12 else 0.0
    return SIGMA2 * Ups / (1 - Ups)


def mc_stats(kind, D, w, rho, n=200, T=6000):
    risks = []
    for seed in range(n):
        x, y, w_true, _t1, _t2, Q0, lam0 = make_stream_fac(
            seed=seed, D=D, T=T, sigma=np.sqrt(SIGMA2), spectrum=kind,
            drift="none", noise="gauss")
        w_star = w_true[-1]
        # Newest sample gets the smallest weight; the streams here are
        # stationary (drift="none"), so the ordering does not change the
        # estimate, but non-stationary copies must reverse the arange.
        sw = w * (1 - w) ** np.arange(T)
        Xw = x * np.sqrt(sw)[:, None]
        A = Xw.T @ Xw
        B = Xw.T @ (y * np.sqrt(sw))
        W = np.linalg.solve(A + rho * np.eye(D), B + rho * w_star)
        dW = W - w_star
        proj = Q0.T @ dW                             # population risk in O(D^2)
        risks.append(float(np.sum(lam0 * proj ** 2)))
    r = np.array(risks)
    return r.mean(), r.std(ddof=1) / np.sqrt(n)


def main() -> None:
    rows = []
    print(f"{'spec':<8} {'zeta':>5} {'rho':>6} {'D':>5} | {'MC':>9} {'DE':>9} "
          f"{'bias%':>7} {'se%':>6} {'|bias|/se':>9}")
    for kind in ("iso", "decay3"):
        for zeta in ZETAS:
            for rho in RHOS:
                for D in DS:
                    w = zeta / D
                    lam = spectrum(kind, D)
                    mc, se = mc_stats(kind, D, w, rho)
                    de = de_risk(lam, w, rho)
                    bias = (mc - de) / de
                    rows.append(dict(spectrum=kind, zeta=zeta, rho=rho, D=D,
                                     mc=mc, de=de, bias=bias,
                                     se=se / de, ratio=abs(bias) / (se / de)))
                    print(f"{kind:<8} {zeta:>5} {rho:>6} {D:>5} | {mc:>9.5f} "
                          f"{de:>9.5f} {100*bias:>6.2f}% {100*se/de:>5.2f}% "
                          f"{abs(bias)/(se/de):>9.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "finite_d_conc.csv", index=False)
    sub = df[df.D >= 128]
    print(f"\nD>=128: max |bias| {100*sub.bias.abs().max():.2f}%, "
          f"max |bias|/se {sub.ratio.max():.2f}; "
          f"median |bias|/se {sub.ratio.median():.2f}")
    print("saved results/finite_d_conc.csv")


if __name__ == "__main__":
    main()
