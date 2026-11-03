"""Dead-direction diagnostic: mean-field plug-in vs exact effective dimension.

The window law uses the exact coefficient c' = sigma^2 d_eff(rho) with
d_eff = sum_i (1 - s_i)^2; the runtime rule uses the mean-field plug-in
c_hat = sigma^2 D (1 - s_bar)^2.  On a spectrum with a fraction p of dead
directions (mu_i = 0, so s_i = 1), Jensen gives c_hat = (1-p) c', i.e. the
ratio r = c'/c_hat = 1/(1-p), and the leading cubed-window gain is
2(r - 1).  The plug-in iteration contracts only while r < 3/2, so a spectrum
with more than one third dead directions breaks it.

This script tabulates r and the gain across dead fractions and anchor
strengths, and simulates the cubed-window iteration with the plug-in and with
the exact coefficient on a 50%-dead spectrum.

Writes results/dead_direction.csv.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIGMA2 = 0.01
D = 128


def spectrum(p: float, kind: str = "iso"):
    alive = int(round((1.0 - p) * D))
    lam = np.ones(D) if kind == "iso" else np.logspace(0, -3, D)
    lam[alive:] = 0.0
    return lam


def coeffs(lam, rho):
    s = rho / (lam + rho)                     # dead directions give s = 1
    d_eff = float(np.sum((1.0 - s) ** 2))
    s_bar = float(np.mean(s))
    c_hat = D * (1.0 - s_bar) ** 2
    return d_eff, c_hat


def iterate(lam, rho, exact, omega_star, iters=400, w_min=5e-4, w_max=0.2):
    d_eff, c_hat_unit = coeffs(lam, rho)
    c_true = SIGMA2 * d_eff
    c_hat = SIGMA2 * (d_eff if exact else c_hat_unit)
    nu2 = c_true * omega_star ** 3 / 4.0
    w = w_min
    for _ in range(iters):
        excess = c_true * w / (2.0 - w) + nu2 / (w * w)
        nu2_hat = max(0.0, (excess - c_hat * w / (2.0 - w)) * w * w)
        w = float(np.clip((4.0 * nu2_hat / max(c_hat, 1e-12)) ** (1.0 / 3.0),
                          w_min, w_max))
    return w


def main() -> None:
    rows = []
    print(f"{'p':>5} {'rho':>6} {'d_eff':>9} {'c_hat':>9} {'ratio':>7} "
          f"{'gain':>7} {'contracts':>9}")
    for p in (0.0, 0.25, 0.5, 0.75, 0.9):
        lam = spectrum(p)
        for rho in (1e-2, 1e-1):
            d_eff, c_hat = coeffs(lam, rho)
            ratio = d_eff / max(c_hat, 1e-12)
            gain = 2.0 * (ratio - 1.0)
            rows.append(dict(dead_fraction=p, rho=rho, d_eff=d_eff,
                             c_hat=c_hat, ratio=ratio, gain=gain,
                             plug_in_contracts=bool(gain < 1.0)))
            print(f"{p:>5.2f} {rho:>6.3f} {d_eff:>9.4f} {c_hat:>9.4f} "
                  f"{ratio:>7.3f} {gain:>7.3f} {str(gain < 1.0):>9}")

    # Simulated cubed-window iteration on the 50%-dead spectrum.
    omega_star = 0.02
    lam = spectrum(0.5)
    w_plugin = iterate(lam, 1e-2, exact=False, omega_star=omega_star)
    w_exact = iterate(lam, 1e-2, exact=True, omega_star=omega_star)
    print(f"\n50%-dead spectrum, rho=0.01, exact optimum w*={omega_star}:")
    print(f"  plug-in iteration final w = {w_plugin:.4f} (cap 0.20)")
    print(f"  exact-d_eff iteration final w = {w_exact:.4f}")
    rows.append(dict(dead_fraction=0.5, rho=1e-2, d_eff=float("nan"),
                     c_hat=float("nan"), ratio=2.0, gain=2.0,
                     plug_in_contracts=False, w_plugin=w_plugin,
                     w_exact=w_exact, omega_star=omega_star))
    pd.DataFrame(rows).to_csv(ROOT / "results" / "dead_direction.csv",
                              index=False)
    print("\nsaved results/dead_direction.csv")


if __name__ == "__main__":
    main()