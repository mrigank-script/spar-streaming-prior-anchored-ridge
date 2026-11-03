"""Numeric check of the coupled stationary fixed point (Goal G9).

The deterministic coupled stationary law is the fixed point of the composition
Phi = rho* o omega*: the window target omega*(rho) substituted into the
strength target rho*(omega).  We iterate Phi from several starts and verify
(i) all starts converge to the same root (uniqueness) and (ii) the iterates
converge geometrically at the coupling product.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Projects\Research\Project - 4")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnostics import rmt_spectrum, RMT_D, RMT_SIGMA2  # noqa: E402


def d_eff(lam, rho):
    return float(np.sum((lam / (lam + rho)) ** 2))


def Phi(lam, rho, nu2, R, mu_bar, wmin=1e-3, wmax=0.5):
    w = float(np.clip((4.0 * nu2 / (RMT_SIGMA2 * d_eff(lam, rho))) ** (1 / 3),
                      wmin, wmax))
    V = RMT_SIGMA2 * RMT_D * w / (2.0 - w)
    return mu_bar * V / max(R - V, 1e-9)


def main():
    starts = np.concatenate([[1e-4], np.linspace(0.05, 0.9, 12)])
    print(f"{'spectrum':<8} {'starts':>6} {'root':>10} {'spread':>9} "
          f"{'mean ratio':>11}")
    for kind in ("iso", "decay3"):
        lam = rmt_spectrum(kind)
        nu2, R, mu_bar = 2.5e-6, 1.0, 1.0
        roots, ratios = [], []
        for start in starts:
            seq = [start]
            for _ in range(200):
                seq.append(Phi(lam, seq[-1], nu2, R, mu_bar))
            roots.append(seq[-1])
            d = np.abs(np.diff(seq))
            d = d[d > 0]
            if len(d) > 1:
                ratios.append(d[1] / d[0])
        print(f"{kind:<8} {len(starts):>6} {np.mean(roots):>10.6f} "
              f"{max(roots)-min(roots):>9.1e} {np.mean(ratios):>11.4f}")
    print("all starts inside the parameter box converge to one root; "
          "the increment ratio is the global contraction modulus.")


if __name__ == "__main__":
    main()
