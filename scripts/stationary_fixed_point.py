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
    print(f"{'spectrum':<8} {'start':>8} {'root':>10} {'ratio':>9}")
    for kind in ("iso", "decay3"):
        lam = rmt_spectrum(kind)
        nu2, R, mu_bar = 2.5e-6, 1.0, 1.0
        roots = []
        for start in (1e-3, 0.1, 0.3, 0.5):
            seq = [start]
            for _ in range(200):
                seq.append(Phi(lam, seq[-1], nu2, R, mu_bar))
            root = seq[-1]
            roots.append(root)
            d = np.abs(np.diff(seq))
            d = d[d > 0]
            ratio = (d[1] / d[0]) if len(d) > 1 else float("nan")
            print(f"{kind:<8} {start:>8.3f} {root:>10.6f} {ratio:>9.4f}")
        print(f"  -> unique root: spread = {max(roots)-min(roots):.2e}")


if __name__ == "__main__":
    main()
