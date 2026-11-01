"""Figure 2: pull-identity eigenspace geometry.

Panel (a): 2D eigenspace showing the anchored solution W as an exact
per-direction interpolation between the data solution and the anchor --
W lies OFF the uniform-pull segment (the WiSE-FT analog). Panel (b): the
pull-weight curve s(mu) = rho/(mu+rho) with the two operating points.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import figstyle

OI = figstyle.OI
BLUE, VERM, GREEN = OI["ours"], OI["danger"], OI["anchor"]


def _panel_a(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 3.2)
    ax.set_ylim(-1.0, 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    # Coordinate axes (eigen-directions)
    ax.annotate("", xy=(3.0, -0.1), xytext=(-1.3, -0.1),
                arrowprops=dict(arrowstyle="-|>", color="#999999", lw=0.8))
    ax.annotate("", xy=(-1.3, 2.9), xytext=(-1.3, -0.1),
                arrowprops=dict(arrowstyle="-|>", color="#999999", lw=0.8))
    ax.text(1.6, -0.42, "excited direction ($\\mu_1$ large)", ha="center", fontsize=7.2, color="#444444")
    ax.text(-1.25, 2.95, "weakly excited ($\\mu_2$ small)", ha="left", va="bottom", fontsize=7.2, color="#444444")

    w_hat = (2.2, 0.7)
    g = (-0.8, 2.4)
    s1, s2 = 0.15, 0.80
    W = ((1 - s1) * w_hat[0] + s1 * g[0], (1 - s2) * w_hat[1] + s2 * g[1])

    # Off-segment wedge
    ax.add_patch(Polygon([w_hat, g, W], closed=True, facecolor=VERM, alpha=0.08,
                         edgecolor="none"))

    # Uniform-pull segment (WiSE-FT analog)
    ax.plot([w_hat[0], g[0]], [w_hat[1], g[1]], ls="--", color="#a0a0a0", lw=1.0)
    ax.text(0.10, 1.30, "uniform pull\n$s_i = s$ (WiSE-FT)", fontsize=6.8,
            color="#666666", ha="right", va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))

    # Per-direction interpolation guidelines through W
    ax.plot([W[0], W[0]], [w_hat[1], g[1]], ls=":", color="#777777", lw=0.8)
    ax.plot([g[0], w_hat[0]], [W[1], W[1]], ls=":", color="#777777", lw=0.8)

    # Solutions
    ax.plot(*w_hat, marker="o", ms=5.0, color=BLUE, zorder=5)
    ax.text(w_hat[0] + 0.12, w_hat[1] - 0.05, "$\\hat w$", fontsize=8.5,
            color=BLUE, fontweight="bold", va="center")

    ax.plot(*g, marker="s", ms=5.0, color=GREEN, zorder=5)
    ax.text(g[0] - 0.15, g[1] + 0.10, "$G$", fontsize=8.5,
            color=GREEN, fontweight="bold", ha="right", va="bottom")

    ax.plot(*W, marker="*", ms=9.5, color=VERM, mec=VERM, zorder=6)
    ax.text(W[0] + 0.22, W[1] + 0.05, "$W$\n(SPAR-Pull)", fontsize=8.0,
            color=VERM, fontweight="bold", va="center", ha="left")

    # Per-direction annotations with generous clearance
    ax.text(W[0], 0.38, "$s_1 = 0.15$:\ndata wins", fontsize=6.8,
            color=BLUE, ha="center", va="top",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.9))

    ax.text(W[0] + 0.20, 1.35, "$s_2 = 0.80$:\nanchor holds", fontsize=6.8,
            color=GREEN, ha="left", va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.9))

    # Pull identity equation at the bottom
    ax.text(0.65, -0.72,
            "$W_i - w_i = (1{-}s_i)(\\hat w_i - w_i) + s_i(G_i - w_i)$",
            fontsize=7.2, ha="center", va="center",
            bbox=dict(facecolor="#f9f9f9", edgecolor="#d5d5d5", boxstyle="round,pad=0.3", lw=0.5))


def _panel_b(ax):
    x = np.logspace(-2, 2, 200)
    ax.semilogx(x, 1.0 / (1.0 + x), color="#444444", lw=1.2)
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("normalized excitation $\\mu/\\rho$", fontsize=7.8)
    ax.set_ylabel("pull weight $s_i$", fontsize=7.8)
    ax.tick_params(labelsize=7.0)

    # Operating points
    for sv, color, name, val_str, offset in (
        (5.67, BLUE, "$s_1$", "0.15", (6, 2)),
        (0.25, GREEN, "$s_2$", "0.80", (6, 2)),
    ):
        yval = 1.0 / (1.0 + sv)
        ax.plot([sv, sv], [0, yval], ls=":", color=color, lw=0.8)
        ax.plot([1e-2, sv], [yval, yval], ls=":", color=color, lw=0.8)
        ax.plot(sv, yval, marker="o", ms=4.5, color=color, zorder=5)
        ax.annotate(f"{name} = {val_str}", xy=(sv, yval),
                    xytext=offset, textcoords="offset points",
                    fontsize=7.2, color=color, fontweight="bold")

    ax.axhline(1.0, color="#888888", lw=0.6, ls=":")
    ax.axhline(0.0, color="#888888", lw=0.6, ls=":")
    ax.text(1.2e-2, 1.04, "anchor holds ($s \\to 1$)", fontsize=6.8, color="#555555", va="bottom")
    ax.text(1.2e-2, 0.04, "data wins ($s \\to 0$)", fontsize=6.8, color="#555555", va="bottom")


def build():
    fig, axes = plt.subplots(1, 2, figsize=(3.80, 2.40),
                             gridspec_kw={"width_ratios": [1.45, 1.0],
                                          "wspace": 0.28})

    _panel_a(axes[0])
    axes[0].set_title("(a) Eigenspace interpolation", loc="left", fontsize=8.2, fontweight="bold", pad=4)

    _panel_b(axes[1])
    axes[1].set_title("(b) Pull weight curve", loc="left", fontsize=8.2, fontweight="bold", pad=4)

    figstyle.save(fig, "F2_geometry")
    return "SUCCESS"


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    build()

