"""Figure 1: SPAR system schematic (full-width architectural flow diagram).

Stream -> leaky statistics -> anchored solve -> prediction -> residual
telemetry -> two self-calibrating estimators (strength rho*, window w*) feeding
back into the solve and the statistics; anchor store with the pull-identity
refresh (beta_i = 1 - s_i). Panel (b): per-direction pull weights.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Wedge, Circle, Rectangle, Ellipse

import figstyle

OI = figstyle.OI
BLUE, VERM, GREEN, GRAY = OI["ours"], OI["danger"], OI["anchor"], "#666666"


def _box(ax, x, y, w, h, edge, lw=0.9, face="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25",
                                facecolor=face, edgecolor=edge, lw=lw))


def _pill(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.5",
                                facecolor="white", edgecolor=GRAY, lw=0.5))


def _arrow(ax, p0, p1, color, style="-", rad=0.0, lw=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8,
                                 color=color, lw=lw, linestyle=style,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def _step(ax, n, x, y):
    ax.add_patch(Circle((x, y), 1.15, facecolor="white", edgecolor="black", lw=0.7, zorder=10))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=7.2, fontweight="bold", zorder=11)


def _gauge(ax, cx, cy, needle_deg, name, formula, sign, fx, fy):
    r = 2.1
    ax.add_patch(Wedge((cx, cy), r, 15, 165, width=0.55,
                       facecolor="#f2f2f2", edgecolor="#444444", lw=0.7))
    a = np.deg2rad(needle_deg)
    ax.plot([cx, cx + (r - 0.45) * np.cos(a)], [cy, cy + (r - 0.45) * np.sin(a)],
            color=VERM, lw=1.1, zorder=5)
    ax.plot([cx], [cy], marker="o", ms=2.4, color="#444444", zorder=6)
    ax.text(cx, cy + r + 0.8, name, ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.text(fx, fy, formula, ha="left", va="center", fontsize=7.2, color="#222222")
    ax.text(fx, fy - 1.6, sign, ha="left", va="center", fontsize=6.8,
            style="italic", color="#555555")


def _cylinder(ax, x0, y0, w, h, edge):
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white", edgecolor=edge, lw=1.0))
    ax.add_patch(Ellipse((x0 + w, y0 + h / 2), 1.6, h, facecolor="white",
                         edgecolor=edge, lw=1.0))
    ax.add_patch(Ellipse((x0, y0 + h / 2), 1.6, h, facecolor="white",
                         edgecolor=edge, lw=1.0))


def build():
    fig = plt.figure(figsize=(6.75, 2.90), layout=None)

    # ========================== PANEL (a) ==========================
    # Aspect ratio: 100 : 26.5
    ax = fig.add_axes([0.015, 0.35, 0.970, 0.63])
    ax.set_xlim(0, 100)
    ax.set_ylim(7.0, 33.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0.5, 32.8, "(a) SPAR streaming architecture", fontsize=9.2, fontweight="bold")

    # 1 stream cylinder
    _cylinder(ax, 1.8, 14.3, 7.8, 7.2, BLUE)
    ax.text(5.7, 19.8, "stream", ha="center", fontsize=7.8, fontweight="bold")
    ax.text(5.7, 17.9, "$(x_t, y_t)$", ha="center", fontsize=7.5)
    ax.text(5.7, 16.0, "$t = 1, \\dots, T$", ha="center", fontsize=6.8, color="#444444")
    _step(ax, 1, 1.0, 22.2)

    # 2 leaky statistics
    _box(ax, 13.8, 14.0, 18.2, 7.6, GRAY)
    ax.text(22.9, 19.9, "leaky statistics", ha="center", fontsize=8.0, fontweight="bold")
    ax.text(22.9, 17.5, "$A \\leftarrow \\alpha A + \\phi\\phi^\\top,\\; B \\leftarrow \\alpha B + \\phi y$",
            ha="center", fontsize=7.0)
    ax.text(22.9, 15.4, "$n \\leftarrow \\alpha n + 1$", ha="center", fontsize=7.0)
    _step(ax, 2, 31.0, 21.0)

    # 3 anchored solve
    _box(ax, 38.0, 13.8, 20.4, 8.0, BLUE, lw=1.2)
    ax.text(48.2, 20.2, "anchored solve", ha="center", fontsize=8.5, fontweight="bold", color=BLUE)
    ax.text(48.2, 16.6, "$W = (A + n\\rho I)^{-1}(B + n\\rho G)$",
            ha="center", fontsize=7.5)
    _step(ax, 3, 57.4, 21.0)

    # 4 prediction
    circ = Circle((67.6, 17.8), 2.5, facecolor="white", edgecolor=BLUE, lw=1.1)
    ax.add_patch(circ)
    ax.text(67.6, 18.8, "predict", ha="center", va="center", fontsize=7.2)
    ax.text(67.6, 16.8, "$\\hat y_t$", ha="center", va="center", fontsize=7.8, fontweight="bold")
    _step(ax, 4, 70.8, 20.8)

    # 5 residual telemetry
    _box(ax, 74.8, 13.8, 23.6, 8.0, GRAY, face="#fafafa")
    ax.text(86.6, 20.2, "residual telemetry", ha="center", fontsize=8.0, fontweight="bold")
    ax.text(86.6, 18.6, "$r_t = y_t - \\hat y_t$", ha="center", fontsize=7.5)
    _pill(ax, 76.0, 15.0, 6.6, 1.8)
    _pill(ax, 83.3, 15.0, 6.6, 1.8)
    _pill(ax, 90.6, 15.0, 6.6, 1.8)
    ax.text(79.3, 15.9, "fast EMA", ha="center", va="center", fontsize=6.8)
    ax.text(86.6, 15.9, "slow EMA", ha="center", va="center", fontsize=6.8)
    ax.text(93.9, 15.9, "detector", ha="center", va="center", fontsize=6.8)
    _step(ax, 5, 97.4, 21.0)

    # data path arrows
    _arrow(ax, (10.0, 17.9), (13.4, 17.9), BLUE, lw=1.2)
    _arrow(ax, (32.7, 17.8), (37.3, 17.8), BLUE, lw=1.2)
    _arrow(ax, (59.1, 17.8), (64.5, 17.8), BLUE, lw=1.2)
    _arrow(ax, (70.6, 17.8), (74.1, 17.8), BLUE, lw=1.2)

    # control path: telemetry -> bus -> estimators
    # bus line at y = 24.2
    _arrow(ax, (76.5, 22.0), (76.5, 24.2), VERM, style="--", lw=0.9)
    _arrow(ax, (76.5, 24.2), (18.5, 24.2), VERM, style="--", lw=0.9)
    ax.text(78.5, 24.7, "$\\hat\\sigma^2,\\; \\hat\\nu^2_\\Sigma,\\; \\hat C$",
            ha="left", va="bottom", fontsize=7.2, color="#222222",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))

    # 6 estimators
    # inputs to estimators (going UP from bus at y=24.2)
    _arrow(ax, (24.0, 24.2), (24.0, 26.0), VERM, style="--", lw=0.9)
    _arrow(ax, (49.0, 24.2), (49.0, 26.0), VERM, style="--", lw=0.9)

    _gauge(ax, 24.0, 26.4, 60, "window estimator $\\omega^*$",
           "$\\omega^* = (4\\hat\\nu^2 / \\hat c')^{1/3}$", "$\\hat\\nu \\uparrow \\Rightarrow \\omega \\uparrow$",
           27.4, 28.3)
    _gauge(ax, 49.0, 26.4, 120, "strength estimator $\\rho^*$",
           "$\\rho^* = \\bar\\mu \\hat V / \\hat C$", "$\\hat C \\uparrow \\Rightarrow \\rho \\downarrow$",
           52.4, 28.3)

    # Step (6) badge centered at the top between the two estimators
    _step(ax, 6, 38.2, 30.5)

    # feedback outputs from estimators (going DOWN to statistics / solve)
    # Drift output: alpha = 1 - w into leaky statistics
    _arrow(ax, (20.5, 26.2), (20.5, 22.1), VERM, style="--", lw=0.9)
    ax.text(19.8, 23.1, "$\\alpha = 1 - \\omega$", ha="right", va="center", fontsize=7.0,
            color=VERM, bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.9))

    # Conflict output: rho into anchored solve
    _arrow(ax, (45.5, 26.2), (45.5, 22.3), VERM, style="--", lw=0.9)
    ax.text(44.8, 23.1, "$\\rho$", ha="right", va="center", fontsize=7.5,
            color=VERM, bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.9))

    # 7 anchor store + refresh
    _cylinder(ax, 42.5, 7.5, 11.2, 3.8, GREEN)
    ax.text(48.1, 10.1, "anchor $G$", ha="center", fontsize=7.8, fontweight="bold", color="#0a6653")
    ax.text(48.1, 8.5, "prior / previous fit", ha="center", fontsize=6.8, color=GRAY)
    _arrow(ax, (48.1, 11.5), (48.1, 13.3), GREEN, style="-.", lw=1.0)
    ax.text(49.2, 12.3, "$G$", ha="left", va="center", fontsize=7.5, color="#0a6653", fontweight="bold")

    # SPAR refresh arrow curving from solve to anchor
    _arrow(ax, (57.0, 13.5), (55.0, 11.0), GREEN, style="-.", rad=-0.38, lw=1.0)
    ax.text(59.5, 12.0, "SPAR refresh", ha="left", va="center", fontsize=7.2,
            color="#0a6653", fontweight="bold")
    ax.text(59.5, 10.3, "$\\beta_i = 1 - s_i$", ha="left", va="center",
            fontsize=7.2, color="#0a6653")
    _step(ax, 7, 69.5, 11.2)

    # repeat glyph (top-right corner)
    _arrow(ax, (84.0, 24.5), (89.5, 27.2), BLUE, rad=0.45, lw=1.0)
    ax.text(90.5, 27.2, "repeat per step", ha="left", va="center", fontsize=7.0, color=BLUE)

    # ========================== PANEL (b) & LEGEND ==========================
    # Legend strip (bottom-left)
    axl = fig.add_axes([0.02, 0.04, 0.44, 0.28])
    axl.set_xlim(0, 10)
    axl.set_ylim(0, 3.2)
    axl.axis("off")

    _box(axl, 0.2, 0.1, 9.6, 3.0, "#d0d0d0", lw=0.6, face="#fbfbfb")
    axl.text(0.5, 2.7, "Information & Control Paths", fontsize=7.5, fontweight="bold", color="#333333")

    axl.plot([0.5, 1.6], [1.95, 1.95], color=BLUE, lw=1.3)
    axl.text(1.9, 1.95, "Data path: stream $(x_t, y_t) \\to$ solve $\\to \\hat y_t$",
             va="center", fontsize=7.0, color="#222222")

    axl.plot([0.5, 1.6], [1.20, 1.20], color=VERM, lw=1.1, ls="--")
    axl.text(1.9, 1.20, "Control path: residuals $\\to$ estimators ($\\omega^*, \\rho^*$)",
             va="center", fontsize=7.0, color="#222222")

    axl.plot([0.5, 1.6], [0.45, 0.45], color=GREEN, lw=1.1, ls="-.")
    axl.text(1.9, 0.45, "Anchor path: prior $G$ & refresh $\\beta_i = 1 - s_i$",
             va="center", fontsize=7.0, color="#222222")

    # Panel (b): per-direction pull
    axb = fig.add_axes([0.52, 0.07, 0.46, 0.25])
    axb.set_title("(b) per-direction pull weights (Equation 1)", fontsize=8.0, pad=3, fontweight="bold")

    mu = np.array([1.0, 0.55, 0.30, 0.16, 0.09, 0.05])
    rho = 0.15
    s = rho / (mu + rho)
    xs = np.arange(6)
    w_bar = 0.35

    # Grouped bars: excitation mu_i and pull weight s_i
    axb.bar(xs - w_bar / 2, mu, width=w_bar, facecolor="#e0e0e0", edgecolor="#666666",
            lw=0.6, label="$\\mu_i$ (excitation)")
    axb.bar(xs + w_bar / 2, s, width=w_bar, facecolor=VERM, edgecolor=VERM,
            alpha=0.85, lw=0.6, label="$s_i = \\rho/(\\mu_i + \\rho)$ (pull)")

    axb.set_xlim(-0.8, 5.8)
    axb.set_ylim(0, 1.40)
    axb.set_xticks(xs)
    axb.set_xticklabels([f"dir {i+1}" for i in range(6)], fontsize=7.0)
    axb.set_yticks([0.0, 0.5, 1.0])
    axb.set_yticklabels(["0", "0.5", "1"], fontsize=7.0)
    axb.grid(axis="y", alpha=0.25, ls=":")

    axb.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=6.8, framealpha=0.9,
               borderpad=0.2, handletextpad=0.3, columnspacing=0.8)
    axb.text(0.0, 1.08, "data wins\n($s_1 \\approx 0.13$)", fontsize=6.8, color=BLUE,
             ha="center", va="bottom", fontweight="bold")
    axb.text(5.0, 0.95, "anchor holds\n($s_6 \\approx 0.75$)", fontsize=6.8, color=GREEN,
             ha="center", va="bottom", fontweight="bold")

    figstyle.save(fig, "F1_schematic")
    return "SUCCESS"


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    build()

