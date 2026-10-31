"""Shared figure style for all paper figures.

Single source of truth for rcParams, the Okabe-Ito semantic palette,
physical size presets, and the dual PDF+PNG exporter.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
})

# Okabe-Ito, colorblind-safe; single semantic assignment across ALL figures.
OI = {
    "ours":   "#0072B2",   # blue    - SPAR / our arms
    "danger": "#D55E00",   # vermillion - divergence / cliff / W star
    "anchor": "#009E73",   # green   - anchors, priors, WiSE-FT comparisons
    "warn":   "#E69F00",   # orange  - secondary family (large rho)
    "sky":    "#56B4E9",   # sky     - secondary ours-variant (SPAR scalar)
    "ref":    "#000000",   # black   - theory references / floor lines
    "base":   "#7F7F7F",   # gray    - baselines (Kalman, leaky, etc.)
}

SINGLE_COL = (3.25, 2.4)    # inches, one column
DOUBLE_COL = (6.75, 2.6)    # inches, figure* full width

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


def save(fig, name):
    """Write results/figures/{name}.pdf (vector) and .png (300 dpi preview)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved {name}.pdf/.png -> {FIG_DIR}")
