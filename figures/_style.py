"""Shared matplotlib style for all figures in this paper.

Nature-quality: minimal, colorblind-friendly, serif body, single-column
width ~3.4 inch unless overridden.
"""

import matplotlib.pyplot as plt

# Okabe-Ito colorblind-friendly palette
OKABE_ITO = {
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

# Single- and double-column widths (inches)
COL_SINGLE = 3.4
COL_DOUBLE = 7.0


def set_style():
    plt.rcParams.update(
        {
            # fonts — serif to match Typst's New Computer Modern
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "legend.borderaxespad": 0.4,
            "legend.handletextpad": 0.5,
            "legend.labelspacing": 0.3,
            "legend.borderpad": 0.2,
            # line widths
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            # ticks on all sides, pointing in
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            # figure layout
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.constrained_layout.use": True,
        }
    )
