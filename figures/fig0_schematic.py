#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
"""Schematic of the slow plasma-anisotropy crossing.

Three-panel figure showing (a) the focusing strengths k_x, k_y vs
propagation coordinate xi, and the resulting transverse trajectories
(b) x(xi) and (c) y(xi) of a witness electron through the stage.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).parent))
from _style import COL_DOUBLE, OKABE_ITO, set_style  # noqa: E402

set_style()

XI_MIN, XI_MAX = -1.0, 1.0
XI_STAR = 0.0
K_BETA = 1.0
ETA = 0.35
N_CYCLES = 8
SIG_K = 7.0
VIOLET_SIGMA = 0.09

BLUE = OKABE_ITO["blue"]
RED = OKABE_ITO["vermillion"]
VIOLET = OKABE_ITO["purple"]


def sigmoid(xi, k=SIG_K, x0=XI_STAR):
    return 1.0 / (1.0 + np.exp(-k * (xi - x0)))


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0


BLUE_RGB = hex_to_rgb(BLUE)
RED_RGB = hex_to_rgb(RED)
VIOLET_RGB = hex_to_rgb(VIOLET)


def colour_along(xi, base_rgb):
    """Blend base colour toward violet near the crossing point."""
    env = np.exp(-((xi - XI_STAR) ** 2) / (2 * VIOLET_SIGMA ** 2))
    colours = np.tile(base_rgb, (len(xi), 1))
    vfrac = 0.85 * env[:, None]
    return colours * (1 - vfrac) + VIOLET_RGB * vfrac


def line_collection(xi, y, colours, lw=1.3):
    pts = np.column_stack([xi, y])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    return LineCollection(segs, colors=colours[:-1], linewidths=lw)


def main():
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(COL_DOUBLE, 3.1),
        gridspec_kw={"height_ratios": [0.9, 1.3, 1.3], "hspace": 0.10},
        sharex=True,
    )

    xi = np.linspace(XI_MIN, XI_MAX, 2000)
    sw = sigmoid(xi)
    phase = 2 * np.pi * N_CYCLES * (xi - XI_MIN) / (XI_MAX - XI_MIN)

    # (a) k_x(xi), k_y(xi)
    ax0 = axes[0]
    kx = K_BETA + ETA * (1 - 2 * (xi - XI_MIN) / (XI_MAX - XI_MIN))
    ky = K_BETA - ETA * (1 - 2 * (xi - XI_MIN) / (XI_MAX - XI_MIN))
    ax0.plot(xi, kx, color=BLUE, lw=1.4, label=r"$k_x$")
    ax0.plot(xi, ky, color=RED, lw=1.4, label=r"$k_y$")
    ax0.axvline(XI_STAR, color="0.55", lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax0.set_ylabel(r"$k_{x,y}\,/\,k_\beta$")
    ax0.set_yticks([K_BETA - ETA, K_BETA, K_BETA + ETA])
    ax0.set_yticklabels(["0.65", "1.00", "1.35"])
    ax0.set_ylim(K_BETA - 1.4 * ETA, K_BETA + 1.4 * ETA)
    ax0.legend(
        loc="center right",
        handlelength=1.2,
        ncol=2,
        columnspacing=0.7,
        bbox_to_anchor=(0.98, 0.5),
    )

    # (b) x(xi)
    ax1 = axes[1]
    amp_x = 1.0 - sw
    x_traj = amp_x * np.cos(phase)
    col_x = colour_along(xi, BLUE_RGB)
    ax1.add_collection(line_collection(xi, x_traj, col_x))
    ax1.axvline(XI_STAR, color="0.55", lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax1.set_ylabel(r"$x(\xi)$")
    ax1.set_ylim(-1.25, 1.25)
    ax1.set_yticks([-1, 0, 1])

    # (c) y(xi)
    ax2 = axes[2]
    amp_y = sw
    y_traj = amp_y * np.cos(phase + np.pi / 2)
    col_y = colour_along(xi, RED_RGB)
    ax2.add_collection(line_collection(xi, y_traj, col_y))
    ax2.axvline(XI_STAR, color="0.55", lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax2.set_ylabel(r"$y(\xi)$")
    ax2.set_ylim(-1.25, 1.25)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_xlabel(r"$\xi$")

    label_bbox = dict(
        boxstyle="square,pad=0.18",
        facecolor="white",
        edgecolor="none",
        alpha=0.95,
    )
    for ax, label in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.text(
            0.012,
            0.93,
            label,
            transform=ax.transAxes,
            fontsize=8.5,
            ha="left",
            va="top",
            bbox=label_bbox,
        )

    for ax in axes:
        ax.set_xlim(XI_MIN, XI_MAX)

    out = Path(__file__).parent / "fig0_schematic.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
