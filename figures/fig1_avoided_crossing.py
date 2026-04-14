#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
"""Avoided crossing of the normal-mode frequencies.

Plot Omega_+/- vs the plasma anisotropy delta = (k_x - k_y)/k_beta for
several values of the dimensionless solenoid coupling w = omega_L/omega_beta.
At w = 0 the modes cross; for w > 0 the degeneracy lifts with a minimum
gap of 2 omega_L at delta = 0.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _style import COL_SINGLE, GOLDEN, OKABE_ITO, set_style  # noqa: E402

set_style()


def normal_modes(delta, w):
    """Return Omega_+/- normalised to omega_beta.

    Parameters: k_x = 1 + delta/2, k_y = 1 - delta/2,
    coupling w = omega_L / omega_beta.
    """
    Kx = 1.0 + 0.5 * delta
    Ky = 1.0 - 0.5 * delta
    Omsum = Kx + Ky + 4.0 * w**2
    disc = (Kx - Ky) ** 2 + 8.0 * w**2 * (Kx + Ky + 2.0 * w**2)
    Omega_p_sq = 0.5 * (Omsum + np.sqrt(disc))
    Omega_m_sq = 0.5 * (Omsum - np.sqrt(disc))
    return np.sqrt(Omega_p_sq), np.sqrt(Omega_m_sq)


def main():
    fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE / GOLDEN))

    delta = np.linspace(-1.0, 1.0, 401)

    # Bare (uncoupled) crossing
    bare_x = np.sqrt(1.0 + 0.5 * delta)
    bare_y = np.sqrt(1.0 - 0.5 * delta)
    ax.plot(delta, bare_x, color="0.55", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.plot(delta, bare_y, color="0.55", lw=0.7, ls=(0, (3, 2)), zorder=1)

    # Dressed curves for two coupling strengths
    w_values = [0.05, 0.15]
    colors = [OKABE_ITO["blue"], OKABE_ITO["vermillion"]]
    for w, c in zip(w_values, colors):
        Op, Om = normal_modes(delta, w)
        ax.plot(delta, Op, color=c, lw=1.2, zorder=3, label=rf"$w = {w}$")
        ax.plot(delta, Om, color=c, lw=1.2, zorder=3)

    ax.set_xlabel(r"$(k_x - k_y)/k_\beta$")
    ax.set_ylabel(r"$\Omega_\pm / \omega_\beta$")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.62, 1.50)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.2,
        ncol=2,
        columnspacing=0.8,
    )

    out = Path(__file__).parent / "fig1_avoided_crossing.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
