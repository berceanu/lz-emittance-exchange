#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
"""Emittance exchange fraction vs crossing length for LWFA parameters.

Plot 1 - P_hop as a function of the anisotropy crossing length (cm)
for two parameter sets. Shaded band marks typical single-stage LWFA
length (1--10 cm).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _style import COL_SINGLE, GOLDEN, OKABE_ITO, set_style  # noqa: E402

set_style()

# Physical constants (SI)
E = 1.602e-19  # electron charge
M = 9.109e-31  # electron mass
C = 2.998e8  # speed of light
EPS0 = 8.854e-12


def omega_p(n_p_cgs):
    """Plasma frequency (rad/s) from density in cm^-3."""
    n_si = n_p_cgs * 1e6
    return np.sqrt(n_si * E * E / (M * EPS0))


def omega_beta(n_p_cgs, gamma):
    """Betatron frequency (rad/s)."""
    return omega_p(n_p_cgs) / np.sqrt(2 * gamma)


def omega_L(gamma, B0):
    """Larmor frequency (rad/s)."""
    return E * B0 / (2 * gamma * M)


def phop_of_length(ell_m, n_p_cgs, gamma, B0):
    """Landau-Zener hop probability for crossing length ell (metres)."""
    wB = omega_beta(n_p_cgs, gamma)
    wL = omega_L(gamma, B0)
    arg = 4 * np.pi * wL**2 * ell_m / (C * wB)
    return np.exp(-arg)


def main():
    fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE / GOLDEN))

    ell_cm = np.linspace(0.1, 40.0, 800)
    ell_m = ell_cm * 1e-2

    cases = [
        {
            "label": r"$B_0 = 20$ T, $\gamma = 10^3$",
            "n_p": 1e17, "gamma": 1e3, "B0": 20.0,
            "color": OKABE_ITO["blue"],
        },
        {
            "label": r"$B_0 = 40$ T, $\gamma = 500$",
            "n_p": 1e17, "gamma": 500.0, "B0": 40.0,
            "color": OKABE_ITO["vermillion"],
        },
    ]

    for case in cases:
        phop = phop_of_length(ell_m, case["n_p"], case["gamma"], case["B0"])
        exch = 1 - phop
        ax.plot(
            ell_cm, exch * 100,
            color=case["color"], lw=1.3, label=case["label"],
        )

    ax.axvspan(1.0, 10.0, color="0.90", alpha=0.7, zorder=0)

    for ref in (10, 50):
        ax.axhline(ref, color="0.65", lw=0.4, ls=(0, (1, 2)), zorder=1)

    ax.set_xlabel(r"$\ell_{\rm cross}$ (cm)")
    ax.set_ylabel(r"$1 - P_{\rm hop}$ (%)")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper left", handlelength=1.2)

    out = Path(__file__).parent / "fig3_accessibility.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
