#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["matplotlib", "numpy", "scipy"]
# ///
"""Numerical verification of the Landau-Zener formula.

Plot of the ratio P_hop(numerical) / P_hop(LZ prediction) vs the
adiabaticity parameter pi*w^2*T/eta, showing where the LZ formula is
accurate and where it starts to deviate.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).parent))
from _style import COL_SINGLE, GOLDEN, OKABE_ITO, set_style  # noqa: E402

set_style()

# Canonical 4x4 symplectic form
J4 = np.array(
    [[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]], dtype=float
)


def dyn_matrix(Kx, Ky, w):
    """Dynamical matrix for canonical state (x, y, p_x, p_y)."""
    return np.array(
        [
            [0.0, -w, 1.0, 0.0],
            [w, 0.0, 0.0, 1.0],
            [-(Kx + w * w), 0.0, 0.0, -w],
            [0.0, -(Ky + w * w), w, 0.0],
        ]
    )


def mode_decompose(z, Kx, Ky, w):
    """Return (J+, J-) actions for phase-space state z in the frozen Hamiltonian."""
    M = dyn_matrix(Kx, Ky, w)
    evals, evecs = np.linalg.eig(M)
    pos = [i for i in range(4) if evals[i].imag > 1e-10]
    if len(pos) != 2:
        return (np.nan, np.nan)
    Omegas = [evals[i].imag for i in pos]
    order = sorted(range(2), key=lambda i: -Omegas[i])
    result = []
    for idx in order:
        v = evecs[:, pos[idx]]
        vdagJv = np.conj(v) @ (J4 @ v)
        scale = np.sqrt(np.abs(1j * vdagJv))
        v = v / scale
        amp = 1j * np.conj(v) @ (J4 @ z)
        result.append(float(np.abs(amp) ** 2))
    return tuple(result)


def mode_plus_eigvec(Kx, Ky, w):
    """Return the mode-+ eigenvector, normalised so that i v^dag J v = 1."""
    M = dyn_matrix(Kx, Ky, w)
    evals, evecs = np.linalg.eig(M)
    pos = [i for i in range(4) if evals[i].imag > 1e-10]
    Omegas = [evals[i].imag for i in pos]
    idx_plus = pos[int(np.argmax(Omegas))]
    v = evecs[:, idx_plus]
    vdagJv = np.conj(v) @ (J4 @ v)
    v = v / np.sqrt(np.abs(1j * vdagJv))
    return v


def eom_canonical(t, z, w, eta, T):
    """Equations of motion in canonical variables (x, y, p_x, p_y)."""
    Kx = 1.0 + eta * (1 - 2 * t / T)
    Ky = 1.0 - eta * (1 - 2 * t / T)
    x, y, px, py = z
    return [
        px - w * y,
        py + w * x,
        -(Kx + w * w) * x - w * py,
        -(Ky + w * w) * y + w * px,
    ]


def integrate_canonical(z0, w, eta, T):
    """Integrate canonical EOM with high accuracy, return final state."""
    sol = solve_ivp(
        eom_canonical,
        (0.0, T),
        z0,
        args=(w, eta, T),
        method="DOP853",
        rtol=1e-11,
        atol=1e-11,
        dense_output=False,
    )
    return sol.y[:, -1]


def run_single(w, eta, T):
    """Single particle starting in dressed mode+ at t=0, phase=0."""
    KxI = 1.0 + eta
    KyI = 1.0 - eta
    KxF = 1.0 - eta
    KyF = 1.0 + eta
    vplus = mode_plus_eigvec(KxI, KyI, w)
    z0 = np.sqrt(2.0) * vplus.real
    zF = integrate_canonical(z0, w, eta, T)
    Jp, Jm = mode_decompose(zF, KxF, KyF, w)
    return Jm / (Jp + Jm)


def run_ensemble(w, eta, T, N=50, seed=17):
    """Ensemble of N particles with random phases on the dressed mode+."""
    KxI = 1.0 + eta
    KyI = 1.0 - eta
    KxF = 1.0 - eta
    KyF = 1.0 + eta
    vplus = mode_plus_eigvec(KxI, KyI, w)
    rng = np.random.default_rng(seed)
    phops = []
    for _ in range(N):
        phi = rng.uniform(0, 2 * np.pi)
        z0 = np.sqrt(2.0) * (np.cos(phi) * vplus.real - np.sin(phi) * vplus.imag)
        zF = integrate_canonical(z0, w, eta, T)
        Jp, Jm = mode_decompose(zF, KxF, KyF, w)
        phops.append(Jm / (Jp + Jm))
    phops = np.array(phops)
    return phops.mean(), phops.std()


def main():
    fig, ax = plt.subplots(figsize=(COL_SINGLE, COL_SINGLE / GOLDEN))

    # (w, eta, T) -- chosen to span ~2 decades in adiabaticity pi*w^2*T/eta
    cases = [
        # --- non-adiabatic (fast) regime: small adiabaticity parameter ---
        (0.005, 0.5, 200.0),     # adiab ~ 0.03
        (0.02, 0.5, 20.0),       # adiab ~ 0.05
        (0.02, 0.5, 50.0),       # adiab ~ 0.13
        (0.02, 0.5, 100.0),      # adiab ~ 0.25
        # --- intermediate regime ---
        (0.02, 0.5, 200.0),      # adiab ~ 0.50
        (0.01, 0.5, 1000.0),     # adiab ~ 0.63
        (0.02, 0.5, 500.0),      # adiab ~ 1.26
        (0.015, 0.5, 1000.0),    # adiab ~ 1.41
        (0.02, 0.7, 1000.0),     # adiab ~ 1.80
        (0.02, 0.5, 1000.0),     # adiab ~ 2.51
        (0.02, 0.4, 1000.0),     # adiab ~ 3.14
        (0.02, 0.3, 1000.0),     # adiab ~ 4.19
        # --- adiabatic (slow) regime: large adiabaticity parameter ---
        (0.02, 0.5, 2000.0),     # adiab ~ 5.03
        (0.03, 0.5, 1000.0),     # adiab ~ 5.65
        (0.025, 0.5, 2000.0),    # adiab ~ 7.85
        (0.03, 0.5, 2000.0),     # adiab ~ 11.3
        (0.02, 0.5, 5000.0),     # adiab ~ 12.6
    ]

    adiab_single = []
    ratio_single = []
    adiab_ens = []
    ratio_ens_mean = []
    ratio_ens_err = []

    print("Running LZ verification data...")
    for w, eta, T in cases:
        adiab = np.pi * w**2 * T / eta
        pred = np.exp(-adiab)
        phop_s = run_single(w, eta, T)
        phop_m, phop_sig = run_ensemble(w, eta, T, N=50)

        adiab_single.append(adiab)
        ratio_single.append(phop_s / pred)
        adiab_ens.append(adiab)
        ratio_ens_mean.append(phop_m / pred)
        # Propagate ensemble std to ratio: sigma_ratio = sigma_phop / pred
        ratio_ens_err.append(phop_sig / pred)

        print(
            f"  w={w} eta={eta} T={T}: adiab={adiab:.2f}  pred={pred:.3e}  "
            f"single={phop_s:.3e} (ratio={phop_s / pred:.4f})  "
            f"ens={phop_m:.3e} +/- {phop_sig:.1e} (ratio={phop_m / pred:.4f})"
        )

    adiab_single = np.array(adiab_single)
    ratio_single = np.array(ratio_single)
    adiab_ens = np.array(adiab_ens)
    ratio_ens_mean = np.array(ratio_ens_mean)
    ratio_ens_err = np.array(ratio_ens_err)

    # Perfect agreement line + +/-2% shaded band
    ax.axhline(1.0, color="0.5", lw=0.7, ls="--", zorder=1)
    ax.axhspan(0.98, 1.02, color="0.90", alpha=0.4, zorder=0, label=r"$\pm 2\%$")

    ax.scatter(
        adiab_single,
        ratio_single,
        s=20,
        marker="o",
        facecolor="white",
        edgecolor=OKABE_ITO["blue"],
        lw=0.9,
        zorder=3,
        label="single trajectory",
    )
    ax.errorbar(
        adiab_ens,
        ratio_ens_mean,
        yerr=ratio_ens_err,
        fmt="^",
        color=OKABE_ITO["vermillion"],
        ms=3.5,
        lw=0,
        elinewidth=0.7,
        capsize=1.0,
        zorder=4,
        label=r"ensemble ($N{=}50$)",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\pi \omega^2 T / \eta$")
    ax.set_ylabel(r"$P_{\rm hop}^{\rm (num)} \;/\; P_{\rm hop}^{\rm (LZ)}$")
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.0,
              columnspacing=0.8, ncol=1)

    out = Path(__file__).parent / "fig2_lz_verification.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
