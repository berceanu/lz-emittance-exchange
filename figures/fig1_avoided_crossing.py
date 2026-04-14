#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["matplotlib", "numpy", "scipy>=1.11"]
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
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

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


def numerical_eigenfreqs(Kx, Ky, w, n_periods=200):
    """Integrate the coupled EOM and extract eigenfrequencies via FFT.

    EOM (velocity form):
        x'' + 2w y' + Kx x = 0
        y'' - 2w x' + Ky y = 0

    State vector: [x, x', y, y'].
    Returns the two dominant angular frequencies sorted as (Omega_+, Omega_-).
    """

    def rhs(t, state):
        x, xd, y, yd = state
        xdd = -Kx * x - 2.0 * w * yd
        ydd = -Ky * y + 2.0 * w * xd
        return [xd, xdd, yd, ydd]

    # Characteristic frequency scale for setting integration time
    omega_char = np.sqrt(0.5 * (Kx + Ky))
    T_beta = 2.0 * np.pi / omega_char
    t_end = n_periods * T_beta

    # Initial condition: offset in x only -> excites both normal modes
    y0 = [1.0, 0.0, 0.0, 0.0]

    # Dense output with enough points for clean FFT
    n_pts = 2**16
    t_eval = np.linspace(0, t_end, n_pts)

    sol = solve_ivp(rhs, [0, t_end], y0, method="DOP853",
                    t_eval=t_eval, rtol=1e-12, atol=1e-14)

    # Combine x and y spectra: a mode that is weak in x may be strong in y
    window = np.hanning(n_pts)
    dt = t_eval[1] - t_eval[0]
    n_padded = 4 * n_pts
    freqs = np.fft.rfftfreq(n_padded, d=dt) * 2.0 * np.pi  # angular freq
    df = freqs[1] - freqs[0]

    spec_x = np.abs(np.fft.rfft(sol.y[0] * window, n=n_padded))
    spec_y = np.abs(np.fft.rfft(sol.y[2] * window, n=n_padded))
    spectrum = spec_x + spec_y
    spectrum[0] = 0.0

    # Restrict to the physical band [0.5, 2.0] to avoid aliases
    band = (freqs >= 0.5) & (freqs <= 2.0)
    spec_band = np.zeros_like(spectrum)
    spec_band[band] = spectrum[band]

    # Find the two tallest peaks using scipy.signal.find_peaks
    min_dist = max(3, int(0.02 / df))
    peaks, _ = find_peaks(spec_band, height=0.01 * spec_band.max(),
                          distance=min_dist)
    if len(peaks) >= 2:
        heights = spec_band[peaks]
        top2 = peaks[np.argsort(heights)[-2:]]
    else:
        # Fallback: argmax + mask
        idx1 = np.argmax(spec_band)
        s2 = spec_band.copy()
        s2[max(0, idx1 - 3):idx1 + 4] = 0.0
        idx2 = np.argmax(s2)
        top2 = np.array([idx1, idx2])

    # Parabolic interpolation around each peak for sub-bin accuracy
    refined = []
    for idx in top2:
        if 0 < idx < len(spectrum) - 1:
            a, b, c = spectrum[idx - 1], spectrum[idx], spectrum[idx + 1]
            denom = 2.0 * (2 * b - a - c)
            if abs(denom) > 1e-15:
                shift = (a - c) / denom
                refined.append(freqs[idx] + shift * df)
            else:
                refined.append(freqs[idx])
        else:
            refined.append(freqs[idx])

    return max(refined), min(refined)  # Omega_+, Omega_-


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

    # Numerical eigenfrequencies via ODE integration + FFT
    # Concentrate sample points near delta=0 where the gap is
    delta_num = np.concatenate([
        np.linspace(-1.0, -0.3, 4),
        np.linspace(-0.25, 0.25, 10),
        np.linspace(0.3, 1.0, 4),
    ])
    for w, c in zip(w_values, colors):
        Om_p_num = np.empty_like(delta_num)
        Om_m_num = np.empty_like(delta_num)
        for i, d in enumerate(delta_num):
            Kx = 1.0 + 0.5 * d
            Ky = 1.0 - 0.5 * d
            Om_p_num[i], Om_m_num[i] = numerical_eigenfreqs(Kx, Ky, w)
        ax.scatter(delta_num, Om_p_num, s=12, facecolors="none",
                   edgecolors=c, linewidths=0.7, zorder=4)
        ax.scatter(delta_num, Om_m_num, s=12, facecolors="none",
                   edgecolors=c, linewidths=0.7, zorder=4)

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
