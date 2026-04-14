#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
"""3D schematic of the magnetised LWFA emittance exchange setup.

Shows electron trajectory transitioning from horizontal to vertical
oscillation inside a solenoid-wrapped capillary.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, str(Path(__file__).parent))
from _style import COL_DOUBLE, OKABE_ITO, set_style  # noqa: E402

set_style()


def smoothstep(x):
    """Hermite smoothstep: 0 for x<0, 1 for x>1, smooth in between."""
    s = np.clip(x, 0, 1)
    return s * s * (3 - 2 * s)


def main():
    fig = plt.figure(figsize=(COL_DOUBLE, COL_DOUBLE * 0.45))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # Propagation along z
    z_len = 10.0
    z_cross = 5.0
    trans_width = 2.5

    # --- Electron trajectory ---
    n_pts = 2000
    z = np.linspace(0, z_len, n_pts)
    sig = smoothstep((z - z_cross + trans_width / 2) / trans_width)

    amp = 0.55
    freq = 6.5  # betatron oscillations over the length
    phase = 2 * np.pi * freq * z / z_len

    x_traj = amp * (1 - sig) * np.sin(phase)
    y_traj = amp * sig * np.cos(phase)

    # Color gradient: blue (x-mode) -> violet -> red (y-mode)
    colors = np.zeros((n_pts, 4))
    colors[:, 0] = sig          # R
    colors[:, 1] = 0.15         # G
    colors[:, 2] = 1 - sig      # B
    colors[:, 3] = 1.0          # A

    # Plot as colored line segments
    points = np.column_stack([z, x_traj, y_traj]).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_colors = 0.5 * (colors[:-1] + colors[1:])
    lc = Line3DCollection(segments, colors=seg_colors, linewidths=1.4,
                          zorder=5)
    ax.add_collection3d(lc)

    # --- Solenoid helix ---
    sol_rad = 1.05
    n_turns = 14
    t_sol = np.linspace(0, 2 * np.pi * n_turns, 800)
    z_sol = t_sol / (2 * np.pi * n_turns) * z_len
    x_sol = sol_rad * np.cos(t_sol)
    y_sol = sol_rad * np.sin(t_sol)

    # Split into front and back halves for depth ordering
    front = y_sol >= 0
    back = ~front

    ax.plot(z_sol[back], x_sol[back], y_sol[back],
            color="0.75", lw=0.6, zorder=1)
    ax.plot(z_sol[front], x_sol[front], y_sol[front],
            color="0.55", lw=0.8, zorder=8)

    # --- Capillary (two horizontal lines + end caps) ---
    cap_r = 0.18
    z_cap = np.array([0.1, z_len - 0.1])
    for sign in [1, -1]:
        ax.plot(z_cap, [0, 0], [sign * cap_r, sign * cap_r],
                color=OKABE_ITO["skyblue"], lw=1.5, alpha=0.5, zorder=2)
    # Top edge
    for sign in [1, -1]:
        ax.plot(z_cap, [sign * cap_r, sign * cap_r], [0, 0],
                color=OKABE_ITO["skyblue"], lw=1.5, alpha=0.5, zorder=2)

    # --- Laser arrow ---
    ax.quiver(-1.5, 0, 0, 1.3, 0, 0,
              arrow_length_ratio=0.2, color=OKABE_ITO["orange"],
              lw=2.5, zorder=10)
    ax.text(-1.8, 0, 0.3, "laser", fontsize=7.5,
            color=OKABE_ITO["orange"], ha="center")

    # --- B-field arrow ---
    ax.quiver(z_len + 0.3, 0, 0, 0.8, 0, 0,
              arrow_length_ratio=0.25, color="darkred",
              lw=1.8, zorder=10)
    ax.text(z_len + 1.5, 0, 0.25, r"$B_0$", fontsize=9,
            color="darkred", ha="center")

    # --- Crossing marker ---
    ax.plot([z_cross, z_cross], [-0.9, 0.9], [0, 0],
            ls="--", color="0.5", lw=0.6, zorder=3)
    ax.text(z_cross, -1.1, 0, r"$\xi^*$", fontsize=8,
            color="0.4", ha="center")

    # --- Mode labels ---
    ax.text(1.5, 0, -0.8, r"$x$-mode", fontsize=7.5,
            color=OKABE_ITO["blue"], ha="center")
    ax.text(8.5, 0, 0.8, r"$y$-mode", fontsize=7.5,
            color=OKABE_ITO["vermillion"], ha="center")

    # --- Axis labels ---
    ax.text(-0.5, 0.9, 0, "$x$", fontsize=7, color="0.4", ha="center")
    ax.text(-0.5, 0, 0.9, "$y$", fontsize=7, color="0.4", ha="center")

    # --- Viewing angle ---
    ax.view_init(elev=18, azim=-60)

    # --- Clean up ---
    ax.set_xlim(-1.5, z_len + 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    out = Path(__file__).parent / "fig_setup3d.pdf"
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
