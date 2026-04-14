#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["pyvista", "numpy"]
# ///
"""3D schematic of the magnetised LWFA emittance exchange setup.

Uses PyVista for proper 3D rendering with smooth tubes and lighting.
"""

from pathlib import Path

import numpy as np
import pyvista as pv


def smoothstep(x):
    s = np.clip(x, 0, 1)
    return s * s * (3 - 2 * s)


def make_trajectory(n_pts=2000):
    """Electron trajectory: x-mode -> circular -> y-mode."""
    z = np.linspace(0, 10, n_pts)
    z_cross, width = 5.0, 2.5
    sig = smoothstep((z - z_cross + width / 2) / width)

    amp = 0.45
    freq = 6.5
    phase = 2 * np.pi * freq * z / 10

    x = amp * (1 - sig) * np.sin(phase)
    y = amp * sig * np.cos(phase)

    points = np.column_stack([z, x, y])
    colors = np.zeros((n_pts, 3), dtype=np.uint8)
    colors[:, 0] = (sig * 220).astype(np.uint8)         # R
    colors[:, 1] = 30                                     # G
    colors[:, 2] = ((1 - sig) * 220).astype(np.uint8)   # B

    spline = pv.Spline(points, n_points=n_pts)
    spline["colors"] = colors[:n_pts]
    return spline


def make_solenoid(n_turns=16, radius=0.9, length=10.0, n_pts_per_turn=60):
    """Helical solenoid coil."""
    n_pts = n_turns * n_pts_per_turn
    t = np.linspace(0, 2 * np.pi * n_turns, n_pts)
    z = t / (2 * np.pi * n_turns) * length
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    points = np.column_stack([z, x, y])
    return pv.Spline(points, n_points=n_pts)


def make_capillary(radius=0.15, length=10.0):
    """Semi-transparent capillary cylinder."""
    cyl = pv.Cylinder(center=(5, 0, 0), direction=(1, 0, 0),
                      radius=radius, height=length, resolution=40)
    return cyl


def make_arrow(start, direction, length=1.0):
    """Simple arrow."""
    return pv.Arrow(start=start, direction=direction, scale=length,
                    tip_length=0.3, tip_radius=0.08, shaft_radius=0.03)


def main():
    pv.global_theme.background = "white"
    pv.global_theme.font.color = "black"

    pl = pv.Plotter(off_screen=True, window_size=[1800, 700])

    # Electron trajectory as colored tube
    traj = make_trajectory()
    tube = traj.tube(radius=0.035)
    tube["colors"] = traj["colors"]
    pl.add_mesh(tube, scalars="colors", rgb=True, smooth_shading=True)

    # Solenoid helix as thin tube
    sol = make_solenoid()
    sol_tube = sol.tube(radius=0.02)
    pl.add_mesh(sol_tube, color="gray", opacity=0.4, smooth_shading=True)

    # Capillary
    cap = make_capillary()
    pl.add_mesh(cap, color="lightskyblue", opacity=0.12,
                smooth_shading=True)

    # Laser arrow
    laser = make_arrow(start=(-2.0, 0, 0), direction=(1, 0, 0), length=1.5)
    pl.add_mesh(laser, color="#E69F00", smooth_shading=True)
    pl.add_text("laser", position=(80, 320), font_size=11,
                color="#E69F00")

    # B-field arrow
    b_arrow = make_arrow(start=(10.5, 0, 0), direction=(1, 0, 0),
                         length=1.0)
    pl.add_mesh(b_arrow, color="darkred", smooth_shading=True)
    pl.add_text("B₀", position=(1620, 320), font_size=13,
                color="darkred")

    # Crossing marker (thin line at z=5)
    cross_line = pv.Line((-0.8, 0, 5), (0.8, 0, 5))
    pl.add_mesh(cross_line, color="gray", line_width=1.5,
                style="wireframe")
    pl.add_text("ξ*", position=(870, 200), font_size=10, color="gray")

    # Mode labels
    pl.add_text("x-mode", position=(250, 400), font_size=10,
                color="#0072B2")
    pl.add_text("y-mode", position=(1350, 400), font_size=10,
                color="#D55E00")

    # Camera
    pl.camera_position = [
        (5, -6, 4),    # camera location
        (5, 0, 0),     # focal point
        (0, 0, 1),     # view up
    ]

    out = Path(__file__).parent / "fig_setup3d.png"
    pl.screenshot(str(out), transparent_background=False)
    print(f"wrote {out}")
    pl.close()


if __name__ == "__main__":
    main()
