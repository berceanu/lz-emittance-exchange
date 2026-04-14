"""Avoided crossing test with a skew quadrupole (Bazzani-style).

H = (px^2 + py^2)/2 + (1/2)[a(t) x^2 + b(t) y^2 + 2 q x y]

where a = 1 + 3t/T, b = 4 - 3t/T, q = constant.

Tracks mode A (smaller eigenvalue) continuously using the analytic eigenvector
angle theta = (1/2) arctan2(2q, b - a). Rotates into the mode frame and
computes instantaneous actions JA and JB.
"""

import numpy as np
from scipy.integrate import solve_ivp


def mode_a_angle(a, b, q):
    """Angle theta such that (cos theta, sin theta) is the eigenvector
    of the smaller eigenvalue of [[a, q], [q, b]].

    Uses arctan2 for continuous tracking through the avoided crossing.
    """
    return 0.5 * np.arctan2(2 * q, b - a)


def run_coupled(T, q0):
    """Integrate the EOM and return time-series of (t, JA, JB, lambdaA, lambdaB, theta).

    Initial condition: x=1, y=0, px=0, py=0 (on mode A at t=0).
    """

    def eom(t, u):
        x, y, px, py = u
        a = 1.0 + 3.0 * t / T
        b = 4.0 - 3.0 * t / T
        return [
            px,
            py,
            -a * x - q0 * y,
            -q0 * x - b * y,
        ]

    u0 = [1.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(eom, [0, T], u0, method="DOP853", rtol=1e-12, atol=1e-12,
                    dense_output=True)

    t_sample = np.linspace(0, T, 401)
    data = []

    for tt in t_sample:
        a = 1.0 + 3.0 * tt / T
        b = 4.0 - 3.0 * tt / T
        theta = mode_a_angle(a, b, q0)
        lambda_A = (a + b) / 2 - np.sqrt((a - b) ** 2 / 4 + q0**2)
        lambda_B = (a + b) / 2 + np.sqrt((a - b) ** 2 / 4 + q0**2)

        x, y, px, py = sol.sol(tt)

        XA = np.cos(theta) * x + np.sin(theta) * y
        XB = -np.sin(theta) * x + np.cos(theta) * y
        PA = np.cos(theta) * px + np.sin(theta) * py
        PB = -np.sin(theta) * px + np.cos(theta) * py

        JA = (PA**2 + lambda_A * XA**2) / (2 * np.sqrt(lambda_A))
        JB = (PB**2 + lambda_B * XB**2) / (2 * np.sqrt(lambda_B))

        data.append((tt, JA, JB, lambda_A, lambda_B, theta))

    return data


def main():
    print("==== Eigenvalue crossing with fixed skew coupling ====")
    print("wx^2(0)=1, wx^2(T)=4; wy^2(0)=4, wy^2(T)=1; q = fixed")
    print()

    for q0 in [0.01, 0.1, 0.3]:
        label = {0.01: "very weak", 0.1: "moderate", 0.3: "strong"}[q0]
        print(f"q = {q0} ({label} coupling)")
        for T in [100.0, 500.0, 2000.0, 10000.0]:
            data = run_coupled(T, q0)
            _, JA_f, JB_f, *_ = data[-1]
            print(f"  T = {T:8.1f}  JA(T) = {JA_f:.4g}  JB(T) = {JB_f:.4g}")
        print()

    print("==== Interpretation ====")
    print("With continuous eigenvector tracking:")
    print("- Adiabatic (large T, q not too small): particle stays in mode A")
    print("  (smaller-eigenvalue mode). Expected: JA ~ 0.5, JB ~ 0.")
    print("- Diabatic (small T or q -> 0): jump at the crossing, JA <-> JB exchange.")


if __name__ == "__main__":
    main()
