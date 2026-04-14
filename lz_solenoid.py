"""Numerical verification of the Landau-Zener emittance exchange formula
for the solenoid + anisotropic-plasma LWFA model.

Verifies P_hop = exp(-pi w^2 T / eta) by integrating the velocity-form EOM,
decomposing the final state into symplectic normal modes, and comparing the
numerically measured hopping probability with the analytic prediction.
"""

import numpy as np
from scipy.integrate import solve_ivp

# Canonical symplectic form J4
J4 = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
], dtype=float)


def dyn_matrix_canonical(Kx, Ky, w):
    """Canonical dynamical matrix dz/dt = M z for z = (x, y, px, py)."""
    return np.array([
        [0, -w, 1, 0],
        [w, 0, 0, 1],
        [-(Kx + w**2), 0, 0, -w],
        [0, -(Ky + w**2), w, 0],
    ])


def mode_decomp(z, Kx, Ky, w):
    """Decompose canonical state z into normal-mode actions.

    Returns (Omega_plus, Omega_minus, J_plus, J_minus) where mode + has the
    higher frequency.
    """
    M = dyn_matrix_canonical(Kx, Ky, w)
    evals, evecs_T = np.linalg.eig(M)
    evecs = evecs_T.T  # evecs[i] is the i-th eigenvector

    pos_idx = np.where(evals.imag > 1e-10)[0]
    if len(pos_idx) != 2:
        return 0.0, 0.0, 0.0, 0.0

    omegas = evals[pos_idx].imag
    vs = evecs[pos_idx]

    # Sort descending by frequency: mode 0 = "+", mode 1 = "-"
    order = np.argsort(-omegas)
    omegas = omegas[order]
    vs = vs[order]

    J_actions = np.empty(2)
    for s in range(2):
        v = vs[s].copy()
        vdag_J_v = v.conj() @ (J4 @ v)
        scale = np.sqrt(np.abs(1j * vdag_J_v))
        v /= scale
        amp = 1j * v.conj() @ (J4 @ z)
        J_actions[s] = np.abs(amp) ** 2

    return omegas[0], omegas[1], J_actions[0], J_actions[1]


def run_solenoid(w, eta, T):
    """Integrate velocity-form EOM and return canonical states at t=0 and t=T.

    Initial condition: x=1, y=0, vx=0, vy=0 (on bare x-mode).
    """

    def eom(t, u):
        x, y, vx, vy = u
        Kx = 1.0 + eta * (1 - 2 * t / T)
        Ky = 1.0 - eta * (1 - 2 * t / T)
        return [
            vx,
            vy,
            -Kx * x - 2 * w * vy,
            -Ky * y + 2 * w * vx,
        ]

    u0 = [1.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(eom, [0, T], u0, method="DOP853", rtol=1e-12, atol=1e-12)

    xI, yI, vxI, vyI = u0
    xF, yF, vxF, vyF = sol.y[:, -1]

    # Convert velocity to canonical momentum: px = vx + w*y, py = vy - w*x
    zI = np.array([xI, yI, vxI + w * yI, vyI - w * xI])
    zF = np.array([xF, yF, vxF + w * yF, vyF - w * xF])
    return zI, zF


def report(label, w, eta, T):
    """Run one parameter set and return formatted result line."""
    zI, zF = run_solenoid(w, eta, T)

    KxI, KyI = 1.0 + eta, 1.0 - eta
    KxF, KyF = 1.0 - eta, 1.0 + eta

    dI = mode_decomp(zI, KxI, KyI, w)
    dF = mode_decomp(zF, KxF, KyF, w)

    P_hop = dF[3] / (dF[2] + dF[3])
    P_pred = np.exp(-np.pi * w**2 * T / eta)
    J_ratio = (dF[2] + dF[3]) / (dI[2] + dI[3])

    print(
        f"  {label}  w={w:.4f}  eta={eta:.2f}  T={T:.1f}"
        f"   Ppred={P_pred:.3e}   Pnum={P_hop:.3e}   Jf/Ji={J_ratio:.6f}"
    )


def main():
    print("==== Landau-Zener verification: solenoid + anisotropic LWFA ====")
    print("Proper symplectic mode actions used.")
    print("Analytic prediction: P_hop = exp(-pi w^2 T / eta)")
    print()

    print("--- Scan 1: vary T, fixed w=0.02, eta=0.5 ---")
    for T in [200.0, 500.0, 1000.0, 2000.0, 5000.0]:
        report("S1", 0.02, 0.5, T)
    print()

    print("--- Scan 2: vary w, fixed T=1000, eta=0.5 ---")
    for w in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
        report("S2", w, 0.5, 1000.0)
    print()

    print("--- Scan 3: vary eta, fixed T=1000, w=0.02 ---")
    for eta in [0.2, 0.3, 0.4, 0.5, 0.7]:
        report("S3", 0.02, eta, 1000.0)


if __name__ == "__main__":
    main()
