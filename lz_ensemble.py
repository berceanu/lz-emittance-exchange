"""Ensemble Landau-Zener verification.

Propagates distributions of initial conditions through the solenoid
emittance-exchange crossing and measures beam-averaged P_hop.

Two ensembles:
  (A) Dressed: particles on the mode-+ eigenstate with random phase.
  (B) Bare: Gaussian in (x, px) matched to the bare plasma mode, y=py=0.
"""

import numpy as np
from scipy.integrate import solve_ivp

J4 = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
], dtype=float)


def dyn_matrix_canonical(Kx, Ky, w):
    return np.array([
        [0, -w, 1, 0],
        [w, 0, 0, 1],
        [-(Kx + w**2), 0, 0, -w],
        [0, -(Ky + w**2), w, 0],
    ])


def mode_plus_eigen(Kx, Ky, w):
    """Return (Omega_plus, v_plus) for the higher-frequency mode.

    v_plus is normalized so that i * v^dag J v = 1.
    """
    M = dyn_matrix_canonical(Kx, Ky, w)
    evals, evecs_T = np.linalg.eig(M)
    evecs = evecs_T.T

    pos_idx = np.where(evals.imag > 1e-10)[0]
    # Pick the eigenvector with the largest imaginary part
    best = pos_idx[np.argmax(evals[pos_idx].imag)]
    Om_plus = evals[best].imag
    v_plus = evecs[best].copy()

    vdag_J_v = v_plus.conj() @ (J4 @ v_plus)
    v_plus /= np.sqrt(np.abs(1j * vdag_J_v))
    return Om_plus, v_plus


def mode_decomp(z, Kx, Ky, w):
    """Return [J_plus, J_minus] for canonical state z."""
    M = dyn_matrix_canonical(Kx, Ky, w)
    evals, evecs_T = np.linalg.eig(M)
    evecs = evecs_T.T

    pos_idx = np.where(evals.imag > 1e-10)[0]
    if len(pos_idx) != 2:
        return np.array([0.0, 0.0])

    omegas = evals[pos_idx].imag
    vs = evecs[pos_idx]

    order = np.argsort(-omegas)
    vs = vs[order]

    J_actions = np.empty(2)
    for s in range(2):
        v = vs[s].copy()
        vdag_J_v = v.conj() @ (J4 @ v)
        v /= np.sqrt(np.abs(1j * vdag_J_v))
        amp = 1j * v.conj() @ (J4 @ z)
        J_actions[s] = np.abs(amp) ** 2

    return J_actions


def propagate_canon(z0, w, eta, T):
    """Propagate canonical initial condition z0 = (x, y, px, py) through the crossing.

    Converts to velocity coords for integration, then back to canonical at t=T.
    """
    # vx = dx/dt = px - w*y, vy = dy/dt = py + w*x
    vxI = z0[2] - w * z0[1]
    vyI = z0[3] + w * z0[0]

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

    u0 = [z0[0], z0[1], vxI, vyI]
    sol = solve_ivp(eom, [0, T], u0, method="DOP853", rtol=1e-12, atol=1e-12)

    xF, yF, vxF, vyF = sol.y[:, -1]
    return np.array([xF, yF, vxF + w * yF, vyF - w * xF])


def run_ensemble_dressed(w, eta, T, N, J0, seed=17):
    """Dressed ensemble: all particles launched with J_+ = J0 exactly.

    z_canonical = sqrt(2*J0) * Re(v_plus * exp(i*phi)), phi uniform in [0, 2pi).
    """
    KxI, KyI = 1.0 + eta, 1.0 - eta
    KxF, KyF = 1.0 - eta, 1.0 + eta
    _, v_plus = mode_plus_eigen(KxI, KyI, w)

    rng = np.random.default_rng(seed)
    phop_list = np.empty(N)

    for i in range(N):
        phi = rng.uniform(0, 2 * np.pi)
        # Re[v * exp(i*phi)] with amplitude sqrt(2*J0) gives action J0
        z0 = np.sqrt(2 * J0) * (
            np.cos(phi) * v_plus.real - np.sin(phi) * v_plus.imag
        )
        zF = propagate_canon(z0, w, eta, T)
        JF = mode_decomp(zF, KxF, KyF, w)
        phop_list[i] = JF[1] / (JF[0] + JF[1])

    return phop_list.mean(), phop_list.std()


def run_ensemble_bare(w, eta, T, N, J0, seed=23):
    """Bare ensemble: Gaussian in (x, px) matched to bare plasma mode, y=py=0.

    sigma_x = sqrt(J0 / sqrt(Kx)), sigma_px = sqrt(J0 * sqrt(Kx)).
    """
    KxI = 1.0 + eta
    KxF, KyF = 1.0 - eta, 1.0 + eta

    sigma_x = np.sqrt(J0 / np.sqrt(KxI))
    sigma_px = np.sqrt(J0 * np.sqrt(KxI))

    rng = np.random.default_rng(seed)
    phop_list = np.empty(N)

    for i in range(N):
        z0 = np.array([
            rng.normal(0, sigma_x),
            0.0,
            rng.normal(0, sigma_px),
            0.0,
        ])
        zF = propagate_canon(z0, w, eta, T)
        JF = mode_decomp(zF, KxF, KyF, w)
        phop_list[i] = JF[1] / (JF[0] + JF[1])

    return phop_list.mean(), phop_list.std()


def main():
    print("==== Ensemble LZ verification ====")
    print()
    print("All runs: eta = 0.5, J0 = 1, N = 200.")
    print("Dressed ensemble: all particles launched with J+ = J0 exactly.")
    print("Bare ensemble: Gaussian in (x, px) only, with O(w^2) admixture in mode -.")
    print()

    eta = 0.5
    N = 200
    J0 = 1.0

    print(" Scan over T at w = 0.02 ")
    print(f"  {'T':>7s}    {'LZ pred':>9s}    {'<P>_dressed':>20s}    {'<P>_bare':>20s}")
    for T in [200.0, 500.0, 1000.0, 2000.0]:
        w = 0.02
        pred = np.exp(-np.pi * w**2 * T / eta)
        dm, ds = run_ensemble_dressed(w, eta, T, N, J0)
        bm, bs = run_ensemble_bare(w, eta, T, N, J0)
        print(
            f"  {T:7.1f}    {pred:9.3e}"
            f"    {dm:.3e} +/- {ds:.3e}"
            f"    {bm:.3e} +/- {bs:.3e}"
        )
    print()

    print(" Scan over w at T = 1000 ")
    print(f"  {'w':>7s}    {'LZ pred':>9s}    {'<P>_dressed':>20s}    {'<P>_bare':>20s}")
    for w in [0.005, 0.01, 0.02, 0.03]:
        T = 1000.0
        pred = np.exp(-np.pi * w**2 * T / eta)
        dm, ds = run_ensemble_dressed(w, eta, T, N, J0)
        bm, bs = run_ensemble_bare(w, eta, T, N, J0)
        print(
            f"  {w:7.4f}    {pred:9.3e}"
            f"    {dm:.3e} +/- {ds:.3e}"
            f"    {bm:.3e} +/- {bs:.3e}"
        )


if __name__ == "__main__":
    main()
