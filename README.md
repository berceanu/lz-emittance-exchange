# Landau-Zener Emittance Exchange

Numerical verification and figure generation for:

> A. C. Berceanu and A. Del Dotto, "Landau-Zener Emittance Exchange of Betatron Modes in Magnetised Laser Wakefield Accelerators," *Symmetry* (2026).

## Scripts

| File | Description |
|------|-------------|
| `lz_solenoid.py` | Single-particle verification of the LZ exchange formula against ODE integration |
| `lz_ensemble.py` | Ensemble verification (dressed vs bare initial conditions) and injection-matching floor |
| `bazzani_crossing.py` | Avoided crossing with skew-quadrupole coupling (Bazzani-style reference case) |

## Figures

| File | Figure |
|------|--------|
| `figures/fig0_schematic.py` | Schematic of the crossing and amplitude transfer |
| `figures/fig1_avoided_crossing.py` | Normal-mode frequencies vs plasma anisotropy |
| `figures/fig2_lz_verification.py` | Numerical vs analytic hop probability |
| `figures/fig3_accessibility.py` | Emittance exchange fraction vs crossing length |

## Requirements

Python 3.10+ with numpy, scipy, and matplotlib:

```
pip install numpy scipy matplotlib
```

## Usage

```bash
python lz_solenoid.py
python lz_ensemble.py
python bazzani_crossing.py
cd figures && python fig0_schematic.py
```

## License

MIT
