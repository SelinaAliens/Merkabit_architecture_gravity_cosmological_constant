# merkabit-architecture

Simulation code for the Merkabit framework Papers 20-24: deriving gravity, the cosmological constant, Newton's constant, and the gauge hierarchy from E6 Coxeter geometry on the Eisenstein lattice.

**Zero free parameters.** Every fundamental constant derived from `{h=12, dim(E6)=78, dim(D4)=28, rank=6, |PSL(2,7)|=168, pi}`.

## Results Summary

| Constant | Formula | Derived | Measured | Accuracy |
|----------|---------|---------|----------|----------|
| Lambda | sqrt(3/2) x exp(-47x78x12 / 25pi) | 2.876e-122 | 2.87e-122 | 0.2% |
| alpha^-1 | 168 - 28 - 3 | 137 | 137.036 | 0.026% |
| v (Higgs) | m_P x exp(-12pi - 1/72) / 2 | 255.1 GeV | 255.0 GeV | 0.056% |
| v_phys | v_tree x 28/29 | 246.3 GeV | 246.22 GeV | 0.05% |
| m_W | v x 47/144 | 80.40 GeV | 80.377 GeV | 0.03% |
| G_eff | 1/N_spinor^2 = 1/4 | 0.250 | 0.254 | 1.7% |
| gamma_Berry | 47/50 | 0.940 | 0.9400068 | 7 ppm |

## Structure

```
core/                    -- Shared infrastructure (gates, states, Berry phase)
paper_20_gravity/        -- Sims 1-4: torsion gravity, dark matter (sedenion)
paper_21_quantisation/   -- Sims 5-6, 9: orbital quantisation, CPT, equivalence
paper_23_lambda/         -- Sims 10-12: cosmological constant, gamma = 47/50
paper_24_hierarchy/      -- Sims 13-14: G_eff, hierarchy, Weyl anti-gravity
output/                  -- Simulation output files
```

## Quick Start

```bash
pip install numpy scipy
python paper_23_lambda/sim10_monopole.py        # Lambda from monopole suppression
python paper_23_lambda/sim11_sqrt32.py           # The sqrt(3/2) factor
python paper_24_hierarchy/sim13_closure_scale.py  # v = m_P * exp(-12pi) / 2
python paper_24_hierarchy/sim14b_octonion_envelope.py  # Weyl anti-gravity
```

## Papers

- **Paper 20**: Gravity and Dark Matter from the Eisenstein Lattice
- **Paper 21**: Orbital Quantisation, CPT, and the Equivalence Principle from Torsion Geometry
- **Paper 23**: The Cosmological Constant from Vacuum Monopole Suppression and the Berry Phase as Geometric Complement
- **Paper 24**: Newton's Constant, the Ternary-Binary Coupling, and the Hierarchy as a Winding Problem

All papers available on [Zenodo](https://zenodo.org/communities/merkabit).

## The Key Identity

The fine structure constant and the cosmological constant share a common geometric root:

```
alpha^-1 = |PSL(2,7)| - dim(D4) - rank(E6)/2 = 168 - 28 - 3 = 137
gamma    = (dim(E6) - dim(D4) - rank/2) / (dim(E6) - dim(D4)) = 47/50
```

Both subtract the matter sector |B_31| = 31 from different total counts. The electromagnetic coupling counts the non-matter elements of PSL(2,7). The vacuum Berry phase counts the non-matter dimensions of E6.

## The Hierarchy

```
v = m_P x exp(-12pi) / 2
```

Euler's identity on the real axis. Each of h=12 Coxeter steps accumulates pi radians of Berry phase. The closure probability is exp(-12pi) ~ 10^-17. That probability times the Planck mass is the electroweak scale.

## License

Open source. The mathematics belongs to everyone.

## Author

Selina Stenberg with Claude Anthropic, 2026
