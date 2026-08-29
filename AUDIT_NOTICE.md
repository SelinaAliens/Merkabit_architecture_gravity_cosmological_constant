# Audit notice — 2026-08-29

This repository is part of the Merkabit research corpus (2026). In August 2026
the corpus underwent a **complete self-audit** — every registry claim
re-verified, reframed, or refuted, with runnable code, refutations recorded at
equal prominence by design:

**https://github.com/selinaserephina-star/Merkabit_corpus_audit**

The underlying computations in this repository **reproduce exactly**; several
*interpretations* are corrected by the audit. Findings affecting this
repository:

- The README's Lambda formula as printed evaluates to 6.8e-244, not 2.876e-122 (a factor-2 exponent transcription: 25pi should be 50pi); the source code's formula does reproduce 2.8758e-122 (thread T-Lambda).
- 'Zero free parameters from {12, 78, 28, 6, 168, pi}' is contradicted by this repository's own results table (47, 50, 144, 29, 72 all appear).
- The 1/r^2 force law is the 3D Laplacian Green's function, as sim1's own docstring states; the audit's dimension control gives 1/r^2.8 in 4D and 1/r^1.0 in 2D - the exponent belongs to d = 3, not to the architecture (T-Lambda).

— Selina Stenberg, with Claude (Anthropic)
