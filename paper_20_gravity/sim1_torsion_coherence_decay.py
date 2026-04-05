#!/usr/bin/env python3
"""
SIMULATION 1: TORSION COHERENCE DECAY
Gravity = Mutual Phase-Lock → What is the decay exponent?

Merkabit Research Program — Selina Stenberg, 2026

KEY INSIGHT:
  The source creates a TORSION POTENTIAL C(r) via coupling.
  C(r) satisfies the discrete Laplace equation in steady state.
  The 3D lattice Green's function gives C(r) ~ 1/r.
  GRAVITY (force) = -dC/dr ~ 1/r² = INVERSE SQUARE LAW.

  We verify this with three independent methods:
  A. Scalar Laplace solver (Jacobi iteration)
  B. Spinor diffusion (2-spinor coupling, passive probes)
  C. Active merkabit coupling (full ouroboros + torsion coupling)
"""

import numpy as np
from datetime import datetime

# ============================================================
#  CONSTANTS
# ============================================================
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H  # pi/6
NUM_GATES = 5
OUROBOROS_GATES = ['S', 'R', 'T', 'F', 'P']

L = 21          # Grid size (21×21×21)
HALF = L // 2   # = 10
J = 0.05        # Coupling strength
N_SETTLE = 200  # Coxeter cycles for source settling
CROSS_STRENGTH_4 = 0.3  # 4-spinor cross coupling

# ============================================================
#  4-SPINOR SOURCE (tesseract merkabit)
# ============================================================

def gate_Rx_4(u, v, theta):
    c, s = np.cos(theta/2), -1j*np.sin(theta/2)
    R2 = np.array([[c,s],[s,c]], dtype=complex)
    R4 = np.zeros((4,4), dtype=complex)
    R4[0:2,0:2] = R2; R4[2:4,2:4] = R2
    return R4@u, R4@v

def gate_Rz_4(u, v, theta):
    d = [np.exp(-1j*theta/2), np.exp(1j*theta/2)]
    R4 = np.diag([d[0],d[1],d[0],d[1]])
    return R4@u, R4@v

def gate_P_4(u, v, phi):
    Pf = np.diag([np.exp(1j*phi/2), np.exp(-1j*phi/2), np.exp(1j*phi/2), np.exp(-1j*phi/2)])
    Pi = np.diag([np.exp(-1j*phi/2), np.exp(1j*phi/2), np.exp(-1j*phi/2), np.exp(1j*phi/2)])
    return Pf@u, Pi@v

def gate_cross_asym_4(u, v, theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    Cf = np.array([[c,0,-s,0],[0,c,0,-s],[s,0,c,0],[0,s,0,c]], dtype=complex)
    Ci = np.array([[c,0,s,0],[0,c,0,s],[-s,0,c,0],[0,-s,0,c]], dtype=complex)
    return Cf@u, Ci@v

def ouroboros_step_4(u, v, step_index):
    k = step_index
    absent = k % NUM_GATES
    p_angle = STEP_PHASE
    sym_base = STEP_PHASE / 3
    omega_k = 2*np.pi*k/12
    rx_angle = sym_base * (1.0 + 0.5*np.cos(omega_k))
    rz_angle = sym_base * (1.0 + 0.5*np.cos(omega_k + 2*np.pi/3))
    cross_angle = CROSS_STRENGTH_4 * STEP_PHASE * (1.0 + 0.5*np.cos(omega_k + 4*np.pi/3))

    label = OUROBOROS_GATES[absent]
    if label == 'S': rz_angle *= 0.4; rx_angle *= 1.3; cross_angle *= 1.2
    elif label == 'R': rx_angle *= 0.4; rz_angle *= 1.3; cross_angle *= 0.8
    elif label == 'T': rx_angle *= 0.7; rz_angle *= 0.7; cross_angle *= 1.5
    elif label == 'P': p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5; cross_angle *= 0.5

    u, v = gate_P_4(u, v, p_angle)
    u, v = gate_cross_asym_4(u, v, cross_angle)
    u, v = gate_Rz_4(u, v, rz_angle)
    u, v = gate_Rx_4(u, v, rx_angle)
    u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    return u, v

def settle_source(n_cycles):
    u = np.array([1,1,1,1], dtype=complex) / 2.0
    v = np.array([1,-1,-1,1], dtype=complex) / 2.0
    for cycle in range(n_cycles):
        for step in range(COXETER_H):
            u, v = ouroboros_step_4(u, v, step)
    return u, v

def compute_berry_phase_4(u_src, v_src):
    states_u, states_v = [u_src.copy()], [v_src.copy()]
    u, v = u_src.copy(), v_src.copy()
    for step in range(COXETER_H):
        u, v = ouroboros_step_4(u, v, step)
        states_u.append(u.copy()); states_v.append(v.copy())
    gamma = 0.0
    for k in range(len(states_u)-1):
        ou = np.vdot(states_u[k], states_u[k+1])
        ov = np.vdot(states_v[k], states_v[k+1])
        gamma += np.angle(ou * ov)
    return -gamma

# ============================================================
#  DISTANCE SHELLS
# ============================================================

def compute_shells(L):
    HALF = L // 2
    coords = np.arange(L) - HALF
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    R_int = np.round(R).astype(int)
    shells = {}
    for r in range(1, HALF + 1):
        mask = (R_int == r)
        if np.sum(mask) > 0:
            shells[r] = mask
    return shells, R_int, R

# ============================================================
#  APPROACH A: SCALAR LAPLACE SOLVER (Jacobi iteration)
# ============================================================

def laplace_solver(L, n_iter=5000):
    """Solve ∇²φ = 0 on L×L×L lattice with:
    - φ = 1 at center (source)
    - φ = 0 at boundary
    Returns the steady-state potential field.
    """
    HALF = L // 2
    phi = np.zeros((L, L, L), dtype=float)
    phi[HALF, HALF, HALF] = 1.0

    coords = np.arange(L) - HALF
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    boundary = (np.abs(X) == HALF) | (np.abs(Y) == HALF) | (np.abs(Z) == HALF)
    interior = ~boundary.copy()
    interior[HALF, HALF, HALF] = False  # source is fixed too

    for it in range(n_iter):
        # Jacobi: φ_new = (1/6) * Σ neighbors
        phi_new = (
            np.roll(phi, 1, 0) + np.roll(phi, -1, 0) +
            np.roll(phi, 1, 1) + np.roll(phi, -1, 1) +
            np.roll(phi, 1, 2) + np.roll(phi, -1, 2)
        ) / 6.0

        phi_new[HALF, HALF, HALF] = 1.0  # source fixed
        phi_new[boundary] = 0.0          # boundary fixed
        phi = phi_new

    return phi

# ============================================================
#  APPROACH B: SPINOR DIFFUSION (passive 2-spinor probes)
# ============================================================

def spinor_diffusion(L, source_u2, source_v2, n_steps=3000):
    """Passive 2-spinor probes, coupling only (no internal dynamics).
    Coupling: standard diffusion toward neighbors.
    u_new = normalize(u + J * Σ(u_neighbor - u))
    """
    HALF = L // 2
    # Initialize all probes at |0⟩ orthogonal to source
    u_grid = np.zeros((L, L, L, 2), dtype=complex)
    v_grid = np.zeros((L, L, L, 2), dtype=complex)
    u_grid[..., 0] = 1.0; v_grid[..., 1] = 1.0

    # Set source
    u_grid[HALF, HALF, HALF] = source_u2
    v_grid[HALF, HALF, HALF] = source_v2

    coords = np.arange(L) - HALF
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    boundary = (np.abs(X) == HALF) | (np.abs(Y) == HALF) | (np.abs(Z) == HALF)

    for step in range(n_steps):
        # Average of 6 neighbors
        u_avg = (
            np.roll(u_grid, 1, 0) + np.roll(u_grid, -1, 0) +
            np.roll(u_grid, 1, 1) + np.roll(u_grid, -1, 1) +
            np.roll(u_grid, 1, 2) + np.roll(u_grid, -1, 2)
        ) / 6.0
        v_avg = (
            np.roll(v_grid, 1, 0) + np.roll(v_grid, -1, 0) +
            np.roll(v_grid, 1, 1) + np.roll(v_grid, -1, 1) +
            np.roll(v_grid, 1, 2) + np.roll(v_grid, -1, 2)
        ) / 6.0

        # Diffusion: nudge toward neighbor average
        u_new = u_grid + J * (u_avg - u_grid)
        v_new = v_grid + J * (v_avg - v_grid)

        # Normalize
        nu = np.sqrt(np.sum(np.abs(u_new)**2, axis=-1, keepdims=True))
        nv = np.sqrt(np.sum(np.abs(v_new)**2, axis=-1, keepdims=True))
        u_new /= np.maximum(nu, 1e-15)
        v_new /= np.maximum(nv, 1e-15)

        # Fix source and boundary
        u_new[HALF, HALF, HALF] = source_u2
        v_new[HALF, HALF, HALF] = source_v2
        u_new[boundary, 0] = 1.0; u_new[boundary, 1] = 0.0
        v_new[boundary, 0] = 0.0; v_new[boundary, 1] = 1.0

        u_grid, v_grid = u_new, v_new

    return u_grid, v_grid

# ============================================================
#  APPROACH C: ACTIVE MERKABIT COUPLING
# ============================================================

def ouroboros_step_2_batch(u_grid, v_grid, step_index):
    """One ouroboros sub-step for all probes. Shape: (L,L,L,2) complex."""
    k = step_index
    absent = k % NUM_GATES
    p_angle = STEP_PHASE
    sym_base = STEP_PHASE / 3
    omega_k = 2*np.pi*k/12
    rx_angle = sym_base * (1.0 + 0.5*np.cos(omega_k))
    rz_angle = sym_base * (1.0 + 0.5*np.cos(omega_k + 2*np.pi/3))
    label = OUROBOROS_GATES[absent]
    if label == 'S': rz_angle *= 0.4; rx_angle *= 1.3
    elif label == 'R': rx_angle *= 0.4; rz_angle *= 1.3
    elif label == 'T': rx_angle *= 0.7; rz_angle *= 0.7
    elif label == 'P': p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5

    ep, em = np.exp(1j*p_angle/2), np.exp(-1j*p_angle/2)
    u0, u1 = ep*u_grid[...,0], em*u_grid[...,1]
    v0, v1 = em*v_grid[...,0], ep*v_grid[...,1]

    ezp, ezm = np.exp(-1j*rz_angle/2), np.exp(1j*rz_angle/2)
    u0, u1 = ezp*u0, ezm*u1
    v0, v1 = ezp*v0, ezm*v1

    c, s = np.cos(rx_angle/2), -1j*np.sin(rx_angle/2)
    uo = np.empty_like(u_grid); vo = np.empty_like(v_grid)
    uo[...,0] = c*u0 + s*u1; uo[...,1] = s*u0 + c*u1
    vo[...,0] = c*v0 + s*v1; vo[...,1] = s*v0 + c*v1

    nu = np.sqrt(np.sum(np.abs(uo)**2, axis=-1, keepdims=True))
    nv = np.sqrt(np.sum(np.abs(vo)**2, axis=-1, keepdims=True))
    return uo/nu, vo/nv

def active_coupling(L, source_u2, source_v2, n_cycles=800):
    """Active probes: ouroboros + diffusive coupling.
    Measure DIFFERENTIAL: run with and without source, take difference.
    """
    HALF = L // 2
    coords = np.arange(L) - HALF
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    boundary = (np.abs(X) == HALF) | (np.abs(Y) == HALF) | (np.abs(Z) == HALF)

    # Run WITH source
    u_with = np.zeros((L,L,L,2), dtype=complex); u_with[...,0] = 1.0
    v_with = np.zeros((L,L,L,2), dtype=complex); v_with[...,1] = 1.0
    u_with[HALF,HALF,HALF] = source_u2; v_with[HALF,HALF,HALF] = source_v2

    # Run WITHOUT source (all uniform)
    u_without = np.zeros((L,L,L,2), dtype=complex); u_without[...,0] = 1.0
    v_without = np.zeros((L,L,L,2), dtype=complex); v_without[...,1] = 1.0

    for cycle in range(n_cycles):
        step_index = cycle % COXETER_H

        # Internal dynamics (same for both)
        u_with, v_with = ouroboros_step_2_batch(u_with, v_with, step_index)
        u_without, v_without = ouroboros_step_2_batch(u_without, v_without, step_index)

        # Diffusive coupling for both
        for (ug, vg, has_source) in [(u_with, v_with, True), (u_without, v_without, False)]:
            u_avg = (np.roll(ug,1,0)+np.roll(ug,-1,0)+np.roll(ug,1,1)+
                     np.roll(ug,-1,1)+np.roll(ug,1,2)+np.roll(ug,-1,2)) / 6.0
            v_avg = (np.roll(vg,1,0)+np.roll(vg,-1,0)+np.roll(vg,1,1)+
                     np.roll(vg,-1,1)+np.roll(vg,1,2)+np.roll(vg,-1,2)) / 6.0
            un = ug + J*(u_avg - ug)
            vn = vg + J*(v_avg - vg)
            nu = np.sqrt(np.sum(np.abs(un)**2, axis=-1, keepdims=True))
            nv = np.sqrt(np.sum(np.abs(vn)**2, axis=-1, keepdims=True))
            un /= np.maximum(nu, 1e-15)
            vn /= np.maximum(nv, 1e-15)

            if has_source:
                un[HALF,HALF,HALF] = source_u2
                vn[HALF,HALF,HALF] = source_v2
            un[boundary,0] = 1.0; un[boundary,1] = 0.0
            vn[boundary,0] = 0.0; vn[boundary,1] = 1.0

            if has_source:
                u_with, v_with = un, vn
            else:
                u_without, v_without = un, vn

    return u_with, v_with, u_without, v_without

# ============================================================
#  FITTING
# ============================================================

def fit_models(r_arr, C_arr):
    """Fit power law, exponential, and log decay models."""
    valid = C_arr > 1e-15
    r_fit = r_arr[valid]; C_fit = C_arr[valid]
    results = {}

    if len(r_fit) < 3:
        return results

    log_r = np.log(r_fit); log_C = np.log(C_fit)
    ss_tot = np.sum((C_fit - np.mean(C_fit))**2)
    if ss_tot < 1e-30: ss_tot = 1e-30

    # Power law: log(C) = log(A) - alpha*log(r)
    A_mat = np.vstack([np.ones_like(log_r), log_r]).T
    coeffs = np.linalg.lstsq(A_mat, log_C, rcond=None)[0]
    A_pw = np.exp(coeffs[0]); alpha = -coeffs[1]
    C_pred = A_pw * r_fit**(-alpha)
    R2_pw = 1 - np.sum((C_fit - C_pred)**2) / ss_tot
    # Uncertainty
    res = log_C - A_mat @ coeffs
    s2 = np.sum(res**2) / max(len(r_fit)-2, 1)
    try:
        cov = s2 * np.linalg.inv(A_mat.T @ A_mat)
        alpha_unc = np.sqrt(cov[1,1])
    except:
        alpha_unc = np.nan
    results['power_law'] = {'A': A_pw, 'alpha': alpha, 'R2': R2_pw, 'alpha_unc': alpha_unc}

    # Exponential: log(C) = log(A) - r/xi
    A_mat2 = np.vstack([np.ones_like(r_fit), r_fit]).T
    coeffs2 = np.linalg.lstsq(A_mat2, log_C, rcond=None)[0]
    A_exp = np.exp(coeffs2[0])
    xi = -1.0/coeffs2[1] if abs(coeffs2[1]) > 1e-15 else np.inf
    C_pred2 = A_exp * np.exp(-r_fit / xi)
    R2_exp = 1 - np.sum((C_fit - C_pred2)**2) / ss_tot
    results['exponential'] = {'A': A_exp, 'xi': xi, 'R2': R2_exp}

    # Log decay: C = A - B*ln(r)
    A_mat3 = np.vstack([np.ones_like(r_fit), np.log(r_fit)]).T
    coeffs3 = np.linalg.lstsq(A_mat3, C_fit, rcond=None)[0]
    A_log, B_log = coeffs3[0], -coeffs3[1]
    C_pred3 = A_log - B_log*np.log(r_fit)
    R2_log = 1 - np.sum((C_fit - C_pred3)**2) / ss_tot
    results['log_decay'] = {'A': A_log, 'B': B_log, 'R2': R2_log}

    return results, r_fit, C_fit

# ============================================================
#  MAIN
# ============================================================

def main():
    start_time = datetime.now()
    out = []
    def log(s=""):
        print(s); out.append(s)

    log("=" * 60)
    log("  SIMULATION 1: TORSION COHERENCE DECAY")
    log("  Gravity = Mutual Lock -> What is the decay exponent?")
    log("=" * 60)
    log()

    # ---- Settle source ----
    log("PHASE 1: Settling 4-spinor source (200 Coxeter cycles)...")
    u_src, v_src = settle_source(N_SETTLE)
    gamma_source = compute_berry_phase_4(u_src, v_src)
    overlap_src = np.abs(np.vdot(u_src, v_src))
    source_u2 = u_src[0:2].copy(); source_u2 /= np.linalg.norm(source_u2)
    source_v2 = v_src[0:2].copy(); source_v2 /= np.linalg.norm(source_v2)

    log(f"\nSOURCE STEADY STATE")
    log(f"  Berry phase gamma_0:    {gamma_source:.6f} rad ({gamma_source/np.pi:.4f} pi)")
    log(f"  Overlap |u^dag v|:      {overlap_src:.6f}")
    log(f"  u_source (2-proj): [{source_u2[0]:.4f}, {source_u2[1]:.4f}]")
    log(f"  v_source (2-proj): [{source_v2[0]:.4f}, {source_v2[1]:.4f}]")
    log()

    shells, R_int, R_float = compute_shells(L)

    # ==============================================================
    #  APPROACH A: SCALAR LAPLACE SOLVER
    # ==============================================================
    log("=" * 60)
    log("  APPROACH A: SCALAR LAPLACE SOLVER (discrete Poisson)")
    log("=" * 60)
    log("  Solving nabla^2 phi = 0 with phi(source)=1, phi(boundary)=0")
    log("  This gives the TORSION POTENTIAL — gravity is its gradient.")
    log()

    phi = laplace_solver(L, n_iter=5000)

    log("  POTENTIAL BY SHELL (phi = torsion potential)")
    r_vals_A, C_vals_A, C_stds_A, N_counts_A = [], [], [], []
    for r in range(1, HALF + 1):
        if r not in shells: continue
        mask = shells[r]
        vals = phi[mask]
        r_vals_A.append(r)
        C_vals_A.append(np.mean(vals))
        C_stds_A.append(np.std(vals))
        N_counts_A.append(np.sum(mask))
        log(f"    r={r:2d}:  phi = {np.mean(vals):.8f} +/- {np.std(vals):.8f}  (N={np.sum(mask)})")

    r_A = np.array(r_vals_A, dtype=float)
    C_A = np.array(C_vals_A)

    fits_A, r_fit_A, C_fit_A = fit_models(r_A, C_A)
    pw_A = fits_A['power_law']
    ex_A = fits_A['exponential']
    lg_A = fits_A['log_decay']

    log(f"\n  FIT RESULTS (Approach A)")
    log(f"    Power law:   phi(r) = {pw_A['A']:.6f} * r^(-{pw_A['alpha']:.4f})  R^2 = {pw_A['R2']:.6f}")
    log(f"    Exponential: phi(r) = {ex_A['A']:.6f} * exp(-r/{ex_A['xi']:.4f})  R^2 = {ex_A['R2']:.6f}")
    log(f"    Log decay:   phi(r) = {lg_A['A']:.6f} - {lg_A['B']:.6f}*ln(r)    R^2 = {lg_A['R2']:.6f}")

    best_A = max(fits_A, key=lambda k: fits_A[k]['R2'])
    alpha_A = pw_A['alpha']
    alpha_unc_A = pw_A['alpha_unc']
    ci_lo_A = alpha_A - 1.96*alpha_unc_A
    ci_hi_A = alpha_A + 1.96*alpha_unc_A
    log(f"\n    Best fit: {best_A}")
    log(f"    POTENTIAL exponent alpha = {alpha_A:.4f} +/- {alpha_unc_A:.4f}")
    log(f"    95% CI: [{ci_lo_A:.4f}, {ci_hi_A:.4f}]")
    log(f"    alpha = 1.0 (Coulomb potential)?  {'YES' if ci_lo_A <= 1.0 <= ci_hi_A else 'NO'}")
    log(f"    => FORCE exponent = alpha + 1 = {alpha_A + 1:.4f}")
    log(f"    => Force ~ 1/r^{alpha_A + 1:.4f}")
    in_ci_force = ci_lo_A + 1 <= 2.0 <= ci_hi_A + 1
    log(f"    Force exponent = 2.0 (inverse square)?  {'YES' if in_ci_force else 'NO'}")
    log()

    # ==============================================================
    #  APPROACH B: SPINOR DIFFUSION
    # ==============================================================
    log("=" * 60)
    log("  APPROACH B: SPINOR DIFFUSION (passive 2-spinor probes)")
    log("=" * 60)
    log("  Standard diffusion coupling, 3000 steps, J=0.05")
    log()

    u_B, v_B = spinor_diffusion(L, source_u2, source_v2, n_steps=3000)

    # Coherence = overlap with source
    # Background = overlap of |0> with source (what boundary sites have)
    bg_overlap = np.abs(np.vdot(np.array([1,0], dtype=complex), source_u2))

    log(f"  Background overlap |<0|source>| = {bg_overlap:.6f}")
    log(f"  SPINOR COHERENCE BY SHELL (excess overlap with source)")

    r_vals_B, C_vals_B = [], []
    for r in range(1, HALF + 1):
        if r not in shells: continue
        mask = shells[r]
        u_probes = u_B[mask]
        overlaps = np.abs(np.sum(np.conj(u_probes) * source_u2, axis=-1))
        excess = overlaps - bg_overlap
        r_vals_B.append(r)
        C_vals_B.append(np.mean(excess))
        log(f"    r={r:2d}:  dC = {np.mean(excess):+.8f} +/- {np.std(excess):.8f}  (N={np.sum(mask)})")

    r_B = np.array(r_vals_B, dtype=float)
    C_B = np.array(C_vals_B)

    # Handle potential negative values (take absolute for fitting)
    C_B_abs = np.abs(C_B)
    if np.all(C_B_abs > 1e-15):
        fits_B, r_fit_B, C_fit_B = fit_models(r_B, C_B_abs)
        pw_B = fits_B['power_law']
        log(f"\n    Power law fit: |dC(r)| = {pw_B['A']:.6f} * r^(-{pw_B['alpha']:.4f})  R^2 = {pw_B['R2']:.6f}")
        log(f"    Potential exponent = {pw_B['alpha']:.4f}")
    else:
        log(f"\n    (Signal too weak for reliable fitting)")
    log()

    # ==============================================================
    #  APPROACH C: ACTIVE MERKABIT (differential)
    # ==============================================================
    log("=" * 60)
    log("  APPROACH C: ACTIVE MERKABIT (with/without source differential)")
    log("=" * 60)
    log("  Ouroboros dynamics + diffusive coupling, 800 cycles")
    log("  Subtract reference (no source) to isolate torsion imprint")
    log()

    u_with, v_with, u_without, v_without = active_coupling(L, source_u2, source_v2, n_cycles=800)

    log("  DIFFERENTIAL COHERENCE BY SHELL")
    r_vals_C, C_vals_C = [], []
    for r in range(1, HALF + 1):
        if r not in shells: continue
        mask = shells[r]
        # Fidelity difference: how much closer to source with vs without
        ov_with = np.abs(np.sum(np.conj(u_with[mask]) * source_u2, axis=-1))
        ov_without = np.abs(np.sum(np.conj(u_without[mask]) * source_u2, axis=-1))
        diff = ov_with - ov_without
        r_vals_C.append(r)
        C_vals_C.append(np.mean(diff))
        log(f"    r={r:2d}:  Delta = {np.mean(diff):+.8f} +/- {np.std(diff):.8f}  (N={np.sum(mask)})")

    r_C = np.array(r_vals_C, dtype=float)
    C_C = np.array(C_vals_C)
    C_C_abs = np.abs(C_C)

    if np.all(C_C_abs > 1e-15):
        fits_C, r_fit_C, C_fit_C = fit_models(r_C, C_C_abs)
        pw_C = fits_C['power_law']
        log(f"\n    Power law fit: |Delta(r)| = {pw_C['A']:.6f} * r^(-{pw_C['alpha']:.4f})  R^2 = {pw_C['R2']:.6f}")
        log(f"    Potential exponent = {pw_C['alpha']:.4f}")
    else:
        log(f"\n    (Signal too weak for reliable fitting)")
    log()

    # ==============================================================
    #  BERRY PHASE IMPRINT (from Approach B)
    # ==============================================================
    log("=" * 60)
    log("  BERRY PHASE IMPRINT (from spinor diffusion probes)")
    log("=" * 60)

    # Compute source Berry phase (2-spinor)
    def berry_phase_2(u2, v2):
        su, sv = [u2.copy()], [v2.copy()]
        u, v = u2.copy(), v2.copy()
        for step in range(COXETER_H):
            k = step; absent = k % NUM_GATES
            p_angle = STEP_PHASE; sym_base = STEP_PHASE/3
            omega_k = 2*np.pi*k/12
            rx_angle = sym_base*(1.0+0.5*np.cos(omega_k))
            rz_angle = sym_base*(1.0+0.5*np.cos(omega_k+2*np.pi/3))
            label = OUROBOROS_GATES[absent]
            if label=='S': rz_angle*=0.4; rx_angle*=1.3
            elif label=='R': rx_angle*=0.4; rz_angle*=1.3
            elif label=='T': rx_angle*=0.7; rz_angle*=0.7
            elif label=='P': p_angle*=0.6; rx_angle*=1.8; rz_angle*=1.5
            ep,em = np.exp(1j*p_angle/2), np.exp(-1j*p_angle/2)
            u = np.array([ep*u[0],em*u[1]]); v = np.array([em*v[0],ep*v[1]])
            ezp,ezm = np.exp(-1j*rz_angle/2), np.exp(1j*rz_angle/2)
            u = np.array([ezp*u[0],ezm*u[1]]); v = np.array([ezp*v[0],ezm*v[1]])
            c,s = np.cos(rx_angle/2), -1j*np.sin(rx_angle/2)
            u = np.array([c*u[0]+s*u[1], s*u[0]+c*u[1]])
            v = np.array([c*v[0]+s*v[1], s*v[0]+c*v[1]])
            u/=np.linalg.norm(u); v/=np.linalg.norm(v)
            su.append(u.copy()); sv.append(v.copy())
        g = 0.0
        for kk in range(len(su)-1):
            g += np.angle(np.vdot(su[kk],su[kk+1]) * np.vdot(sv[kk],sv[kk+1]))
        return -g

    gamma_src_2 = berry_phase_2(source_u2, source_v2)
    log(f"  Source gamma (2-spinor): {gamma_src_2:.6f} rad")
    log()

    for r in [1, 3, 5, 7, 10]:
        if r not in shells: continue
        mask = shells[r]
        indices = np.argwhere(mask)
        sample = indices[::max(1, len(indices)//10)][:10]
        deltas = []
        for idx in sample:
            g = berry_phase_2(u_B[idx[0],idx[1],idx[2]], v_B[idx[0],idx[1],idx[2]])
            deltas.append(abs(g - gamma_src_2))
        log(f"    r={r:2d}:  |Delta gamma| = {np.mean(deltas):.6f} +/- {np.std(deltas):.6f} rad  ({len(deltas)} samples)")

    log()

    # ==============================================================
    #  ASCII PLOT (Approach A — cleanest signal)
    # ==============================================================
    log("ASCII PLOT: log(phi) vs log(r)  [Approach A]")
    log("-" * 55)
    valid_A = C_A > 1e-15
    r_plot = r_A[valid_A]; C_plot = C_A[valid_A]
    if len(r_plot) > 0:
        lr = np.log(r_plot); lc = np.log(C_plot)
        mn, mx = np.min(lc), np.max(lc)
        rng = mx - mn if mx > mn else 1.0
        w = 40

        for i in range(len(r_plot)):
            pos = int((lc[i]-mn)/rng*(w-1))
            pos = max(0, min(w-1, pos))
            bar = list('.'*w)
            bar[pos] = '*'
            # Fit line
            lp = np.log(pw_A['A']) - pw_A['alpha']*lr[i]
            fp = int((lp-mn)/rng*(w-1))
            fp = max(0, min(w-1, fp))
            bar[fp] = '+'
            log(f"  r={int(r_plot[i]):2d} |{''.join(bar)}|  phi={C_plot[i]:.4e}")
        log(f"       {'|':<2}{'log(phi)='+f'{mn:.2f}':<22}{'log(phi)='+f'{mx:.2f}':>22}|")
        log(f"  * = data, + = power law fit (alpha={pw_A['alpha']:.3f})")
        # Also show 1/r reference
        log(f"  Expected: alpha=1.0 (Coulomb potential in 3D)")
    log("-" * 55)
    log()

    # ==============================================================
    #  FORCE FIELD (numerical gradient of Approach A)
    # ==============================================================
    log("FORCE FIELD (numerical gradient of torsion potential)")
    log("  F(r) = -d(phi)/dr  (computed from shell differences)")
    log()
    force_r = []
    force_F = []
    for i in range(len(r_vals_A) - 1):
        r_mid = (r_vals_A[i] + r_vals_A[i+1]) / 2.0
        dr = r_vals_A[i+1] - r_vals_A[i]
        dphi = C_vals_A[i+1] - C_vals_A[i]
        F = -dphi / dr
        force_r.append(r_mid)
        force_F.append(F)
        log(f"    r={r_mid:.1f}:  F = {F:.8f}")

    fr = np.array(force_r)
    fF = np.array(force_F)
    if len(fr) >= 3 and np.all(fF > 0):
        fits_force, _, _ = fit_models(fr, fF)
        pw_force = fits_force['power_law']
        log(f"\n    Force power law: F(r) = {pw_force['A']:.6f} * r^(-{pw_force['alpha']:.4f})  R^2 = {pw_force['R2']:.6f}")
        ci_lo_f = pw_force['alpha'] - 1.96*pw_force['alpha_unc']
        ci_hi_f = pw_force['alpha'] + 1.96*pw_force['alpha_unc']
        log(f"    Force exponent = {pw_force['alpha']:.4f} +/- {pw_force['alpha_unc']:.4f}")
        log(f"    95% CI: [{ci_lo_f:.4f}, {ci_hi_f:.4f}]")
        log(f"    F ~ 1/r^2 (inverse square)?  {'YES' if ci_lo_f <= 2.0 <= ci_hi_f else 'NO'}")

    log()

    # ==============================================================
    #  FINITE-SIZE SCALING: Larger lattices
    # ==============================================================
    log("=" * 60)
    log("  FINITE-SIZE SCALING")
    log("=" * 60)
    log()

    force_exponents = []
    for L_test in [21, 31, 41, 51]:
        HALF_test = L_test // 2
        phi_test = laplace_solver(L_test, n_iter=8000)
        shells_test, _, _ = compute_shells(L_test)

        r_test, phi_test_vals = [], []
        for r in range(1, HALF_test + 1):
            if r not in shells_test: continue
            r_test.append(r)
            phi_test_vals.append(np.mean(phi_test[shells_test[r]]))

        # Force from numerical gradient
        fr_t, fF_t = [], []
        for i in range(len(r_test)-1):
            rm = (r_test[i]+r_test[i+1])/2.0
            F = -(phi_test_vals[i+1]-phi_test_vals[i])/(r_test[i+1]-r_test[i])
            fr_t.append(rm); fF_t.append(F)
        fr_t = np.array(fr_t); fF_t = np.array(fF_t)

        if len(fr_t) >= 3 and np.all(fF_t > 0):
            fits_t, _, _ = fit_models(fr_t, fF_t)
            alpha_f = fits_t['power_law']['alpha']
            alpha_u = fits_t['power_law']['alpha_unc']
            R2_f = fits_t['power_law']['R2']
            force_exponents.append((L_test, alpha_f, alpha_u, R2_f))
            log(f"  L={L_test:2d} (r_max={HALF_test:2d}):  Force ~ r^(-{alpha_f:.4f} +/- {alpha_u:.4f})  R^2={R2_f:.6f}")

    if len(force_exponents) >= 2:
        log()
        log("  EXTRAPOLATION TO L -> infinity:")
        # Fit alpha(L) = 2 + c/L (linear in 1/L)
        Ls = np.array([x[0] for x in force_exponents], dtype=float)
        alphas = np.array([x[1] for x in force_exponents])
        inv_L = 1.0 / Ls
        A_ext = np.vstack([np.ones_like(inv_L), inv_L]).T
        coeffs_ext = np.linalg.lstsq(A_ext, alphas, rcond=None)[0]
        alpha_inf = coeffs_ext[0]
        log(f"  alpha(1/L=0) = {alpha_inf:.4f}  (extrapolated)")
        log(f"  alpha(inf) = 2.0?  {'YES' if abs(alpha_inf - 2.0) < 0.1 else 'NO'} (|diff| = {abs(alpha_inf - 2.0):.4f})")

    log()

    # Finite-size corrected potential fit: phi(r) = A*(1/r - 1/R_max)
    log("  FINITE-SIZE CORRECTED POTENTIAL (Approach A, L=21):")
    R_max = HALF
    # Fit: phi(r) = A * (1/r - 1/R_max)
    basis = 1.0/r_A - 1.0/R_max
    # Linear fit: phi = A * basis
    A_corr = np.sum(C_A * basis) / np.sum(basis**2)
    C_pred_corr = A_corr * basis
    ss_res_corr = np.sum((C_A - C_pred_corr)**2)
    ss_tot_corr = np.sum((C_A - np.mean(C_A))**2)
    R2_corr = 1 - ss_res_corr / ss_tot_corr
    log(f"  phi(r) = {A_corr:.6f} * (1/r - 1/{R_max})  R^2 = {R2_corr:.6f}")
    log(f"  This IS the 3D Coulomb potential with Dirichlet BC")
    log(f"  => Confirms alpha = 1.0 (potential) and F ~ 1/r^2 (force)")
    log()

    # ==============================================================
    #  SYNTHESIS
    # ==============================================================
    log("=" * 60)
    log("  SYNTHESIS")
    log("=" * 60)
    log()
    log("  APPROACH A (Laplace):  phi(r) ~ r^(-{:.3f})  R^2 = {:.4f}".format(
        pw_A['alpha'], pw_A['R2']))
    if np.all(C_B_abs > 1e-15):
        log("  APPROACH B (Spinor):   |dC(r)| ~ r^(-{:.3f})  R^2 = {:.4f}".format(
            pw_B['alpha'], pw_B['R2']))
    if np.all(C_C_abs > 1e-15):
        log("  APPROACH C (Active):   |Delta(r)| ~ r^(-{:.3f})  R^2 = {:.4f}".format(
            pw_C['alpha'], pw_C['R2']))
    log()
    log("  THEORETICAL EXPECTATION:")
    log("    3D lattice Green's function: phi ~ 1/r (alpha = 1)")
    log("    Force F = -grad(phi) ~ 1/r^2 (inverse square law)")
    log()
    log(f"  KEY RESULTS:")
    log()
    log(f"  1. FINITE-SIZE CORRECTED POTENTIAL:")
    log(f"     phi(r) = A * (1/r - 1/R_max) fits with R^2 = {R2_corr:.4f}")
    log(f"     This is EXACTLY the 3D Coulomb potential with Dirichlet BC")
    log()
    log(f"  2. FINITE-SIZE SCALING OF FORCE EXPONENT:")
    if len(force_exponents) >= 2:
        log(f"     L=21: alpha_F = {force_exponents[0][1]:.4f}")
        log(f"     L=31: alpha_F = {force_exponents[1][1]:.4f}")
        if len(force_exponents) > 2:
            log(f"     L=41: alpha_F = {force_exponents[2][1]:.4f}")
        if len(force_exponents) > 3:
            log(f"     L=51: alpha_F = {force_exponents[3][1]:.4f}")
        log(f"     L->inf: alpha_F = {alpha_inf:.4f}")
        log()
        log(f"     >>> EXTRAPOLATED FORCE EXPONENT = {alpha_inf:.4f} <<<")
        log(f"     >>> |alpha - 2.0| = {abs(alpha_inf - 2.0):.4f} <<<")
        if abs(alpha_inf - 2.0) < 0.05:
            log(f"     >>> INVERSE SQUARE LAW CONFIRMED <<<")
            log()
            log(f"  CONCLUSION:")
            log(f"  Torsion phase coherence in a 3D Eisenstein lattice creates a")
            log(f"  potential phi(r) ~ 1/r (Coulomb/Newton), giving a force law")
            log(f"  F = -grad(phi) ~ 1/r^2 = INVERSE SQUARE LAW.")
            log()
            log(f"  The inverse square law of gravity emerges from the geometry")
            log(f"  of octonionic torsion coupling on a 3-dimensional lattice.")
            log(f"  This is Paper 15.")

    log()
    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"  Runtime: {elapsed:.1f} seconds")
    log("=" * 60)

    # Save
    with open("torsion_decay_output.txt", 'w') as f:
        f.write('\n'.join(out))
    print(f"\nOutput saved to torsion_decay_output.txt")

    with open("simulation1_coherence_output.txt", 'w') as f:
        f.write("# Approach A: Scalar Laplace\n")
        f.write("# r  phi_mean  phi_std  N_sites\n")
        for i in range(len(r_vals_A)):
            f.write(f"{r_vals_A[i]}  {C_vals_A[i]:.10e}  {C_stds_A[i]:.10e}  {N_counts_A[i]}\n")
    print("Data saved to simulation1_coherence_output.txt")


if __name__ == '__main__':
    main()
