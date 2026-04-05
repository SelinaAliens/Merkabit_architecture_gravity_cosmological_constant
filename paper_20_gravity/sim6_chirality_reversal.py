#!/usr/bin/env python3
"""
SIMULATION 6: CHIRALITY REVERSAL -- THE ANTI-GRAVITY TEST
Does opposite torsion winding repel?

Merkabit Research Program -- Selina Stenberg, 2026

Normal chirality:   u -> gate(u),  v -> gate_conj(v)  (forward/backward)
Reversed chirality: u -> gate_conj(u), v -> gate(v)   (backward/forward)
"""

import numpy as np
from datetime import datetime

# ============================================================
#  CONSTANTS
# ============================================================
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H
NUM_GATES = 5
OUROBOROS_GATES = ['S', 'R', 'T', 'F', 'P']
CROSS_STRENGTH = 0.3
J_WEAK = 0.05    # Weak coupling for Tests B, C
J_STRONG = 0.2   # Strong coupling for Test D

# ============================================================
#  4-SPINOR GATES
# ============================================================

def make_Rx4(theta):
    c, s = np.cos(theta/2), -1j*np.sin(theta/2)
    R2 = np.array([[c,s],[s,c]], dtype=complex)
    R4 = np.zeros((4,4), dtype=complex); R4[:2,:2] = R2; R4[2:,2:] = R2
    return R4

def make_Rz4(theta):
    return np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2),
                    np.exp(-1j*theta/2), np.exp(1j*theta/2)])

def make_P4_fwd(phi):
    return np.diag([np.exp(1j*phi/2), np.exp(-1j*phi/2),
                    np.exp(1j*phi/2), np.exp(-1j*phi/2)])

def make_P4_inv(phi):
    return np.diag([np.exp(-1j*phi/2), np.exp(1j*phi/2),
                    np.exp(-1j*phi/2), np.exp(1j*phi/2)])

def make_cross_fwd(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c,0,-s,0],[0,c,0,-s],[s,0,c,0],[0,s,0,c]], dtype=complex)

def make_cross_inv(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c,0,s,0],[0,c,0,s],[-s,0,c,0],[0,-s,0,c]], dtype=complex)

def compute_gate_params(step_index):
    """Compute gate angles for a given ouroboros step."""
    k = step_index
    absent = k % NUM_GATES
    p_angle = STEP_PHASE
    sym_base = STEP_PHASE / 3
    omega_k = 2 * np.pi * k / 12
    rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
    cross_angle = CROSS_STRENGTH * STEP_PHASE * (1.0 + 0.5 * np.cos(omega_k + 4*np.pi/3))

    label = OUROBOROS_GATES[absent]
    if label == 'S': rz_angle *= 0.4; rx_angle *= 1.3; cross_angle *= 1.2
    elif label == 'R': rx_angle *= 0.4; rz_angle *= 1.3; cross_angle *= 0.8
    elif label == 'T': rx_angle *= 0.7; rz_angle *= 0.7; cross_angle *= 1.5
    elif label == 'P': p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5; cross_angle *= 0.5

    return p_angle, rx_angle, rz_angle, cross_angle

# ============================================================
#  NORMAL OUROBOROS (forward chirality)
# ============================================================

def ouroboros_step_normal(u, v, step_index):
    """Normal chirality: u gets forward gates, v gets conjugate gates."""
    p_angle, rx_angle, rz_angle, cross_angle = compute_gate_params(step_index)

    # P gate: asymmetric (u forward, v inverse)
    u = make_P4_fwd(p_angle) @ u
    v = make_P4_inv(p_angle) @ v

    # Cross gate: asymmetric (u forward, v inverse)
    u = make_cross_fwd(cross_angle) @ u
    v = make_cross_inv(cross_angle) @ v

    # Rz, Rx: symmetric (same for both)
    Rz = make_Rz4(rz_angle)
    Rx = make_Rx4(rx_angle)
    u = Rx @ Rz @ u
    v = Rx @ Rz @ v

    u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    return u, v

# ============================================================
#  REVERSED OUROBOROS (backward chirality)
# ============================================================

def ouroboros_step_reversed(u, v, step_index):
    """Reversed chirality: u gets conjugate gates, v gets forward gates.
    This is the ONLY difference from normal — gate assignment is swapped.
    """
    p_angle, rx_angle, rz_angle, cross_angle = compute_gate_params(step_index)

    # P gate: REVERSED asymmetry (u inverse, v forward)
    u = make_P4_inv(p_angle) @ u
    v = make_P4_fwd(p_angle) @ v

    # Cross gate: REVERSED (u inverse, v forward)
    u = make_cross_inv(cross_angle) @ u
    v = make_cross_fwd(cross_angle) @ v

    # Rz, Rx: still symmetric
    Rz = make_Rz4(rz_angle)
    Rx = make_Rx4(rx_angle)
    u = Rx @ Rz @ u
    v = Rx @ Rz @ v

    u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    return u, v

# ============================================================
#  SETTLE AND BERRY PHASE
# ============================================================

def settle(n_cycles, step_fn):
    """Settle a merkabit using the given step function."""
    u = np.array([1, 1, 1, 1], dtype=complex) / 2.0
    v = np.array([1, -1, -1, 1], dtype=complex) / 2.0
    for cycle in range(n_cycles):
        for step in range(COXETER_H):
            u, v = step_fn(u, v, step)
    return u, v

def berry_phase(u0, v0, step_fn):
    """Compute Berry phase over one Coxeter cycle."""
    su, sv = [u0.copy()], [v0.copy()]
    u, v = u0.copy(), v0.copy()
    for step in range(COXETER_H):
        u, v = step_fn(u, v, step)
        su.append(u.copy()); sv.append(v.copy())
    g = 0.0
    for k in range(len(su)-1):
        g += np.angle(np.vdot(su[k], su[k+1]) * np.vdot(sv[k], sv[k+1]))
    return -g

# ============================================================
#  TORSION COUPLING
# ============================================================

def apply_coupling(u_probe, v_probe, u_source, v_source, d, J):
    """Torsion coupling at distance d with strength J."""
    if d < 0.5: return u_probe, v_probe
    coupling = J / d
    ov_u = np.vdot(u_source, u_probe)
    ov_v = np.vdot(v_source, v_probe)
    u_new = u_probe + coupling * ov_u * u_source
    v_new = v_probe + coupling * np.conj(ov_v) * v_source
    u_new /= np.linalg.norm(u_new); v_new /= np.linalg.norm(v_new)
    return u_new, v_new

# ============================================================
#  LAPLACE POTENTIAL (from Sim 1)
# ============================================================

def laplace_potential_1d(max_d, n_iter=5000):
    """1D discrete Laplace potential for distance analysis.
    phi(0) = 1 (source), phi(max_d+1) = 0 (boundary).
    Equivalent to phi(d) = 1 - d/(max_d+1) for 1D.
    For 3D comparison, use the known 1/r form.
    """
    # In 3D: phi(d) ~ A * (1/d - 1/R_max) from Sim 1
    R_max = max_d + 1
    phi = {}
    for d in range(1, max_d + 1):
        phi[d] = 1.0 / d - 1.0 / R_max
    # Normalize so phi(1) is the reference
    phi_1 = phi[1]
    for d in phi:
        phi[d] /= phi_1
    return phi

# ============================================================
#  MAIN SIMULATION
# ============================================================

def main():
    start = datetime.now()
    out = []
    def log(s=""):
        print(s); out.append(s)

    log("=" * 64)
    log("  SIMULATION 6: CHIRALITY REVERSAL -- THE ANTI-GRAVITY TEST")
    log("  Does opposite torsion winding repel?")
    log("=" * 64)
    log()

    # ==============================================================
    #  TEST A: REVERSED CHIRALITY SELF-COHERENCE
    # ==============================================================
    log("=" * 64)
    log("  TEST A: REVERSED CHIRALITY SELF-COHERENCE")
    log("=" * 64)
    log()

    # Normal
    u_norm, v_norm = settle(300, ouroboros_step_normal)
    gamma_norm = berry_phase(u_norm, v_norm, ouroboros_step_normal)
    overlap_norm = np.abs(np.vdot(u_norm, v_norm))

    # Reversed
    u_rev, v_rev = settle(300, ouroboros_step_reversed)
    gamma_rev = berry_phase(u_rev, v_rev, ouroboros_step_reversed)
    overlap_rev = np.abs(np.vdot(u_rev, v_rev))

    log(f"  NORMAL chirality:")
    log(f"    Berry phase gamma_norm = {gamma_norm:.6f} rad ({gamma_norm/np.pi:.4f} pi)")
    log(f"    Zero-point overlap |u^dag v| = {overlap_norm:.6f}")
    log(f"    u = [{', '.join(f'{x:.3f}' for x in u_norm)}]")
    log()
    log(f"  REVERSED chirality:")
    log(f"    Berry phase gamma_rev  = {gamma_rev:.6f} rad ({gamma_rev/np.pi:.4f} pi)")
    log(f"    Zero-point overlap |u^dag v| = {overlap_rev:.6f}")
    log(f"    u = [{', '.join(f'{x:.3f}' for x in u_rev)}]")
    log()

    ratio = gamma_rev / gamma_norm if abs(gamma_norm) > 1e-10 else np.nan
    diff_from_neg = abs(gamma_rev + gamma_norm)
    log(f"  Ratio gamma_rev / gamma_norm = {ratio:.6f}")
    log(f"  |gamma_rev + gamma_norm| = {diff_from_neg:.6f}")
    is_exact_reversal = diff_from_neg < 0.01 * abs(gamma_norm)
    log(f"  Is gamma_rev = -gamma_norm? {'YES' if is_exact_reversal else 'NO'} (diff = {diff_from_neg:.6f})")
    rev_self_sustaining = overlap_rev < 0.3
    log(f"  Reversed reaches zero-point: {'YES' if rev_self_sustaining else 'NO'} (overlap = {overlap_rev:.4f})")
    log(f"  VERDICT: Reversed chirality is self-sustaining: {'YES' if rev_self_sustaining else 'NO'}")
    log()

    # ==============================================================
    #  TEST B: SAME-CHIRALITY PAIR (Normal + Normal)
    # ==============================================================
    log("=" * 64)
    log("  TEST B: SAME-CHIRALITY BASELINE (Normal + Normal)")
    log("=" * 64)
    log()

    max_d = 12
    N_cycles_test = 200
    N_realizations = 20  # Average over random perturbations

    log(f"  Source: settled normal merkabit at origin")
    log(f"  Probe: normal merkabit at distance d")
    log(f"  Coupling J = {J_WEAK}, {N_cycles_test} cycles, {N_realizations} realizations")
    log()

    # Pre-compute source history for retarded coupling
    src_hist_u, src_hist_v = [u_norm.copy()], [v_norm.copy()]
    u_s, v_s = u_norm.copy(), v_norm.copy()
    for s in range(N_cycles_test * COXETER_H + max_d + 10):
        u_s, v_s = ouroboros_step_normal(u_s, v_s, s % COXETER_H)
        src_hist_u.append(u_s.copy()); src_hist_v.append(v_s.copy())

    log(f"  {'d':>3s}   {'L_nn':>8s}   {'phi_nn':>8s}   {'gradient':>12s}")
    log(f"  {'-'*3}   {'-'*8}   {'-'*8}   {'-'*12}")

    d_values = list(range(1, max_d + 1))
    L_nn = {}
    phi_nn = {}

    for d in d_values:
        coherences_all = []
        for trial in range(N_realizations):
            # Random perturbation on initial probe state
            u_probe = np.array([1, 0, 0, 0], dtype=complex)
            v_probe = np.array([0, 0, 0, 1], dtype=complex)
            pert = 0.01 * (np.random.randn(4) + 1j * np.random.randn(4))
            u_probe = u_probe + pert; u_probe /= np.linalg.norm(u_probe)
            pert = 0.01 * (np.random.randn(4) + 1j * np.random.randn(4))
            v_probe = v_probe + pert; v_probe /= np.linalg.norm(v_probe)

            gs = 0
            for cycle in range(N_cycles_test):
                for step in range(COXETER_H):
                    u_probe, v_probe = ouroboros_step_normal(u_probe, v_probe, step)
                    ret_idx = max(0, min(gs - d, len(src_hist_u)-1))
                    u_probe, v_probe = apply_coupling(
                        u_probe, v_probe, src_hist_u[ret_idx], src_hist_v[ret_idx], d, J_WEAK)
                    gs += 1

            coh = np.abs(np.vdot(u_probe, u_norm))
            coherences_all.append(coh)

        L_nn[d] = np.mean(coherences_all)
        phi_nn[d] = L_nn[d]  # Use coherence as potential proxy

    # Compute gradient sign
    for d in d_values:
        if d < max_d:
            grad = phi_nn[d+1] - phi_nn[d]
            grad_str = "attractive" if grad < -1e-6 else "repulsive" if grad > 1e-6 else "neutral"
        else:
            grad_str = "--"
        log(f"  {d:3d}   {L_nn[d]:.6f}   {phi_nn[d]:.6f}   {grad_str}")

    log()

    # ==============================================================
    #  TEST C: OPPOSITE-CHIRALITY (Normal + Reversed)
    # ==============================================================
    log("=" * 64)
    log("  TEST C: OPPOSITE-CHIRALITY (Normal source + Reversed probe)")
    log("=" * 64)
    log()

    log(f"  Source: settled NORMAL merkabit")
    log(f"  Probe: REVERSED chirality merkabit at distance d")
    log(f"  Coupling J = {J_WEAK}, {N_cycles_test} cycles, {N_realizations} realizations")
    log()

    log(f"  {'d':>3s}   {'L_nr':>8s}   {'phi_nr':>8s}   {'gradient':>12s}   {'L_nr/L_nn':>10s}")
    log(f"  {'-'*3}   {'-'*8}   {'-'*8}   {'-'*12}   {'-'*10}")

    L_nr = {}
    phi_nr = {}

    for d in d_values:
        coherences_all = []
        for trial in range(N_realizations):
            u_probe = np.array([1, 0, 0, 0], dtype=complex)
            v_probe = np.array([0, 0, 0, 1], dtype=complex)
            pert = 0.01 * (np.random.randn(4) + 1j * np.random.randn(4))
            u_probe = u_probe + pert; u_probe /= np.linalg.norm(u_probe)
            pert = 0.01 * (np.random.randn(4) + 1j * np.random.randn(4))
            v_probe = v_probe + pert; v_probe /= np.linalg.norm(v_probe)

            gs = 0
            for cycle in range(N_cycles_test):
                for step in range(COXETER_H):
                    # Probe uses REVERSED ouroboros
                    u_probe, v_probe = ouroboros_step_reversed(u_probe, v_probe, step)
                    # Coupling to NORMAL source (retarded)
                    ret_idx = max(0, min(gs - d, len(src_hist_u)-1))
                    u_probe, v_probe = apply_coupling(
                        u_probe, v_probe, src_hist_u[ret_idx], src_hist_v[ret_idx], d, J_WEAK)
                    gs += 1

            coh = np.abs(np.vdot(u_probe, u_norm))
            coherences_all.append(coh)

        L_nr[d] = np.mean(coherences_all)
        phi_nr[d] = L_nr[d]

    for d in d_values:
        if d < max_d:
            grad = phi_nr[d+1] - phi_nr[d]
            grad_str = "attractive" if grad < -1e-6 else "repulsive" if grad > 1e-6 else "neutral"
        else:
            grad_str = "--"
        ratio_d = L_nr[d] / L_nn[d] if L_nn[d] > 1e-10 else np.nan
        log(f"  {d:3d}   {L_nr[d]:.6f}   {phi_nr[d]:.6f}   {grad_str:>12s}   {ratio_d:.6f}")

    log()

    # Fit both profiles
    d_arr = np.array(d_values, dtype=float)
    L_nn_arr = np.array([L_nn[d] for d in d_values])
    L_nr_arr = np.array([L_nr[d] for d in d_values])

    # Fit phi = A/d + B (linear in 1/d)
    inv_d = 1.0 / d_arr
    for label, L_arr in [("Normal-Normal", L_nn_arr), ("Normal-Reversed", L_nr_arr)]:
        A_mat = np.vstack([inv_d, np.ones_like(inv_d)]).T
        coeffs = np.linalg.lstsq(A_mat, L_arr, rcond=None)[0]
        A_fit, B_fit = coeffs[0], coeffs[1]
        L_pred = A_fit * inv_d + B_fit
        ss_res = np.sum((L_arr - L_pred)**2)
        ss_tot = np.sum((L_arr - np.mean(L_arr))**2)
        R2 = 1 - ss_res / max(ss_tot, 1e-15)
        sign = "ATTRACTIVE" if A_fit > 0 else "REPULSIVE" if A_fit < 0 else "NEUTRAL"
        log(f"  {label:20s}: phi(d) = {A_fit:+.6f}/d + {B_fit:.6f}  R^2={R2:.4f}  [{sign}]")

    log()

    # Overall ratio
    mean_ratio = np.mean(L_nr_arr / np.maximum(L_nn_arr, 1e-10))
    log(f"  Mean L_nr / L_nn = {mean_ratio:.4f}")

    # Determine the force difference
    # Compare gradient profiles
    grad_nn = np.diff(L_nn_arr)
    grad_nr = np.diff(L_nr_arr)
    n_attractive_nn = np.sum(grad_nn < -1e-6)
    n_attractive_nr = np.sum(grad_nr < -1e-6)
    n_repulsive_nr = np.sum(grad_nr > 1e-6)
    log(f"  Normal-Normal: {n_attractive_nn}/{len(grad_nn)} shells attractive")
    log(f"  Normal-Reversed: {n_attractive_nr}/{len(grad_nr)} attractive, {n_repulsive_nr}/{len(grad_nr)} repulsive")

    # Sign of interaction
    grad_nr_mean = np.mean(grad_nr[:6])  # Inner shells (more reliable)
    grad_nn_mean = np.mean(grad_nn[:6])
    if abs(grad_nr_mean) < abs(grad_nn_mean) * 0.3:
        force_type = "NON-INTERACTING"
    elif grad_nr_mean * grad_nn_mean > 0:
        force_type = "ATTRACTIVE (same sign as normal)"
    else:
        force_type = "REPULSIVE (opposite sign)"

    log(f"\n  FORCE TYPE: {force_type}")
    log(f"    Inner-shell gradient (nn): {grad_nn_mean:+.6f}")
    log(f"    Inner-shell gradient (nr): {grad_nr_mean:+.6f}")
    if abs(grad_nn_mean) > 1e-8:
        log(f"    Ratio: {grad_nr_mean/grad_nn_mean:+.4f}")
    log()

    # ==============================================================
    #  TEST D: CLOSE ENCOUNTER (d=1, 500 cycles)
    # ==============================================================
    log("=" * 64)
    log("  TEST D: CLOSE ENCOUNTER (d=1, J=0.2, 500 cycles)")
    log("=" * 64)
    log()

    N_encounter = 500

    # Start both at settled states
    u_n, v_n = u_norm.copy(), v_norm.copy()
    u_r, v_r = u_rev.copy(), v_rev.copy()

    norm_self_coh = []    # |u_n^dag v_n| - normal's self-coherence
    rev_self_coh = []     # |u_r^dag v_r| - reversed's self-coherence
    cross_lock = []       # |u_n^dag u_r| - cross-lock
    cross_lock_v = []     # |v_n^dag v_r|

    for cycle in range(N_encounter):
        for step in range(COXETER_H):
            # Internal dynamics
            u_n, v_n = ouroboros_step_normal(u_n, v_n, step)
            u_r, v_r = ouroboros_step_reversed(u_r, v_r, step)

            # Mutual coupling (strong, d=1)
            # Normal -> Reversed
            u_r, v_r = apply_coupling(u_r, v_r, u_n, v_n, 1.0, J_STRONG)
            # Reversed -> Normal
            u_n, v_n = apply_coupling(u_n, v_n, u_r, v_r, 1.0, J_STRONG)

        # Record observables
        norm_self_coh.append(np.abs(np.vdot(u_n, v_n)))
        rev_self_coh.append(np.abs(np.vdot(u_r, v_r)))
        cross_lock.append(np.abs(np.vdot(u_n, u_r)))
        cross_lock_v.append(np.abs(np.vdot(v_n, v_r)))

    norm_coh = np.array(norm_self_coh)
    rev_coh = np.array(rev_self_coh)
    cross_u = np.array(cross_lock)
    cross_v = np.array(cross_lock_v)

    # Report
    log(f"  Normal self-coherence |u_n^dag v_n|:")
    log(f"    Initial: {norm_coh[0]:.4f}  Final: {norm_coh[-1]:.4f}")
    log(f"    Mean(last 50): {np.mean(norm_coh[-50:]):.4f} +/- {np.std(norm_coh[-50:]):.4f}")
    norm_trend = "STABLE" if np.std(norm_coh[-50:]) < 0.2 else "OSCILLATING"
    if np.mean(norm_coh[-50:]) < 0.05:
        norm_trend = "DECAYED"
    log(f"    Status: {norm_trend}")
    log()

    log(f"  Reversed self-coherence |u_r^dag v_r|:")
    log(f"    Initial: {rev_coh[0]:.4f}  Final: {rev_coh[-1]:.4f}")
    log(f"    Mean(last 50): {np.mean(rev_coh[-50:]):.4f} +/- {np.std(rev_coh[-50:]):.4f}")
    rev_trend = "STABLE" if np.std(rev_coh[-50:]) < 0.2 else "OSCILLATING"
    if np.mean(rev_coh[-50:]) < 0.05:
        rev_trend = "DECAYED"
    log(f"    Status: {rev_trend}")
    log()

    log(f"  Cross-lock |u_n^dag u_r|:")
    log(f"    Initial: {cross_u[0]:.4f}  Final: {cross_u[-1]:.4f}")
    log(f"    Mean(last 50): {np.mean(cross_u[-50:]):.4f} +/- {np.std(cross_u[-50:]):.4f}")
    cross_trend = "GROWING" if cross_u[-1] > cross_u[0] + 0.1 else \
                  "ZERO" if np.mean(cross_u[-50:]) < 0.05 else "OSCILLATING"
    log(f"    Status: {cross_trend}")
    log()

    log(f"  Cross-lock |v_n^dag v_r|:")
    log(f"    Mean(last 50): {np.mean(cross_v[-50:]):.4f} +/- {np.std(cross_v[-50:]):.4f}")
    log()

    # Time evolution (sampled)
    log("  TIME EVOLUTION (every 50 cycles):")
    log(f"  {'cycle':>6s}  {'|un.vn|':>8s}  {'|ur.vr|':>8s}  {'|un.ur|':>8s}  {'|vn.vr|':>8s}")
    for i in range(0, N_encounter, 50):
        log(f"  {i:6d}  {norm_coh[i]:8.4f}  {rev_coh[i]:8.4f}  {cross_u[i]:8.4f}  {cross_v[i]:8.4f}")
    log(f"  {N_encounter-1:6d}  {norm_coh[-1]:8.4f}  {rev_coh[-1]:8.4f}  {cross_u[-1]:8.4f}  {cross_v[-1]:8.4f}")
    log()

    # Classify outcome
    both_alive = (np.mean(norm_coh[-50:]) > 0.05) and (np.mean(rev_coh[-50:]) > 0.05)
    both_dead = (np.mean(norm_coh[-50:]) < 0.05) and (np.mean(rev_coh[-50:]) < 0.05)
    cross_locked = np.mean(cross_u[-50:]) > 0.3

    if both_dead:
        outcome = "ANNIHILATION"
    elif cross_locked:
        outcome = "LOCK"
    elif both_alive:
        outcome = "COEXISTENCE"
    else:
        outcome = "ASYMMETRIC"

    log(f"  OUTCOME: {outcome}")

    if outcome == "LOCK":
        # Combined Berry phase
        gamma_combined_n = berry_phase(u_n, v_n, ouroboros_step_normal)
        gamma_combined_r = berry_phase(u_r, v_r, ouroboros_step_reversed)
        log(f"    Normal Berry phase after contact: {gamma_combined_n:.6f} rad")
        log(f"    Reversed Berry phase after contact: {gamma_combined_r:.6f} rad")
        log(f"    Sum gamma_n + gamma_r = {gamma_combined_n + gamma_combined_r:.6f}")
        log(f"    Is sum ~ 0 (cancellation)? {'YES' if abs(gamma_combined_n + gamma_combined_r) < 0.1 else 'NO'}")
    log()

    # ==============================================================
    #  ANTI-GRAVITY VERDICT
    # ==============================================================
    log("=" * 64)
    log("  ANTI-GRAVITY VERDICT")
    log("=" * 64)
    log()

    log(f"  Reversed chirality is self-sustaining:          {'YES' if rev_self_sustaining else 'NO'}")
    log(f"    gamma_rev = {gamma_rev:.4f} rad")
    log(f"    gamma_norm = {gamma_norm:.4f} rad")
    log(f"    Exact sign reversal: {'YES' if is_exact_reversal else 'NO'}")
    log()

    is_repulsive = "REPULSIVE" in force_type
    is_non_interacting = "NON-INTERACTING" in force_type
    log(f"  Opposite chirality pairs repel:                 {'YES' if is_repulsive else 'NO'}")
    log(f"  Force type: {force_type}")
    log()

    if is_repulsive:
        log(f"  Repulsion strength vs attraction:")
        log(f"    Gradient ratio: {abs(grad_nr_mean/grad_nn_mean):.4f}")
        log(f"    Equal strength: {'YES' if abs(abs(grad_nr_mean) - abs(grad_nn_mean)) < 0.3*abs(grad_nn_mean) else 'NO'}")
    log()

    log(f"  Close encounter outcome: {outcome}")
    log()

    # ==============================================================
    #  INTERPRETATION
    # ==============================================================
    log("=" * 64)
    log("  INTERPRETATION")
    log("=" * 64)
    log()

    if is_exact_reversal:
        log("  The reversed chirality merkabit has gamma_rev = -gamma_norm.")
        log("  This is EXACT CPT conjugation at the torsion level:")
        log("    C (charge) = trit sign flip")
        log("    P (parity) = chirality reversal (u<->v gate swap)")
        log("    T (time)   = Berry phase sign flip")
        log("  All three are accomplished by the SINGLE operation of")
        log("  swapping forward/conjugate gate assignments.")
    else:
        log(f"  gamma_rev/gamma_norm = {ratio:.4f} (not exactly -1)")
        log(f"  The chirality reversal modifies but does not exactly negate")
        log(f"  the Berry phase. The asymmetric gates (P, Cross) do not")
        log(f"  have exact conjugation symmetry in the 4-spinor case.")

    log()

    if is_repulsive:
        log("  OPPOSITE CHIRALITY REPELS: this is geometric anti-gravity.")
        log("  The torsion field of reversed-chirality matter has the")
        log("  opposite gradient sign, pushing normal matter away.")
        log("  This is a candidate mechanism for dark energy.")
    elif is_non_interacting:
        log("  OPPOSITE CHIRALITY IS NON-INTERACTING: the torsion fields")
        log("  decouple when chirality is mismatched. This is consistent")
        log("  with Sim 4 (sedenion dark matter is invisible to octonionic")
        log("  gauge bosons), but now at the chirality level rather than")
        log("  the algebraic sector level.")
    else:
        log("  OPPOSITE CHIRALITY STILL ATTRACTS: gravity does not depend")
        log("  on chirality. The lattice Laplacian (which determines the")
        log("  1/r potential) is chirality-blind. Both chiralities create")
        log("  the same potential because the Laplace equation is parity-")
        log("  invariant. This is consistent with the equivalence principle.")
        log()
        log("  This means: gravity is TRULY UNIVERSAL -- it does not care")
        log("  about the internal winding direction of the torsion structure.")
        log("  Only the PRESENCE of a self-sustaining torsion structure")
        log("  (mass) matters, not its chirality.")

    log()
    elapsed = (datetime.now() - start).total_seconds()
    log(f"  Runtime: {elapsed:.1f} seconds")
    log("=" * 64)

    with open("chirality_reversal_output.txt", 'w') as f:
        f.write('\n'.join(out))
    print(f"\nOutput saved to chirality_reversal_output.txt")


if __name__ == '__main__':
    np.random.seed(42)
    main()
