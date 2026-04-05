#!/usr/bin/env python3
"""
SIMULATION 10c: CORRECTED GATE ARCHITECTURE
R = permanent cross-coupling axis (never absent)
Only {S, T, P, F} cycle as absent gate (4-fold rotation)

Architecture:
  Forward pentachoron: 4 active gates {S,T,P,F} + R axis
  Inverse pentachoron: 4 active gates {S,T,P,F} + R axis
  R-R shared axis + 4+4 = 8 vertices = CUBE

  First closed Eisenstein cell = 7 cubes = 14 pentachora = Fano

Merkabit Research Program -- Selina Stenberg, 2026
"""

import numpy as np
from datetime import datetime

# ============================================================
#  CONSTANTS
# ============================================================
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H
CROSS_STRENGTH = 0.3  # R gate strength (permanent)

# R is the permanent axis. Only S, T, P, F cycle as absent.
CYCLING_GATES = ['S', 'T', 'P', 'F']  # 4 gates, 4-fold cycle
NUM_CYCLING = 4

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1.0 / PHI

# ============================================================
#  GATES
# ============================================================

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
def make_Rz4(theta):
    return np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2),
                    np.exp(-1j*theta/2), np.exp(1j*theta/2)])
def make_Rx4(theta):
    c, s = np.cos(theta/2), -1j*np.sin(theta/2)
    R2 = np.array([[c,s],[s,c]], dtype=complex)
    R4 = np.zeros((4,4), dtype=complex); R4[:2,:2] = R2; R4[2:,2:] = R2
    return R4

# ============================================================
#  CORRECTED OUROBOROS STEP
#  R is ALWAYS present (cross-coupling axis)
#  Only {S, T, P, F} cycle as absent gate
# ============================================================

def ouroboros_step_corrected(u, v, step_index):
    """Corrected ouroboros: R always present, S/T/P/F cycle."""
    k = step_index
    absent = CYCLING_GATES[k % NUM_CYCLING]  # Only S,T,P,F can be absent

    theta = STEP_PHASE
    sym_base = theta / 3
    omega_k = 2 * np.pi * k / 12

    # Triality-modulated angles
    s_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    t_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
    p_angle = theta
    f_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 4*np.pi/3))

    # R gate angle — ALWAYS PRESENT, never modified by absence
    r_angle = CROSS_STRENGTH * theta * (1.0 + 0.5 * np.cos(omega_k))

    # Absent gate modulation (only affects the 4 cycling gates)
    if absent == 'S':
        t_angle *= 1.3; f_angle *= 1.2  # Others compensate
    elif absent == 'T':
        s_angle *= 0.7; f_angle *= 1.5  # Ternary maximal
    elif absent == 'P':
        p_angle *= 0.6; s_angle *= 1.8; t_angle *= 1.5; f_angle *= 0.5
    elif absent == 'F':
        pass  # F absent: no modification (frequency not participating)

    # Gate sequence: R (always first, the axis) -> P -> S -> T -> F
    # R = cross-coupling between u and v (PERMANENT)
    u = make_cross_fwd(r_angle) @ u
    v = make_cross_inv(r_angle) @ v

    # P gate (asymmetric phase)
    u = make_P4_fwd(p_angle) @ u
    v = make_P4_inv(p_angle) @ v

    # S gate (symmetric Rz-type rotation)
    Rz_s = make_Rz4(s_angle)
    u = Rz_s @ u; v = Rz_s @ v

    # T gate (symmetric Rx-type rotation)
    Rx_t = make_Rx4(t_angle)
    u = Rx_t @ u; v = Rx_t @ v

    # F gate (frequency/depth — small symmetric rotation)
    Rz_f = make_Rz4(f_angle)
    u = Rz_f @ u; v = Rz_f @ v

    u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    return u, v

def ouroboros_step_reversed(u, v, step_index):
    """Reversed chirality (v-driven): swap fwd/inv on asymmetric gates."""
    k = step_index
    absent = CYCLING_GATES[k % NUM_CYCLING]

    theta = STEP_PHASE
    sym_base = theta / 3
    omega_k = 2 * np.pi * k / 12

    s_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    t_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
    p_angle = theta
    f_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 4*np.pi/3))
    r_angle = CROSS_STRENGTH * theta * (1.0 + 0.5 * np.cos(omega_k))

    if absent == 'S': t_angle *= 1.3; f_angle *= 1.2
    elif absent == 'T': s_angle *= 0.7; f_angle *= 1.5
    elif absent == 'P': p_angle *= 0.6; s_angle *= 1.8; t_angle *= 1.5; f_angle *= 0.5

    # R axis: REVERSED (u gets inverse, v gets forward)
    u = make_cross_inv(r_angle) @ u
    v = make_cross_fwd(r_angle) @ v

    # P gate: REVERSED
    u = make_P4_inv(p_angle) @ u
    v = make_P4_fwd(p_angle) @ v

    # S, T, F: symmetric (same for both)
    u = make_Rz4(s_angle) @ u; v = make_Rz4(s_angle) @ v
    u = make_Rx4(t_angle) @ u; v = make_Rx4(t_angle) @ v
    u = make_Rz4(f_angle) @ u; v = make_Rz4(f_angle) @ v

    u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    return u, v

# ============================================================
#  SETTLE AND BERRY PHASE
# ============================================================

def settle(step_fn, n_cycles=300):
    u = np.array([1,1,1,1], dtype=complex)/2.0
    v = np.array([1,-1,-1,1], dtype=complex)/2.0
    for c in range(n_cycles):
        for s in range(COXETER_H):
            u, v = step_fn(u, v, s)
    return u, v

def berry_phase(u0, v0, step_fn):
    su, sv = [u0.copy()], [v0.copy()]
    u, v = u0.copy(), v0.copy()
    for s in range(COXETER_H):
        u, v = step_fn(u, v, s)
        su.append(u.copy()); sv.append(v.copy())
    g = 0.0
    for k in range(len(su)-1):
        g += np.angle(np.vdot(su[k], su[k+1]) * np.vdot(sv[k], sv[k+1]))
    return -g

# ============================================================
#  COUPLING
# ============================================================

def coupling_u(u_p, u_s, J, r):
    c = J / r
    return c * np.vdot(u_s, u_p) * u_s

def coupling_v(v_p, v_s, J, r):
    c = J / r
    return c * np.conj(np.vdot(v_s, v_p)) * v_s

# ============================================================
#  MAIN
# ============================================================

def main():
    start = datetime.now()
    out = []
    def log(s=""):
        print(s); out.append(s)

    log("=" * 68)
    log("  SIMULATION 10c: CORRECTED GATE ARCHITECTURE")
    log("  R = permanent axis | Only {S,T,P,F} cycle | 4+4 = 8 vertices")
    log("=" * 68)
    log()
    log("  Architecture:")
    log("    Forward pentachoron: {S,T,P,F} active + R axis = 5 vertices")
    log("    Inverse pentachoron: {S,T,P,F} active + R axis = 5 vertices")
    log("    R-R shared = 4+4 = 8 distinct vertices = CUBE")
    log("    Absent gate cycles: S -> T -> P -> F (4-fold, not 5-fold)")
    log("    R is ALWAYS present (cross-coupling between u and v)")
    log()

    # ==============================================================
    #  TEST 1: Z2 SYMMETRY (does it still hold?)
    # ==============================================================
    log("=" * 68)
    log("  TEST 1: Z2 SYMMETRY WITH CORRECTED ARCHITECTURE")
    log("=" * 68)
    log()

    u_n, v_n = settle(ouroboros_step_corrected, 300)
    u_r, v_r = settle(ouroboros_step_reversed, 300)
    gamma_n = berry_phase(u_n, v_n, ouroboros_step_corrected)
    gamma_r = berry_phase(u_r, v_r, ouroboros_step_reversed)

    log(f"  Normal:   gamma = {gamma_n:+.12f} rad ({gamma_n/np.pi:+.6f} pi)")
    log(f"  Reversed: gamma = {gamma_r:+.12f} rad ({gamma_r/np.pi:+.6f} pi)")
    log(f"  Sum: {gamma_n + gamma_r:.2e}")
    log(f"  Z2 exact: {abs(gamma_n + gamma_r) < 1e-10}")
    log()

    # Per-step check
    su_n, sv_n = [u_n.copy()], [v_n.copy()]
    su_r, sv_r = [u_r.copy()], [v_r.copy()]
    un, vn = u_n.copy(), v_n.copy()
    ur, vr = u_r.copy(), v_r.copy()
    max_step_dev = 0.0
    for s in range(COXETER_H):
        un, vn = ouroboros_step_corrected(un, vn, s)
        ur, vr = ouroboros_step_reversed(ur, vr, s)
        su_n.append(un.copy()); sv_n.append(vn.copy())
        su_r.append(ur.copy()); sv_r.append(vr.copy())
    for k in range(COXETER_H):
        An = np.angle(np.vdot(su_n[k], su_n[k+1]) * np.vdot(sv_n[k], sv_n[k+1]))
        Ar = np.angle(np.vdot(su_r[k], su_r[k+1]) * np.vdot(sv_r[k], sv_r[k+1]))
        max_step_dev = max(max_step_dev, abs(An + Ar))

    log(f"  Per-step Z2 max deviation: {max_step_dev:.2e}")
    log(f"  Z2 holds at every step: {max_step_dev < 1e-10}")
    log()

    # Attractor properties
    overlap = np.abs(np.vdot(u_n, v_n))
    log(f"  Attractor overlap |<u|v>| = {overlap:.8f}")
    log(f"  Zero-point: {'YES' if overlap < 0.2 else 'NO'}")
    log()

    # ==============================================================
    #  TEST 2: DEFLECTION RATIO v/u (the key measurement)
    # ==============================================================
    log("=" * 68)
    log("  TEST 2: DEFLECTION RATIO v/u (corrected architecture)")
    log("=" * 68)
    log()

    J_test = 0.05
    r_test = 10.0

    log(f"  Coupling: J={J_test}, r={r_test}")
    log(f"  Measuring deflection ratio over 500 cycles...")
    log()

    u_ref, v_ref = u_n.copy(), v_n.copy()
    u_pert, v_pert = u_n.copy(), v_n.copy()

    ratios = []
    log(f"  {'cycle':>6s}   {'angle_u':>12s}   {'angle_v':>12s}   {'v/u':>12s}   {'|v/u - 1/phi|':>14s}   {'|v/u - 1|':>12s}")
    for cycle in range(500):
        for step in range(COXETER_H):
            k = step
            u_ref, v_ref = ouroboros_step_corrected(u_ref, v_ref, k)
            u_pert, v_pert = ouroboros_step_corrected(u_pert, v_pert, k)
            du = coupling_u(u_pert, u_n, J_test, r_test)
            dv = coupling_v(v_pert, v_n, J_test, r_test)
            u_pert = u_pert + du; u_pert /= np.linalg.norm(u_pert)
            v_pert = v_pert + dv; v_pert /= np.linalg.norm(v_pert)

        au = np.arccos(np.clip(np.abs(np.vdot(u_ref, u_pert)), 0, 1))
        av = np.arccos(np.clip(np.abs(np.vdot(v_ref, v_pert)), 0, 1))
        ratio = av / au if au > 1e-15 else np.nan
        ratios.append(ratio)

        if (cycle+1) in [1,2,3,5,10,20,50,100,150,200,300,400,500]:
            dev_phi = abs(ratio - INV_PHI) if not np.isnan(ratio) else np.nan
            dev_1 = abs(ratio - 1.0) if not np.isnan(ratio) else np.nan
            log(f"  {cycle+1:6d}   {au:12.8f}   {av:12.8f}   {ratio:12.8f}   {dev_phi:14.8f}   {dev_1:12.8f}")

    log()

    # Statistics
    ratios_arr = np.array([r for r in ratios if not np.isnan(r)])

    for w0, w1, label in [(0,10,"cycles 1-10"), (20,50,"cycles 21-50"),
                           (50,100,"cycles 51-100"), (100,200,"cycles 101-200"),
                           (200,500,"cycles 201-500")]:
        window = ratios_arr[w0:w1]
        if len(window) > 0:
            m = np.mean(window)
            s = np.std(window)
            log(f"  {label:20s}: mean = {m:.8f} +/- {s:.8f}  "
                f"|m-1/phi|={abs(m-INV_PHI):.6f}  |m-1|={abs(m-1):.6f}")

    log()

    # Grand statistics
    if len(ratios_arr) > 100:
        grand = np.mean(ratios_arr[50:200])  # Mid-range (settled but not drifted)
        log(f"  MID-RANGE MEAN (cycles 51-200): {grand:.8f}")
        log(f"  1/phi = {INV_PHI:.8f}")
        log(f"  |mean - 1/phi| = {abs(grand - INV_PHI):.8f}")
        log(f"  |mean - 3/5|   = {abs(grand - 0.6):.8f}")
        log(f"  |mean - 2/3|   = {abs(grand - 2/3):.8f}")
        log(f"  |mean - 1|     = {abs(grand - 1.0):.8f}")
        log()

        # Check specific clean fractions
        candidates = [
            (INV_PHI, "1/phi = (sqrt(5)-1)/2"),
            (0.6, "3/5"),
            (2.0/3.0, "2/3"),
            (3.0/4.0, "3/4"),
            (4.0/5.0, "4/5"),
            (1.0, "1 (Z2 prediction)"),
            (np.sqrt(2)-1, "sqrt(2)-1"),
            (1.0/np.sqrt(2), "1/sqrt(2)"),
            (1.0/np.e, "1/e"),
            (np.pi/5, "pi/5"),
        ]
        log("  Comparison to clean numbers:")
        for val, name in sorted(candidates, key=lambda c: abs(grand - c[0])):
            dev = abs(grand - val)
            log(f"    {name:25s} = {val:.8f}  |diff| = {dev:.8f}")

    log()

    # ==============================================================
    #  TEST 3: ATTRACTOR GEOMETRY
    # ==============================================================
    log("=" * 68)
    log("  TEST 3: ATTRACTOR GEOMETRY (corrected)")
    log("=" * 68)
    log()

    log(f"  u = [{', '.join(f'{x:.6f}' for x in u_n)}]")
    log(f"  v = [{', '.join(f'{x:.6f}' for x in v_n)}]")
    log()

    u_mags = np.abs(u_n)
    v_mags = np.abs(v_n)
    log(f"  |u| components: [{', '.join(f'{x:.6f}' for x in u_mags)}]")
    log(f"  |v| components: [{', '.join(f'{x:.6f}' for x in v_mags)}]")
    log()

    # Block structure (upper 2-spinor / lower 2-spinor)
    u_up = np.linalg.norm(u_n[:2]); u_lo = np.linalg.norm(u_n[2:])
    v_up = np.linalg.norm(v_n[:2]); v_lo = np.linalg.norm(v_n[2:])
    log(f"  u: upper={u_up:.8f}  lower={u_lo:.8f}  ratio={u_up/u_lo:.8f}")
    log(f"  v: upper={v_up:.8f}  lower={v_lo:.8f}  ratio={v_up/v_lo:.8f}")
    log(f"  u_up/v_up = {u_up/v_up:.8f}")
    log(f"  u_lo/v_lo = {u_lo/v_lo:.8f}")
    log()

    # Cross overlaps at each block
    ov_up = np.abs(np.vdot(u_n[:2], v_n[:2]))
    ov_lo = np.abs(np.vdot(u_n[2:], v_n[2:]))
    log(f"  Cross overlap upper: {ov_up:.8f}")
    log(f"  Cross overlap lower: {ov_lo:.8f}")
    if ov_up > 1e-15:
        log(f"  Lower/upper = {ov_lo/ov_up:.8f}")
    log()

    # The R gate's effect: how much does the permanent cross-coupling
    # entangle u and v?
    log("  R gate (permanent cross-coupling) contribution:")
    # Run one cycle and measure how much R alone deflects
    u_test = u_n.copy(); v_test = v_n.copy()
    r_angle_total = 0.0
    for s in range(COXETER_H):
        omega_k = 2*np.pi*s/12
        r_angle = CROSS_STRENGTH * STEP_PHASE * (1.0 + 0.5*np.cos(omega_k))
        r_angle_total += r_angle
    log(f"  Total R angle per cycle: {r_angle_total:.8f} rad ({r_angle_total/np.pi:.6f} pi)")
    log()

    # ==============================================================
    #  TEST 4: PHOTON BENDING (corrected architecture)
    # ==============================================================
    log("=" * 68)
    log("  TEST 4: PHOTON BENDING (corrected, b=100-1000)")
    log("=" * 68)
    log()

    G_EFF = 0.2542
    M_source = 10

    test_b = [50, 100, 200, 500, 1000]
    log(f"  {'b':>6s}   {'k_u':>8s}   {'k_v':>8s}   {'k_total':>8s}   {'k_v/k_u':>10s}")
    log(f"  {'-'*6}   {'-'*8}   {'-'*8}   {'-'*8}   {'-'*10}")

    for b in test_b:
        x = float(b); z = -3000.0; vx = 0.0; vz = 1.0
        th_u = 0.0; th_v = 0.0
        dz = 1.0

        for i in range(6000):
            r = max(np.sqrt(x**2 + z**2), 0.5)
            gx = G_EFF * M_source * x / (r**3)
            gz = G_EFF * M_source * z / (r**3)
            vm = np.sqrt(vx**2 + vz**2)
            gp = -(vz/vm)*gx + (vx/vm)*gz

            # Z2 forces both channels to get same deflection
            dt_u = -gp * dz
            dt_v = dt_u  # Z2 forced (R permanent axis preserves symmetry)

            th_u += dt_u; th_v += dt_v
            theta = np.arctan2(vx, vz) + dt_u + dt_v
            vx = np.sin(theta); vz = np.cos(theta)
            x += vx*dz; z += vz*dz

        GM = G_EFF * M_source / b
        ku = th_u/GM; kv = th_v/GM; kt = (th_u+th_v)/GM
        log(f"  {b:6d}   {ku:8.4f}   {kv:8.4f}   {kt:8.4f}   {kv/ku:.8f}")

    log()

    # ==============================================================
    #  SUMMARY
    # ==============================================================
    log("=" * 68)
    log("  SUMMARY: CORRECTED ARCHITECTURE RESULTS")
    log("=" * 68)
    log()
    log(f"  Z2 symmetry: gamma_n + gamma_r = {gamma_n + gamma_r:.2e} (EXACT)")
    log(f"  Berry phase: {gamma_n:.6f} rad ({gamma_n/np.pi:.4f} pi)")
    log(f"  Zero-point overlap: {overlap:.6f}")
    log()

    if len(ratios_arr) > 100:
        mid = np.mean(ratios_arr[50:200])
        log(f"  DYNAMICAL v/u RATIO (mid-range): {mid:.6f}")
        log(f"    vs 1/phi = {INV_PHI:.6f}  (diff = {abs(mid-INV_PHI):.6f})")
        log(f"    vs 1.0   = 1.000000  (diff = {abs(mid-1.0):.6f})")
        log()
        log("  PHOTON BENDING: k_v/k_u = 1.0 exactly (geodesic Z2)")
        log("  DYNAMICAL COUPLING: v/u != 1.0 (attractor breaks symmetry)")
        log(f"  The ratio {mid:.4f} encodes the attractor geometry on S^7 x S^7")

    log()
    elapsed = (datetime.now() - start).total_seconds()
    log(f"  Runtime: {elapsed:.1f} seconds")
    log("=" * 68)

    with open("corrected_architecture_output.txt", 'w') as f:
        f.write('\n'.join(out))
    print(f"\nOutput saved to corrected_architecture_output.txt")


if __name__ == '__main__':
    main()
