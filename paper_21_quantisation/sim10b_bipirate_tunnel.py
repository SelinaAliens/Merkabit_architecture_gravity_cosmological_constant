#!/usr/bin/env python3
"""
SIMULATION 10b: THE BIPIRATE TUNNEL RATIO
Is the v/u deflection ratio exactly 1/phi (golden ratio)?

The torsion tunnel is BIPIRATE: it takes unequally from Mer and Ka.
The ratio is 1/phi because the tunnel is a Fibonacci spiral on S^7.

If v/u = 1/phi:
  Total enhancement = 1 + 1/phi = phi = (1+sqrt(5))/2
  k_dynamical = 2 * phi = 1 + sqrt(5) ~ 3.236

For PHOTONS (geodesic, not dynamical): Z2 forces ratio = 1 -> k = 4
For MASSIVE BODIES (dynamical, through attractor): ratio = 1/phi -> k = 2*phi
"""

import numpy as np
from datetime import datetime

PHI = (1 + np.sqrt(5)) / 2  # 1.6180339887...
INV_PHI = 1.0 / PHI          # 0.6180339887...

# ============================================================
#  GATES (same as all previous sims)
# ============================================================
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H
NUM_GATES = 5
OUROBOROS_GATES = ['S', 'R', 'T', 'F', 'P']
CROSS_STRENGTH = 0.3

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

def compute_gate_params(step_index):
    k = step_index; absent = k % NUM_GATES
    p = STEP_PHASE; sb = STEP_PHASE/3; ok = 2*np.pi*k/12
    rx = sb*(1+0.5*np.cos(ok))
    rz = sb*(1+0.5*np.cos(ok+2*np.pi/3))
    cr = CROSS_STRENGTH*STEP_PHASE*(1+0.5*np.cos(ok+4*np.pi/3))
    label = OUROBOROS_GATES[absent]
    if label=='S': rz*=0.4; rx*=1.3; cr*=1.2
    elif label=='R': rx*=0.4; rz*=1.3; cr*=0.8
    elif label=='T': rx*=0.7; rz*=0.7; cr*=1.5
    elif label=='P': p*=0.6; rx*=1.8; rz*=1.5; cr*=0.5
    return p, rx, rz, cr

def step_normal(u, v, k):
    p, rx, rz, cr = compute_gate_params(k)
    u = make_P4_fwd(p)@u; v = make_P4_inv(p)@v
    u = make_cross_fwd(cr)@u; v = make_cross_inv(cr)@v
    Rz = make_Rz4(rz); Rx = make_Rx4(rx)
    u = Rx@Rz@u; v = Rx@Rz@v
    u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    return u, v

def settle(n_cycles=300):
    u = np.array([1,1,1,1], dtype=complex)/2.0
    v = np.array([1,-1,-1,1], dtype=complex)/2.0
    for c in range(n_cycles):
        for s in range(COXETER_H):
            u, v = step_normal(u, v, s)
    return u, v

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
    log("  SIMULATION 10b: THE BIPIRATE TUNNEL RATIO")
    log("  Is v/u deflection = 1/phi = 0.6180339887...?")
    log("=" * 68)
    log()
    log(f"  phi = (1+sqrt(5))/2 = {PHI:.10f}")
    log(f"  1/phi = phi - 1    = {INV_PHI:.10f}")
    log()

    # Settle source
    u_s, v_s = settle(300)
    log(f"  Settled source: |<u|v>| = {np.abs(np.vdot(u_s, v_s)):.6f}")
    log()

    # ==============================================================
    #  HIGH-PRECISION RATIO MEASUREMENT
    # ==============================================================
    log("=" * 68)
    log("  HIGH-PRECISION RATIO MEASUREMENT")
    log("=" * 68)
    log()

    # Sweep coupling parameters to check if ratio is universal
    J_values = [0.01, 0.02, 0.05, 0.1]
    r_values = [3.0, 5.0, 10.0, 20.0]
    N_cycles_list = [50, 100, 200, 500]

    log("  Test 1: Ratio vs coupling strength J (r=10, 200 cycles)")
    log(f"  {'J':>8s}   {'angle_u':>12s}   {'angle_v':>12s}   {'v/u ratio':>12s}   {'|ratio-1/phi|':>14s}")
    log(f"  {'-'*8}   {'-'*12}   {'-'*12}   {'-'*12}   {'-'*14}")

    r_fixed = 10.0
    n_fixed = 200
    for J in J_values:
        u_ref, v_ref = u_s.copy(), v_s.copy()
        u_pert, v_pert = u_s.copy(), v_s.copy()

        for step in range(n_fixed * COXETER_H):
            k = step % COXETER_H
            u_ref, v_ref = step_normal(u_ref, v_ref, k)
            u_pert, v_pert = step_normal(u_pert, v_pert, k)
            du = coupling_u(u_pert, u_s, J, r_fixed)
            dv = coupling_v(v_pert, v_s, J, r_fixed)
            u_pert = u_pert + du; u_pert /= np.linalg.norm(u_pert)
            v_pert = v_pert + dv; v_pert /= np.linalg.norm(v_pert)

        au = np.arccos(np.clip(np.abs(np.vdot(u_ref, u_pert)), 0, 1))
        av = np.arccos(np.clip(np.abs(np.vdot(v_ref, v_pert)), 0, 1))
        ratio = av / au if au > 1e-15 else np.nan
        dev = abs(ratio - INV_PHI)
        log(f"  {J:8.3f}   {au:12.8f}   {av:12.8f}   {ratio:12.8f}   {dev:14.8f}")

    log()

    log("  Test 2: Ratio vs distance r (J=0.05, 200 cycles)")
    log(f"  {'r':>8s}   {'angle_u':>12s}   {'angle_v':>12s}   {'v/u ratio':>12s}   {'|ratio-1/phi|':>14s}")
    log(f"  {'-'*8}   {'-'*12}   {'-'*12}   {'-'*12}   {'-'*14}")

    J_fixed = 0.05
    for r in r_values:
        u_ref, v_ref = u_s.copy(), v_s.copy()
        u_pert, v_pert = u_s.copy(), v_s.copy()

        for step in range(n_fixed * COXETER_H):
            k = step % COXETER_H
            u_ref, v_ref = step_normal(u_ref, v_ref, k)
            u_pert, v_pert = step_normal(u_pert, v_pert, k)
            du = coupling_u(u_pert, u_s, J_fixed, r)
            dv = coupling_v(v_pert, v_s, J_fixed, r)
            u_pert = u_pert + du; u_pert /= np.linalg.norm(u_pert)
            v_pert = v_pert + dv; v_pert /= np.linalg.norm(v_pert)

        au = np.arccos(np.clip(np.abs(np.vdot(u_ref, u_pert)), 0, 1))
        av = np.arccos(np.clip(np.abs(np.vdot(v_ref, v_pert)), 0, 1))
        ratio = av / au if au > 1e-15 else np.nan
        dev = abs(ratio - INV_PHI)
        log(f"  {r:8.1f}   {au:12.8f}   {av:12.8f}   {ratio:12.8f}   {dev:14.8f}")

    log()

    log("  Test 3: Ratio vs settle time (J=0.05, r=10)")
    log(f"  {'cycles':>8s}   {'angle_u':>12s}   {'angle_v':>12s}   {'v/u ratio':>12s}   {'|ratio-1/phi|':>14s}")
    log(f"  {'-'*8}   {'-'*12}   {'-'*12}   {'-'*12}   {'-'*14}")

    for nc in N_cycles_list:
        u_ref, v_ref = u_s.copy(), v_s.copy()
        u_pert, v_pert = u_s.copy(), v_s.copy()

        for step in range(nc * COXETER_H):
            k = step % COXETER_H
            u_ref, v_ref = step_normal(u_ref, v_ref, k)
            u_pert, v_pert = step_normal(u_pert, v_pert, k)
            du = coupling_u(u_pert, u_s, J_fixed, r_fixed)
            dv = coupling_v(v_pert, v_s, J_fixed, r_fixed)
            u_pert = u_pert + du; u_pert /= np.linalg.norm(u_pert)
            v_pert = v_pert + dv; v_pert /= np.linalg.norm(v_pert)

        au = np.arccos(np.clip(np.abs(np.vdot(u_ref, u_pert)), 0, 1))
        av = np.arccos(np.clip(np.abs(np.vdot(v_ref, v_pert)), 0, 1))
        ratio = av / au if au > 1e-15 else np.nan
        dev = abs(ratio - INV_PHI)
        log(f"  {nc:8d}   {au:12.8f}   {av:12.8f}   {ratio:12.8f}   {dev:14.8f}")

    log()

    # ==============================================================
    #  CYCLE-BY-CYCLE RATIO EVOLUTION
    # ==============================================================
    log("=" * 68)
    log("  RATIO EVOLUTION (cycle by cycle, J=0.05, r=10)")
    log("=" * 68)
    log()

    u_ref, v_ref = u_s.copy(), v_s.copy()
    u_pert, v_pert = u_s.copy(), v_s.copy()

    cycle_ratios = []
    log(f"  {'cycle':>6s}   {'v/u ratio':>12s}   {'|r - 1/phi|':>14s}")
    for cycle in range(500):
        for step in range(COXETER_H):
            k = step % COXETER_H
            u_ref, v_ref = step_normal(u_ref, v_ref, k)
            u_pert, v_pert = step_normal(u_pert, v_pert, k)
            du = coupling_u(u_pert, u_s, J_fixed, r_fixed)
            dv = coupling_v(v_pert, v_s, J_fixed, r_fixed)
            u_pert = u_pert + du; u_pert /= np.linalg.norm(u_pert)
            v_pert = v_pert + dv; v_pert /= np.linalg.norm(v_pert)

        au = np.arccos(np.clip(np.abs(np.vdot(u_ref, u_pert)), 0, 1))
        av = np.arccos(np.clip(np.abs(np.vdot(v_ref, v_pert)), 0, 1))
        ratio = av / au if au > 1e-15 else np.nan
        cycle_ratios.append(ratio)

        if (cycle + 1) % 50 == 0 or cycle < 10:
            dev = abs(ratio - INV_PHI) if not np.isnan(ratio) else np.nan
            log(f"  {cycle+1:6d}   {ratio:12.8f}   {dev:14.8f}")

    log()

    ratios_arr = np.array([r for r in cycle_ratios if not np.isnan(r)])

    # Statistics over different windows
    log("  CONVERGENCE STATISTICS:")
    for window_start, window_end, label in [(0, 50, "early (1-50)"),
                                             (100, 200, "mid (101-200)"),
                                             (300, 500, "late (301-500)")]:
        window = ratios_arr[window_start:window_end]
        if len(window) > 0:
            m = np.mean(window)
            s = np.std(window)
            dev = abs(m - INV_PHI)
            log(f"    {label:20s}: mean = {m:.8f} +/- {s:.8f}  |mean - 1/phi| = {dev:.8f}")

    log()

    # ==============================================================
    #  THE STRUCTURAL ORIGIN
    # ==============================================================
    log("=" * 68)
    log("  THE STRUCTURAL ORIGIN: WHY 1/phi?")
    log("=" * 68)
    log()

    # Measure the asymmetric gate contributions
    log("  The asymmetric gates (P and Cross) treat u and v differently.")
    log("  P gate: u gets exp(+i*phi/2), v gets exp(-i*phi/2)")
    log("  Cross gate: u gets forward rotation, v gets inverse rotation")
    log()
    log("  The P gate phase per step:")
    p_phases_u = []
    p_phases_v = []
    cross_angles_u = []
    cross_angles_v = []
    for k in range(COXETER_H):
        p, rx, rz, cr = compute_gate_params(k)
        # P gate phases
        p_phases_u.append(p / 2)   # u gets +p/2
        p_phases_v.append(-p / 2)  # v gets -p/2
        # Cross angles
        cross_angles_u.append(cr / 2)   # forward
        cross_angles_v.append(-cr / 2)  # inverse

    total_p_u = sum(p_phases_u)
    total_p_v = sum(p_phases_v)
    total_cr_u = sum(cross_angles_u)
    total_cr_v = sum(cross_angles_v)

    log(f"  Total P phase (u): {total_p_u:.8f}")
    log(f"  Total P phase (v): {total_p_v:.8f}")
    log(f"  Ratio |v/u| (P):   {abs(total_p_v/total_p_u):.8f}")
    log()
    log(f"  Total Cross angle (u): {total_cr_u:.8f}")
    log(f"  Total Cross angle (v): {total_cr_v:.8f}")
    log(f"  Ratio |v/u| (Cross):   {abs(total_cr_v/total_cr_u):.8f}")
    log()

    # The combined asymmetric action per cycle
    # u gets: P_fwd * Cross_fwd = accumulates phase total_p_u + total_cr_u
    # v gets: P_inv * Cross_inv = accumulates phase total_p_v + total_cr_v
    total_asym_u = abs(total_p_u + total_cr_u)
    total_asym_v = abs(total_p_v + total_cr_v)
    asym_ratio = total_asym_v / total_asym_u if total_asym_u > 1e-15 else np.nan

    log(f"  Total asymmetric action (u): {total_asym_u:.8f}")
    log(f"  Total asymmetric action (v): {total_asym_v:.8f}")
    log(f"  Ratio |v/u| (total asym):    {asym_ratio:.8f}")
    log(f"  1/phi =                       {INV_PHI:.8f}")
    log(f"  |ratio - 1/phi| =             {abs(asym_ratio - INV_PHI):.8f}")
    log()

    # Per-step asymmetry analysis
    log("  Per-step asymmetric contribution:")
    log(f"  {'step':>4s}   {'asym_u':>10s}   {'asym_v':>10s}   {'v/u':>10s}")
    step_ratios = []
    for k in range(COXETER_H):
        au_k = abs(p_phases_u[k] + cross_angles_u[k])
        av_k = abs(p_phases_v[k] + cross_angles_v[k])
        r_k = av_k / au_k if au_k > 1e-15 else np.nan
        step_ratios.append(r_k)
        log(f"  {k:4d}   {au_k:10.6f}   {av_k:10.6f}   {r_k:10.6f}")

    log()

    # ==============================================================
    #  ATTRACTOR GEOMETRY ON S^7
    # ==============================================================
    log("=" * 68)
    log("  ATTRACTOR GEOMETRY ON S^7")
    log("=" * 68)
    log()

    log(f"  u_settled = [{', '.join(f'{x:.6f}' for x in u_s)}]")
    log(f"  v_settled = [{', '.join(f'{x:.6f}' for x in v_s)}]")
    log()

    # Component magnitudes
    u_mags = np.abs(u_s)
    v_mags = np.abs(v_s)
    log(f"  |u| components: [{', '.join(f'{x:.6f}' for x in u_mags)}]")
    log(f"  |v| components: [{', '.join(f'{x:.6f}' for x in v_mags)}]")
    log()

    # The u and v spinors settle to DIFFERENT points on S^7
    # But related by the Z2 symmetry
    # Check: does the component distribution differ by 1/phi?
    u_upper = np.linalg.norm(u_s[:2])
    u_lower = np.linalg.norm(u_s[2:])
    v_upper = np.linalg.norm(v_s[:2])
    v_lower = np.linalg.norm(v_s[2:])

    log(f"  u: upper block |u[0:2]| = {u_upper:.8f}")
    log(f"  u: lower block |u[2:4]| = {u_lower:.8f}")
    log(f"  u: upper/lower = {u_upper/u_lower:.8f}")
    log()
    log(f"  v: upper block |v[0:2]| = {v_upper:.8f}")
    log(f"  v: lower block |v[2:4]| = {v_lower:.8f}")
    log(f"  v: upper/lower = {v_upper/v_lower:.8f}")
    log()

    # Cross overlaps
    uv_upper = np.abs(np.vdot(u_s[:2], v_s[:2]))
    uv_lower = np.abs(np.vdot(u_s[2:], v_s[2:]))
    log(f"  Cross overlap upper: |<u[0:2]|v[0:2]>| = {uv_upper:.8f}")
    log(f"  Cross overlap lower: |<u[2:4]|v[2:4]>| = {uv_lower:.8f}")
    if uv_upper > 1e-15:
        log(f"  Lower/upper cross overlap = {uv_lower/uv_upper:.8f}")
    log()

    # ==============================================================
    #  VERDICT
    # ==============================================================
    log("=" * 68)
    log("  VERDICT")
    log("=" * 68)
    log()

    # Collect all measured ratios
    all_late_ratios = ratios_arr[200:] if len(ratios_arr) > 200 else ratios_arr
    if len(all_late_ratios) > 0:
        grand_mean = np.mean(all_late_ratios)
        grand_std = np.std(all_late_ratios)
        dev_from_phi = abs(grand_mean - INV_PHI)
        dev_from_1 = abs(grand_mean - 1.0)

        log(f"  Grand mean v/u ratio (late time): {grand_mean:.8f} +/- {grand_std:.8f}")
        log(f"  1/phi = {INV_PHI:.8f}")
        log(f"  |mean - 1/phi| = {dev_from_phi:.8f}")
        log(f"  |mean - 1.0|   = {dev_from_1:.8f}")
        log()

        closer_to_phi = dev_from_phi < dev_from_1
        log(f"  Closer to 1/phi than to 1.0: {closer_to_phi}")
        log()

        if dev_from_phi < 0.05:
            log(f"  RESULT: v/u ratio = 1/phi to within {dev_from_phi:.4f}")
            log()
            log("  The torsion tunnel is BIPIRATE:")
            log("    - The u-spinor (Mer, forward) deflects by delta")
            log("    - The v-spinor (Ka, backward) deflects by delta/phi")
            log("    - Total = delta * (1 + 1/phi) = delta * phi")
            log()
            log(f"  For photon bending (dynamical):")
            log(f"    k = 2 * phi = {2*PHI:.6f}")
            log()
            log(f"  For photon bending (geodesic, Z2-forced):")
            log(f"    k = 2 * 2 = 4")
            log()
            log("  The distinction: geodesic = instantaneous (both channels")
            log("  see the same gradient simultaneously). Dynamical = through")
            log("  the attractor (channels evolve differently due to bipirate")
            log("  asymmetry of P and Cross gates).")
            log()
            log(f"  phi = (1+sqrt(5))/2 appears because the tesseract's")
            log(f"  asymmetric gates (P, Cross) create a Fibonacci spiral")
            log(f"  on S^7 x S^7. The attractor sits at the golden ratio")
            log(f"  partition of the torsion between forward and backward.")
        else:
            log(f"  RESULT: v/u ratio = {grand_mean:.6f}")
            log(f"  Not converged to 1/phi. Deviation = {dev_from_phi:.4f}")
            log(f"  May need more cycles or different parameters.")

    log()
    elapsed = (datetime.now() - start).total_seconds()
    log(f"  Runtime: {elapsed:.1f} seconds")
    log("=" * 68)

    with open("bipirate_tunnel_output.txt", 'w') as f:
        f.write('\n'.join(out))
    print(f"\nOutput saved to bipirate_tunnel_output.txt")


if __name__ == '__main__':
    main()
