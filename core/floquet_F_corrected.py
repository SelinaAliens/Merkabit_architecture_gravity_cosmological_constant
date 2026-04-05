#!/usr/bin/env python3
"""
FLOQUET F DERIVATION — CORRECTED GATE ARCHITECTURE
R = permanent cross-coupling axis, {S,T,P,F} cycle (4-fold)

Original: F = 0.696778, -ln(F) = 0.3613, alpha^-1 = 137 - ln(F)/10
Corrected: F = ?, does the identity still hold?

Merkabit Research Program — Selina Stenberg, 2026
"""

import numpy as np
from datetime import datetime

try:
    import mpmath
    mpmath.mp.dps = 120
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

# ============================================================
#  CONSTANTS
# ============================================================
T_FLOQUET = 12
STEP_PHASE = 2 * np.pi / T_FLOQUET  # pi/6
ALPHA_INV_CODATA = 137.035999084
CYCLING_GATES = ['S', 'T', 'P', 'F']  # R excluded — permanent axis
NUM_CYCLING = 4
CROSS_STRENGTH = 0.3  # R gate coupling strength

# ============================================================
#  CORRECTED GATE ANGLES
#  R is never absent. Only {S,T,P,F} cycle.
# ============================================================

def get_gate_angles(k):
    """Return all gate angles for step k. R is always present."""
    absent = CYCLING_GATES[k % NUM_CYCLING]

    theta = STEP_PHASE  # pi/6
    sym_base = theta / 3  # pi/18
    omega_k = 2 * np.pi * k / T_FLOQUET

    # Base angles with triality modulation
    s_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    t_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
    p_angle = theta
    f_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 4*np.pi/3))
    r_angle = CROSS_STRENGTH * theta * (1.0 + 0.5 * np.cos(omega_k))

    # Absent gate modulation (only S,T,P,F can be absent)
    if absent == 'S':
        t_angle *= 1.3; f_angle *= 1.2
    elif absent == 'T':
        s_angle *= 0.7; f_angle *= 1.5
    elif absent == 'P':
        p_angle *= 0.6; s_angle *= 1.8; t_angle *= 1.5; f_angle *= 0.5
    elif absent == 'F':
        pass  # No modification

    return r_angle, p_angle, s_angle, t_angle, f_angle

# ============================================================
#  STEP UNITARY (C^2 x C^2 = C^4 representation)
#  Order: R(cross,asymm) -> P(phase,asymm) -> S(Rz,symm) -> T(Rx,symm) -> F(Rz,symm)
# ============================================================

def step_unitary(k):
    """Build 4x4 unitary for step k with corrected gate sequence."""
    r_ang, p_ang, s_ang, t_ang, f_ang = get_gate_angles(k)

    # R gate: cross-coupling (asymmetric, always present)
    # In C^2 x C^2 basis: forward cross on first factor, inverse on second
    # cross_fwd(theta) on qubit 1, cross_inv(theta) on qubit 2
    # For 2-spinor representation: R acts as exp(i*r*sigma_y/2) x exp(-i*r*sigma_y/2)
    cr = np.cos(r_ang/2); sr = np.sin(r_ang/2)
    R_fwd = np.array([[cr, -sr], [sr, cr]], dtype=complex)
    R_inv = np.array([[cr, sr], [-sr, cr]], dtype=complex)
    U_R = np.kron(R_fwd, R_inv)

    # P gate: asymmetric phase
    Pf = np.diag([np.exp(1j*p_ang/2), np.exp(-1j*p_ang/2)])
    Pi = np.diag([np.exp(-1j*p_ang/2), np.exp(1j*p_ang/2)])
    U_P = np.kron(Pf, Pi)

    # S gate: symmetric Rz
    Rz_s = np.diag([np.exp(-1j*s_ang/2), np.exp(1j*s_ang/2)])
    U_S = np.kron(Rz_s, Rz_s)

    # T gate: symmetric Rx
    ct = np.cos(t_ang/2); st = -1j*np.sin(t_ang/2)
    Rx_t = np.array([[ct, st], [st, ct]], dtype=complex)
    U_T = np.kron(Rx_t, Rx_t)

    # F gate: symmetric Rz
    Rz_f = np.diag([np.exp(-1j*f_ang/2), np.exp(1j*f_ang/2)])
    U_F = np.kron(Rz_f, Rz_f)

    # Sequence: F -> T -> S -> P -> R (right-to-left matrix multiplication)
    return U_F @ U_T @ U_S @ U_P @ U_R

# ============================================================
#  OLD (INCORRECT) STEP UNITARY FOR COMPARISON
# ============================================================

OLD_GATES = ['S', 'R', 'T', 'F', 'P']

def get_gate_angles_old(k):
    absent = k % 5
    gate_label = OLD_GATES[absent]
    p_angle = STEP_PHASE
    sym_base = STEP_PHASE / 3
    omega_k = 2 * np.pi * k / T_FLOQUET
    rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))
    if gate_label == 'S': rz_angle *= 0.4; rx_angle *= 1.3
    elif gate_label == 'R': rx_angle *= 0.4; rz_angle *= 1.3
    elif gate_label == 'T': rx_angle *= 0.7; rz_angle *= 0.7
    elif gate_label == 'P': p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5
    return p_angle, rz_angle, rx_angle

def step_unitary_old(k):
    p_angle, rz_angle, rx_angle = get_gate_angles_old(k)
    Pf = np.diag([np.exp(1j*p_angle/2), np.exp(-1j*p_angle/2)])
    Pi = np.diag([np.exp(-1j*p_angle/2), np.exp(1j*p_angle/2)])
    U_P = np.kron(Pf, Pi)
    Rz = np.diag([np.exp(-1j*rz_angle/2), np.exp(1j*rz_angle/2)])
    U_Rz = np.kron(Rz, Rz)
    c = np.cos(rx_angle/2); s = -1j*np.sin(rx_angle/2)
    Rx = np.array([[c,s],[s,c]], dtype=complex)
    U_Rx = np.kron(Rx, Rx)
    return U_Rx @ U_Rz @ U_P

# ============================================================
#  FLOQUET UNITARY AND F COMPUTATION
# ============================================================

def floquet_unitary(step_fn):
    U = np.eye(4, dtype=complex)
    for k in range(T_FLOQUET):
        U = step_fn(k) @ U
    return U

def compute_F(step_fn):
    """F = |<psi_0|U_F|psi_0>|^2 where psi_0 = |0> x |1> = [0,1,0,0]"""
    U_F = floquet_unitary(step_fn)
    psi0 = np.array([0, 1, 0, 0], dtype=complex)
    amp = np.vdot(psi0, U_F @ psi0)
    return abs(amp)**2, U_F

# ============================================================
#  HIGH-PRECISION MPMATH COMPUTATION
# ============================================================

def compute_F_mpmath(corrected=True):
    if not HAS_MPMATH:
        return None, None

    pi = mpmath.pi
    T = 12

    def mp_step(k):
        theta = 2*pi/12
        sb = theta/3
        ok = 2*pi*k/12

        if corrected:
            absent = ['S','T','P','F'][int(k) % 4]
            s_a = sb*(1+mpmath.mpf('0.5')*mpmath.cos(ok))
            t_a = sb*(1+mpmath.mpf('0.5')*mpmath.cos(ok+2*pi/3))
            p_a = theta
            f_a = sb*(1+mpmath.mpf('0.5')*mpmath.cos(ok+4*pi/3))
            r_a = mpmath.mpf('0.3')*theta*(1+mpmath.mpf('0.5')*mpmath.cos(ok))

            if absent=='S': t_a*=mpmath.mpf('1.3'); f_a*=mpmath.mpf('1.2')
            elif absent=='T': s_a*=mpmath.mpf('0.7'); f_a*=mpmath.mpf('1.5')
            elif absent=='P': p_a*=mpmath.mpf('0.6'); s_a*=mpmath.mpf('1.8'); t_a*=mpmath.mpf('1.5'); f_a*=mpmath.mpf('0.5')
        else:
            gates = ['S','R','T','F','P']
            absent = gates[int(k) % 5]
            rx = sb*(1+mpmath.mpf('0.5')*mpmath.cos(ok))
            rz = sb*(1+mpmath.mpf('0.5')*mpmath.cos(ok+2*pi/3))
            p_a = theta
            if absent=='S': rz*=mpmath.mpf('0.4'); rx*=mpmath.mpf('1.3')
            elif absent=='R': rx*=mpmath.mpf('0.4'); rz*=mpmath.mpf('1.3')
            elif absent=='T': rx*=mpmath.mpf('0.7'); rz*=mpmath.mpf('0.7')
            elif absent=='P': p_a*=mpmath.mpf('0.6'); rx*=mpmath.mpf('1.8'); rz*=mpmath.mpf('1.5')
            # Old: P -> Rz -> Rx, no R gate, no S/F separation
            s_a = rz; t_a = rx; f_a = mpmath.mpf(0); r_a = mpmath.mpf(0)

        # Build 4x4 unitary
        U = mpmath.matrix(4, 4)
        for i in range(4): U[i,i] = 1

        def apply_kron(Af, Ai):
            """Apply kron(Af, Ai) to U (in-place via temp)."""
            nonlocal U
            T = mpmath.matrix(4, 4)
            for i in range(2):
                for j in range(2):
                    for m in range(2):
                        for n in range(2):
                            for p in range(4):
                                T[2*i+m, p] += Af[i,j] * Ai[m,n] * U[2*j+n, p]
            return T

        if corrected:
            # R gate (permanent cross-coupling)
            cr = mpmath.cos(r_a/2); sr = mpmath.sin(r_a/2)
            Rf = mpmath.matrix([[cr, -sr], [sr, cr]])
            Ri = mpmath.matrix([[cr, sr], [-sr, cr]])
            U = apply_kron(Rf, Ri)

        # P gate
        ep = mpmath.exp(1j*p_a/2); em = mpmath.exp(-1j*p_a/2)
        Pf = mpmath.matrix([[ep, 0], [0, em]])
        Pi = mpmath.matrix([[em, 0], [0, ep]])
        U = apply_kron(Pf, Pi)

        # S gate (Rz)
        es = mpmath.exp(-1j*s_a/2); ems = mpmath.exp(1j*s_a/2)
        Rs = mpmath.matrix([[es, 0], [0, ems]])
        U = apply_kron(Rs, Rs)

        # T gate (Rx)
        ct = mpmath.cos(t_a/2); st = -1j*mpmath.sin(t_a/2)
        Rt = mpmath.matrix([[ct, st], [st, ct]])
        U = apply_kron(Rt, Rt)

        if corrected and abs(f_a) > 1e-20:
            # F gate (Rz)
            ef = mpmath.exp(-1j*f_a/2); emf = mpmath.exp(1j*f_a/2)
            Rf = mpmath.matrix([[ef, 0], [0, emf]])
            U = apply_kron(Rf, Rf)

        return U

    # Accumulate full Floquet unitary
    UF = mpmath.matrix(4, 4)
    for i in range(4): UF[i,i] = 1
    for k in range(T):
        Uk = mp_step(k)
        Tnew = mpmath.matrix(4, 4)
        for i in range(4):
            for j in range(4):
                for m in range(4):
                    Tnew[i,j] += Uk[i,m] * UF[m,j]
        UF = Tnew

    amp = UF[1,1]
    F = abs(amp)**2
    neg_ln_F = -mpmath.log(F)
    return F, neg_ln_F

# ============================================================
#  MAIN
# ============================================================

def main():
    start = datetime.now()
    out = []
    def log(s=""):
        print(s); out.append(s)

    log("=" * 68)
    log("  FLOQUET F DERIVATION — CORRECTED GATE ARCHITECTURE")
    log("  R = permanent axis | {S,T,P,F} cycle | 4-fold")
    log("=" * 68)
    log()

    # ==============================================================
    #  BASELINE: Old vs Corrected F
    # ==============================================================
    log("=" * 68)
    log("  BASELINE COMPARISON: OLD vs CORRECTED")
    log("=" * 68)
    log()

    F_old, UF_old = compute_F(step_unitary_old)
    F_new, UF_new = compute_F(step_unitary)

    neg_ln_old = -np.log(F_old)
    neg_ln_new = -np.log(F_new)

    alpha_inv_old = 137 - np.log(F_old) / 10
    alpha_inv_new = 137 - np.log(F_new) / 10

    log(f"  OLD (5-fold, R cycles):")
    log(f"    F = {F_old:.15f}")
    log(f"    -ln(F) = {neg_ln_old:.15f}")
    log(f"    alpha^-1 = 137 - ln(F)/10 = {alpha_inv_old:.12f}")
    log()

    log(f"  CORRECTED (4-fold, R permanent):")
    log(f"    F = {F_new:.15f}")
    log(f"    -ln(F) = {neg_ln_new:.15f}")
    log(f"    alpha^-1 = 137 - ln(F)/10 = {alpha_inv_new:.12f}")
    log()

    log(f"  CODATA alpha^-1 = {ALPHA_INV_CODATA:.12f}")
    log(f"  Old deviation:  |alpha_old - CODATA| = {abs(alpha_inv_old - ALPHA_INV_CODATA):.6e}")
    log(f"  New deviation:  |alpha_new - CODATA| = {abs(alpha_inv_new - ALPHA_INV_CODATA):.6e}")
    log()

    closer = "CORRECTED" if abs(alpha_inv_new - ALPHA_INV_CODATA) < abs(alpha_inv_old - ALPHA_INV_CODATA) else "OLD"
    log(f"  Closer to CODATA: {closer}")
    log()

    # ==============================================================
    #  SERIES EXPANSION: -ln(F) = c2/V^2 + c3/V^3 + ...
    # ==============================================================
    log("=" * 68)
    log("  SERIES EXPANSION CHECK")
    log("=" * 68)
    log()

    V = 10.0
    log(f"  -ln(F_new) = {neg_ln_new:.15f}")
    log(f"  36/V^2 = {36/V**2:.15f}")
    log(f"  Residual after c2: {neg_ln_new - 36/V**2:.15f}")
    log()

    # Check if c2 = 36 still holds
    # c2 = -ln(F) * V^2 in the limit
    c2_eff = neg_ln_new * V**2
    log(f"  Effective c2 = -ln(F) * V^2 = {c2_eff:.6f}")
    log(f"  Expected c2 = 36 (E6 positive roots)")
    log(f"  |c2 - 36| = {abs(c2_eff - 36):.6f}")
    log()

    # Residual analysis
    r1 = neg_ln_new - 36/V**2
    log(f"  After subtracting 36/V^2: residual = {r1:.15f}")
    c3_eff = r1 * V**3
    log(f"  Effective c3 = residual * V^3 = {c3_eff:.6f}")
    log(f"  Original c3 = 58/45 = {58/45:.6f}")
    log(f"  |c3_eff - 58/45| = {abs(c3_eff - 58/45):.6f}")
    log()

    # ==============================================================
    #  HIGH-PRECISION (mpmath)
    # ==============================================================
    if HAS_MPMATH:
        log("=" * 68)
        log("  HIGH-PRECISION COMPUTATION (mpmath, 120 digits)")
        log("=" * 68)
        log()

        log("  Computing corrected F at high precision...")
        F_hp, nln_hp = compute_F_mpmath(corrected=True)
        if F_hp is not None:
            log(f"  F_corrected (hp) = {mpmath.nstr(F_hp, 30)}")
            log(f"  -ln(F) (hp) = {mpmath.nstr(nln_hp, 30)}")
            alpha_hp = 137 - nln_hp / 10
            log(f"  alpha^-1 (hp) = {mpmath.nstr(alpha_hp, 20)}")
            log(f"  CODATA = {ALPHA_INV_CODATA}")
            dev_hp = abs(float(alpha_hp) - ALPHA_INV_CODATA)
            log(f"  |deviation| = {dev_hp:.6e}")
            log()

            # c2 from high precision
            c2_hp = float(nln_hp) * 100
            log(f"  c2 (hp) = {c2_hp:.10f}")
            log(f"  |c2 - 36| = {abs(c2_hp - 36):.10f}")

        log()
        log("  Computing OLD F at high precision for comparison...")
        F_hp_old, nln_hp_old = compute_F_mpmath(corrected=False)
        if F_hp_old is not None:
            log(f"  F_old (hp) = {mpmath.nstr(F_hp_old, 30)}")
            log(f"  -ln(F_old) (hp) = {mpmath.nstr(nln_hp_old, 30)}")
            alpha_hp_old = 137 - nln_hp_old / 10
            log(f"  alpha^-1_old (hp) = {mpmath.nstr(alpha_hp_old, 20)}")
            dev_hp_old = abs(float(alpha_hp_old) - ALPHA_INV_CODATA)
            log(f"  |deviation_old| = {dev_hp_old:.6e}")
        log()
    else:
        log("  (mpmath not available, skipping high-precision)")
        log()

    # ==============================================================
    #  FLOQUET EIGENVALUES
    # ==============================================================
    log("=" * 68)
    log("  FLOQUET EIGENVALUE COMPARISON")
    log("=" * 68)
    log()

    eig_old = np.linalg.eigvals(UF_old)
    eig_new = np.linalg.eigvals(UF_new)

    log(f"  OLD Floquet eigenvalues:")
    for i, e in enumerate(sorted(eig_old, key=lambda x: np.angle(x))):
        log(f"    lambda_{i} = {e:.8f}  |lambda| = {abs(e):.10f}  phase = {np.angle(e)/np.pi:.6f} pi")

    log(f"\n  CORRECTED Floquet eigenvalues:")
    for i, e in enumerate(sorted(eig_new, key=lambda x: np.angle(x))):
        log(f"    lambda_{i} = {e:.8f}  |lambda| = {abs(e):.10f}  phase = {np.angle(e)/np.pi:.6f} pi")

    log()

    # Quasi-energies
    log("  Quasi-energies (epsilon = -i*ln(lambda)/T):")
    for label, eigs in [("OLD", eig_old), ("CORRECTED", eig_new)]:
        phases = np.sort(np.angle(eigs))
        qe = phases / T_FLOQUET
        log(f"  {label}: {[f'{q:.6f}' for q in qe]}")
    log()

    # ==============================================================
    #  EVEN SYMMETRY CHECK
    # ==============================================================
    log("=" * 68)
    log("  EVEN SYMMETRY: F(s) = F(-s)?")
    log("=" * 68)
    log()

    # Scale the gate angles by factor s and compute F(s)
    test_s = [0.5, 0.8, 1.0, 1.2, 1.5]
    log(f"  {'s':>6s}   {'F(s)':>15s}   {'F(-s)':>15s}   {'|diff|':>12s}")
    for s in test_s:
        def scaled_step(k, scale=s):
            r_a, p_a, s_a, t_a, f_a = get_gate_angles(k)
            r_a *= scale; p_a *= scale; s_a *= scale; t_a *= scale; f_a *= scale
            cr = np.cos(r_a/2); sr = np.sin(r_a/2)
            Rf = np.array([[cr,-sr],[sr,cr]], dtype=complex)
            Ri = np.array([[cr,sr],[-sr,cr]], dtype=complex)
            U = np.kron(Rf, Ri)
            Pf = np.diag([np.exp(1j*p_a/2), np.exp(-1j*p_a/2)])
            Pi = np.diag([np.exp(-1j*p_a/2), np.exp(1j*p_a/2)])
            U = np.kron(Pf, Pi) @ U
            Rs = np.diag([np.exp(-1j*s_a/2), np.exp(1j*s_a/2)])
            U = np.kron(Rs, Rs) @ U
            ct = np.cos(t_a/2); st = -1j*np.sin(t_a/2)
            Rt = np.array([[ct,st],[st,ct]], dtype=complex)
            U = np.kron(Rt, Rt) @ U
            Rfz = np.diag([np.exp(-1j*f_a/2), np.exp(1j*f_a/2)])
            U = np.kron(Rfz, Rfz) @ U
            return U
        def scaled_step_neg(k, scale=-s):
            return scaled_step(k, scale)

        Fs = compute_F(lambda k: scaled_step(k, s))[0]
        Fns = compute_F(lambda k: scaled_step(k, -s))[0]
        log(f"  {s:6.2f}   {Fs:15.12f}   {Fns:15.12f}   {abs(Fs-Fns):12.2e}")

    log()

    # ==============================================================
    #  SUMMARY
    # ==============================================================
    log("=" * 68)
    log("  SUMMARY")
    log("=" * 68)
    log()
    log(f"  OLD:       F = {F_old:.12f}   alpha^-1 = {alpha_inv_old:.9f}")
    log(f"  CORRECTED: F = {F_new:.12f}   alpha^-1 = {alpha_inv_new:.9f}")
    log(f"  CODATA:                         alpha^-1 = {ALPHA_INV_CODATA:.9f}")
    log()
    log(f"  OLD     |delta| = {abs(alpha_inv_old - ALPHA_INV_CODATA):.6e}")
    log(f"  CORRECT |delta| = {abs(alpha_inv_new - ALPHA_INV_CODATA):.6e}")
    log()

    if abs(alpha_inv_new - ALPHA_INV_CODATA) < 0.1:
        log("  alpha^-1 identity PRESERVED under corrected architecture")
        if abs(alpha_inv_new - ALPHA_INV_CODATA) < abs(alpha_inv_old - ALPHA_INV_CODATA):
            log("  CORRECTED architecture gives BETTER match to CODATA")
        else:
            log("  OLD architecture was closer (but used incorrect gates)")
    else:
        log(f"  alpha^-1 identity BROKEN: deviation = {abs(alpha_inv_new - ALPHA_INV_CODATA):.4f}")
        log("  The series expansion coefficients need re-derivation")

    log()
    elapsed = (datetime.now() - start).total_seconds()
    log(f"  Runtime: {elapsed:.1f} seconds")
    log("=" * 68)

    with open("floquet_F_corrected_output.txt", 'w') as f:
        f.write('\n'.join(out))
    print(f"\nOutput saved to floquet_F_corrected_output.txt")

if __name__ == '__main__':
    main()
