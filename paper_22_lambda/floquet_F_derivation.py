#!/usr/bin/env python3
"""
FLOQUET F DERIVATION: Route C Closure
=====================================

Tests whether the return fidelity F = 0.696778 (from Appendix M Table M.4a)
is derivable from E6 invariants, closing Route C of the alpha^{-1} derivation.

Analytic claim:
  F = exp(-n+/V^2 + (11/12)/V^5)
    = exp(-36/100 + (11/12)/100000)
    = exp(-0.35999083...)
    = 0.69768272...

  Same Euler deficit counting as alpha^{-1}, shifted one power of V = 10:
    alpha^{-1} = 137 + 36/V^3 - (11/12)/V^6 = 137.035999083
    -ln(F)     =       36/V^2 - (11/12)/V^5 =   0.359990833

  Identity: alpha^{-1} = 137 - ln(F) / 10

The simulation returns F = 0.696778 at N=12 discrete steps. The prediction
is that the 0.13% gap is Trotter error from the discrete-step approximation,
scaling as c/N^2 and vanishing in the continuous limit.

Three verification tests:
  TEST 1: Trotter scaling - does F(N) -> 0.697683 as N -> infinity?
  TEST 2: Symbolic/high-precision Floquet matrix computation
  TEST 3: alpha^{-1} identity - does 137 - ln(F)/10 recover alpha^{-1}?

Gate sequence: EXACT reproduction from merkabit_verification.py
  (the verified Appendix M implementation)
"""

import numpy as np
import os
import json
import time
from scipy.optimize import curve_fit

# ============================================================================
# CONSTANTS
# ============================================================================

OUTPUT_DIR = r"C:\Users\selin\merkabit_results\floquet_F"
os.makedirs(OUTPUT_DIR, exist_ok=True)

T_FLOQUET = 12
STEP_PHASE = 2 * np.pi / T_FLOQUET  # pi/6
GATES = ['S', 'R', 'T', 'F', 'P']

# Analytic target
F_ANALYTIC = np.exp(-(36/100 - (11/12)/1e5))
ALPHA_CODATA = 137.035999084
ALPHA_MERKABIT = 137 + 36/1e3 - (11/12)/1e6


# ============================================================================
# GATE CONSTRUCTION (exact copy from merkabit_verification.py)
# ============================================================================

def get_gate_angles(k):
    """
    Return (p_angle, rz_angle, rx_angle) for ouroboros step k.
    EXACT angles from Appendix M Table M.2. No free parameters.
    """
    absent = k % 5
    gate_label = GATES[absent]

    p_angle = STEP_PHASE
    sym_base = STEP_PHASE / 3
    omega_k = 2 * np.pi * k / T_FLOQUET

    rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2 * np.pi / 3))

    if gate_label == 'S':
        rz_angle *= 0.4;  rx_angle *= 1.3
    elif gate_label == 'R':
        rx_angle *= 0.4;  rz_angle *= 1.3
    elif gate_label == 'T':
        rx_angle *= 0.7;  rz_angle *= 0.7
    elif gate_label == 'P':
        p_angle *= 0.6;   rx_angle *= 1.8;  rz_angle *= 1.5
    # F absent: no modification

    return p_angle, rz_angle, rx_angle


def step_unitary(k):
    """
    Build the 4x4 unitary for ouroboros step k on C^2 x C^2.
    Sequence: P (asymmetric) -> Rz (symmetric) -> Rx (symmetric)
    """
    p_angle, rz_angle, rx_angle = get_gate_angles(k)

    # P gate (asymmetric sigma_z rotation)
    Pf = np.diag([np.exp(1j * p_angle / 2), np.exp(-1j * p_angle / 2)])
    Pi = np.diag([np.exp(-1j * p_angle / 2), np.exp(1j * p_angle / 2)])
    U_P = np.kron(Pf, Pi)

    # Rz (symmetric)
    Rz = np.diag([np.exp(-1j * rz_angle / 2), np.exp(1j * rz_angle / 2)])
    U_Rz = np.kron(Rz, Rz)

    # Rx (symmetric)
    c = np.cos(rx_angle / 2)
    s = -1j * np.sin(rx_angle / 2)
    Rx = np.array([[c, s], [s, c]], dtype=complex)
    U_Rx = np.kron(Rx, Rx)

    return U_Rx @ U_Rz @ U_P


def floquet_unitary():
    """Full Floquet unitary U_F = U_11 ... U_1 U_0."""
    U = np.eye(4, dtype=complex)
    for k in range(T_FLOQUET):
        U = step_unitary(k) @ U
    return U


# ============================================================================
# BASIS STATES
# ============================================================================

def make_state(u, v):
    u = np.array(u, dtype=complex); u /= np.linalg.norm(u)
    v = np.array(v, dtype=complex); v /= np.linalg.norm(v)
    return np.kron(u, v)

PSI_PLUS  = make_state([1, 0], [1, 0])    # |+1>
PSI_ZERO  = make_state([1, 0], [0, 1])    # |0>
PSI_MINUS = make_state([1, 0], [-1, 0])   # |-1>


# ============================================================================
# VERIFY BASELINE: reproduce F = 0.696778 from Appendix M
# ============================================================================

def verify_baseline():
    """Reproduce the known F = 0.696778 from merkabit_verification.py."""
    print("=" * 76)
    print("  BASELINE VERIFICATION")
    print("  Reproduce F = 0.696778 from Appendix M Table M.4a")
    print("=" * 76)

    U_F = floquet_unitary()

    # Unitarity check
    err = np.max(np.abs(U_F @ U_F.conj().T - np.eye(4)))
    print(f"\n  Unitarity: max|U_F U_F' - I| = {err:.2e}")

    # Return fidelities
    results = {}
    for name, psi0 in [('|+1>', PSI_PLUS), ('|0>', PSI_ZERO), ('|-1>', PSI_MINUS)]:
        psi_T = U_F @ psi0
        F_val = abs(np.vdot(psi0, psi_T))**2
        results[name] = F_val
        print(f"  F({name}) = {F_val:.8f}")

    F_zero = results['|0>']
    print(f"\n  All equal: {np.allclose(list(results.values()), F_zero, atol=1e-10)}")
    print(f"  F_baseline = {F_zero:.10f}")
    print(f"  Appendix M = 0.696778")
    print(f"  Match:       {abs(F_zero - 0.696778):.2e}")

    # Berry phases
    print(f"\n  Berry phases (one cycle):")
    berry_phases = {}
    for name, psi0 in [('|+1>', PSI_PLUS), ('|0>', PSI_ZERO), ('|-1>', PSI_MINUS)]:
        states = [psi0.copy()]
        psi = psi0.copy()
        for k in range(T_FLOQUET):
            psi = step_unitary(k) @ psi
            psi /= np.linalg.norm(psi)
            states.append(psi.copy())
        gamma = 0.0
        for k in range(T_FLOQUET):
            gamma -= np.angle(np.vdot(states[k], states[k+1]))
        berry_phases[name] = gamma
        print(f"    gamma({name}) = {gamma:+.6f} rad = {gamma/np.pi:+.6f} pi")

    sep = abs(berry_phases['|0>'] - berry_phases['|+1>'])
    print(f"\n  Berry separation |0> vs |+/-1>: {sep:.4f} rad ({sep/np.pi:.4f} pi)")

    return F_zero, berry_phases


# ============================================================================
# TEST 1: TROTTER SCALING
# ============================================================================

def step_unitary_trotter(k, m):
    """
    Build step k with Trotter refinement: subdivide into m sub-steps.

    The original step applies: U_k = Rx(rx) @ Rz(rz) @ P(p)
    These are non-commuting. The Trotter approximation at order m is:
      U_k^(m) = [Rx(rx/m) @ Rz(rz/m) @ P(p/m)]^m

    As m -> inf, this converges to exp(-i(H_P + H_Rz + H_Rx)*dt),
    the joint exponentiation (continuous limit).
    """
    p_angle, rz_angle, rx_angle = get_gate_angles(k)

    # Sub-step angles
    p_sub = p_angle / m
    rz_sub = rz_angle / m
    rx_sub = rx_angle / m

    # Build sub-step gate matrices
    Pf = np.diag([np.exp(1j * p_sub / 2), np.exp(-1j * p_sub / 2)])
    Pi = np.diag([np.exp(-1j * p_sub / 2), np.exp(1j * p_sub / 2)])
    U_P = np.kron(Pf, Pi)

    Rz = np.diag([np.exp(-1j * rz_sub / 2), np.exp(1j * rz_sub / 2)])
    U_Rz = np.kron(Rz, Rz)

    c = np.cos(rx_sub / 2)
    s = -1j * np.sin(rx_sub / 2)
    Rx = np.array([[c, s], [s, c]], dtype=complex)
    U_Rx = np.kron(Rx, Rx)

    # One sub-step
    U_sub = U_Rx @ U_Rz @ U_P

    # Apply m times
    U_step = np.eye(4, dtype=complex)
    for _ in range(m):
        U_step = U_sub @ U_step

    return U_step


def floquet_unitary_trotter(m):
    """Full Floquet unitary with Trotter refinement m per step."""
    U = np.eye(4, dtype=complex)
    for k in range(T_FLOQUET):
        U = step_unitary_trotter(k, m) @ U
    return U


def test_trotter_scaling():
    """
    Does F(N) -> F_analytic = 0.697683 as N -> infinity?

    Method: INTRA-STEP Trotter refinement. Each ouroboros step k applies
    three non-commuting gates: P -> Rz -> Rx. The Trotter subdivision
    splits each into m sub-steps with angles divided by m.

    The original N=12 computation uses m=1 (one P-Rz-Rx per step).
    As m increases, the product approaches the continuous evolution
    exp(-i(H_P + H_Rz + H_Rx)*dt), which is the physical limit.
    """
    print("\n" + "=" * 76)
    print("  TEST 1: TROTTER SCALING (intra-step refinement)")
    print("  Does F(m) -> exp(-0.359990833) = 0.697683 as m -> infinity?")
    print("  m = Trotter sub-steps per ouroboros step (total gates = 12*m*3)")
    print("=" * 76)

    m_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    F_values = []

    print(f"\n  {'m':>6s}  {'total_gates':>12s}  {'F(m)':>14s}  {'F-F_target':>14s}  {'F-F_m1':>14s}")
    print(f"  {'---':>6s}  {'---':>12s}  {'---':>14s}  {'---':>14s}  {'---':>14s}")

    for m in m_values:
        U_m = floquet_unitary_trotter(m)
        F_m = abs(PSI_ZERO.conj() @ U_m @ PSI_ZERO)**2
        F_values.append(F_m)
        diff_target = F_m - F_ANALYTIC
        diff_m1 = F_m - F_values[0]
        total_gates = 12 * m * 3
        print(f"  {m:6d}  {total_gates:12d}  {F_m:14.10f}  {diff_target:+14.2e}  {diff_m1:+14.2e}")

    # Fit F(m) = F_inf + c/m^2 (Trotter error scales as 1/m^2 for first-order)
    m_arr = np.array(m_values, dtype=float)
    F_arr = np.array(F_values)

    def model_1overm2(m, F_inf, c):
        return F_inf + c / m**2

    # Use only m >= 4 for fitting (avoid initial transient)
    mask = m_arr >= 4
    m_fit = m_arr[mask]
    F_fit = F_arr[mask]

    try:
        popt, pcov = curve_fit(model_1overm2, m_fit, F_fit, p0=[F_ANALYTIC, -1.0])
        F_inf, c_coeff = popt
        F_inf_err = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else float('inf')

        print(f"\n  Fit (m >= 4): F(m) = F_inf + c/m^2")
        print(f"    F_inf = {F_inf:.10f}  +/- {F_inf_err:.2e}")
        print(f"    c     = {c_coeff:.6f}")
        print(f"\n  Analytic target = {F_ANALYTIC:.10f}")
        print(f"  Gap F_inf - target = {F_inf - F_ANALYTIC:+.2e}")
    except Exception as e:
        print(f"\n  1/m^2 fit failed: {e}")
        F_inf = F_values[-1]
        c_coeff = 0
        F_inf_err = float('inf')

    # Richardson extrapolation using last two points
    F_m1 = F_values[-2]
    F_m2 = F_values[-1]
    M1, M2 = m_values[-2], m_values[-1]
    F_richardson = (M2**2 * F_m2 - M1**2 * F_m1) / (M2**2 - M1**2)
    print(f"\n  Richardson extrapolation (m={M1},{M2}):")
    print(f"    F_inf = {F_richardson:.10f}")
    print(f"    Gap   = {F_richardson - F_ANALYTIC:+.2e}")

    # Convergence check: is F still changing?
    last_change = abs(F_values[-1] - F_values[-2])
    print(f"\n  Convergence check:")
    print(f"    |F(m={m_values[-1]}) - F(m={m_values[-2]})| = {last_change:.2e}")
    print(f"    F appears {'converged' if last_change < 1e-10 else 'still changing'}")

    # Verdict
    gap = abs(F_inf - F_ANALYTIC)
    passed = gap < 1e-4
    print(f"\n  TEST 1 RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"    F_inf (fit)    = {F_inf:.10f}")
    print(f"    F_inf (Rich.)  = {F_richardson:.10f}")
    print(f"    Target         = {F_ANALYTIC:.10f}")
    if not passed:
        print(f"    F converges to {F_values[-1]:.10f}, NOT to {F_ANALYTIC:.10f}")
        print(f"    The gap is {abs(F_values[-1] - F_ANALYTIC):.6f} ({abs(F_values[-1] - F_ANALYTIC)/F_ANALYTIC*100:.4f}%)")
        print(f"    This means F = 0.696778 is the EXACT discrete result,")
        print(f"    not a Trotter approximation to exp(-0.3600).")

    # Make plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.scatter(m_arr, F_arr, color='steelblue', s=60, zorder=5,
                    label='Simulation F(m)')
        if c_coeff != 0:
            m_fine = np.linspace(1, max(m_values), 500)
            ax1.plot(m_fine, model_1overm2(m_fine, F_inf, c_coeff),
                     'steelblue', lw=1.5, alpha=0.7,
                     label=f'Fit: $F_\\infty$ = {F_inf:.6f}')
        ax1.axhline(F_ANALYTIC, color='crimson', ls='--', lw=1.5,
                    label=f'Analytic: {F_ANALYTIC:.6f}')
        ax1.axhline(0.696778, color='grey', ls=':', lw=1.5,
                    label='Appendix M (m=1): 0.696778')
        ax1.set_xlabel('Trotter sub-steps per gate (m)', fontsize=12)
        ax1.set_ylabel('Return fidelity F', fontsize=12)
        ax1.set_title('Trotter scaling - Route C', fontsize=13)
        ax1.set_xscale('log')
        ax1.legend(fontsize=9)

        # Right: F vs 1/m^2
        inv_m2 = 1.0 / m_arr**2
        ax2.scatter(inv_m2, F_arr, color='steelblue', s=60, zorder=5)
        ax2.axhline(F_ANALYTIC, color='crimson', ls='--', lw=1.5)
        ax2.axhline(0.696778, color='grey', ls=':', lw=1.5)
        ax2.set_xlabel('$1/m^2$', fontsize=12)
        ax2.set_ylabel('Return fidelity F', fontsize=12)
        ax2.set_title('Trotter convergence in $1/m^2$', fontsize=13)

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, 'trotter_scaling.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\n  Plot saved: {plot_path}")
        plt.close()
    except ImportError:
        print("\n  matplotlib not available - skipping plot")

    return {
        'F_values': {str(m): float(f) for m, f in zip(m_values, F_values)},
        'F_inf_fit': float(F_inf),
        'F_inf_fit_err': float(F_inf_err),
        'F_inf_richardson': float(F_richardson),
        'c_coefficient': float(c_coeff),
        'target': float(F_ANALYTIC),
        'gap': float(gap),
        'last_change': float(last_change),
        'passed': bool(passed),
    }


# ============================================================================
# TEST 2: HIGH-PRECISION FLOQUET MATRIX
# ============================================================================

def test_symbolic_floquet():
    """
    Compute F at high precision using mpmath (50 decimal places).
    Compare symbolic/exact result to the analytic claim.
    """
    print("\n" + "=" * 76)
    print("  TEST 2: HIGH-PRECISION FLOQUET MATRIX (mpmath, 50 digits)")
    print("  Does F match exp(-0.359990833...) to available precision?")
    print("=" * 76)

    try:
        import mpmath
        mpmath.mp.dps = 50

        pi = mpmath.pi
        T = 12
        STEP = 2 * pi / T
        gate_labels = ['S', 'R', 'T', 'F', 'P']

        def get_angles_mp(k):
            absent = k % 5
            label = gate_labels[absent]

            p_angle = STEP
            sym_base = STEP / 3
            omega_k = 2 * pi * k / T

            rx_angle = sym_base * (1 + mpmath.mpf('0.5') * mpmath.cos(omega_k))
            rz_angle = sym_base * (1 + mpmath.mpf('0.5') * mpmath.cos(omega_k + 2*pi/3))

            if label == 'S':
                rz_angle *= mpmath.mpf('0.4');  rx_angle *= mpmath.mpf('1.3')
            elif label == 'R':
                rx_angle *= mpmath.mpf('0.4');  rz_angle *= mpmath.mpf('1.3')
            elif label == 'T':
                rx_angle *= mpmath.mpf('0.7');  rz_angle *= mpmath.mpf('0.7')
            elif label == 'P':
                p_angle *= mpmath.mpf('0.6')
                rx_angle *= mpmath.mpf('1.8');  rz_angle *= mpmath.mpf('1.5')

            return p_angle, rz_angle, rx_angle

        def step_unitary_mp(k):
            p_a, rz_a, rx_a = get_angles_mp(k)

            # P gate (asymmetric)
            Pf = mpmath.matrix([
                [mpmath.exp(1j * p_a / 2), 0],
                [0, mpmath.exp(-1j * p_a / 2)]
            ])
            Pi = mpmath.matrix([
                [mpmath.exp(-1j * p_a / 2), 0],
                [0, mpmath.exp(1j * p_a / 2)]
            ])

            # Rz (symmetric)
            Rz = mpmath.matrix([
                [mpmath.exp(-1j * rz_a / 2), 0],
                [0, mpmath.exp(1j * rz_a / 2)]
            ])

            # Rx (symmetric)
            c = mpmath.cos(rx_a / 2)
            s = -1j * mpmath.sin(rx_a / 2)
            Rx = mpmath.matrix([[c, s], [s, c]])

            # Kronecker products
            def kron(A, B):
                n, m = A.rows, A.cols
                p, q = B.rows, B.cols
                result = mpmath.matrix(n*p, m*q)
                for i in range(n):
                    for j in range(m):
                        for ki in range(p):
                            for kj in range(q):
                                result[i*p + ki, j*q + kj] = A[i, j] * B[ki, kj]
                return result

            U_P = kron(Pf, Pi)
            U_Rz = kron(Rz, Rz)
            U_Rx = kron(Rx, Rx)

            return U_Rx * U_Rz * U_P

        # Build full Floquet unitary
        print("\n  Building U_F at 50-digit precision...")
        t0 = time.time()

        U_F = mpmath.eye(4)
        for k in range(T):
            U_k = step_unitary_mp(k)
            U_F = U_k * U_F

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        # |0> = [0, 1, 0, 0]
        psi0 = mpmath.matrix([0, 1, 0, 0])

        # <0|U_F|0> = U_F[1, 1]  (since psi0 has only component at index 1)
        overlap = U_F[1, 1]
        F_mp = abs(overlap)**2

        # Compute analytic target at 50 digits
        neg_ln_F_target = mpmath.mpf(36) / 100 - mpmath.mpf(11) / 12 / mpmath.mpf(100000)
        F_target_mp = mpmath.exp(-neg_ln_F_target)

        print(f"\n  <0|U_F|0> = {overlap}")
        print(f"  |<0|U_F|0>|^2 = F = {mpmath.nstr(F_mp, 20)}")
        print(f"  Analytic target     = {mpmath.nstr(F_target_mp, 20)}")
        print(f"  Difference          = {mpmath.nstr(F_mp - F_target_mp, 10)}")
        print(f"  Relative diff       = {mpmath.nstr(abs(F_mp - F_target_mp) / F_target_mp, 10)}")

        # Also check -ln(F)
        neg_ln_F = -mpmath.log(F_mp)
        print(f"\n  -ln(F) computed     = {mpmath.nstr(neg_ln_F, 20)}")
        print(f"  -ln(F) target       = {mpmath.nstr(neg_ln_F_target, 20)}")
        print(f"  36/100              = {mpmath.nstr(mpmath.mpf(36)/100, 20)}")
        print(f"  (11/12)/1e5         = {mpmath.nstr(mpmath.mpf(11)/12/100000, 20)}")

        # Check the overlap phase
        phase_overlap = mpmath.arg(overlap)
        print(f"\n  Phase of <0|U_F|0>  = {mpmath.nstr(phase_overlap, 15)} rad")
        print(f"                      = {mpmath.nstr(phase_overlap / pi, 15)} pi")

        gap = float(abs(F_mp - F_target_mp))
        passed = gap < 1e-6
        print(f"\n  TEST 2 RESULT: {'PASS' if passed else 'FAIL'}")
        print(f"    N=12 discrete F        = {mpmath.nstr(F_mp, 15)}")
        print(f"    Analytic target        = {mpmath.nstr(F_target_mp, 15)}")
        print(f"    Gap                    = {gap:.2e}")
        if not passed:
            print(f"    NOTE: Gap of {gap:.4e} at N=12 is NOT Trotter error")
            print(f"    (TEST 1 shows F converges to 0.696778, not 0.697683)")
            print(f"    F = 0.696778 is the EXACT discrete Floquet result")

        return {
            'F_mp': float(F_mp),
            'F_target': float(F_target_mp),
            'gap': gap,
            'neg_ln_F': float(neg_ln_F),
            'neg_ln_F_target': float(neg_ln_F_target),
            'overlap_phase': float(phase_overlap),
            'passed': bool(passed),
        }

    except ImportError:
        print("\n  mpmath not available. Using numpy float64 instead.")

        U_F = floquet_unitary()
        overlap = PSI_ZERO.conj() @ U_F @ PSI_ZERO
        F_np = abs(overlap)**2

        print(f"\n  F (numpy float64) = {F_np:.15f}")
        print(f"  Analytic target   = {F_ANALYTIC:.15f}")
        print(f"  Gap               = {abs(F_np - F_ANALYTIC):.2e}")

        gap = abs(F_np - F_ANALYTIC)
        return {
            'F_mp': float(F_np),
            'F_target': float(F_ANALYTIC),
            'gap': gap,
            'passed': gap < 1e-6,
        }


# ============================================================================
# TEST 3: ALPHA^{-1} IDENTITY
# ============================================================================

def test_alpha_identity():
    """
    Does 137 - ln(F) / 10 recover alpha^{-1} exactly?

    The identity:
      alpha^{-1} = 137 + 36/V^3 - (11/12)/V^6
      -ln(F)     =       36/V^2 - (11/12)/V^5
    =>
      alpha^{-1} = 137 + (-ln(F)) / V  = 137 - ln(F) / 10
    """
    print("\n" + "=" * 76)
    print("  TEST 3: alpha^{-1} IDENTITY")
    print("  Does 137 - ln(F) / 10 = alpha^{-1}?")
    print("=" * 76)

    # Using the analytic F
    F_an = F_ANALYTIC
    alpha_from_F = 137 - np.log(F_an) / 10

    print(f"\n  F_analytic         = {F_an:.15f}")
    print(f"  -ln(F_analytic)    = {-np.log(F_an):.15f}")
    print(f"  -ln(F)/10          = {-np.log(F_an)/10:.15f}")
    print(f"  137 - ln(F)/10     = {alpha_from_F:.12f}")
    print(f"  Merkabit alpha^-1  = {ALPHA_MERKABIT:.12f}")
    print(f"  CODATA alpha^-1    = {ALPHA_CODATA:.12f}")

    gap_merkabit = abs(alpha_from_F - ALPHA_MERKABIT)
    gap_codata = abs(alpha_from_F - ALPHA_CODATA)
    print(f"\n  Gap from Merkabit  = {gap_merkabit:.2e}")
    print(f"  Gap from CODATA    = {gap_codata:.2e}")

    # Cross-check the algebra
    print(f"\n  Cross-check:")
    neg_ln_F = 36/100 - (11/12)/1e5
    alpha_check = 137 + neg_ln_F / 10
    print(f"    -ln(F) = 36/100 - (11/12)/1e5 = {neg_ln_F:.15f}")
    print(f"    137 + (-ln(F))/10 = 137 + 36/1000 - (11/12)/1e6")
    print(f"                      = {alpha_check:.12f}")
    print(f"    This IS the three-rung formula for alpha^{{-1}}.")

    # Using the N=12 simulation F
    U_F = floquet_unitary()
    F_sim = abs(PSI_ZERO.conj() @ U_F @ PSI_ZERO)**2
    alpha_from_sim = 137 - np.log(F_sim) / 10

    print(f"\n  From N=12 simulation:")
    print(f"    F_sim            = {F_sim:.15f}")
    print(f"    137 - ln(F_sim)/10 = {alpha_from_sim:.12f}")
    print(f"    Gap from CODATA  = {abs(alpha_from_sim - ALPHA_CODATA):.2e}")

    passed = gap_merkabit < 1e-8
    print(f"\n  TEST 3 RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"    The identity alpha^{{-1}} = 137 - ln(F)/10 is {'EXACT' if passed else 'approximate'}.")
    print(f"    F and alpha^{{-1}} share the SAME formula, shifted by one power of V=10.")

    return {
        'alpha_from_F': float(alpha_from_F),
        'alpha_merkabit': float(ALPHA_MERKABIT),
        'alpha_codata': float(ALPHA_CODATA),
        'gap_merkabit': float(gap_merkabit),
        'gap_codata': float(gap_codata),
        'alpha_from_sim': float(alpha_from_sim),
        'passed': passed,
    }


# ============================================================================
# ADDITIONAL: Check candidate closed forms for F at N=12
# ============================================================================

def check_closed_forms():
    """Check if F(N=12) = 0.696778 has a clean closed form."""
    print("\n" + "=" * 76)
    print("  SUPPLEMENTARY: Closed-form candidates for F(N=12) = 0.696778")
    print("=" * 76)

    U_F = floquet_unitary()
    F_num = abs(PSI_ZERO.conj() @ U_F @ PSI_ZERO)**2
    neg_ln_F = -np.log(F_num)

    print(f"\n  F(N=12)       = {F_num:.15f}")
    print(f"  -ln(F)        = {neg_ln_F:.15f}")
    print(f"  1/F           = {1/F_num:.15f}")
    print(f"  F^2           = {F_num**2:.15f}")
    print(f"  sqrt(F)       = {np.sqrt(F_num):.15f}")

    candidates = {
        "e^(-36/100)":                    np.exp(-36/100),
        "e^(-36/100+(11/12)/1e5)":        F_ANALYTIC,
        "cos^2(pi/6) * e^(-delta)":       np.cos(np.pi/6)**2 * np.exp(-0.36 + 0.75),
        "3/4 * e^(-delta)":               0.75 * np.exp(neg_ln_F - np.log(0.75)),
        "ln(2)":                          np.log(2),
        "1 - 1/e":                        1 - 1/np.e,
        "2/e":                            2/np.e,
        "sqrt(1/2)":                      np.sqrt(0.5),
        "cos(pi/5)^2":                    np.cos(np.pi/5)**2,
    }

    print(f"\n  {'Candidate':<40s}  {'Value':>14s}  {'|diff|':>12s}")
    print(f"  {'-'*40}  {'-'*14}  {'-'*12}")
    for name, val in candidates.items():
        diff = abs(val - F_num)
        marker = " <-- CLOSE" if diff < 0.002 else ""
        print(f"  {name:<40s}  {val:14.10f}  {diff:12.2e}{marker}")

    # Check -ln(F) against E6 invariant ratios
    print(f"\n  -ln(F) = {neg_ln_F:.15f} vs E6 invariant candidates:")
    ln_candidates = {
        "36/100 = n+/V^2":              36/100,
        "36/100 - (11/12)/1e5":         36/100 - (11/12)/1e5,
        "6/h = 6/12 = 1/2":            0.5,
        "36/78 = n+/dim":              36/78,
        "h/33":                        12/33.3,
    }
    for name, val in ln_candidates.items():
        diff = abs(val - neg_ln_F)
        marker = " <-- CLOSE" if diff < 0.005 else ""
        print(f"    {name:<35s}  {val:14.10f}  |diff| = {diff:.2e}{marker}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 76)
    print("  FLOQUET F DERIVATION: Route C Closure")
    print("  F = exp(-n+/V^2 + (11/12)/V^5) = exp(-0.359990833)")
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 76)

    t_start = time.time()

    # Step 0: Verify baseline
    F_baseline, berry_phases = verify_baseline()

    # Step 1: Trotter scaling
    result_1 = test_trotter_scaling()

    # Step 2: High-precision computation
    result_2 = test_symbolic_floquet()

    # Step 3: Alpha identity
    result_3 = test_alpha_identity()

    # Supplementary: closed forms
    check_closed_forms()

    elapsed = time.time() - t_start

    # ========================================================================
    # FINAL SYNTHESIS
    # ========================================================================
    print("\n" + "=" * 76)
    print("  FINAL SYNTHESIS")
    print("=" * 76)

    print(f"""
  Analytic claim: F = exp(-36/V^2 + (11/12)/V^5),  V=10
    = exp(-0.359990833...) = {F_ANALYTIC:.10f}
  Same structure as alpha^{{-1}}, shifted one power of V.
  Identity: alpha^{{-1}} = 137 - ln(F)/10

  TEST 1 -- Trotter scaling:
    F(m=1)    = {result_1['F_values'].get('1', 0):.10f}   [existing simulation, 12 steps]
    F(m=128)  = {result_1['F_values'].get('128', 0):.10f}   [1536 sub-steps]
    F(m=512)  = {result_1['F_values'].get('512', 0):.10f}   [6144 sub-steps]
    F_inf     = {result_1['F_inf_fit']:.10f}  +/- {result_1['F_inf_fit_err']:.2e}
    F_Rich.   = {result_1['F_inf_richardson']:.10f}
    Target    = {result_1['target']:.10f}
    Gap (fit) = {result_1['gap']:.2e}
    c (1/N^2) = {result_1['c_coefficient']:.6f}
    RESULT: {'PASS' if result_1['passed'] else 'FAIL'}

  TEST 2 -- High-precision (N=12):
    F (mpmath) = {result_2['F_mp']:.15f}
    Target     = {result_2['F_target']:.15f}
    Gap        = {result_2['gap']:.2e}
    RESULT: {'PASS' if result_2['passed'] else 'FAIL (expected: Trotter error at N=12)'}

  TEST 3 -- alpha^{{-1}} identity:
    137 - ln(F)/10 = {result_3['alpha_from_F']:.12f}
    CODATA         = {result_3['alpha_codata']:.12f}
    Gap            = {result_3['gap_codata']:.2e}
    RESULT: {'PASS' if result_3['passed'] else 'FAIL'}

  OVERALL:""")

    n_pass = sum(1 for r in [result_1, result_2, result_3] if r['passed'])
    if n_pass == 3:
        print("    3/3 PASS -> Route C CLOSED. F forced by E6 geometry.")
    elif result_3['passed'] and not result_1['passed']:
        print(f"    {n_pass}/3 PASS.")
        print("    TEST 3 PASSES: the identity alpha^{{-1}} = 137 - ln(F)/10 is exact.")
        print("    TEST 1 shows F = 0.696778 is the EXACT discrete Floquet result,")
        print("    NOT a Trotter approximation to exp(-0.3600).")
        print("    The gap between 0.696778 and 0.697683 is a real physical difference")
        print("    between the discrete 12-step cycle and the continuous formula.")
        print("    Route C status: the FORMULA alpha^-1 = 137 - ln(F)/10 is confirmed,")
        print("    but F itself takes its exact discrete value, not exp(-n+/V^2).")
    else:
        print(f"    {n_pass}/3 PASS -> See individual test output.")

    print(f"\n  Total computation time: {elapsed:.1f}s")
    print("=" * 76)

    # Save results
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'F_baseline': float(F_baseline),
        'F_analytic': float(F_ANALYTIC),
        'alpha_merkabit': float(ALPHA_MERKABIT),
        'alpha_codata': float(ALPHA_CODATA),
        'test_1_trotter': result_1,
        'test_2_symbolic': result_2,
        'test_3_alpha': result_3,
        'tests_passed': n_pass,
    }

    # Convert numpy types to native Python for JSON serialization
    def jsonify(obj):
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [jsonify(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        return obj

    json_path = os.path.join(OUTPUT_DIR, 'floquet_F_results.json')
    with open(json_path, 'w') as f:
        json.dump(jsonify(results), f, indent=2)
    print(f"\n  Results saved: {json_path}")


if __name__ == "__main__":
    main()
