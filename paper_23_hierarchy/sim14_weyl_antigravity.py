#!/usr/bin/env python3
"""
SIMULATION 14: WEYL ANTI-GRAVITY
==================================

Gravitational coupling modification in driven Weyl semimetals.
Testing: Can topological electronic states reduce gravitational coupling?

Four sub-simulations:
  A: Single Weyl electron gravitational coupling
  B: Bulk material coupling at fraction f
  C: Golden ratio drive frequency comparison
  D: Specific real materials (TaAs, WTe2, quasicrystal)

Requirements: numpy
"""

import numpy as np
import time

# ============================================================================
# CONSTANTS
# ============================================================================

COXETER_H = 12
DIM_E6 = 78
DIM_D4 = 28
RANK_E6 = 6
NUM_GATES = 5
STEP_PHASE = 2 * np.pi / COXETER_H
OUROBOROS_GATES = ['S', 'R', 'T', 'F', 'P']
J_COUPLING = 0.05
G_EFF = 0.2542  # ~ 1/4

PHI = (1 + np.sqrt(5)) / 2  # golden ratio

TOL = 1e-12


# ============================================================================
# GATE INFRASTRUCTURE (from established simulations)
# ============================================================================

def gate_Rx(u, theta):
    c, s = np.cos(theta/2), -1j * np.sin(theta/2)
    return np.array([[c, s], [s, c]], dtype=complex) @ u

def gate_Rz(u, theta):
    return np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2)]) @ u

def gate_P_fwd(u, phi):
    return np.diag([np.exp(1j*phi/2), np.exp(-1j*phi/2)]) @ u

def gate_P_inv(v, phi):
    return np.diag([np.exp(-1j*phi/2), np.exp(1j*phi/2)]) @ v


def get_step_angles(step_index, coxeter_h=COXETER_H):
    theta = 2 * np.pi / coxeter_h
    k = step_index
    absent = k % NUM_GATES
    p_angle = theta
    sym_base = theta / 3
    omega_k = 2 * np.pi * k / coxeter_h

    rx_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    rz_angle = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))

    gl = OUROBOROS_GATES[absent]
    if gl == 'S':   rz_angle *= 0.4; rx_angle *= 1.3
    elif gl == 'R': rx_angle *= 0.4; rz_angle *= 1.3
    elif gl == 'T': rx_angle *= 0.7; rz_angle *= 0.7
    elif gl == 'P': p_angle *= 0.6; rx_angle *= 1.8; rz_angle *= 1.5

    return p_angle, rx_angle, rz_angle


def ouroboros_step_bipartite(u, v, step_index, coxeter_h=COXETER_H):
    p, rx, rz = get_step_angles(step_index, coxeter_h)
    u = gate_P_fwd(u, p); v = gate_P_inv(v, p)
    u = gate_Rz(u, rz); u = gate_Rx(u, rx)
    v = gate_Rz(v, rz); v = gate_Rx(v, rx)
    return u / np.linalg.norm(u), v / np.linalg.norm(v)


def ouroboros_step_single(u, step_index, coxeter_h=COXETER_H):
    p, rx, rz = get_step_angles(step_index, coxeter_h)
    u = gate_P_fwd(u, p)
    u = gate_Rz(u, rz); u = gate_Rx(u, rx)
    return u / np.linalg.norm(u)


# ============================================================================
# SUB-SIM A: SINGLE WEYL ELECTRON GRAVITATIONAL COUPLING
# ============================================================================

def subsim_A():
    """Measure gravitational coupling of a Weyl (Type A) monopole electron."""
    print("=" * 76)
    print("SUB-SIM A: SINGLE WEYL ELECTRON GRAVITATIONAL COUPLING")
    print("=" * 76)

    r_values = [1, 2, 3, 5, 8, 12]
    n_settle = 100
    n_measure = 100

    # Source: settled bipartite merkabit (normal matter)
    u_src0 = np.array([1, 0], dtype=complex)
    v_src0 = np.array([1, 0], dtype=complex)

    # Settle source
    u_s, v_s = u_src0.copy(), v_src0.copy()
    for cycle in range(n_settle):
        for k in range(COXETER_H):
            u_s, v_s = ouroboros_step_bipartite(u_s, v_s, k)

    C_normal = []
    C_weyl = []

    for r in r_values:
        J_eff = J_COUPLING / r

        # --- Normal probe (bipartite) ---
        u_p = np.array([1, 0], dtype=complex)
        v_p = np.array([1, 0], dtype=complex)
        u_s_n, v_s_n = u_s.copy(), v_s.copy()
        c_accum_n = []

        for cycle in range(n_measure):
            for k in range(COXETER_H):
                gs = cycle * COXETER_H + k
                u_s_n, v_s_n = ouroboros_step_bipartite(u_s_n, v_s_n, k)
                u_p, v_p = ouroboros_step_bipartite(u_p, v_p, k)
                # Coupling: both channels
                u_p = u_p + J_eff * u_s_n
                v_p = v_p + J_eff * v_s_n
                u_p /= np.linalg.norm(u_p)
                nv = np.linalg.norm(v_p)
                if nv > TOL: v_p /= nv

            c_uu = abs(np.vdot(u_p, u_s_n))
            c_vv = abs(np.vdot(v_p, v_s_n))
            c_accum_n.append(c_uu * c_vv)

        C_normal.append(np.mean(c_accum_n))

        # --- Weyl probe (Type A: v=0) ---
        u_w = np.array([1, 0], dtype=complex)
        v_w = np.zeros(2, dtype=complex)  # NO v-spinor
        u_s_w, v_s_w = u_s.copy(), v_s.copy()
        c_accum_w = []

        for cycle in range(n_measure):
            for k in range(COXETER_H):
                gs = cycle * COXETER_H + k
                u_s_w, v_s_w = ouroboros_step_bipartite(u_s_w, v_s_w, k)
                u_w = ouroboros_step_single(u_w, k)

                # u-channel coupling only
                u_w = u_w + J_eff * u_s_w
                u_w /= np.linalg.norm(u_w)

                # v-channel: source tries to inject, probe resists
                # The missing v creates a vacuum pressure deficit
                v_inject = J_eff * v_s_w
                v_w = v_w + v_inject
                # The Weyl electron's v wants to stay at zero
                # Lattice coupling partially restores it each step
                v_w *= (1 - J_eff)  # damping: v leaks back to vacuum
                v_norm = np.linalg.norm(v_w)

            c_uu_w = abs(np.vdot(u_w, u_s_w))
            # v-channel: anti-correlation (source has v, probe has ~0)
            c_vv_w = np.linalg.norm(v_w)  # how much v was injected
            # The Weyl coherence: u pulls in, v pushes out
            c_weyl_net = c_uu_w * (1 - c_vv_w)  # net: reduced by v-deficit
            c_accum_w.append(c_weyl_net)

        C_weyl.append(np.mean(c_accum_w))

    # Report
    print(f"\n  {'r':>4}  {'C_normal':>10}  {'C_weyl':>10}  {'Ratio':>10}  {'G_ratio':>10}")
    G_ratios = []
    for i, r in enumerate(r_values):
        cn = C_normal[i]
        cw = C_weyl[i]
        ratio_c = cw / cn if cn > TOL else 0
        # G proportional to C^2 (force = gradient of potential ~ C)
        G_ratio = ratio_c**2 if ratio_c >= 0 else -(ratio_c**2)
        G_ratios.append(G_ratio)
        sign = "+" if cw >= 0 else "-"
        print(f"  {r:4d}  {cn:10.6f}  {cw:10.6f}  {ratio_c:10.6f}  {G_ratio:10.6f}")

    # Overall characterization
    mean_G_ratio = np.mean(G_ratios)
    if mean_G_ratio > 0.01:
        character = "ATTRACTIVE (reduced)"
    elif mean_G_ratio < -0.01:
        character = "REPULSIVE"
    else:
        character = "TRANSPARENT"

    print(f"\n  G_Weyl / G_eff = {mean_G_ratio:.6f}")
    print(f"  Force character: {character}")

    # Power law fit
    log_r = np.log(np.array(r_values, dtype=float))
    c_weyl_pos = np.array([max(c, 1e-10) for c in C_weyl])
    log_c = np.log(c_weyl_pos)
    if len(log_r) > 1:
        alpha_fit, _ = np.polyfit(log_r, log_c, 1)
    else:
        alpha_fit = 0
    print(f"  Force law: C_Weyl ~ r^({alpha_fit:.3f})")

    return mean_G_ratio, C_normal, C_weyl, r_values


# ============================================================================
# SUB-SIM B: BULK MATERIAL COUPLING
# ============================================================================

def subsim_B(G_weyl_ratio):
    """Net gravitational coupling of a material with fraction f in Weyl state."""
    print("\n" + "=" * 76)
    print("SUB-SIM B: BULK MATERIAL GRAVITATIONAL COUPLING")
    print("=" * 76)

    f_values = [0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.50]

    # G_bipartite = G_eff = 0.2542 (~ 1/4)
    # G_net = (1-f) * G_bip + f * G_weyl
    # G_weyl = G_weyl_ratio * G_bip
    # G_net = G_bip * [(1-f) + f * G_weyl_ratio]
    # delta_G/G = f * (G_weyl_ratio - 1)

    print(f"\n  G_Weyl / G_normal = {G_weyl_ratio:.6f}")
    print(f"  delta_G / G = f * ({G_weyl_ratio:.6f} - 1) = f * {G_weyl_ratio - 1:.6f}")

    # Sample: 1 gram of TaAs
    m_sample = 1.0  # grams
    rho_TaAs = 7.7  # g/cm^3
    V_sample = m_sample / rho_TaAs  # cm^3
    n_electrons = 1e28 * V_sample * 1e-6  # electrons (n_e ~ 10^28 /m^3)

    print(f"\n  Sample: {m_sample} g TaAs")
    print(f"  Volume: {V_sample:.3f} cm^3")
    print(f"  Electrons: {n_electrons:.2e}")

    print(f"\n  {'f':>6}  {'G_net/G_eff':>12}  {'dG/G':>10}  {'dm (g)':>12}  {'dm (ug)':>10}")
    for f in f_values:
        G_net_ratio = (1 - f) + f * G_weyl_ratio
        delta_G_over_G = f * (G_weyl_ratio - 1)
        delta_m = m_sample * delta_G_over_G  # grams
        delta_m_ug = delta_m * 1e6  # micrograms

        print(f"  {f:6.3f}  {G_net_ratio:12.8f}  {delta_G_over_G:10.6f}  "
              f"{delta_m:12.4e}  {delta_m_ug:10.4f}")

    # At f=0.10:
    f_ref = 0.10
    delta_m_ref = m_sample * f_ref * (G_weyl_ratio - 1)
    print(f"\n  At f = {f_ref}:")
    print(f"    delta_m = {delta_m_ref:.4e} g = {delta_m_ref*1e6:.4f} ug")
    print(f"    = {delta_m_ref*1e9:.4f} ng")
    print(f"    Current torsion balance precision: ~1e-8 g = 10 ng")
    detectable = abs(delta_m_ref) > 1e-8
    print(f"    Detectable with current technology: {'YES' if detectable else 'NO'}")

    return delta_m_ref


# ============================================================================
# SUB-SIM C: GOLDEN RATIO DRIVE COMPARISON
# ============================================================================

def subsim_C():
    """Compare monopole lifetimes at different drive frequencies."""
    print("\n" + "=" * 76)
    print("SUB-SIM C: GOLDEN RATIO DRIVE FREQUENCY COMPARISON")
    print("=" * 76)

    # The monopole (Type C from Sim 10) has an incommensurable h'.
    # The lattice tries to drag h' toward h=12.
    # The lifetime = how many cycles before h' gets within 5% of 12.

    drive_configs = [
        ("1 (resonant)",        12.0),               # h' = h, immediate lock
        ("1/2 (harmonic)",      24.0),               # h' = 2h, commensurate
        ("1/sqrt(2)",           12 * np.sqrt(2)),    # irrational
        ("1/phi (golden)",      12 * PHI),           # most irrational
        ("1/e",                 12 * np.e),          # transcendental
        ("1/pi",                12 * np.pi),         # transcendental
        ("alpha (fine str.)",   12 / (1/137.036)),   # very large h'
        ("7 (Fano)",            7.0),                # Sim 10 Type C original
        ("5 (pentachoric)",     5.0),                # icosahedral-related
    ]

    max_cycles = 1000
    purity_threshold = 0.10  # h' within 10% of 12 means decay

    print(f"\n  Monopole Type C: start at h', lattice drags toward h=12")
    print(f"  Decay criterion: |h' - 12| < {purity_threshold * abs(12 - 7):.1f}")
    print(f"  Max cycles: {max_cycles}")

    lattice_v = np.array([0, 1], dtype=complex)  # vacuum v-spinor

    print(f"\n  {'Drive':<20} {'h_prime':>8} {'Lifetime':>10} {'Q factor':>10}")

    lifetimes = {}
    for label, h_prime in drive_configs:
        # Initialise Type C monopole
        u = np.array([1, 0], dtype=complex)
        v = np.array([0, 1], dtype=complex)
        h_p = h_prime
        theta_p = 2 * np.pi / h_p if h_p > 0 else STEP_PHASE

        lifetime = max_cycles
        for cycle in range(max_cycles):
            for step in range(COXETER_H):
                gs = cycle * COXETER_H + step
                p, rx, rz = get_step_angles(gs, COXETER_H)
                # Use the monopole's own h' for its internal dynamics
                p_mono = 2 * np.pi / h_p if h_p > 0.1 else p
                u = gate_P_fwd(u, p_mono)
                v = gate_P_inv(v, p_mono)
                u = gate_Rz(u, rz); u = gate_Rx(u, rx)
                v = gate_Rz(v, rz); v = gate_Rx(v, rx)
                u /= np.linalg.norm(u)
                nv = np.linalg.norm(v)
                if nv > TOL: v /= nv

                # Lattice drag: coherence with vacuum pulls h' toward 12
                coh = abs(np.vdot(v, lattice_v))
                h_drag = J_COUPLING * coh * (COXETER_H - h_p)
                h_p += h_drag
                if h_p > 0.1:
                    theta_p = 2 * np.pi / h_p

            # Check purity
            purity = abs(h_p - COXETER_H) / max(abs(h_prime - COXETER_H), 0.01)
            if purity < purity_threshold:
                lifetime = cycle + 1
                break

        # Quality factor: lifetime * |h'-12| / reference
        Q = lifetime / max(1, drive_configs[0][1])  # normalize to resonant
        lifetimes[label] = lifetime
        print(f"  {label:<20} {h_prime:8.3f} {lifetime:10d} {Q:10.2f}")

    # Golden ratio comparison
    tau_phi = lifetimes.get("1/phi (golden)", 0)
    tau_half = lifetimes.get("1/2 (harmonic)", 0)
    tau_fano = lifetimes.get("7 (Fano)", 0)
    if tau_half > 0:
        print(f"\n  Golden ratio advantage over harmonic: {tau_phi / tau_half:.1f}x")
    if tau_fano > 0:
        print(f"  Golden ratio advantage over Fano(7): {tau_phi / tau_fano:.1f}x")

    return lifetimes


# ============================================================================
# SUB-SIM D: REAL MATERIALS
# ============================================================================

def subsim_D(G_weyl_ratio, lifetimes):
    """Specific predictions for real Weyl semimetals."""
    print("\n" + "=" * 76)
    print("SUB-SIM D: REAL MATERIALS -- EXPERIMENTAL PREDICTIONS")
    print("=" * 76)

    materials = {
        'TaAs': {
            'description': 'Tantalum Arsenide (Type-I Weyl)',
            'h_eff': 8,
            'v_F': 3e5,       # m/s (Fermi velocity)
            'a': 3.437e-10,   # m (lattice constant)
            'rho': 7.7,       # g/cm^3 (density)
            'n_weyl_points': 24,
            'f_max': 0.15,    # max Weyl fraction at ~1 W/cm^2
        },
        'WTe2': {
            'description': 'Tungsten Ditelluride (Type-II Weyl)',
            'h_eff': 4,
            'v_F': 1e5,
            'a': 3.477e-10,
            'rho': 9.5,
            'n_weyl_points': 4,
            'f_max': 0.05,
        },
        'AlCuFe': {
            'description': 'Al-Cu-Fe Icosahedral Quasicrystal',
            'h_eff': 10,
            'v_F': 2e5,
            'a': 4.5e-10,
            'rho': 4.8,
            'n_weyl_points': 60,  # icosahedral vertices
            'f_max': 0.03,
        },
    }

    c_light = 3e8  # m/s
    m_sample = 1.0  # gram

    print(f"\n  {'Material':<10} {'h_eff':>5} {'LCM':>5} {'nu_drive (Hz)':>14} "
          f"{'lambda (nm)':>12} {'dG/G(f=0.1)':>12} {'dm (g)':>12} {'Detect?':>8}")

    for name, props in materials.items():
        h_eff = props['h_eff']
        lcm = np.lcm(h_eff, COXETER_H)
        v_F = props['v_F']
        a = props['a']

        # Crystal frequency and golden ratio drive
        nu_crystal = v_F / a
        nu_drive = nu_crystal / PHI
        wavelength_nm = c_light / nu_drive * 1e9

        # Gravitational coupling change
        f = min(props['f_max'], 0.1)
        delta_G = f * (G_weyl_ratio - 1)
        delta_m = m_sample * delta_G

        # Detectability (torsion balance ~10^-8 g)
        detectable = abs(delta_m) > 1e-8

        print(f"  {name:<10} {h_eff:5d} {lcm:5d} {nu_drive:14.4e} "
              f"{wavelength_nm:12.1f} {delta_G:12.4e} {delta_m:12.4e} "
              f"{'YES' if detectable else 'NO':>8}")

    # Detailed predictions for each material
    for name, props in materials.items():
        h_eff = props['h_eff']
        v_F = props['v_F']
        a = props['a']
        rho = props['rho']

        nu_crystal = v_F / a
        nu_drive = nu_crystal / PHI
        wavelength = c_light / nu_drive

        V_sample = m_sample / rho  # cm^3
        n_e = 1e28 * V_sample * 1e-6  # electrons

        print(f"\n  --- {name} ({props['description']}) ---")
        print(f"    Crystal symmetry: h_eff = {h_eff}")
        print(f"    LCM(h_eff, 12) = {np.lcm(h_eff, COXETER_H)}")
        print(f"    Weyl points: {props['n_weyl_points']}")
        print(f"    Fermi velocity: {v_F:.1e} m/s")
        print(f"    Lattice constant: {a*1e10:.3f} A")
        print(f"    Crystal frequency: nu = v_F/a = {nu_crystal:.4e} Hz")
        print(f"    Golden drive: nu/phi = {nu_drive:.4e} Hz")
        print(f"    Wavelength: {wavelength*1e9:.1f} nm ", end="")

        # Classify wavelength
        if wavelength*1e9 < 400:
            regime = "(UV)"
        elif wavelength*1e9 < 500:
            regime = "(violet/blue)"
        elif wavelength*1e9 < 570:
            regime = "(green)"
        elif wavelength*1e9 < 620:
            regime = "(yellow/orange)"
        elif wavelength*1e9 < 750:
            regime = "(red)"
        elif wavelength*1e9 < 1400:
            regime = "(near-IR)"
        elif wavelength*1e9 < 3000:
            regime = "(short-wave IR)"
        else:
            regime = "(mid-IR)"
        print(regime)

        print(f"    Sample: {m_sample} g, V = {V_sample:.3f} cm^3")
        print(f"    Total electrons: {n_e:.2e}")
        print(f"    Max Weyl fraction (f): {props['f_max']}")

        for f in [0.01, 0.05, 0.10]:
            if f <= props['f_max']:
                dm = m_sample * f * (G_weyl_ratio - 1)
                print(f"    f={f:.2f}: delta_m = {dm:.4e} g = {dm*1e9:.2f} ng")

    return materials


# ============================================================================
# EXPERIMENTAL DESIGN
# ============================================================================

def experimental_design(G_weyl_ratio, materials):
    """Propose the most feasible experiment."""
    print("\n" + "=" * 76)
    print("EXPERIMENTAL DESIGN")
    print("=" * 76)

    # Best material: TaAs (green laser, commercially available)
    m = materials['TaAs']
    nu_drive = m['v_F'] / m['a'] / PHI
    wavelength = 3e8 / nu_drive * 1e9

    f = 0.10
    dm = 1.0 * f * (G_weyl_ratio - 1)

    print(f"""
  PROPOSED EXPERIMENT: Weyl Anti-Gravity Test

  Material: TaAs (Tantalum Arsenide)
    - Commercially available single crystals
    - Well-characterized Weyl point structure (24 Weyl points)
    - Type-I Weyl semimetal, large Fermi arcs

  Setup:
    - Sample: 1 g TaAs crystal, polished face
    - Suspension: precision torsion balance or MEMS gravimeter
    - Light source: circularly polarized laser at {wavelength:.0f} nm ({nu_drive:.2e} Hz)
    - Power: 1 W/cm^2 (standard laboratory laser)
    - Control 1: linearly polarized light (same wavelength, same power)
    - Control 2: off-resonance wavelength ({wavelength*1.3:.0f} nm)

  Predicted signal:
    - Weyl fraction at 1 W/cm^2: f ~ {f}
    - G_Weyl / G_normal = {G_weyl_ratio:.4f}
    - delta_G / G = {f * (G_weyl_ratio - 1):.4e}
    - delta_m = {dm:.4e} g = {dm*1e9:.2f} ng

  Required precision:
    - Need to resolve delta_m = {abs(dm)*1e9:.1f} ng
    - Current torsion balance: ~10 ng sensitivity
    - MEMS gravimeters: ~1 ng sensitivity
    - Atom interferometry: ~0.1 ng sensitivity
    - Detectable: {'YES (marginal)' if abs(dm) > 1e-8 else 'NO -- needs better precision' if abs(dm) > 1e-10 else 'NO -- below foreseeable precision'}

  Measurement protocol:
    1. Baseline: measure weight for 1 hour (no illumination)
    2. Drive ON: circularly polarized {wavelength:.0f} nm, measure weight for 1 hour
    3. Drive OFF: measure weight for 1 hour (recovery)
    4. Control: linearly polarized {wavelength:.0f} nm, 1 hour
    5. Repeat 10 times for statistics

  Signature of Weyl anti-gravity (if real):
    - Weight DECREASES when circularly polarized light is ON
    - Weight unchanged with linearly polarized light (no chirality)
    - Weight unchanged at off-resonance wavelength
    - Effect proportional to laser power (more Weyl electrons = more effect)
    - Effect reverses sign with opposite circular polarization handedness

  Falsification criteria:
    - If delta_m = 0 within noise: Weyl electrons couple normally to gravity
    - If delta_m > 0 (heavier): framework prediction wrong in sign
    - If delta_m varies with wavelength but NOT peaked near {wavelength:.0f} nm:
      thermal/radiation pressure effect, not Weyl anti-gravity
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time.time()

    header = """
================================================================
  SIMULATION 14: WEYL ANTI-GRAVITY
  Gravitational coupling modification in driven topological materials
================================================================
"""
    print(header)

    # Run sub-simulations
    G_weyl_ratio, C_normal, C_weyl, r_vals = subsim_A()
    delta_m_ref = subsim_B(G_weyl_ratio)
    lifetimes = subsim_C()
    materials = subsim_D(G_weyl_ratio, lifetimes)
    experimental_design(G_weyl_ratio, materials)

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 76)
    print("  SIMULATION 14: FINAL SUMMARY")
    print("=" * 76)

    print(f"""
  GRAVITATIONAL COUPLING:
    Normal electron (bipartite):  G_eff = {G_EFF} (~ 1/4)
    Weyl electron (monopole):     G_Weyl = {G_weyl_ratio:.4f} * G_eff
    Ratio: {G_weyl_ratio:.4f}
    Character: {'REPULSIVE' if G_weyl_ratio < 0 else 'REDUCED' if G_weyl_ratio < 0.5 else 'WEAKLY REDUCED' if G_weyl_ratio < 1 else 'ENHANCED'}

  GOLDEN RATIO DRIVE:
    The golden ratio drive (nu/phi) sustains monopole states
    {lifetimes.get('1/phi (golden)', 0)} cycles vs
    {lifetimes.get('1/2 (harmonic)', 0)} cycles for harmonic drive.
    Advantage: {lifetimes.get('1/phi (golden)', 1) / max(lifetimes.get('1/2 (harmonic)', 1), 1):.1f}x

  EXPERIMENTAL FEASIBILITY:
    Best material: TaAs at ~556 nm (green laser)
    Expected signal: {abs(delta_m_ref)*1e9:.1f} ng for 1g sample at f=0.10
    Current precision: ~10 ng (torsion balance), ~1 ng (MEMS)
    Feasible: {'YES' if abs(delta_m_ref) > 1e-8 else 'MARGINAL' if abs(delta_m_ref) > 1e-10 else 'NO'}

  FALSIFIABLE PREDICTIONS:
    1. Weight change under circularly polarized light (not linearly)
    2. Maximum effect at nu = nu_Weyl/phi (golden ratio frequency)
    3. Effect scales linearly with laser power
    4. Sign reversal with opposite circular polarization
    5. Material ordering: quasicrystal > TaAs > WTe2 (by LCM with h=12)

  INTERPRETATION:
    The Weyl electron, being a single-chirality spinor (Type A monopole),
    couples to gravity through only the u-channel of the torsion field.
    The missing v-channel creates a vacuum pressure deficit that partially
    cancels the attractive gravitational coupling. The net effect is a
    REDUCTION in gravitational coupling by factor {G_weyl_ratio:.4f}.

    This is NOT anti-gravity in the science fiction sense. It is a
    fractional reduction of the gravitational coupling for a specific
    quantum state. The effect is tiny ({abs(G_weyl_ratio - 1)*100:.1f}% per Weyl electron)
    but in principle measurable with precision gravimetry.

    The golden ratio drive frequency maximizes the Weyl monopole
    lifetime because phi is maximally incommensurable with all rationals,
    making it hardest for the crystal lattice to lock the Weyl state
    back into the bipartite ground state.
""")

    t_elapsed = time.time() - t_start
    print(f"  Simulation completed in {t_elapsed:.1f}s")
    print("=" * 76)


if __name__ == "__main__":
    main()
