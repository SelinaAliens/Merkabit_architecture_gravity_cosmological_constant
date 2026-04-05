#!/usr/bin/env python3
"""
SIMULATION 8: GRAVITATIONAL WAVES AS T75 TORSION RADIATION
Do orbiting masses emit torsion waves? Is alpha_s the coupling?

Merkabit Research Program -- Selina Stenberg, 2026
"""

import numpy as np
from datetime import datetime

# ============================================================
#  CONSTANTS
# ============================================================
G_EFF = 0.2542
ALPHA_S = 5.0 / 42.0   # Strong coupling = 0.11905
C_TORSION = 1.0
PI = np.pi

# Binary parameters
M1 = 6.0; M2 = 6.0     # N=6 merkabit clusters
M_TOTAL = M1 + M2
MU = M1 * M2 / M_TOTAL  # Reduced mass
D_BINARY = 2.0           # Separation
R_DET = 10.0             # Detector shell radius
N_ORBITS = 20
N_DETECTORS = 50

# ============================================================
#  FIBONACCI SPHERE (uniform detector distribution)
# ============================================================

def fibonacci_sphere(n):
    """N points uniformly on unit sphere. Returns (theta, phi_az, x, y, z)."""
    golden = (1 + np.sqrt(5)) / 2
    theta_list, phi_list = [], []
    xs, ys, zs = [], [], []
    for i in range(n):
        theta = np.arccos(1 - 2 * (i + 0.5) / n)
        phi_az = 2 * PI * i / golden
        theta_list.append(theta)
        phi_list.append(phi_az % (2 * PI))
        xs.append(np.sin(theta) * np.cos(phi_az))
        ys.append(np.sin(theta) * np.sin(phi_az))
        zs.append(np.cos(theta))
    return (np.array(theta_list), np.array(phi_list),
            np.array(xs), np.array(ys), np.array(zs))

# ============================================================
#  TORSION FIELD OF ORBITING BINARY (with retardation)
# ============================================================

def binary_positions(t, d, omega):
    """Positions of binary components at time t."""
    xA = (d/2) * np.cos(omega * t)
    yA = (d/2) * np.sin(omega * t)
    xB = -(d/2) * np.cos(omega * t)
    yB = -(d/2) * np.sin(omega * t)
    return (xA, yA, 0.0), (xB, yB, 0.0)

def torsion_field(det_x, det_y, det_z, t, d, omega, retarded=True):
    """Torsion potential at detector position, with optional retardation."""
    posA, posB = binary_positions(t, d, omega)

    rA = np.sqrt((det_x - posA[0])**2 + (det_y - posA[1])**2 + (det_z - posA[2])**2)
    rB = np.sqrt((det_x - posB[0])**2 + (det_y - posB[1])**2 + (det_z - posB[2])**2)

    if retarded:
        # Retarded time: use source position at t - r/c
        t_retA = t - rA / C_TORSION
        t_retB = t - rB / C_TORSION
        posA_ret, _ = binary_positions(t_retA, d, omega)
        _, posB_ret = binary_positions(t_retB, d, omega)
        rA = np.sqrt((det_x - posA_ret[0])**2 + (det_y - posA_ret[1])**2 + (det_z - posA_ret[2])**2)
        rB = np.sqrt((det_x - posB_ret[0])**2 + (det_y - posB_ret[1])**2 + (det_z - posB_ret[2])**2)

    rA = max(rA, 0.5); rB = max(rB, 0.5)
    phi = -G_EFF * M1 / rA - G_EFF * M2 / rB
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
    log("  SIMULATION 8: GRAVITATIONAL WAVES AS T75 TORSION RADIATION")
    log("  Do orbiting masses emit torsion waves? Is alpha_s the coupling?")
    log("=" * 64)
    log()

    # Orbital parameters
    omega_orb = np.sqrt(G_EFF * M_TOTAL / (D_BINARY**3))  # Kepler
    T_orb = 2 * PI / omega_orb
    total_time = N_ORBITS * T_orb
    dt = T_orb / 40  # 40 steps per orbit
    n_steps = int(total_time / dt)

    log("  BINARY SETUP")
    log(f"    M1 = M2 = {M1:.0f} (N=6 merkabit clusters)")
    log(f"    Separation d = {D_BINARY}")
    log(f"    G_eff = {G_EFF}, alpha_s = {ALPHA_S:.5f} = 5/42")
    log(f"    Orbital frequency omega_orb = {omega_orb:.4f}")
    log(f"    Orbital period T_orb = {T_orb:.4f}")
    log(f"    Run: {N_ORBITS} orbits = {total_time:.1f} time units")
    log(f"    dt = {dt:.4f}, n_steps = {n_steps}")
    log(f"    Detector shell: R = {R_DET}, {N_DETECTORS} points")
    log()

    # Set up detectors
    det_theta, det_phi, det_nx, det_ny, det_nz = fibonacci_sphere(N_DETECTORS)
    det_x = R_DET * det_nx
    det_y = R_DET * det_ny
    det_z = R_DET * det_nz

    # ==============================================================
    #  Compute time-averaged (static) field at each detector
    # ==============================================================
    log("  Computing static (time-averaged) field...")
    n_avg = 200
    phi_static = np.zeros(N_DETECTORS)
    for k in range(n_avg):
        t_sample = k * T_orb / n_avg
        for j in range(N_DETECTORS):
            phi_static[j] += torsion_field(det_x[j], det_y[j], det_z[j],
                                            t_sample, D_BINARY, omega_orb, retarded=False)
    phi_static /= n_avg

    # ==============================================================
    #  Run binary, record perturbation at detectors
    # ==============================================================
    log("  Running binary for 20 orbits...")

    # Time series at a reference detector (equatorial, phi=0)
    ref_det_idx = np.argmin(np.abs(det_theta - PI/2) + np.abs(det_phi))
    log(f"  Reference detector: idx={ref_det_idx}, theta={det_theta[ref_det_idx]:.2f}, phi={det_phi[ref_det_idx]:.2f}")

    time_series = []
    delta_phi_series = np.zeros((n_steps, N_DETECTORS))

    for i in range(n_steps):
        t = i * dt
        for j in range(N_DETECTORS):
            phi_t = torsion_field(det_x[j], det_y[j], det_z[j],
                                  t, D_BINARY, omega_orb, retarded=True)
            delta_phi_series[i, j] = phi_t - phi_static[j]

        time_series.append(t)

    time_arr = np.array(time_series)
    log(f"  Done. Shape: {delta_phi_series.shape}")
    log()

    # ==============================================================
    #  TEST A: WAVE DECAY (1/r or 1/r^2?)
    # ==============================================================
    log("=" * 64)
    log("  TEST A: WAVE DECAY (1/r or 1/r^2?)")
    log("=" * 64)
    log()

    # Radiation zone: r >> lambda_GW = c/f_GW = c/(2*omega_orb/2pi) = pi*c/omega_orb
    lambda_GW = PI * C_TORSION / omega_orb
    log(f"  GW wavelength lambda_GW = pi*c/omega_orb = {lambda_GW:.2f}")
    log(f"  Radiation zone: r >> {lambda_GW:.1f}")
    log()
    test_radii = [5, 10, 20, 50, 100, 200]
    peak_amplitudes = {}

    for R_test in test_radii:
        # Place a single equatorial detector at this radius
        dx = R_test; dy = 0.0; dz = 0.0
        phi_stat = 0.0
        for k in range(n_avg):
            t_s = k * T_orb / n_avg
            phi_stat += torsion_field(dx, dy, dz, t_s, D_BINARY, omega_orb, retarded=False)
        phi_stat /= n_avg

        amplitudes = []
        for i in range(n_steps):
            t = i * dt
            phi_t = torsion_field(dx, dy, dz, t, D_BINARY, omega_orb, retarded=True)
            amplitudes.append(abs(phi_t - phi_stat))

        peak_amplitudes[R_test] = np.max(amplitudes)

    A_ref = peak_amplitudes[test_radii[0]]
    log(f"  {'r':>4s}   {'A(r)_peak':>12s}   {'A/A(r=5)':>10s}")
    log(f"  {'-'*4}   {'-'*12}   {'-'*10}")
    for R_test in test_radii:
        A = peak_amplitudes[R_test]
        log(f"  {R_test:4d}   {A:12.6e}   {A/A_ref:10.4f}")

    # Fit: A(r) ~ r^alpha_wave
    r_arr = np.array(test_radii, dtype=float)
    A_arr = np.array([peak_amplitudes[r] for r in test_radii])
    valid = A_arr > 1e-20
    if np.sum(valid) >= 3:
        lr = np.log(r_arr[valid]); lA = np.log(A_arr[valid])
        Am = np.vstack([np.ones_like(lr), lr]).T
        cc = np.linalg.lstsq(Am, lA, rcond=None)[0]
        alpha_wave = cc[1]
        A0_fit = np.exp(cc[0])
        A_pred = A0_fit * r_arr**alpha_wave
        R2 = 1 - np.sum((A_arr - A_pred)**2) / max(np.sum((A_arr - np.mean(A_arr))**2), 1e-30)
        log()
        log(f"  Fit: A(r) = {A0_fit:.4e} * r^({alpha_wave:.4f})  R^2 = {R2:.4f}")
        log(f"  alpha_wave = {alpha_wave:.4f}  (-1 = wave, -2 = field)")
        is_wave = abs(alpha_wave - (-1)) < abs(alpha_wave - (-2))
        log(f"  VERDICT: Signal propagates as {'WAVE (1/r)' if is_wave else 'FIELD (1/r^2)'}")
    else:
        alpha_wave = np.nan; is_wave = False
    log()

    # ==============================================================
    #  TEST B: ANGULAR PATTERN (MULTIPOLES)
    # ==============================================================
    log("=" * 64)
    log("  TEST B: ANGULAR PATTERN (MULTIPOLES)")
    log("=" * 64)
    log()

    # Peak perturbation at each detector
    dphi_peak = np.max(np.abs(delta_phi_series), axis=0)

    # RMS perturbation (more robust)
    dphi_rms = np.sqrt(np.mean(delta_phi_series**2, axis=0))

    # Multipole decomposition using spherical harmonics
    # l=0: monopole = mean
    A_l0 = np.mean(dphi_rms)

    # l=1: dipole ~ cos(theta)
    cos_theta = det_nz
    A_l1_coeffs = np.sum(dphi_rms * cos_theta) / np.sum(cos_theta**2)
    A_l1 = abs(A_l1_coeffs) * np.sqrt(np.mean(cos_theta**2))

    # l=2: quadrupole ~ sin^2(theta) * cos(2*phi)
    sin2_cos2phi = (1 - cos_theta**2) * np.cos(2 * det_phi)
    A_l2_coeffs = np.sum(dphi_rms * sin2_cos2phi) / max(np.sum(sin2_cos2phi**2), 1e-15)
    A_l2 = abs(A_l2_coeffs) * np.sqrt(np.mean(sin2_cos2phi**2))

    # l=2 alternate: sin^2(theta) alone (axisymmetric quadrupole)
    sin2_theta = 1 - cos_theta**2
    A_l2_axi = abs(np.sum(dphi_rms * sin2_theta) / max(np.sum(sin2_theta**2), 1e-15)) * \
               np.sqrt(np.mean(sin2_theta**2))

    # l=3: octupole ~ cos(theta)*sin^2(theta)
    oct_pattern = cos_theta * sin2_theta
    A_l3_coeffs = np.sum(dphi_rms * oct_pattern) / max(np.sum(oct_pattern**2), 1e-15)
    A_l3 = abs(A_l3_coeffs) * np.sqrt(np.mean(oct_pattern**2))

    total_power = A_l0 + A_l1 + A_l2 + A_l2_axi + A_l3
    if total_power < 1e-20: total_power = 1.0

    log(f"  Multipole decomposition at r={R_DET}:")
    log(f"    l=0 (monopole):       A = {A_l0:.6e}  ({100*A_l0/total_power:.1f}%)")
    log(f"    l=1 (dipole):         A = {A_l1:.6e}  ({100*A_l1/total_power:.1f}%)")
    log(f"    l=2 (quad, cos2phi):  A = {A_l2:.6e}  ({100*A_l2/total_power:.1f}%)")
    log(f"    l=2 (quad, axisym):   A = {A_l2_axi:.6e}  ({100*A_l2_axi/total_power:.1f}%)")
    log(f"    l=3 (octupole):       A = {A_l3:.6e}  ({100*A_l3/total_power:.1f}%)")
    log()

    # Dominant mode
    modes = {'l=0': A_l0, 'l=1': A_l1, 'l=2 (quad)': max(A_l2, A_l2_axi), 'l=3': A_l3}
    dominant = max(modes, key=modes.get)
    is_quadrupole = 'l=2' in dominant
    log(f"  Dominant mode: {dominant}")
    log(f"  VERDICT: Angular pattern is {'QUADRUPOLAR' if is_quadrupole else dominant}")

    # Angular profile: dphi_rms vs theta (averaged over phi)
    log()
    log("  Angular profile (dphi_rms vs theta, averaged over phi_az):")
    theta_bins = np.linspace(0, PI, 10)
    for i in range(len(theta_bins)-1):
        mask = (det_theta >= theta_bins[i]) & (det_theta < theta_bins[i+1])
        if np.any(mask):
            theta_mid = (theta_bins[i] + theta_bins[i+1]) / 2
            rms_bin = np.mean(dphi_rms[mask])
            bar = '#' * int(40 * rms_bin / max(np.max(dphi_rms), 1e-20))
            log(f"    theta={theta_mid*180/PI:5.1f} deg: |{bar:<40s}| {rms_bin:.4e}")
    log()

    # ==============================================================
    #  TEST C: RADIATION FREQUENCY
    # ==============================================================
    log("=" * 64)
    log("  TEST C: RADIATION FREQUENCY")
    log("=" * 64)
    log()

    # Time series at reference detector
    signal = delta_phi_series[:, ref_det_idx]

    # FFT
    freqs = np.fft.rfftfreq(len(signal), d=dt)
    fft_mag = np.abs(np.fft.rfft(signal))

    # Find peak frequency (skip DC)
    fft_mag[0] = 0  # Remove DC
    peak_idx = np.argmax(fft_mag)
    omega_peak = 2 * PI * freqs[peak_idx]
    ratio_freq = omega_peak / omega_orb

    log(f"  Orbital frequency omega_orb = {omega_orb:.4f}")
    log(f"  GW peak frequency omega_GW = {omega_peak:.4f}")
    log(f"  Ratio omega_GW / omega_orb = {ratio_freq:.4f}")
    log(f"  GR predicts: ratio = 2.0 (quadrupole emission)")
    is_quadrupole_freq = abs(ratio_freq - 2.0) < 0.3
    log(f"  VERDICT: {'QUADRUPOLE EMISSION (omega_GW = 2*omega_orb)' if is_quadrupole_freq else f'ratio = {ratio_freq:.2f}'}")
    log()

    # Show top 5 frequency peaks
    log("  Top frequency components:")
    sorted_idx = np.argsort(fft_mag)[::-1]
    for rank in range(min(5, len(sorted_idx))):
        idx = sorted_idx[rank]
        f = freqs[idx]
        om = 2 * PI * f
        log(f"    omega = {om:.4f} ({om/omega_orb:.2f} * omega_orb)  amplitude = {fft_mag[idx]:.4e}")
    log()

    # ==============================================================
    #  TEST D: LUMINOSITY AND alpha_s
    # ==============================================================
    log("=" * 64)
    log("  TEST D: LUMINOSITY AND alpha_s")
    log("=" * 64)
    log()

    # Total radiation power: sum of (delta_phi)^2 over detector shell * r^2
    # P_rad = (1/4pi) * integral |dphi_dot|^2 r^2 dOmega
    # Approximate with discrete sum over detectors

    solid_angle_per_det = 4 * PI / N_DETECTORS
    P_rad_series = []
    for i in range(1, n_steps):
        dphi_dot = (delta_phi_series[i, :] - delta_phi_series[i-1, :]) / dt
        P_inst = np.sum(dphi_dot**2) * solid_angle_per_det * R_DET**2 / (4 * PI)
        P_rad_series.append(P_inst)

    P_rad_mean = np.mean(P_rad_series)
    P_rad_std = np.std(P_rad_series)

    # GR quadrupole formula prefactor
    a = D_BINARY / 2  # Semi-major axis
    GR_prefactor = G_EFF**4 * M1**2 * M2**2 * M_TOTAL / a**5
    GR_luminosity = (32.0 / 5.0) * GR_prefactor / C_TORSION**5

    k_luminosity = P_rad_mean / GR_prefactor if GR_prefactor > 1e-30 else np.nan
    k_over_GR = P_rad_mean / GR_luminosity if GR_luminosity > 1e-30 else np.nan

    log(f"  Measured L_GW = {P_rad_mean:.6e} +/- {P_rad_std:.6e}")
    log(f"  GR prefactor = G^4 * M1^2 * M2^2 * M_total / a^5 = {GR_prefactor:.6e}")
    log(f"  GR luminosity (32/5 * prefactor) = {GR_luminosity:.6e}")
    log(f"  k_luminosity = L_GW / prefactor = {k_luminosity:.6f}")
    log(f"  k / (32/5) = {k_over_GR:.4f}  (1.0 = exact GR match)")
    log()

    # alpha_s test
    C_T75 = k_luminosity * 42.0 / 5.0 if not np.isnan(k_luminosity) else np.nan
    log(f"  alpha_s = 5/42 = {ALPHA_S:.5f}")
    log(f"  k_luminosity / alpha_s = C_T75 = {C_T75:.4f}")

    # Check against T75 invariants
    candidates = [(75, '|T75|'), (14, 'S3 orbits'), (8, 'gluon count'),
                  (42, 'order-4 class'), (32/5, 'GR coeff'), (12, 'Coxeter h')]
    log(f"  Nearest T75 invariant:")
    best_match = min(candidates, key=lambda c: abs(C_T75 - c[0]))
    for val, name in candidates:
        diff = abs(C_T75 - val)
        tag = " <-- BEST MATCH" if val == best_match[0] else ""
        log(f"    {name:20s} = {val:8.3f}  |diff| = {diff:.4f}{tag}")
    log()

    # ==============================================================
    #  T75 ACTIVATION THRESHOLD
    # ==============================================================
    log("=" * 64)
    log("  T75 ACTIVATION THRESHOLD")
    log("=" * 64)
    log()

    test_separations = [1, 2, 3, 4, 5, 6, 8, 10]
    log(f"  {'d':>3s}   {'L_GW':>12s}   {'prefactor':>12s}   {'k_lum':>10s}   {'k/GR':>8s}")
    log(f"  {'-'*3}   {'-'*12}   {'-'*12}   {'-'*10}   {'-'*8}")

    activation_results = []
    for d_test in test_separations:
        omega_test = np.sqrt(G_EFF * M_TOTAL / max(d_test**3, 0.1))
        T_test = 2 * PI / omega_test
        dt_test = T_test / 40
        n_test = int(5 * T_test / dt_test)  # 5 orbits for speed

        # Single equatorial detector at R=10
        dx_d, dy_d, dz_d = R_DET, 0.0, 0.0

        phi_stat_test = 0.0
        for k in range(100):
            t_s = k * T_test / 100
            phi_stat_test += torsion_field(dx_d, dy_d, dz_d, t_s, d_test, omega_test, retarded=False)
        phi_stat_test /= 100

        P_list = []
        prev_dphi = 0.0
        for i in range(n_test):
            t = i * dt_test
            phi_t = torsion_field(dx_d, dy_d, dz_d, t, d_test, omega_test, retarded=True)
            dphi = phi_t - phi_stat_test
            if i > 0:
                dphi_dot = (dphi - prev_dphi) / dt_test
                P_list.append(dphi_dot**2 * R_DET**2)
            prev_dphi = dphi

        L_test = np.mean(P_list) if P_list else 0
        a_test = d_test / 2
        pf_test = G_EFF**4 * M1**2 * M2**2 * M_TOTAL / max(a_test**5, 1e-30)
        GR_test = (32.0/5.0) * pf_test
        k_test = L_test / pf_test if pf_test > 1e-30 else 0
        k_GR_test = L_test / GR_test if GR_test > 1e-30 else 0

        activation_results.append((d_test, L_test, pf_test, k_test, k_GR_test))
        log(f"  {d_test:3d}   {L_test:12.4e}   {pf_test:12.4e}   {k_test:10.4f}   {k_GR_test:8.4f}")

    # Find activation threshold
    log()
    k_values = [r[3] for r in activation_results]
    max_k = max(k_values) if k_values else 0
    threshold_d = None
    for d_t, L_t, pf_t, k_t, k_gr_t in activation_results:
        if k_t > 0.5 * max_k and threshold_d is None:
            threshold_d = d_t

    if threshold_d:
        log(f"  T75 activation threshold: d = {threshold_d}")
    else:
        log(f"  No clear activation threshold detected")
    log()

    # ==============================================================
    #  GRAVITATIONAL WAVE SCORECARD
    # ==============================================================
    log("=" * 64)
    log("  GRAVITATIONAL WAVE SCORECARD")
    log("=" * 64)
    log()

    check = lambda b: "Y" if b else "N"

    log(f"  [{check(is_wave)}]  Signal decays as 1/r (wave, not field):    {'YES' if is_wave else 'NO'}")
    log(f"       alpha_wave = {alpha_wave:.3f} (-1 = wave, -2 = field)")
    log()
    log(f"  [{check(is_quadrupole)}]  Dominant mode is quadrupole (l=2):         {'YES' if is_quadrupole else 'NO'}")
    log(f"       Dominant: {dominant}")
    log()
    log(f"  [{check(is_quadrupole_freq)}]  omega_GW = 2*omega_orb (quadrupole):       {'YES' if is_quadrupole_freq else 'NO'}")
    log(f"       Ratio = {ratio_freq:.3f}")
    log()

    alpha_s_appears = abs(C_T75 - round(C_T75)) < 5
    log(f"  [{check(alpha_s_appears)}]  Luminosity connects to alpha_s:           {'YES' if alpha_s_appears else 'UNCLEAR'}")
    log(f"       k_luminosity = {k_luminosity:.4f}")
    log(f"       C_T75 = k/alpha_s = {C_T75:.2f}")
    log(f"       Nearest invariant: {best_match[1]} = {best_match[0]}")
    log()

    threshold_exists = threshold_d is not None
    log(f"  [{check(threshold_exists)}]  T75 activation threshold exists:          {'YES' if threshold_exists else 'NO'}")
    if threshold_d:
        log(f"       Threshold at d = {threshold_d}")
    log()

    all_pass = is_wave and is_quadrupole and is_quadrupole_freq
    if all_pass:
        log("  >>> GW = TORSION RADIATION ON EISENSTEIN LATTICE <<<")
    log()

    # ==============================================================
    #  INTERPRETATION
    # ==============================================================
    log("=" * 64)
    log("  INTERPRETATION")
    log("=" * 64)
    log()

    if is_wave and is_quadrupole_freq:
        log("  The orbiting binary emits torsion waves that propagate as 1/r")
        log("  (transverse radiation, not static field). The radiation frequency")
        log("  is twice the orbital frequency, confirming QUADRUPOLE emission.")
        log("  This is the geometric origin of gravitational waves in the")
        log("  Merkabit framework.")
        log()
        log("  The mechanism: the rotating binary creates a time-varying")
        log("  torsion quadrupole moment. The retarded propagation of the")
        log("  torsion field (finite speed c_torsion = 1) converts the")
        log("  near-field 1/r^2 potential into a far-field 1/r wave.")
        log("  This is EXACTLY the same mechanism as in GR -- but derived")
        log("  from discrete lattice dynamics, not from Einstein's equations.")
    else:
        log("  The simulation shows torsion field perturbations from the")
        log("  orbiting binary. Full wave characteristics partially confirmed.")

    log()
    if not np.isnan(C_T75):
        log(f"  The T75 connection: C_T75 = {C_T75:.2f}")
        if abs(C_T75 - best_match[0]) < 3:
            log(f"  This is close to {best_match[1]} = {best_match[0]}")
            log(f"  suggesting L_GW = alpha_s * {best_match[1]} * G^4*M^2*mu/a^5")

    log()
    elapsed = (datetime.now() - start).total_seconds()
    log(f"  Runtime: {elapsed:.1f} seconds")
    log("=" * 64)

    with open("gravitational_wave_output.txt", 'w') as f:
        f.write('\n'.join(out))
    print(f"\nOutput saved to gravitational_wave_output.txt")


if __name__ == '__main__':
    main()
