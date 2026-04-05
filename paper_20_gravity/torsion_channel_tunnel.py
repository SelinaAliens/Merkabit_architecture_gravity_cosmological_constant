#!/usr/bin/env python3
"""
TORSION CHANNEL + TUNNEL: Two-Component Scaling Test
=====================================================

Key insight (Selina Stenberg):
  INTRA-cell torsion CHANNEL: cross-gate counter-rotation of u and v
    within each cell. Scales with VOLUME (N_cells).
  INTER-cell torsion TUNNEL: cross-chirality T_AB between cells.
    Scales with SURFACE (N_tunnels ~ N^(2/3)).

Previous test only measured the tunnel. The channel was missing because
the static M = <u|v> + <v|u> is purely imaginary by duality.

But the CROSS-GATE breaks this duality:
  M_static[i,j] = J(r) * (<u_i|v_j> + <v_i|u_j>)     [purely imaginary]
  H_cross[i,j]  = κ * J(r) * (<u_i|u_j> - <v_i|v_j>)  [COMPLEX, torsion]

The cross-gate couples forward-to-forward and inverse-to-inverse with
OPPOSITE sign = counter-rotation = torsion. This is NOT imaginary because
<u_i|u_j> has both real and imaginary parts for generic SU(2) spinors.

Full Hamiltonian:
  H = Σ_i [M_i + κ*H_cross_i] + λ * Σ_{<ij>} T_ij
       ↑ imaginary (GOE)  ↑ complex (torsion channel)  ↑ complex (torsion tunnel)
"""

import numpy as np
import sys, time
from pathlib import Path
from scipy import stats
from scipy.integrate import quad
from scipy.linalg import eigvalsh
from scipy.special import gamma as gamma_fn

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(r"C:\Users\selin\merkabit_results\torsion_scaling")
OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# Architecture constants
RANK_E6 = 6
DIM_E6 = 78
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H  # pi/6
XI = 3.0
F_RETURN = 0.696778
ALPHA_EM_INV = 137.035999083
OMEGA_EISEN = np.exp(2j * np.pi / 3)
UNIT_VECTORS_AB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]

ALPHA_EM = 1.0 / ALPHA_EM_INV
ALPHA_W  = 1.0 / 29.5
ALPHA_S  = 0.1179

# Default cross-gate strength from tesseract simulation
KAPPA_DEFAULT = 0.3  # cross_strength in ouroboros_step_4


# ============================================================================
# CELL INFRASTRUCTURE (same as before)
# ============================================================================

def eisenstein(a, b):
    return a + b * OMEGA_EISEN

class HexagonalCell:
    def __init__(self, centre_a=0, centre_b=0, cell_id=0):
        self.centre_ab = (centre_a, centre_b)
        self.cell_id = cell_id
        self.nodes_ab = [(centre_a, centre_b)]
        for da, db in UNIT_VECTORS_AB:
            self.nodes_ab.append((centre_a + da, centre_b + db))
        self.N = len(self.nodes_ab)
        self.sites = np.array([eisenstein(a, b) for a, b in self.nodes_ab])
        self.neighbours = {i: [] for i in range(self.N)}
        self.edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if abs(self.sites[i] - self.sites[j]) < 1.15:
                    self.neighbours[i].append(j)
                    self.neighbours[j].append(i)
                    self.edges.append((i, j))
        self.is_boundary = np.array([len(self.neighbours[i]) < 6 for i in range(self.N)])
        self.boundary_indices = [i for i in range(self.N) if self.is_boundary[i]]
        self.sublattice = np.array([(a + b) % 3 for a, b in self.nodes_ab])

def build_multi_cell_lattice(n_cells):
    cell_centres = []
    R = int(np.sqrt(n_cells)) + 3
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            cell_centres.append((2 * a, 2 * b))
    cell_centres.sort(key=lambda ab: abs(eisenstein(*ab)))
    cell_centres = cell_centres[:n_cells]
    return [HexagonalCell(ca, cb, cell_id=idx) for idx, (ca, cb) in enumerate(cell_centres)]

def assign_cell_spinors(cell, omega_base=1.0):
    N = cell.N
    U = np.zeros((N, 2), dtype=complex)
    V = np.zeros((N, 2), dtype=complex)
    omegas = np.zeros(N)
    L_scale = np.max(np.abs(cell.sites)) + 1.0
    for i in range(N):
        a, b = cell.nodes_ab[i]
        phase = np.pi * (a - b) / RANK_E6
        r = abs(cell.sites[i] - eisenstein(*cell.centre_ab))
        theta = np.pi * r / L_scale
        u = np.array([np.cos(theta/2) * np.exp(1j*phase),
                       1j * np.sin(theta/2) * np.exp(-1j*phase)], dtype=complex)
        u /= np.linalg.norm(u)
        v = np.array([-np.conj(u[1]), np.conj(u[0])], dtype=complex)
        U[i], V[i] = u, v
        omegas[i] = omega_base * (1.0 if cell.sublattice[i] == 0 else
                                   0.9 if cell.sublattice[i] == 1 else -1.0)
    return U, V, omegas


# ============================================================================
# INTRA-CELL OPERATORS
# ============================================================================

def build_intra_cell_M(cell, U, V, xi=XI):
    """Static R/R-bar merger: PURELY IMAGINARY."""
    N = cell.N
    M = np.zeros((N, N), dtype=complex)
    for i in range(N):
        M[i, i] = np.real(np.vdot(U[i], V[i]))
        for j in range(i + 1, N):
            J = np.exp(-abs(cell.sites[i] - cell.sites[j]) / xi)
            coupling = J * (np.vdot(U[i], V[j]) + np.vdot(V[i], U[j]))
            M[i, j] = coupling
            M[j, i] = np.conj(coupling)
    return M


def build_intra_cell_torsion(cell, U, V, xi=XI):
    """
    TORSION CHANNEL: cross-gate counter-rotation term.

    H_cross[i,j] = J(r) * (<u_i|u_j> - <v_i|v_j>)

    This is the Hamiltonian representation of the asymmetric cross-gate:
      Cf rotates u by +theta, Ci rotates v by -theta.
    The generator of this rotation couples forward-forward and inverse-inverse
    with OPPOSITE signs. By Noether, the conserved current is:
      <u|u> - <v|v> = torsion current

    This is COMPLEX (has real and imaginary parts) because <u_i|u_j>
    is a generic inner product of SU(2) spinors from different lattice sites.
    """
    N = cell.N
    H = np.zeros((N, N), dtype=complex)
    for i in range(N):
        # Diagonal: <u_i|u_i> - <v_i|v_i> = 1 - 1 = 0 always
        # (both normalised) so no diagonal torsion
        for j in range(i + 1, N):
            J = np.exp(-abs(cell.sites[i] - cell.sites[j]) / xi)
            uu = np.vdot(U[i], U[j])
            vv = np.vdot(V[i], V[j])
            torsion = J * (uu - vv)
            H[i, j] = torsion
            H[j, i] = np.conj(torsion)
    return H


# ============================================================================
# INTER-CELL TUNNEL (same as before)
# ============================================================================

def find_boundary_pairs(cell_A, cell_B, threshold=1.15):
    pairs = []
    for i in cell_A.boundary_indices:
        for j in cell_B.boundary_indices:
            if abs(cell_A.sites[i] - cell_B.sites[j]) < threshold:
                pairs.append((i, j))
    return pairs

def build_tunnel_T(cell_A, cell_B, U_A, V_A, U_B, V_B,
                   omegas_A, omegas_B, J_tunnel=1.0, xi=XI):
    N_A, N_B = cell_A.N, cell_B.N
    T = np.zeros((N_A, N_B), dtype=complex)
    sigma_res = 0.1
    for i_A, j_B in find_boundary_pairs(cell_A, cell_B):
        r = abs(cell_A.sites[i_A] - cell_B.sites[j_B])
        J = J_tunnel * np.exp(-r / xi)
        omega_sum = omegas_A[i_A] + omegas_B[j_B]
        resonance = np.exp(-omega_sum**2 / (2 * sigma_res**2))
        uv_cross = np.vdot(U_A[i_A], V_B[j_B])
        T[i_A, j_B] = J * resonance * uv_cross
    return T


# ============================================================================
# FULL HAMILTONIAN WITH BOTH TORSION COMPONENTS
# ============================================================================

def build_full_H(cells, spinors_list, kappa=0.3, lam=1.0, xi=XI):
    """
    H = Σ_i [M_i + κ*H_cross_i] + λ * Σ_{<ij>} T_ij

    kappa: intra-cell torsion channel strength (cross-gate)
    lam: inter-cell torsion tunnel strength
    """
    n_cells = len(cells)
    sizes = [c.N for c in cells]
    total = sum(sizes)
    offsets = np.cumsum([0] + sizes[:-1]).tolist()

    H = np.zeros((total, total), dtype=complex)

    # Diagonal blocks: M_i (imaginary) + kappa * H_cross_i (complex)
    for c_idx in range(n_cells):
        cell = cells[c_idx]
        U, V, om = spinors_list[c_idx]
        M = build_intra_cell_M(cell, U, V, xi)
        H_cross = build_intra_cell_torsion(cell, U, V, xi)
        o = offsets[c_idx]
        H[o:o+cell.N, o:o+cell.N] = M + kappa * H_cross

    # Off-diagonal: tunnel T_ij
    n_tunnels = 0
    for c_i in range(n_cells):
        for c_j in range(c_i + 1, n_cells):
            if abs(cells[c_i].sites[0] - cells[c_j].sites[0]) > 4.0:
                continue
            U_i, V_i, om_i = spinors_list[c_i]
            U_j, V_j, om_j = spinors_list[c_j]
            T = build_tunnel_T(cells[c_i], cells[c_j], U_i, V_i, U_j, V_j,
                               om_i, om_j, J_tunnel=lam, xi=xi)
            o_i, o_j = offsets[c_i], offsets[c_j]
            H[o_i:o_i+cells[c_i].N, o_j:o_j+cells[c_j].N] = T
            H[o_j:o_j+cells[c_j].N, o_i:o_i+cells[c_i].N] = T.conj().T
            if np.any(np.abs(T) > 1e-12):
                n_tunnels += 1

    H = (H + H.conj().T) / 2.0
    return H, n_tunnels


# ============================================================================
# SPECTRAL TOOLS (same as before)
# ============================================================================

def unfold_spectrum(eigenvalues, poly_degree=10):
    evals = np.sort(eigenvalues)
    N = len(evals)
    if N < 5:
        spacings = np.diff(evals)
        if len(spacings) > 0 and np.mean(spacings) > 0:
            spacings /= np.mean(spacings)
        return evals, spacings
    cdf = np.arange(1, N + 1) / N
    deg = min(poly_degree, max(3, N // 10))
    coeffs = np.polyfit(evals, cdf * N, deg)
    N_smooth = np.polyval(coeffs, evals)
    spacings = np.diff(N_smooth)
    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s
    return N_smooth, spacings

def fit_beta(spacings):
    pos = spacings[spacings > 0.02]
    if len(pos) < 8:
        pos = spacings[spacings > 0.005]
    if len(pos) < 5:
        return 0.0, 0.0
    small = pos[pos < np.percentile(pos, 40)]
    if len(small) < 5:
        small = pos[pos < np.median(pos)]
    if len(small) < 3:
        return 0.0, 0.0
    n_bins = max(5, len(small) // 3)
    hist, edges = np.histogram(small, bins=n_bins, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    mask = hist > 0
    if np.sum(mask) < 3:
        return 0.0, 0.0
    log_s = np.log(centres[mask])
    log_p = np.log(hist[mask])
    coeffs = np.polyfit(log_s, log_p, 1)
    beta = coeffs[0]
    y_pred = np.polyval(coeffs, log_s)
    ss_res = np.sum((log_p - y_pred)**2)
    ss_tot = np.sum((log_p - np.mean(log_p))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return beta, r2


# ============================================================================
# MEASUREMENT
# ============================================================================

def measure(cells, spinors_list, kappa, lam):
    H, n_tun = build_full_H(cells, spinors_list, kappa=kappa, lam=lam)
    N = H.shape[0]

    re_n = np.linalg.norm(np.real(H))
    im_n = np.linalg.norm(np.imag(H))
    h_n = np.linalg.norm(H)
    tau = re_n / h_n if h_n > 0 else 0
    reim = re_n / im_n if im_n > 0 else np.inf

    evals = eigvalsh(H)
    bw = evals[-1] - evals[0]
    pos = evals[evals > 1e-12]
    pf = len(pos) / N
    gap = pos[0] if len(pos) > 0 else 0
    gbw = gap / bw if bw > 0 else 0

    _, spacings = unfold_spectrum(evals)
    ps = spacings[spacings > 0]
    beta, beta_r2 = fit_beta(ps)

    return {
        'kappa': kappa, 'lambda': lam, 'N_cells': len(cells), 'N_total': N,
        'n_tunnels': n_tun, 'tau': tau, 'reim': reim,
        're_norm': re_n, 'im_norm': im_n,
        'bw': bw, 'gap': gap, 'gap_bw': gbw, 'pos_frac': pf,
        'beta': beta, 'beta_r2': beta_r2,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 76)
    print("TORSION CHANNEL + TUNNEL: Two-Component Scaling Test")
    print("=" * 76)

    # First: verify that H_cross is NOT purely imaginary
    print("\n--- Verification: H_cross has real component ---")
    cell = HexagonalCell(0, 0, 0)
    U, V, om = assign_cell_spinors(cell)
    M = build_intra_cell_M(cell, U, V)
    Hc = build_intra_cell_torsion(cell, U, V)

    print(f"  M (static):  Re = {np.linalg.norm(np.real(M)):.6f}, Im = {np.linalg.norm(np.imag(M)):.6f}")
    print(f"  H_cross:     Re = {np.linalg.norm(np.real(Hc)):.6f}, Im = {np.linalg.norm(np.imag(Hc)):.6f}")
    print(f"  M purely imaginary: {np.linalg.norm(np.real(M)) < 1e-10}")
    print(f"  H_cross has Re:     {np.linalg.norm(np.real(Hc)) > 1e-10}")

    # Decompose H_cross
    re_hc = np.linalg.norm(np.real(Hc))
    im_hc = np.linalg.norm(np.imag(Hc))
    print(f"  H_cross Re/Im = {re_hc/im_hc:.4f}" if im_hc > 0 else "  H_cross purely real")
    print(f"  H_cross Re fraction = {re_hc / np.linalg.norm(Hc):.4f}")

    # ========================================================================
    # SWEEP 1: kappa (channel) vs lambda (tunnel) at fixed N=7
    # ========================================================================

    print("\n" + "=" * 76)
    print("SWEEP 1: κ (channel) × λ (tunnel) at N=7 cells")
    print("=" * 76)

    N_fix = 7
    cells = build_multi_cell_lattice(N_fix)
    spinors = [assign_cell_spinors(c, omega_base=1.0 if c.cell_id % 2 == 0 else -1.0)
               for c in cells]

    kappas = [0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    lambdas = [0, 0.1, 0.5, 1.0, 2.0, 5.0]

    grid_tau = np.zeros((len(kappas), len(lambdas)))
    grid_beta = np.zeros((len(kappas), len(lambdas)))
    grid_reim = np.zeros((len(kappas), len(lambdas)))

    print(f"\n  {'κ':>5} {'λ':>5} {'τ':>8} {'Re/Im':>8} {'β':>8} {'pos_f':>7}")
    print("  " + "-" * 55)

    for i, kp in enumerate(kappas):
        for j, lm in enumerate(lambdas):
            r = measure(cells, spinors, kappa=kp, lam=lm)
            grid_tau[i, j] = r['tau']
            grid_beta[i, j] = r['beta']
            grid_reim[i, j] = r['reim']
            print(f"  {kp:5.1f} {lm:5.1f} {r['tau']:8.4f} {r['reim']:8.4f} {r['beta']:8.3f} {r['pos_frac']:7.3f}")

    # ========================================================================
    # SWEEP 2: N scaling with BOTH channel and tunnel at kappa=0.3, lambda=1
    # ========================================================================

    print("\n" + "=" * 76)
    print("SWEEP 2: N scaling with κ=0.3 (channel) + λ=1.0 (tunnel)")
    print("=" * 76)

    N_cells_list = [1, 2, 3, 7, 19, 37]
    kp_fix = 0.3
    lm_fix = 1.0

    print(f"\n  {'N':>4} {'N_tot':>6} {'τ':>8} {'Re/Im':>8} {'β':>8} {'gap/bw':>8} {'pos_f':>7} {'#tun':>5}")
    print("  " + "-" * 65)

    n_scaling = []
    for nc in N_cells_list:
        cells_n = build_multi_cell_lattice(nc)
        sp_n = [assign_cell_spinors(c, omega_base=1.0 if c.cell_id % 2 == 0 else -1.0)
                for c in cells_n]
        r = measure(cells_n, sp_n, kappa=kp_fix, lam=lm_fix)
        n_scaling.append(r)
        print(f"  {nc:4d} {r['N_total']:6d} {r['tau']:8.4f} {r['reim']:8.4f} "
              f"{r['beta']:8.3f} {r['gap_bw']:8.5f} {r['pos_frac']:7.3f} {r['n_tunnels']:5d}")

    # Compare: channel only (lambda=0), tunnel only (kappa=0), both
    print("\n" + "=" * 76)
    print("SWEEP 3: Channel-only vs Tunnel-only vs Both (N scaling)")
    print("=" * 76)

    configs = [
        ("Channel only (κ=0.3, λ=0)", 0.3, 0.0),
        ("Tunnel only  (κ=0,   λ=1)", 0.0, 1.0),
        ("Both         (κ=0.3, λ=1)", 0.3, 1.0),
        ("Strong both  (κ=1.0, λ=5)", 1.0, 5.0),
    ]

    for label, kp, lm in configs:
        print(f"\n  {label}:")
        print(f"  {'N':>4} {'τ':>8} {'Re/Im':>8} {'β':>8}")
        print("  " + "-" * 35)
        for nc in N_cells_list:
            cells_n = build_multi_cell_lattice(nc)
            sp_n = [assign_cell_spinors(c, omega_base=1.0 if c.cell_id % 2 == 0 else -1.0)
                    for c in cells_n]
            r = measure(cells_n, sp_n, kappa=kp, lam=lm)
            print(f"  {nc:4d} {r['tau']:8.4f} {r['reim']:8.4f} {r['beta']:8.3f}")

    # ========================================================================
    # KEY TEST: Does channel torsion scale with volume?
    # ========================================================================

    print("\n" + "=" * 76)
    print("KEY TEST: Torsion scaling — channel (volume) vs tunnel (surface)")
    print("=" * 76)

    print(f"\n  {'N':>4} {'τ_chan':>8} {'τ_tun':>8} {'τ_both':>8} "
          f"{'Δτ=both-sum':>12} {'τ_chan/N':>9} {'τ_tun/Ntun':>10}")
    print("  " + "-" * 75)

    for nc in N_cells_list:
        cells_n = build_multi_cell_lattice(nc)
        sp_n = [assign_cell_spinors(c, omega_base=1.0 if c.cell_id % 2 == 0 else -1.0)
                for c in cells_n]

        r_chan = measure(cells_n, sp_n, kappa=0.3, lam=0.0)
        r_tun = measure(cells_n, sp_n, kappa=0.0, lam=1.0)
        r_both = measure(cells_n, sp_n, kappa=0.3, lam=1.0)

        tc = r_chan['tau']
        tt = r_tun['tau']
        tb = r_both['tau']
        delta = tb - (tc + tt)  # nonlinear interaction term

        tc_per_cell = tc / nc
        tt_per_tun = tt / r_tun['n_tunnels'] if r_tun['n_tunnels'] > 0 else 0

        print(f"  {nc:4d} {tc:8.4f} {tt:8.4f} {tb:8.4f} "
              f"{delta:12.4f} {tc_per_cell:9.5f} {tt_per_tun:10.5f}")

    # ========================================================================
    # EFFECTIVE COUPLING WITH BOTH COMPONENTS
    # ========================================================================

    print("\n" + "=" * 76)
    print("EFFECTIVE COUPLING: α_eff from channel + tunnel torsion")
    print("=" * 76)

    print(f"\n  Model: α_eff = α_EM / (1 - τ_total)")
    print(f"  Known: α_EM=1/137, α_W=1/30, α_S=0.118")
    print()

    # Try different (kappa, lambda) to see if we can reach known values
    test_configs = [
        ("Single merkabit, no torsion", 1, 0.0, 0.0),
        ("1 cell, channel only κ=0.3", 1, 0.3, 0.0),
        ("1 cell, channel strong κ=1", 1, 1.0, 0.0),
        ("7 cells, both κ=0.3 λ=1", 7, 0.3, 1.0),
        ("7 cells, strong κ=1 λ=5", 7, 1.0, 5.0),
        ("19 cells, both κ=0.3 λ=1", 19, 0.3, 1.0),
        ("19 cells, strong κ=1 λ=5", 19, 1.0, 5.0),
        ("37 cells, both κ=0.3 λ=1", 37, 0.3, 1.0),
        ("37 cells, strong κ=1 λ=5", 37, 1.0, 5.0),
        ("37 cells, max κ=5 λ=10", 37, 5.0, 10.0),
    ]

    print(f"  {'Config':>38} {'τ':>7} {'α_eff':>10} {'α⁻¹':>8} {'match':>10}")
    print("  " + "-" * 80)

    for label, nc, kp, lm in test_configs:
        cells_n = build_multi_cell_lattice(nc)
        sp_n = [assign_cell_spinors(c, omega_base=1.0 if c.cell_id % 2 == 0 else -1.0)
                for c in cells_n]
        r = measure(cells_n, sp_n, kappa=kp, lam=lm)
        tau = r['tau']
        if tau < 0.999:
            a_eff = ALPHA_EM / (1 - tau)
            a_inv = 1.0 / a_eff
        else:
            a_eff = np.inf
            a_inv = 0

        match = ""
        if abs(a_inv - 137) < 5: match = "≈ EM"
        elif abs(a_inv - 30) < 10: match = "≈ Weak"
        elif a_inv < 15: match = "≈ Strong"

        print(f"  {label:>38} {tau:7.4f} {a_eff:10.6f} {a_inv:8.2f} {match:>10}")

    # ========================================================================
    # PLOTTING
    # ========================================================================

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Torsion Channel + Tunnel: Two-Component Scaling',
                 fontsize=14, fontweight='bold')

    # (a) κ-λ phase diagram of torsion fraction
    ax = axes[0, 0]
    im = ax.imshow(grid_tau, aspect='auto', origin='lower', cmap='viridis',
                   extent=[-0.5, len(lambdas)-0.5, -0.5, len(kappas)-0.5])
    ax.set_xticks(range(len(lambdas))); ax.set_xticklabels(lambdas)
    ax.set_yticks(range(len(kappas))); ax.set_yticklabels(kappas)
    ax.set_xlabel('$\\lambda$ (tunnel)'); ax.set_ylabel('$\\kappa$ (channel)')
    ax.set_title('(a) Torsion $\\tau(\\kappa, \\lambda)$ at N=7')
    plt.colorbar(im, ax=ax)
    for i in range(len(kappas)):
        for j in range(len(lambdas)):
            ax.text(j, i, f'{grid_tau[i,j]:.3f}', ha='center', va='center', fontsize=7,
                    color='white' if grid_tau[i,j] > 0.3 else 'black')

    # (b) κ-λ phase diagram of beta
    ax = axes[0, 1]
    im = ax.imshow(grid_beta, aspect='auto', origin='lower', cmap='RdYlBu_r',
                   extent=[-0.5, len(lambdas)-0.5, -0.5, len(kappas)-0.5],
                   vmin=-1, vmax=2)
    ax.set_xticks(range(len(lambdas))); ax.set_xticklabels(lambdas)
    ax.set_yticks(range(len(kappas))); ax.set_yticklabels(kappas)
    ax.set_xlabel('$\\lambda$ (tunnel)'); ax.set_ylabel('$\\kappa$ (channel)')
    ax.set_title('(b) Level repulsion $\\beta(\\kappa, \\lambda)$ at N=7')
    plt.colorbar(im, ax=ax)
    for i in range(len(kappas)):
        for j in range(len(lambdas)):
            ax.text(j, i, f'{grid_beta[i,j]:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if grid_beta[i,j] > 0.5 else 'black')

    # (c) Channel vs tunnel vs both torsion scaling
    ax = axes[0, 2]
    ns = np.array([r['N_cells'] for r in n_scaling])
    taus = np.array([r['tau'] for r in n_scaling])
    ax.plot(ns, taus, 'ko-', ms=8, lw=2, label='Both (κ=0.3, λ=1)')

    # Also plot channel-only and tunnel-only
    taus_chan = []
    taus_tun = []
    for nc in N_cells_list:
        cells_n = build_multi_cell_lattice(nc)
        sp_n = [assign_cell_spinors(c, omega_base=1.0 if c.cell_id % 2 == 0 else -1.0)
                for c in cells_n]
        r_c = measure(cells_n, sp_n, kappa=0.3, lam=0.0)
        r_t = measure(cells_n, sp_n, kappa=0.0, lam=1.0)
        taus_chan.append(r_c['tau'])
        taus_tun.append(r_t['tau'])

    ax.plot(N_cells_list, taus_chan, 'bs--', ms=6, lw=1.5, label='Channel only (κ=0.3)')
    ax.plot(N_cells_list, taus_tun, 'r^--', ms=6, lw=1.5, label='Tunnel only (λ=1)')
    ax.set_xlabel('Number of cells N')
    ax.set_ylabel('Torsion fraction $\\tau$')
    ax.set_title('(c) Channel vs tunnel scaling')
    ax.legend(fontsize=8)

    # (d) Effective coupling force hierarchy
    ax = axes[1, 0]
    # Strong config
    alpha_effs = []
    for nc in N_cells_list:
        cells_n = build_multi_cell_lattice(nc)
        sp_n = [assign_cell_spinors(c, omega_base=1.0 if c.cell_id % 2 == 0 else -1.0)
                for c in cells_n]
        r = measure(cells_n, sp_n, kappa=1.0, lam=5.0)
        tau = r['tau']
        alpha_effs.append(ALPHA_EM / (1 - tau) if tau < 0.999 else 1.0)
    alpha_effs = np.array(alpha_effs)

    ax.semilogy(N_cells_list, alpha_effs, 'ro-', ms=8, lw=2, label='κ=1, λ=5')

    # Medium config
    alpha_med = []
    for nc in N_cells_list:
        cells_n = build_multi_cell_lattice(nc)
        sp_n = [assign_cell_spinors(c, omega_base=1.0 if c.cell_id % 2 == 0 else -1.0)
                for c in cells_n]
        r = measure(cells_n, sp_n, kappa=0.3, lam=1.0)
        tau = r['tau']
        alpha_med.append(ALPHA_EM / (1 - tau) if tau < 0.999 else 1.0)
    alpha_med = np.array(alpha_med)
    ax.semilogy(N_cells_list, alpha_med, 'bo-', ms=6, lw=1.5, label='κ=0.3, λ=1')

    ax.axhline(ALPHA_EM, ls='--', color='blue', alpha=0.5, label='$\\alpha_{EM}$')
    ax.axhline(ALPHA_W, ls='--', color='green', alpha=0.5, label='$\\alpha_W$')
    ax.axhline(ALPHA_S, ls='--', color='red', alpha=0.5, label='$\\alpha_S$')
    ax.set_xlabel('N cells'); ax.set_ylabel('$\\alpha_{eff}$')
    ax.set_title('(d) Force hierarchy')
    ax.legend(fontsize=7)

    # (e) Re/Im ratio scaling
    ax = axes[1, 1]
    for label, kp, lm, color, marker in [
        ('κ=0.3, λ=0', 0.3, 0.0, 'blue', 's'),
        ('κ=0, λ=1', 0.0, 1.0, 'red', '^'),
        ('κ=0.3, λ=1', 0.3, 1.0, 'black', 'o'),
        ('κ=1, λ=5', 1.0, 5.0, 'green', 'D'),
    ]:
        reims = []
        for nc in N_cells_list:
            cells_n = build_multi_cell_lattice(nc)
            sp_n = [assign_cell_spinors(c, omega_base=1.0 if c.cell_id % 2 == 0 else -1.0)
                    for c in cells_n]
            r = measure(cells_n, sp_n, kappa=kp, lam=lm)
            reims.append(r['reim'])
        ax.plot(N_cells_list, reims, f'{marker}-', color=color, ms=6, lw=1.5, label=label)
    ax.axhline(1.0, ls=':', color='gray', alpha=0.5)
    ax.set_xlabel('N cells'); ax.set_ylabel('Re/Im ratio')
    ax.set_title('(e) Re/Im scaling by config')
    ax.legend(fontsize=7)

    # (f) Channel torsion per cell (extensivity test)
    ax = axes[1, 2]
    taus_chan_per = [t/nc for t, nc in zip(taus_chan, N_cells_list)]
    ax.plot(N_cells_list, taus_chan_per, 'bs-', ms=8, lw=2)
    ax.set_xlabel('N cells')
    ax.set_ylabel('$\\tau_{channel}$ / N')
    ax.set_title('(f) Channel torsion per cell (extensivity)')
    ax.axhline(taus_chan_per[-1], ls='--', color='gray', alpha=0.5,
               label=f'asymptote ≈ {taus_chan_per[-1]:.5f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT / 'torsion_channel_tunnel.png', dpi=200, bbox_inches='tight')
    plt.savefig(OUT / 'torsion_channel_tunnel.pdf', bbox_inches='tight')
    print(f"\nFigures saved to {OUT}")


if __name__ == '__main__':
    main()
