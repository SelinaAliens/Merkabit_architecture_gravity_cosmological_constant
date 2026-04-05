#!/usr/bin/env python3
"""
TORSION TUNNEL SCALING TEST — Force Hierarchy from N-Cell Lattice
=================================================================

Hypothesis (Selina Stenberg, March 31 2026):
  - Single merkabit: intra-cell M purely imaginary → GOE → electromagnetic (α_EM ~ 1/137)
  - Tunnel T_AB: cross-chirality, complex → breaks time-reversal → weak force analog
  - N-cell lattice: accumulated torsion from many tunnels → strong coupling

Test: measure torsion fraction τ(N), beta(N), and effective coupling α_eff(N)
as functions of cell count N and tunnel strength λ.

Reuses infrastructure from tunnel_operator_simulation.py.
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
from matplotlib.colors import LogNorm

OUT = Path(r"C:\Users\selin\merkabit_results\torsion_scaling")
OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ============================================================================
# ARCHITECTURE CONSTANTS
# ============================================================================
RANK_E6 = 6
DIM_E6 = 78
COXETER_H = 12
XI = 3.0
F_RETURN = 0.696778
ALPHA_EM_INV = 137.035999083  # fine structure constant inverse
OMEGA_EISEN = np.exp(2j * np.pi / 3)
UNIT_VECTORS_AB = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]

# Known coupling constants (PDG values)
ALPHA_EM = 1.0 / ALPHA_EM_INV       # ~ 0.00730
ALPHA_W  = 1.0 / 29.5               # ~ 0.0339 (at M_Z scale)
ALPHA_S  = 0.1179                    # at M_Z = 91.2 GeV (PDG 2024)

# ============================================================================
# CELL & SPINOR INFRASTRUCTURE (from tunnel_operator_simulation.py)
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


def build_intra_cell_M(cell, U, V, xi=XI):
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


def build_full_hamiltonian(cells, spinors_list, J_tunnel=1.0, xi=XI):
    n_cells = len(cells)
    sizes = [c.N for c in cells]
    total = sum(sizes)
    offsets = np.cumsum([0] + sizes[:-1]).tolist()

    H = np.zeros((total, total), dtype=complex)

    for c_idx in range(n_cells):
        U, V, om = spinors_list[c_idx]
        M = build_intra_cell_M(cells[c_idx], U, V, xi)
        o = offsets[c_idx]
        H[o:o+cells[c_idx].N, o:o+cells[c_idx].N] = M

    n_tunnels = 0
    for c_i in range(n_cells):
        for c_j in range(c_i + 1, n_cells):
            if abs(cells[c_i].sites[0] - cells[c_j].sites[0]) > 4.0:
                continue
            U_i, V_i, om_i = spinors_list[c_i]
            U_j, V_j, om_j = spinors_list[c_j]
            T = build_tunnel_T(cells[c_i], cells[c_j], U_i, V_i, U_j, V_j,
                               om_i, om_j, J_tunnel=J_tunnel, xi=xi)
            o_i, o_j = offsets[c_i], offsets[c_j]
            H[o_i:o_i+cells[c_i].N, o_j:o_j+cells[c_j].N] = T
            H[o_j:o_j+cells[c_j].N, o_i:o_i+cells[c_i].N] = T.conj().T
            if np.any(np.abs(T) > 1e-12):
                n_tunnels += 1

    H = (H + H.conj().T) / 2.0
    return H, n_tunnels


# ============================================================================
# SPECTRAL ANALYSIS
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


def make_wigner_cdf(beta):
    a = 2.0 * gamma_fn((beta+2)/2)**(beta+1) / gamma_fn((beta+1)/2)**(beta+2)
    b = (gamma_fn((beta+2)/2) / gamma_fn((beta+1)/2))**2
    def cdf(s):
        if np.isscalar(s):
            val, _ = quad(lambda x: a * x**beta * np.exp(-b * x**2), 0, max(s, 0))
            return val
        return np.array([quad(lambda x: a * x**beta * np.exp(-b * x**2), 0, max(si, 0))[0] for si in s])
    return cdf


def ks_tests(spacings):
    pos = spacings[spacings > 0]
    if len(pos) < 5:
        return {'goe': (1, 0), 'gue': (1, 0), 'poi': (1, 0)}
    results = {}
    for name, beta_val in [('goe', 1), ('gue', 2)]:
        cdf = make_wigner_cdf(beta_val)
        ks, p = stats.kstest(pos, cdf)
        results[name] = (ks, p)
    ks_p, p_p = stats.kstest(pos, lambda s: 1 - np.exp(-s))
    results['poi'] = (ks_p, p_p)
    return results


# ============================================================================
# MAIN SCALING TEST
# ============================================================================

def measure_point(cells, spinors_list, lam):
    """Measure all observables at a single (N, lambda) point."""
    N_cells = len(cells)

    if N_cells == 1 or lam == 0:
        # Pure intra-cell
        if N_cells == 1:
            U, V, om = spinors_list[0]
            H = build_intra_cell_M(cells[0], U, V)
            n_tunnels = 0
        else:
            H, n_tunnels = build_full_hamiltonian(cells, spinors_list, J_tunnel=0)
    else:
        H, n_tunnels = build_full_hamiltonian(cells, spinors_list, J_tunnel=lam)

    N_total = H.shape[0]

    # Torsion measures
    re_norm = np.linalg.norm(np.real(H))
    im_norm = np.linalg.norm(np.imag(H))
    h_norm = np.linalg.norm(H)
    torsion_frac = re_norm / h_norm if h_norm > 0 else 0
    re_im_ratio = re_norm / im_norm if im_norm > 0 else np.inf

    # Eigenvalues
    evals = eigvalsh(H)

    # Spectral properties
    bandwidth = evals[-1] - evals[0]
    pos_evals = evals[evals > 1e-12]
    neg_evals = evals[evals < -1e-12]
    n_pos = len(pos_evals)
    pos_frac = n_pos / N_total

    # Spectral gap (smallest positive eigenvalue)
    gap = pos_evals[0] if len(pos_evals) > 0 else 0
    gap_bw_ratio = gap / bandwidth if bandwidth > 0 else 0

    # Level statistics
    _, spacings = unfold_spectrum(evals)
    pos_spacings = spacings[spacings > 0]

    beta, beta_r2 = fit_beta(pos_spacings)
    ks = ks_tests(pos_spacings)

    return {
        'N_cells': N_cells,
        'N_total': N_total,
        'lambda': lam,
        'n_tunnels': n_tunnels,
        'torsion_frac': torsion_frac,
        're_im_ratio': re_im_ratio,
        're_norm': re_norm,
        'im_norm': im_norm,
        'bandwidth': bandwidth,
        'gap': gap,
        'gap_bw': gap_bw_ratio,
        'pos_frac': pos_frac,
        'beta': beta,
        'beta_r2': beta_r2,
        'ks_goe': ks['goe'][0],
        'ks_gue': ks['gue'][0],
        'ks_poi': ks['poi'][0],
        'p_goe': ks['goe'][1],
        'p_gue': ks['gue'][1],
    }


def main():
    print("=" * 76)
    print("TORSION TUNNEL SCALING TEST — Force Hierarchy from N-Cell Lattice")
    print("=" * 76)

    # Grid
    N_cells_list = [1, 2, 3, 7, 19, 37]
    lambda_list = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    all_results = []

    for nc in N_cells_list:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  N = {nc} cells ({nc*7} eigenvalues)")
        print(f"{'='*60}")

        cells = build_multi_cell_lattice(nc)

        # Assign spinors (alternating omega_base by cell ID)
        spinors = []
        for c in cells:
            ob = 1.0 if c.cell_id % 2 == 0 else -1.0
            spinors.append(assign_cell_spinors(c, omega_base=ob))

        # Count total possible tunnels
        total_pairs = 0
        for i in range(nc):
            for j in range(i+1, nc):
                if abs(cells[i].sites[0] - cells[j].sites[0]) <= 4.0:
                    total_pairs += len(find_boundary_pairs(cells[i], cells[j]))

        print(f"  Total boundary pairs: {total_pairs}")

        print(f"\n  {'lam':>6} {'tau':>7} {'Re/Im':>7} {'beta':>7} {'R2':>6} "
              f"{'KS_GOE':>7} {'KS_GUE':>7} {'gap/bw':>7} {'pos_f':>6} {'#tun':>5}")
        print("  " + "-" * 80)

        for lam in lambda_list:
            r = measure_point(cells, spinors, lam)
            all_results.append(r)
            print(f"  {lam:6.1f} {r['torsion_frac']:7.4f} {r['re_im_ratio']:7.4f} "
                  f"{r['beta']:7.3f} {r['beta_r2']:6.3f} "
                  f"{r['ks_goe']:7.4f} {r['ks_gue']:7.4f} "
                  f"{r['gap_bw']:7.4f} {r['pos_frac']:6.3f} {r['n_tunnels']:5d}")

        dt = time.time() - t0
        print(f"  [{dt:.1f}s]")

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    print("\n" + "=" * 76)
    print("SCALING ANALYSIS")
    print("=" * 76)

    # Convert to arrays for analysis
    data = {k: np.array([r[k] for r in all_results]) for k in all_results[0]}

    # --- Test 1: Torsion fraction vs N at fixed lambda ---
    print("\n--- Test 1: Torsion fraction τ(N) at λ=1.0 ---")
    for nc in N_cells_list:
        mask = (data['N_cells'] == nc) & (np.isclose(data['lambda'], 1.0))
        if np.any(mask):
            tau = data['torsion_frac'][mask][0]
            nt = data['n_tunnels'][mask][0]
            print(f"  N={nc:3d}: τ = {tau:.5f}, n_tunnels = {nt}")

    # --- Test 2: Beta vs N at optimal lambda ---
    print("\n--- Test 2: Peak beta(N) across lambda sweep ---")
    for nc in N_cells_list:
        mask = data['N_cells'] == nc
        if np.any(mask):
            betas = data['beta'][mask]
            lams = data['lambda'][mask]
            best_idx = np.argmax(betas)
            print(f"  N={nc:3d}: peak β = {betas[best_idx]:.3f} at λ = {lams[best_idx]:.1f}")

    # --- Test 3: Spectral gap ratio ---
    print("\n--- Test 3: Gap/bandwidth ratio at λ=1.0 ---")
    for nc in N_cells_list:
        mask = (data['N_cells'] == nc) & (np.isclose(data['lambda'], 1.0))
        if np.any(mask):
            gbw = data['gap_bw'][mask][0]
            print(f"  N={nc:3d}: gap/bw = {gbw:.5f}  (1/3 = {1/3:.5f})")

    # --- Test 4: Effective coupling from torsion ---
    print("\n--- Test 4: Effective coupling α_eff(N) ---")
    print("  Model: α_eff = α_EM + τ × (torsion contribution)")
    print("  Alternative: α_eff = α_EM / (1 - τ)")
    print()
    print(f"  {'N':>4} {'τ(λ=1)':>8} {'α=αEM/(1-τ)':>12} {'α⁻¹':>8} {'β':>6} {'Target':>12}")

    for nc in N_cells_list:
        mask = (data['N_cells'] == nc) & (np.isclose(data['lambda'], 1.0))
        if np.any(mask):
            tau = data['torsion_frac'][mask][0]
            beta = data['beta'][mask][0]
            if tau < 1:
                alpha_eff = ALPHA_EM / (1 - tau)
                alpha_inv = 1.0 / alpha_eff
            else:
                alpha_eff = np.inf
                alpha_inv = 0.0

            target = ""
            if nc == 1:
                target = f"EM: 1/{ALPHA_EM_INV:.0f}"
            elif nc <= 3:
                target = f"Weak: 1/{1/ALPHA_W:.0f}"
            elif nc >= 7:
                target = f"Strong: 1/{1/ALPHA_S:.1f}"

            print(f"  {nc:4d} {tau:8.5f} {alpha_eff:12.6f} {alpha_inv:8.2f} {beta:6.3f} {target}")

    # --- Test 5: Beta scaling law ---
    print("\n--- Test 5: β(N) at λ=1.0 — scaling toward GUE ---")
    ns_test = []
    betas_test = []
    for nc in N_cells_list:
        mask = (data['N_cells'] == nc) & (np.isclose(data['lambda'], 1.0))
        if np.any(mask):
            ns_test.append(nc)
            betas_test.append(data['beta'][mask][0])
    ns_test = np.array(ns_test)
    betas_test = np.array(betas_test)

    if len(ns_test) > 3:
        # Fit: beta(N) = 2 - A * N^(-gamma)
        # Or: beta = a + b*log(N)
        mask_fit = ns_test > 1
        if np.sum(mask_fit) >= 3:
            log_n = np.log(ns_test[mask_fit])
            betas_f = betas_test[mask_fit]
            coeffs = np.polyfit(log_n, betas_f, 1)
            print(f"  Linear fit: β ≈ {coeffs[1]:.3f} + {coeffs[0]:.3f} × ln(N)")
            print(f"  β = 2 (GUE) predicted at N = {np.exp((2 - coeffs[1]) / coeffs[0]):.0f} cells")

    # --- Torsion per tunnel ---
    print("\n--- Torsion per tunnel ---")
    for nc in N_cells_list:
        mask = (data['N_cells'] == nc) & (np.isclose(data['lambda'], 1.0))
        if np.any(mask):
            tau = data['torsion_frac'][mask][0]
            nt = data['n_tunnels'][mask][0]
            tau_per = tau / nt if nt > 0 else 0
            print(f"  N={nc:3d}: τ/tunnel = {tau_per:.6f} (τ={tau:.5f}, tunnels={nt})")

    # --- Positive eigenvalue fraction and F-connection ---
    print("\n--- Positive eigenvalue fraction (F-connection: -ln(F) = 0.361) ---")
    for nc in N_cells_list:
        mask = (data['N_cells'] == nc) & (np.isclose(data['lambda'], 1.0))
        if np.any(mask):
            pf = data['pos_frac'][mask][0]
            print(f"  N={nc:3d}: pos_frac = {pf:.4f}  (−ln(F) = 0.3613)")

    # ========================================================================
    # PLOTTING
    # ========================================================================

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Torsion Tunnel Scaling: Force Hierarchy from N-Cell Lattice',
                 fontsize=14, fontweight='bold')

    colors_n = {1: 'black', 2: 'blue', 3: 'purple', 7: 'green', 19: 'orange', 37: 'red'}

    # (a) Torsion fraction vs lambda for each N
    ax = axes[0, 0]
    for nc in N_cells_list:
        mask = data['N_cells'] == nc
        ax.plot(data['lambda'][mask], data['torsion_frac'][mask],
                'o-', color=colors_n[nc], label=f'N={nc}', lw=1.5, ms=5)
    ax.set_xlabel('Tunnel strength $\\lambda$')
    ax.set_ylabel('Torsion fraction $\\tau = ||\\mathrm{Re}(H)|| / ||H||$')
    ax.set_title('(a) Torsion fraction')
    ax.legend(fontsize=8)

    # (b) Beta vs lambda for each N
    ax = axes[0, 1]
    for nc in N_cells_list:
        mask = data['N_cells'] == nc
        ax.plot(data['lambda'][mask], data['beta'][mask],
                'o-', color=colors_n[nc], label=f'N={nc}', lw=1.5, ms=5)
    ax.axhline(1.0, ls='--', color='gray', alpha=0.5, label='GOE (β=1)')
    ax.axhline(2.0, ls='--', color='red', alpha=0.5, label='GUE (β=2)')
    ax.set_xlabel('Tunnel strength $\\lambda$')
    ax.set_ylabel('Level repulsion $\\beta$')
    ax.set_title('(b) Beta (GOE→GUE transition)')
    ax.legend(fontsize=7)

    # (c) Torsion fraction vs N at lambda=1
    ax = axes[0, 2]
    ns_plot, taus_plot, betas_plot = [], [], []
    for nc in N_cells_list:
        mask = (data['N_cells'] == nc) & (np.isclose(data['lambda'], 1.0))
        if np.any(mask):
            ns_plot.append(nc)
            taus_plot.append(data['torsion_frac'][mask][0])
            betas_plot.append(data['beta'][mask][0])
    ns_plot = np.array(ns_plot)
    taus_plot = np.array(taus_plot)
    betas_plot = np.array(betas_plot)

    ax.plot(ns_plot, taus_plot, 'ko-', ms=8, lw=2)
    ax.set_xlabel('Number of cells N')
    ax.set_ylabel('Torsion fraction $\\tau$ at $\\lambda$=1')
    ax.set_title('(c) Torsion scaling with N')
    for i, nc in enumerate(ns_plot):
        ax.annotate(f'N={nc}', (nc, taus_plot[i]), textcoords='offset points',
                    xytext=(5, 5), fontsize=8)

    # (d) Effective coupling (force hierarchy)
    ax = axes[1, 0]
    alpha_effs = []
    for i, nc in enumerate(ns_plot):
        if taus_plot[i] < 1:
            alpha_effs.append(ALPHA_EM / (1 - taus_plot[i]))
        else:
            alpha_effs.append(1.0)
    alpha_effs = np.array(alpha_effs)
    alpha_inv_effs = 1.0 / alpha_effs

    ax.semilogy(ns_plot, alpha_effs, 'ko-', ms=8, lw=2, label='$\\alpha_{eff}(N)$')
    ax.axhline(ALPHA_EM, ls='--', color='blue', lw=1.5, label=f'$\\alpha_{{EM}}$ = 1/{ALPHA_EM_INV:.0f}')
    ax.axhline(ALPHA_W, ls='--', color='green', lw=1.5, label=f'$\\alpha_{{W}}$ = 1/30')
    ax.axhline(ALPHA_S, ls='--', color='red', lw=1.5, label=f'$\\alpha_{{S}}$ = 0.118')
    ax.axhline(1.0, ls=':', color='gray', alpha=0.5, label='$\\alpha$ = 1')
    ax.set_xlabel('Number of cells N')
    ax.set_ylabel('Effective coupling $\\alpha_{eff}$')
    ax.set_title('(d) Force hierarchy from torsion scaling')
    ax.legend(fontsize=7, loc='upper left')
    for i, nc in enumerate(ns_plot):
        ax.annotate(f'1/{alpha_inv_effs[i]:.1f}', (nc, alpha_effs[i]),
                    textcoords='offset points', xytext=(5, -10), fontsize=8)

    # (e) Beta vs N at lambda=1
    ax = axes[1, 1]
    ax.plot(ns_plot, betas_plot, 'ko-', ms=8, lw=2)
    ax.axhline(1.0, ls='--', color='blue', alpha=0.5, label='GOE')
    ax.axhline(2.0, ls='--', color='red', alpha=0.5, label='GUE')
    ax.fill_between([0, 40], 0.85, 1.15, color='blue', alpha=0.1)
    ax.fill_between([0, 40], 1.85, 2.15, color='red', alpha=0.1)
    ax.set_xlabel('Number of cells N')
    ax.set_ylabel('$\\beta$ at $\\lambda$=1')
    ax.set_title('(e) Level repulsion scaling')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 40)

    # (f) Re/Im ratio vs N at lambda=1
    ax = axes[1, 2]
    re_ims = []
    for nc in N_cells_list:
        mask = (data['N_cells'] == nc) & (np.isclose(data['lambda'], 1.0))
        if np.any(mask):
            re_ims.append(data['re_im_ratio'][mask][0])
    re_ims = np.array(re_ims)
    ax.plot(ns_plot, re_ims, 'ko-', ms=8, lw=2)
    ax.axhline(1.0, ls='--', color='gray', alpha=0.5, label='Re = Im (balance)')
    ax.set_xlabel('Number of cells N')
    ax.set_ylabel('$||\\mathrm{Re}(H)|| / ||\\mathrm{Im}(H)||$ at $\\lambda$=1')
    ax.set_title('(f) Re/Im ratio scaling')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / 'torsion_scaling.png', dpi=200, bbox_inches='tight')
    plt.savefig(OUT / 'torsion_scaling.pdf', bbox_inches='tight')
    print(f"\nFigures saved to {OUT}")

    # ========================================================================
    # PHASE DIAGRAM HEATMAP
    # ========================================================================

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('Phase Diagrams: Torsion, Beta, Effective Coupling', fontsize=13)

    for ax, key, title, cmap in [
        (axes2[0], 'torsion_frac', 'Torsion τ', 'viridis'),
        (axes2[1], 'beta', 'Level repulsion β', 'RdYlBu_r'),
        (axes2[2], 're_im_ratio', 'Re/Im ratio', 'magma'),
    ]:
        grid = np.full((len(lambda_list), len(N_cells_list)), np.nan)
        for i, lam in enumerate(lambda_list):
            for j, nc in enumerate(N_cells_list):
                mask = (data['N_cells'] == nc) & (np.isclose(data['lambda'], lam))
                if np.any(mask):
                    grid[i, j] = data[key][mask][0]

        im = ax.imshow(grid, aspect='auto', origin='lower', cmap=cmap,
                       extent=[-0.5, len(N_cells_list)-0.5, -0.5, len(lambda_list)-0.5])
        ax.set_xticks(range(len(N_cells_list)))
        ax.set_xticklabels(N_cells_list)
        ax.set_yticks(range(len(lambda_list)))
        ax.set_yticklabels(lambda_list)
        ax.set_xlabel('N cells')
        ax.set_ylabel('$\\lambda$')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        # Annotate values
        for i in range(len(lambda_list)):
            for j in range(len(N_cells_list)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f'{grid[i,j]:.3f}', ha='center', va='center',
                            fontsize=6, color='white' if grid[i,j] > np.nanmedian(grid) else 'black')

    plt.tight_layout()
    plt.savefig(OUT / 'torsion_phase_diagram.png', dpi=200, bbox_inches='tight')
    print(f"Phase diagram saved.")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n" + "=" * 76)
    print("FINAL SUMMARY")
    print("=" * 76)

    print(f"""
TORSION TUNNEL SCALING TEST RESULTS
====================================

Architecture: HexagonalCell (7-node Eisenstein) × N cells
Tunnel: cross-chirality T_AB, forward(A) ↔ inverse(B)

TORSION FRACTION τ(N) at λ=1.0:""")
    for i, nc in enumerate(ns_plot):
        print(f"  N={nc:3d}: τ = {taus_plot[i]:.5f}")

    print(f"""
EFFECTIVE COUPLING α_eff(N) = α_EM / (1 - τ):""")
    for i, nc in enumerate(ns_plot):
        print(f"  N={nc:3d}: α_eff = {alpha_effs[i]:.6f}  (α⁻¹ = {alpha_inv_effs[i]:.2f})")

    print(f"""
KNOWN FORCE COUPLINGS:
  α_EM  = {ALPHA_EM:.6f}  (1/{ALPHA_EM_INV:.0f})
  α_W   = {ALPHA_W:.6f}  (1/30)
  α_S   = {ALPHA_S:.4f}      (1/8.5)

LEVEL REPULSION β(N) at λ=1.0:""")
    for i, nc in enumerate(ns_plot):
        print(f"  N={nc:3d}: β = {betas_plot[i]:.3f}")

    print()

    plt.show()


if __name__ == '__main__':
    main()
