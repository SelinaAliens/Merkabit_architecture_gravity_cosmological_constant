#!/usr/bin/env python3
"""
SIMULATION 14B: WEYL ANTI-GRAVITY IN THE OCTONIONIC ENVELOPE
==============================================================

The full merkabit has 8 octonion channels, each carrying a (u, v) spinor pair.
A Weyl electron doesn't break ALL 8 channels — it breaks a SUBSET determined
by the crystal symmetry.

The octonion multiplication table governs inter-channel coupling.
Non-associativity means the ORDER of channel interactions matters,
creating phase corrections to the gravitational coupling.

Key question: does the 8-channel structure AMPLIFY or SUPPRESS the
Weyl gravitational coupling modification found in Sim 14?

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
NUM_GATES = 5
STEP_PHASE = 2 * np.pi / COXETER_H
OUROBOROS_GATES = ['S', 'R', 'T', 'F', 'P']
J_COUPLING = 0.05
G_EFF = 0.2542
PHI = (1 + np.sqrt(5)) / 2
TOL = 1e-12

# ============================================================================
# OCTONION ALGEBRA
# ============================================================================

# The 8 octonion basis elements: e0 (real), e1-e7 (imaginary)
# Multiplication table: e_i * e_j = sign * e_k
# Encoded by the Fano plane: the 7 oriented lines of PG(2,2)

# Fano plane triples (i,j,k) such that e_i * e_j = e_k
# Each triple is a directed cycle: (i,j) -> k, (j,k) -> i, (k,i) -> j
FANO_TRIPLES = [
    (1, 2, 3),  # e1*e2 = e3
    (1, 4, 5),  # e1*e4 = e5
    (1, 7, 6),  # e1*e7 = e6 (note: e7*e1 = -e6)
    (2, 4, 6),  # e2*e4 = e6
    (2, 5, 7),  # e2*e5 = e7
    (3, 4, 7),  # e3*e4 = e7
    (3, 6, 5),  # e3*e6 = e5
]


def octonion_multiply_table():
    """Build the full 8x8 octonion multiplication table.
    Returns (product_index, sign) for each pair (i, j).
    """
    # Table[i][j] = (k, sign) means e_i * e_j = sign * e_k
    table = {}

    # e0 * e_j = e_j, e_i * e0 = e_i
    for i in range(8):
        table[(0, i)] = (i, +1)
        table[(i, 0)] = (i, +1)

    # e_i * e_i = -e_0 for i > 0
    for i in range(1, 8):
        table[(i, i)] = (0, -1)

    # Fano triples: (a,b,c) means e_a * e_b = +e_c
    for a, b, c in FANO_TRIPLES:
        table[(a, b)] = (c, +1)
        table[(b, a)] = (c, -1)  # anti-commutative
        table[(b, c)] = (a, +1)
        table[(c, b)] = (a, -1)
        table[(c, a)] = (b, +1)
        table[(a, c)] = (b, -1)

    return table


OCT_TABLE = octonion_multiply_table()


def oct_mult(x, y):
    """Multiply two octonions x, y (each length-8 array)."""
    result = np.zeros(8, dtype=complex)
    for i in range(8):
        for j in range(8):
            k, sign = OCT_TABLE[(i, j)]
            result[k] += sign * x[i] * y[j]
    return result


def oct_conj(x):
    """Octonion conjugate: negate imaginary parts."""
    xc = -x.copy()
    xc[0] = x[0]
    return xc


def oct_norm_sq(x):
    """Squared norm: x * conj(x)."""
    return np.real(np.sum(x * np.conj(x)))


# ============================================================================
# GATE INFRASTRUCTURE
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


def get_step_angles(step_index):
    theta = STEP_PHASE
    k = step_index
    absent = k % NUM_GATES
    p_angle = theta
    sym_base = theta / 3
    omega_k = 2 * np.pi * k / COXETER_H

    rx = sym_base * (1.0 + 0.5 * np.cos(omega_k))
    rz = sym_base * (1.0 + 0.5 * np.cos(omega_k + 2*np.pi/3))

    gl = OUROBOROS_GATES[absent]
    if gl == 'S':   rz *= 0.4; rx *= 1.3
    elif gl == 'R': rx *= 0.4; rz *= 1.3
    elif gl == 'T': rx *= 0.7; rz *= 0.7
    elif gl == 'P': p_angle *= 0.6; rx *= 1.8; rz *= 1.5

    return p_angle, rx, rz


# ============================================================================
# OCTONIONIC MERKABIT STATE
# ============================================================================

class OctonionMerkabit:
    """
    Full 8-channel merkabit: 8 × (u, v) spinor pairs.
    Channel 0 = scalar (real part), channels 1-7 = imaginary octonion units.
    Each channel carries an independent (u, v) dual-spinor.
    Inter-channel coupling is governed by the octonion multiplication table.
    """

    def __init__(self, n_channels=8):
        self.n = n_channels
        # Each channel: (u, v) spinor pair
        self.u = [np.array([1, 0], dtype=complex) for _ in range(n_channels)]
        self.v = [np.array([0, 1], dtype=complex) for _ in range(n_channels)]
        # Channel activity: 1.0 = bipartite, 0.0 = monopole (v=0)
        self.channel_active = np.ones(n_channels)

    def set_weyl_channels(self, weyl_mask):
        """Set which channels are in Weyl (monopole) state.
        weyl_mask: array of 8 booleans. True = Weyl (v=0)."""
        for i in range(self.n):
            if weyl_mask[i]:
                self.v[i] = np.zeros(2, dtype=complex)
                self.channel_active[i] = 0.0

    def normalize(self):
        for i in range(self.n):
            nu = np.linalg.norm(self.u[i])
            if nu > TOL: self.u[i] /= nu
            nv = np.linalg.norm(self.v[i])
            if nv > TOL: self.v[i] /= nv

    def total_coherence(self, other):
        """
        Octonionic coherence between two merkabit states.
        Uses the FULL octonion product structure, not just channel-by-channel.

        The coherence is the octonion inner product:
        C = Re(sum_channels u_self^dag . u_other * v_self^dag . v_other)
        weighted by the octonion multiplication table for inter-channel coupling.
        """
        # Channel-by-channel coherence (diagonal terms)
        c_diagonal = 0.0
        for i in range(self.n):
            c_uu = np.vdot(self.u[i], other.u[i])
            c_vv = np.vdot(self.v[i], other.v[i])
            c_diagonal += np.real(c_uu * c_vv)

        # Inter-channel coherence (off-diagonal, from octonion non-commutativity)
        # For each Fano triple (a,b,c): the product e_a * e_b = e_c creates
        # a coupling between channels a, b, and c.
        c_fano = 0.0
        for a, b, c in FANO_TRIPLES:
            # Forward: channel a × channel b -> contributes to channel c
            c_ab_u = np.vdot(self.u[a], other.u[b])
            c_ab_v = np.vdot(self.v[a], other.v[b])
            c_c_u = np.vdot(self.u[c], other.u[c])

            # The triple coupling: how much the product of channels a,b
            # is coherent with channel c
            triple_coh = np.real(c_ab_u * c_ab_v * np.conj(c_c_u))
            c_fano += triple_coh

            # Reverse: channel b × channel a -> -e_c (anti-commutative)
            # This gives the NON-ASSOCIATIVE correction
            c_ba_u = np.vdot(self.u[b], other.u[a])
            c_ba_v = np.vdot(self.v[b], other.v[a])
            triple_rev = np.real(c_ba_u * c_ba_v * np.conj(c_c_u))
            c_fano -= triple_rev  # anti-commutative: subtract

        # Total: diagonal + Fano correction / normalization
        c_total = c_diagonal / self.n + c_fano / (self.n * len(FANO_TRIPLES))

        return c_total


def ouroboros_step_octonion(state, step_index, J=J_COUPLING, vacuum=None):
    """
    One ouroboros step on the full 8-channel octonionic state.
    Each channel evolves independently under the gate sequence.
    Inter-channel coupling occurs through the Fano triple structure.
    """
    p, rx, rz = get_step_angles(step_index)

    for i in range(state.n):
        if state.channel_active[i] > 0.5:
            # Bipartite channel: full (u, v) evolution
            state.u[i] = gate_P_fwd(state.u[i], p)
            state.v[i] = gate_P_inv(state.v[i], p)
            state.u[i] = gate_Rz(state.u[i], rz)
            state.u[i] = gate_Rx(state.u[i], rx)
            state.v[i] = gate_Rz(state.v[i], rz)
            state.v[i] = gate_Rx(state.v[i], rx)
        else:
            # Weyl channel: u-only evolution
            state.u[i] = gate_P_fwd(state.u[i], p)
            state.u[i] = gate_Rz(state.u[i], rz)
            state.u[i] = gate_Rx(state.u[i], rx)
            # v stays at zero (or near-zero with lattice injection)

    # Inter-channel coupling through Fano triples
    # Each triple (a,b,c): channel c gets a kick from the product of a and b
    for a, b, c in FANO_TRIPLES:
        # The Fano coupling strength: J_fano = J / n_triples
        J_fano = J / (7 * state.n)

        # u-channel coupling: u_c += J_fano * (u_a . u_b)_proj
        u_ab = np.vdot(state.u[a], state.u[b]) * state.u[a]
        state.u[c] = state.u[c] + J_fano * u_ab

        # v-channel coupling (only if both a,b are bipartite)
        if state.channel_active[a] > 0.5 and state.channel_active[b] > 0.5:
            v_ab = np.vdot(state.v[a], state.v[b]) * state.v[a]
            if state.channel_active[c] > 0.5:
                state.v[c] = state.v[c] + J_fano * v_ab
            # If c is Weyl: v-injection from the Fano coupling
            # This is how the octonion structure LEAKS bipartite content
            # into Weyl channels through non-associative cross-talk
            else:
                state.v[c] = state.v[c] + J_fano * 0.1 * v_ab  # suppressed leak

    # Vacuum coupling (if provided)
    if vacuum is not None:
        for i in range(state.n):
            J_eff = J / (1 + 0)  # no r-dependence here, applied externally
            state.u[i] = state.u[i] + J_eff * vacuum.u[i]
            nu = np.linalg.norm(state.u[i])
            if nu > TOL: state.u[i] /= nu

            if state.channel_active[i] > 0.5:
                state.v[i] = state.v[i] + J_eff * vacuum.v[i]
            else:
                # Weyl channel: vacuum tries to inject v, but damped
                state.v[i] = state.v[i] + J_eff * 0.1 * vacuum.v[i]
                state.v[i] *= 0.95  # partial v-leak back to vacuum

            nv = np.linalg.norm(state.v[i])
            if nv > TOL: state.v[i] /= nv

    state.normalize()
    return state


# ============================================================================
# CRYSTAL SYMMETRY MASKS
# ============================================================================

def get_weyl_mask(crystal_type):
    """
    Which octonion channels are in Weyl (monopole) state for each crystal?

    The crystal symmetry determines which channels break chirality.
    TaAs (C4v, h_eff=8): 4 of 8 channels are Weyl (half-monopole)
    WTe2 (C2v, h_eff=4): 2 of 8 channels are Weyl (quarter-monopole)
    AlCuFe (icosahedral, h_eff=10): 5 of 8 channels (Fano+scalar)
    Full monopole: all 7 imaginary channels are Weyl, scalar bipartite
    """
    masks = {
        'full_monopole': [False, True, True, True, True, True, True, True],
        'TaAs':          [False, True, False, True, False, True, False, True],  # C4: alternate
        'WTe2':          [False, True, True, False, False, False, False, False],  # C2: first pair
        'AlCuFe':        [False, True, True, True, True, True, False, False],  # icosa: 5 of 7
        'single_channel':[False, True, False, False, False, False, False, False],  # minimal
        'bipartite':     [False, False, False, False, False, False, False, False],  # all normal
    }
    return np.array(masks[crystal_type], dtype=bool)


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def measure_octonionic_gravity(crystal_type, r_values, n_settle=80, n_measure=80):
    """
    Measure the gravitational coupling of an octonionic Weyl state.
    Source: normal octonionic merkabit (all channels bipartite).
    Probe: Weyl octonionic merkabit (some channels monopole).
    """
    weyl_mask = get_weyl_mask(crystal_type)
    n_weyl = np.sum(weyl_mask)
    n_bip = 8 - n_weyl

    # Source: settled bipartite
    source = OctonionMerkabit()
    for cycle in range(n_settle):
        for k in range(COXETER_H):
            ouroboros_step_octonion(source, k)

    # Reference: bipartite probe (no Weyl channels)
    C_normal_arr = []
    for r in r_values:
        J_eff = J_COUPLING / r
        probe_n = OctonionMerkabit()
        src_n = OctonionMerkabit()
        # Copy settled source
        for i in range(8):
            src_n.u[i] = source.u[i].copy()
            src_n.v[i] = source.v[i].copy()

        c_acc = []
        for cycle in range(n_measure):
            for k in range(COXETER_H):
                ouroboros_step_octonion(src_n, k)
                ouroboros_step_octonion(probe_n, k)
                # Coupling
                for i in range(8):
                    probe_n.u[i] = probe_n.u[i] + J_eff * src_n.u[i]
                    probe_n.v[i] = probe_n.v[i] + J_eff * src_n.v[i]
                probe_n.normalize()

            c_acc.append(probe_n.total_coherence(src_n))

        C_normal_arr.append(np.mean(c_acc))

    # Weyl probe
    C_weyl_arr = []
    for r in r_values:
        J_eff = J_COUPLING / r
        probe_w = OctonionMerkabit()
        probe_w.set_weyl_channels(weyl_mask)
        src_w = OctonionMerkabit()
        for i in range(8):
            src_w.u[i] = source.u[i].copy()
            src_w.v[i] = source.v[i].copy()

        c_acc = []
        for cycle in range(n_measure):
            for k in range(COXETER_H):
                ouroboros_step_octonion(src_w, k)
                ouroboros_step_octonion(probe_w, k)
                for i in range(8):
                    probe_w.u[i] = probe_w.u[i] + J_eff * src_w.u[i]
                    nu = np.linalg.norm(probe_w.u[i])
                    if nu > TOL: probe_w.u[i] /= nu

                    if probe_w.channel_active[i] > 0.5:
                        probe_w.v[i] = probe_w.v[i] + J_eff * src_w.v[i]
                    else:
                        # Weyl channel: suppressed v-coupling
                        probe_w.v[i] = probe_w.v[i] + J_eff * 0.1 * src_w.v[i]
                        probe_w.v[i] *= 0.95
                    nv = np.linalg.norm(probe_w.v[i])
                    if nv > TOL: probe_w.v[i] /= nv

            c_acc.append(probe_w.total_coherence(src_w))

        C_weyl_arr.append(np.mean(c_acc))

    return np.array(C_normal_arr), np.array(C_weyl_arr), n_weyl


def main():
    t_start = time.time()

    print("""
================================================================
  SIMULATION 14B: WEYL ANTI-GRAVITY IN THE OCTONIONIC ENVELOPE
  8-channel merkabit with Fano plane inter-channel coupling
================================================================
""")

    r_values = np.array([1, 2, 3, 5, 8])

    # Test all crystal types
    crystal_types = ['bipartite', 'single_channel', 'WTe2', 'TaAs', 'AlCuFe', 'full_monopole']

    all_results = {}

    for ctype in crystal_types:
        mask = get_weyl_mask(ctype)
        n_weyl = np.sum(mask)
        print(f"  --- {ctype} ({n_weyl}/8 Weyl channels) ---")
        print(f"  Mask: {mask.astype(int)}")

        C_n, C_w, nw = measure_octonionic_gravity(ctype, r_values, n_settle=60, n_measure=60)

        print(f"  {'r':>4}  {'C_normal':>10}  {'C_weyl':>10}  {'Ratio':>10}  {'G_ratio':>10}")
        G_ratios = []
        for i, r in enumerate(r_values):
            cn = C_n[i]; cw = C_w[i]
            ratio = cw / cn if abs(cn) > TOL else 0
            G_r = ratio**2 if ratio >= 0 else -(ratio**2)
            G_ratios.append(G_r)
            print(f"  {r:4d}  {cn:10.6f}  {cw:10.6f}  {ratio:10.6f}  {G_r:10.6f}")

        mean_G = np.mean(G_ratios)
        all_results[ctype] = {
            'n_weyl': nw,
            'C_normal': C_n,
            'C_weyl': C_w,
            'G_ratio': mean_G,
            'G_ratios': G_ratios,
        }
        print(f"  Mean G_Weyl/G_normal = {mean_G:.6f}")
        print()

    # ================================================================
    # COMPARISON: Octonionic vs Single-Channel
    # ================================================================
    print("=" * 76)
    print("  COMPARISON: OCTONIONIC ENVELOPE vs SINGLE-CHANNEL")
    print("=" * 76)

    # Sim 14 (single channel) gave G_Weyl/G_normal = 0.897
    G_sim14 = 0.897

    print(f"\n  {'Crystal':<18} {'Weyl/8':>6} {'G_oct/G_norm':>12} {'vs Sim14':>10} {'Enhancement':>12}")
    for ctype in crystal_types:
        res = all_results[ctype]
        g = res['G_ratio']
        nw = res['n_weyl']
        vs_sim14 = g / G_sim14 if G_sim14 != 0 else 0
        # Enhancement: how much more/less effect than single-channel
        # The single-channel had delta_G/G = -0.103
        # With octonionic: delta_G/G = (g - 1)
        delta_oct = g - 1
        delta_single = G_sim14 - 1
        enhance = delta_oct / delta_single if delta_single != 0 else 0
        print(f"  {ctype:<18} {nw:4d}/8  {g:12.6f}  {vs_sim14:10.4f}  {enhance:12.4f}x")

    # ================================================================
    # THE KEY QUESTION: NON-ASSOCIATIVE AMPLIFICATION
    # ================================================================
    print(f"\n{'=' * 76}")
    print("  NON-ASSOCIATIVE AMPLIFICATION ANALYSIS")
    print("=" * 76)

    # The Fano plane has 7 triples. Each triple involves 3 of 7 imaginary channels.
    # When Weyl channels participate in a Fano triple, the non-associative
    # correction is MODIFIED because the v-spinor is absent in those channels.

    # Count how many Fano triples are affected for each crystal type
    print(f"\n  {'Crystal':<18} {'Weyl ch':>8} {'Fano affected':>14} {'Fano intact':>12} {'Ratio':>8}")
    for ctype in crystal_types:
        mask = get_weyl_mask(ctype)
        n_affected = 0
        n_intact = 0
        for a, b, c in FANO_TRIPLES:
            if mask[a] or mask[b] or mask[c]:
                n_affected += 1
            else:
                n_intact += 1
        total = n_affected + n_intact
        ratio = n_intact / total if total > 0 else 0
        print(f"  {ctype:<18} {np.sum(mask):6d}/7  {n_affected:14d}  {n_intact:12d}  {ratio:8.3f}")

    # The intact Fano fraction determines how much of the non-associative
    # structure survives. If ALL triples are affected, the octonion structure
    # is fully broken -> max gravitational modification.
    # If NO triples affected (bipartite), full structure -> normal gravity.

    # ================================================================
    # THE PHYSICAL PREDICTION
    # ================================================================
    print(f"\n{'=' * 76}")
    print("  PHYSICAL PREDICTIONS WITH OCTONIONIC ENVELOPE")
    print("=" * 76)

    # For TaAs:
    g_taas = all_results['TaAs']['G_ratio']
    delta_g_taas = g_taas - 1
    f = 0.10
    m_sample = 1.0  # gram
    dm_taas = m_sample * f * delta_g_taas

    print(f"\n  TaAs (4/8 Weyl channels):")
    print(f"    G_oct/G_normal = {g_taas:.6f}")
    print(f"    delta_G/G = {delta_g_taas:.6f} ({delta_g_taas*100:.2f}%)")
    print(f"    At f=0.10: delta_m = {dm_taas:.4e} g = {dm_taas*1e6:.2f} ug")

    # Compare to Sim 14 single-channel
    dm_sim14 = m_sample * f * (G_sim14 - 1)
    print(f"\n  Comparison to single-channel (Sim 14):")
    print(f"    Sim 14: delta_G/G = {G_sim14-1:.4f}, delta_m = {dm_sim14:.4e} g")
    print(f"    Octonionic: delta_G/G = {delta_g_taas:.4f}, delta_m = {dm_taas:.4e} g")
    if dm_sim14 != 0:
        print(f"    Octonionic / single-channel = {dm_taas/dm_sim14:.4f}x")

    # ================================================================
    # INTERPRETATION
    # ================================================================
    print(f"\n{'=' * 76}")
    print("  INTERPRETATION")
    print("=" * 76)

    # Determine the overall pattern
    G_vals = [(ctype, all_results[ctype]['n_weyl'], all_results[ctype]['G_ratio'])
              for ctype in crystal_types]

    print(f"\n  Pattern: G_ratio vs number of Weyl channels:")
    print(f"  {'Weyl/8':>6}  {'G_ratio':>10}  {'delta_G/G':>10}")
    for ctype, nw, g in G_vals:
        print(f"  {nw:4d}/8   {g:10.6f}  {g-1:10.6f}  ({ctype})")

    # Is there a simple relationship?
    # G_ratio = 1 - n_weyl/8 * delta_per_channel?
    # Or: G_ratio = (1 - n_weyl/8)^2 (bipartite fraction squared)?
    print(f"\n  --- Fitting G_ratio vs Weyl fraction ---")
    nw_arr = np.array([all_results[ct]['n_weyl'] for ct in crystal_types])
    g_arr = np.array([all_results[ct]['G_ratio'] for ct in crystal_types])

    # Model 1: linear G = 1 - alpha * n_weyl/8
    frac = nw_arr / 8.0
    # Fit: G = 1 - alpha * frac
    if np.var(frac) > TOL:
        alpha_lin = (1 - np.mean(g_arr)) / np.mean(frac) if np.mean(frac) > TOL else 0
    else:
        alpha_lin = 0
    print(f"  Linear model: G = 1 - {alpha_lin:.4f} * (n_weyl/8)")
    for ct in crystal_types:
        nw = all_results[ct]['n_weyl']
        g_pred = 1 - alpha_lin * nw / 8
        g_act = all_results[ct]['G_ratio']
        print(f"    {ct:<18}: predicted {g_pred:.4f}, actual {g_act:.4f}, diff {g_act-g_pred:.4f}")

    # Model 2: quadratic G = (1 - n_weyl/8)^2 (bipartite fraction squared)
    print(f"\n  Quadratic model: G = (1 - n_weyl/8)^2")
    for ct in crystal_types:
        nw = all_results[ct]['n_weyl']
        g_pred = (1 - nw/8)**2
        g_act = all_results[ct]['G_ratio']
        print(f"    {ct:<18}: predicted {g_pred:.4f}, actual {g_act:.4f}, diff {g_act-g_pred:.4f}")

    # Model 3: G = (n_bip/8)^2 + Fano_correction
    print(f"\n  The octonionic envelope modifies the gravitational coupling through:")
    print(f"  1. Direct channel reduction: fewer bipartite channels = less coupling")
    print(f"  2. Fano triple disruption: broken triples lose non-associative correction")
    print(f"  3. V-spinor leakage: Fano coupling can inject v-content into Weyl channels")

    t_elapsed = time.time() - t_start
    print(f"\n  Simulation completed in {t_elapsed:.1f}s")
    print("=" * 76)


if __name__ == "__main__":
    main()
