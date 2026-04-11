"""
value_iteration.py

Stochastic Shortest Path (SSP) value iteration over the RRAM Markov chain.

Problem formulation:
  - States:   s ∈ {0, …, 64}     (quantized resistance / conductance)
  - Targets:  t ∈ {0, …, 64}     (desired final state)
  - Actions:  a ∈ {0, …, 255}    (128 SET + 128 RESET vwl values)
  - Cost:     1 per pulse applied
  - Terminal: s == t              (zero cost, absorbing)

Bellman update:
  V[s, t] = 0                                if s == t
  V[s, t] = min_a { 1 + P[a,s,:] · V[:,t] } otherwise

Convergence: ‖V_new − V‖∞ < ε

Outputs (saved to markov/):
  value_function.npy   : float32 [65, 65]   V[s, t] = expected pulses to reach t from s
  policy.npy           : int16   [65, 65]   policy[s, t] = optimal action index
  policy_vwl.npy       : int16   [65, 65]   optimal vwl DAC code
  policy_type.npy      : str     [65, 65]   'SET' or 'RESET' (saved as uint8: 0=SET, 1=RESET)

Also produces:
  value_function.png   : heatmap of V[s, t]
  policy_vwl.png       : heatmap of optimal vwl per (s, target)
  policy_type.png      : SET vs RESET choice per (s, target)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

OUT_DIR = Path(__file__).parent

S         = 65
N_ACTIONS = 256
MAX_ITER     = 50_000
EPS_VALUE    = 1e-2   # absolute V tolerance — adequate when values are O(10–2000 pulses)
EPS_POLICY   = 200    # stop if policy unchanged for this many consecutive iterations
INF          = 1e9    # sentinel for unreachable


# ── value iteration ───────────────────────────────────────────────────────────

def run_value_iteration(probs):
    """
    probs: float32 [N_ACTIONS, S, S]
           probs[a, s, :] = P(s' | s, a).  All-zero row → action unavailable.

    Uses optimistic (V=0) initialisation, which converges upward to the true
    optimal cost-to-go V*[s,t] = expected number of pulses to reach t from s.
    This avoids the INF-bleed problem that occurs with pessimistic initialisation
    when unavailable-action rows have small positive transition probabilities.

    Returns:
      V      : float32 [S, S]   INF for truly unreachable (s,t) pairs
      policy : int32   [S, S]   action index, -1 for terminal/unreachable
    """
    # Mask: valid action for (a, s) iff row sums to > 0
    valid_action = probs.sum(axis=2) > 0    # [N_ACTIONS, S]
    # Optimistic init: V=0 everywhere; terminal stays 0 throughout
    V             = np.zeros((S, S), dtype=np.float64)
    policy        = np.full((S, S), -1,  dtype=np.int32)
    policy_stable = 0   # consecutive iters with unchanged policy

    for iteration in range(MAX_ITER):
        # Q[a, s, t] = 1 + sum_{s'} P[a,s,s'] * V[s',t]
        # probs: [A,S,S']  V: [S',T]  → tensordot over S' axis → [A,S,T]
        Q = 1.0 + np.tensordot(probs, V, axes=([2], [0]))   # [N_ACTIONS, S, S]

        # Mask unavailable (action, s) pairs — broadcast [A,S] mask across last dim T
        Q[~valid_action, :] = INF

        # Best action per (s, t)
        V_new  = Q.min(axis=0)    # [S, S]
        best_a = Q.argmin(axis=0) # [S, S]

        # Re-impose terminal condition
        np.fill_diagonal(V_new, 0.0)
        np.fill_diagonal(best_a, -1)

        delta = np.abs(V_new - V).max()

        # Track policy stability
        policy_changed = not np.array_equal(best_a, policy)
        policy_stable  = 0 if policy_changed else policy_stable + 1

        V      = V_new
        policy = best_a

        if (iteration + 1) % 500 == 0:
            off_diag = ~np.eye(S, dtype=bool)
            print(f"  iter {iteration+1:5d}  Δ={delta:.2e}  "
                  f"policy_stable={policy_stable}  "
                  f"mean_V={V[off_diag].mean():.2f}")

        if delta < EPS_VALUE:
            print(f"  Converged (ΔV < {EPS_VALUE}) at iteration {iteration+1}")
            break
        if policy_stable >= EPS_POLICY:
            print(f"  Policy stable for {EPS_POLICY} iterations — stopping at iter {iteration+1}  (Δ={delta:.2e})")
            break
    else:
        print(f"  WARNING: did not converge after {MAX_ITER} iterations (Δ={delta:.2e})")

    # Mark truly unreachable pairs as INF
    V[V >= INF / 2] = np.inf

    return V.astype(np.float32), policy.astype(np.int32)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_value_function(V, ax):
    display = V.copy()
    display[V >= INF / 2] = np.nan

    im = ax.imshow(display.T, origin='lower', aspect='equal',
                   cmap='viridis_r', interpolation='none')
    ax.set_xlabel('Initial state (s)', fontsize=11)
    ax.set_ylabel('Target state (t)', fontsize=11)
    ax.set_title('Expected pulses to reach target: V[s, t]', fontsize=12)
    ax.set_xlim(-0.5, S - 0.5)
    ax.set_ylim(-0.5, S - 0.5)
    plt.colorbar(im, ax=ax, label='Expected # pulses', shrink=0.8)

    reachable = (V < INF / 2).sum() - S   # exclude diagonal
    ax.text(0.02, 0.97,
            f"Reachable (s≠t) pairs: {reachable}/{S*S - S}\n"
            f"Max expected pulses:   {display[~np.isnan(display)].max():.1f}\n"
            f"Mean expected pulses:  {display[~np.isnan(display)].mean():.2f}",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))


def plot_policy_vwl(policy, ax):
    """Heatmap of optimal vwl for each (s, target)."""
    # Derive vwl from action index: action < 128 → vwl = action*2; else vwl = (action-128)*2
    vwl_map = np.where(policy < 128, policy * 2, (policy - 128) * 2).astype(float)
    vwl_map[policy == -1] = np.nan    # terminal / unreachable

    im = ax.imshow(vwl_map.T, origin='lower', aspect='equal',
                   cmap='plasma', interpolation='none',
                   vmin=0, vmax=254)
    ax.set_xlabel('Initial state (s)', fontsize=11)
    ax.set_ylabel('Target state (t)', fontsize=11)
    ax.set_title('Optimal vwl DAC code per (s, target)', fontsize=12)
    ax.set_xlim(-0.5, S - 0.5)
    ax.set_ylim(-0.5, S - 0.5)
    plt.colorbar(im, ax=ax, label='vwl DAC code', shrink=0.8)


def plot_policy_type(policy, ax):
    """SET (blue) vs RESET (orange) choice per (s, target)."""
    type_map = np.where(policy == -1, -1,
               np.where(policy < 128, 0, 1)).astype(float)  # 0=SET, 1=RESET, -1=N/A
    type_map[type_map == -1] = np.nan

    cmap = mcolors.ListedColormap(['#2255e0', '#e06020'])
    im = ax.imshow(type_map.T, origin='lower', aspect='equal',
                   cmap=cmap, interpolation='none', vmin=0, vmax=1)
    ax.set_xlabel('Initial state (s)', fontsize=11)
    ax.set_ylabel('Target state (t)', fontsize=11)
    ax.set_title('Optimal pulse type: SET (blue) vs RESET (orange)', fontsize=12)
    ax.set_xlim(-0.5, S - 0.5)
    ax.set_ylim(-0.5, S - 0.5)

    n_set   = (type_map == 0).sum()
    n_reset = (type_map == 1).sum()
    ax.text(0.02, 0.97,
            f"SET:   {n_set} pairs\nRESET: {n_reset} pairs",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading transition probabilities...")
    probs = np.load(OUT_DIR / 'transition_probs.npy')   # [256, 65, 65]
    print(f"  Shape: {probs.shape},  dtype: {probs.dtype}")
    print(f"  Valid (action,s) rows: {(probs.sum(axis=2) > 0).sum()}")

    print("\nRunning value iteration...")
    V, policy = run_value_iteration(probs)

    # Save outputs
    np.save(OUT_DIR / 'value_function.npy', V)
    np.save(OUT_DIR / 'policy.npy', policy)

    # Derived: vwl and type arrays
    vwl_arr  = np.where(policy < 128, policy * 2, (policy - 128) * 2).astype(np.int16)
    type_arr = np.where(policy == -1, -1, np.where(policy < 128, 0, 1)).astype(np.int8)
    np.save(OUT_DIR / 'policy_vwl.npy',  vwl_arr)
    np.save(OUT_DIR / 'policy_type.npy', type_arr)
    print("Saved value_function.npy, policy.npy, policy_vwl.npy, policy_type.npy")

    # Summary stats
    reachable_mask = (V < INF / 2)
    off_diag = ~np.eye(S, dtype=bool)
    print(f"\n=== Value function summary ===")
    print(f"  Reachable (s≠t) pairs: {(reachable_mask & off_diag).sum()} / {S*S - S}")
    reachable_vals = V[reachable_mask & off_diag]
    print(f"  Min expected pulses:   {reachable_vals.min():.2f}")
    print(f"  Max expected pulses:   {reachable_vals.max():.2f}")
    print(f"  Mean expected pulses:  {reachable_vals.mean():.2f}")

    # Print a small slice of the policy for sanity check
    print("\n  Sample V[s=32, t=0..64]:", np.round(V[32, ::8], 2))
    print("  Sample policy[s=32, t=0..64]:", policy[32, ::8])

    # --- Figures ---
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_value_function(V, ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'value_function.png', dpi=150, bbox_inches='tight')
    print("\nSaved value_function.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_policy_vwl(policy, ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'policy_vwl.png', dpi=150, bbox_inches='tight')
    print("Saved policy_vwl.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_policy_type(policy, ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'policy_type.png', dpi=150, bbox_inches='tight')
    print("Saved policy_type.png")
