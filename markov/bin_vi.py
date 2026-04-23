"""
bin_vi.py

Bin-level value iteration for MLC RRAM programming.

Instead of targeting a single exact resistance state, the policy targets a
resistance *bin* — any state within the bin counts as success.  This matches
real MLC operation: a 2-bit cell has 4 target windows, a 3-bit cell has 8.

Bin definitions (equal-width partition of [0, 64]):
  n_levels=4  (2-bit):  [0-15], [16-31], [32-47], [48-64]
  n_levels=8  (3-bit):  [0-7], [8-15], ..., [56-64]

Bellman equation:
  V[s, b] = 0                                      if s ∈ bin_b
  V[s, b] = min_a { 1 + Σ_{s'} T[a,s,s'] · V[s',b] }   otherwise

Usage:
  python bin_vi.py --pw-bins 1 --n-levels 4
  python bin_vi.py --pw-bins 4 --n-levels 8

Output files (saved to markov/):
  bin_value_pw{K}_lv{L}.npy   : float32 [65, L]  V[s, b] = expected pulses to bin b
  bin_policy_pw{K}_lv{L}.npy  : int32   [65, L]  policy[s, b] = optimal action index
"""

import argparse
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent

S       = 65
INF     = 1e9
MAX_ITER    = 50_000
EPS_VALUE   = 1e-2
EPS_POLICY  = 200


def make_bin_membership(n_levels):
    """
    Returns int array [S] where bin_membership[s] = bin index of state s.
    Equal-width partition of [0, 64] into n_levels bins.
    """
    membership = np.minimum(
        (np.arange(S) * n_levels) // S,
        n_levels - 1
    ).astype(np.int32)
    return membership


def run_bin_vi(probs, n_levels):
    """
    Solve the bin-level SSP via value iteration.

    probs : float64 [N_ACTIONS, S, S]

    Returns
    -------
    V      : float32 [S, n_levels]  expected pulses; inf for unreachable
    policy : int32   [S, n_levels]  action index; -1 for terminal/unreachable
    bin_membership : int32 [S]
    """
    bin_membership = make_bin_membership(n_levels)
    valid_action   = probs.sum(axis=2) > 0   # [N_ACTIONS, S]

    V      = np.zeros((S, n_levels), dtype=np.float64)
    policy = np.full((S, n_levels), -1, dtype=np.int32)

    # Pre-compute terminal mask: terminal[s, b] = True iff s ∈ bin b
    terminal = np.zeros((S, n_levels), dtype=bool)
    for b in range(n_levels):
        terminal[:, b] = (bin_membership == b)

    policy_stable = 0

    for iteration in range(MAX_ITER):
        # Q[a, s, b] = 1 + sum_{s'} T[a,s,s'] * V[s',b]
        Q = 1.0 + np.tensordot(probs, V, axes=([2], [0]))   # [N_ACTIONS, S, n_levels]
        Q[~valid_action, :] = INF

        V_new  = Q.min(axis=0)    # [S, n_levels]
        best_a = Q.argmin(axis=0) # [S, n_levels]

        # Re-impose terminal condition
        V_new[terminal]  = 0.0
        best_a[terminal] = -1

        delta = np.abs(V_new - V).max()
        policy_changed = not np.array_equal(best_a, policy)
        policy_stable  = 0 if policy_changed else policy_stable + 1

        V      = V_new
        policy = best_a

        if (iteration + 1) % 500 == 0:
            non_terminal = ~terminal
            print(f"  iter {iteration+1:5d}  Δ={delta:.2e}  "
                  f"policy_stable={policy_stable}  "
                  f"mean_V={V[non_terminal].mean():.2f}")

        if delta < EPS_VALUE:
            print(f"  Converged (ΔV < {EPS_VALUE}) at iteration {iteration+1}")
            break
        if policy_stable >= EPS_POLICY:
            print(f"  Policy stable for {EPS_POLICY} iters — stopping at iter {iteration+1}  (Δ={delta:.2e})")
            break
    else:
        print(f"  WARNING: did not converge after {MAX_ITER} iterations")

    V[V >= INF / 2] = np.inf
    return V.astype(np.float32), policy.astype(np.int32), bin_membership


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bin-level value iteration for MLC RRAM.')
    parser.add_argument('--pw-bins',  type=int, default=1,
                        help='pw bin count used when building the transition matrix (default: 1)')
    parser.add_argument('--n-levels', type=int, default=4,
                        help='Number of resistance levels / bins (4=2-bit, 8=3-bit, default: 4)')
    args = parser.parse_args()

    probs_path = OUT_DIR / f'transition_probs_pw{args.pw_bins}.npy'
    if not probs_path.exists():
        raise FileNotFoundError(
            f"{probs_path} not found. Run: python build_chain.py --pw-bins {args.pw_bins}"
        )

    print(f"Loading {probs_path.name}...")
    probs = np.load(probs_path)
    print(f"  Shape: {probs.shape}  ({args.pw_bins} pw bins, {probs.shape[0]} total actions)")

    print(f"\nRunning bin-level value iteration  (n_levels={args.n_levels})...")
    V, policy, bin_membership = run_bin_vi(probs, args.n_levels)

    suffix = f'_pw{args.pw_bins}_lv{args.n_levels}'
    np.save(OUT_DIR / f'bin_value{suffix}.npy',  V)
    np.save(OUT_DIR / f'bin_policy{suffix}.npy', policy)
    print(f"Saved bin_value{suffix}.npy, bin_policy{suffix}.npy")

    # Summary
    print(f"\n=== Bin-level value function summary ===")
    print(f"  Bin membership: {[list(np.where(bin_membership==b)[0][[0,-1]]) for b in range(args.n_levels)]}")
    for b in range(args.n_levels):
        non_terminal = bin_membership != b
        v_col = V[non_terminal, b]
        finite = v_col[np.isfinite(v_col)]
        print(f"  Bin {b}: mean={finite.mean():.2f}  max={finite.max():.2f}  "
              f"unreachable={np.isinf(v_col).sum()}")
