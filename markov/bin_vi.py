"""
bin_vi.py

Bin-level value iteration for MLC RRAM programming.

Supports two bin-definition modes:

  equal  — equal-width partition of [0, 64] into n_levels bins.
            Simple baseline; each bin is ~8–16 states wide.

  pba    — PBA-style write windows derived from retention characterisation data.
            Each write window is only 4–5 ADC states wide (matching the actual
            hardware target precision). Gaps between windows are non-terminal:
            the policy must navigate out of them. This gives a realistic,
            non-trivial BER and enables direct comparison with RADAR/PBA results.

            Also stores read windows [Rlow, Rhigh) — wider post-drift acceptance
            zones used when computing read-time BER after relaxation.

Bellman equation (both modes):
  V[s, b] = 0                                      if s ∈ write_window_b
  V[s, b] = min_a { 1 + Σ_{s'} T[a,s,s'] · V[s',b] }   otherwise

Gap states (pba mode only, bin_membership == -1) are never terminal for any
bin; the policy is free to use SET or RESET pulses to escape them.

Usage:
  python bin_vi.py --pw-bins 1 --n-levels 4
  python bin_vi.py --pw-bins 1 --n-levels 4 --bin-type pba
  python bin_vi.py --pw-bins 4 --n-levels 8 --bin-type pba

Output files (saved to markov/):
  bin_value_pw{K}_lv{L}_{type}.npy   : float32 [65, L]
  bin_policy_pw{K}_lv{L}_{type}.npy  : int32   [65, L]
  bin_windows_lv{L}_{type}.npz       : write_windows, read_windows arrays (pba only)
"""

import argparse
import numpy as np
from pathlib import Path

OUT_DIR   = Path(__file__).parent
MODEL_DIR = OUT_DIR.parent / 'model'

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


def make_pba_bins(n_levels, model_path=None):
    """
    Compute PBA-style bin boundaries from retention characterisation data.

    Uses the same greedy algorithm as dala.py: finds the maximum set of
    non-overlapping write windows that achieves the minimum BER, then refines
    the read windows to cover the full [0, 64] range with no gaps.

    Returns
    -------
    bin_membership : int32 [S]
        bin_membership[s] = b  if s ∈ write window of bin b
        bin_membership[s] = -1 if s is in a gap between write windows
    write_windows  : int32 [n_levels, 2]  columns = [tmin, tmax)
    read_windows   : int32 [n_levels, 2]  columns = [rlow, rhigh)  (post-drift)
    """
    if model_path is None:
        model_path = MODEL_DIR / 'retention1s.csv'

    # Load retention distributions indexed by write level
    distributions = {}
    with open(model_path) as f:
        for line in f:
            tokens = line.strip().split(',')
            tmin, tmax = int(tokens[0]), int(tokens[1])
            distr = sorted([int(x) for x in tokens[2:]])
            distributions[(tmin, tmax)] = distr

    def get_write_window(tmin, ber_frac):
        """Empirical [tmin, tmax) write window at given BER fraction."""
        tmax_key = tmin + 4
        if (tmin, tmax_key) not in distributions:
            return None
        distr = distributions[(tmin, tmax_key)]
        n = len(distr)
        n_discard = max(1, int(ber_frac * n / 2))
        return distr[n_discard], distr[-n_discard] + 1   # [rlow, rhigh)

    def level_inference(ber_frac):
        """Greedy: find all non-overlapping write windows at given BER frac."""
        candidates = []
        for tmin in range(0, 60):
            result = get_write_window(tmin, ber_frac)
            if result is None:
                continue
            rlow, rhigh = result
            candidates.append([rlow, rhigh, tmin, tmin + 4])
        # Sort by rhigh, then greedily select non-overlapping levels
        candidates.sort(key=lambda x: x[1])
        levels = [candidates[0]]
        for c in candidates[1:]:
            if c[0] >= levels[-1][1] and c[2] >= levels[-1][3]:
                levels.append(c)
        return levels

    # Binary search for minimum BER that yields exactly n_levels
    lo, hi, best = 0.0, 1.0, None
    for _ in range(50):
        mid = (lo + hi) / 2
        lvls = level_inference(mid)
        if len(lvls) >= n_levels:
            best = lvls[:n_levels]   # take the first n_levels (lowest conductance first)
            hi = mid
        else:
            lo = mid

    if best is None or len(best) < n_levels:
        raise RuntimeError(f"Could not find {n_levels}-level PBA allocation.")

    # Refine: close gaps between adjacent read windows → cover full [0,64]
    for i in range(1, n_levels):
        merge = (best[i - 1][1] + best[i][0]) // 2
        best[i - 1][1] = merge
        best[i][0]     = merge
    best[0][0]           = 0
    best[n_levels - 1][1] = S  # exclusive upper bound

    read_windows  = np.array([[b[0], b[1]] for b in best], dtype=np.int32)
    write_windows = np.array([[b[2], b[3]] for b in best], dtype=np.int32)

    # Build bin membership from write windows; gaps → -1
    bin_membership = np.full(S, -1, dtype=np.int32)
    for b, (tmin, tmax) in enumerate(write_windows):
        bin_membership[tmin:tmax] = b

    return bin_membership, write_windows, read_windows


def run_bin_vi(probs, n_levels, bin_membership=None):
    """
    Solve the bin-level SSP via value iteration.

    probs          : float64 [N_ACTIONS, S, S]
    bin_membership : int32 [S] or None.  If None, uses equal-width bins.
                     -1 entries are gap states (never terminal for any bin).

    Returns
    -------
    V      : float32 [S, n_levels]  expected pulses; inf for unreachable
    policy : int32   [S, n_levels]  action index; -1 for terminal/unreachable
    bin_membership : int32 [S]  (the one actually used)
    """
    if bin_membership is None:
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
    parser.add_argument('--pw-bins',  type=int, default=1)
    parser.add_argument('--n-levels', type=int, default=4)
    parser.add_argument('--bin-type', choices=['equal', 'pba'], default='equal',
                        help='equal = equal-width bins; pba = PBA write-window bins (default: equal)')
    args = parser.parse_args()

    probs_path = OUT_DIR / f'transition_probs_pw{args.pw_bins}.npy'
    if not probs_path.exists():
        raise FileNotFoundError(
            f"{probs_path} not found. Run: python build_chain.py --pw-bins {args.pw_bins}"
        )

    print(f"Loading {probs_path.name}...")
    probs = np.load(probs_path)
    print(f"  Shape: {probs.shape}  ({args.pw_bins} pw bins, {probs.shape[0]} total actions)")

    if args.bin_type == 'pba':
        print(f"\nComputing PBA-style bins (n_levels={args.n_levels})...")
        bin_membership, write_windows, read_windows = make_pba_bins(args.n_levels)
        print(f"  Write windows: {write_windows.tolist()}")
        print(f"  Read  windows: {read_windows.tolist()}")
        print(f"  Gap states:    {(bin_membership == -1).sum()} / {S}")
    else:
        bin_membership = make_bin_membership(args.n_levels)
        write_windows = read_windows = None

    print(f"\nRunning bin-level value iteration  (n_levels={args.n_levels}, bin_type={args.bin_type})...")
    V, policy, _ = run_bin_vi(probs, args.n_levels, bin_membership=bin_membership)

    suffix = f'_pw{args.pw_bins}_lv{args.n_levels}_{args.bin_type}'
    np.save(OUT_DIR / f'bin_value{suffix}.npy',  V)
    np.save(OUT_DIR / f'bin_policy{suffix}.npy', policy)
    print(f"Saved bin_value{suffix}.npy, bin_policy{suffix}.npy")
    if args.bin_type == 'pba':
        np.savez(OUT_DIR / f'bin_windows_lv{args.n_levels}_pba.npz',
                 write_windows=write_windows, read_windows=read_windows)
        print(f"Saved bin_windows_lv{args.n_levels}_pba.npz")

    # Summary
    print(f"\n=== Bin-level value function summary ===")
    for b in range(args.n_levels):
        non_terminal = (bin_membership != b)
        v_col = V[non_terminal, b]
        finite = v_col[np.isfinite(v_col)]
        if len(finite):
            print(f"  Bin {b}: mean={finite.mean():.2f}  max={finite.max():.2f}  "
                  f"unreachable={np.isinf(v_col).sum()}")
