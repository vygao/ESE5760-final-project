"""
monte_carlo.py

Empirical validation and distribution characterization of the optimal policy
via Monte Carlo rollouts over the learned transition model.

For each (s, t) pair the policy is simulated N_TRIALS times:
  - At each step: apply policy[current, t], sample next state from T[action, current, :]
  - Terminate when current == t (success) or MAX_STEPS is hit (timeout)

Outputs (saved to markov/):
  mc_pulse_counts.npy   : int32   [65, 65, N_TRIALS]  pulses per rollout (MAX_STEPS = timeout)
  mc_timed_out.npy      : bool    [65, 65, N_TRIALS]  True if rollout hit MAX_STEPS
  mc_summary.csv        : per-(s,t): mean, std, p50, p90, p99, success_rate, V_theoretical

Figures:
  mc_mean_vs_theoretical.png  : scatter of empirical mean vs V[s,t]
  mc_percentiles.png          : heatmaps of p50, p90, p99
  mc_histograms.png           : pulse-count distributions for selected (s,t) pairs
  mc_interleaving.png         : SET/RESET switch frequency per rollout
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent

S         = 65
N_TRIALS  = 1_000
MAX_STEPS = 1_500   # ~5x theoretical max (303); hard cap for truly stuck rollouts
SEED      = 42


# ── rollout engine ────────────────────────────────────────────────────────────

def run_monte_carlo(probs, policy, V, n_trials=N_TRIALS, max_steps=MAX_STEPS, seed=SEED,
                    record_actions=False):
    """
    Vectorised Monte Carlo rollout over all (s, t) pairs simultaneously.

    probs   : float64 [N_ACTIONS, S, S]  transition probabilities
    policy  : int32   [S, S]             policy[s,t] = action index, -1 = terminal/unreachable
    V       : float32 [S, S]             value function (inf = unreachable)

    record_actions : if True, also returns action_sequences list for interleaving analysis
                     (only feasible for small n_trials due to memory)

    Returns
    -------
    pulse_counts : int32 [S, S, n_trials]
    timed_out    : bool  [S, S, n_trials]
    action_log   : list of (s, t, actions_array) or None
    """
    rng = np.random.default_rng(seed)

    N_PAIRS = S * S
    s_start = np.repeat(np.arange(S), S)  # [N_PAIRS]  start state for each pair
    t_arr   = np.tile(np.arange(S), S)    # [N_PAIRS]  target state for each pair

    # Pairs where V == inf are unreachable — treat as permanently timed-out
    reachable = np.isfinite(V).ravel()    # [N_PAIRS]

    # current[p, k] = current state of pair p, trial k
    current = np.tile(s_start[:, None], (1, n_trials)).astype(np.int32)  # [N_PAIRS, n_trials]

    pulse_counts = np.zeros((N_PAIRS, n_trials), dtype=np.int32)

    # Done from the start: diagonal (s==t) or unreachable
    done = (s_start == t_arr)[:, None] * np.ones(n_trials, dtype=bool)
    done[~reachable] = True

    action_log = [] if record_actions else None

    for step in range(max_steps):
        active_pairs = (~done).any(axis=1)  # [N_PAIRS]  at least one trial still running
        if not active_pairs.any():
            break

        active = ~done  # [N_PAIRS, n_trials]

        # Look up action for each active (pair, trial): policy[current[p,k], target[p]]
        t_broadcast = np.broadcast_to(t_arr[:, None], current.shape)
        action = policy[current, t_broadcast]  # [N_PAIRS, n_trials]

        # Flatten active rollouts for batch sampling
        p_idx, k_idx = np.where(active)
        if len(p_idx) == 0:
            break

        a_flat = action[p_idx, k_idx]
        s_flat = current[p_idx, k_idx]

        # Transition distributions for active rollouts
        trans = probs[a_flat, s_flat, :]  # [N_active, S]

        # Vectorised categorical sampling via inverse-CDF / searchsorted
        cdf    = np.cumsum(trans, axis=1)         # [N_active, S]
        u      = rng.uniform(size=len(p_idx))     # [N_active]
        next_s = (cdf < u[:, None]).sum(axis=1).clip(0, S - 1)  # [N_active]

        current[p_idx, k_idx] = next_s
        pulse_counts[active]  += 1

        reached = (current == t_arr[:, None])
        done   |= reached

    timed_out = ~done
    timed_out[~reachable] = True
    # Unreachable / timed-out: set pulse_counts to max_steps as sentinel
    pulse_counts[timed_out] = max_steps

    return (
        pulse_counts.reshape(S, S, n_trials).astype(np.int32),
        timed_out.reshape(S, S, n_trials),
        action_log,
    )


# ── interleaving analysis ─────────────────────────────────────────────────────

def count_set_reset_switches(probs, policy, V, n_trials=200, max_steps=MAX_STEPS, seed=99):
    """
    For each (s, t) pair, run n_trials rollouts and count SET↔RESET switches per rollout.
    A switch is any step where pulse type changes from the previous step.

    Returns
    -------
    mean_switches : float32 [S, S]   mean number of type switches per successful rollout
    """
    N_ACTIONS_HALF = probs.shape[0] // 2  # SET = [0, N//2), RESET = [N//2, N)

    rng = np.random.default_rng(seed)
    N_PAIRS = S * S
    s_start = np.repeat(np.arange(S), S)
    t_arr   = np.tile(np.arange(S), S)
    reachable = np.isfinite(V).ravel()

    current      = np.tile(s_start[:, None], (1, n_trials)).astype(np.int32)
    done         = (s_start == t_arr)[:, None] * np.ones(n_trials, dtype=bool)
    done[~reachable] = True

    switches      = np.zeros((N_PAIRS, n_trials), dtype=np.int32)
    prev_type     = np.full((N_PAIRS, n_trials), -1, dtype=np.int8)  # -1 = no previous pulse

    for step in range(max_steps):
        if (~done).sum() == 0:
            break

        active = ~done
        t_broadcast = np.broadcast_to(t_arr[:, None], current.shape)
        action = policy[current, t_broadcast]

        p_idx, k_idx = np.where(active)
        if len(p_idx) == 0:
            break

        a_flat = action[p_idx, k_idx]
        s_flat = current[p_idx, k_idx]

        pulse_type = (a_flat >= N_ACTIONS_HALF).astype(np.int8)  # 0=SET, 1=RESET

        switched = (
            (prev_type[p_idx, k_idx] != -1) &
            (pulse_type != prev_type[p_idx, k_idx])
        )
        switches[p_idx[switched], k_idx[switched]] += 1
        prev_type[p_idx, k_idx] = pulse_type

        trans  = probs[a_flat, s_flat, :]
        cdf    = np.cumsum(trans, axis=1)
        u      = rng.uniform(size=len(p_idx))
        next_s = (cdf < u[:, None]).sum(axis=1).clip(0, S - 1)

        current[p_idx, k_idx] = next_s
        done |= (current == t_arr[:, None])

    # Average switches only over successful (completed) rollouts
    switch_sum   = switches.astype(np.float32)
    success_mask = done.reshape(N_PAIRS, n_trials)  # True = reached target
    n_success    = success_mask.sum(axis=1).clip(1, None)
    mean_sw      = (switch_sum * success_mask).sum(axis=1) / n_success
    return mean_sw.reshape(S, S).astype(np.float32)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_mean_vs_theoretical(mc_mean, V):
    finite = np.isfinite(V) & ~np.eye(S, dtype=bool)
    x = V[finite]
    y = mc_mean[finite]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=4, alpha=0.3, color='steelblue')
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'r--', lw=1, label='y = x')
    ax.set_xlabel('Theoretical V[s, t]  (value iteration)', fontsize=11)
    ax.set_ylabel('Empirical mean pulses  (Monte Carlo)', fontsize=11)
    ax.set_title('MC empirical mean vs. theoretical expected cost', fontsize=12)
    ax.legend(fontsize=9)

    residuals = y - x
    ax.text(0.03, 0.97,
            f"Mean residual: {residuals.mean():+.2f}\n"
            f"Max  residual: {residuals.max():+.2f}\n"
            f"RMSE:          {np.sqrt((residuals**2).mean()):.2f}",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'mc_mean_vs_theoretical.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved mc_mean_vs_theoretical.png")


def plot_percentile_heatmaps(p50, p90, p99):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, data, label in zip(axes, [p50, p90, p99], ['p50', 'p90', 'p99']):
        d = data.astype(float)
        d[d >= MAX_STEPS] = np.nan
        im = ax.imshow(d.T, origin='lower', aspect='equal',
                       cmap='viridis_r', interpolation='none')
        ax.set_xlabel('Initial state s', fontsize=10)
        ax.set_ylabel('Target state t', fontsize=10)
        ax.set_title(f'MC {label} pulse count', fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Pulses')
    plt.suptitle('Monte Carlo pulse-count percentiles', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'mc_percentiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved mc_percentiles.png")


def plot_histograms(pulse_counts, timed_out, pairs):
    """
    pairs: list of (s, t, label) tuples to plot histograms for.
    """
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (s, t, label) in zip(axes, pairs):
        counts = pulse_counts[s, t]
        to     = timed_out[s, t]
        success_counts = counts[~to].astype(float)

        ax.hist(success_counts, bins=40, color='steelblue', alpha=0.8, edgecolor='white')
        ax.axvline(success_counts.mean(), color='red', lw=1.5, linestyle='--',
                   label=f'mean={success_counts.mean():.1f}')
        ax.axvline(np.percentile(success_counts, 90), color='orange', lw=1.5, linestyle=':',
                   label=f'p90={np.percentile(success_counts, 90):.0f}')
        ax.set_xlabel('Pulses to reach target', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        timeout_pct = to.mean() * 100
        ax.set_title(f'{label}  (s={s}→t={t})\n'
                     f'timeout={timeout_pct:.1f}%  n={len(success_counts)}', fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'mc_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved mc_histograms.png")


def plot_interleaving(mean_switches):
    fig, ax = plt.subplots(figsize=(7, 6))
    d = mean_switches.copy()
    d[~np.isfinite(d)] = np.nan
    np.fill_diagonal(d, np.nan)

    im = ax.imshow(d.T, origin='lower', aspect='equal',
                   cmap='hot_r', interpolation='none')
    ax.set_xlabel('Initial state s', fontsize=11)
    ax.set_ylabel('Target state t', fontsize=11)
    ax.set_title('Mean SET↔RESET type switches per rollout', fontsize=12)
    plt.colorbar(im, ax=ax, label='Mean switches', shrink=0.8)

    finite = d[np.isfinite(d)]
    frac_any = (finite > 0).mean()
    ax.text(0.03, 0.97,
            f"Pairs with ≥1 switch: {frac_any*100:.1f}%\n"
            f"Mean switches (all):  {finite.mean():.2f}",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'mc_interleaving.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved mc_interleaving.png")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading model artifacts...")
    probs  = np.load(OUT_DIR / 'transition_probs.npy')   # [1024, 65, 65]
    policy = np.load(OUT_DIR / 'policy.npy')              # [65, 65]
    V      = np.load(OUT_DIR / 'value_function.npy')      # [65, 65]
    print(f"  probs:  {probs.shape}  policy: {policy.shape}  V: {V.shape}")

    print(f"\nRunning Monte Carlo ({N_TRIALS} trials × {S}×{S} pairs, max {MAX_STEPS} steps)...")
    pulse_counts, timed_out, _ = run_monte_carlo(probs, policy, V)
    np.save(OUT_DIR / 'mc_pulse_counts.npy', pulse_counts)
    np.save(OUT_DIR / 'mc_timed_out.npy',   timed_out)
    print("Saved mc_pulse_counts.npy, mc_timed_out.npy")

    # Summary statistics per (s, t)
    print("\nComputing summary statistics...")
    success_mask = ~timed_out   # [S, S, N_TRIALS]

    # Replace timed-out entries with NaN for statistics
    mc_float = pulse_counts.astype(np.float32)
    mc_float[timed_out] = np.nan

    with np.errstate(all='ignore'):
        emp_mean = np.nanmean(mc_float, axis=2)
        emp_std  = np.nanstd( mc_float, axis=2)
        p50      = np.nanpercentile(mc_float, 50, axis=2)
        p90      = np.nanpercentile(mc_float, 90, axis=2)
        p99      = np.nanpercentile(mc_float, 99, axis=2)
    success_rate = success_mask.mean(axis=2)

    # Flatten to DataFrame
    rows = []
    for s in range(S):
        for t in range(S):
            if s == t:
                continue
            rows.append({
                's': s, 't': t,
                'V_theoretical':   float(V[s, t]),
                'mc_mean':         float(emp_mean[s, t]),
                'mc_std':          float(emp_std[s, t]),
                'mc_p50':          float(p50[s, t]),
                'mc_p90':          float(p90[s, t]),
                'mc_p99':          float(p99[s, t]),
                'success_rate':    float(success_rate[s, t]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'mc_summary.csv', index=False)
    print("Saved mc_summary.csv")

    # Print overall stats
    finite = df[df['V_theoretical'] < 1e8]
    print(f"\n=== Monte Carlo summary ===")
    print(f"  Reachable (s≠t) pairs:  {len(finite)}")
    print(f"  Overall success rate:   {finite['success_rate'].mean()*100:.1f}%")
    print(f"  Mean pulses (empirical):{finite['mc_mean'].mean():.2f}  "
          f"(theoretical: {finite['V_theoretical'].mean():.2f})")
    print(f"  p90 mean across pairs:  {finite['mc_p90'].mean():.2f}")
    print(f"  p99 mean across pairs:  {finite['mc_p99'].mean():.2f}")
    print(f"  Max p99 across pairs:   {finite['mc_p99'].max():.2f}")

    # Figures
    print("\nGenerating figures...")
    plot_mean_vs_theoretical(emp_mean, V)
    plot_percentile_heatmaps(p50, p90, p99)

    # Histogram pairs: easy same-direction, hard cross-range, random mid-range
    off_diag = ~np.eye(S, dtype=bool)
    finite_mask = np.isfinite(V) & off_diag
    V_vals = V.copy()
    V_vals[~finite_mask] = np.inf

    s_easy, t_easy = np.unravel_index(V_vals.argmin(), (S, S))
    s_hard, t_hard = np.unravel_index(
        np.where(np.isfinite(V_vals), V_vals, -1).argmax(), (S, S)
    )
    hist_pairs = [
        (int(s_easy), int(t_easy), f'easiest (V={V[s_easy,t_easy]:.1f})'),
        (32,          16,          f'mid-range (V={V[32,16]:.1f})'),
        (int(s_hard), int(t_hard), f'hardest (V={V[s_hard,t_hard]:.1f})'),
    ]
    plot_histograms(pulse_counts, timed_out, hist_pairs)

    # Interleaving analysis
    print("\nRunning interleaving analysis (200 trials)...")
    mean_switches = count_set_reset_switches(probs, policy, V, n_trials=200)
    np.save(OUT_DIR / 'mc_mean_switches.npy', mean_switches)
    plot_interleaving(mean_switches)
    print("Saved mc_mean_switches.npy")
