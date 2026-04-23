"""
run_experiments.py

Experiment orchestrator: sweeps (pw_bins, n_levels) configurations and produces
BER and write-latency comparison figures.

Experiment grid (configurable at top of file):
  pw_bins  ∈ [1, 2, 4]     — number of pulse-width bins in the action space
  n_levels ∈ [4, 8]         — number of resistance levels (2-bit, 3-bit)

For each configuration:
  1. Build transition matrix if not already cached  (build_chain.py)
  2. Run bin-level value iteration if not cached    (bin_vi.py)
  3. Run Monte Carlo rollouts from HRS (state 0) for each target bin
  4. Compute BER and expected write pulses

Primary figure of merit: BER — fraction of rollouts that fail to land in the
correct resistance bin within a given write budget.

Additional metrics:
  - Expected write pulses (lower bound on write latency)
  - BER vs. write budget curves (trade-off between speed and accuracy)

Usage:
  python run_experiments.py             # run all configs, skip cached
  python run_experiments.py --rebuild   # force rebuild of all caches
  python run_experiments.py --n-trials 2000  # more MC trials

NOTE: Building transition matrices for pw_bins > 1 requires re-reading the
full sweep CSVs (~minutes per config). Results are cached in markov/ as .npy files.
"""

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR  = Path(__file__).parent
DATA_DIR = OUT_DIR.parent / 'data'

# ── experiment grid ───────────────────────────────────────────────────────────

PW_BINS_LIST  = [1, 2, 4]    # pw_bins configurations to evaluate
N_LEVELS_LIST = [4, 8]       # resistance levels (2-bit, 3-bit)

START_STATE = 2              # HRS — state 0 has no observed outgoing transitions;
                             # state 2 is the lowest fully-covered state (1024 valid actions)
N_TRIALS    = 2_000          # MC rollouts per (start, target_bin)
MAX_STEPS   = 1_500          # hard cap per rollout
SEED        = 42

# Write-budget checkpoints for BER vs. latency curves
BUDGETS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]

S = 65


# ── helpers ───────────────────────────────────────────────────────────────────

def bin_membership(n_levels):
    return np.minimum((np.arange(S) * n_levels) // S, n_levels - 1).astype(np.int32)


def ensure_transition_matrix(pw_bins, rebuild=False):
    path = OUT_DIR / f'transition_probs_pw{pw_bins}.npy'
    if path.exists() and not rebuild:
        print(f"  [cached] {path.name}")
        return np.load(path)
    print(f"  Building transition matrix (pw_bins={pw_bins})...")
    result = subprocess.run(
        [sys.executable, str(OUT_DIR / 'build_chain.py'), '--pw-bins', str(pw_bins)],
        check=True
    )
    return np.load(path)


def ensure_bin_policy(pw_bins, n_levels, rebuild=False):
    suffix = f'_pw{pw_bins}_lv{n_levels}'
    v_path = OUT_DIR / f'bin_value{suffix}.npy'
    p_path = OUT_DIR / f'bin_policy{suffix}.npy'
    if v_path.exists() and p_path.exists() and not rebuild:
        print(f"  [cached] {v_path.name}")
        return np.load(v_path), np.load(p_path)
    print(f"  Running bin value iteration (pw_bins={pw_bins}, n_levels={n_levels})...")
    subprocess.run(
        [sys.executable, str(OUT_DIR / 'bin_vi.py'),
         '--pw-bins', str(pw_bins), '--n-levels', str(n_levels)],
        check=True
    )
    return np.load(v_path), np.load(p_path)


# ── Monte Carlo (bin-level, from fixed start state) ───────────────────────────

def run_bin_mc(probs, policy, membership, start_state, n_trials, max_steps, seed):
    """
    Run MC rollouts starting from `start_state`, targeting each bin in turn.

    Returns
    -------
    pulse_counts : int32  [n_levels, n_trials]  pulses used; max_steps = timed out
    final_bin    : int32  [n_levels, n_trials]  bin of final state
    timed_out    : bool   [n_levels, n_trials]
    """
    rng = np.random.default_rng(seed)
    n_levels = policy.shape[1]

    pulse_counts = np.zeros((n_levels, n_trials), dtype=np.int32)
    final_bin    = np.full((n_levels, n_trials), -1, dtype=np.int32)
    timed_out    = np.zeros((n_levels, n_trials), dtype=bool)

    for b in range(n_levels):
        current = np.full(n_trials, start_state, dtype=np.int32)
        done    = np.zeros(n_trials, dtype=bool)

        # If start_state is already in target bin, mark immediately done
        if membership[start_state] == b:
            done[:] = True

        for step in range(max_steps):
            if done.all():
                break
            active = ~done
            a = policy[current[active], b]          # [n_active]
            s = current[active]                     # [n_active]

            trans = probs[a, s, :]                  # [n_active, S]
            cdf   = np.cumsum(trans, axis=1)
            u     = rng.uniform(size=active.sum())
            next_s = (cdf < u[:, None]).sum(axis=1).clip(0, S - 1)

            current[active]       = next_s
            pulse_counts[b, active] += 1

            just_reached = membership[current] == b
            done |= just_reached

        timed_out[b] = ~done
        final_bin[b] = membership[current]
        pulse_counts[b, timed_out[b]] = max_steps  # sentinel

    return pulse_counts, final_bin, timed_out


# ── BER computation ───────────────────────────────────────────────────────────

def compute_ber_vs_budget(pulse_counts, timed_out, budgets):
    """
    BER(budget) = fraction of rollouts where the policy hasn't succeeded within `budget` pulses.
    For each target bin and budget, BER = mean(pulse_counts > budget OR timed_out).

    Returns ber : float [n_levels, len(budgets)]
    """
    n_levels = pulse_counts.shape[0]
    ber = np.zeros((n_levels, len(budgets)))
    for i, budget in enumerate(budgets):
        failed = (pulse_counts > budget) | timed_out
        ber[:, i] = failed.mean(axis=1)
    return ber


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_ber_vs_budget(results, n_levels, budgets):
    """
    One figure per n_levels: BER vs. write budget, one curve per pw_bins config.
    Averaged over non-trivial target bins (bins that require actual programming from HRS).
    """
    membership = bin_membership(n_levels)
    trivial_bin = membership[START_STATE]   # bin containing HRS — 0 pulses needed

    fig, ax = plt.subplots(figsize=(7, 5))
    colors  = plt.cm.viridis(np.linspace(0.15, 0.85, len(PW_BINS_LIST)))

    for (pw_bins, nl, ber, _), color in zip(results, colors):
        if nl != n_levels:
            continue
        non_trivial = [b for b in range(n_levels) if b != trivial_bin]
        mean_ber = ber[non_trivial, :].mean(axis=0)
        label = f'pw_bins={pw_bins}  ({pw_bins * 1024} actions)'
        ax.semilogy(budgets, mean_ber + 1e-6, marker='o', markersize=4,
                    label=label, color=color)

    ax.set_xlabel('Write budget (pulses)', fontsize=11)
    ax.set_ylabel('BER  (fraction of failed writes)', fontsize=11)
    ax.set_title(f'BER vs. write budget — {n_levels}-level ({int(np.log2(n_levels))}-bit) cell',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(0, max(budgets))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    plt.tight_layout()
    out = OUT_DIR / f'exp_ber_vs_budget_lv{n_levels}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out.name}")


def plot_expected_pulses(results):
    """
    Bar chart: expected write pulses per (pw_bins, n_levels) config, averaged over non-trivial bins.
    """
    labels, means, stds = [], [], []
    for pw_bins, n_levels, ber, pulse_counts in results:
        membership  = bin_membership(n_levels)
        trivial_bin = membership[START_STATE]
        non_trivial = [b for b in range(n_levels) if b != trivial_bin]
        # Expected pulses = mean over successful rollouts only
        pc = pulse_counts[non_trivial, :]
        to = pc >= MAX_STEPS
        success_counts = np.where(to, np.nan, pc.astype(float))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = np.nanmean(success_counts)
            s = np.nanstd(success_counts)
        labels.append(f'pw={pw_bins}\nlv={n_levels}')
        means.append(m)
        stds.append(s)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=plt.cm.viridis(np.linspace(0.2, 0.8, len(labels))),
                  edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Expected write pulses', fontsize=11)
    ax.set_title('Expected write pulses by configuration\n(mean ± std over non-trivial bins, from HRS)',
                 fontsize=11)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{m:.1f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    out = OUT_DIR / 'exp_expected_pulses.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out.name}")


def plot_ber_heatmap(results):
    """
    Heatmap of BER at a fixed representative budget for all (pw_bins × n_levels) configs.
    """
    representative_budget = 50
    budget_idx = min(range(len(BUDGETS)), key=lambda i: abs(BUDGETS[i] - representative_budget))

    pw_list = sorted(set(r[0] for r in results))
    lv_list = sorted(set(r[1] for r in results))
    grid = np.full((len(lv_list), len(pw_list)), np.nan)

    for pw_bins, n_levels, ber, _ in results:
        membership  = bin_membership(n_levels)
        trivial_bin = membership[START_STATE]
        non_trivial = [b for b in range(n_levels) if b != trivial_bin]
        mean_ber = ber[non_trivial, budget_idx].mean()
        i = lv_list.index(n_levels)
        j = pw_list.index(pw_bins)
        grid[i, j] = mean_ber

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(grid, cmap='RdYlGn_r', vmin=0, vmax=grid[np.isfinite(grid)].max() * 1.1,
                   aspect='auto')
    ax.set_xticks(range(len(pw_list)))
    ax.set_xticklabels([f'pw_bins={p}' for p in pw_list])
    ax.set_yticks(range(len(lv_list)))
    ax.set_yticklabels([f'{lv}-level' for lv in lv_list])
    ax.set_title(f'BER @ {BUDGETS[budget_idx]}-pulse budget', fontsize=12)
    plt.colorbar(im, ax=ax, label='BER')
    for i in range(len(lv_list)):
        for j in range(len(pw_list)):
            ax.text(j, i, f'{grid[i,j]:.3f}', ha='center', va='center',
                    fontsize=10, color='black')
    plt.tight_layout()
    out = OUT_DIR / 'exp_ber_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out.name}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MLC programming experiments.')
    parser.add_argument('--rebuild',  action='store_true',
                        help='Force rebuild of all cached transition matrices and policies')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS,
                        help=f'MC trials per (start, target_bin) (default: {N_TRIALS})')
    args = parser.parse_args()

    rng_seed = SEED
    results  = []   # list of (pw_bins, n_levels, ber [n_levels, budgets], pulse_counts [n_levels, trials])

    for pw_bins in PW_BINS_LIST:
        print(f"\n{'='*60}")
        print(f"pw_bins = {pw_bins}  ({pw_bins * 1024} total actions)")
        print(f"{'='*60}")

        probs = ensure_transition_matrix(pw_bins, rebuild=args.rebuild)

        for n_levels in N_LEVELS_LIST:
            print(f"\n  --- n_levels={n_levels} ({int(np.log2(n_levels))}-bit) ---")

            V, policy = ensure_bin_policy(pw_bins, n_levels, rebuild=args.rebuild)
            membership = bin_membership(n_levels)

            mc_path = OUT_DIR / f'exp_mc_pw{pw_bins}_lv{n_levels}.npy'
            if mc_path.exists() and not args.rebuild:
                print(f"  [cached] {mc_path.name}")
                pulse_counts = np.load(mc_path)
                to_path = OUT_DIR / f'exp_mc_to_pw{pw_bins}_lv{n_levels}.npy'
                timed_out = np.load(to_path)
            else:
                print(f"  Running MC ({args.n_trials} trials per bin)...")
                pulse_counts, final_bin, timed_out = run_bin_mc(
                    probs, policy, membership,
                    start_state=START_STATE,
                    n_trials=args.n_trials,
                    max_steps=MAX_STEPS,
                    seed=rng_seed,
                )
                np.save(mc_path, pulse_counts)
                np.save(OUT_DIR / f'exp_mc_to_pw{pw_bins}_lv{n_levels}.npy', timed_out)

            ber = compute_ber_vs_budget(pulse_counts, timed_out, BUDGETS)
            results.append((pw_bins, n_levels, ber, pulse_counts))

            # Per-bin summary
            trivial_bin = membership[START_STATE]
            print(f"  Results (start=state {START_STATE}, {args.n_trials} trials):")
            for b in range(n_levels):
                pc = pulse_counts[b]
                to = timed_out[b]
                success = pc[~to]
                if b == trivial_bin:
                    print(f"    Bin {b} (HRS, trivial): 0 pulses by definition")
                    continue
                ber_50  = (pc > 50).mean()
                ber_100 = (pc > 100).mean()
                print(f"    Bin {b}: mean={success.mean():.1f}  p90={np.percentile(success,90):.0f}"
                      f"  BER@50={ber_50:.4f}  BER@100={ber_100:.4f}"
                      f"  timeout={to.mean()*100:.1f}%")

    # ── figures ───────────────────────────────────────────────────────────────
    print("\nGenerating comparison figures...")
    for n_levels in N_LEVELS_LIST:
        plot_ber_vs_budget(results, n_levels, BUDGETS)
    plot_expected_pulses(results)
    plot_ber_heatmap(results)

    # Summary CSV
    rows = []
    for pw_bins, n_levels, ber, pulse_counts in results:
        membership  = bin_membership(n_levels)
        trivial_bin = membership[START_STATE]
        for b in range(n_levels):
            if b == trivial_bin:
                continue
            pc = pulse_counts[b]
            to_mask = pc >= MAX_STEPS
            success = pc[~to_mask].astype(float)
            row = {
                'pw_bins': pw_bins,
                'n_levels': n_levels,
                'target_bin': b,
                'mean_pulses': float(np.nanmean(success)) if len(success) else np.nan,
                'p90_pulses':  float(np.nanpercentile(success, 90)) if len(success) else np.nan,
                'p99_pulses':  float(np.nanpercentile(success, 99)) if len(success) else np.nan,
                'timeout_rate': float(to_mask.mean()),
            }
            for budget in [10, 20, 50, 100, 200]:
                row[f'ber_at_{budget}'] = float(((pc > budget) | to_mask).mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'exp_summary.csv', index=False)
    print("Saved exp_summary.csv")
    print("\nDone.")
