"""
run_experiments.py

Experiment orchestrator: sweeps (pw_bins × n_levels × bin_type) configurations
and produces BER and write-latency comparison figures.

Experiment grid (configurable at top of file):
  pw_bins   ∈ [1, 2, 4]        — pulse-width bin count in the action space
  n_levels  ∈ [4, 8]            — resistance levels (2-bit, 3-bit)
  bin_types ∈ ['equal', 'pba']  — equal-width vs. PBA write-window bins

PBA bins are 4–5 ADC states wide (matching real hardware write tolerance),
creating large gaps between windows.  BER is non-trivial and comparable to
RADAR/PBA results from the literature.

Two BER metrics are reported:
  write_ber — fraction of MC rollouts that fail to land in the write window
              within the write budget (what our MDP directly controls)
  read_ber  — fraction where the post-drift ADC readout falls outside the
              read window (what RADAR/PBA report; requires drift model)

Drift model: for each final write state s, sample a post-drift state from the
empirical retention distribution P(readout | write ≈ s) at t = 1 s.

Usage:
  python run_experiments.py             # run all configs, skip cached
  python run_experiments.py --rebuild   # force rebuild of all caches
  python run_experiments.py --bin-type pba   # only PBA bins
  python run_experiments.py --n-trials 2000
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

OUT_DIR   = Path(__file__).parent
DATA_DIR  = OUT_DIR.parent / 'data'
MODEL_DIR = OUT_DIR.parent / 'model'

# ── experiment grid ───────────────────────────────────────────────────────────

PW_BINS_LIST  = [1, 2, 4]
N_LEVELS_LIST = [4, 8]
BIN_TYPES     = ['equal', 'pba']   # set to ['pba'] to only run PBA bins

# Cell-to-cell variability sweep (ADC units, additive offset std dev)
# sigma=0 → ideal pooled model (current baseline, BER=0)
# sigma=2 → moderate variability (~1 write window width)
# sigma=4 → strong variability (~1 write window width for 8-level tight bins)
SIGMA_CELL_LIST = [0, 1, 2, 3, 4]

START_STATE = 2              # HRS — state 0 has no valid outgoing actions
N_TRIALS    = 2_000
MAX_STEPS   = 1_500
SEED        = 42

BUDGETS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]

S = 65


# ── helpers ───────────────────────────────────────────────────────────────────

def get_equal_membership(n_levels):
    import sys; sys.path.insert(0, str(OUT_DIR.parent))
    from markov.bin_vi import make_bin_membership
    return make_bin_membership(n_levels), None, None


def get_pba_membership(n_levels):
    import sys; sys.path.insert(0, str(OUT_DIR.parent))
    from markov.bin_vi import make_pba_bins
    membership, write_windows, read_windows = make_pba_bins(n_levels)
    return membership, write_windows, read_windows


def get_membership(n_levels, bin_type):
    if bin_type == 'pba':
        return get_pba_membership(n_levels)
    return get_equal_membership(n_levels)


def build_drift_model():
    """
    Build empirical drift model P(readout | write_state) from retention data.
    Returns drift_table: list of 65 sorted readout arrays.
    drift_table[s] = sorted list of post-drift ADC values when writing to state s.
    """
    distributions = {}
    with open(MODEL_DIR / 'retention1s.csv') as f:
        for line in f:
            tokens = line.strip().split(',')
            tmin, tmax = int(tokens[0]), int(tokens[1])
            distr = sorted([int(x) for x in tokens[2:]])
            distributions[tmin] = distr   # key by tmin

    drift_table = []
    for s in range(S):
        key = min(max(s, 0), 59)
        drift_table.append(np.array(distributions.get(key, [s]), dtype=np.int32))
    return drift_table


def sample_drift(final_states, drift_table, rng):
    """
    For each final write state, sample a post-drift readout state.
    final_states: int array [N]
    Returns post_drift: int array [N]
    """
    post_drift = np.empty_like(final_states)
    for i, s in enumerate(final_states):
        distr = drift_table[int(s)]
        post_drift[i] = distr[rng.integers(0, len(distr))]
    return post_drift


def ensure_transition_matrix(pw_bins, rebuild=False):
    path = OUT_DIR / f'transition_probs_pw{pw_bins}.npy'
    if path.exists() and not rebuild:
        print(f"  [cached] {path.name}")
        return np.load(path)
    print(f"  Building transition matrix (pw_bins={pw_bins})...")
    subprocess.run(
        [sys.executable, str(OUT_DIR / 'build_chain.py'), '--pw-bins', str(pw_bins)],
        check=True
    )
    return np.load(path)


def ensure_bin_policy(pw_bins, n_levels, bin_type, rebuild=False):
    suffix = f'_pw{pw_bins}_lv{n_levels}_{bin_type}'
    v_path = OUT_DIR / f'bin_value{suffix}.npy'
    p_path = OUT_DIR / f'bin_policy{suffix}.npy'
    if v_path.exists() and p_path.exists() and not rebuild:
        print(f"  [cached] {v_path.name}")
        return np.load(v_path), np.load(p_path)
    print(f"  Running bin value iteration (pw_bins={pw_bins}, n_levels={n_levels}, bin_type={bin_type})...")
    subprocess.run(
        [sys.executable, str(OUT_DIR / 'bin_vi.py'),
         '--pw-bins', str(pw_bins), '--n-levels', str(n_levels), '--bin-type', bin_type],
        check=True
    )
    return np.load(v_path), np.load(p_path)


# ── Monte Carlo (bin-level, from fixed start state) ───────────────────────────

def run_bin_mc(probs, policy, membership, start_state, n_trials, max_steps, seed,
               sigma_cell=0.0):
    """
    Run MC rollouts starting from `start_state`, targeting each bin in turn.

    sigma_cell : float
        Standard deviation of a fixed per-cell conductance offset (in ADC units).
        Models cell-to-cell variability: each trial draws δ ~ N(0, sigma_cell²)
        once before its rollout begins. Every transition outcome is shifted by
        this fixed δ (clamped to [0, 64]).  The policy observes the shifted
        (actual) state and reacts, but its action-value estimates were computed
        under the unbiased model — so it partially but not perfectly adapts,
        producing non-zero BER for large sigma_cell.

    Returns
    -------
    pulse_counts  : int32 [n_levels, n_trials]
    final_states  : int32 [n_levels, n_trials]
    timed_out     : bool  [n_levels, n_trials]
    """
    rng = np.random.default_rng(seed)
    n_levels = policy.shape[1]

    pulse_counts = np.zeros((n_levels, n_trials), dtype=np.int32)
    final_states = np.full((n_levels, n_trials), start_state, dtype=np.int32)
    timed_out    = np.zeros((n_levels, n_trials), dtype=bool)

    # Draw fixed per-cell offsets once (same across bins for consistency)
    if sigma_cell > 0:
        cell_delta = np.round(rng.normal(0, sigma_cell, size=n_trials)).astype(np.int32)
    else:
        cell_delta = np.zeros(n_trials, dtype=np.int32)

    for b in range(n_levels):
        current = np.full(n_trials, start_state, dtype=np.int32)
        done    = np.zeros(n_trials, dtype=bool)

        if membership[start_state] == b:
            done[:] = True

        for step in range(max_steps):
            if done.all():
                break
            active = ~done
            a = policy[current[active], b]
            s = current[active]

            trans  = probs[a, s, :]
            cdf    = np.cumsum(trans, axis=1)
            u      = rng.uniform(size=active.sum())
            next_s = (cdf < u[:, None]).sum(axis=1)

            # Apply per-cell conductance offset and clamp to valid range
            next_s = np.clip(next_s + cell_delta[active], 0, S - 1).astype(np.int32)

            current[active]          = next_s
            pulse_counts[b, active] += 1
            done |= (membership[current] == b)

        timed_out[b]    = ~done
        final_states[b] = current
        pulse_counts[b, timed_out[b]] = max_steps

    return pulse_counts, final_states, timed_out


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
    One figure per n_levels: write BER (solid) and read BER (dashed) vs. write budget.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    styles  = {'equal': '-', 'pba': '--'}
    colors  = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c'}

    for (pw_bins, nl, bin_type, ber, read_ber, pulse_counts, ww, rw) in results:
        if nl != n_levels:
            continue
        membership, _, _ = get_membership(n_levels, bin_type)
        trivial = int(membership[START_STATE]) if membership[START_STATE] >= 0 else -1
        non_trivial = [b for b in range(n_levels) if b != trivial]
        if not non_trivial:
            continue
        mean_ber      = ber[non_trivial, :].mean(axis=0)
        mean_read_ber = read_ber[non_trivial, :].mean(axis=0)
        ls    = styles.get(bin_type, '-')
        color = colors.get(pw_bins, 'gray')
        label = f'{bin_type} pw={pw_bins}'
        ax.semilogy(budgets, mean_ber + 1e-6, ls=ls, marker='o', markersize=3,
                    color=color, label=f'{label} (write BER)')
        if bin_type == 'pba':
            ax.semilogy(budgets, mean_read_ber + 1e-6, ls=':', marker='s', markersize=3,
                        color=color, alpha=0.6, label=f'{label} (read BER+drift)')

    ax.set_xlabel('Write budget (pulses)', fontsize=11)
    ax.set_ylabel('BER', fontsize=11)
    ax.set_title(f'BER vs. write budget — {n_levels}-level ({int(np.log2(n_levels))}-bit) cell\n'
                 f'solid=write BER, dotted=read BER after drift  (pba bins only)', fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(0, max(budgets))

    plt.tight_layout()
    out = OUT_DIR / f'exp_ber_vs_budget_lv{n_levels}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out.name}")


def plot_expected_pulses(results):
    """
    Bar chart: expected write pulses per (pw_bins, n_levels, bin_type) config.
    """
    labels, means, stds = [], [], []
    for pw_bins, n_levels, bin_type, ber, read_ber, pulse_counts, ww, rw in results:
        membership, _, _ = get_membership(n_levels, bin_type)
        trivial_bin = int(membership[START_STATE]) if membership[START_STATE] >= 0 else -1
        non_trivial = [b for b in range(n_levels) if b != trivial_bin]
        pc = pulse_counts[non_trivial, :]
        to = pc >= MAX_STEPS
        success_counts = np.where(to, np.nan, pc.astype(float))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = np.nanmean(success_counts)
            s = np.nanstd(success_counts)
        labels.append(f'pw={pw_bins}\nlv={n_levels}\n{bin_type}')
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

    # One heatmap per bin_type
    for bt in set(r[2] for r in results):
        pw_list = sorted(set(r[0] for r in results if r[2] == bt))
        lv_list = sorted(set(r[1] for r in results if r[2] == bt))
        grid = np.full((len(lv_list), len(pw_list)), np.nan)

        for pw_bins, n_levels, bin_type, ber, read_ber, pulse_counts, ww, rw in results:
            if bin_type != bt:
                continue
            membership, _, _ = get_membership(n_levels, bin_type)
            trivial = int(membership[START_STATE]) if membership[START_STATE] >= 0 else -1
            non_trivial = [b for b in range(n_levels) if b != trivial]
            mean_ber = ber[non_trivial, budget_idx].mean()
            i = lv_list.index(n_levels)
            j = pw_list.index(pw_bins)
            grid[i, j] = mean_ber

        if np.all(np.isnan(grid)):
            continue

        fig, ax = plt.subplots(figsize=(5, 3.5))
        finite = grid[np.isfinite(grid)]
        vmax = finite.max() * 1.1 if len(finite) else 1.0
        im = ax.imshow(grid, cmap='RdYlGn_r', vmin=0, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(pw_list)))
        ax.set_xticklabels([f'pw={p}' for p in pw_list])
        ax.set_yticks(range(len(lv_list)))
        ax.set_yticklabels([f'{lv}-level' for lv in lv_list])
        ax.set_title(f'Write BER @ {BUDGETS[budget_idx]}-pulse budget ({bt} bins)', fontsize=11)
        plt.colorbar(im, ax=ax, label='Write BER')
        for i in range(len(lv_list)):
            for j in range(len(pw_list)):
                v = grid[i, j]
                ax.text(j, i, f'{v:.3f}' if np.isfinite(v) else 'N/A',
                        ha='center', va='center', fontsize=10)
        plt.tight_layout()
        out = OUT_DIR / f'exp_ber_heatmap_{bt}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {out.name}")


def plot_ber_vs_sigma(sigma_results, fixed_budget=50):
    """
    BER vs. sigma_cell for each (pw_bins, n_levels) at a fixed write budget.
    sigma_results: list of (pw_bins, n_levels, sigma, mean_ber_over_non_trivial_bins)
    One figure per n_levels.
    """
    for n_levels in N_LEVELS_LIST:
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c'}

        for pw_bins in PW_BINS_LIST:
            xs = [r[2] for r in sigma_results if r[0] == pw_bins and r[1] == n_levels]
            ys = [r[3] for r in sigma_results if r[0] == pw_bins and r[1] == n_levels]
            if not xs:
                continue
            ax.plot(xs, ys, marker='o', color=colors.get(pw_bins, 'gray'),
                    label=f'pw_bins={pw_bins} ({pw_bins*1024} actions)')

        ax.set_xlabel('σ_cell  (per-cell conductance offset, ADC units)', fontsize=11)
        ax.set_ylabel(f'Write BER  (budget = {fixed_budget} pulses)', fontsize=11)
        ax.set_title(f'BER vs. cell-to-cell variability — {n_levels}-level '
                     f'({int(np.log2(n_levels))}-bit), PBA bins\n'
                     f'Write window width = 4 ADC states', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.05)
        plt.tight_layout()
        out = OUT_DIR / f'exp_ber_vs_sigma_lv{n_levels}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {out.name}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MLC programming experiments.')
    parser.add_argument('--rebuild',   action='store_true')
    parser.add_argument('--n-trials',  type=int, default=N_TRIALS)
    parser.add_argument('--bin-type',  choices=['equal', 'pba', 'both'], default='both',
                        help='Which bin type(s) to run (default: both)')
    args = parser.parse_args()

    bin_types = ['equal', 'pba'] if args.bin_type == 'both' else [args.bin_type]

    drift_table = build_drift_model()
    drift_rng   = np.random.default_rng(SEED + 1)

    results = []  # (pw_bins, n_levels, bin_type, ber [n_levels, budgets],
                  #  read_ber [n_levels, budgets], pulse_counts, write_windows, read_windows)

    for bin_type in bin_types:
      for pw_bins in PW_BINS_LIST:
        print(f"\n{'='*60}")
        print(f"bin_type={bin_type}  pw_bins={pw_bins}  ({pw_bins * 1024} actions)")
        print(f"{'='*60}")

        probs = ensure_transition_matrix(pw_bins, rebuild=args.rebuild)

        for n_levels in N_LEVELS_LIST:
            print(f"\n  --- n_levels={n_levels} ({int(np.log2(n_levels))}-bit) ---")

            membership, write_windows, read_windows = get_membership(n_levels, bin_type)
            V, policy = ensure_bin_policy(pw_bins, n_levels, bin_type, rebuild=args.rebuild)

            mc_path = OUT_DIR / f'exp_mc_pw{pw_bins}_lv{n_levels}_{bin_type}.npy'
            fs_path = OUT_DIR / f'exp_mc_fs_pw{pw_bins}_lv{n_levels}_{bin_type}.npy'
            to_path = OUT_DIR / f'exp_mc_to_pw{pw_bins}_lv{n_levels}_{bin_type}.npy'

            if mc_path.exists() and not args.rebuild:
                print(f"  [cached] {mc_path.name}")
                pulse_counts  = np.load(mc_path)
                final_states  = np.load(fs_path)
                timed_out     = np.load(to_path)
            else:
                print(f"  Running MC ({args.n_trials} trials per bin)...")
                pulse_counts, final_states, timed_out = run_bin_mc(
                    probs, policy, membership,
                    start_state=START_STATE,
                    n_trials=args.n_trials,
                    max_steps=MAX_STEPS,
                    seed=SEED,
                )
                np.save(mc_path, pulse_counts)
                np.save(fs_path, final_states)
                np.save(to_path, timed_out)

            # Write BER vs budget
            ber = compute_ber_vs_budget(pulse_counts, timed_out, BUDGETS)

            # Read BER: apply drift to final states, check against read windows
            if read_windows is not None:
                read_ber = np.zeros((n_levels, len(BUDGETS)))
                for bi, budget in enumerate(BUDGETS):
                    for b in range(n_levels):
                        within_budget = ~timed_out[b] & (pulse_counts[b] <= budget)
                        if within_budget.sum() == 0:
                            read_ber[b, bi] = 1.0
                            continue
                        fs = final_states[b, within_budget]
                        post_drift = sample_drift(fs, drift_table, drift_rng)
                        rlow, rhigh = read_windows[b]
                        miss = (post_drift < rlow) | (post_drift >= rhigh)
                        # cells that timed out also count as failures
                        n_fail = miss.sum() + (~within_budget).sum()
                        read_ber[b, bi] = n_fail / args.n_trials
            else:
                read_ber = ber   # equal bins: read == write

            results.append((pw_bins, n_levels, bin_type, ber, read_ber,
                            pulse_counts, write_windows, read_windows))

            # Per-bin summary
            trivial_bin = int(membership[START_STATE]) if membership[START_STATE] >= 0 else -1
            print(f"  Results (start=state {START_STATE}):")
            for b in range(n_levels):
                pc = pulse_counts[b]; to = timed_out[b]
                success = pc[~to]
                if b == trivial_bin:
                    print(f"    Bin {b}: trivial (already in bin)")
                    continue
                if len(success) == 0:
                    print(f"    Bin {b}: all timed out")
                    continue
                ber_20  = (pc > 20).mean()
                ber_100 = (pc > 100).mean()
                print(f"    Bin {b}: mean={success.mean():.1f}  p90={np.percentile(success,90):.0f}"
                      f"  write_BER@20={ber_20:.4f}  write_BER@100={ber_100:.4f}"
                      f"  timeout={to.mean()*100:.1f}%")

    # ── sigma sweep (PBA bins only, fixed budget) ─────────────────────────────
    print("\nRunning cell-variability (sigma) sweep on PBA bins...")
    sigma_results = []   # (pw_bins, n_levels, sigma, mean_ber_at_budget50)
    budget_50_idx = min(range(len(BUDGETS)), key=lambda i: abs(BUDGETS[i] - 50))

    for pw_bins in PW_BINS_LIST:
        probs = ensure_transition_matrix(pw_bins, rebuild=False)
        for n_levels in N_LEVELS_LIST:
            membership, _, _ = get_membership(n_levels, 'pba')
            V, policy = ensure_bin_policy(pw_bins, n_levels, 'pba', rebuild=False)
            trivial = int(membership[START_STATE]) if membership[START_STATE] >= 0 else -1
            non_trivial = [b for b in range(n_levels) if b != trivial]

            for sigma in SIGMA_CELL_LIST:
                cache = OUT_DIR / f'exp_sigma_pw{pw_bins}_lv{n_levels}_s{sigma}.npy'
                cache_to = OUT_DIR / f'exp_sigma_to_pw{pw_bins}_lv{n_levels}_s{sigma}.npy'
                if cache.exists() and not args.rebuild:
                    pc = np.load(cache); to = np.load(cache_to)
                else:
                    print(f"  sigma={sigma}  pw={pw_bins}  lv={n_levels}")
                    pc, _, to = run_bin_mc(
                        probs, policy, membership,
                        start_state=START_STATE, n_trials=args.n_trials,
                        max_steps=MAX_STEPS, seed=SEED, sigma_cell=float(sigma),
                    )
                    np.save(cache, pc); np.save(cache_to, to)

                ber_at_50 = compute_ber_vs_budget(pc, to, BUDGETS)
                mean_ber = ber_at_50[non_trivial, budget_50_idx].mean()
                sigma_results.append((pw_bins, n_levels, sigma, mean_ber))
                print(f"  pw={pw_bins} lv={n_levels} σ={sigma}  BER@50={mean_ber:.4f}")

    plot_ber_vs_sigma(sigma_results, fixed_budget=BUDGETS[budget_50_idx])

    # ── figures ───────────────────────────────────────────────────────────────
    print("\nGenerating comparison figures...")
    for n_levels in N_LEVELS_LIST:
        plot_ber_vs_budget(results, n_levels, BUDGETS)
    plot_expected_pulses(results)
    plot_ber_heatmap(results)

    # Summary CSV
    rows = []
    for pw_bins, n_levels, bin_type, ber, read_ber, pulse_counts, ww, rw in results:
        membership  = get_membership(n_levels, bin_type)[0]
        trivial_bin = int(membership[START_STATE]) if membership[START_STATE] >= 0 else -1
        for b in range(n_levels):
            if b == trivial_bin:
                continue
            pc      = pulse_counts[b]
            to_mask = pc >= MAX_STEPS
            success = pc[~to_mask].astype(float)
            row = {
                'pw_bins': pw_bins, 'n_levels': n_levels, 'bin_type': bin_type,
                'target_bin': b,
                'mean_pulses': float(np.nanmean(success)) if len(success) else np.nan,
                'p90_pulses':  float(np.nanpercentile(success, 90)) if len(success) else np.nan,
                'p99_pulses':  float(np.nanpercentile(success, 99)) if len(success) else np.nan,
                'timeout_rate': float(to_mask.mean()),
            }
            for budget in [10, 20, 50, 100, 200]:
                row[f'write_ber_at_{budget}'] = float(((pc > budget) | to_mask).mean())
                bi = BUDGETS.index(budget) if budget in BUDGETS else None
                if bi is not None:
                    row[f'read_ber_at_{budget}'] = float(read_ber[b, bi])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'exp_summary.csv', index=False)
    print("Saved exp_summary.csv")
    print("\nDone.")
