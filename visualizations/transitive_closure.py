"""
transitive_closure.py

Computes multi-step reachability from the observed single-step transitions.
If s -> a is observed AND a -> b is observed, then b is reachable from s in 2
steps — even if that exact (s, b) pair was never directly measured.

Uses both setsweep (SET) and resetsweep (RESET). Pulses of either type can
be used at each step (i.e., the chain can mix SET and RESET freely).

Outputs:
  - closure_reachability.png  : which (s, s') pairs are reachable in N steps
  - closure_by_steps.png      : how reachability grows as step limit increases
  - closure_gaps.png          : states that are NEVER reachable from each start
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent

S = 65  # resistance states 0–64


# ── data loading (same as single_step_reachability.py) ──────────────────────
# NOTE: vbsl and pw ranges differ between chips — chip1 uses vbsl step=8 and
# 12 pw values; chip2 uses vbsl step=4 and 13 pw values (adds pw=3968).
# Safe to merge here since we only use (s, s_next) pairs for reachability.

def load_transitions(filepath):
    df = pd.read_csv(filepath, sep='\t', header=None, on_bad_lines='skip')
    ncols = len(df.columns)
    n_cells = (ncols - 5) // 2
    df.columns = (
        ['addr', 't', 'vwl', 'vbsl', 'pw']
        + [f'gi{i}' for i in range(n_cells)]
        + (['artifact'] if (ncols - 5) % 2 == 1 else [])
        + [f'gf{i}' for i in range(n_cells)]
    )
    records = []
    for _, row in df.iterrows():
        for k in range(n_cells):
            s      = int(row[f'gi{k}'])
            s_next = int(row[f'gf{k}'])
            if 0 <= s < S and 0 <= s_next < S:
                records.append((s, s_next))
    return records


def load_all(filepaths):
    """Concatenate transitions from multiple sweep files."""
    all_records = []
    for f in filepaths:
        all_records.extend(load_transitions(f))
    return all_records


def build_adjacency(transitions):
    """Bool adjacency matrix A[s, s'] = True if (s->s') ever observed."""
    A = np.zeros((S, S), dtype=bool)
    for s, sn in transitions:
        A[s, sn] = True
    return A


# ── transitive closure via repeated matrix OR-multiplication ─────────────────

def transitive_closure(A, max_steps=64):
    """
    Compute reachability[s, s', k] = can s reach s' in exactly ≤k steps.
    Returns:
      reach[k] : bool (S,S) matrix for step limit k  (k=1..max_steps)
      new_at[k]: number of NEW (s,s') pairs first reachable at step k
    """
    reach = A.copy()           # step 1: direct transitions
    history = [None, reach.copy()]   # history[k] = reachability at step k
    new_at  = [0, int(reach.sum())]

    for step in range(2, max_steps + 1):
        # reach_new[s, s'] = any intermediate t s.t. reach[s,t] AND A[t,s']
        reach_new = (reach @ A.astype(np.uint8)).astype(bool) | reach
        added = int(reach_new.sum()) - int(reach.sum())
        new_at.append(added)
        history.append(reach_new.copy())
        reach = reach_new
        if added == 0:
            # Converged: fill remaining steps with same matrix
            for _ in range(step + 1, max_steps + 1):
                history.append(reach_new.copy())
                new_at.append(0)
            break

    converged_at = next((k for k in range(1, len(new_at)) if new_at[k] == 0 and k > 1), max_steps)
    return history, new_at, converged_at


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_closure_reachability(single_step, full_closure, ax):
    """Show what's reachable in 1 step vs. full closure."""
    only_multi = full_closure & ~single_step
    never      = ~full_closure

    rgb = np.ones((S, S, 3))
    rgb[full_closure.T]  = [0.6, 0.8, 1.0]   # light blue = reachable (any steps)
    rgb[only_multi.T]    = [0.2, 0.6, 0.2]   # green = multi-step only
    rgb[single_step.T]   = [0.1, 0.3, 0.8]   # dark blue = single-step

    ax.imshow(rgb, origin='lower', aspect='equal', interpolation='none')
    ax.plot([0, S-1], [0, S-1], color='red', linewidth=0.8,
            linestyle='--', alpha=0.6)
    ax.set_xlabel('Initial resistance (s)', fontsize=11)
    ax.set_ylabel('Final resistance (s\')', fontsize=11)
    ax.set_title('Reachability: single-step vs. full transitive closure', fontsize=12)
    ax.set_xlim(-0.5, S - 0.5)
    ax.set_ylim(-0.5, S - 0.5)

    patches = [
        mpatches.Patch(color=[0.1, 0.3, 0.8], label='Reachable in 1 step'),
        mpatches.Patch(color=[0.2, 0.6, 0.2], label='Reachable in 2+ steps only'),
        mpatches.Patch(color='white',          label='Never reachable'),
    ]
    ax.legend(handles=patches, loc='upper left', fontsize=9, framealpha=0.9)

    n_single = int(single_step.sum())
    n_multi  = int(only_multi.sum())
    n_never  = int(never.sum())
    ax.text(0.98, 0.02,
            f"1-step:       {n_single:4d} pairs\n"
            f"2+-step only: {n_multi:4d} pairs\n"
            f"Never:        {n_never:4d} pairs",
            transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))


def plot_growth(new_at, converged_at, ax):
    """Cumulative reachable pairs as step count increases."""
    steps = np.arange(len(new_at))
    cumulative = np.cumsum(new_at)

    ax.plot(steps[1:], cumulative[1:], 'o-', color='steelblue',
            markersize=4, linewidth=1.8)
    ax.axvline(converged_at, color='red', linestyle='--', linewidth=1,
               label=f'Converged at step {converged_at}')
    ax.axhline(S * S, color='gray', linestyle=':', linewidth=1,
               label=f'Full coverage ({S}×{S}={S*S})')
    ax.set_xlabel('Max pulse steps allowed', fontsize=11)
    ax.set_ylabel('Cumulative reachable (s → s\') pairs', fontsize=11)
    ax.set_title('How reachability grows with more steps', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, min(converged_at + 3, len(new_at) - 1))
    ax.grid(True, alpha=0.3)


def plot_gaps(full_closure, ax):
    """For each starting state, how many targets are NEVER reachable."""
    unreachable_count = (~full_closure).sum(axis=1)  # per initial state
    colors = ['#d73027' if u > 0 else '#4dac26' for u in unreachable_count]
    ax.bar(np.arange(S), unreachable_count, color=colors, width=0.8)
    ax.set_xlabel('Initial resistance (s)', fontsize=11)
    ax.set_ylabel('# target states NEVER reachable', fontsize=11)
    ax.set_title('Unreachable targets per starting state\n(red = has gaps, green = fully connected)', fontsize=12)
    ax.set_xlim(-1, S)
    ax.axhline(0, color='black', linewidth=0.5)

    n_fully_connected = int((unreachable_count == 0).sum())
    ax.text(0.98, 0.97,
            f"Fully connected states: {n_fully_connected}/{S}",
            transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    set_files   = [DATA_DIR / 'setsweep1.csv',  DATA_DIR / 'setsweep2.csv']
    reset_files = [DATA_DIR / 'resetsweep1.csv', DATA_DIR / 'resetsweep2.csv']

    print("Loading SET sweeps (chip1 + chip2)...")
    t_set = load_all(set_files)
    print(f"  {len(t_set):,} transitions")

    print("Loading RESET sweeps (chip1 + chip2)...")
    t_rst = load_all(reset_files)
    print(f"  {len(t_rst):,} transitions")

    # Combined adjacency: a pulse can be SET or RESET at each step
    A_set  = build_adjacency(t_set)
    A_rst  = build_adjacency(t_rst)
    A_both = A_set | A_rst
    single_step = A_both.copy()

    print("\nComputing transitive closure (combined SET + RESET)...")
    history, new_at, converged_at = transitive_closure(A_both, max_steps=64)
    full_closure = history[converged_at]
    print(f"  Converged at step {converged_at}")
    print(f"  Total reachable pairs: {int(full_closure.sum())} / {S*S}")
    print(f"  Fully connected states (can reach all 65): {int((full_closure.sum(axis=1)==S).sum())}/{S}")

    # Per-step breakdown
    print("\n  Step-by-step new pairs:")
    for k in range(1, converged_at + 1):
        print(f"    step {k:2d}: +{new_at[k]:4d} new pairs  (cumulative: {sum(new_at[:k+1]):5d})")

    # --- Figure 1: single-step vs closure ---
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_closure_reachability(single_step, full_closure, ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'closure_reachability.png', dpi=150, bbox_inches='tight')
    print("\nSaved closure_reachability.png")

    # --- Figure 2: growth curve ---
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_growth(new_at, converged_at, ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'closure_by_steps.png', dpi=150, bbox_inches='tight')
    print("Saved closure_by_steps.png")

    # --- Figure 3: gaps per starting state ---
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_gaps(full_closure, ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'closure_gaps.png', dpi=150, bbox_inches='tight')
    print("Saved closure_gaps.png")
