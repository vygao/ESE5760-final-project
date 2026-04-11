"""
single_step_reachability.py

For each observed initial resistance state, maps every final resistance state
reachable in exactly ONE pulse — across all pulse parameters (vwl, vbsl, pw).
Uses both setsweep (SET pulses, resistance goes down) and resetsweep (RESET
pulses, resistance goes up).

Outputs:
  - transition_map_set.png   : single-step reachability for SET pulses
  - transition_map_reset.png : single-step reachability for RESET pulses
  - transition_map_both.png  : combined SET/RESET overlay
  - transition_coverage.png  : per-state summary (how many final states reachable)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA_DIR  = Path(__file__).parent.parent / "data"
OUT_DIR   = Path(__file__).parent

S = 65  # resistance states 0-64


def load_transitions(filepath, chip=None):
    """Load sweep CSV and return DataFrame of (s, s_next, vwl, vbsl, pw, chip).

    Column layout detected automatically:
      - 36 cols: 5 header + 15 gi + 1 artifact + 15 gf  (setsweep1)
      - 37 cols: 5 header + 16 gi + 16 gf               (all others)

    NOTE: vbsl and pw ranges differ between chips — chip1 uses vbsl step=8
    and 12 pw values; chip2 uses vbsl step=4 and 13 pw values (adds pw=3968).
    Safe to merge for reachability analysis; condition on chip if needed later.
    """
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
                records.append((s, s_next, row['vwl'], row['vbsl'], row['pw'], chip))
    return pd.DataFrame(records, columns=['s', 's_next', 'vwl', 'vbsl', 'pw', 'chip'])


def load_all(set_files, reset_files):
    """Load and concatenate multiple SET and RESET sweep files."""
    set_parts   = [load_transitions(f, chip=f.stem) for f in set_files]
    reset_parts = [load_transitions(f, chip=f.stem) for f in reset_files]
    t_set = pd.concat(set_parts,   ignore_index=True)
    t_rst = pd.concat(reset_parts, ignore_index=True)
    return t_set, t_rst


def build_reachability(transitions):
    """Return (reachable[S,S] bool, count[S,S] int) matrices."""
    count = np.zeros((S, S), dtype=np.int32)
    for _, row in transitions.iterrows():
        count[int(row['s']), int(row['s_next'])] += 1
    reachable = count > 0
    return reachable, count


def plot_reachability(reachable, count, title, ax, cmap='Blues', show_count=True):
    """Heatmap of observed transitions; diagonal shown in red."""
    display = count.astype(float)
    display[display == 0] = np.nan

    im = ax.imshow(display.T, origin='lower', aspect='equal',
                   cmap=cmap, interpolation='none')
    # Diagonal: no change
    ax.plot([0, S-1], [0, S-1], color='red', linewidth=0.8, alpha=0.5,
            linestyle='--', label='no change')
    ax.set_xlabel('Initial condutance (s)', fontsize=11)
    ax.set_ylabel('Final conductance (s\')', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.5, S - 0.5)
    ax.set_ylim(-0.5, S - 0.5)
    plt.colorbar(im, ax=ax, label='# transitions observed', shrink=0.8)

    # Annotate coverage
    observed_s   = reachable.any(axis=1).sum()
    observed_s_next = reachable.any(axis=0).sum()
    total_pairs  = reachable.sum()
    ax.text(0.02, 0.97,
            f"Initial states observed: {observed_s}/65\n"
            f"Final states reached:    {observed_s_next}/65\n"
            f"Unique (s→s') pairs:     {total_pairs}",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def plot_combined(reach_set, count_set, reach_rst, count_rst, ax):
    """Overlay SET (blue, alpha) and RESET (orange, alpha) — overlap blends to brown/purple."""
    # Two RGBA layers composited via two imshow calls
    rgba_rst = np.zeros((S, S, 4))
    rgba_rst[reach_rst.T] = [0.95, 0.45, 0.05, 0.6]  # orange, semi-transparent

    rgba_set = np.zeros((S, S, 4))
    rgba_set[reach_set.T] = [0.15, 0.35, 0.90, 0.6]  # blue, semi-transparent

    ax.set_facecolor('white')
    ax.imshow(rgba_rst, origin='lower', aspect='equal', interpolation='none')
    ax.imshow(rgba_set, origin='lower', aspect='equal', interpolation='none')
    ax.plot([0, S-1], [0, S-1], color='red', linewidth=0.8, alpha=0.6,
            linestyle='--')
    ax.set_xlabel('Initial conductance (s)', fontsize=11)
    ax.set_ylabel('Final conductance (s\')', fontsize=11)
    ax.set_title('All observed single-step transitions: SET vs RESET\n(overlap = both)', fontsize=12)
    ax.set_xlim(-0.5, S - 0.5)
    ax.set_ylim(-0.5, S - 0.5)

    patches = [
        mpatches.Patch(color=[0.15, 0.35, 0.90], label='SET only'),
        mpatches.Patch(color=[0.95, 0.45, 0.05], label='RESET only'),
        mpatches.Patch(color=[0.45, 0.35, 0.50], label='Both (overlap)'),
        mpatches.Patch(color='white', ec='lightgray', label='Never observed'),
    ]
    ax.legend(handles=patches, loc='upper left', fontsize=8, framealpha=0.9)


def plot_coverage_summary(reach_set, reach_rst, ax_top, ax_bot):
    """Per-state bar charts: how many final states are reachable."""
    states = np.arange(S)

    n_reachable_set = reach_set.sum(axis=1)   # for each s, how many s' via SET
    n_reachable_rst = reach_rst.sum(axis=1)   # for each s, how many s' via RESET

    # How many initial states have at least 1 observation
    observed_set = reach_set.any(axis=1)
    observed_rst = reach_rst.any(axis=1)

    ax_top.bar(states[observed_set], n_reachable_set[observed_set],
               color='steelblue', alpha=0.8, width=0.8, label='SET')
    ax_top.set_ylabel('# distinct s\' reachable')
    ax_top.set_title('SET pulses: reachable final states per initial state')
    ax_top.set_xlim(-1, S)
    ax_top.legend()

    ax_bot.bar(states[observed_rst], n_reachable_rst[observed_rst],
               color='darkorange', alpha=0.8, width=0.8, label='RESET')
    ax_bot.set_xlabel('Initial resistance (s)')
    ax_bot.set_ylabel('# distinct s\' reachable')
    ax_bot.set_title('RESET pulses: reachable final states per initial state')
    ax_bot.set_xlim(-1, S)
    ax_bot.legend()


if __name__ == '__main__':
    set_files   = [DATA_DIR / 'setsweep1.csv',   DATA_DIR / 'setsweep2.csv']
    reset_files = [DATA_DIR / 'resetsweep1.csv',  DATA_DIR / 'resetsweep2.csv']

    print("Loading SET sweeps (chip1 + chip2)...")
    t_set, t_rst = load_all(set_files, reset_files)
    print(f"  SET:   {len(t_set):,} transitions")
    print(f"  RESET: {len(t_rst):,} transitions")

    reach_set, count_set = build_reachability(t_set)
    reach_rst, count_rst = build_reachability(t_rst)

    # --- Combined SET + RESET ---
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_combined(reach_set, count_set, reach_rst, count_rst, ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'transition_map_both.png', dpi=150, bbox_inches='tight')
    print("Saved transition_map_both.png")

    # --- Per-state coverage summary ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    plot_coverage_summary(reach_set, reach_rst, ax1, ax2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'transition_coverage.png', dpi=150, bbox_inches='tight')
    print("Saved transition_coverage.png")

    # --- Print summary stats ---
    print("\n=== Summary ===")
    for label, reach, count in [('SET', reach_set, count_set), ('RESET', reach_rst, count_rst)]:
        init_states  = reach.any(axis=1).sum()
        final_states = reach.any(axis=0).sum()
        pairs        = reach.sum()
        self_loops   = np.diag(reach).sum()
        print(f"\n{label}:")
        print(f"  Initial states with any observation : {init_states}/65")
        print(f"  Final states ever reached           : {final_states}/65")
        print(f"  Unique (s → s') pairs               : {pairs}")
        print(f"  Self-loops (s → s, no change)       : {self_loops}")
        print(f"  Avg # final states per initial state: {reach.sum(axis=1)[reach.any(axis=1)].mean():.1f}")
