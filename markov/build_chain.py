"""
build_chain.py

Builds a transition probability matrix T[action, s, s'] from the four sweep files.

Action space (1024 total) — expanded to include vbsl:
  vbsl is binned into 4 groups of width 8:
    bin 0: vbsl  0– 7
    bin 1: vbsl  8–15
    bin 2: vbsl 16–23
    bin 3: vbsl 24–31

  Actions   0–511: SET   pulses, index = vwl//2 * 4 + vbsl_bin
  Actions 512–1023: RESET pulses, index = 512 + vwl//2 * 4 + vbsl_bin

pw is still pooled (marginalized over) — sensitivity analysis showed pw has
~40% the effect of vwl for SET and ~154% for RESET.  Adding pw bins would
multiply the action space by another 4x with diminishing data density; left as
a future extension.

NOTE on SET vs RESET hardware distinction:
  Even though vwl, vbsl, pw DAC ranges are identical in the sweep data, SET and
  RESET drive different physical lines on the Ember chip:
    SET   → controls BL (bitline) voltage   (bl_dac_set_lvl_cycle)
    RESET → controls SL (sourceline) voltage (sl_dac_rst_lvl_cycle)
  Current flows in opposite directions, forming vs disrupting the filament.
  Whether freely interleaving SET and RESET pulses is realizable in a real
  programming sequence is an open hardware question.

Output files (saved to markov/):
  transition_counts.npy  : int32   [1024, 65, 65]  raw observation counts
  transition_probs.npy   : float64 [1024, 65, 65]  row-normalized probabilities
  action_info.csv        : action index → (type, vwl, vbsl_bin, vbsl_range)
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent

S          = 65     # resistance states 0–64
N_VWL      = 128    # vwl ∈ {0,2,...,254} → index vwl//2
N_VBSL_BIN = 4      # vbsl bins: [0-7], [8-15], [16-23], [24-31]
N_ACTIONS  = 2 * N_VWL * N_VBSL_BIN   # 1024: 512 SET + 512 RESET
MIN_COUNT  = 3      # minimum observations before trusting a transition row


def vbsl_to_bin(vbsl):
    """Map vbsl DAC code (0–31) to bin index 0–3."""
    return np.clip(vbsl // 8, 0, N_VBSL_BIN - 1)


def action_index(pulse_type, vwl, vbsl):
    """
    Compute scalar or array action index.
    SET:   vwl//2 * 4 + vbsl_bin          (0–511)
    RESET: 512 + vwl//2 * 4 + vbsl_bin   (512–1023)
    """
    base = (vwl // 2) * N_VBSL_BIN + vbsl_to_bin(vbsl)
    if pulse_type == 'reset':
        base = base + N_VWL * N_VBSL_BIN
    return base


# ── data loading ──────────────────────────────────────────────────────────────

def load_transitions(filepath, pulse_type):
    """
    Load one sweep CSV.  Returns (action_idx, s, s_next) int arrays.
    """
    df = pd.read_csv(filepath, sep='\t', header=None, on_bad_lines='skip')
    ncols = len(df.columns)
    n_cells = (ncols - 5) // 2
    colnames = (
        ['addr', 't', 'vwl', 'vbsl', 'pw']
        + [f'gi{i}' for i in range(n_cells)]
        + (['artifact'] if (ncols - 5) % 2 == 1 else [])
        + [f'gf{i}' for i in range(n_cells)]
    )
    df.columns = colnames

    gi_cols = [f'gi{i}' for i in range(n_cells)]
    gf_cols = [f'gf{i}' for i in range(n_cells)]

    gi_vals   = df[gi_cols].values.ravel().astype(np.int32)
    gf_vals   = df[gf_cols].values.ravel().astype(np.int32)
    vwl_vals  = np.repeat(df['vwl'].values,  n_cells).astype(np.int32)
    vbsl_vals = np.repeat(df['vbsl'].values, n_cells).astype(np.int32)

    valid = (gi_vals >= 0) & (gi_vals < S) & (gf_vals >= 0) & (gf_vals < S)
    gi_vals   = gi_vals[valid]
    gf_vals   = gf_vals[valid]
    vwl_vals  = vwl_vals[valid]
    vbsl_vals = vbsl_vals[valid]

    a_idx = action_index(pulse_type, vwl_vals, vbsl_vals)
    return a_idx, gi_vals, gf_vals


def build_count_matrix(file_pairs):
    counts = np.zeros((N_ACTIONS, S, S), dtype=np.int32)
    for fpath, ptype in file_pairs:
        print(f"  Loading {fpath.name} ({ptype})...")
        a_idx, gi, gf = load_transitions(fpath, ptype)
        np.add.at(counts, (a_idx, gi, gf), 1)
        print(f"    → {len(gi):,} valid transitions")
    return counts


def normalize(counts, min_count=MIN_COUNT):
    """Row-normalize counts → probabilities. Rows with < min_count obs → zeroed."""
    row_totals = counts.sum(axis=2)                        # [N_ACTIONS, S]
    valid = row_totals >= min_count
    row_totals_safe = np.where(valid, row_totals, 1)
    probs = counts.astype(np.float64) / row_totals_safe[:, :, np.newaxis]
    probs[~valid] = 0.0
    return probs


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    file_pairs = [
        (DATA_DIR / 'setsweep1.csv',   'set'),
        (DATA_DIR / 'setsweep2.csv',   'set'),
        (DATA_DIR / 'resetsweep1.csv', 'reset'),
        (DATA_DIR / 'resetsweep2.csv', 'reset'),
    ]

    print(f"Building transition count matrix [{N_ACTIONS}, {S}, {S}]...")
    counts = build_count_matrix(file_pairs)

    print(f"\nNormalizing to probabilities (MIN_COUNT={MIN_COUNT})...")
    probs = normalize(counts, MIN_COUNT)

    np.save(OUT_DIR / 'transition_counts.npy', counts)
    np.save(OUT_DIR / 'transition_probs.npy',  probs)
    print("Saved transition_counts.npy and transition_probs.npy")

    # Action metadata CSV
    vbsl_bin_ranges = ['0-7', '8-15', '16-23', '24-31']
    rows = []
    for ptype, offset in [('SET', 0), ('RESET', N_VWL * N_VBSL_BIN)]:
        for vi in range(N_VWL):
            for bi in range(N_VBSL_BIN):
                rows.append({
                    'action_idx': offset + vi * N_VBSL_BIN + bi,
                    'type':       ptype,
                    'vwl':        vi * 2,
                    'vbsl_bin':   bi,
                    'vbsl_range': vbsl_bin_ranges[bi],
                })
    pd.DataFrame(rows).to_csv(OUT_DIR / 'action_info.csv', index=False)
    print("Saved action_info.csv")

    # Summary stats
    valid_rows = (counts.sum(axis=2) >= MIN_COUNT)   # [N_ACTIONS, S]
    n_set_half   = N_VWL * N_VBSL_BIN
    print(f"\n=== Action coverage ===")
    print(f"  Total (action, s) pairs with >= {MIN_COUNT} obs: {valid_rows.sum()} / {N_ACTIONS * S}")
    print(f"  SET   actions with any valid row: {valid_rows[:n_set_half].any(axis=1).sum()} / {n_set_half}")
    print(f"  RESET actions with any valid row: {valid_rows[n_set_half:].any(axis=1).sum()} / {n_set_half}")

    n_valid_actions = valid_rows.sum(axis=0)   # [65]
    print(f"\n  States with 0 valid actions:  {(n_valid_actions == 0).sum()}")
    print(f"  Avg valid actions per state:  {n_valid_actions.mean():.1f}")
    print(f"  Min valid actions per state:  {n_valid_actions.min()} (state {n_valid_actions.argmin()})")

    # Per vbsl-bin: how many (action, s) rows are valid
    print(f"\n=== Coverage by vbsl bin ===")
    for bi, rng in enumerate(vbsl_bin_ranges):
        set_idxs   = [vi * N_VBSL_BIN + bi for vi in range(N_VWL)]
        rst_idxs   = [n_set_half + vi * N_VBSL_BIN + bi for vi in range(N_VWL)]
        set_valid  = valid_rows[set_idxs].sum()
        rst_valid  = valid_rows[rst_idxs].sum()
        print(f"  vbsl {rng:>5s}  SET valid rows: {set_valid:5d}/{N_VWL*S}   "
              f"RESET valid rows: {rst_valid:5d}/{N_VWL*S}")
