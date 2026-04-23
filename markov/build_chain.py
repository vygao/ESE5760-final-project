"""
build_chain.py

Builds a transition probability matrix T[action, s, s'] from the four sweep files.

Action space — vbsl always binned into 4 groups; pw binning is configurable:

  N_PW_BIN = 1 (default, pools all pw):
    Actions   0–511:  SET,   index = vwl//2 * 4 + vbsl_bin
    Actions 512–1023: RESET, index = 512 + vwl//2 * 4 + vbsl_bin
    Total: 1024 actions

  N_PW_BIN = K (K > 1, log-scale pw bins):
    Actions [0, 512K):    SET,   index = vwl//2 * 4*K + vbsl_bin * K + pw_bin
    Actions [512K, 1024K): RESET, index = 512K + vwl//2 * 4*K + vbsl_bin * K + pw_bin
    Total: 1024*K actions

pw is binned on a log2 scale across the full observed range (1–128 ns).
NOTE: RESET sweeps have narrower pw coverage (chip1: 1–8 ns, chip2: 1–2 ns),
so pw binning primarily benefits SET actions.

Usage:
  python build_chain.py              # pw pooled (N_PW_BIN=1), saves transition_probs_pw1.npy
  python build_chain.py --pw-bins 2  # 2 pw bins
  python build_chain.py --pw-bins 4  # 4 pw bins

NOTE on SET vs RESET hardware distinction:
  SET   → controls BL (bitline) voltage   (bl_dac_set_lvl_cycle)
  RESET → controls SL (sourceline) voltage (sl_dac_rst_lvl_cycle)
  Current flows in opposite directions. Whether freely interleaving SET and RESET
  pulses is realizable on Ember hardware is an open question.

Output files (saved to markov/):
  transition_counts_pw{K}.npy  : int32   [1024*K, 65, 65]
  transition_probs_pw{K}.npy   : float64 [1024*K, 65, 65]
  action_info_pw{K}.csv        : action index → (type, vwl, vbsl_bin, pw_bin, pw_range)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent

S          = 65     # resistance states 0–64
N_VWL      = 128    # vwl ∈ {0,2,...,254} → index vwl//2
N_VBSL_BIN = 4      # vbsl bins: [0-7], [8-15], [16-23], [24-31]
MIN_COUNT  = 3      # minimum observations before trusting a transition row

# pw values observed across all sweep files: 1,2,4,8,16,32,64,128 (powers of 2)
# log2 values: 0,1,2,3,4,5,6,7  →  range [0,7], span = 8 steps
_PW_LOG2_SPAN = 8


def vbsl_to_bin(vbsl):
    """Map vbsl DAC code (0–31) to bin index 0–3."""
    return np.clip(vbsl // 8, 0, N_VBSL_BIN - 1)


def pw_to_bin(pw, n_pw_bins):
    """
    Map pulse width (ns, power-of-2) to log-scale bin index.

    With n_pw_bins=1 all pw collapse to bin 0 (pooled).
    With n_pw_bins=K the 8 log2-steps are divided evenly:
      bin = min(floor(log2(pw) * K / 8), K-1)

    pw=1  → log2=0   pw=2 → 1   pw=4 → 2   pw=8  → 3
    pw=16 → log2=4   pw=32→ 5   pw=64→ 6   pw=128→ 7
    """
    if n_pw_bins == 1:
        return np.zeros_like(pw, dtype=np.int32) if hasattr(pw, '__len__') else 0
    log2_pw = np.floor(np.log2(np.clip(pw, 1, None))).astype(np.int32)
    return np.clip(log2_pw * n_pw_bins // _PW_LOG2_SPAN, 0, n_pw_bins - 1)


def action_index(pulse_type, vwl, vbsl, pw, n_pw_bins):
    """
    Compute scalar or array action index.

    Layout per half (SET or RESET):
      vwl_idx * N_VBSL_BIN * n_pw_bins  +  vbsl_bin * n_pw_bins  +  pw_bin

    SET  actions: [0,                    N_VWL*N_VBSL_BIN*n_pw_bins)
    RESET actions: [N_VWL*N_VBSL_BIN*n_pw_bins, 2*N_VWL*N_VBSL_BIN*n_pw_bins)
    """
    stride = N_VBSL_BIN * n_pw_bins
    base   = (vwl // 2) * stride + vbsl_to_bin(vbsl) * n_pw_bins + pw_to_bin(pw, n_pw_bins)
    if pulse_type == 'reset':
        base = base + N_VWL * stride
    return base


# ── data loading ──────────────────────────────────────────────────────────────

def load_transitions(filepath, pulse_type, n_pw_bins):
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
    pw_vals   = np.repeat(df['pw'].values,   n_cells).astype(np.int32)

    valid = (gi_vals >= 0) & (gi_vals < S) & (gf_vals >= 0) & (gf_vals < S)
    gi_vals   = gi_vals[valid]
    gf_vals   = gf_vals[valid]
    vwl_vals  = vwl_vals[valid]
    vbsl_vals = vbsl_vals[valid]
    pw_vals   = pw_vals[valid]

    a_idx = action_index(pulse_type, vwl_vals, vbsl_vals, pw_vals, n_pw_bins)
    return a_idx, gi_vals, gf_vals


def build_count_matrix(file_pairs, n_pw_bins):
    n_actions = 2 * N_VWL * N_VBSL_BIN * n_pw_bins
    counts = np.zeros((n_actions, S, S), dtype=np.int32)
    for fpath, ptype in file_pairs:
        print(f"  Loading {fpath.name} ({ptype})...")
        a_idx, gi, gf = load_transitions(fpath, ptype, n_pw_bins)
        np.add.at(counts, (a_idx, gi, gf), 1)
        print(f"    → {len(gi):,} valid transitions")
    return counts


def normalize(counts, min_count=MIN_COUNT):
    """Row-normalize counts → probabilities. Rows with < min_count obs → zeroed."""
    row_totals = counts.sum(axis=2)
    valid = row_totals >= min_count
    row_totals_safe = np.where(valid, row_totals, 1)
    probs = counts.astype(np.float64) / row_totals_safe[:, :, np.newaxis]
    probs[~valid] = 0.0
    return probs


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build RRAM Markov transition matrix.')
    parser.add_argument('--pw-bins', type=int, default=1,
                        help='Number of pulse-width bins (1=pool all pw, 2/4=log-scale bins). Default: 1')
    args = parser.parse_args()
    n_pw_bins = args.pw_bins

    suffix = f'_pw{n_pw_bins}'
    n_actions = 2 * N_VWL * N_VBSL_BIN * n_pw_bins

    file_pairs = [
        (DATA_DIR / 'setsweep1.csv',   'set'),
        (DATA_DIR / 'setsweep2.csv',   'set'),
        (DATA_DIR / 'resetsweep1.csv', 'reset'),
        (DATA_DIR / 'resetsweep2.csv', 'reset'),
    ]

    print(f"Building transition count matrix [{n_actions}, {S}, {S}]  (pw_bins={n_pw_bins})...")
    counts = build_count_matrix(file_pairs, n_pw_bins)

    print(f"\nNormalizing to probabilities (MIN_COUNT={MIN_COUNT})...")
    probs = normalize(counts, MIN_COUNT)

    np.save(OUT_DIR / f'transition_counts{suffix}.npy', counts)
    np.save(OUT_DIR / f'transition_probs{suffix}.npy',  probs)
    print(f"Saved transition_counts{suffix}.npy and transition_probs{suffix}.npy")

    # Action metadata CSV
    vbsl_bin_ranges = ['0-7', '8-15', '16-23', '24-31']
    pw_bin_ranges   = [f'log2-bin-{b}' for b in range(n_pw_bins)]
    rows = []
    n_set_half = N_VWL * N_VBSL_BIN * n_pw_bins
    for ptype, offset in [('SET', 0), ('RESET', n_set_half)]:
        for vi in range(N_VWL):
            for bi in range(N_VBSL_BIN):
                for pi in range(n_pw_bins):
                    rows.append({
                        'action_idx': offset + vi * N_VBSL_BIN * n_pw_bins + bi * n_pw_bins + pi,
                        'type':       ptype,
                        'vwl':        vi * 2,
                        'vbsl_bin':   bi,
                        'vbsl_range': vbsl_bin_ranges[bi],
                        'pw_bin':     pi,
                    })
    pd.DataFrame(rows).to_csv(OUT_DIR / f'action_info{suffix}.csv', index=False)
    print(f"Saved action_info{suffix}.csv")

    # Summary stats
    valid_rows = (counts.sum(axis=2) >= MIN_COUNT)
    print(f"\n=== Action coverage (pw_bins={n_pw_bins}) ===")
    print(f"  Total (action, s) pairs with >= {MIN_COUNT} obs: {valid_rows.sum()} / {n_actions * S}")
    print(f"  SET   actions with any valid row: {valid_rows[:n_set_half].any(axis=1).sum()} / {n_set_half}")
    print(f"  RESET actions with any valid row: {valid_rows[n_set_half:].any(axis=1).sum()} / {n_set_half}")

    n_valid_actions = valid_rows.sum(axis=0)
    print(f"\n  States with 0 valid actions:  {(n_valid_actions == 0).sum()}")
    print(f"  Avg valid actions per state:  {n_valid_actions.mean():.1f}")
    print(f"  Min valid actions per state:  {n_valid_actions.min()} (state {n_valid_actions.argmin()})")
