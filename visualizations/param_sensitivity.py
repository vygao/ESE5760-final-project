"""
param_sensitivity.py

Answers: does pooling over vbsl and pw lose significant information?

For a fixed vwl, we measure how much the mean final state (and its distribution)
changes as we vary vbsl and pw independently.  If large → pooling is lossy and
we should expand the action space.  If small → pooling is a safe approximation.

Methodology
-----------
For each (pulse_type, vwl, s_initial) triplet, compute mean(s_final) separately
for each vbsl value and each pw value.  The "spread" = max - min across parameter
values quantifies how much vbsl/pw matter beyond vwl alone.

Outputs
-------
  param_sensitivity_set.png   : SET  — vbsl effect + pw effect per vwl
  param_sensitivity_reset.png : RESET — same
  param_spread_summary.png    : Spread of mean(Δs) by vbsl vs pw vs vwl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent

S = 65


# ── data loading ──────────────────────────────────────────────────────────────

def load_sweep(filepath):
    """Return DataFrame with columns: vwl, vbsl, pw, s, s_next, delta."""
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

    gi  = df[gi_cols].values.ravel().astype(np.int32)
    gf  = df[gf_cols].values.ravel().astype(np.int32)
    vwl  = np.repeat(df['vwl'].values,  n_cells).astype(np.int32)
    vbsl = np.repeat(df['vbsl'].values, n_cells).astype(np.int32)
    pw   = np.repeat(df['pw'].values,   n_cells).astype(np.int32)

    valid = (gi >= 0) & (gi < S) & (gf >= 0) & (gf < S)
    return pd.DataFrame({
        'vwl':    vwl[valid],
        'vbsl':   vbsl[valid],
        'pw':     pw[valid],
        's':      gi[valid],
        's_next': gf[valid],
        'delta':  gf[valid] - gi[valid],
    })


# ── analysis helpers ──────────────────────────────────────────────────────────

def spread_by_param(df, param, group_cols, min_count=10):
    """
    For each combination in group_cols, compute mean(delta) for each unique
    value of `param`.  Return a DataFrame with column 'spread' = max-min of
    those means, and 'n_values' = how many param values were seen.
    """
    rows = []
    for key, grp in df.groupby(group_cols):
        by_param = (
            grp.groupby(param)['delta']
               .agg(['mean', 'count'])
               .query('count >= @min_count')
        )
        if len(by_param) < 2:
            continue
        spread = by_param['mean'].max() - by_param['mean'].min()
        rows.append({**dict(zip(group_cols, key if isinstance(key, tuple) else (key,))),
                     'spread': spread, 'n_values': len(by_param)})
    return pd.DataFrame(rows)


def representative_vwl_values(df, n=5):
    """Pick n vwl values spread across the distribution (by count)."""
    counts = df.groupby('vwl')['delta'].count()
    # pick from low / mid / high range
    vwl_vals = sorted(counts.index)
    idxs = np.linspace(0, len(vwl_vals) - 1, n, dtype=int)
    return [vwl_vals[i] for i in idxs]


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_param_effect(df, pulse_type, out_path):
    """
    For 5 representative vwl values: show how mean(Δs) varies with vbsl and pw
    (lines per vbsl/pw value, x-axis = initial state s).
    """
    vwl_vals = representative_vwl_values(df, n=5)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey='row')
    fig.suptitle(f'{pulse_type} pulses: effect of vbsl (top) and pw (bottom) on mean Δs\n'
                 f'(each line = one parameter value; fixed vwl per column)',
                 fontsize=13)

    for col, vwl in enumerate(vwl_vals):
        sub = df[df['vwl'] == vwl]

        for row, (param, cmap_name) in enumerate([('vbsl', 'Blues'), ('pw', 'Oranges')]):
            ax = axes[row, col]
            param_vals = sorted(sub[param].unique())
            cmap = plt.get_cmap(cmap_name)
            colors = cmap(np.linspace(0.35, 0.9, len(param_vals)))

            for pval, color in zip(param_vals, colors):
                grp = sub[sub[param] == pval]
                agg = (grp.groupby('s')['delta']
                          .agg(['mean', 'count'])
                          .query('count >= 5'))
                if len(agg) < 3:
                    continue
                ax.plot(agg.index, agg['mean'], color=color, linewidth=1.2,
                        label=f'{param}={pval}', alpha=0.85)

            ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
            ax.set_title(f'vwl={vwl}', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'mean Δs  ({param} varies)', fontsize=9)
            if row == 1:
                ax.set_xlabel('initial state s', fontsize=9)
            ax.legend(fontsize=6, loc='upper right', ncol=2)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"Saved {out_path.name}")


def plot_spread_summary(df_set, df_rst):
    """
    Boxplots: distribution of spread values (max_mean - min_mean of Δs) when
    varying vbsl, pw, or vwl — for both pulse types.
    Shows at a glance which parameter drives the most variance in outcome.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, df, title in [(axes[0], df_set, 'SET'), (axes[1], df_rst, 'RESET')]:
        # Spread when vbsl varies (fixed vwl, s)
        sp_vbsl = spread_by_param(df, 'vbsl', ['vwl', 's'])['spread']
        # Spread when pw varies (fixed vwl, s)
        sp_pw   = spread_by_param(df, 'pw',   ['vwl', 's'])['spread']
        # Spread when vwl varies (fixed vbsl, pw, s) — upper reference
        sp_vwl  = spread_by_param(df, 'vwl',  ['vbsl', 'pw', 's'])['spread']

        data   = [sp_vbsl.values, sp_pw.values, sp_vwl.values]
        labels = ['vbsl\n(fixed vwl, s)', 'pw\n(fixed vwl, s)', 'vwl\n(fixed vbsl,pw,s)']
        colors = ['steelblue', 'darkorange', 'forestgreen']

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Spread of mean Δs  (max − min across param values)', fontsize=10)
        ax.set_title(f'{title} pulses: how much does each parameter matter?', fontsize=11)
        ax.grid(True, alpha=0.25, axis='y')

        # Annotate medians
        for i, d in enumerate(data, 1):
            med = np.median(d)
            ax.text(i, med + 0.3, f'{med:.1f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'param_spread_summary.png', dpi=150, bbox_inches='tight')
    print("Saved param_spread_summary.png")


def print_stats(df_set, df_rst):
    print("\n=== Spread of mean Δs (max-min across parameter values) ===")
    for label, df in [('SET', df_set), ('RESET', df_rst)]:
        sp_vbsl = spread_by_param(df, 'vbsl', ['vwl', 's'])['spread']
        sp_pw   = spread_by_param(df, 'pw',   ['vwl', 's'])['spread']
        sp_vwl  = spread_by_param(df, 'vwl',  ['vbsl', 'pw', 's'])['spread']
        print(f"\n{label}:")
        print(f"  vbsl spread | median={sp_vbsl.median():.2f}  mean={sp_vbsl.mean():.2f}  "
              f"p90={sp_vbsl.quantile(.9):.2f}")
        print(f"  pw   spread | median={sp_pw.median():.2f}  mean={sp_pw.mean():.2f}  "
              f"p90={sp_pw.quantile(.9):.2f}")
        print(f"  vwl  spread | median={sp_vwl.median():.2f}  mean={sp_vwl.mean():.2f}  "
              f"p90={sp_vwl.quantile(.9):.2f}")
        ratio_vbsl = sp_vbsl.median() / sp_vwl.median()
        ratio_pw   = sp_pw.median()   / sp_vwl.median()
        print(f"  → vbsl effect is {ratio_vbsl:.0%} of vwl effect (by median spread)")
        print(f"  → pw   effect is {ratio_pw:.0%} of vwl effect (by median spread)")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading SET sweeps...")
    df_set = pd.concat([
        load_sweep(DATA_DIR / 'setsweep1.csv'),
        load_sweep(DATA_DIR / 'setsweep2.csv'),
    ], ignore_index=True)
    print(f"  {len(df_set):,} transitions")

    print("Loading RESET sweeps...")
    df_rst = pd.concat([
        load_sweep(DATA_DIR / 'resetsweep1.csv'),
        load_sweep(DATA_DIR / 'resetsweep2.csv'),
    ], ignore_index=True)
    print(f"  {len(df_rst):,} transitions")

    print("\nGenerating per-parameter effect plots...")
    plot_param_effect(df_set, 'SET',   OUT_DIR / 'param_sensitivity_set.png')
    plot_param_effect(df_rst, 'RESET', OUT_DIR / 'param_sensitivity_reset.png')

    print("\nGenerating spread summary...")
    plot_spread_summary(df_set, df_rst)

    print_stats(df_set, df_rst)
