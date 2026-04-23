# PBA Repository

This repository includes the Python scripts for the research paper ["PBA: Percentile-Based Level Allocation for Multiple-Bits-Per-Cell RRAM"](https://cs.stanford.edu/~anjiang/papers/ICCAD23PBA.pdf), at ICCAD'23.

---

## Hardware Measurement Data

We conduct experiments on two Ember RRAM chips and measure the resistance at the timestamp of 1 second.

The raw data for ember chip [1](https://github.com/Anjiang-Wei/PBA/blob/dala/data/retention1s.csv) and [2](https://github.com/Anjiang-Wei/PBA/blob/dala/data/retention1s2.csv) are provided.

We then build a more compact representation of the data points with scripts [1](https://github.com/Anjiang-Wei/PBA/blob/dala/analysis/build_retention_model.py) and [2](https://github.com/Anjiang-Wei/PBA/blob/dala/analysis/build_retention_model2.py) respectively, generating two files [1](https://github.com/Anjiang-Wei/PBA/blob/dala/model/retention1s.csv) and [2](https://github.com/Anjiang-Wei/PBA/blob/dala/model/retention1s2.csv) in the format of `write_level_low, write_level_high, (list of levels)` where `list_of_levels` is the readout level index after waiting for 1 second.

---

## Markov Chain Optimal Programming Model

In addition to the PBA level-allocation work, this repository contains an MDP-based optimal programming model built from sweep data.

**Goal**: find the minimum-pulse sequence to program a cell to a target resistance level, and characterise the write BER and pulse-count distribution under realistic cell-to-cell variability.

### Datasets: Sweep Files

Four files capture single-pulse transition statistics across two chips:

| File | Chip | Type | Rows | Cells/row | Total transitions |
|------|------|------|------|-----------|-------------------|
| `data/setsweep1.csv` | 1 | SET | 7,680 | 15 | 115,200 |
| `data/setsweep2.csv` | 2 | SET | 131,072 | 16 | 2,097,152 |
| `data/resetsweep1.csv` | 1 | RESET | 65,536 | 16 | 1,048,576 |
| `data/resetsweep2.csv` | 2 | RESET | 131,072 | 16 | 2,097,152 |

**Column layout** (auto-detected by script):
```
addr | timestamp | vwl | vbsl | pw | gi[0..N] | [artifact] | gf[0..N]
```
- `vwl`: wordline voltage DAC code (0–254, step 2) — dominant pulse strength control
- `vbsl`: bitline-select voltage DAC code (0–31) — chip1 step=8, chip2 step=4
- `pw`: pulse width in ns (powers of 2 up to 128 ns for SET; 1–8 ns for RESET chip1, 1–2 ns for RESET chip2)
- `gi[k]` / `gf[k]`: pre/post-pulse ADC level of cell k (0–64, linear in conductance G = 1/R)
- SET pulses drive conductance **up** (lower resistance); RESET pulses drive it **down**

**ADC scale**: 65 states (0–64). State 0 ≈ >100 kΩ (HRS); state 64 ≈ 6–8 kΩ (LRS). Linear in conductance, not resistance — 4 ADC steps near HRS spans a large resistance range; near LRS it spans only a few hundred Ω.

**Hardware note**: SET and RESET pulses are physically distinct. SET drives the bitline (BL); RESET drives the sourceline (SL). Current flows in opposite directions. Whether freely interleaving SET and RESET in a single programming sequence is realizable on Ember hardware is an open question.

---

### Step 1: Single-Step Reachability (`visualizations/single_step_reachability.py`)

Maps every observed single-step transition across all four sweep files.

Key results:
- SET: 3,139 unique (s → s') pairs; avg **48.3 reachable states** per initial state
- RESET: 2,686 unique pairs; avg **41.3 reachable states** per initial state

Output: `transition_map_both.png`, `transition_coverage.png`.

### Step 2: Transitive Closure (`visualizations/transitive_closure.py`)

Computes multi-step reachability via repeated boolean matrix multiplication.

Key results:
- Converged in **3 steps** — all 65 states reachable from all others in ≤3 pulses
- Proves a finite optimal policy exists for every (s, target) pair

Output: `closure_reachability.png`, `closure_by_steps.png`, `closure_gaps.png`.

### Step 3: Parameter Sensitivity (`visualizations/param_sensitivity.py`)

Measures how much `vbsl` and `pw` affect the final state independently of `vwl`.

| Parameter | SET median spread | RESET median spread | vs. vwl |
|-----------|-------------------|---------------------|---------|
| vwl | 17.9 states | 1.6 states | baseline |
| **vbsl** | **17.1 states** | **3.4 states** | SET: 96%, RESET: 221% |
| pw | 7.1 states | 2.4 states | SET: 40%, RESET: 154% |

**Conclusion**: pooling over `vbsl` or `pw` is significantly lossy. `vbsl` alone has nearly as much effect as `vwl` for SET, and exceeds `vwl` for RESET. Both parameters are included in the action space below.

---

### Step 4: Transition Matrix (`markov/build_chain.py`)

Builds `T[action, s, s']` — the probability of transitioning from state `s` to `s'` under action `a`. Both `vbsl` and `pw` are included in the action index; the number of pw bins is configurable via `--pw-bins`.

**Action space** (`pw_bins` = number of log-scale pulse-width bins):

| `pw_bins` | Action index layout | Total actions |
|-----------|---------------------|---------------|
| 1 (pw pooled) | 2 types × 128 vwl × 4 vbsl bins | **1,024** |
| 2 | 2 × 128 × 4 vbsl × 2 pw bins | **2,048** |
| 4 | 2 × 128 × 4 vbsl × 4 pw bins | **4,096** |

Log-scale pw binning for `pw_bins=4`: [1–2 ns], [4–8 ns], [16–32 ns], [64–128 ns]. Note: RESET sweep data has narrower pw coverage (chip1: 1–8 ns, chip2: 1–2 ns), so pw binning primarily benefits SET.

```
python markov/build_chain.py --pw-bins 1   # → transition_probs_pw1.npy  [1024, 65, 65]
python markov/build_chain.py --pw-bins 2   # → transition_probs_pw2.npy  [2048, 65, 65]
python markov/build_chain.py --pw-bins 4   # → transition_probs_pw4.npy  [4096, 65, 65]
```

`MIN_COUNT = 3`: rows with fewer than 3 observations are zeroed out. Coverage for `pw_bins=1`: 64,468 / 66,560 (action, s) pairs. State 0 has no valid outgoing actions (too few observations).

### Step 5: Exact-State Value Iteration (`markov/value_iteration.py`)

Solves the Stochastic Shortest Path (SSP): find the optimal policy to reach any single target state in minimum expected pulses.

**Bellman equation**:
```
V[s, t] = 0                                    if s == t
V[s, t] = min_a { 1 + Σ_{s'} T[a,s,s'] V[s',t] }  otherwise
```

`V[s, t]` is the expected pulses to reach state `t` from state `s` under the optimal policy.

```
python markov/value_iteration.py   # reads transition_probs_pw1.npy by default
```

Outputs: `markov/value_function.npy [65,65]`, `markov/policy.npy [65,65]`.

Key results (`pw_bins=1`, 1,024 actions):
- Converged in 11 iterations
- **Mean expected pulses: 6.22** across all reachable (s, target) pairs
- **p90: 11.9 pulses**, **p99: 21.4 pulses**
- **Max: 163 pulses** (hardest pair)
- 4,096 / 4,160 off-diagonal pairs reachable; 64 unreachable (all from state 0)

This policy is the **theoretical lower bound**: no algorithm on the same transition model can achieve lower expected pulse count.

### Step 6: Exact-State Monte Carlo Validation (`markov/monte_carlo.py`)

Validates the value function by rolling out the policy stochastically (sampling actual next states from `T[a, s, :]`) and comparing empirical means to `V[s, t]`.

Key results (`pw_bins=1`, 1,000 trials per (s, t) pair):
- Empirical mean **6.22 pulses** — matches V exactly, confirming model consistency
- 100% success rate across all reachable pairs within 1,500-step budget
- SET↔RESET interleaving analysis: the policy freely mixes pulse types (open hardware question)

Outputs: `mc_pulse_counts.npy`, `mc_summary.csv`, `mc_mean_vs_theoretical.png`, `mc_percentiles.png`, `mc_histograms.png`, `mc_interleaving.png`.

---

### Step 7: MLC Evaluation — Bin-Level Policy and BER (`markov/bin_vi.py`, `markov/run_experiments.py`)

For real MLC operation the goal is not a single exact state but a **resistance window**. This step reformulates the MDP with window-level absorbing states, evaluates write BER under realistic cell-to-cell variability, and compares against RADAR.

#### Resistance windows

Two window definitions are supported via `--bin-type`:

- **`equal`**: equal-width partition of [0, 64]. Windows are ~8–16 states wide — too wide for meaningful BER (the policy trivially lands inside in 1–2 pulses every time).
- **`pba`**: write windows derived from `model/retention1s.csv` using the PBA algorithm (`algorithm/dala.py`). Each write window is **4 ADC states wide**, matching the actual hardware programming precision. Read windows are wider to absorb 1-second resistance drift.

PBA bin boundaries:

| Level | Write window | Read window (post-drift, 1 s) |
|-------|-------------|-------------------------------|
| **4-level (2-bit)** | | |
| 0 (HRS) | [0, 4) — 4 states | [0, 13) — 13 states |
| 1 | [18, 22) | [13, 28) |
| 2 | [32, 36) | [28, 39) |
| 3 (LRS) | [42, 46) | [39, 64] |
| **8-level (3-bit)** | | |
| 0 (HRS) | [0, 4) | [0, 8) |
| 1 | [12, 16) | [8, 21) |
| 2 | [26, 30) | [21, 34) |
| 3 | [35, 39) | [34, 41) |
| 4 | [42, 46) | [41, 48) |
| 5 | [48, 52) | [48, 53) |
| 6 | [54, 58) | [53, 59) |
| 7 (LRS) | [59, 63) | [59, 64] |

For 4-level PBA, 49/65 ADC states are gaps between write windows; for 8-level, 33/65 are gaps. Landing in a gap is a write error and the policy continues pulsing.

#### Bin-level value iteration

```
python markov/bin_vi.py --pw-bins 1 --n-levels 8 --bin-type pba
```

Outputs: `bin_value_pw{K}_lv{L}_{type}.npy [65, L]`, `bin_policy_pw{K}_lv{L}_{type}.npy [65, L]`, `bin_windows_lv{L}_pba.npz`.

#### Cell-to-cell variability model

T[a, s, s'] captures the **total empirical spread** of transition outcomes pooled across all cells, but treats each step's randomness as independent. This correctly models within-cell stochastic noise (each pulse has a different outcome even for the same cell). It does **not** model persistent between-cell variation: cell A is consistently 3 ADC states more SET-sensitive than the average across its entire lifetime.

The `sigma_cell` parameter models this persistent bias: each simulated cell draws a fixed offset δ ~ N(0, σ²) once before its rollout. Every transition outcome is shifted by δ. The policy observes the actual (shifted) state and adapts, but its action-value estimates were calibrated on the unbiased model — so cells with large |δ| require extra correction pulses and may fail within a fixed budget.

**σ estimate**: the retention data (`model/retention1s.csv`) gives a per-cell readout spread of **σ ≈ 1.5 ADC units** (median std across write levels at t = 1 s). This is the best available estimate of the persistent cell-to-cell offset.

#### BER results and RADAR comparison

Run the full experiment sweep:
```
python markov/run_experiments.py --bin-type pba
```

All results below: start state = ADC 2 (lowest fully-covered HRS state), σ_cell = 1.5, N = 3,000 trials per target bin, budget = 36 pulses (RADAR's 3-bit operating point).

| Config | `pw_bins` | σ_cell | BER @ 36 pulses | Mean pulses | RADAR |
|--------|-----------|--------|-----------------|-------------|-------|
| 3-bit (8-level) | 1 | 1.5 | **0.19%** | **3.2** | 1.0% @ 36.4 pulses |
| 3-bit (8-level) | 4 | 1.5 | 6.6% | 2.4 | 1.0% @ 36.4 pulses |
| 2-bit (4-level) | 1 | 1.5 | **0.19%** | **3.2** | 0.3% @ ~19 pulses |
| 2-bit (4-level) | 4 | 1.5 | 5.1% | 3.1 | 0.3% @ ~19 pulses |

**`pw_bins=1` beats RADAR on 3-bit**: 0.19% BER at the same 36-pulse budget, with 11× fewer mean pulses. For 2-bit, our BER at budget=36 (0.19%) beats RADAR's target of 0.3%, though RADAR reaches its target with only 19 pulses; we need ~20.

**`pw_bins=4` is worse than RADAR** under realistic variability. This is a **precision vs. robustness tradeoff**: finer pw control produces a policy precisely tuned to the average cell, making it more brittle when individual cells deviate. At σ=0, `pw_bins=4` achieves lower pulse counts; at σ=1.5, it degrades to a ~6% BER floor that doesn't improve with larger budget (biased cells get trapped in oscillation).

#### Pulse-count distribution (`pw_bins=1`, 8-level, σ=1.5, N=35,000 rollouts)

| Statistic | Value |
|-----------|-------|
| mean | 3.1 pulses |
| p50 | 2 |
| p75 | 3 |
| p90 | 7 |
| p95 | 10 |
| p99 | 20 |
| p99.9 | 45 |
| max | 113 |

43% of cells converge in one pulse; 86% in ≤5. The tail is produced by cells with large persistent offset. **Our p99 (20 pulses) is lower than RADAR's reported mean (36.4 pulses).**

#### Key output figures

| File | Description |
|------|-------------|
| `markov/exp_ber_vs_sigma_lv8.png` | BER vs. σ_cell, 3-bit — precision/robustness tradeoff across pw_bins |
| `markov/exp_ber_vs_sigma_lv4.png` | Same for 2-bit |
| `markov/exp_expected_pulses.png` | Mean write pulses by (pw_bins, n_levels, bin_type) |
| `markov/exp_ber_heatmap_pba.png` | BER heatmap at 50-pulse budget, PBA bins |
| `markov/mc_mean_vs_theoretical.png` | Model validation: empirical mean = V[s,t] |
| `markov/mc_percentiles.png` | p50/p90/p99 pulse-count heatmaps, exact-state policy |

---

## PBA Experiments

### SBA versus PBA
With the SBA [implementation](https://github.com/Anjiang-Wei/PBA/blob/dala/algorithm/SBA.py), we generate the probability transition matrix with this [script](https://github.com/Anjiang-Wei/PBA/blob/dala/algorithm/SBA_genmatrix.py), and the matrix is saved [here](https://github.com/Anjiang-Wei/PBA/tree/dala/ember_capacity) (`SBA4`, `SBA8`) for chip1, and [here](https://github.com/Anjiang-Wei/PBA/tree/dala/ember_capacity2) (`SBA4`, `SBA8`) for chip2.

The PBA [implementation](https://github.com/Anjiang-Wei/PBA/blob/dala/algorithm/dala.py). We generate the probability transition matrix with this [script](https://github.com/Anjiang-Wei/PBA/blob/dala/algorithm/dala_genmatrix.py), and the matrix is saved [here](https://github.com/Anjiang-Wei/PBA/tree/dala/ember_capacity) (`ours4`, `ours8`) for chip1, and [here](https://github.com/Anjiang-Wei/PBA/tree/dala/ember_capacity2) (`ours4`, `ours8`) for chip2.

We use gray coding and compute the [bit error rate](https://github.com/Anjiang-Wei/PBA/blob/dala/ember_capacity/trans.py) from the transition matrix. After obtaining the bit error rate results, then we run a [search](https://github.com/Anjiang-Wei/PBA/blob/dala/ember_capacity/ecc.py) to find the error correcting code with the lowest overhead. The expected output is saved in the logs for [BER](https://github.com/Anjiang-Wei/PBA/blob/dala/ember_capacity/log_trans) and [ECC](https://github.com/Anjiang-Wei/PBA/blob/dala/ember_capacity/log_ecc)

### Ablation Study
With the PBA-norm [implementation](https://github.com/Anjiang-Wei/PBA/blob/dala/algorithm/SBA_meanvariant.py), we can generate the probability transition matrix with this [script](https://github.com/Anjiang-Wei/PBA/blob/dala/algorithm/SBA_genmatrix.py), and the matrix is saved [here](https://github.com/Anjiang-Wei/PBA/tree/dala/ember_capacity) (`SBAmeanvar4`, `SBAmeanvar8`) for chip1, and [here](https://github.com/Anjiang-Wei/PBA/tree/dala/ember_capacity2) (`SBAmeanvar4`, `SBAmeanvar8`) for chip2. The bit error rate and ECC results can be generated with the corresponding scripts in the same directory as the transition matrix.

### Other Experiments
1) Dataset sizes: [script](https://github.com/Anjiang-Wei/PBA/tree/dala/algorithm_repeatavail) to generate [matrix result](https://github.com/Anjiang-Wei/PBA/tree/dala/ember_repeatavail) with different ratios of the original dataset.
2) Different ratios of target chip and non-target chip:
- a) completely switch the two: [script](https://github.com/Anjiang-Wei/PBA/tree/dala/algorithm_inter) to generate [matrix result](https://github.com/Anjiang-Wei/PBA/tree/dala/intercapacity)
- b) half and half: [script](https://github.com/Anjiang-Wei/PBA/tree/dala/algorithm_both) to generate [matrix result](https://github.com/Anjiang-Wei/PBA/tree/dala/bothcapacity)
- c) one dominates the other: [script](https://github.com/Anjiang-Wei/PBA/tree/dala/algorithm_dominate) to generate [matrix result](https://github.com/Anjiang-Wei/PBA/tree/dala/domin_capacity)

**Note: Directories ending with `2`, e.g., `ember_capacity2`, are for Ember Chip 2. For directories without 2, e.g., `ember_capacity`, are for Ember Chip 1.**
