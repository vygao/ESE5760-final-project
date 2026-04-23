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

**Goal**: given a cell's current resistance state, find the minimum-pulse sequence to reach any target state.

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
- `pw`: pulse width in ns (powers of 2, up to 2048; chip2 adds pw=3968)
- `gi[k]`: pre-pulse resistance state of cell k (ADC level 0–64, linear in conductance)
- `gf[k]`: post-pulse resistance state of cell k
- SET pulses drive resistance **down** (higher conductance); RESET pulses drive it **up**

**Note**: ADC levels are linear in conductance (G = 1/R), not resistance. ADC≈64 corresponds to ~6–8 kΩ (LRS); ADC≈2 corresponds to >100 kΩ (HRS).

### Step 1: Single-Step Reachability (`visualizations/single_step_reachability.py`)

Maps every observed single-step transition across all four files. Output: `transition_map_both.png`, `transition_coverage.png`.

Key results:
- SET: 3,139 unique (s → s') pairs; avg 48.3 reachable states per initial state
- RESET: 2,686 unique pairs; avg 41.3 reachable states per initial state

### Step 2: Transitive Closure (`visualizations/transitive_closure.py`)

Computes multi-step reachability via repeated boolean matrix multiplication. Output: `closure_reachability.png`, `closure_by_steps.png`, `closure_gaps.png`.

Key results:
- Converged in **3 steps** — all 65 states are reachable from all others in ≤3 pulses
- This proves a finite optimal policy exists for every (s, target) pair

### Step 3: Parameter Sensitivity (`visualizations/param_sensitivity.py`)

Measures how much `vbsl` and `pw` affect final state independently of `vwl`. Computes the spread of mean Δs across parameter values for each (vwl, s) combination.

Key results:

| Parameter | SET median spread | RESET median spread | vs. vwl effect |
|-----------|-------------------|---------------------|----------------|
| vwl       | 17.9 states       | 1.6 states          | baseline       |
| **vbsl**  | **17.1 states**   | **3.4 states**      | SET: 96%, RESET: 221% |
| pw        | 7.1 states        | 2.4 states          | SET: 40%, RESET: 154% |

**Conclusion**: pooling over `vbsl` and `pw` is significantly lossy — `vbsl` alone has nearly as much effect as `vwl` for SET pulses, and exceeds `vwl`'s effect for RESET pulses. Expanding the action space beyond `(type, vwl)` is recommended.

### Step 4: Transition Matrix (`markov/build_chain.py`)

Builds `T[action, s, s']` — the probability of transitioning from state `s` to `s'` under action `a`.

Current action space (baseline, vbsl/pw pooled):
- Actions 0–127: SET pulses, indexed by `vwl // 2`
- Actions 128–255: RESET pulses, indexed by `128 + vwl // 2`

Outputs: `markov/transition_counts.npy` [256, 65, 65], `markov/transition_probs.npy` [256, 65, 65], `markov/action_info.csv`.

Parameters: `MIN_COUNT = 3` (rows with fewer observations are zeroed out).

Coverage: 16,231 / 16,640 (action, s) pairs have sufficient observations. Only state 0 (minimum conductance) has no valid outgoing actions.

### Step 5: Value Iteration (`markov/value_iteration.py`)

Solves the Stochastic Shortest Path (SSP) problem via value iteration to compute the optimal policy.

**Bellman equation**:
```
V[s, t] = 0                               if s == t (already at target)
V[s, t] = min_a { 1 + P[a,s,:] · V[:,t] }  otherwise
```

`V[s, t]` = expected number of pulses to reach target `t` from state `s` using the optimal policy. Converges from V=0 (optimistic initialization) upward.

Outputs: `markov/value_function.npy`, `markov/policy.npy`, `markov/policy_vwl.npy`, `markov/policy_type.npy`.

Key results (baseline, vbsl/pw pooled):
- Converged in 328 iterations (policy stability criterion)
- **Mean expected pulses: 24.8** across all reachable (s, target) pairs
- **Max expected pulses: 303** (s=32 → t=0, cross-range against gradient)
- 4,096 / 4,160 off-diagonal pairs reachable; 64 unreachable (all from state 0)

This policy is the **optimal baseline**: no algorithm operating on the same transition model can do better in expectation.

### Step 6: Monte Carlo Simulation (`markov/monte_carlo.py`)

**Goal**: validate the value function empirically and characterize the full pulse-count distribution (not just the mean).

Value iteration produces *expected* pulse counts under the model. Monte Carlo simulation rolls out the policy stochastically — sampling actual next states from `T[a, s, :]` at each step — and records what actually happens over many trials.

**Planned outputs**: `markov/mc_pulse_counts.npy` [65, 65, N_trials], summary stats CSV.

Procedure:
1. Load `transition_probs.npy` and `policy.npy`
2. For each (s, t) pair, run N rollouts:
   - Apply `policy[current, t]`, sample next state from `T[action, current, :]`
   - Repeat until `current == t` or a max-pulse timeout is hit
   - Record total pulse count and whether the rollout succeeded
3. Compute per-(s,t): empirical mean, standard deviation, 90th/99th percentile, success rate

**Validation**: plot empirical mean vs `V[s, t]` — should match closely if the transition model is consistent. Large deviations flag model/data issues.

**Distribution analysis**: the mean alone hides heavy tails. Plot pulse-count histograms for representative (s, t) pairs (e.g. easy same-direction, hard cross-range). Percentiles matter for real system design — a 99th-percentile worst case of 500 pulses is very different from a mean of 25.

**SET/RESET interleaving analysis**: track how often the optimal policy switches between SET and RESET mid-sequence. This is an open hardware question — SET and RESET are physically distinct at the circuit level (different voltage lines driven), so freely interleaved sequences may not be realizable on Ember without further validation.

Key results (1024-action model, vbsl-expanded):
- 100% success rate across all reachable (s, target) pairs within 1500 steps
- **Mean pulses (empirical): 6.22** — matches theoretical value function exactly, validating the model
- **p90: 11.9 pulses**, **p99: 21.4 pulses** — tails are moderate
- **Max p99: 163 pulses** (worst-case pair)

Outputs: `markov/mc_pulse_counts.npy`, `markov/mc_timed_out.npy`, `markov/mc_summary.csv`, `markov/mc_mean_switches.npy`, plus figures `mc_mean_vs_theoretical.png`, `mc_percentiles.png`, `mc_histograms.png`, `mc_interleaving.png`.

---

### Step 7: Pulse-Width-Binned Action Space and Bin-Level MDP (`markov/build_chain.py`, `markov/bin_vi.py`)

#### Action space

The action space is parameterised by `pw_bins` — the number of log-scale pulse-width bins. `pw_bins=1` pools all pulse widths (baseline); higher values separate short from long pulses:

| `pw_bins` | Composition | Total actions |
|-----------|-------------|---------------|
| 1 (pooled) | 2 types × 128 vwl × 4 vbsl bins × 1 pw bin | **1,024** |
| 2 | 2 × 128 × 4 × 2 pw bins | **2,048** |
| 4 | 2 × 128 × 4 × 4 pw bins | **4,096** |

pw values in the data are powers of 2: chip1 uses 1–128 ns, chip2 uses 1–16 ns (RESET sweeps even narrower: chip1 1–8 ns, chip2 1–2 ns). The log-scale binning for `pw_bins=4` is: [1–2 ns], [4–8 ns], [16–32 ns], [64–128 ns].

Build transition matrices for each `pw_bins` value:
```
python markov/build_chain.py --pw-bins 1   # → transition_probs_pw1.npy [1024, 65, 65]
python markov/build_chain.py --pw-bins 2   # → transition_probs_pw2.npy [2048, 65, 65]
python markov/build_chain.py --pw-bins 4   # → transition_probs_pw4.npy [4096, 65, 65]
```

#### Bin-level value iteration (`markov/bin_vi.py`)

For MLC evaluation, the target is a resistance *window*, not a single state. Two bin-definition modes:

- **`equal`**: equal-width partition of [0, 64] into N levels (~8–16 states per bin). Too wide to produce meaningful BER — any single SET pulse from HRS lands inside the window.
- **`pba`**: PBA-derived write windows computed from `model/retention1s.csv`. Each write window is **4 ADC states wide**, matching real hardware programming precision. 49/65 states are gaps between windows (for 4-level); landing in a gap is an error.

PBA bin boundaries (derived from characterisation data, `dala.py` algorithm):

| Level | Write window | Read window (post-drift, 1 s) |
|-------|-------------|-------------------------------|
| **4-level (2-bit)** | | |
| 0 (HRS) | [0, 4) | [0, 13) |
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

Write windows are 4 ADC states wide; read windows are wider to absorb 1-second resistance drift.

Run bin-level value iteration for each configuration:
```
python markov/bin_vi.py --pw-bins 1 --n-levels 8 --bin-type pba
```
Outputs: `bin_value_pw{K}_lv{L}_{type}.npy [65, L]`, `bin_policy_pw{K}_lv{L}_{type}.npy [65, L]`

---

### Step 8: BER Evaluation and RADAR Comparison (`markov/run_experiments.py`)

#### What the Monte Carlo transition noise models — and what it doesn't

The transition matrix T[a, s, s'] captures the **total spread** of observed outcomes across all cells — for the same action from the same state, the next state is drawn from this distribution. Running the MC simulation with T already samples from the full population variance, but treats each step's noise as **independent**.

This correctly models *within-cell stochasticity* (per-pulse quantum noise). It does **not** model *between-cell variation*: the fact that cell A is persistently 3 ADC states more SET-sensitive than the average. That persistent bias is a fixed property of each physical cell — the same offset applies to every pulse in that cell's programming sequence — so the policy's corrections are systematically less effective.

The `sigma_cell` parameter adds this persistent-bias component: each simulated cell draws one fixed offset δ ~ N(0, σ²) before its rollout, and every transition in that rollout is shifted by δ. The policy observes the actual (shifted) state and reacts, but its action-value estimates were calibrated on the unbiased model, so it partially but not perfectly compensates.

**σ estimate from data**: the retention data (`model/retention1s.csv`) gives a readout spread of **σ ≈ 1.5 ADC units** (median std across write levels at t = 1 s). This is the best available estimate of the persistent cell-to-cell component because it is measured on many cells under the same write condition.

#### Results: BER vs. RADAR

All results below use:
- Start state: ADC state 2 (lowest fully-characterised HRS state; state 0 has no valid outgoing transitions)
- Bin type: PBA write windows (4 states wide)
- Write budget: 36 pulses (matching RADAR's reported operating point for 3-bit)
- σ_cell = 1.5 ADC units (matched to retention data spread)
- N = 3,000 trials per target bin, 7 non-trivial bins averaged for 8-level, 3 for 4-level

| Config | pw_bins | σ_cell | BER @ 36 pulses | Mean pulses (successes) | RADAR reference |
|--------|---------|--------|-----------------|-------------------------|-----------------|
| 3-bit (8-level) | 1 | 1.5 | **0.19%** | **3.2** | 1.0% @ 36.4 pulses |
| 3-bit (8-level) | 4 | 1.5 | 6.6% | 2.4 | 1.0% @ 36.4 pulses |
| 2-bit (4-level) | 1 | 1.5 | **0.19%** | **3.2** | 0.3% @ ~19 pulses |
| 2-bit (4-level) | 4 | 1.5 | 5.1% | 3.1 | 0.3% @ ~19 pulses |

**pw_bins=1 (pooled) beats RADAR on 3-bit**: 0.19% BER vs. 1.0% at the same 36-pulse budget, with 11× fewer mean pulses. The 2-bit comparison is tighter: at budget=36 our BER (0.19%) also beats RADAR's 0.3%, but RADAR reaches its 0.3% target at only 19 pulses; we would need ~budget=20 to achieve equivalent BER.

**pw_bins=4 is worse than RADAR** under realistic variability. Finer pulse-width control produces a more brittle policy: it picks precisely calibrated fine-grained actions that are optimal for the average cell but more sensitive to individual cell deviations. This is a **precision vs. robustness tradeoff** — higher granularity improves performance without variability (σ=0) but degrades faster as σ increases.

#### Pulse-count distribution (pw_bins=1, 8-level, σ=1.5)

The distribution is heavily right-skewed:

| Percentile | Pulse count |
|------------|-------------|
| p50 | 2 |
| p75 | 3 |
| p90 | 7 |
| p95 | 10 |
| p99 | 20 |
| p99.9 | 45 |
| max | 113 |
| mean | 3.1 |

43% of cells converge in a single pulse; 86% in ≤5 pulses. The long tail (p99 = 20 pulses) is produced by cells with large persistent offset (large |δ|) that require iterative correction. The tail behaviour is qualitatively consistent with RADAR's observation that "some cells take upward of 100 pulses" while the majority converge quickly. Notably, **our p99 (20 pulses) is lower than RADAR's reported mean (36.4 pulses)**.

#### Precision vs. robustness: BER vs. σ_cell

Figure `markov/exp_ber_vs_sigma_lv8.png` and `markov/exp_ber_vs_sigma_lv4.png` show BER at budget=50 pulses as a function of σ_cell for each pw_bins configuration:

- At σ=0 (ideal model, no persistent bias): BER=0 for all configurations
- At σ≥2: pw_bins=1 has substantially lower BER than pw_bins=4
- pw_bins=4 at σ=1.5 has a ~6% BER floor that does not decrease with larger budget — cells with large |δ| get trapped in correction cycles because the fine-grained policy over-commits to precise actions that the biased cell consistently misses

#### Key figures

| File | Description |
|------|-------------|
| `markov/exp_ber_vs_sigma_lv8.png` | BER vs. σ_cell, 3-bit, all pw_bins — shows precision/robustness tradeoff |
| `markov/exp_ber_vs_sigma_lv4.png` | Same for 2-bit |
| `markov/exp_expected_pulses.png` | Mean write pulses by (pw_bins, n_levels, bin_type) |
| `markov/exp_ber_heatmap_pba.png` | BER heatmap at budget=50, PBA bins |
| `markov/mc_mean_vs_theoretical.png` | Model validation: MC mean matches V[s,t] exactly |
| `markov/mc_percentiles.png` | p50/p90/p99 pulse-count heatmaps for exact-state policy |

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
