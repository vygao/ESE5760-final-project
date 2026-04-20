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

### Step 7: Pulse-Width-Binned Action Space (`markov/build_chain_pw.py`)

**Goal**: replace pw pooling with explicit pw bins, giving the policy finer control over pulse strength and a more realistic action space.

The current model pools over all pulse widths, collapsing them into a single marginal distribution per (vwl, vbsl, type). Parameter sensitivity (Step 3) showed pw has ~40% the effect of vwl for SET and ~154% for RESET — especially for RESET, this pooling is significantly lossy.

**pw values in the data** (powers of 2): chip1 uses 1–128 ns, chip2 uses 1–16 ns. A natural log-scale binning:

| Bin | pw range (ns) | Rationale |
|-----|--------------|-----------|
| 0   | 1–2          | shortest pulses — fine trimming |
| 1   | 4–8          | medium-short |
| 2   | 16–32        | medium-long |
| 3   | 64–128       | longest — coarse jumps |

This expands the action space from 1024 to **4096** (2 types × 128 vwl × 4 vbsl bins × 4 pw bins). Data density will be ~4x thinner per action — `MIN_COUNT` threshold becomes more important. Pairs with insufficient data in a given pw bin fall back to the pooled estimate or are masked out.

Procedure: duplicate `build_chain.py`, add `pw_to_bin()` mapping, include pw bin in the action index formula, re-run value iteration and Monte Carlo on the new transition matrix.

Expected outcome: lower mean expected pulses and tighter pulse-count distributions, particularly for RESET-dominated programming sequences.

---

### Step 8: Multi-Bit Cell Evaluation and Comparison to RADAR/PBA

**Goal**: translate the pulse-level MDP results into the metric that matters for real memory systems — bit error rate (BER) when programming a 2-bit (4-level) or 3-bit (8-level) MLC cell — and compare directly against RADAR and PBA.

#### Bin definitions

Partition the 65 ADC states into equal-width bins (or PBA-style percentile bins):

| Configuration | Bins | States per bin |
|--------------|------|----------------|
| 2-bit MLC    | 4    | ~16 states     |
| 3-bit MLC    | 8    | ~8 states      |

Each bin represents one resistance level; a cell is correctly programmed if it lands anywhere within the target bin.

#### MDP reformulation (bin-level targets)

Re-solve value iteration with bin-level absorbing states: any state within the target bin counts as done. This yields `V_bin[s, b]` — the expected pulses to reach bin `b` from state `s` — and a corresponding optimal bin-level policy. This is the correct formulation for comparing against RADAR (which also targets a resistance window, not a single state).

The per-state policy from Step 5 is a suboptimal approximation for this task: it over-commits to a single centroid state and may waste pulses trying to hit an exact level when the cell is already inside the target window.

#### Monte Carlo BER measurement

Simulate the bin-level policy and record:
- **Success rate**: fraction of rollouts that land in the correct bin
- **Expected pulses**: total pulses per successful program operation
- **BER**: per-bit error rate under gray coding, as a function of write budget (pulses)

#### Comparison baseline

RADAR is an iterative write-verify scheme: apply a pulse, read back, repeat until within tolerance. PBA/SBA (from this repo) optimize level allocation but use a fixed write algorithm. The MDP policy is the **theoretical optimum** under the learned transition model — it provides a lower bound on expected pulses for any algorithm operating on the same hardware.

Report the gap: how many more pulses does RADAR use vs. the MDP optimum for the same target BER?

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
