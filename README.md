# Study 1 — How large is the energy share of temporal state in brain-inspired SNNs?

**[English](README.md) | [中文](README.zh.md)**

> Preliminary Study 1 of the NSF proposal *Time as a First-Class Resource*
> Data collected: 2026-11-16
> Data basis: **all 2,264 samples of the SHD test split**

---

## 0. What this report answers

The goal of Study 1 in the white paper:

> "in delay-trained and multi-compartment models, **delay storage and neuron-state traffic, not spikes**, account for a substantial share of energy at matched accuracy. Spike-based estimates therefore misjudge these models."

Split into two falsifiable criteria:

| Criterion | Definition | White-paper threshold |
|---|---|---|
| **(a)** | Temporal state (state memory + delay storage + leakage) as a share of per-inference energy | ≥ 10% |
| **(b)** | Ranking by spike count or by SOP ≠ ranking by the full model | at least one comparison |

White-paper wording:

> "Temporal state is a non-trivial share of energy in **at least one** brain-inspired model, and the SOP-only ranking differs from the full-model ranking in **at least one comparison**."

---

## 1. The three models

| Model | Source | Params | Neurons | Epochs | Checkpoint |
|---|---|---|---|---|---|
| **delay 352** | DelRec (`alexmaxad/DelRec`), learned synaptic delays | 306,612 | 724 (incl. output LIF) | 7 | best validation |
| **plain LIF** | PMSN (`xychen-comp/...-PMSN`), $d_{state}=1$ | 187,988 | 704 | 5 | last epoch |
| **multi-compartment** | PMSN, $d_{state}=4$ | 200,660 | 704 | 3 | last epoch |

**Why these three**: they match the white paper's "three small SNNs on SHD" — a plain LIF baseline, a delay-trained network, and a multi-compartment network. PMSN with $d_{state}=1$ is a first-order IIR, i.e. equivalent to LIF, so it shares one codebase with the $d_{state}=4$ variant; the only architectural difference is the number of compartments.

**Why PMSN uses the last epoch rather than best**: MC peaks at epoch 1 and drops ~1 point by epoch 2. Taking best would raise it by 1.5–2 points and widen the gap to LIF to 3–4 points, breaking matched accuracy.

### Three time-step resolutions

Same 1.4 s span, different frame lengths:

| T | Frame | delay | LIF | MC | Spread |
|---|---|---|---|---|---|
| 50 | 28.0 ms | 86.56% | 88.96% | 91.21% | 4.65 pt |
| **100** | **14.0 ms** | **87.24%** | **86.40%** | **88.30%** | **1.90 pt** ✅ |
| **140** | **10.0 ms** | **87.99%** | **87.10%** | **88.96%** | **1.86 pt** ✅ |

**T=100 and T=140 are the main results** (spread <2 pt). T=50 is kept as cross-validation but the three models differ by 4.65 pt there.

---

## 2. What we changed and how we trained

### 2.0 Unifying the time axis

**All three models, all three resolutions, 9 training runs in total, use one fixed-duration binning:**

$$\text{duration} = \frac{1400\ \text{ms}}{T}\ \text{ms/frame}, \quad T \in \{50, 100, 140\}$$

**So frame $t$ always corresponds to the same physical interval $[t \cdot \text{duration},\ (t+1) \cdot \text{duration})$ across all three models.**

**This is the precondition for comparison**: the two upstream codebases bin differently by default (DelRec by frame count, PMSN by `linspace` equal division). Without unification, "the same T" would mean different frame lengths, and SOP, delay values, and state-access counts would all lose comparability. Concrete changes in §2.1 and §2.2.

### 2.1 DelRec: three patches

| # | Location | Change | What breaks without it |
|---|---|---|---|
| 1 | `configs/perf_SHD.py` | Re-derive all delay hyperparameters from `time_step` | Delay semantics drift when changing resolution |
| 2 | `src/delrec/datasets/loaders.py` | Replace `pad_sequence_collate` with `fixed_collate_T{T}`; bin with `duration=` | **Equal-division binning (`frames_number`) drops accuracy from 87% to 70.40%** |
| 3 | `src/delrec/utils.py` | Temporal pooling `max` → `mean` | Mismatch with the official checkpoint |

**Why binning must use fixed duration**: DelRec's delays are discrete frame counts shared across samples. With equal-division binning, one frame covers different physical durations for differently-lengthed samples, destroying the delay semantics. Verified by element-wise comparison: fixed-duration binning is identical to `integrate_events_by_fixed_duration_shd(duration=...)`.

### 2.1b Reworking the delay network's training pipeline (upstream → ours)

**Upstream behaviour:**

| Stage | Original DelRec |
|---|---|
| Binning | `config.time_step` is passed to `integrate_events_by_fixed_duration_shd(duration=...)`, but downstream `collate_fn=pad_sequence_collate` keeps each sample's actual frame count |
| Batch assembly | Variable-length padding, **samples within a batch have different frame counts** |
| Temporal pooling | `m, _ = torch.max(output, 0)` (max over time) |
| Delay hyperparameters | Hard-coded at `@10 ms`: `max_rec_delay=30`, `max_feedforward_delay=250//time_step`, `sigma_init=10.0`, `init_recdel_offset=10`, `delay_std_init=15` |

**What we changed:**

| Stage | After | Why |
|---|---|---|
| Binning | **Fixed `duration = 1400/T` ms**, crop over-long, zero-pad short | Preserve "one frame = fixed physical duration" |
| Batch assembly | **`fixed_collate_T{T}`: crop / zero-pad to a constant T frames** | Delays are in frames; if frame counts differ within a batch, the same delay value means different physical durations |
| Temporal pooling | **`m = torch.mean(output, 0)`** | Matches the official checkpoint's convention |
| Delay hyperparameters | **All re-derived from physical time** (divided by `time_step`), see the table below | Delay semantics stay at 300 ms / 250 ms / 100 ms / 150 ms across resolutions |

**Note that `max_feedforward_delay` was written as `250//time_step` — it already did the physical-time conversion, but the other four were hard-coded `@10 ms`. Changing `time_step` alone without re-deriving these five would make the effective delays more than double at T=50.**

**Hyperparameters derived from physical time** (divided by `time_step` to get frames):

| Parameter | Physical time | T=50 (28 ms) | T=100 (14 ms) | T=140 (10 ms) |
|---|---|---|---|---|
| `time_step` | — | 28 | 14 | 10 |
| `max_rec_delay` | 300 ms | 11 | 21 | 30 |
| `max_feedforward_delay` | 250 ms | 9 | 17 | 25 |
| `sigma_init` | 100 ms | 3.5 | 7.0 | 10.0 |
| `init_recdel_offset` | 100 ms | 4 | 7 | 10 |
| `delay_std_init` | 150 ms | 5 | 11 | 15 |

`patch_delrec.py --T {T}` derives this table automatically.

### 2.2 PMSN: two patches

| # | Change | What breaks without it |
|---|---|---|
| 1 | `np.linspace` equal-division binning → fixed `duration = 1400/T` ms | Its "one frame" would not match DelRec's, so the two families could not be compared at the same T |
| 2 | `bn=False` → `bn=True` | Checkpoints were trained with `bn=True`; a mismatch raises in `load_state_dict` |

**PMSN is a continuous-time-constant SSM and is insensitive to the binning scheme** (verified: accuracy unchanged after switching). The binning change exists purely to align the time axis with the delay network.

### 2.3 Training commands

See `patches/TRAINING.md`. Core commands:

```bash
# delay (must use --kernel v2, otherwise CuPy/triton shared-memory overflow)
python experiments/train.py --dataset shd --model SNN_recurrent_delays \
  --seeds 1111 --epochs 7 --kernel v2 --out /content/work/exp/delay_w352_T100_ts14_7ep

# PMSN
python SHD_PMSN.py --data_path /content/shd --T 100 --d_state 1 --epochs 5 \
  --ckpt_path /content/work/exp/pmsn_lif_T100_fill_5ep --device cuda:0
```

### 2.4 No dropout at inference

All three models used dropout during training (delay: feedforward 0.4 / recurrent 0.2; PMSN: 0.2), **but every measurement runs under `eval()`, where dropout is the identity and costs nothing**. All reported SOP, spike counts, and flit hops come from dropout-free forward passes.

---

## 3. Mesh design and references

### 3.1 Topology

| Config | Cores | Routing | Hop range |
|---|---|---|---|
| 2×2 | 4 | 2D mesh, XY dimension-order | 1–2 |
| 4×4 | 16 | same | 1–6 |
| 8×8 | 64 | same | 1–14 |

**Hop count = Manhattan distance** (shortest-path table in `net/topology.py`).

**No 2-core configuration**: a 1D direct link has a degenerate hop distribution (all 1), which is not comparable with the heavy-tailed 2D mesh under a single topology family, so it was removed.

### 3.2 Neuron-to-core mapping

**Sequential mapping** (the default of Loihi's Nx SDK and the SpiNNaker compiler):

$$\text{core}(i) = \left\lfloor \frac{i \cdot n_{cores}}{n_{total}} \right\rfloor, \quad n_{total} = 724$$

**This is not a communication-optimizing algorithm; it is a deterministic, reproducible, connectivity-agnostic baseline.**

**Known cost**: the output layer has only 20 neurons, all landing on the last core.

| Mesh | hidden1 cores | hidden2 cores | **out cores** |
|---|---|---|---|
| 2×2 | 0–1 | 1–3 | **3** |
| 4×4 | 0–7 | 7–15 | **15** |
| 8×8 | 0–31 | 31–63 | **62–63** |

**So hidden2 → out traffic converges on 1–2 cores, creating a structural hotspot.** This is part of the "unoptimized upper bound", see §8.

### 3.3 How communication is counted

**A spike is sent to "every core that holds downstream neurons" (multicast semantics); hop count = the sum of shortest-path lengths to each target core.**

Since hidden1 and hidden2 are fully connected, each hidden1 spike genuinely must reach every core holding hidden2 neurons, so the multicast semantics is correct.

**However, summing per-target-core path lengths is equivalent to independent unicast per target, which is an upper bound on a real multicast tree** (see §8).

### 3.4 Per-core capacity

| Item | Value at 8×8 | Loihi limit |
|---|---|---|
| Neurons/core | 11 (hidden1/hidden2), 20 (out) | 1,024 |
| Synapses/core | ~3,800 (hidden2), 7,040 (out) | 128 K |

**Well below the limits; capacity is not a constraint.**

---

## 4. Calculation method

### 4.0 Why we built an analytical model instead of using an existing simulator

**We surveyed the available open SNN energy-evaluation tools and found none that fits this study.** The requirement is narrow but specific: a 45 nm **digital CMOS** cost model that covers Study 1's five categories (including **delay storage** and **leakage of stored state**), handles **learned delays** and **multi-compartment** neurons, and can be driven by the PyTorch models from two upstream codebases.

| Tool | What it models | Why it does not fit |
|---|---|---|
| **SpikeSim** (Yale) | Analog IMC crossbar (RRAM/SRAM), 65 nm | Architecture mismatch: crossbar tiles and ADC/DIFF pipelines, not digital MAC arrays. Its latency model ($\alpha$, `passes`) has no counterpart here |
| **NeuroSim** | Analog/IMC crossbar | Same mismatch as SpikeSim |
| **BookSim** | NoC latency and throughput | Models the interconnect only; no SNN energy, no state or delay storage |
| **snnTorch / SpikingJelly** | Training frameworks | Report MAC/SOP counts at best; no hardware energy model |
| **Lava / Nx SDK** | Targets Loihi 1 | Loihi 1's neuron model is CUBA LIF with **exactly two internal state variables** — the synaptic response current $u_i(t)$ and the membrane potential $v_i(t)$ (Davies et al. 2018, p.3). Our multi-compartment model needs **4 compartments × 2 real state variables = 8**, which cannot be mapped onto Loihi 1 at all. See the note below |
| **NeuroBench** | Hardware-independent metrics | Explicitly provides SOP and activation sparsity, **not** hardware energy |
| **SNNToolBox / N2D2** | Quantization and compilation | Energy modules are coarse (SOP-only) |

**Conclusion**: no existing tool jointly provides (i) a 45 nm digital-CMOS cost basis, (ii) per-category accounting of delay storage and state memory, (iii) support for learned-delay and multi-compartment models, and (iv) a path from these two training codebases. **Hence a transparent analytical model, where every term is a counted quantity times a literature unit price** — which is also what the white paper's Study 1 Setup asks for ("a transparent CMOS energy model with characterized per-operation, per-access, and per-bit-of-storage costs").

#### Loihi 1's structural limits (why its energy numbers cannot be reused)

**Beyond the process node, three hard architectural constraints make Loihi 1 unusable as the platform for this study:**

| Constraint | Loihi 1 | This study needs |
|---|---|---|
| **State variables per neuron** | **2** — $u_i$ (synaptic current) and $v_i$ (membrane potential), fixed CUBA LIF (Davies et al. 2018, p.3). Dendritic compartments exist, but each carries the same fixed $u/v$ dynamics; there is no way to instantiate an arbitrary $N$-th-order state-space model | **8** for the 4-compartment PMSN (4 compartments × real + imaginary parts). It simply does not fit |
| **Synaptic delay** | A per-synapse field in the 5-tuple $(i, j, \text{wgt}, \text{dly}, \text{tag})$ — one delay value attached to each synapse | A **per-neuron** learned delay with a triangular kernel of half-width $\sigma$ (DelRec). Different semantics; Loihi cannot represent the kernel spread |
| **Timing model** | Asynchronous, no global clock ($t_{NoC}$ is not a design parameter); each core iterates over its neuron compartments independently | Clock-driven, per-time-step updates, with an explicit $t_{inf} = \text{MAC count} / (n_{cores} F)$ |

**Consequences for reuse of Loihi measurements:**

- **23.6 pJ per synaptic spike op** is an end-to-end measured figure. It bundles weight RMW, state update, neuron update, and routing, so it **cannot be decomposed** into Study 1's five categories (synaptic / communication / delay / state / leakage). Using it would collapse the very distinction Study 1 set out to measure.
- **81 / 52 pJ per neuron update** likewise includes the full RMW loop over the neuron's state banks, with no way to isolate the delay-buffer and multi-compartment terms.
- **3.0 / 4.0 pJ per tile hop** *is* usable — it is the one quantity whose scope (router + link, one flit) matches our `flit_hops` definition, and we use it as $E_{hop}$ (§4.4), cross-checked against Gorski's 45 nm router power.

**So the honest position is**: Loihi 1 supplies one transferable unit price ($E_{hop}$) and a qualitative confirmation that communication dominates at large mesh, but it cannot host either the delay-trained or the multi-compartment model, and its compound energy figures cannot be decomposed into the accounting this study requires.

**This is a real limitation, not just a convenience**: a cycle-accurate simulator would additionally capture buffer contention, arbitration, and congestion, all of which are currently outside the model (§8.2, §8.3). It also means the *absolute* numbers are not directly comparable with published tool-based results.

### 4.1 Time model

$$t_{inf} = \frac{\text{SOP} + N_{neur} \cdot T \cdot d_{state}}{n_{cores} \cdot F \cdot \text{par}}, \quad F = 2\ \text{GHz},\ \text{par} = 1$$

**Physical meaning**: total MAC count divided by (cores × frequency × per-cycle parallelism).

**`par = 1` means each core completes one MAC per cycle — an optimistic assumption**, see §8 limitation 1.

### 4.2 The five Study 1 categories

Per the white paper's Study 1 Procedure (p.12):

| # | Category | Formula |
|---|---|---|
| 1 | **synaptic operations** | $\text{SOP} \times E_{AC} + N_{neur} T d_{state} \times E_{MAC}$ |
| 2 | **communication** | $fh \times E_{hop}$ |
| 3 | **delay storage** | $N_{delay} \cdot T \cdot 2 \times E_{access}$ |
| 4 | **state memory** | $N_{neur} T d_{state} \cdot 2 \times E_{access}$ |
| 5 | **leakage** | $bits_{state+buffer} \times P_{leak} \times t_{inf}$ |

**$\int P_{idle}dt$, clock tree, data conversion, and feature extraction are not included** — that is the white paper's full §4.3 formula, which Study 1 does not require.

### 4.3 Weight leakage reported separately

**Weights occupy SRAM and leak, but they are independent of the temporal mechanism.** Folding them into the "temporal state share" would dilute the signal (they are 99.6% of all bits).

**So weight leakage is reported in its own column and excluded from criterion (a)'s numerator.**

### 4.4 Unit prices and sources

| Parameter | Value | Source |
|---|---|---|
| $E_{AC}$ | **0.9 pJ** | Horowitz 2014, Fig 1.1.9 (45nm 0.9V, 32-bit FP Add) |
| $E_{MAC}$ | **4.6 pJ** | Ibid. (FP Add 0.9 + FP Mult 3.7) |
| $E_{access}$ | **10 pJ** | Ibid. (8 KB SRAM read) |
| $P_{leak}$ | **11.9 nW/bit** | Horowitz 2014, p.2 (100 mW/MB, 45–32nm) |
| $E_{hop}$ | **3.5 pJ** | Davies et al. 2018 (Loihi), Table 2 (3.0/4.0 pJ per tile hop) |
| **Weight access** | **0** | Dampfhoffer 2023, L2372 (spatially expanded, "remove the weights accesses") |
| Buffer width | **1 bit/slot** | `delay_layers_pytorch.py:299` |
| $F$ | **2 GHz** | Self-defined (measurement baseline) |
| par | **1** | Self-defined (no literature basis) |

**Cross-check for $E_{hop}$**: dividing Gorski 2016's 45nm router $P_{dyn}$ ($P_{total}$ 16–40 mW minus the dominant $P_{stat}$, i.e. ~5–7.3 mW) by the throughput $1.5\times10^9$ flit/s yields 3.3–4.9 pJ/hop, consistent with Loihi's measured value.

---

## 5. Counting results (prices excluded)

### 5.1 Literature basis for the counting conventions

**All four counting definitions come from Dampfhoffer et al. 2023's LIF+inst energy model (Eq. 3.4 and 3.7):**

**Synaptic operations (Eq. 3.4, L2007):**

$$E_{IF+inst} = N_{syn} \cdot N_{spikes/syn} \cdot (E_R^{weight} + E_R^{state} + E_W^{state} + E_{AC})$$

**Per-timestep neuron update (Eq. 3.7, L2037–2038):**

$$E_{LIF+inst} = \underbrace{N_{syn} N_{spikes/syn}(\cdots)}_{\text{event-driven}} + \underbrace{N_{neur} T (E_R^{state} + E_W^{state} + E_{MAC})}_{\text{clock-driven}}$$

**Provenance of the four counts:**

| Count | Our formula | Dampfhoffer basis |
|---|---|---|
| **SOP** | spike × fanout | the $N_{syn} N_{spikes/syn}$ term of Eq. 3.4 |
| **neuron ops** | $N_{neur} \times T \times d_{state}$ | $N_{neur} T$ in Eq. 3.7; the compartment count $d_{state}$ is our PMSN extension |
| **state accesses** | neuron ops **× 2** | $E_R^{state} + E_W^{state}$ in Eq. 3.7; L2032–2034 states "**reading the state, multiplying it with a constant, and writing back the result**" |
| **buffer accesses** | $N_{delay} \times T \times 2$ | no direct source; analogized from the read+write pattern of state accesses (read the expiring slot + write the current spike) |

**Why only Eq. 3.7's second term, not the first**: Dampfhoffer L2039–2044 explains these are two **mutually exclusive** implementation routes — the first is event-driven (update state only when a spike arrives), the second is clock-driven (update all neurons every step), and the original text notes "the update at each timestep is more efficient". Our models are clock-driven (`single_step_forward` at `delay_layers_pytorch.py:95`), so only the second term applies.

**The following are all single-sample averages over **all 2,264 SHD test samples**.**

### 5.2 Raw counts

| T | Model | spikes | SOP | neuron ops | state acc. | buffer acc. |
|---|---|---|---|---|---|---|
| **50** | delay | 3,909 | 1,431,230 | 36,200 | 72,400 | 35,200 |
| | LIF | 8,688 | 1,389,789 | 35,200 | 70,400 | 0 |
| | MC | 9,565 | 1,533,348 | 140,800 | 281,600 | 0 |
| **100** | delay | 7,667 | 2,804,570 | 72,400 | 144,800 | 70,400 |
| | LIF | 13,930 | 1,808,584 | 70,400 | 140,800 | 0 |
| | MC | 15,672 | 2,343,106 | 281,600 | 563,200 | 0 |
| **140** | delay | 10,891 | 3,985,830 | 101,360 | 202,720 | 98,560 |
| | LIF | 19,523 | 2,589,276 | 98,560 | 197,120 | 0 |
| | MC | 21,026 | 2,937,174 | 394,240 | 788,480 | 0 |

**neuron ops = $N_{neur} \times T \times d_{state}$; state accesses = neuron ops × 2 (read + write); buffer accesses = $N_{delay} \times T \times 2$.**

### 5.3 Storage bits

| Model | weight bits | state bits | buffer bits (T=100) |
|---|---|---|---|
| delay | 9,811,584 | 23,168 | 14,432 |
| LIF | 6,015,616 | 22,528 | 0 |
| MC | 6,421,120 | 90,112 | 0 |

**Weights account for 99.6% of all bits and dominate leakage energy.**

### 5.4 Communication hops

| T | Model | 2×2 | 4×4 | 8×8 |
|---|---|---|---|---|
| 50 | delay | 12,468 | 85,422 | 614,151 |
| | LIF | 28,587 | 198,392 | 1,436,650 |
| | MC | 31,381 | 220,566 | 1,591,048 |
| 100 | delay | 24,572 | 169,371 | 1,220,020 |
| | LIF | 45,118 | 308,479 | 2,213,959 |
| | MC | 50,835 | 357,068 | 2,576,434 |
| 140 | delay | 34,980 | 238,466 | 1,706,995 |
| | LIF | 63,203 | 434,092 | 3,125,954 |
| | MC | 68,429 | 474,574 | 3,409,407 |

**delay's hop count is only 43–56% of LIF's — sparser spikes buy less communication.**

### 5.5 Energy after applying unit prices (µJ / inference)

| T | Mesh | Model | synaptic | state | delay | leakage | comm | **total** | weights leak |
|---|---|---|---|---|---|---|---|---|---|
| **50** | 2×2 | delay | 1.455 | 0.724 | 0.352 | 0.071 | 0.044 | **2.646** | 21.417 |
| | | LIF | 1.413 | 0.704 | 0 | 0.048 | 0.100 | **2.265** | 12.751 |
| | | MC | 2.028 | 2.816 | 0 | 0.224 | 0.110 | **5.178** | 15.990 |
| | 4×4 | delay | 1.455 | 0.724 | 0.352 | 0.018 | 0.299 | **2.847** | 5.354 |
| | | LIF | 1.413 | 0.704 | 0 | 0.012 | 0.694 | **2.823** | 3.188 |
| | | MC | 2.028 | 2.816 | 0 | 0.056 | 0.772 | **5.672** | 3.998 |
| | 8×8 | delay | 1.455 | 0.724 | 0.352 | 0.005 | 2.150 | **4.685** | 1.339 |
| | | LIF | 1.413 | 0.704 | 0 | 0.003 | 5.028 | **7.148** | 0.797 |
| | | MC | 2.028 | 2.816 | 0 | 0.014 | 5.569 | **10.426** | 0.999 |
| **100** | 2×2 | delay | 2.857 | 1.448 | 0.704 | 0.161 | 0.086 | **5.256** | 41.989 |
| | | LIF | 1.952 | 1.408 | 0 | 0.063 | 0.158 | **3.580** | 16.814 |
| | | MC | 3.404 | 5.632 | 0 | 0.352 | 0.178 | **9.566** | 25.070 |
| | 4×4 | delay | 2.857 | 1.448 | 0.704 | 0.040 | 0.593 | **5.642** | 10.497 |
| | | LIF | 1.952 | 1.408 | 0 | 0.016 | 1.080 | **4.455** | 4.203 |
| | | MC | 3.404 | 5.632 | 0 | 0.088 | 1.250 | **10.374** | 6.267 |
| | 8×8 | delay | 2.857 | 1.448 | 0.704 | 0.010 | 4.270 | **9.289** | 2.624 |
| | | LIF | 1.952 | 1.408 | 0 | 0.004 | 7.749 | **11.112** | 1.051 |
| | | MC | 3.404 | 5.632 | 0 | 0.022 | 9.018 | **18.076** | 1.567 |
| **140** | 2×2 | delay | 4.053 | 2.027 | 0.986 | 0.244 | 0.122 | **7.432** | 59.651 |
| | | LIF | 2.783 | 1.971 | 0 | 0.090 | 0.221 | **5.066** | 24.051 |
| | | MC | 4.457 | 7.885 | 0 | 0.447 | 0.240 | **13.028** | 31.820 |
| | 4×4 | delay | 4.053 | 2.027 | 0.986 | 0.061 | 0.835 | **7.962** | 14.913 |
| | | LIF | 2.783 | 1.971 | 0 | 0.023 | 1.519 | **6.297** | 6.013 |
| | | MC | 4.457 | 7.885 | 0 | 0.112 | 1.661 | **14.114** | 7.955 |
| | 8×8 | delay | 4.053 | 2.027 | 0.986 | 0.015 | 5.974 | **13.056** | 3.728 |
| | | LIF | 2.783 | 1.971 | 0 | 0.006 | 10.941 | **15.701** | 1.503 |
| | | MC | 4.457 | 7.885 | 0 | 0.028 | 11.933 | **24.303** | 1.989 |

**`weights leak` is reported separately and does not enter `total`.**

---

## 6. Criteria results

### 6.1 (a) Temporal-state share ≥ 10%

**Holds in 8 of 9 configurations.**

| T | Mesh | delay | LIF | MC | (a) |
|---|---|---|---|---|---|
| 50 | 2×2 | 43.4% | 33.2% | 58.7% | ✅ |
| 50 | 4×4 | 38.4% | 25.4% | 50.6% | ✅ |
| **50** | **8×8** | 23.1% | **9.9%** | 27.1% | ❌ |
| 100 | 2×2 | 44.0% | 41.1% | 62.6% | ✅ |
| 100 | 4×4 | 38.9% | 32.0% | 55.1% | ✅ |
| 100 | 8×8 | 23.3% | 12.7% | 31.3% | ✅ |
| 140 | 2×2 | 43.8% | 40.7% | 64.0% | ✅ |
| 140 | 4×4 | 38.6% | 31.7% | 56.7% | ✅ |
| 140 | 8×8 | 23.2% | 12.6% | 32.6% | ✅ |

**The single miss is T=50 at 8×8, where LIF reaches 9.9% — 0.1 point short.** Reason: LIF has no buffer and $d_{state}=1$, so at 64 cores $t_{inf}$ is only 11.5 µs and leakage is essentially zero.

#### Note on the scope of criterion (a)

**The white paper's threshold reads "a non-trivial share of energy in *at least one* brain-inspired model"** — i.e. the criterion is satisfied if **any** of the three models crosses 10% in a given configuration.

**The table above is judged more strictly**: it requires **all three** models to cross 10%, which is why T=50 at 8×8 is marked ❌.

**Under the white paper's own wording, (a) holds in 9/9 configurations** — in that same cell the delay model reaches **23.1%** and the multi-compartment model **27.1%**.

| Reading | (a) holds |
|---|---|
| White paper's wording: **any** model ≥ 10% | **9 / 9** |
| Stricter reading used in the table above: **all** models ≥ 10% | 8 / 9 |

**The lowest value across all nine configurations and all three models is 9.9%** (LIF, T=50, 8×8); **the maximum in every configuration is 27.1–64.0%**. So the premise of Study 1 — that temporal state is a non-trivial energy share in at least one brain-inspired model — holds everywhere, by either reading.

**`results/study1_data.json` was generated with the stricter `all` reading** (the script's `a` flag is `all(...)`); the per-model `ts_share` values are stored there, so either reading can be recomputed from the JSON without re-running the measurement.

### 6.2 (b) Ranking flips

| T | Mesh | spike order | SOP order | energy order | spike≠E | SOP≠E |
|---|---|---|---|---|---|---|
| 50 | 2×2 | DL<LIF<MC | LIF<DL<MC | LIF<DL<MC | **✅** | ❌ |
| 50 | 4×4 | DL<LIF<MC | LIF<DL<MC | LIF<DL<MC | **✅** | ❌ |
| 50 | 8×8 | DL<LIF<MC | LIF<DL<MC | **DL**<LIF<MC | ❌ | **✅** |
| 100 | 2×2 | DL<LIF<MC | LIF<MC<**DL** | LIF<**DL**<MC | **✅** | **✅** |
| 100 | 4×4 | DL<LIF<MC | LIF<MC<**DL** | LIF<**DL**<MC | **✅** | **✅** |
| 100 | 8×8 | DL<LIF<MC | LIF<MC<**DL** | **DL**<LIF<MC | ❌ | **✅** |
| 140 | 2×2 | DL<LIF<MC | LIF<MC<**DL** | LIF<**DL**<MC | **✅** | **✅** |
| 140 | 4×4 | DL<LIF<MC | LIF<MC<**DL** | LIF<**DL**<MC | **✅** | **✅** |
| 140 | 8×8 | DL<LIF<MC | LIF<MC<**DL** | **DL**<LIF<MC | ❌ | **✅** |

**At least one flip: 9/9.**

### 6.3 Summary

| Criterion | Holds (strict) | Holds (white-paper scope) |
|---|---|---|
| (a) ts ≥ 10% | 8/9 | **9/9** |
| (b) spike ≠ energy | 6/9 | 6/9 |
| (b) SOP ≠ energy | 7/9 | 7/9 |
| **(b) at least one flip** | **9/9** | **9/9** |

---

## 7. Analysis

### 7.1 Two simplified estimates mis-rank delay in opposite directions

**T=100, 2×2:**

| Estimate | Order | Verdict on delay |
|---|---|---|
| **spike count** | delay(7.7k) < LIF(13.9k) < MC(15.7k) | **cheapest** (actually 2nd) |
| **SOP** | LIF(1.81M) < MC(2.34M) < delay(2.80M) | **most expensive** (actually 2nd) |
| **full model** | LIF(3.58) < delay(5.26) < MC(9.57) | 2nd |

**Why they disagree:**

- **Spike count** sees delay's sparsity (3,909–10,891, i.e. 45–56% of LIF) but not its buffer cost (0.70 µJ, 13% of its own total).
- **SOP** sees delay's recurrent fanout of 372 (20 output + 352 recurrent), which cancels the sparsity advantage, but not the multi-compartment network's state cost.

### 7.2 MC's compartment cost is visible only in the full model

| T=100, 2×2 | delay | LIF | **MC** |
|---|---|---|---|
| spikes | 7,667 | 13,930 | 15,672 |
| SOP | 2.80M | 1.81M | 2.34M |
| **state** | 1.448 | 1.408 | **5.632 (4.0×)** |
| **leakage** | 0.161 | 0.063 | **0.352 (5.6×)** |

**$d_{state}=4$ multiplies both state accesses and bits by 4, but SOP contains no such factor.**

### 7.3 At large mesh, communication overtakes state

| T=100, 8×8 | delay | LIF | MC |
|---|---|---|---|
| comm | **4.270** | 7.749 | 9.018 |
| comm share | 46% | 70% | 50% |

**delay's hop count is only 55% of LIF's, so at 64 cores it becomes the cheapest (9.29 vs 11.11).**

**SOP fails completely here: it ranks delay as most expensive (2.80M) when it is actually cheapest.**

### 7.4 Effect of time-step resolution

**delay's accuracy rises monotonically as frames get finer (86.56 → 87.24 → 87.99), while both PMSN variants peak at 28 ms.**

**So the two model families respond to resolution in opposite directions:**

| | Mechanism | Preference |
|---|---|---|
| delay | discrete frame-level delays | **finer resolution** (finer frames express delay granularity more precisely) |
| PMSN | continuous time constants | **28 ms** (closer to SHD phoneme durations of 50–200 ms) |

**This is precisely what Metrics ③ (state cost vs time-step resolution) is meant to quantify.**

---

## 8. Placement and cost-model choices affect the result

**This is the report's most important caveat: the absolute numbers depend on the following five modelling choices, each of which has alternatives.**

### 8.1 Placement is not optimized

**Sequential mapping puts the entire output layer on the last core (core 15 at 4×4, cores 62–63 at 8×8).**

**Effect**: hidden2 → out traffic forms a hotspot and hop counts are systematically inflated. **A communication-aware placement (e.g. spreading the output layer) would lower flit_hops and thus the comm term.**

**At 8×8 the comm term is 46–70% of the total, so this choice matters most.**

### 8.2 Multicast is computed as a unicast upper bound

**Current hop count = sum of per-target-core shortest paths, equivalent to sending an independent unicast to each target.**

**A real NoC multicast uses a replication tree, copying only at path forks.**

**Effect**: flit_hops is overestimated 2–3×, and the comm term scales accordingly. **All three models are inflated by the same factor, so the ordering is unaffected.**

### 8.3 Multiple spikes in the same timestep are not aggregated

**Each spike is counted independently. Physically, multiple spikes from the same source core to the same target set within one timestep could share one flit payload.**

**Effect**: same direction as §8.2, further overestimating comm.

### 8.4 Time model par = 1 has no literature basis

**Assumes a 32-bit FP MAC completes in one cycle at 2 GHz.**

**Effect**: if a MAC actually needs 4 cycles, $t_{inf}$ is 4× larger and $P_{leak} \times t_{inf}$ scales accordingly.

**But Study 1's leakage is only 0.2–3.7% (excluding weights), so criteria (a)(b) are unaffected. Weight leakage (41.99 µJ at T=100, 2×2) is heavily affected, but it is reported separately.**

### 8.5 Mixed process nodes

| Parameter | Node |
|---|---|
| $E_{AC}$ / $E_{MAC}$ / $E_{access}$ / $P_{leak}$ | **45 nm** |
| $E_{hop}$ | **14 nm** (Loihi) |

**Node scaling (14→45 nm, about 2.75×) would make $E_{hop}$ ≈ 9.6 pJ/hop, scaling the comm term up accordingly.**

### 8.6 How to read these numbers

**Trustworthy:**
- The relative ranking of the three models (under one set of modelling assumptions)
- The order of magnitude of the temporal-state share (tens of percent, not 1% and not 90%)
- The conclusion that "simplified estimates mis-rank"

**Not trustworthy:**
- Absolute µJ values (could differ by 2–3×)
- Any cross-work comparison of the form "X times more efficient than chip Y"

---

## 9. Reproduction

### 9.1 Dependencies

- Python 3.11+, PyTorch, h5py, numpy, matplotlib
- DelRec source (`alexmaxad/DelRec`) and the SHD dataset
- Three model checkpoints (paths in §9.4)

**Training was done on Google Colab with its default environment; measurement runs on local CPU (`map_location="cpu"`). Exact package versions were not pinned beyond `spikingjelly@d4fee3a1`.**

### 9.2 From scratch

```bash
# 1. Train (~2 hours total; full commands in patches/TRAINING.md)
python patches/patch_delrec.py --root /content/DelRec --T 100
python patches/patch_pmsn.py   --root /content/PMSN   --T 100
# ... 9 runs in total

# 2. Measure + energy (~12 min)
#    NOTE: despite the filename, this runs ALL 2,264 test samples by default.
python scripts/study1_10samples.py

# 3. Plot (~5 s)
python scripts/plot_study1.py

# 4. Consistency check (10-sample subset vs all samples, ~2 min)
python scripts/check_10_samples.py
```

**Note on the filename `study1_10samples.py`**: the name is historical — the script was first written to quick-check 10 samples. **Its default is now all 2,264 samples** (`N_TEST = -1` and `--n` defaults to `-1`). Every number in this report comes from the full 2,264 samples. Use `--n 10` for a fast smoke test, and `--save <path>` to export the JSON result. The separate script `check_10_samples.py` is the cross-validation tool that compares a 10-sample subset against the full set (§9.3); it is not the source of the reported data.

**In `scripts/study1_10samples.py`, `N_TEST = -1` means all 2,264 samples; set it to 10 for a quick check.**

### 9.3 Data reliability

**Cross-validation, 10 samples vs 2,264 samples:**

| Model | T | SOP ratio | flit_hops ratio |
|---|---|---|---|
| delay | 50 / 100 / 140 | 1.010 / 0.992 / 0.979 | 1.009 / 0.991 / 0.981 |
| LIF | 50 / 100 / 140 | 1.004 / 1.015 / 1.001 | 1.014 / 1.013 / 1.000 |
| MC | 50 / 100 / 140 | 1.014 / 1.002 / 1.007 | 1.018 / 1.003 / 1.012 |

**All deviations <2.1%, so the 2,264-sample average has converged.**

### 9.4 Checkpoint locations

```
新SNN\delrec_runs\
├── delay_T50_ts28_7ep\delay_w352_T50_ts28_7ep\
├── delay_w352_T100_ts14_7ep\delay_w352_T100_ts14_7ep\
├── delay_T140_ts10_7ep\delay_w352_T140_ts10_7ep\
├── pmsn_T50_ckpts\pmsn_{lif,mc}_T50_fill_{5,3}ep\...\checkpoint.pth
├── pmsn_new_ckpts\pmsn_{lif,mc}_T100_fill_{5,3}ep\...\checkpoint.pth
└── pmsn_T140_ckpts\pmsn_{lif,mc}_T140_fill_{5,3}ep\...\checkpoint.pth
```

**`DELAY_RUNS` and `PMSN_RUNS` at the top of `scripts/study1_10samples.py` hold absolute paths and must be edited when relocating.**

---

## 10. Known limitations

| # | Limitation | Impact | See |
|---|---|---|---|
| 1 | `par = 1` has no literature basis | $t_{inf}$ and weight leakage | §8.4 |
| 2 | $E_{hop}$ is a 14nm value mixed into a 45nm model | comm term | §8.5 |
| 3 | $P_{leak}$ is an 8 MB cache value (small SRAM leaks more) | leakage underestimated | `PARAMETERS.md` |
| 4 | flit_hops is a unicast upper bound | comm overestimated 2–3× | §8.2 |
| 5 | Same-timestep spikes not aggregated | comm further overestimated | §8.3 |
| 6 | Placement not optimized | output-layer hotspot | §8.1 |
| 7 | FP32 conflicts with the white paper's §7.1 "low-precision" | narrative only (INT32 differs <0.5%) | `PARAMETERS.md` |
| 8 | $\int P_{idle}dt$ excluded | matches Study 1's five categories; not a defect | §4.2 |
| 9 | Weight leakage reported separately | a reporting choice, not a defect | §4.3 |

---

## 11. Figures

| File | Question it answers |
|---|---|
| `figs/study1_main_T100_2x2.png` | The white paper's requested form: stacked energy breakdown of the three models with spike counts overlaid (T=100, 2×2) |
| `figs/study1_grid_resolution_mesh.png` | How resolution and mesh size change the energy composition (3×2 grid) |
| `figs/study1_ranking_flip.png` | Ranking by the three estimates at each resolution, marking which pair flips |

---

## 12. References

| Use | Reference |
|---|---|
| FP32 unit prices | Horowitz, *Computing's Energy Problem (and what we can do about it)*, ISSCC 2014, Fig 1.1.9 |
| Per-hop communication | Davies et al., *Loihi: A Neuromorphic Manycore Processor with On-Chip Learning*, IEEE Micro 2018, Table 2 |
| Architecture assumption (resident weights) | Dampfhoffer et al., *Are SNNs Really More Energy Efficient Than ANNs?*, 2023, §3.3 |
| Router power (cross-check) | Gorski, *Dissertation*, 2016, Table I.5 / I.7 |
| delay model | DelRec (`alexmaxad/DelRec`) |
| PMSN model | Chen et al., *PMSN: A Parallel Multi-compartment Spiking Neuron*, arXiv:2408.14917 |
| Criteria source | *Time as a First-Class Resource*, §4.3 and Preliminary Studies, p.11–12 |
