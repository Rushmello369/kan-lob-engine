# kan-lob-engine

Multiplier-free FPGA inference for short-term price direction prediction on limit order
book data, using a Kolmogorov–Arnold Network compiled entirely into lookup tables —
targeting **zero DSP and zero BRAM** usage.

> **Status:** exploratory. This is a learning vehicle for a recent line of FPGA research,
> not a finished result. Numbers below are placeholders until measured.

---

## Purpose

To work through a 2026 result — KANs compiled to pure LUT logic — end to end, from
quantization-aware training in PyTorch down to synthesised RTL and a timing report, and
to apply it to a domain it has not been applied to publicly: limit order book signals.

The point is the full hardware–software path, not the trading signal. Every stage
(training → quantization → truth-table export → generated Verilog → bit-exact
verification → synthesis) has to be built and understood, because that path is where the
actual learning is.

## Why KAN

A LUT neural network pre-enumerates a whole neuron into a truth table, so inference is
table lookup with no arithmetic. The problem is that table size grows as `2^(β×F)` —
exponential in fan-in `F` — which forces brutal sparsity and bit-width limits.

A KAN puts the learnable function on each **edge** rather than a weight, and nodes only
add. Each edge is a single-input function, so its table is `2^β` entries regardless of
fan-in, and hardware cost grows **linearly** instead of exponentially. That is the whole
argument: it removes the constraint that makes conventional LUT networks awkward to
scale.

The order book side is a good fit for the same reason it is a good fit for FPGAs
generally — small feature vectors, hard latency requirements, and a genuine premium on
*deterministic* latency rather than merely low average latency.

## What this is expected to achieve

A comparison table on identical data and an identical device, with real measurements:

| Implementation | Macro F1 | LUT | DSP | Fmax (MHz) | Latency |
| --- | --- | --- | --- | --- | --- |
| Floating-point KAN (software reference) | — | – | – | – | CPU |
| Fixed-point MLP (`hls4ml`, uses DSP) | — | — | — | — | — |
| LogicNets / PolyLUT (same data) | — | — | 0 | — | — |
| **kan-lob-engine** | — | — | 0 | — | — |

Alongside that:

- A bit-exact match between the Python fixed-point model and the generated RTL.
- An accuracy-vs-resource Pareto curve over bit width, grid size and pruning ratio.
- Plots of the learned edge functions, as a readable account of what the model picked up
  about order imbalance and price direction.

No latency target is promised in advance. Whatever the tools report is what gets
reported, along with an explanation of why.

## A caveat on the data

FI-2010 is used as a **fixed benchmark for comparing implementations**, not as evidence
of a tradeable signal. Its published headline accuracies are partly an artefact of
whole-dataset normalisation and smoothed multi-tick labels.

---

## Layout

```
src/kanlob/     training, quantization, truth-table export, Verilog generation
rtl/src/        hand-written RTL          rtl/generated/  exported tables (gitignored)
tb/             cocotb testbenches        synth/          Vivado scripts and reports
notebooks/      exploration               results/        figures and comparison tables
docs/           project plan              data/           FI-2010 (gitignored)
```

## Reading

- [`docs/kan-lob-engine_plan.md`](docs/kan-lob-engine_plan.md) — full plan, model lineage,
  roadmap and phase deliverables.
