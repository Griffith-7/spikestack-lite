# SpikeStack Lite

A lightweight **3-in-1 spiking transformer** library that combines three
solutions from the source repository set into one GPU-friendly model:

| # | Source solution | Role in SpikeStack Lite | Fidelity |
|---|---|---|---|
| 1 | **Spire** (`source_repos/spire`) | `encode/spire.py` — dithered multi-spike **time-repetition** encoder with trainable per-channel frontiers | Repetition-code mechanism (C1/H2) carried over; the theory (R(D) bounds, channel model) is out of scope for the on-chip forward pass |
| 2 | **AstroHebbian** (`source_repos/astrocyte_hebbian`) | `nn/attention.py` — 100% spiking **linear attention** (Hebbian `KᵀV` trace + key-mass norm). Ported near-verbatim | Original headline: 85.42% psMNIST, 3.1x faster than dense Transformer |
| 3 | **SpikeSkip** (`source_repos/spikeskip`) | `sparse/` — CUDA **CSR sparse engine** powering the FFN when it pays off; multi-timestep variant reuses weights across the T repetition planes in one tight C++ loop | Engine packaged here; JIT-compiles on first import, falls back to dense if no CUDA/toolchain |

## Design decisions (explicitly NOT included)

- **GSMC** (`source_repos/gsmc`) — the gated sequential memory cell is
  *deliberately skipped*: SpikeStack Lite trades long-range sequential memory
  for fully parallel GPU processing (a direct spiking replacement for
  standard Transformers rather than an RNN).
- **Exact-SNN** (`source_repos/exact_snn`) — not used; the library trains with
  surrogate gradients (as in the AstroHebbian source).

## Install

```bash
pip install -e .          # picks up pyproject.toml + CUDA source as package data
```

Requirements: Python ≥ 3.9, PyTorch ≥ 2.0, torchvision ≥ 0.15.
For the SpikeSkip CUDA engine: NVIDIA GPU, CUDA Toolkit, and MSVC/GCC.
The engine auto-detects CUDA_HOME + MSVC on import and JIT-compiles once
(torch caches the build). If any piece is missing it degrades to dense
matmul with a warning — the model still works.

## Quickstart

```python
import torch
from spikestack_lite.models.transformer import SpikingTransformer

model = SpikingTransformer(
    input_dim=48, d_model=128, seq_len=64, num_classes=10,
    num_heads=4, num_layers=1, n_repeats=3,
    use_spikeskip=True,          # auto-disabled if the CUDA engine isn't available
)
x = torch.randn(2, 64, 48)       # (batch, seq_len, input_dim)
logits = model(x)                # (batch, num_classes)
```

## Training CIFAR-10

**Recommended tiny profile** (best accuracy-per-cost on a 4 GB GPU):

```bash
python train_cifar10_lite.py --tiny
```

This is the single-block, 128-wide dense model (206k params) with temporal
T=3 rate pooling, FC1 spike threshold 2, augment + cosine, 10 epochs —
measured **56.95%** test accuracy at ~26 s/epoch / ~100 MB. It essentially
matches the 57.13% result from the much larger 1.13M-param config at 1/5.5x
the parameters and half the wall time (see "Measured accuracy" below).

Other presets / raw knobs:

```bash
python train_cifar10_lite.py                       # default 5 epochs, seed 0
python train_cifar10_lite.py --epochs 10 --augment --cosine   # stronger recipe
python train_cifar10_lite.py --lr 1e-3 --seed 42
python train_cifar10_lite.py --no-spikeskip --expansion 8     # same-size dense FFN comparison
```

Key flags:

| flag | effect |
|---|---|
| `--epochs / --batch-size / --seed` | run configuration |
| `--layers` | number of stacked AstroHebbian blocks |
| `--repeats` | Spire repetition count (`1` = pure binary, `>1` = graded multi-spike) |
| `--lr` | AdamW learning rate (default `1e-3`) |
| `--augment` | random 4px-pad crop + horizontal flip |
| `--cosine` | cosine-anneal the LR over the run |
| `--exp-name --save-dir` | checkpoint location (`<save-dir>/<exp-name>/best.pt` + `final.pt`) |
| `--resume <path>` | continue from a saved checkpoint |
| `--no-spikeskip` | force dense FFN (engine auto-falls-back anyway if unbuildable) |
| `--expansion` | FFN expansion (default: 8 with SpikeSkip, else 4) |
| `--temporal` | treat the Spire repetition planes as a time axis `(T, B, N, d)` so the FFN runs the multi-timestep SpikeSkip kernel (mean over time == graded code) |
| `--threshold` | harder FC1 spike threshold (subtracted pre-heaviside); >0 pushes FC2 sparsity above ~95% at a small accuracy cost (tune per model size: 2 for the 128-wide profile, 3-4 for wide FFNs) |
| `--tiny` | recommended tiny profile (temporal + threshold 2 + augment + cosine, 10 epochs, dense FFN); overrides the config knobs above |

Data path resolution order: `$SNN_DATA_DIR` env var → local `./data` →
the original absolute CIFAR folder → auto-download into `./data`.

## Tests

```bash
python -m pytest tests -v        # CPU + CUDA; engine tests auto-skip if no GPU
```

Covers the encoder's binary/graded code shapes (incl. the temporal-plane
equivalence: mean over time == graded code), surrogate gradients, the
3-in-1 model (stacking, reproducibility, gradient flow through the CUDA
engine — a regression for the FFN being silently frozen), exact
sparse-vs-dense forward/backward parity for `SparseLinear` (single- and
multi-timestep), and the CPU-input safety guard.

## Benchmarks

```bash
python bench.py --steps 200      # ms/step + sparsity + peak VRAM per config
```

Reports per-step wall time, FFN input sparsity, and peak VRAM for dense and
SpikeSkip configs (including temporal `T = n_repeats` mode with a
`--threshold` sweep). Measured on a 4 GB laptop RTX at batch 128, 60 steps:

```
dense FFN 512  (default no-skips)  12.9 ms/step   sparsity=50.6%   205 MB
dense FFN 1024 (same geometry)     16.5 ms/step   sparsity=52.9%   307 MB
SpikeSkip FFN 1024 (CUDA kernel)   31.7 ms/step   sparsity=52.6%   374 MB

--- temporal (T=3), 4096-wide FFN (expansion=32) ---
dense temporal 4096 (ref)        106.0 ms/step   sparsity=91.7%  2672 MB
SpikeSkip temporal 4096 thr=1.0  129.0 ms/step   sparsity=91.8%   WIN=lose
SpikeSkip temporal 4096 thr=2.0  112.7 ms/step   sparsity=95.1%   WIN=lose
SpikeSkip temporal 4096 thr=3.0  105.2 ms/step   sparsity=98.2%   WIN
SpikeSkip temporal 4096 thr>=4   102.9 ms/step   sparsity>99%     WIN
```

What this means, measured rather than guessed:

- On the **isolated 4096-wide FC2 slice**, the multi-timestep kernel is
  **1.3-2.1x faster than cuBLAS** once FC2 input sparsity clears ~99.5%
  (`bench.py` microbench with GEMM-equivalent geometry).
- **End-to-end at model level** the kernel closes the gap to a ~1.01-1.03x
  win at `--threshold 3` (98% sparsity). The rest of the FFN — the dense
  `fc1` (`128 -> 4096`) projection, which dominates FFN FLOPs — plus the
  attention/encoder bound the end-to-end gain. SpikeSkip accelerates the
  *wide output-side* projection; it is not a free pass for the whole FFN.
- The kernel was also fixed along the way: the original one-thread-per-row
  CSR build was ~32x *uncoalesced* (warps touched 32 rows), and the prefix
  sum ran a single serial thread over all rows (an 8k-deep latency chain per
  timestep at batch 8192). The engine now uses a coalesced warp-per-row build
  + a two-phase parallel prefix sum.

The strictly-sparse SweetSpot is therefore: **wide FFN (`--expansion 32`),
temporal mode, and a spike threshold in the thr≈3-4 range** — where skipping
maintains beat cuBLAS even though the whole-model speedup is modest.

## Task geometry vs. vanilla transformers (honest comparison)

The spiking mechanism's win/loss against a same-geometry analog transformer
is decided by **sequence length N vs. hidden d**, not by "SNNs vs
transformers":

| model (same dims: d=128, 1 block, batch 64) | N=784 (psMNIST-style, sequential) | N=64 (CIFAR patches) |
|---|---|---|
| AstroHebbian spiking attention | **52.7 ms/step**, no N×N matrix | 16.9 ms/step (temporal) |
| dense softmax transformer | 168.3 ms/step | **4.1 ms/step** |
| spiking wins | **3.2x faster** (matches the source repo's "3.1x") | loses on acc/speed/VRAM |

Reason (FLOP crossover): dense attention is **O(N²·d)**, linear (Hebbian)
attention is **O(N·d²)**. When **N ≫ d** the spiking linear attention wins
both speed and memory (psMNIST, N=784); when **N ≈ d** the dense softmax
attention is cheaper and the spiking model's surrogate overhead dominates
(CIFAR patches, N=64). The library's three mechanisms never outperform a
dense transformer on the short-sequence regime — they buy accuracy via
temporal/threshold regularization and sparsity (99.9% FC2) for neuromorphic
deployment, not raw CUDA speed.

Reproduce the fairness duel:

```bash
python train_vanilla_baseline.py --augment --cosine   # same budget as --tiny
```

## Measured accuracy

Best test accuracy so far (4 GB laptop RTX, seed 0, batch 64):

| config | params | test acc | wall time |
|---|---|---|---|
| **`--tiny`** (temporal, thr 2, aug+cosine, 10 ep) | **206k** | **56.95%** | ~26 s/epoch |
| wide temporal-4096 thr 3 (10 ep, aug+cosine) | 1.13M | **57.13%** | ~52 s/epoch |
| plain spike lite (5 ep, no tricks) | 206k | 51.42% | ~12 s/epoch |
| **vanilla analog transformer, same budget** | 212k | **62.37%** | ~4.1 ms/step |

```bash
python train_cifar10_lite.py --tiny                 # recommended tiny profile
python train_cifar10_lite.py --epochs 10 --temporal --threshold 3 \
    --expansion 32 --augment --cosine --exp-name temporal-4096-thr3
```

The wide run hit **57.13%** at FC2 input sparsity **99.9%** throughout
training — +5.7pp over the 51.42% non-temporal, no-aug baseline, while the
FFN output-projection ran the sparse engine in its winning regime for the
whole run. The tiny profile trades only 0.18pp of that for **1/5.5x the
parameters and a 2x faster wall clock** — the recommended default.

Honest bottom line: on CIFAR-10 (N=64) a same-budget vanilla transformer
beats this library on accuracy (62.37 vs 56.95%), speed (4.1 vs 16.9
ms/step) and VRAM (64 vs 269 MB). The spiking mechanisms are the *right*
choice where they win legitimately: long-sequence tasks (N≫d) and on
neuromorphic hardware, where zero-skipping and 99.9% sparsity pay off.
Running the spike model on a short-sequence GPU task against a dense
transformer is the wrong arena and will lose — see "Task geometry" above.

## Module layout

```
spikestack_lite/
├── encode/spire.py        SpireEncoder: dithered multi-spike repetition code
├── nn/attention.py        AstroHebbian spiking linear attention (+ FFN/block)
├── sparse/                SpikeSkip CUDA engine + SparseLinear
│   └── src/sparse_linear.cu
└── models/transformer.py  SpikingTransformer (the 3-in-1 model)
tests/                     pytest suite (encode / attention / transformer / sparse / trainer)
bench.py                   benchmark harness (ms/step, sparsity, peak VRAM)
train_cifar10_lite.py       training entry point (--tiny = recommended profile)
train_vanilla_baseline.py   same-budget vanilla transformer for honest comparisons
```

## Reproducibility

Seeds are centralized in `train_cifar10_lite.py` (`--seed`, default 0 and
applied to `torch` + `random`). Set `SNN_DATA_DIR` to pin the dataset path.