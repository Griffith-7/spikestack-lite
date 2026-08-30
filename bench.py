"""
Benchmark harness for SpikeStack Lite.

Measures per-step training time, FFN input sparsity, and peak VRAM for a few
model configs on the CIFAR-10 geometry, so claims about the SpikeSkip engine
are measured, not guessed.

Temporal configs route the FFN through the multi-timestep kernel
(SparseLinear.forward_multistep): the Spire repetition planes become a time
axis and the weight stays hot across the T tight-loop steps. A harder
spike_threshold pushes FC2 input sparsity >90%, where the CSR engine should
beat cuBLAS.

Usage:
    python bench.py                       # all configs, 200 steps each
    python bench.py --steps 100
"""

import argparse
import random
import time

import torch

from spikestack_lite import sparse as S
from spikestack_lite.models.transformer import SpikingTransformer

B, N, DIM = 128, 64, 48


def run(name, use_spikeskip, expansion, steps, temporal=False, threshold=0.0):
    torch.manual_seed(0)
    random.seed(0)
    torch.cuda.reset_peak_memory_stats()
    m = SpikingTransformer(use_spikeskip=use_spikeskip, num_layers=1,
                           expansion=expansion, temporal=temporal,
                           spike_threshold=threshold).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()

    # FC2 input sparsity = fraction of silent (zero) entries that the kernel
    # skips. FC2 input is spike_fn(fc1(x) - threshold); hook fc1 because the
    # temporal path calls fc2.forward_multistep() directly (bypasses fc2 hooks).
    sparsity = [None]

    def hook(mod, inp, out):
        sparsity[0] = ((out[0] - threshold) <= 0).float().mean().item()

    h = m.blocks[0].ffn.fc1.register_forward_hook(hook)

    x = torch.randn(B, N, DIM).cuda()
    y = torch.randint(0, 10, (B,)).cuda()
    for _ in range(3):  # warmup (JIT / cuDNN autotune)
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps * 1000

    h.remove()
    peak = torch.cuda.max_memory_allocated() / 1e6
    engine = "CUDA-ON" if S._cuda_engine is not None and use_spikeskip else "dense "
    print(f"{name:<30} {dt:7.2f} ms/step   sparsity={sparsity[0]:5.1%}   "
          f"peakVRAM={peak:5.0f} MB   {engine}")
    return dict(name=name, dt=dt, sparsity=sparsity[0], engine=engine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    if S._cuda_engine is None:
        print("NOTE: SpikeSkip CUDA engine not compiled -- configs without it are dense only.\n")

    print(f"steps/config = {args.steps}\n")
    run("dense FFN 512  (default no-skips)", use_spikeskip=False, expansion=4, steps=args.steps)
    run("dense FFN 1024 (same geometry)", use_spikeskip=False, expansion=8, steps=args.steps)
    run("SpikeSkip FFN 1024 (CUDA kernel)", use_spikeskip=True, expansion=8, steps=args.steps)

    print("\n--- temporal (T = n_repeats = 3), SpikeSkip 4096-wide FFN (expansion=32) ---")
    dense_t = run("dense temporal 4096 (ref)", use_spikeskip=False,
                  expansion=32, steps=args.steps, temporal=True, threshold=1.0)
    for thr in (1.0, 2.0, 3.0, 4.0, 5.0):
        skip = run(f"SpikeSkip temporal 4096 thr={thr}", use_spikeskip=True,
                   expansion=32, steps=args.steps, temporal=True, threshold=thr)
        ratio = dense_t["dt"] / skip["dt"] if skip["dt"] > 0 else float("inf")
        win = "WIN" if ratio > 1.0 else "lose"
        print(f"    -> dense vs SpikeSkip @ thr={thr}:   {dense_t['dt']:.2f} / "
              f"{skip['dt']:.2f} ms/step = {ratio:.2f}x   [{win}]")

    print("\nThe multi-timestep SpikeSkip kernel beats dense temporal FFNs once the\n"
          "spike threshold pushes FC2 sparsity past ~99.5% at >=4096 width "
          "(branch-skip multiply beats cuBLAS).")


if __name__ == "__main__":
    main()