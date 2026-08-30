"""
GPU Transformer Parity Profiler for SpikeStack Lite.

Directly compares SpikingTransformer against a standard Dense Softmax Transformer
on standard NVIDIA GPUs, measuring GPU training time, throughput (samples/sec),
peak VRAM allocated, and VRAM memory savings.

Usage:
    python -m spikestack_lite.utils.gpu_profiler
    python -m spikestack_lite.utils.gpu_profiler --seq-len 512 --batch-size 64
"""

import argparse
import random
import time
import torch
import torch.nn as nn

from spikestack_lite.models.transformer import SpikingTransformer


class StandardDenseTransformer(nn.Module):
    """Standard Dense Softmax Transformer baseline with O(N^2) attention."""

    def __init__(self, d_model=128, seq_len=64, num_classes=10, num_heads=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            activation="relu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        out = self.transformer(x)
        return self.classifier(out.mean(dim=1))


def profile_gpu_parity(batch_size=64, seq_len=128, d_model=128, num_classes=10,
                       num_layers=2, steps=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=" * 70)
    print(f" SPIKESTACK-LITE GPU TRANSFORMER PARITY PROFILER")
    print(f" Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f" Geometry: Batch={batch_size}, SeqLen={seq_len}, d_model={d_model}, Layers={num_layers}")
    print(f"=" * 70)

    # 1. Profile Standard Dense Softmax Transformer
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(0)
    random.seed(0)

    dense_model = StandardDenseTransformer(
        d_model=d_model, seq_len=seq_len, num_classes=num_classes,
        num_heads=4, num_layers=num_layers
    ).to(device)

    opt_dense = torch.optim.AdamW(dense_model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)

    # Warmup
    for _ in range(5):
        opt_dense.zero_grad()
        crit(dense_model(x), y).backward()
        opt_dense.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        opt_dense.zero_grad()
        crit(dense_model(x), y).backward()
        opt_dense.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dense_dt = (time.perf_counter() - t0) / steps * 1000.0
    dense_throughput = (batch_size / (dense_dt / 1000.0))
    dense_vram = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0

    # 2. Profile SpikingTransformer (Spikestack-Lite 5-in-1 Pipeline)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(0)
    random.seed(0)

    spike_model = SpikingTransformer(
        input_dim=d_model, d_model=d_model, seq_len=seq_len, num_classes=num_classes,
        num_heads=4, num_layers=num_layers, n_repeats=3, temporal=True,
        use_spikeskip=True, use_gsmc=True, use_exact=True
    ).to(device)

    opt_spike = torch.optim.AdamW(spike_model.parameters(), lr=1e-3)

    # Warmup
    for _ in range(5):
        opt_spike.zero_grad()
        out = spike_model(x)
        loss = crit(out, y) + spike_model.latency_loss(spike_model._pooled_for_latency, y)
        loss.backward()
        opt_spike.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        opt_spike.zero_grad()
        out = spike_model(x)
        loss = crit(out, y) + spike_model.latency_loss(spike_model._pooled_for_latency, y)
        loss.backward()
        opt_spike.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    spike_dt = (time.perf_counter() - t0) / steps * 1000.0
    spike_throughput = (batch_size / (spike_dt / 1000.0))
    spike_vram = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0

    # Summary Report
    vram_savings = ((dense_vram - spike_vram) / max(1e-6, dense_vram)) * 100.0
    speed_ratio = dense_dt / spike_dt

    print(f"\n--- BENCHMARK RESULTS ---")
    print(f"{'Model Architecture':<35} | {'Speed (ms/step)':<15} | {'Throughput (samp/s)':<20} | {'Peak VRAM (MB)':<15}")
    print(f"-" * 95)
    print(f"{'Standard Dense Softmax Transformer':<35} | {dense_dt:15.2f} | {dense_throughput:20.1f} | {dense_vram:15.1f}")
    print(f"{'SpikingTransformer (5-in-1 Pipeline)':<35} | {spike_dt:15.2f} | {spike_throughput:20.1f} | {spike_vram:15.1f}")
    print(f"-" * 95)
    print(f"VRAM Memory Savings:  {vram_savings:+.1f}% vs. Standard Transformer")
    print(f"Speed Ratio:          {speed_ratio:.2f}x standard transformer speed")
    print(f"=" * 70)


def main():
    parser = argparse.ArgumentParser(description="GPU Transformer Parity Profiler")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    profile_gpu_parity(
        batch_size=args.batch_size, seq_len=args.seq_len,
        d_model=args.d_model, num_layers=args.layers, steps=args.steps
    )


if __name__ == "__main__":
    main()
