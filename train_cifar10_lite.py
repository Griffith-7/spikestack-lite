import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from spikestack_lite.models.transformer import SpikingTransformer

# ==============================================================
# DEFAULT COMPUTATION DIMENSIONS
# ==============================================================
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 5
SEQ_LEN = 64
INPUT_DIM = 48
D_MODEL = 128
NUM_HEADS = 4

# Pre-existing CIFAR folder on this machine (used only as a last fallback).
LEGACY_DATA_DIR = r"C:\Users\sumit\Downloads\snn-solutations\data\cifar-10-python"


def resolve_data_dir():
    """Portable data path: env var -> local ./data -> legacy absolute -> download."""
    env = os.environ.get("SNN_DATA_DIR")
    if env and os.path.isdir(env):
        return env
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if os.path.isdir(os.path.join(local, "cifar-10-python", "cifar-10-batches-py")):
        return os.path.join(local, "cifar-10-python")
    if os.path.isdir(os.path.join(local, "cifar-10-batches-py")):
        return local
    if os.path.isdir(LEGACY_DATA_DIR):
        return LEGACY_DATA_DIR
    os.makedirs(local, exist_ok=True)
    return local



def patchify(images):
    """(B, 3, 32, 32) -> (B, 64, 48) non-overlapping 4x4 patches."""
    B = images.shape[0]
    patches = images.unfold(2, 4, 4).unfold(3, 4, 4)
    patches = patches.contiguous().view(B, 3, 64, 16)
    patches = patches.permute(0, 2, 3, 1).contiguous().view(B, 64, 48)
    return patches


def build_parser():
    parser = argparse.ArgumentParser(description="Train SpikeStack Lite on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader num_workers (Windows: >0 spawns loader processes)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3, help="Spire dithered repetition count (n_repeats)")
    parser.add_argument("--lr", type=float, default=1e-3, help="peak learning rate (AdamW)")
    parser.add_argument("--augment", action="store_true",
                        help="CIFAR-10 augmentation: random 4px-pad crop + horizontal flip")
    parser.add_argument("--cosine", action="store_true",
                        help="cosine annealing of the learning rate over --epochs")
    parser.add_argument("--exp-name", type=str, default="lite",
                        help="experiment name (subdir under --save-dir)")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="root directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="path to a saved .pt checkpoint to resume from")
    parser.add_argument("--no-spikeskip", action="store_true",
                        help="Disable SpikeSkip FFN (it is on by default; auto-fallback to dense if the CUDA engine is unavailable)")
    parser.add_argument("--expansion", type=int, default=None,
                        help="FFN expansion factor (default: 8 with SpikeSkip, else 4). "
                             "Paired with --no-spikeskip, e.g. 8 gives a same-size dense FE comparison.")
    parser.add_argument("--temporal", action="store_true",
                        help="Use the Spire repetition planes as a time axis (T, B, N, d); the "
                             "FFN runs the multi-timestep SpikeSkip kernel. Mean over time == graded code.")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Harder FC1 spike threshold (subtracted pre-heaviside). >0 pushes FC2 "
                             "sparsity >90%% at a small accuracy cost.")
    parser.add_argument("--tiny", action="store_true",
                        help="Recommended tiny profile: temporal + threshold 2 + augment + cosine, "
                             "10 epochs, dense FFN (206k params, ~57%% CIFAR-10 at ~26 s/epoch). "
                             "Overrides epochs/repeats/temporal/threshold/augment/cosine/no-spikeskip.")
    parser.add_argument("--gsmc", action="store_true",
                        help="Enable the fused GSMC memory stage over the Spire T-planes "
                             "(requires --temporal). Adds long-range memory at a controlled "
                             "~1.5-2x cost on that stage (tunable via --repeats/T).")
    parser.add_argument("--exact", action="store_true",
                        help="Enable the Exact-SNN soft-latency readout head (surrogate warm-start, "
                             "temperature-annealed) as a regularizer on top of the surrogate path.")
    parser.add_argument("--exact-weight", type=float, default=1.0,
                        help="Weight of the exact latency loss relative to the CE loss.")
    parser.add_argument("--conv-stem", action="store_true",
                        help="Use a small convolutional stem so the model takes raw (3,32,32) "
                             "images instead of patchified 48-dim tokens (breaks the FC-only "
                             "~10%% CIFAR floor).")
    parser.add_argument("--conv-kernels", type=int, default=32,
                        help="Channels for the conv stem's two conv layers.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable Automatic Mixed Precision (FP16 autocast + GradScaler) for faster GPU Tensor Core throughput")
    return parser



def make_transform(augment=False):
    t = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if augment:
        t = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] + t
    return transforms.Compose(t)


def model_input(images, conv_stem):
    """Raw images for the conv-stem model, patchified tokens otherwise."""
    return images if conv_stem else patchify(images)


def evaluate(model, loader, device, conv_stem=False):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(model_input(images, conv_stem))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def save_checkpoint(path, model, optimizer, epoch, best_acc, args):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "args": vars(args),
    }, path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = resolve_data_dir()
    print(f"Using data dir: {data_dir}")

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=make_transform(args.augment))
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=make_transform(augment=False))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = SpikingTransformer(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        seq_len=SEQ_LEN,
        num_classes=10,
        num_heads=NUM_HEADS,
        num_layers=args.layers,
        n_repeats=args.repeats,
        use_spikeskip=not args.no_spikeskip,
        expansion=args.expansion,
        temporal=args.temporal,
        spike_threshold=args.threshold,
        use_gsmc=args.gsmc,
        use_exact=args.exact,
        exact_weight=args.exact_weight,
        use_conv_stem=args.conv_stem,
        conv_kernels=args.conv_kernels,
    ).to(device)
    if args.gsmc and not args.temporal:
        raise SystemExit("--gsmc requires --temporal (GSMC consumes the Spire T-planes).")
    filt = args.conv_stem  # conv-stem model consumes raw images
    print(f"SpikingTransformer params: {sum(p.numel() for p in model.parameters()):,}  "
          f"FFN in_features={model.blocks[0].ffn.fc2.in_features}  lr={args.lr}  "
          f"augment={args.augment}  cosine={args.cosine}  temporal={args.temporal}  "
          f"threshold={args.threshold}  gsmc={args.gsmc}  exact={args.exact}  "
          f"conv_stem={args.conv_stem}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pt")
    final_path = os.path.join(ckpt_dir, "final.pt")

    best_acc = -1.0
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", -1.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best_acc={best_acc:.2f})")

    # Per-epoch FC2 input sparsity tracking (only on SpikeSkip layers).
    # FC2 input == spike_fn(fc1(x) - threshold) => silent exactly where fc1 <= threshold.
    # Hook fc1: the temporal path calls fc2.forward_multistep() directly, which
    # bypasses nn.Module forward-hooks on fc2.
    _fc2_sparsity = [0.0]
    _fc2_calls = [0]

    def _fc2_hook(mod, inp, out):
        _fc2_sparsity[0] += float(((out[0] - args.threshold) <= 0).float().mean().item())
        _fc2_calls[0] += 1

    _sparsity_hooks = []
    for _blk in model.blocks:
        if type(_blk.ffn.fc2).__name__ == "SparseLinear":
            _sparsity_hooks.append(_blk.ffn.fc1.register_forward_hook(_fc2_hook))

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    print("\n--- Starting CIFAR-10 Training on SpikeStack Lite ---")
    total_start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        _fc2_sparsity[0] = 0.0
        _fc2_calls[0] = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=args.amp):
                outputs = model(model_input(images, filt))
                loss = criterion(outputs, labels)
                if args.exact:
                    loss = loss + model.latency_loss(model._pooled_for_latency, labels)


            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()


            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}], Step [{i + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%")

        if args.cosine:
            scheduler.step()

        # Anneal the exact-SNN latency head's sharpness (soft -> sharp TTFS).
        if args.exact:
            model.anneal_exact((epoch - start_epoch + 1) / max(1, args.epochs))

        epoch_time = time.time() - start_time
        fc2_sparsity = (100.0 * _fc2_sparsity[0] / _fc2_calls[0]) if _fc2_calls[0] else float("nan")
        print(f"=== Epoch {epoch + 1} Summary: Avg Loss: {running_loss / len(train_loader):.4f}, "
              f"Train Acc: {100. * correct / total:.2f}%, Time: {epoch_time:.2f}s, "
              f"FC2 sparsity: {fc2_sparsity:.1f}% ===")

        test_acc = evaluate(model, test_loader, device, conv_stem=filt)
        print(f"*** Test Accuracy: {test_acc:.2f}% ***\n")

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(best_path, model, optimizer, epoch, best_acc, args)
            print(f"    [saved new best -> {best_path}]")

    save_checkpoint(final_path, model, optimizer, args.epochs - 1, best_acc, args)
    for _h in _sparsity_hooks:
        _h.remove()
    total_time = time.time() - total_start_time
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Best test acc: {best_acc:.2f}%  (ckpt: {best_path})")


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.tiny:
        args.epochs = 10
        args.repeats = 3
        args.temporal = True
        args.threshold = 2.0
        args.augment = True
        args.cosine = True
        args.no_spikeskip = True
        if args.exp_name == "lite":
            args.exp_name = "tiny"
    train(args)
