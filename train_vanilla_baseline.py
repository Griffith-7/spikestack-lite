import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from . import train_cifar10_lite as ssl
except ImportError:
    import train_cifar10_lite as ssl

D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 1
FFN_EXPANSION = 4


class VanillaTransformer(nn.Module):
    """Same-geometry analog baseline for the spiking model.

    Single block, d_model=128, 4 heads, FFN 128->512->128, mean-over-sequence
    readout -- identical topology/budget to the SpikeStack Lite tiny profile,
    but with dense softmax attention and GELU instead of Hebbian spikes.
    """

    def __init__(self, input_dim=48, seq_len=64, d_model=D_MODEL,
                 num_classes=10, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                 expansion=FFN_EXPANSION):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * expansion,
            dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True, bias=False,
        )
        self.blocks = nn.ModuleList([layer for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        for blk in self.blocks:
            x = blk(x)
        return self.classifier(x.mean(dim=1))


def build_parser():
    p = argparse.ArgumentParser(description="Train vanilla analog transformer baseline (same-geometry)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=ssl.DEFAULT_BATCH_SIZE)
    p.add_argument("--workers", type=int, default=0, help="DataLoader num_workers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", type=int, default=NUM_LAYERS)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--cosine", action="store_true")
    p.add_argument("--exp-name", type=str, default="vanilla")
    p.add_argument("--save-dir", type=str, default="checkpoints")
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = ssl.resolve_data_dir()
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False,
                                     transform=ssl.make_transform(args.augment))
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False,
                                    transform=ssl.make_transform(augment=False))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = VanillaTransformer(num_layers=args.layers).to(device)
    print(f"VanillaTransformer params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pt")

    best_acc = -1.0
    print("\n--- Starting CIFAR-10 Training on Vanilla Transformer ---")
    total_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss = correct = total = 0
        start_time = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(ssl.patchify(images))
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        if args.cosine:
            scheduler.step()
        epoch_time = time.time() - start_time

        test_acc = ssl.evaluate(model, test_loader, device)
        print(f"=== Epoch {epoch + 1} Summary: Avg Loss: {running_loss / len(train_loader):.4f}, "
              f"Train Acc: {100. * correct / total:.2f}%, Time: {epoch_time:.2f}s ===")
        print(f"*** Test Accuracy: {test_acc:.2f}% ***")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model_state": model.state_dict(), "best_acc": best_acc}, best_path)

    total_time = time.time() - total_start
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Best test acc: {best_acc:.2f}%  (ckpt: {best_path})")
    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
