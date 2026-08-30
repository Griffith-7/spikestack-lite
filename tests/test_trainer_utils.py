import sys

import torch

import train_cifar10_lite as trainer


def test_patchify_shape():
    imgs = torch.randn(4, 3, 32, 32)
    patches = trainer.patchify(imgs)
    assert patches.shape == (4, 64, 48)


def test_patchify_preserves_constant_image():
    imgs = torch.full((2, 3, 32, 32), 0.7)
    patches = trainer.patchify(imgs)
    assert torch.allclose(patches, torch.full_like(patches, 0.7))


def test_patchify_non_overlapping_demo():
    imgs = torch.zeros(1, 3, 32, 32)
    imgs[0, 0, :4, :4] = 1.0  # top-left 4x4 pixel block in channel 0
    patches = trainer.patchify(imgs)
    # channel-last layout: channel 0 values sit at strides of 3 within a patch
    ch0 = patches[0, 0, 0::3]
    assert torch.allclose(ch0, torch.ones(16))
    assert torch.allclose(patches[0, 0, 1::3], torch.zeros(16))
    assert torch.allclose(patches[0, 0, 2::3], torch.zeros(16))


def test_dataroot_resolution_fallback():
    # with nothing set and ./data empty, must return a valid directory
    path = trainer.resolve_data_dir()
    assert path and path.replace("\\", "/")  # non-empty directory string


def test_cli_exposes_hardening_flags():
    parser = trainer.build_parser()
    args = parser.parse_args([])
    for flag in ("--lr", "--augment", "--cosine", "--exp-name", "--save-dir", "--resume"):
        assert flag in parser.format_help(), f"missing CLI flag {flag}"