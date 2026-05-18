#!/usr/bin/env python3
"""Практика 6: DCGAN + опциональный Self-Attention."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h * w).transpose(1, 2)
        k = self.key(x).view(b, -1, h * w)
        v = self.value(x).view(b, -1, h * w)

        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        return self.gamma * out + x


class DCGenerator(nn.Module):
    def __init__(self, latent_dim: int, channels: int, use_attention: bool):
        super().__init__()
        blocks: List[nn.Module] = [
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        ]
        if use_attention:
            blocks.append(SelfAttention2d(128))
        blocks.extend(
            [
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DCDiscriminator(nn.Module):
    def __init__(self, channels: int, use_attention: bool, dropout: float = 0.2):
        super().__init__()
        blocks: List[nn.Module] = [
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        ]
        if use_attention:
            blocks.append(SelfAttention2d(128))
        blocks.extend(
            [
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(dropout),
            ]
        )
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.Conv2d(256, 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.classifier(self.features(x))
        return out.view(-1, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


def compute_stats(feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = feats.mean(dim=0)
    centered = feats - mu
    cov = centered.t().mm(centered) / max(feats.size(0) - 1, 1)
    return mu, cov


def sqrtm_psd(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh((mat + mat.t()) * 0.5)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.t()


def fid_score(real_feats: torch.Tensor, fake_feats: torch.Tensor) -> float:
    mu_r, cov_r = compute_stats(real_feats)
    mu_f, cov_f = compute_stats(fake_feats)
    diff = mu_r - mu_f

    cov_r_sqrt = sqrtm_psd(cov_r)
    mid = cov_r_sqrt @ cov_f @ cov_r_sqrt
    covmean = sqrtm_psd(mid)
    fid = diff.dot(diff) + torch.trace(cov_r + cov_f - 2 * covmean)
    return float(fid.item())


def get_dataset(name: str):
    if name == "mnist":
        transform = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        return ds, 1
    if name == "cifar10":
        transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        return ds, 3
    raise ValueError("dataset должен быть mnist или cifar10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Практика 6: DCGAN")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--out-dir", type=str, default="outputs/pr6")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_dcgan(args: argparse.Namespace, use_attention: bool, loader: DataLoader, channels: int, device: torch.device, out_dir: Path) -> float:
    G = DCGenerator(args.latent_dim, channels, use_attention).to(device)
    D = DCDiscriminator(channels, use_attention).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    criterion = nn.BCELoss()

    fixed_z = torch.randn(64, args.latent_dim, 1, 1, device=device)

    tag = "attn" if use_attention else "plain"
    for epoch in range(1, args.epochs + 1):
        g_sum = 0.0
        d_sum = 0.0

        for real, _ in loader:
            real = real.to(device)
            bsz = real.size(0)
            ones = torch.ones(bsz, 1, device=device)
            zeros = torch.zeros(bsz, 1, device=device)

            # Train D
            z = torch.randn(bsz, args.latent_dim, 1, 1, device=device)
            fake = G(z).detach()

            loss_d = criterion(D(real), ones) + criterion(D(fake), zeros)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # Train G
            z = torch.randn(bsz, args.latent_dim, 1, 1, device=device)
            gen = G(z)
            loss_g = criterion(D(gen), ones)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            g_sum += loss_g.item()
            d_sum += loss_d.item()

        print(f"[{tag}] epoch={epoch} D={d_sum / len(loader):.4f} G={g_sum / len(loader):.4f}")

        with torch.no_grad():
            samples = (G(fixed_z) + 1) / 2
            save_image(samples, out_dir / f"{tag}_epoch_{epoch:03d}.png", nrow=8)

    # FID-like на признаках дискриминатора.
    D.eval()
    real_feats, fake_feats = [], []
    with torch.no_grad():
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            z = torch.randn(real.size(0), args.latent_dim, 1, 1, device=device)
            fake = G(z)
            real_feats.append(D.extract_features(real).cpu())
            fake_feats.append(D.extract_features(fake).cpu())
            if i >= 20:
                break

    fid = fid_score(torch.cat(real_feats, dim=0), torch.cat(fake_feats, dim=0))
    print(f"[{tag}] FID(feature-space)={fid:.4f}")
    return fid


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, channels = get_dataset(args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fid_plain = train_dcgan(args, use_attention=False, loader=loader, channels=channels, device=device, out_dir=out_dir)
    fid_attn = train_dcgan(args, use_attention=True, loader=loader, channels=channels, device=device, out_dir=out_dir)

    print("\n=== Сравнение архитектур ===")
    print(f"DCGAN без attention: FID={fid_plain:.4f}")
    print(f"DCGAN + Self-Attn : FID={fid_attn:.4f}")


if __name__ == "__main__":
    main()
