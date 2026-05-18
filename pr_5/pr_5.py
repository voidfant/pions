#!/usr/bin/env python3
"""Практика 5: Улучшенный GAN (BatchNorm + Dropout) и FID-оценка."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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


class ImprovedGenerator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ImprovedDiscriminator(nn.Module):
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256 * 4 * 4, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


def compute_stats(feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = feats.mean(dim=0)
    centered = feats - mu
    cov = centered.t().mm(centered) / max(feats.size(0) - 1, 1)
    return mu, cov


def sqrtm_psd(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Для симметричных PSD-матриц достаточно eigendecomposition.
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.t()


def fid_score(real_feats: torch.Tensor, fake_feats: torch.Tensor) -> float:
    mu_r, cov_r = compute_stats(real_feats)
    mu_f, cov_f = compute_stats(fake_feats)

    diff = mu_r - mu_f
    diff_term = diff.dot(diff)

    cov_r_sqrt = sqrtm_psd(cov_r)
    middle = cov_r_sqrt @ cov_f @ cov_r_sqrt
    covmean = sqrtm_psd((middle + middle.t()) * 0.5)

    trace_term = torch.trace(cov_r + cov_f - 2.0 * covmean)
    return float((diff_term + trace_term).item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Практика 5: улучшение GAN")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default="outputs/pr5")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_experiment(
    config: Dict[str, object],
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path,
) -> float:
    G = ImprovedGenerator(args.latent_dim).to(device)
    D = ImprovedDiscriminator(dropout_p=float(config["dropout"])) .to(device)

    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(
        G.parameters(),
        lr=float(config["lr"]),
        betas=(float(config["beta1"]), float(config["beta2"])),
    )
    opt_d = torch.optim.Adam(
        D.parameters(),
        lr=float(config["lr"]),
        betas=(float(config["beta1"]), float(config["beta2"])),
    )

    fixed_z = torch.randn(64, args.latent_dim, 1, 1, device=device)

    for epoch in range(1, args.epochs + 1):
        g_loss_sum = 0.0
        d_loss_sum = 0.0

        for real, _ in loader:
            real = real.to(device)
            bsz = real.size(0)
            ones = torch.ones(bsz, 1, device=device)
            zeros = torch.zeros(bsz, 1, device=device)

            # Дискриминатор.
            z = torch.randn(bsz, args.latent_dim, 1, 1, device=device)
            fake = G(z).detach()

            d_real = D(real)
            d_fake = D(fake)
            loss_d = criterion(d_real, ones) + criterion(d_fake, zeros)

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # Генератор.
            z = torch.randn(bsz, args.latent_dim, 1, 1, device=device)
            gen = G(z)
            pred = D(gen)
            loss_g = criterion(pred, ones)

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            g_loss_sum += loss_g.item()
            d_loss_sum += loss_d.item()

        print(
            f"cfg={config} | epoch={epoch} | D={d_loss_sum / len(loader):.4f} | G={g_loss_sum / len(loader):.4f}"
        )

        with torch.no_grad():
            samples = (G(fixed_z) + 1) / 2
            save_image(samples, out_dir / f"cfg_{config['name']}_epoch_{epoch:03d}.png", nrow=8)

    # Считаем FID-подобную метрику по признакам дискриминатора.
    D.eval()
    real_feats_list: List[torch.Tensor] = []
    fake_feats_list: List[torch.Tensor] = []
    with torch.no_grad():
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            z = torch.randn(real.size(0), args.latent_dim, 1, 1, device=device)
            fake = G(z)

            real_feats_list.append(D.extract_features(real).cpu())
            fake_feats_list.append(D.extract_features(fake).cpu())
            if i >= 20:
                break

    real_feats = torch.cat(real_feats_list, dim=0)
    fake_feats = torch.cat(fake_feats_list, dim=0)
    fid = fid_score(real_feats, fake_feats)
    print(f"cfg={config} | FID (feature-space)={fid:.4f}")
    return fid


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        {"name": "a", "lr": 2e-4, "beta1": 0.5, "beta2": 0.999, "dropout": 0.3},
        {"name": "b", "lr": 1e-4, "beta1": 0.4, "beta2": 0.95, "dropout": 0.4},
    ]

    results = []
    for cfg in configs:
        fid = run_experiment(cfg, loader, device, args, out_dir)
        results.append((cfg, fid))

    results.sort(key=lambda x: x[1])
    print("\n=== Сравнение конфигураций по FID ===")
    for cfg, fid in results:
        print(f"{cfg} -> FID={fid:.4f}")


if __name__ == "__main__":
    main()
