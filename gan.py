#!/usr/bin/env python3
"""GAN group: практики 4, 5, 6 в одном файле."""

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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sqrtm_psd(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh((mat + mat.t()) * 0.5)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.t()


def feats_stats(feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = feats.mean(dim=0)
    c = feats - mu
    cov = c.t().mm(c) / max(feats.size(0) - 1, 1)
    return mu, cov


def fid_score(real_feats: torch.Tensor, fake_feats: torch.Tensor) -> float:
    mu_r, cov_r = feats_stats(real_feats)
    mu_f, cov_f = feats_stats(fake_feats)
    diff = mu_r - mu_f
    cov_r_sqrt = sqrtm_psd(cov_r)
    mid = cov_r_sqrt @ cov_f @ cov_r_sqrt
    covmean = sqrtm_psd(mid)
    fid = diff.dot(diff) + torch.trace(cov_r + cov_f - 2 * covmean)
    return float(fid.item())


# =========================
# Практика 4
# =========================
class PR4Generator(nn.Module):
    def __init__(self, z_dim: int, img_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, img_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class PR4Discriminator(nn.Module):
    def __init__(self, img_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def run_pr4(base_out: Path) -> None:
    print("\n========== Практика 4: Введение в GAN ==========")
    seed_everything(42)

    out_dir = base_out / "pr4"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size, epochs, z_dim, lr, d_steps = 128, 15, 100, 2e-4, 2

    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_dim = 28 * 28
    G = PR4Generator(z_dim, img_dim).to(device)
    D = PR4Discriminator(img_dim).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=lr)
    opt_d = torch.optim.Adam(D.parameters(), lr=lr)
    crit = nn.BCELoss()

    fixed_z = torch.randn(64, z_dim, device=device)

    for epoch in range(1, epochs + 1):
        g_sum = d_sum = 0.0
        for real, _ in loader:
            real = real.view(real.size(0), -1).to(device)
            bsz = real.size(0)
            ones = torch.ones(bsz, 1, device=device)
            zeros = torch.zeros(bsz, 1, device=device)

            for _ in range(d_steps):
                z = torch.randn(bsz, z_dim, device=device)
                fake = G(z).detach()
                d_loss = crit(D(real), ones) + crit(D(fake), zeros)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            z = torch.randn(bsz, z_dim, device=device)
            gen = G(z)
            g_loss = crit(D(gen), ones)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            g_sum += g_loss.item()
            d_sum += d_loss.item()

        print(
            f"[PR4] epoch={epoch:02d}/{epochs} D={d_sum / len(loader):.4f} G={g_sum / len(loader):.4f} "
            f"gradD={grad_norm(D):.3f} gradG={grad_norm(G):.3f}"
        )

        with torch.no_grad():
            sample = (G(fixed_z).view(-1, 1, 28, 28) + 1) / 2
            save_image(sample, out_dir / f"epoch_{epoch:03d}.png", nrow=8)


# =========================
# Практика 5
# =========================
class PR5Generator(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 7, 1, 0, bias=False),
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

    def forward(self, z):
        return self.net(z)


class PR5Discriminator(nn.Module):
    def __init__(self, drop: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256 * 4 * 4, 1), nn.Sigmoid())

    def forward(self, x):
        return self.head(self.features(x))

    def extract_features(self, x):
        return self.features(x).flatten(1)


def run_pr5(base_out: Path) -> None:
    print("\n========== Практика 5: Улучшение GAN ==========")
    seed_everything(42)

    out_dir = base_out / "pr5"
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs, batch_size, z_dim = 10, 128, 64

    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [
        {"name": "a", "lr": 2e-4, "beta1": 0.5, "beta2": 0.999, "drop": 0.3},
        {"name": "b", "lr": 1e-4, "beta1": 0.4, "beta2": 0.95, "drop": 0.4},
    ]

    ranking = []
    for cfg in configs:
        G = PR5Generator(z_dim).to(device)
        D = PR5Discriminator(cfg["drop"]).to(device)
        opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
        opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
        crit = nn.BCELoss()
        fixed_z = torch.randn(64, z_dim, 1, 1, device=device)

        for epoch in range(1, epochs + 1):
            g_sum = d_sum = 0.0
            for real, _ in loader:
                real = real.to(device)
                bsz = real.size(0)
                ones = torch.ones(bsz, 1, device=device)
                zeros = torch.zeros(bsz, 1, device=device)

                z = torch.randn(bsz, z_dim, 1, 1, device=device)
                fake = G(z).detach()
                d_loss = crit(D(real), ones) + crit(D(fake), zeros)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                z = torch.randn(bsz, z_dim, 1, 1, device=device)
                gen = G(z)
                g_loss = crit(D(gen), ones)
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

                g_sum += g_loss.item()
                d_sum += d_loss.item()

            print(f"[PR5] cfg={cfg['name']} epoch={epoch} D={d_sum / len(loader):.4f} G={g_sum / len(loader):.4f}")
            with torch.no_grad():
                sample = (G(fixed_z) + 1) / 2
                save_image(sample, out_dir / f"cfg_{cfg['name']}_epoch_{epoch:03d}.png", nrow=8)

        real_feats, fake_feats = [], []
        D.eval()
        with torch.no_grad():
            for i, (real, _) in enumerate(loader):
                real = real.to(device)
                z = torch.randn(real.size(0), z_dim, 1, 1, device=device)
                fake = G(z)
                real_feats.append(D.extract_features(real).cpu())
                fake_feats.append(D.extract_features(fake).cpu())
                if i >= 20:
                    break

        fid = fid_score(torch.cat(real_feats, dim=0), torch.cat(fake_feats, dim=0))
        print(f"[PR5] cfg={cfg['name']} FID(feature-space)={fid:.4f}")
        ranking.append((cfg, fid))

    ranking.sort(key=lambda x: x[1])
    print("[PR5] Рейтинг конфигураций по FID:")
    for cfg, fid in ranking:
        print(f"- {cfg}: FID={fid:.4f}")


# =========================
# Практика 6
# =========================
class PR6SelfAttention2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.q = nn.Conv2d(channels, channels // 8, 1)
        self.k = nn.Conv2d(channels, channels // 8, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).view(b, -1, h * w).transpose(1, 2)
        k = self.k(x).view(b, -1, h * w)
        v = self.v(x).view(b, -1, h * w)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        return self.gamma * out + x


class PR6Generator(nn.Module):
    def __init__(self, z_dim: int, channels: int, use_attn: bool):
        super().__init__()
        layers: List[nn.Module] = [
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        ]
        if use_attn:
            layers.append(PR6SelfAttention2d(128))
        layers.extend(
            [
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class PR6Discriminator(nn.Module):
    def __init__(self, channels: int, use_attn: bool, drop: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop),
        ]
        if use_attn:
            layers.append(PR6SelfAttention2d(128))
        layers.extend(
            [
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(drop),
            ]
        )
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Conv2d(256, 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, x):
        return self.head(self.features(x)).view(-1, 1)

    def extract_features(self, x):
        return self.features(x).flatten(1)


def pr6_dataset(name: str):
    if name == "mnist":
        tr = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        return torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tr), 1
    tr = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tr), 3


def pr6_train_variant(loader, channels, use_attn: bool, out_dir: Path, device: torch.device) -> float:
    z_dim, epochs, lr, beta1, beta2 = 100, 12, 2e-4, 0.5, 0.999

    G = PR6Generator(z_dim, channels, use_attn).to(device)
    D = PR6Discriminator(channels, use_attn).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
    crit = nn.BCELoss()
    fixed_z = torch.randn(64, z_dim, 1, 1, device=device)
    tag = "attn" if use_attn else "plain"

    for epoch in range(1, epochs + 1):
        g_sum = d_sum = 0.0
        for real, _ in loader:
            real = real.to(device)
            bsz = real.size(0)
            ones = torch.ones(bsz, 1, device=device)
            zeros = torch.zeros(bsz, 1, device=device)

            z = torch.randn(bsz, z_dim, 1, 1, device=device)
            fake = G(z).detach()
            d_loss = crit(D(real), ones) + crit(D(fake), zeros)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            z = torch.randn(bsz, z_dim, 1, 1, device=device)
            gen = G(z)
            g_loss = crit(D(gen), ones)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            g_sum += g_loss.item()
            d_sum += d_loss.item()

        print(f"[PR6-{tag}] epoch={epoch} D={d_sum / len(loader):.4f} G={g_sum / len(loader):.4f}")
        with torch.no_grad():
            sample = (G(fixed_z) + 1) / 2
            save_image(sample, out_dir / f"{tag}_epoch_{epoch:03d}.png", nrow=8)

    real_feats, fake_feats = [], []
    D.eval()
    with torch.no_grad():
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            z = torch.randn(real.size(0), z_dim, 1, 1, device=device)
            fake = G(z)
            real_feats.append(D.extract_features(real).cpu())
            fake_feats.append(D.extract_features(fake).cpu())
            if i >= 20:
                break

    return fid_score(torch.cat(real_feats, dim=0), torch.cat(fake_feats, dim=0))


def run_pr6(base_out: Path) -> None:
    print("\n========== Практика 6: DCGAN + Self-Attention ==========")
    seed_everything(42)

    out_dir = base_out / "pr6"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, channels = pr6_dataset("mnist")
    loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fid_plain = pr6_train_variant(loader, channels, False, out_dir, device)
    fid_attn = pr6_train_variant(loader, channels, True, out_dir, device)

    print(f"[PR6] DCGAN без attention FID={fid_plain:.4f}")
    print(f"[PR6] DCGAN + attention FID={fid_attn:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GAN: практики 4,5,6")
    parser.add_argument("--out-dir", type=str, default="outputs/gan")
    args = parser.parse_args()

    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    run_pr4(base_out)
    run_pr5(base_out)
    run_pr6(base_out)

    print("\nGAN: все практики (4-6) выполнены последовательно.")


if __name__ == "__main__":
    main()
