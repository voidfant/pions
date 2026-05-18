#!/usr/bin/env python3
"""Практика 4: Базовый GAN на MNIST (MLP-архитектура)."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, img_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Практика 4: базовый GAN")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d-steps", type=int, default=2, help="Сколько шагов дискриминатора на 1 шаг генератора")
    parser.add_argument("--out-dir", type=str, default="outputs/pr4")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dim = 28 * 28
    G = Generator(latent_dim=args.latent_dim, img_dim=img_dim).to(device)
    D = Discriminator(img_dim=img_dim).to(device)

    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr)

    fixed_noise = torch.randn(64, args.latent_dim, device=device)

    for epoch in range(1, args.epochs + 1):
        loss_g_epoch = 0.0
        loss_d_epoch = 0.0

        for real_imgs, _ in loader:
            real_imgs = real_imgs.view(real_imgs.size(0), -1).to(device)
            bsz = real_imgs.size(0)

            real_labels = torch.ones(bsz, 1, device=device)
            fake_labels = torch.zeros(bsz, 1, device=device)

            # Дискриминатор тренируется чаще генератора.
            for _ in range(args.d_steps):
                z = torch.randn(bsz, args.latent_dim, device=device)
                fake_imgs = G(z).detach()

                d_real = D(real_imgs)
                d_fake = D(fake_imgs)
                loss_d = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)

                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

            # Генератор.
            z = torch.randn(bsz, args.latent_dim, device=device)
            gen_imgs = G(z)
            pred_fake = D(gen_imgs)
            loss_g = criterion(pred_fake, real_labels)

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            loss_g_epoch += loss_g.item()
            loss_d_epoch += loss_d.item()

        g_grad = grad_norm(G)
        d_grad = grad_norm(D)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"loss_D={loss_d_epoch / len(loader):.4f} | "
            f"loss_G={loss_g_epoch / len(loader):.4f} | "
            f"grad_norm_D={d_grad:.3f} | grad_norm_G={g_grad:.3f}"
        )

        with torch.no_grad():
            sample = G(fixed_noise).view(-1, 1, 28, 28)
            sample = (sample + 1) / 2
            save_image(sample, out_dir / f"epoch_{epoch:03d}.png", nrow=8)

    print(f"Сэмплы сохранены в: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
