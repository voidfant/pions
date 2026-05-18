#!/usr/bin/env python3
"""Практика 8: GAT на Cora + сравнение с GCN и t-SNE визуализация."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GCNModel(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: float):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        emb = F.relu(self.conv1(x, edge_index))
        x = F.dropout(emb, p=self.dropout, training=self.training)
        out = self.conv2(x, edge_index)
        return out

    def embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv1(x, edge_index))


class GATModel(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        emb = F.elu(self.gat1(x, edge_index))
        x = F.dropout(emb, p=self.dropout, training=self.training)
        out = self.gat2(x, edge_index)
        return out

    def embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return F.elu(self.gat1(x, edge_index))


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def train_model(model, data, epochs: int, lr: float, wd: float, device: torch.device, tag: str) -> float:
    model.to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        if epoch % 40 == 0 or epoch == 1:
            test_acc = evaluate_model(model, data, device)
            print(f"[{tag}] epoch={epoch:03d} loss={loss.item():.4f} test_acc={test_acc:.4f}")

    return evaluate_model(model, data, device)


def evaluate_model(model, data, device: torch.device) -> float:
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    return accuracy(out[data.test_mask], data.y[data.test_mask])


def tsne_plot(embeddings: np.ndarray, labels: np.ndarray, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, cmap="tab10", alpha=0.85)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Практика 8: GAT + t-SNE")
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--heads", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--out-dir", type=str, default="outputs/pr8")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures
    except ModuleNotFoundError as exc:
        raise SystemExit("Установите зависимости: pip install torch_geometric") from exc

    dataset = Planetoid(root="./data/Cora", name="Cora", transform=NormalizeFeatures())
    data = dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Базовый GCN для сравнения.
    gcn = GCNModel(dataset.num_features, hidden=16, out_channels=dataset.num_classes, dropout=0.5)
    gcn_acc = train_model(gcn, data, args.epochs, lr=0.01, wd=5e-4, device=device, tag="GCN")

    # Эксперимент с разным количеством голов внимания в GAT.
    gat_results: Dict[int, Tuple[GATModel, float]] = {}
    for heads in args.heads:
        gat = GATModel(
            in_channels=dataset.num_features,
            hidden=args.hidden,
            out_channels=dataset.num_classes,
            heads=heads,
            dropout=args.dropout,
        )
        acc = train_model(gat, data, args.epochs, args.lr, args.weight_decay, device, tag=f"GAT(h={heads})")
        gat_results[heads] = (gat, acc)

    best_heads = max(gat_results, key=lambda h: gat_results[h][1])
    best_gat, best_acc = gat_results[best_heads]

    print("\n=== Сравнение GCN vs GAT ===")
    print(f"GCN test accuracy: {gcn_acc:.4f}")
    for heads in sorted(gat_results):
        print(f"GAT heads={heads} test accuracy: {gat_results[heads][1]:.4f}")
    print(f"Лучшая конфигурация GAT: heads={best_heads}, acc={best_acc:.4f}")

    # Визуализация эмбеддингов лучшей GAT-модели и GCN.
    best_gat.eval()
    gcn.eval()
    with torch.no_grad():
        gat_emb = best_gat.embeddings(data.x.to(device), data.edge_index.to(device)).cpu().numpy()
        gcn_emb = gcn.embeddings(data.x.to(device), data.edge_index.to(device)).cpu().numpy()
        labels = data.y.cpu().numpy()

    try:
        tsne_plot(gat_emb, labels, out_dir / f"gat_heads_{best_heads}_tsne.png", f"GAT embeddings (heads={best_heads})")
        tsne_plot(gcn_emb, labels, out_dir / "gcn_tsne.png", "GCN embeddings")
        print(f"t-SNE графики сохранены в: {out_dir.resolve()}")
    except ModuleNotFoundError as exc:
        print("Не удалось построить t-SNE: установите matplotlib и scikit-learn")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
