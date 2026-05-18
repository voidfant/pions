#!/usr/bin/env python3
"""Практика 7: GCN на Cora + сравнение с классическим baseline."""

from __future__ import annotations

import argparse
import random
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def aggregate_neighbors(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    mode: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    """Показывает явную агрегацию соседей (усреднение/сумма)."""
    src, dst = edge_index
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src])
    if mode == "mean":
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg = torch.clamp(deg, min=1.0).unsqueeze(1)
        out = out / deg
    return out


def train_gcn(model, data, epochs: int, lr: float, wd: float, device: torch.device) -> float:
    model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            train_acc = accuracy(out[data.train_mask], data.y[data.train_mask])
            test_acc = evaluate_gcn(model, data, device)
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

    return evaluate_gcn(model, data, device)


def evaluate_gcn(model, data, device: torch.device) -> float:
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    return accuracy(out[data.test_mask], data.y[data.test_mask])


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_logreg_baseline(data) -> float:
    from sklearn.linear_model import LogisticRegression

    x_train = data.x[data.train_mask].cpu().numpy()
    y_train = data.y[data.train_mask].cpu().numpy()
    x_test = data.x[data.test_mask].cpu().numpy()
    y_test = data.y[data.test_mask].cpu().numpy()

    clf = LogisticRegression(max_iter=3000, n_jobs=-1)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    return float(score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Практика 7: GCN на Cora")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    try:
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures
    except ModuleNotFoundError as exc:
        raise SystemExit("Установите зависимости: pip install torch_geometric") from exc

    dataset = Planetoid(root="./data/Cora", name="Cora", transform=NormalizeFeatures())
    data = dataset[0]

    # Демонстрация агрегации соседей.
    agg_mean = aggregate_neighbors(data.x, data.edge_index, mode="mean")
    agg_sum = aggregate_neighbors(data.x, data.edge_index, mode="sum")
    print("Проверка агрегации соседей:")
    print("mean aggregation tensor shape:", tuple(agg_mean.shape))
    print("sum aggregation tensor shape :", tuple(agg_sum.shape))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gcn = GCN(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden,
        out_channels=dataset.num_classes,
        dropout=args.dropout,
    )
    gcn_acc = train_gcn(gcn, data, args.epochs, args.lr, args.weight_decay, device)

    baseline_acc = train_logreg_baseline(data)

    print("\n=== Сравнение методов ===")
    print(f"GCN test accuracy            : {gcn_acc:.4f}")
    print(f"Logistic Regression baseline: {baseline_acc:.4f}")


if __name__ == "__main__":
    main()
