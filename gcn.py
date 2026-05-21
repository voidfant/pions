#!/usr/bin/env python3
"""GCN group: практики 7 и 8 в одном файле."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


# =========================
# Практика 7
# =========================
class PR7GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: float):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def aggregate_neighbors(x: torch.Tensor, edge_index: torch.Tensor, mode: Literal["mean", "sum"] = "mean") -> torch.Tensor:
    src, dst = edge_index
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src])
    if mode == "mean":
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        out = out / torch.clamp(deg, min=1.0).unsqueeze(1)
    return out


def train_pr7_gcn(model, data, device, epochs: int, lr: float, wd: float) -> float:
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

        if epoch % 20 == 0 or epoch == 1:
            tr_acc = accuracy(out[data.train_mask], data.y[data.train_mask])
            te_acc = eval_pr7_gcn(model, data, device)
            print(f"[PR7] epoch={epoch:03d} loss={loss.item():.4f} train_acc={tr_acc:.4f} test_acc={te_acc:.4f}")

    return eval_pr7_gcn(model, data, device)


def eval_pr7_gcn(model, data, device) -> float:
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    return accuracy(out[data.test_mask], data.y[data.test_mask])


def train_logreg_baseline(data) -> float:
    from sklearn.linear_model import LogisticRegression

    x_train = data.x[data.train_mask].cpu().numpy()
    y_train = data.y[data.train_mask].cpu().numpy()
    x_test = data.x[data.test_mask].cpu().numpy()
    y_test = data.y[data.test_mask].cpu().numpy()

    clf = LogisticRegression(max_iter=3000, n_jobs=-1)
    clf.fit(x_train, y_train)
    return float(clf.score(x_test, y_test))


def run_pr7(dataset, data, device) -> float:
    print("\n========== Практика 7: Введение в графовые нейросети ==========")

    agg_mean = aggregate_neighbors(data.x, data.edge_index, "mean")
    agg_sum = aggregate_neighbors(data.x, data.edge_index, "sum")
    print(f"[PR7] aggregation mean shape={tuple(agg_mean.shape)}")
    print(f"[PR7] aggregation sum shape={tuple(agg_sum.shape)}")

    model = PR7GCN(dataset.num_features, hidden=32, out_channels=dataset.num_classes, dropout=0.5)
    gcn_acc = train_pr7_gcn(model, data, device, epochs=200, lr=0.01, wd=5e-4)
    baseline_acc = train_logreg_baseline(data)

    print(f"[PR7] GCN test_acc={gcn_acc:.4f}")
    print(f"[PR7] LogisticRegression test_acc={baseline_acc:.4f}")
    return gcn_acc


# =========================
# Практика 8
# =========================
class PR8GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: float):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        emb = F.relu(self.conv1(x, edge_index))
        x = F.dropout(emb, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

    def embeddings(self, x, edge_index):
        return F.relu(self.conv1(x, edge_index))


class PR8GAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, heads: int, dropout: float):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        emb = F.elu(self.gat1(x, edge_index))
        x = F.dropout(emb, p=self.dropout, training=self.training)
        return self.gat2(x, edge_index)

    def embeddings(self, x, edge_index):
        return F.elu(self.gat1(x, edge_index))


def train_cls_model(model, data, device, epochs: int, lr: float, wd: float, tag: str) -> float:
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
            te_acc = eval_cls_model(model, data, device)
            print(f"[{tag}] epoch={epoch:03d} loss={loss.item():.4f} test_acc={te_acc:.4f}")

    return eval_cls_model(model, data, device)


def eval_cls_model(model, data, device) -> float:
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


def run_pr8(dataset, data, device, out_dir: Path) -> None:
    print("\n========== Практика 8: GNN с вниманием (GAT) ==========")

    gcn = PR8GCN(dataset.num_features, hidden=16, out_channels=dataset.num_classes, dropout=0.5)
    gcn_acc = train_cls_model(gcn, data, device, epochs=240, lr=0.01, wd=5e-4, tag="PR8-GCN")

    heads_list = [1, 4, 8]
    gat_results: Dict[int, Tuple[PR8GAT, float]] = {}
    for heads in heads_list:
        gat = PR8GAT(dataset.num_features, hidden=8, out_channels=dataset.num_classes, heads=heads, dropout=0.6)
        acc = train_cls_model(gat, data, device, epochs=240, lr=0.005, wd=5e-4, tag=f"PR8-GAT(h={heads})")
        gat_results[heads] = (gat, acc)

    best_heads = max(gat_results, key=lambda h: gat_results[h][1])
    best_gat, best_acc = gat_results[best_heads]

    print(f"[PR8] GCN test_acc={gcn_acc:.4f}")
    for h in sorted(gat_results):
        print(f"[PR8] GAT heads={h} test_acc={gat_results[h][1]:.4f}")
    print(f"[PR8] Лучшая GAT конфигурация: heads={best_heads}, acc={best_acc:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    gcn.eval()
    best_gat.eval()
    with torch.no_grad():
        gcn_emb = gcn.embeddings(data.x.to(device), data.edge_index.to(device)).cpu().numpy()
        gat_emb = best_gat.embeddings(data.x.to(device), data.edge_index.to(device)).cpu().numpy()
        labels = data.y.cpu().numpy()

    tsne_plot(gcn_emb, labels, out_dir / "gcn_tsne.png", "GCN embeddings")
    tsne_plot(gat_emb, labels, out_dir / f"gat_heads_{best_heads}_tsne.png", f"GAT embeddings (heads={best_heads})")
    print(f"[PR8] t-SNE сохранены в {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GCN/GAT: практики 7,8")
    parser.add_argument("--out-dir", type=str, default="outputs/gcn")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    try:
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures
    except ModuleNotFoundError as exc:
        raise SystemExit("Установите torch_geometric: pip install torch_geometric") from exc

    dataset = Planetoid(root="./data/Cora", name="Cora", transform=NormalizeFeatures())
    data = dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_pr7(dataset, data, device)
    run_pr8(dataset, data, device, Path(args.out_dir))

    print("\nGCN: практики 7-8 выполнены последовательно.")


if __name__ == "__main__":
    main()
