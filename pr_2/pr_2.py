#!/usr/bin/env python3
"""Практика 2: Реализация механизма внимания и сравнение с RNN."""

from __future__ import annotations

import argparse
import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultiHeadAttentionFromScratch(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model должен делиться на num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous().view(batch, seq_len, heads * head_dim)
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = self._combine_heads(context)
        out = self.out_proj(context)
        return out


class AttentionClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attn = MultiHeadAttentionFromScratch(d_model=d_model, num_heads=num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        h = self.norm(h + self.attn(h))
        pooled = h.mean(dim=1)
        return self.classifier(pooled)


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        _, hidden = self.rnn(h)
        return self.classifier(hidden[-1])


def generate_context_dataset(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Задача требует учитывать весь контекст: класс зависит от удалённых позиций."""
    rng = np.random.default_rng(seed)
    x = rng.integers(low=1, high=vocab_size, size=(n_samples, seq_len), dtype=np.int64)

    # Бинарная метка: XOR двух дальних зависимостей.
    y = ((x[:, 0] > x[:, seq_len // 2]) ^ (x[:, 2] > x[:, -1])).astype(np.int64)

    return torch.from_numpy(x), torch.from_numpy(y)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        batches = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(logits.detach(), yb)
            batches += 1

        model.eval()
        test_acc = 0.0
        test_batches = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                test_acc += accuracy(logits, yb)
                test_batches += 1

        print(
            f"Epoch {epoch}: train_loss={train_loss / batches:.4f}, "
            f"train_acc={train_acc / batches:.4f}, test_acc={test_acc / test_batches:.4f}"
        )

    return test_acc / test_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Практика 2: механизм внимания")
    parser.add_argument("--n-samples", type=int, default=9000)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--vocab-size", type=int, default=80)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    # Проверка формы выхода собственного слоя внимания.
    x_rand = torch.randn(2, args.seq_len, args.d_model)
    attn = MultiHeadAttentionFromScratch(d_model=args.d_model, num_heads=args.num_heads)
    y_rand = attn(x_rand)
    print("Проверка формы MultiHeadAttention:", tuple(x_rand.shape), "->", tuple(y_rand.shape))

    x, y = generate_context_dataset(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )

    split = int(0.8 * len(x))
    train_ds = TensorDataset(x[:split], y[:split])
    test_ds = TensorDataset(x[split:], y[split:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== Attention-модель ===")
    attn_model = AttentionClassifier(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_classes=2,
    )
    attn_acc = train_model(attn_model, train_loader, test_loader, device, args.epochs, args.lr)

    print("\n=== RNN baseline ===")
    rnn_model = RNNClassifier(vocab_size=args.vocab_size, d_model=args.d_model, num_classes=2)
    rnn_acc = train_model(rnn_model, train_loader, test_loader, device, args.epochs, args.lr)

    print("\n=== Сравнение ===")
    print(f"Attention test_acc: {attn_acc:.4f}")
    print(f"RNN       test_acc: {rnn_acc:.4f}")


if __name__ == "__main__":
    main()
