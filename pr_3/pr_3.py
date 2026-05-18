#!/usr/bin/env python3
"""Практика 3: Полная реализация мини-трансформера (encoder + decoder)."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
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
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, s, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, h * d)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v)
        return self.out_proj(self._combine_heads(context))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.ffn(x)
        x = self.norm2(x + self.dropout(ff))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn))

        cross = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(cross))

        ff = self.ffn(x)
        x = self.norm3(x + self.dropout(ff))
        return x


class MiniTransformer(nn.Module):
    """Encoder похож на BERT-часть, decoder похож на GPT-часть."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.pos_dec = PositionalEncoding(d_model, max_len=max_len)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        self.norm_enc = nn.LayerNorm(d_model)
        self.norm_dec = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, tgt_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, src: torch.Tensor, tgt_inp: torch.Tensor) -> torch.Tensor:
        src_emb = self.pos_enc(self.token_emb(src))
        tgt_emb = self.pos_dec(self.token_emb(tgt_inp))

        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory)
        memory = self.norm_enc(memory)

        tgt_mask = self.generate_causal_mask(tgt_inp.size(1), tgt_inp.device)
        out = tgt_emb
        for layer in self.decoder_layers:
            out = layer(out, memory, tgt_mask=tgt_mask)
        out = self.norm_dec(out)

        return self.out_proj(out)


def create_toy_seq2seq_data(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    bos_id: int,
    eos_id: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    core = rng.integers(3, vocab_size - 1, size=(n_samples, seq_len), dtype=np.int64)
    src = np.concatenate([
        np.full((n_samples, 1), bos_id, dtype=np.int64),
        core,
        np.full((n_samples, 1), eos_id, dtype=np.int64),
    ], axis=1)

    # Цель: вывести перевёрнутую последовательность (классическая toy-задача).
    rev = np.flip(core, axis=1)
    tgt = np.concatenate([
        np.full((n_samples, 1), bos_id, dtype=np.int64),
        rev,
        np.full((n_samples, 1), eos_id, dtype=np.int64),
    ], axis=1)
    return torch.from_numpy(src), torch.from_numpy(tgt)


def train_eval_one_config(
    config: Dict[str, int],
    src_train: torch.Tensor,
    tgt_train: torch.Tensor,
    src_test: torch.Tensor,
    tgt_test: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model = MiniTransformer(
        vocab_size=args.vocab_size,
        d_model=config["d_model"],
        num_heads=config["heads"],
        num_layers=config["layers"],
        ff_dim=config["ff_dim"],
        max_len=args.seq_len + 2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(src_train, tgt_train),
        batch_size=args.batch_size,
        shuffle=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for src_b, tgt_b in train_loader:
            src_b = src_b.to(device)
            tgt_b = tgt_b.to(device)
            tgt_inp = tgt_b[:, :-1]
            tgt_out = tgt_b[:, 1:]

            optimizer.zero_grad()
            logits = model(src_b, tgt_inp)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Config={config} | epoch={epoch} | train_loss={total_loss / len(train_loader):.4f}"
        )

    model.eval()
    with torch.no_grad():
        src_t = src_test.to(device)
        tgt_t = tgt_test.to(device)
        logits = model(src_t, tgt_t[:, :-1])
        preds = logits.argmax(dim=-1)
        target = tgt_t[:, 1:]
        token_acc = (preds == target).float().mean().item()

        # Показываем распределение вероятностей для одного примера на первом предсказании.
        probs = torch.softmax(logits[0, 0], dim=-1)
        top_probs, top_ids = torch.topk(probs, k=5)
        top_pairs = ", ".join([f"id={i.item()} p={p.item():.3f}" for i, p in zip(top_ids, top_probs)])
        print(f"Top-5 вероятностей (пример 1, первый токен): {top_pairs}")

    return token_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Практика 3: полный мини-трансформер")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    bos_id, eos_id = 1, 2
    src_train, tgt_train = create_toy_seq2seq_data(
        n_samples=args.n_train,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        bos_id=bos_id,
        eos_id=eos_id,
        seed=args.seed,
    )
    src_test, tgt_test = create_toy_seq2seq_data(
        n_samples=args.n_test,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        bos_id=bos_id,
        eos_id=eos_id,
        seed=args.seed + 1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs: List[Dict[str, int]] = [
        {"layers": 2, "d_model": 64, "heads": 4, "ff_dim": 128},
        {"layers": 3, "d_model": 96, "heads": 6, "ff_dim": 192},
    ]

    print("=== Эксперимент с гиперпараметрами ===")
    results: List[Tuple[Dict[str, int], float]] = []
    for cfg in configs:
        acc = train_eval_one_config(cfg, src_train, tgt_train, src_test, tgt_test, args, device)
        results.append((cfg, acc))
        print(f"Config={cfg} | token_accuracy={acc:.4f}\n")

    results.sort(key=lambda x: x[1], reverse=True)
    print("=== Рейтинг конфигураций ===")
    for cfg, acc in results:
        print(f"{cfg} -> token_accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
