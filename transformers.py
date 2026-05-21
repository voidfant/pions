#!/usr/bin/env python3
"""Transformers group: практики 1, 2, 3 в одном файле."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Практика 1
# =========================
@dataclass
class PR1Result:
    model_name: str
    test_loss: float
    test_accuracy: float


def pr1_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--models", nargs="+", default=["distilbert-base-uncased", "bert-base-uncased"])
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args([])


def pr1_collate_fn(tokenizer, max_length: int):
    def collate(batch):
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc["labels"] = labels
        return enc

    return collate


def pr1_eval(model, loader, device, criterion) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch["labels"]
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = criterion(out.logits, y)
            pred = out.logits.argmax(dim=-1)
            total_loss += loss.item() * y.size(0)
            total_correct += (pred == y).sum().item()
            total_n += y.size(0)
    return total_loss / total_n, total_correct / total_n


def run_pr1() -> None:
    print("\n========== Практика 1: Введение в трансформеры ==========")
    args = pr1_args()
    seed_everything(args.seed)

    try:
        from datasets import load_dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ModuleNotFoundError as exc:
        print("[PR1] Пропуск: установите зависимости `pip install transformers datasets`.")
        return

    dataset = load_dataset("imdb")
    train_ds = dataset["train"].shuffle(seed=args.seed).select(range(args.train_size))
    test_ds = dataset["test"].shuffle(seed=args.seed).select(range(args.test_size))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: List[PR1Result] = []

    for model_name in args.models:
        print(f"\n[PR1] Модель: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=pr1_collate_fn(tokenizer, args.max_length),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=pr1_collate_fn(tokenizer, args.max_length),
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            model.train()
            run_loss, run_correct, run_n = 0.0, 0, 0

            for step, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(device) for k, v in batch.items()}
                y = batch["labels"]

                optimizer.zero_grad()
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = criterion(out.logits, y)
                loss.backward()
                optimizer.step()

                pred = out.logits.argmax(dim=-1)
                run_loss += loss.item() * y.size(0)
                run_correct += (pred == y).sum().item()
                run_n += y.size(0)

                if step % 50 == 0:
                    print(
                        f"[PR1] epoch={epoch} step={step}/{len(train_loader)} "
                        f"loss={run_loss / run_n:.4f} acc={run_correct / run_n:.4f}"
                    )

            tr_loss = run_loss / run_n
            tr_acc = run_correct / run_n
            te_loss, te_acc = pr1_eval(model, test_loader, device, criterion)
            print(
                f"[PR1] epoch={epoch} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                f"test_loss={te_loss:.4f} test_acc={te_acc:.4f}"
            )

        results.append(PR1Result(model_name, te_loss, te_acc))

    results.sort(key=lambda x: x.test_accuracy, reverse=True)
    print("\n[PR1] Сравнение моделей:")
    for r in results:
        print(f"- {r.model_name}: loss={r.test_loss:.4f}, acc={r.test_accuracy:.4f}")


# =========================
# Практика 2
# =========================
class PR2MultiHeadAttention(nn.Module):
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

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, h, s, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, h * d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        w = torch.softmax(scores, dim=-1)
        w = self.dropout(w)
        ctx = w @ v
        return self.out_proj(self._merge(ctx))


class PR2AttentionClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, heads: int, num_classes: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.attn = PR2MultiHeadAttention(d_model, heads)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h = self.norm(h + self.attn(h))
        pooled = h.mean(dim=1)
        return self.fc(pooled)


class PR2RNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_classes: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        _, hidden = self.rnn(h)
        return self.fc(hidden[-1])


def pr2_acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def pr2_generate_data(n_samples: int, seq_len: int, vocab_size: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.integers(1, vocab_size, size=(n_samples, seq_len), dtype=np.int64)
    y = ((x[:, 0] > x[:, seq_len // 2]) ^ (x[:, 2] > x[:, -1])).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def pr2_train(model, train_loader, test_loader, device, epochs: int, lr: float) -> float:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = tr_acc = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            tr_acc += pr2_acc(logits.detach(), yb)
            n += 1

        model.eval()
        te_acc, m = 0.0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                te_acc += pr2_acc(model(xb), yb)
                m += 1

        print(f"[PR2] epoch={epoch} train_loss={tr_loss / n:.4f} train_acc={tr_acc / n:.4f} test_acc={te_acc / m:.4f}")

    return te_acc / m


def run_pr2() -> None:
    print("\n========== Практика 2: Реализация механизма внимания ==========")
    seed_everything(42)

    n_samples, seq_len, vocab_size = 9000, 24, 80
    d_model, heads, batch, epochs, lr = 96, 4, 64, 8, 3e-3

    x_rand = torch.randn(2, seq_len, d_model)
    y_rand = PR2MultiHeadAttention(d_model, heads)(x_rand)
    print(f"[PR2] Проверка формы attention: {tuple(x_rand.shape)} -> {tuple(y_rand.shape)}")

    x, y = pr2_generate_data(n_samples, seq_len, vocab_size, seed=42)
    split = int(0.8 * len(x))
    train_loader = DataLoader(TensorDataset(x[:split], y[:split]), batch_size=batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(x[split:], y[split:]), batch_size=batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[PR2] Attention-модель")
    attn_model = PR2AttentionClassifier(vocab_size, d_model, heads, 2)
    attn_acc = pr2_train(attn_model, train_loader, test_loader, device, epochs, lr)

    print("[PR2] RNN baseline")
    rnn_model = PR2RNNClassifier(vocab_size, d_model, 2)
    rnn_acc = pr2_train(rnn_model, train_loader, test_loader, device, epochs, lr)

    print(f"[PR2] Итог: Attention={attn_acc:.4f}, RNN={rnn_acc:.4f}")


# =========================
# Практика 3
# =========================
class PR3PositionalEncoding(nn.Module):
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


class PR3MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model должен делиться на heads")
        self.h = heads
        self.d = d_model // heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, s, dm = x.shape
        return x.view(b, s, self.h, self.d).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, h, s, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, h * d)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None) -> torch.Tensor:
        qh = self._split(self.q(q))
        kh = self._split(self.k(k))
        vh = self._split(self.v(v))
        scores = qh @ kh.transpose(-2, -1) / math.sqrt(self.d)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        w = self.drop(torch.softmax(scores, dim=-1))
        return self.o(self._merge(w @ vh))


class PR3FFN(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)


class PR3EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = PR3MultiHeadAttention(d_model, heads, dropout)
        self.ffn = PR3FFN(d_model, ff_dim, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.n1(x + self.drop(self.attn(x, x, x, mask)))
        x = self.n2(x + self.drop(self.ffn(x)))
        return x


class PR3DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = PR3MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = PR3MultiHeadAttention(d_model, heads, dropout)
        self.ffn = PR3FFN(d_model, ff_dim, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        x = self.n1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.n2(x + self.drop(self.cross_attn(x, memory, memory, memory_mask)))
        x = self.n3(x + self.drop(self.ffn(x)))
        return x


class PR3MiniTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, heads: int, layers: int, ff_dim: int, max_len: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe_enc = PR3PositionalEncoding(d_model, max_len)
        self.pe_dec = PR3PositionalEncoding(d_model, max_len)
        self.encoder = nn.ModuleList([PR3EncoderLayer(d_model, heads, ff_dim) for _ in range(layers)])
        self.decoder = nn.ModuleList([PR3DecoderLayer(d_model, heads, ff_dim) for _ in range(layers)])
        self.n_enc = nn.LayerNorm(d_model)
        self.n_dec = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(t, t, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, src: torch.Tensor, tgt_inp: torch.Tensor) -> torch.Tensor:
        src = self.pe_enc(self.emb(src))
        tgt = self.pe_dec(self.emb(tgt_inp))

        memory = src
        for layer in self.encoder:
            memory = layer(memory)
        memory = self.n_enc(memory)

        mask = self.causal_mask(tgt_inp.size(1), tgt_inp.device)
        out = tgt
        for layer in self.decoder:
            out = layer(out, memory, tgt_mask=mask)
        out = self.n_dec(out)

        return self.out(out)


def pr3_data(n_samples: int, seq_len: int, vocab_size: int, bos: int, eos: int, seed: int):
    rng = np.random.default_rng(seed)
    core = rng.integers(3, vocab_size - 1, size=(n_samples, seq_len), dtype=np.int64)
    src = np.concatenate([np.full((n_samples, 1), bos), core, np.full((n_samples, 1), eos)], axis=1)
    rev = np.flip(core, axis=1)
    tgt = np.concatenate([np.full((n_samples, 1), bos), rev, np.full((n_samples, 1), eos)], axis=1)
    return torch.from_numpy(src), torch.from_numpy(tgt)


def pr3_train_eval(cfg: Dict[str, int], src_tr, tgt_tr, src_te, tgt_te, device, vocab_size: int, epochs: int, lr: float, batch: int, seq_len: int) -> float:
    model = PR3MiniTransformer(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        heads=cfg["heads"],
        layers=cfg["layers"],
        ff_dim=cfg["ff_dim"],
        max_len=seq_len + 2,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(src_tr, tgt_tr), batch_size=batch, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for src_b, tgt_b in loader:
            src_b, tgt_b = src_b.to(device), tgt_b.to(device)
            tgt_in = tgt_b[:, :-1]
            tgt_out = tgt_b[:, 1:]
            opt.zero_grad()
            logits = model(src_b, tgt_in)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(f"[PR3] cfg={cfg} epoch={epoch} train_loss={loss_sum / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        src_te, tgt_te = src_te.to(device), tgt_te.to(device)
        logits = model(src_te, tgt_te[:, :-1])
        pred = logits.argmax(dim=-1)
        target = tgt_te[:, 1:]
        token_acc = (pred == target).float().mean().item()

        probs = torch.softmax(logits[0, 0], dim=-1)
        top_p, top_i = torch.topk(probs, k=5)
        top = ", ".join([f"id={i.item()} p={p.item():.3f}" for i, p in zip(top_i, top_p)])
        print(f"[PR3] Top-5 вероятностей (пример 1, 1 токен): {top}")

    return token_acc


def run_pr3() -> None:
    print("\n========== Практика 3: Полная реализация трансформера ==========")
    seed_everything(7)

    n_train, n_test, seq_len, vocab_size = 5000, 1000, 10, 70
    batch, epochs, lr = 64, 6, 2e-3
    bos, eos = 1, 2

    src_tr, tgt_tr = pr3_data(n_train, seq_len, vocab_size, bos, eos, seed=7)
    src_te, tgt_te = pr3_data(n_test, seq_len, vocab_size, bos, eos, seed=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [
        {"layers": 2, "d_model": 64, "heads": 4, "ff_dim": 128},
        {"layers": 3, "d_model": 96, "heads": 6, "ff_dim": 192},
    ]

    scores = []
    for cfg in configs:
        acc = pr3_train_eval(cfg, src_tr, tgt_tr, src_te, tgt_te, device, vocab_size, epochs, lr, batch, seq_len)
        scores.append((cfg, acc))
        print(f"[PR3] cfg={cfg} token_acc={acc:.4f}\n")

    scores.sort(key=lambda x: x[1], reverse=True)
    print("[PR3] Рейтинг конфигураций:")
    for cfg, acc in scores:
        print(f"- {cfg}: token_acc={acc:.4f}")


def main() -> None:
    run_pr1()
    run_pr2()
    run_pr3()
    print("\nTransformers: все практики (1-3) выполнены последовательно.")


if __name__ == "__main__":
    main()
