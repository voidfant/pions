#!/usr/bin/env python3
"""Практика 1: Введение в трансформеры (IMDB sentiment)."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


@dataclass
class RunResult:
    model_name: str
    test_loss: float
    test_accuracy: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Практика 1: обучение трансформеров на IMDB")
    parser.add_argument("--models", nargs="+", default=["distilbert-base-uncased", "bert-base-uncased"])
    parser.add_argument("--train-size", type=int, default=4000, help="Сколько train-объектов использовать")
    parser.add_argument("--test-size", type=int, default=1500, help="Сколько test-объектов использовать")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_collate_fn(tokenizer, max_length: int):
    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        texts = [sample["text"] for sample in batch]
        labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        return encoded

    return collate


def evaluate(model, loader, device, criterion) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = criterion(outputs.logits, labels)
            preds = torch.argmax(outputs.logits, dim=-1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def train_one_model(args: argparse.Namespace, model_name: str) -> RunResult:
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"\n=== Модель: {model_name} ===")

    dataset = load_dataset("imdb")
    train_ds = dataset["train"].shuffle(seed=args.seed).select(range(args.train_size))
    test_ds = dataset["test"].shuffle(seed=args.seed).select(range(args.test_size))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    collate_fn = build_collate_fn(tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]

            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs.logits, dim=-1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_count += labels.size(0)

            if step % 50 == 0:
                print(
                    f"Epoch {epoch} | Step {step}/{len(train_loader)} | "
                    f"loss={running_loss / running_count:.4f} | acc={running_correct / running_count:.4f}"
                )

        train_loss = running_loss / running_count
        train_acc = running_correct / running_count
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        print(
            f"Epoch {epoch} завершена: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

    return RunResult(model_name=model_name, test_loss=test_loss, test_accuracy=test_acc)


def print_comparison(results: List[RunResult]) -> None:
    print("\n=== Сравнение предобученных моделей ===")
    ranked = sorted(results, key=lambda x: x.test_accuracy, reverse=True)
    for item in ranked:
        print(f"{item.model_name:30s} | test_loss={item.test_loss:.4f} | test_acc={item.test_accuracy:.4f}")

    best = ranked[0]
    print(f"\nЛучшая модель по точности: {best.model_name} (acc={best.test_accuracy:.4f})")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    try:
        results = [train_one_model(args, model_name) for model_name in args.models]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Не хватает зависимостей. Установите: pip install torch transformers datasets"
        ) from exc

    print_comparison(results)


if __name__ == "__main__":
    main()
