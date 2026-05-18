#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import LABEL_ORDER, classify_metrics, ensure_dir, load_table, map_id_to_label, map_label_to_id, parse_utc, write_json


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def looks_like_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:2]
    except Exception:
        return False
    if len(head) < 2:
        return False
    return head[0].startswith("version https://git-lfs.github.com/spec/v1") and head[1].startswith("oid sha256:")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DistilBERT classifier on labeled events")
    p.add_argument("--input", default="market_nir/data/processed/labeled_events_split.parquet")
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-class-weight", action="store_true")
    p.add_argument("--local-files-only", action="store_true", help="Use only local HF cache/files")
    p.add_argument("--out-dir", default="market_nir/artifacts")
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weight_from_labels(y_ids: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y_ids, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
            pred = probs.argmax(axis=1)
            y_true.append(labels)
            y_pred.append(pred)
            y_prob.append(probs)
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    df = load_table(args.input).copy()
    df["timestamp_utc"] = parse_utc(df["timestamp_utc"])
    df = df.dropna(subset=["text", "label", "split"]).copy()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise SystemExit("Need non-empty train/val/test splits")

    label2id = {label: i for i, label in enumerate(LABEL_ORDER)}
    id2label = {i: label for i, label in enumerate(LABEL_ORDER)}

    model_path = Path(args.model_name)
    if model_path.exists() and model_path.is_dir():
        suspect_files = [
            model_path / "pytorch_model.bin",
            model_path / "model.safetensors",
            model_path / "tf_model.h5",
            model_path / "flax_model.msgpack",
        ]
        for sf in suspect_files:
            if looks_like_lfs_pointer(sf):
                raise SystemExit(
                    f"Local model file '{sf}' is a Git LFS pointer, not real weights. "
                    "Download actual weights (git lfs pull or Hugging Face snapshot_download) and retry."
                )

    train_y = map_label_to_id(train_df["label"].astype(str).values)
    val_y = map_label_to_id(val_df["label"].astype(str).values)
    test_y = map_label_to_id(test_df["label"].astype(str).values)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            local_files_only=args.local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(LABEL_ORDER),
            label2id=label2id,
            id2label=id2label,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        raise SystemExit(
            "Failed to load DistilBERT weights/tokenizer. "
            "Either enable internet access to Hugging Face Hub or provide a local model path to --model-name "
            "(and optionally pass --local-files-only)."
        ) from exc

    train_ds = TextDataset(train_df["text"].astype(str).values, train_y, tokenizer, args.max_len)
    val_ds = TextDataset(val_df["text"].astype(str).values, val_y, tokenizer, args.max_len)
    test_ds = TextDataset(test_df["text"].astype(str).values, test_y, tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.no_class_weight:
        criterion = torch.nn.CrossEntropyLoss()
        class_w = None
    else:
        class_w = class_weight_from_labels(train_y, len(LABEL_ORDER)).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_w)

    best_state = None
    best_val_f1 = -1.0
    bad_epochs = 0
    epoch_logs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batches = 0

        for batch in train_loader:
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn)
            loss = criterion(out.logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            batches += 1

        y_true_val, y_pred_val, _ = evaluate(model, val_loader, device)
        val_macro_f1 = float(f1_score(y_true_val, y_pred_val, average="macro", zero_division=0))

        log = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, batches),
            "val_macro_f1": val_macro_f1,
        }
        epoch_logs.append(log)
        print(log)

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final inference
    y_true_tr, y_pred_tr, y_prob_tr = evaluate(model, train_loader, device)
    y_true_va, y_pred_va, y_prob_va = evaluate(model, val_loader, device)
    y_true_te, y_pred_te, y_prob_te = evaluate(model, test_loader, device)

    def pack(part: pd.DataFrame, y_true, y_pred, y_prob, split_name: str) -> pd.DataFrame:
        out = part[["event_id", "timestamp_utc", "ticker", "ret_h"]].copy().reset_index(drop=True)
        out["split"] = split_name
        out["y_true"] = map_id_to_label(y_true)
        out["y_pred"] = map_id_to_label(y_pred)
        for i, label in enumerate(LABEL_ORDER):
            out[f"prob_{label}"] = y_prob[:, i]
        out["model"] = "distilbert"
        return out

    pred_df = pd.concat(
        [
            pack(train_df, y_true_tr, y_pred_tr, y_prob_tr, "train"),
            pack(val_df, y_true_va, y_pred_va, y_prob_va, "val"),
            pack(test_df, y_true_te, y_pred_te, y_prob_te, "test"),
        ],
        axis=0,
        ignore_index=True,
    )

    metrics = {
        split_name: classify_metrics(g["y_true"], g["y_pred"])
        for split_name, g in pred_df.groupby("split")
    }

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "models")
    ensure_dir(out_dir / "predictions")
    ensure_dir(out_dir / "metrics")

    model_dir = out_dir / "models" / "distilbert_market"
    ensure_dir(model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    pred_path = out_dir / "predictions" / "distilbert_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    metrics_payload = {
        "model": "distilbert",
        "model_name": args.model_name,
        "device": str(device),
        "rows": int(len(pred_df)),
        "splits": metrics,
        "epochs_ran": len(epoch_logs),
        "best_val_macro_f1": float(best_val_f1),
        "params": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_len": args.max_len,
            "class_weight": None if class_w is None else [float(x) for x in class_w.detach().cpu().numpy()],
        },
    }
    write_json(metrics_payload, out_dir / "metrics" / "distilbert_metrics.json")
    write_json({"epochs": epoch_logs}, out_dir / "metrics" / "distilbert_training_log.json")

    print(f"Saved model: {model_dir}")
    print(f"Saved predictions: {pred_path}")
    print(metrics_payload)


if __name__ == "__main__":
    main()
