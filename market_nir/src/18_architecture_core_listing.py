#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import load_table, parse_utc

LABEL_ORDER = ["DOWN", "FLAT", "UP"]
META_COLS = {"event_id", "timestamp_utc", "ticker", "ret_h", "label", "split"}


@dataclass
class ExperimentResult:
    model: str
    family: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    score_ret_corr: float
    top10_hit_rate: float
    top10_cum_return: float
    train_seconds: float
    inference_ms_per_1000: float
    params_count: int


class TabularMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: batch x time x features.
        z = x.transpose(1, 2)
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        return self.head(z)


class TinyTransformer(nn.Module):
    def __init__(self, n_features: int, n_classes: int, seq_len: int):
        super().__init__()
        d_model = 64
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.10,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x) + self.pos[:, : x.shape[1], :]
        z = self.encoder(z)
        return self.head(z[:, -1, :])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Core architecture comparison for market time series")
    parser.add_argument("--input", default="market_nir/data/processed/market_only_dataset.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_market_dataset(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    df = load_table(path).copy()
    df["timestamp_utc"] = parse_utc(df["timestamp_utc"])
    df = df.dropna(subset=["timestamp_utc", "ticker", "ret_h", "label", "split"])
    df = df.sort_values(["timestamp_utc", "ticker"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in META_COLS]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols


def score_from_proba(classes: list[str], proba: np.ndarray) -> np.ndarray:
    up = proba[:, classes.index("UP")] if "UP" in classes else np.zeros(len(proba))
    down = proba[:, classes.index("DOWN")] if "DOWN" in classes else np.zeros(len(proba))
    return up - down


def top10_signal_metrics(score: np.ndarray, ret_h: np.ndarray, cost: float = 0.0005) -> tuple[float, float]:
    tau = float(np.quantile(np.abs(score), 0.90))
    signal = np.where(score > tau, 1.0, np.where(score < -tau, -1.0, 0.0))
    traded = np.abs(signal) > 0
    pnl = signal * ret_h - cost * np.abs(signal)
    hit_rate = float((np.sign(signal[traded]) == np.sign(ret_h[traded])).mean()) if traded.any() else 0.0
    return hit_rate, float(pnl.sum())


def evaluate(
    model_name: str,
    family: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    ret_h: np.ndarray,
    train_seconds: float,
    inference_seconds: float,
    params_count: int,
) -> ExperimentResult:
    corr = float(np.corrcoef(score, ret_h)[0, 1]) if np.std(score) > 0 and np.std(ret_h) > 0 else 0.0
    hit_rate, cum_return = top10_signal_metrics(score, ret_h)
    return ExperimentResult(
        model=model_name,
        family=family,
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        score_ret_corr=corr,
        top10_hit_rate=hit_rate,
        top10_cum_return=cum_return,
        train_seconds=float(train_seconds),
        inference_ms_per_1000=float(inference_seconds / max(len(y_true), 1) * 1_000_000.0),
        params_count=int(params_count),
    )


def train_sklearn_models(df: pd.DataFrame, feature_cols: list[str], seed: int) -> list[ExperimentResult]:
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    x_train = train[feature_cols].values.astype(np.float32)
    y_train = train["label"].astype(str).values
    x_test = test[feature_cols].values.astype(np.float32)
    y_test = test["label"].astype(str).values
    ret_test = test["ret_h"].astype(float).values

    specs = [
        (
            "LogisticRegression",
            "linear",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)),
                ]
            ),
            x_train.shape[1] * len(np.unique(y_train)) + len(np.unique(y_train)),
        ),
        (
            "HistGradientBoosting",
            "tree_boosting",
            HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.03,
                max_depth=5,
                min_samples_leaf=60,
                l2_regularization=0.8,
                random_state=seed,
            ),
            220 * 31,
        ),
    ]

    results: list[ExperimentResult] = []
    for name, family, model, params_count in specs:
        start = time.perf_counter()
        model.fit(x_train, y_train)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        y_pred = model.predict(x_test)
        proba = model.predict_proba(x_test)
        inference_seconds = time.perf_counter() - start

        classes = list(model.classes_ if hasattr(model, "classes_") else model.named_steps["clf"].classes_)
        score = score_from_proba(classes, proba)
        results.append(evaluate(name, family, y_test, y_pred, score, ret_test, train_seconds, inference_seconds, params_count))
    return results


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_seq, y_seq, split_seq, ret_seq = [], [], [], []
    for _, group in df.sort_values(["ticker", "timestamp_utc"]).groupby("ticker", sort=False):
        x = group[feature_cols].values.astype(np.float32)
        y = group["label"].astype(str).values
        splits = group["split"].astype(str).values
        rets = group["ret_h"].astype(float).values
        for i in range(seq_len - 1, len(group)):
            x_seq.append(x[i - seq_len + 1 : i + 1])
            y_seq.append(y[i])
            split_seq.append(splits[i])
            ret_seq.append(rets[i])
    return np.stack(x_seq), np.asarray(y_seq), np.asarray(split_seq), np.asarray(ret_seq)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_torch_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[nn.Module, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_val = -1.0
    start_time = time.perf_counter()
    for _ in range(epochs):
        model.train()
        order = np.random.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = torch.tensor(x_train[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        preds = predict_torch_ids(model, x_val, batch_size)
        val_score = balanced_accuracy_score(y_val, preds)
        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_seconds = time.perf_counter() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_seconds


def predict_torch_ids(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def predict_torch_proba(model: nn.Module, x: np.ndarray, batch_size: int) -> tuple[np.ndarray, float]:
    device = next(model.parameters()).device
    out = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out), time.perf_counter() - start_time


def train_torch_models(df: pd.DataFrame, feature_cols: list[str], args: argparse.Namespace) -> list[ExperimentResult]:
    encoder = LabelEncoder()
    encoder.fit([x for x in LABEL_ORDER if x in set(df["label"].astype(str))])
    classes = list(encoder.classes_)
    n_classes = len(classes)

    scaler = StandardScaler()
    train_mask = df["split"].values == "train"
    df_scaled = df.copy()
    df_scaled.loc[:, feature_cols] = scaler.fit(df.loc[train_mask, feature_cols]).transform(df[feature_cols])

    results: list[ExperimentResult] = []

    train = df_scaled[df_scaled["split"] == "train"]
    val = df_scaled[df_scaled["split"] == "val"]
    test = df_scaled[df_scaled["split"] == "test"]

    x_train = train[feature_cols].values.astype(np.float32)
    y_train = encoder.transform(train["label"].astype(str).values)
    x_val = val[feature_cols].values.astype(np.float32)
    y_val = encoder.transform(val["label"].astype(str).values)
    x_test = test[feature_cols].values.astype(np.float32)
    y_test_raw = test["label"].astype(str).values
    ret_test = test["ret_h"].astype(float).values

    mlp = TabularMLP(len(feature_cols), n_classes)
    mlp, train_seconds = train_torch_model(mlp, x_train, y_train, x_val, y_val, args.epochs, args.batch_size, args.lr)
    proba, inference_seconds = predict_torch_proba(mlp, x_test, args.batch_size)
    y_pred = encoder.inverse_transform(proba.argmax(1))
    score = score_from_proba(classes, proba)
    results.append(evaluate("TorchMLP", "neural_mlp", y_test_raw, y_pred, score, ret_test, train_seconds, inference_seconds, count_params(mlp)))

    x_seq, y_seq_raw, split_seq, ret_seq = make_sequences(df_scaled, feature_cols, args.seq_len)
    y_seq = encoder.transform(y_seq_raw)
    x_train, y_train = x_seq[split_seq == "train"], y_seq[split_seq == "train"]
    x_val, y_val = x_seq[split_seq == "val"], y_seq[split_seq == "val"]
    x_test, y_test = x_seq[split_seq == "test"], y_seq_raw[split_seq == "test"]
    ret_test = ret_seq[split_seq == "test"]

    for name, family, model in [
        ("TemporalCNN", "neural_cnn", TemporalCNN(len(feature_cols), n_classes)),
        ("TinyTransformer", "neural_transformer", TinyTransformer(len(feature_cols), n_classes, args.seq_len)),
    ]:
        model, train_seconds = train_torch_model(model, x_train, y_train, x_val, y_val, args.epochs, args.batch_size, args.lr)
        proba, inference_seconds = predict_torch_proba(model, x_test, args.batch_size)
        y_pred = encoder.inverse_transform(proba.argmax(1))
        score = score_from_proba(classes, proba)
        results.append(evaluate(name, family, y_test, y_pred, score, ret_test, train_seconds, inference_seconds, count_params(model)))

    return results


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    df, feature_cols = load_market_dataset(args.input)

    results = []
    results.extend(train_sklearn_models(df, feature_cols, args.seed))
    results.extend(train_torch_models(df, feature_cols, args))

    table = pd.DataFrame([r.__dict__ for r in results]).sort_values("balanced_accuracy", ascending=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
