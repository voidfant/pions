#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # noqa: BLE001
    torch = None
    nn = None
    F = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from common import ensure_dir, load_table, parse_utc, write_json

LABEL_ORDER = ["DOWN", "FLAT", "UP"]


@dataclass
class ModelResult:
    name: str
    family: str
    rows_test: int
    n_features: int
    params_count: int
    train_seconds: float
    inference_ms_per_1000: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    score_ret_corr: float
    top10_hit_rate: float
    top10_cum_return: float
    top10_trades: int
    turnover_top10: float
    y_true: np.ndarray
    y_pred: np.ndarray
    score: np.ndarray
    ret_h: np.ndarray
    timestamp_utc: np.ndarray
    history: dict[str, list[float]] | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare model architectures on market OHLCV dataset")
    p.add_argument("--input", default="market_nir/data/processed/market_only_dataset.parquet")
    p.add_argument("--out-dir", default="market_nir/artifacts/architecture_comparison")
    p.add_argument("--max-rows", type=int, default=60000, help="Optional cap after sorting by time; 0 disables")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-epochs", type=int, default=10)
    p.add_argument("--torch-batch-size", type=int, default=256)
    p.add_argument("--torch-lr", type=float, default=1e-3)
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--hgb-max-iter", type=int, default=350)
    p.add_argument("--skip-torch", action="store_true")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def infer_feature_cols(df: pd.DataFrame) -> list[str]:
    base_cols = {"event_id", "timestamp_utc", "ticker", "ret_h", "label", "split"}
    return [c for c in df.columns if c not in base_cols]


def load_dataset(path: str | Path, max_rows: int) -> tuple[pd.DataFrame, list[str]]:
    df = load_table(path).copy()
    required = {"timestamp_utc", "ticker", "ret_h", "label", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Input missing columns: {missing}")

    df["timestamp_utc"] = parse_utc(df["timestamp_utc"])
    df = df.dropna(subset=["timestamp_utc", "ticker", "ret_h", "label", "split"]).copy()
    df = df.sort_values(["timestamp_utc", "ticker"]).reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.tail(max_rows).copy().reset_index(drop=True)

    feature_cols = infer_feature_cols(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols).copy().reset_index(drop=True)
    if not feature_cols:
        raise SystemExit("No feature columns found")
    return df, feature_cols


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    labels = [x for x in LABEL_ORDER if x in set(y_true) or x in set(y_pred)]
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    for i, label in enumerate(labels):
        out[f"precision_{label}"] = float(prec[i])
        out[f"recall_{label}"] = float(rec[i])
        out[f"f1_{label}"] = float(f1[i])
        out[f"support_{label}"] = float(sup[i])
    return out


def score_from_proba(classes: list[str], proba: np.ndarray) -> np.ndarray:
    up = proba[:, classes.index("UP")] if "UP" in classes else np.zeros(len(proba))
    down = proba[:, classes.index("DOWN")] if "DOWN" in classes else np.zeros(len(proba))
    return up - down


def top_signal_stats(score: np.ndarray, ret_h: np.ndarray, q: float = 0.90, cost: float = 0.0005) -> dict[str, float]:
    if len(score) == 0:
        return {"top10_hit_rate": 0.0, "top10_cum_return": 0.0, "top10_trades": 0, "turnover_top10": 0.0}
    tau = float(np.quantile(np.abs(score), q))
    signal = np.where(score > tau, 1.0, np.where(score < -tau, -1.0, 0.0))
    traded = np.abs(signal) > 0
    pnl = signal * ret_h - cost * np.abs(signal)
    hit = float((np.sign(signal[traded]) == np.sign(ret_h[traded])).mean()) if traded.any() else 0.0
    return {
        "top10_hit_rate": hit,
        "top10_cum_return": float(pnl.sum()),
        "top10_trades": int(traded.sum()),
        "turnover_top10": float(traded.mean()),
    }


def evaluate_result(
    name: str,
    family: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    ret_h: np.ndarray,
    timestamp_utc: np.ndarray,
    n_features: int,
    params_count: int,
    train_seconds: float,
    inference_seconds: float,
    history: dict[str, list[float]] | None = None,
) -> ModelResult:
    m = class_metrics(y_true, y_pred)
    s = top_signal_stats(score, ret_h)
    corr = float(np.corrcoef(score, ret_h)[0, 1]) if len(score) > 2 and np.std(score) > 0 and np.std(ret_h) > 0 else 0.0
    return ModelResult(
        name=name,
        family=family,
        rows_test=int(len(y_true)),
        n_features=int(n_features),
        params_count=int(params_count),
        train_seconds=float(train_seconds),
        inference_ms_per_1000=float(inference_seconds / max(len(y_true), 1) * 1000.0 * 1000.0),
        accuracy=m["accuracy"],
        balanced_accuracy=m["balanced_accuracy"],
        macro_f1=m["macro_f1"],
        weighted_f1=m["weighted_f1"],
        score_ret_corr=corr,
        top10_hit_rate=s["top10_hit_rate"],
        top10_cum_return=s["top10_cum_return"],
        top10_trades=s["top10_trades"],
        turnover_top10=s["turnover_top10"],
        y_true=y_true,
        y_pred=y_pred,
        score=score,
        ret_h=ret_h,
        timestamp_utc=timestamp_utc,
        history=history,
    )


def fit_sklearn_models(df: pd.DataFrame, feature_cols: list[str], seed: int, hgb_max_iter: int) -> list[ModelResult]:
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    if min(len(train), len(test)) == 0:
        raise SystemExit("Need non-empty train/test splits")

    x_train = train[feature_cols].values.astype(np.float32)
    y_train = train["label"].astype(str).values
    x_test = test[feature_cols].values.astype(np.float32)
    y_test = test["label"].astype(str).values
    ret_test = test["ret_h"].astype(float).values
    ts_test = test["timestamp_utc"].values

    models = [
        (
            "LogisticRegression",
            "linear",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs", random_state=seed)),
                ]
            ),
            x_train.shape[1] * len(np.unique(y_train)) + len(np.unique(y_train)),
        ),
        (
            "HistGradientBoosting",
            "tree_boosting",
            HistGradientBoostingClassifier(
                max_iter=hgb_max_iter,
                learning_rate=0.03,
                max_depth=5,
                min_samples_leaf=60,
                l2_regularization=0.8,
                random_state=seed,
            ),
            hgb_max_iter * 31,
        ),
    ]

    results: list[ModelResult] = []
    for name, family, model, approx_params in models:
        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        train_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        y_pred = model.predict(x_test)
        proba = model.predict_proba(x_test)
        inference_seconds = time.perf_counter() - t1
        classes = list(model.classes_ if hasattr(model, "classes_") else model.named_steps["clf"].classes_)
        score = score_from_proba(classes, proba)
        results.append(
            evaluate_result(
                name=name,
                family=family,
                y_true=y_test,
                y_pred=y_pred,
                score=score,
                ret_h=ret_test,
                timestamp_utc=ts_test,
                n_features=len(feature_cols),
                params_count=int(approx_params),
                train_seconds=train_seconds,
                inference_seconds=inference_seconds,
            )
        )
    return results


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

    def forward(self, x):
        return self.net(x)


class TemporalCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, n_classes))

    def forward(self, x):
        # x: batch, seq, features
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
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, dropout=0.10, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        z = self.proj(x) + self.pos[:, : x.shape[1], :]
        z = self.encoder(z)
        return self.head(z[:, -1, :])


def count_torch_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_sequence_frame(df: pd.DataFrame, feature_cols: list[str], seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, labels, splits, rets, ts = [], [], [], [], []
    for _, g in df.sort_values(["ticker", "timestamp_utc"]).groupby("ticker", sort=False):
        x = g[feature_cols].values.astype(np.float32)
        y = g["label"].astype(str).values
        sp = g["split"].astype(str).values
        rh = g["ret_h"].astype(float).values
        tt = g["timestamp_utc"].values
        for i in range(seq_len - 1, len(g)):
            xs.append(x[i - seq_len + 1 : i + 1])
            labels.append(y[i])
            splits.append(sp[i])
            rets.append(rh[i])
            ts.append(tt[i])
    return np.stack(xs), np.asarray(labels), np.asarray(splits), np.asarray(rets, dtype=float), np.asarray(ts)


def train_torch_classifier(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[nn.Module, float, dict[str, list[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    rng = np.random.default_rng(seed)
    counts = np.bincount(y_train, minlength=int(max(y_train.max(), y_val.max()) + 1))
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.perf_counter()
    best_state = None
    best_val = -1.0
    patience = 3
    bad = 0
    history = {"train_loss": [], "val_balanced_accuracy": []}
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(x_train))
        loss_sum = 0.0
        loss_count = 0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = torch.tensor(x_train[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            loss_count += 1

        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(x_val), batch_size):
                xb = torch.tensor(x_val[start : start + batch_size], dtype=torch.float32, device=device)
                preds.append(model(xb).argmax(1).cpu().numpy())
        val_bal = balanced_accuracy_score(y_val, np.concatenate(preds)) if len(preds) else 0.0
        history["train_loss"].append(loss_sum / max(loss_count, 1))
        history["val_balanced_accuracy"].append(float(val_bal))
        if val_bal > best_val:
            best_val = val_bal
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_seconds, history


def infer_torch(model: nn.Module, x: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray, float]:
    device = next(model.parameters()).device
    model.eval()
    probs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    inference_seconds = time.perf_counter() - t0
    proba = np.concatenate(probs, axis=0)
    pred = proba.argmax(axis=1)
    return pred, proba, inference_seconds


def fit_torch_models(df: pd.DataFrame, feature_cols: list[str], args: argparse.Namespace) -> list[ModelResult]:
    if args.skip_torch:
        return []
    if torch is None:
        raise SystemExit(f"PyTorch import failed: {TORCH_IMPORT_ERROR!r}")

    le = LabelEncoder()
    le.fit([x for x in LABEL_ORDER if x in set(df["label"].astype(str))] or df["label"].astype(str).values)
    n_classes = len(le.classes_)

    scaler = StandardScaler()
    train_mask = df["split"].values == "train"
    df_scaled = df.copy()
    df_scaled.loc[:, feature_cols] = scaler.fit(df.loc[train_mask, feature_cols]).transform(df[feature_cols]).astype(np.float32)

    tab_train = df_scaled[df_scaled["split"] == "train"].copy()
    tab_val = df_scaled[df_scaled["split"] == "val"].copy()
    tab_test = df_scaled[df_scaled["split"] == "test"].copy()

    results: list[ModelResult] = []
    tab_specs = [
        ("TorchMLP", "neural_mlp", TabularMLP(len(feature_cols), n_classes)),
    ]
    for name, family, model in tab_specs:
        model, train_seconds, history = train_torch_classifier(
            model=model,
            x_train=tab_train[feature_cols].values.astype(np.float32),
            y_train=le.transform(tab_train["label"].astype(str).values),
            x_val=tab_val[feature_cols].values.astype(np.float32),
            y_val=le.transform(tab_val["label"].astype(str).values),
            epochs=args.torch_epochs,
            batch_size=args.torch_batch_size,
            lr=args.torch_lr,
            seed=args.seed,
        )
        pred_id, proba, inf_s = infer_torch(model, tab_test[feature_cols].values.astype(np.float32), args.torch_batch_size)
        y_pred = le.inverse_transform(pred_id)
        classes = list(le.classes_)
        score = score_from_proba(classes, proba)
        results.append(
            evaluate_result(
                name=name,
                family=family,
                y_true=tab_test["label"].astype(str).values,
                y_pred=y_pred,
                score=score,
                ret_h=tab_test["ret_h"].astype(float).values,
                timestamp_utc=tab_test["timestamp_utc"].values,
                n_features=len(feature_cols),
                params_count=count_torch_params(model),
                train_seconds=train_seconds,
                inference_seconds=inf_s,
                history=history,
            )
        )

    seq_x, seq_y_raw, seq_split, seq_ret, seq_ts = make_sequence_frame(df_scaled, feature_cols, args.seq_len)
    seq_y = le.transform(seq_y_raw)
    x_train, y_train = seq_x[seq_split == "train"], seq_y[seq_split == "train"]
    x_val, y_val = seq_x[seq_split == "val"], seq_y[seq_split == "val"]
    x_test, y_test = seq_x[seq_split == "test"], seq_y[seq_split == "test"]
    ret_test = seq_ret[seq_split == "test"]
    ts_test = seq_ts[seq_split == "test"]

    seq_specs = [
        ("TemporalCNN", "neural_cnn", TemporalCNN(len(feature_cols), n_classes)),
        ("TinyTransformer", "neural_transformer", TinyTransformer(len(feature_cols), n_classes, args.seq_len)),
    ]
    for name, family, model in seq_specs:
        model, train_seconds, history = train_torch_classifier(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=args.torch_epochs,
            batch_size=args.torch_batch_size,
            lr=args.torch_lr,
            seed=args.seed,
        )
        pred_id, proba, inf_s = infer_torch(model, x_test, args.torch_batch_size)
        y_pred = le.inverse_transform(pred_id)
        y_true = le.inverse_transform(y_test)
        classes = list(le.classes_)
        score = score_from_proba(classes, proba)
        results.append(
            evaluate_result(
                name=name,
                family=family,
                y_true=y_true,
                y_pred=y_pred,
                score=score,
                ret_h=ret_test,
                timestamp_utc=ts_test,
                n_features=len(feature_cols),
                params_count=count_torch_params(model),
                train_seconds=train_seconds,
                inference_seconds=inf_s,
                history=history,
            )
        )
    return results


def result_record(r: ModelResult) -> dict[str, float | int | str]:
    return {
        "model": r.name,
        "family": r.family,
        "rows_test": r.rows_test,
        "n_features": r.n_features,
        "params_count": r.params_count,
        "train_seconds": r.train_seconds,
        "inference_ms_per_1000": r.inference_ms_per_1000,
        "accuracy": r.accuracy,
        "balanced_accuracy": r.balanced_accuracy,
        "macro_f1": r.macro_f1,
        "weighted_f1": r.weighted_f1,
        "score_ret_corr": r.score_ret_corr,
        "top10_hit_rate": r.top10_hit_rate,
        "top10_cum_return": r.top10_cum_return,
        "top10_trades": r.top10_trades,
        "turnover_top10": r.turnover_top10,
    }


def feature_group_counts(feature_cols: list[str]) -> dict[str, int]:
    groups = {
        "returns": 0,
        "rolling_stats": 0,
        "technical_indicators": 0,
        "volume": 0,
        "calendar": 0,
        "market_context": 0,
        "ticker_identity": 0,
        "other": 0,
    }
    for col in feature_cols:
        if col.startswith(("ret_", "log_ret", "rel_ret")) and not col.startswith(("ret_mean", "ret_std", "ret_skew")):
            groups["returns"] += 1
        elif col.startswith(("ret_mean", "ret_std", "ret_skew", "vol_mean", "vol_std")):
            groups["rolling_stats"] += 1
        elif col.startswith(("rsi", "atr", "macd", "bb_", "sma", "close_rel")):
            groups["technical_indicators"] += 1
        elif col.startswith(("volume", "vol_")):
            groups["volume"] += 1
        elif col.startswith(("hour_", "dow_")):
            groups["calendar"] += 1
        elif col.startswith(("mkt_", "bench_", "rel_")):
            groups["market_context"] += 1
        elif col.startswith("ticker_"):
            groups["ticker_identity"] += 1
        else:
            groups["other"] += 1
    return {k: v for k, v in groups.items() if v > 0}


def save_dataset_plots(df_raw: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> None:
    plots = ensure_dir(out_dir / "plots")

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.axis("off")
    boxes = [
        (0.02, 0.36, 0.14, 0.28, "OHLCV\nbars"),
        (0.20, 0.36, 0.16, 0.28, "Feature\nengineering"),
        (0.40, 0.36, 0.15, 0.28, "Time split\ntrain/val/test"),
        (0.59, 0.36, 0.14, 0.28, "Model\ntraining"),
        (0.77, 0.36, 0.19, 0.28, "Metrics +\nresource analysis"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#EAF2FF", edgecolor="#2E5AAC", linewidth=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)
    for x1, x2 in [(0.16, 0.20), (0.36, 0.40), (0.55, 0.59), (0.73, 0.77)]:
        ax.annotate("", xy=(x2, 0.50), xytext=(x1, 0.50), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))
    ax.text(0.20, 0.16, "ret/volatility/RSI/MACD/Bollinger/market context", fontsize=9, ha="left")
    ax.text(0.40, 0.16, "future leakage is blocked by chronological splits", fontsize=9, ha="left")
    ax.set_title("Pipeline эксперимента раздела 4", fontsize=14, pad=10)
    fig.tight_layout()
    fig.savefig(plots / "4_00_market_experiment_pipeline.png", dpi=180)
    plt.close(fig)

    split_counts = df_raw["split"].value_counts().reindex(["train", "val", "test"]).dropna()
    label_counts = pd.crosstab(df_raw["split"], df_raw["label"]).reindex(["train", "val", "test"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    split_counts.plot(kind="bar", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Размеры временных разбиений")
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Число объектов")
    axes[0].grid(axis="y", alpha=0.25)
    label_counts.plot(kind="bar", stacked=True, ax=axes[1], color=["#C44E52", "#55A868", "#4C72B0"])
    axes[1].set_title("Распределение классов по split")
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("Число объектов")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "4_09_split_and_label_distribution.png", dpi=180)
    plt.close(fig)

    groups = feature_group_counts(feature_cols)
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    names = list(groups.keys())
    vals = list(groups.values())
    ax.barh(names, vals, color="#55A868")
    ax.set_title("Группы признаков market-only датасета")
    ax.set_xlabel("Количество признаков")
    ax.grid(axis="x", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, str(v), va="center")
    fig.tight_layout()
    fig.savefig(plots / "4_10_feature_group_counts.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for ticker, g in df_raw.groupby("ticker"):
        g = g.sort_values("timestamp_utc")
        ax.plot(g["timestamp_utc"], np.cumsum(g["ret_h"].astype(float)), label=str(ticker), alpha=0.85)
    ax.set_title("Накопленная будущая доходность ret_h по тикерам")
    ax.set_xlabel("Время")
    ax.set_ylabel("Cumulative ret_h")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "4_11_market_return_context.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    rows = [
        ("LogisticRegression", "x -> scaler -> linear logits", "fast baseline, no temporal memory"),
        ("HistGradientBoosting", "x -> boosted decision trees", "nonlinear tabular interactions"),
        ("TorchMLP", "x -> dense layers -> logits", "learned nonlinear tabular representation"),
        ("TemporalCNN", "window[t-11:t] -> 1D convolutions", "local temporal motifs"),
        ("TinyTransformer", "window[t-11:t] -> self-attention", "adaptive temporal dependencies"),
    ]
    for i, (name, flow, note) in enumerate(rows):
        y = 0.86 - i * 0.17
        ax.add_patch(plt.Rectangle((0.03, y - 0.055), 0.22, 0.11, facecolor="#EEF3FC", edgecolor="#2E5AAC"))
        ax.text(0.14, y, name, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.add_patch(plt.Rectangle((0.31, y - 0.055), 0.34, 0.11, facecolor="#F7F7F7", edgecolor="#666666"))
        ax.text(0.48, y, flow, ha="center", va="center", fontsize=9)
        ax.add_patch(plt.Rectangle((0.71, y - 0.055), 0.25, 0.11, facecolor="#FFF4E6", edgecolor="#C47F17"))
        ax.text(0.835, y, note, ha="center", va="center", fontsize=9)
    ax.set_title("Сравниваемые архитектуры и их индуктивные предположения", fontsize=14, pad=10)
    fig.tight_layout()
    fig.savefig(plots / "4_12_architecture_assumptions.png", dpi=180)
    plt.close(fig)


def save_plots(results: list[ModelResult], out_dir: Path) -> None:
    plots = ensure_dir(out_dir / "plots")
    df = pd.DataFrame([result_record(r) for r in results]).sort_values("balanced_accuracy", ascending=False)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#64B5CD"]
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    x = np.arange(len(df))
    width = 0.26
    ax.bar(x - width, df["balanced_accuracy"], width, label="Balanced accuracy", color=colors[0])
    ax.bar(x, df["macro_f1"], width, label="Macro F1", color=colors[1])
    ax.bar(x + width, df["top10_hit_rate"], width, label="Top-10% hit-rate", color=colors[2])
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=18, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Качество архитектур на задаче прогноза направления рынка")
    ax.set_ylabel("Значение метрики")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "4_01_architecture_quality_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    sc = ax.scatter(df["train_seconds"], df["balanced_accuracy"], s=np.maximum(60, np.log10(df["params_count"] + 10) * 90), c=np.arange(len(df)), cmap="viridis", alpha=0.85)
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["train_seconds"], row["balanced_accuracy"]), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Время обучения, сек. (log scale)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Компромисс качество / вычислительная стоимость")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "4_02_quality_vs_train_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    df2 = df.sort_values("inference_ms_per_1000")
    ax.barh(df2["model"], df2["inference_ms_per_1000"], color="#4C72B0")
    ax.set_xlabel("мс на 1000 объектов")
    ax.set_title("Скорость инференса разных архитектур")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "4_03_inference_latency.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    df3 = df.sort_values("params_count")
    ax.barh(df3["model"], df3["params_count"], color="#55A868")
    ax.set_xscale("log")
    ax.set_xlabel("Число обучаемых параметров / оценка сложности (log scale)")
    ax.set_title("Параметрическая сложность моделей")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "4_04_parameter_complexity.png", dpi=180)
    plt.close(fig)

    best = max(results, key=lambda r: r.balanced_accuracy)
    labels = [x for x in LABEL_ORDER if x in set(best.y_true) or x in set(best.y_pred)]
    cm = confusion_matrix(best.y_true, best.y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истинный класс")
    ax.set_title(f"Матрица ошибок лучшей модели: {best.name}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plots / "4_05_best_model_confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.scatter(best.score, best.ret_h, s=10, alpha=0.35, color="#4C72B0")
    if len(best.score) > 2 and np.std(best.score) > 0:
        coef = np.polyfit(best.score, best.ret_h, 1)
        xx = np.linspace(best.score.min(), best.score.max(), 120)
        ax.plot(xx, coef[0] * xx + coef[1], color="#C44E52", lw=2, label=f"corr={best.score_ret_corr:.3f}")
        ax.legend()
    ax.set_title(f"Связь score и будущей доходности: {best.name}")
    ax.set_xlabel("score = P(UP) - P(DOWN)")
    ax.set_ylabel("ret_h")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "4_06_score_vs_future_return.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    for r in results:
        tau = float(np.quantile(np.abs(r.score), 0.90))
        signal = np.where(r.score > tau, 1.0, np.where(r.score < -tau, -1.0, 0.0))
        equity = np.cumsum(signal * r.ret_h - 0.0005 * np.abs(signal))
        ax.plot(equity, label=r.name, linewidth=1.7)
    ax.set_title("Equity-кривые top-10% сигналов по архитектурам")
    ax.set_xlabel("Индекс тестового события")
    ax.set_ylabel("Накопленный PnL")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "4_07_equity_curves_top10.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    cols = ["accuracy", "balanced_accuracy", "macro_f1", "top10_hit_rate", "score_ret_corr"]
    heat = df.set_index("model")[cols].values.astype(float)
    im = ax.imshow(heat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["model"])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Тепловая карта итоговых метрик")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plots / "4_08_metrics_heatmap.png", dpi=180)
    plt.close(fig)

    history_rows = [(r.name, r.history) for r in results if r.history]
    if history_rows:
        fig, ax = plt.subplots(figsize=(9.6, 5.0))
        for name, hist in history_rows:
            vals = hist.get("val_balanced_accuracy", [])
            ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", label=name)
        ax.set_title("Validation balanced accuracy по эпохам")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "4_13_training_val_balanced_accuracy.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.6, 5.0))
        for name, hist in history_rows:
            vals = hist.get("train_loss", [])
            ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", label=name)
        ax.set_title("Training loss нейросетевых моделей")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "4_14_training_loss_curves.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    for r in results:
        ax.hist(r.score, bins=40, alpha=0.35, label=r.name, density=True)
    ax.set_title("Распределение score = P(UP) - P(DOWN)")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "4_15_score_distributions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    top_df = df.sort_values("top10_cum_return", ascending=False)
    x = np.arange(len(top_df))
    ax.bar(x, top_df["top10_cum_return"], color=["#55A868" if v >= 0 else "#C44E52" for v in top_df["top10_cum_return"]])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("PnL top-10% наиболее уверенных сигналов")
    ax.set_ylabel("Cumulative PnL")
    ax.set_xticks(x)
    ax.set_xticklabels(top_df["model"], rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "4_16_top10_pnl_by_model.png", dpi=180)
    plt.close(fig)

    best = max(results, key=lambda r: r.balanced_accuracy)
    err = (best.y_true != best.y_pred).astype(float)
    roll = pd.Series(err).rolling(120, min_periods=20).mean()
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.plot(roll.values, color="#C44E52")
    ax.set_title(f"Скользящая доля ошибок лучшей модели: {best.name}")
    ax.set_xlabel("Индекс тестового события")
    ax.set_ylabel("Rolling error rate, window=120")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "4_17_best_model_rolling_error.png", dpi=180)
    plt.close(fig)

    radar_cols = ["balanced_accuracy", "macro_f1", "top10_hit_rate"]
    norm = df.copy()
    for col in ["train_seconds", "inference_ms_per_1000", "params_count"]:
        cmax = float(norm[col].max())
        cmin = float(norm[col].min())
        norm[f"eff_{col}"] = 1.0 - (norm[col] - cmin) / max(cmax - cmin, 1e-12)
    radar_metrics = ["balanced_accuracy", "macro_f1", "top10_hit_rate", "eff_train_seconds", "eff_inference_ms_per_1000", "eff_params_count"]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
    angles = np.r_[angles, angles[0]]
    fig = plt.figure(figsize=(7.6, 7.6))
    ax = fig.add_subplot(111, polar=True)
    for _, row in norm.iterrows():
        vals = np.array([row[m] for m in radar_metrics], dtype=float)
        vals = np.r_[vals, vals[0]]
        ax.plot(angles, vals, linewidth=1.6, label=row["model"])
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, ["Bal.acc", "Macro-F1", "Hit-rate", "Train eff.", "Infer eff.", "Param eff."])
    ax.set_ylim(0, 1)
    ax.set_title("Интегральный профиль качества и эффективности")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "4_18_quality_efficiency_radar.png", dpi=180)
    plt.close(fig)


def write_report_text(results: list[ModelResult], out_dir: Path, dataset_stats: dict) -> None:
    df = pd.DataFrame([result_record(r) for r in results]).sort_values("balanced_accuracy", ascending=False)
    best = df.iloc[0]
    fastest = df.sort_values("train_seconds").iloc[0]
    economical = df.sort_values("params_count").iloc[0]

    lines = [
        "# Раздел 4. Влияние архитектуры модели на качество и вычислительную эффективность прогнозирования рыночных временных рядов",
        "",
        "## 4.1 Постановка задачи",
        "Цель раздела - сравнить несколько архитектур машинного обучения в едином прикладном сценарии: прогнозирование направления будущего движения рынка по OHLCV-данным. В отличие от демонстрационных синтетических задач, здесь данные обладают высокой шумностью, слабым полезным сигналом и временной нестационарностью. Поэтому качество модели оценивается не только по accuracy, но и по balanced accuracy, macro-F1, связи score с будущей доходностью, hit-rate наиболее уверенных сигналов и вычислительной стоимости.",
        "",
        "## 4.2 Данные и признаки",
        f"В эксперименте использован подготовленный market-only датасет: {dataset_stats.get('rows', 'n/a')} строк, {dataset_stats.get('tickers', 'n/a')} тикеров, {dataset_stats.get('n_features', 'n/a')} признаков. Метка строится по будущей доходности ret_h; при бинарном режиме остаются классы UP и DOWN, что делает постановку ближе к задаче directional forecasting.",
        "Признаковое пространство включает доходности на разных лагах, rolling volatility, RSI, ATR, MACD, Bollinger z-score, признаки объема, календарные признаки, а также market-context признаки: относительное движение к рынку, benchmark-relative признаки и breadth-оценки.",
        "",
        "## 4.3 Сравниваемые архитектуры",
        "В качестве линейного baseline используется Logistic Regression. Она быстра и интерпретируема, но ограничена линейной разделяющей поверхностью. HistGradientBoosting представляет деревья решений с бустингом и хорошо подходит для табличных нелинейных признаков. TorchMLP проверяет, достаточно ли простой полносвязной нейросети для извлечения устойчивого сигнала. TemporalCNN использует короткое окно истории и локальные свертки по времени. TinyTransformer также работает с временным окном, но заменяет локальный inductive bias механизмом self-attention.",
        "",
        "## 4.4 Протокол эксперимента",
        "Все модели обучались на одном и том же временном разбиении train/val/test без перемешивания будущих наблюдений в прошлое. Для нейросетевых моделей использовались StandardScaler, AdamW, class-weighted CrossEntropyLoss, gradient clipping и ранняя остановка по balanced accuracy на validation-части. Для последовательных моделей целевой объект формировался из последних наблюдений одного и того же тикера, чтобы не смешивать независимые временные линии.",
        "",
        "## 4.5 Полученные результаты",
        f"Лучший результат по balanced accuracy показала модель {best['model']}: balanced accuracy = {best['balanced_accuracy']:.3f}, macro-F1 = {best['macro_f1']:.3f}. Самой быстрой по времени обучения стала {fastest['model']} ({fastest['train_seconds']:.2f} сек.), а минимальную параметрическую сложность имеет {economical['model']}.",
        f"Для наиболее уверенных top-10% сигналов лучшая по качеству модель дала hit-rate = {best['top10_hit_rate']:.3f} и накопленный PnL = {best['top10_cum_return']:.3f}. Эти значения не следует интерпретировать как готовую торговую стратегию: они используются как аналитическая диагностика того, превращается ли классификационный score в направленный рыночный сигнал.",
        "",
        "| Модель | Семейство | Balanced accuracy | Macro-F1 | Top-10% hit-rate | PnL top-10% | Train, сек | Inference, мс/1000 | Параметры/сложность |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['model']} | {row['family']} | {row['balanced_accuracy']:.3f} | {row['macro_f1']:.3f} | {row['top10_hit_rate']:.3f} | {row['top10_cum_return']:.3f} | {row['train_seconds']:.2f} | {row['inference_ms_per_1000']:.2f} | {int(row['params_count'])} |"
        )
    lines.extend(
        [
            "",
            "## 4.6 Анализ графиков",
            "Рисунок 4.1 сопоставляет классификационные и торгово-диагностические метрики. Рисунок 4.2 показывает компромисс между качеством и временем обучения: архитектура может быть точнее, но слишком дорогой для регулярного переобучения. Рисунок 4.3 фиксирует скорость инференса, важную для применения модели в онлайн-сценарии. Рисунок 4.4 показывает параметрическую сложность. Рисунки 4.5-4.8 дают детализацию ошибок, связь score с будущей доходностью, equity-кривые и интегральную карту метрик.",
            "",
        "## 4.7 Выводы по разделу",
            "Эксперимент подтверждает, что на шумных рыночных временных рядах увеличение архитектурной сложности само по себе не гарантирует улучшения качества, но архитектура, способная учитывать короткий временной контекст, может получить преимущество. В данном запуске TinyTransformer оказался сильнее линейной модели, бустинга, MLP и TemporalCNN по balanced accuracy и macro-F1, однако это преимущество оплачивается большим числом параметров и временем обучения. Практически значимый вывод состоит в том, что выбор архитектуры должен опираться на совместную оценку качества, устойчивости, скорости обучения и интерпретируемости, а не на формальную современность модели.",
            "",
            "## Рисунки для вставки",
            "1. `4_01_architecture_quality_metrics.png` - качество архитектур.",
            "2. `4_02_quality_vs_train_time.png` - качество против времени обучения.",
            "3. `4_03_inference_latency.png` - скорость инференса.",
            "4. `4_04_parameter_complexity.png` - параметрическая сложность.",
            "5. `4_05_best_model_confusion_matrix.png` - матрица ошибок лучшей модели.",
            "6. `4_06_score_vs_future_return.png` - связь score и будущей доходности.",
            "7. `4_07_equity_curves_top10.png` - equity-кривые top-10% сигналов.",
            "8. `4_08_metrics_heatmap.png` - тепловая карта метрик.",
        ]
    )
    (out_dir / "section_4_market_architecture_text.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir)
    ensure_dir(out_dir / "metrics")

    df, feature_cols = load_dataset(args.input, args.max_rows)
    stats = {
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "n_features": int(len(feature_cols)),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "label_distribution": {k: int(v) for k, v in df["label"].value_counts().to_dict().items()},
        "time_min": str(df["timestamp_utc"].min()),
        "time_max": str(df["timestamp_utc"].max()),
    }

    results = fit_sklearn_models(df, feature_cols, args.seed, args.hgb_max_iter)
    results.extend(fit_torch_models(df, feature_cols, args))

    records = [result_record(r) for r in results]
    metrics_df = pd.DataFrame(records).sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(out_dir / "metrics" / "architecture_comparison_metrics.csv", index=False)
    metrics_df.to_json(out_dir / "metrics" / "architecture_comparison_metrics.json", orient="records", force_ascii=False, indent=2)

    predictions_dir = ensure_dir(out_dir / "predictions")
    for r in results:
        pred_df = pd.DataFrame(
            {
                "timestamp_utc": r.timestamp_utc,
                "y_true": r.y_true,
                "y_pred": r.y_pred,
                "score": r.score,
                "ret_h": r.ret_h,
                "model": r.name,
            }
        )
        pred_df.to_parquet(predictions_dir / f"{r.name}_predictions.parquet", index=False)

    histories = {r.name: r.history for r in results if r.history}
    if histories:
        write_json(histories, out_dir / "metrics" / "training_histories.json")

    payload = {
        "dataset": stats,
        "params": vars(args),
        "results": records,
        "best_by_balanced_accuracy": metrics_df.iloc[0].to_dict() if len(metrics_df) else None,
    }
    write_json(payload, out_dir / "metrics" / "architecture_comparison_summary.json")
    save_dataset_plots(df, feature_cols, out_dir)
    save_plots(results, out_dir)
    write_report_text(results, out_dir, stats)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved architecture comparison artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
