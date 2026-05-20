#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import LABEL_ORDER, classify_metrics, ensure_dir, load_table, parse_utc, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train high-capacity market-only classifier")
    p.add_argument("--input", default="market_nir/data/processed/market_only_dataset.parquet")
    p.add_argument("--model-type", choices=["hgb", "logreg", "hgb_ensemble"], default="hgb_ensemble")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="market_nir/artifacts")

    p.add_argument("--max-iter", type=int, default=1200)
    p.add_argument("--learning-rate", type=float, default=0.02)
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--min-samples-leaf", type=int, default=16)
    p.add_argument("--l2", type=float, default=0.2)

    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--bootstrap-frac", type=float, default=0.85)
    p.add_argument("--class-balance", choices=["none", "balanced"], default="balanced")
    return p.parse_args()


def class_weight_vector(y: np.ndarray) -> dict[str, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = float(counts.sum())
    k = float(len(classes))
    return {c: total / (k * float(cnt)) for c, cnt in zip(classes, counts)}


def sample_weights(y: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    wmap = class_weight_vector(y)
    return np.array([wmap[v] for v in y], dtype=float)


def build_hgb(args: argparse.Namespace, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2,
        random_state=seed,
    )


def build_logreg(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=seed,
                    solver="lbfgs",
                    multi_class="auto",
                ),
            ),
        ]
    )


def infer_with_model(model, part: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = part[feature_cols].values
    pred = model.predict(x)
    proba = model.predict_proba(x)
    return pred, proba


def main() -> None:
    args = parse_args()
    df = load_table(args.input).copy()
    df["timestamp_utc"] = parse_utc(df["timestamp_utc"])

    required = {"event_id", "timestamp_utc", "ticker", "ret_h", "label", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Input missing columns: {missing}")

    feature_cols = [c for c in df.columns if c not in {"event_id", "timestamp_utc", "ticker", "ret_h", "label", "split"}]
    if not feature_cols:
        raise SystemExit("No feature columns found")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["label", "split", "ret_h"]).copy()
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise SystemExit("Need non-empty train/val/test")

    x_train = train[feature_cols].values
    y_train = train["label"].astype(str).values
    sw_train = sample_weights(y_train, args.class_balance)

    if args.model_type == "logreg":
        model = build_logreg(args.seed)
        model.fit(x_train, y_train)
        models = [model]
        model_name = "market_only_logreg"
    elif args.model_type == "hgb":
        model = build_hgb(args, args.seed)
        model.fit(x_train, y_train, sample_weight=sw_train)
        models = [model]
        model_name = "market_only_hgb"
    else:
        if args.ensemble_size < 2:
            raise SystemExit("--ensemble-size must be >=2 for hgb_ensemble")
        if not (0.2 <= args.bootstrap_frac <= 1.0):
            raise SystemExit("--bootstrap-frac must be in [0.2, 1.0]")

        models = []
        rng = np.random.default_rng(args.seed)
        n = len(train)
        boot_n = max(100, int(n * args.bootstrap_frac))
        for i in range(args.ensemble_size):
            idx = rng.choice(n, size=boot_n, replace=True)
            x_b = x_train[idx]
            y_b = y_train[idx]
            sw_b = sample_weights(y_b, args.class_balance)

            m = build_hgb(args, args.seed + i * 17 + 1)
            m.fit(x_b, y_b, sample_weight=sw_b)
            models.append(m)
        model_name = "market_only_hgb_ensemble"

    def infer(part: pd.DataFrame) -> pd.DataFrame:
        probs_accum = None
        pred_votes: list[np.ndarray] = []
        classes = list(models[0].classes_)

        for m in models:
            pred, probs = infer_with_model(m, part, feature_cols)
            pred_votes.append(pred)
            probs_accum = probs if probs_accum is None else (probs_accum + probs)

        probs_mean = probs_accum / max(1, len(models))

        if len(models) == 1:
            pred_final = pred_votes[0]
        else:
            labels_idx = np.argmax(probs_mean, axis=1)
            pred_final = np.array([classes[j] for j in labels_idx], dtype=object)

        out = part[["event_id", "timestamp_utc", "ticker", "split", "label", "ret_h"]].copy()
        out = out.rename(columns={"label": "y_true"})
        out["y_pred"] = pred_final
        for label in LABEL_ORDER:
            if label in classes:
                j = classes.index(label)
                out[f"prob_{label}"] = probs_mean[:, j]
            else:
                out[f"prob_{label}"] = 0.0
        out["model"] = model_name
        return out

    pred_df = pd.concat([infer(train), infer(val), infer(test)], axis=0, ignore_index=True)
    metrics = {split_name: classify_metrics(g["y_true"], g["y_pred"]) for split_name, g in pred_df.groupby("split")}

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "models")
    ensure_dir(out_dir / "predictions")
    ensure_dir(out_dir / "metrics")

    model_path = out_dir / "models" / f"{model_name}.joblib"
    joblib.dump(
        {
            "models": models,
            "feature_cols": feature_cols,
            "model_name": model_name,
            "params": vars(args),
        },
        model_path,
    )

    pred_path = out_dir / "predictions" / f"{model_name}_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    payload = {
        "model": model_name,
        "rows": int(len(pred_df)),
        "n_features": int(len(feature_cols)),
        "n_models": int(len(models)),
        "features": feature_cols,
        "splits": metrics,
        "params": {
            "model_type": args.model_type,
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "l2": args.l2,
            "ensemble_size": args.ensemble_size,
            "bootstrap_frac": args.bootstrap_frac,
            "class_balance": args.class_balance,
        },
    }
    write_json(payload, out_dir / "metrics" / f"{model_name}_metrics.json")

    print(f"Saved model: {model_path}")
    print(f"Saved predictions: {pred_path}")
    print(payload)


if __name__ == "__main__":
    main()
