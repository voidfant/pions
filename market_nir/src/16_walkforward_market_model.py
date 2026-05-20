#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from common import LABEL_ORDER, classify_metrics, ensure_dir, load_table, parse_utc, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward training/evaluation for market-only model")
    p.add_argument("--input", default="market_nir/data/processed/market_only_dataset.parquet")
    p.add_argument("--out-dir", default="market_nir/artifacts")

    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--step-days", type=int, default=30)
    p.add_argument("--gap-hours", type=int, default=0)
    p.add_argument("--min-train-rows", type=int, default=3000)
    p.add_argument("--min-test-rows", type=int, default=300)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=600)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--min-samples-leaf", type=int, default=60)
    p.add_argument("--l2", type=float, default=0.8)
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


def build_model(args: argparse.Namespace, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2,
        random_state=seed,
    )


def main() -> None:
    args = parse_args()
    df = load_table(args.input).copy()
    df["timestamp_utc"] = parse_utc(df["timestamp_utc"])
    df = df.dropna(subset=["timestamp_utc", "label", "ret_h"]).copy()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    base_cols = {"event_id", "timestamp_utc", "ticker", "ret_h", "label", "split"}
    feature_cols = [c for c in df.columns if c not in base_cols]
    if not feature_cols:
        raise SystemExit("No feature columns found")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols).copy()

    t_min = df["timestamp_utc"].min()
    t_max = df["timestamp_utc"].max()
    if pd.isna(t_min) or pd.isna(t_max):
        raise SystemExit("Bad timestamps")

    train_td = pd.Timedelta(days=args.train_days)
    test_td = pd.Timedelta(days=args.test_days)
    step_td = pd.Timedelta(days=args.step_days)
    gap_td = pd.Timedelta(hours=args.gap_hours)

    eval_start = t_min + train_td + gap_td
    folds = []
    preds = []
    i = 0

    while eval_start + test_td <= t_max:
        train_start = eval_start - gap_td - train_td
        train_end = eval_start - gap_td
        test_end = eval_start + test_td

        tr = df[(df["timestamp_utc"] >= train_start) & (df["timestamp_utc"] < train_end)].copy()
        te = df[(df["timestamp_utc"] >= eval_start) & (df["timestamp_utc"] < test_end)].copy()

        if len(tr) >= args.min_train_rows and len(te) >= args.min_test_rows:
            x_tr = tr[feature_cols].values
            y_tr = tr["label"].astype(str).values
            x_te = te[feature_cols].values

            model = build_model(args, seed=args.seed + i * 17 + 1)
            sw = sample_weights(y_tr, args.class_balance)
            model.fit(x_tr, y_tr, sample_weight=sw)

            classes = list(model.classes_)
            y_pred = model.predict(x_te)
            y_prob = model.predict_proba(x_te)

            out = te[["event_id", "timestamp_utc", "ticker", "ret_h", "label"]].copy()
            out = out.rename(columns={"label": "y_true"})
            out["y_pred"] = y_pred
            out["split"] = "test"
            out["wf_fold"] = i
            for label in LABEL_ORDER:
                if label in classes:
                    j = classes.index(label)
                    out[f"prob_{label}"] = y_prob[:, j]
                else:
                    out[f"prob_{label}"] = 0.0
            out["model"] = "market_only_hgb_walkforward"
            preds.append(out)

            folds.append(
                {
                    "fold": i,
                    "train_start": str(train_start),
                    "train_end": str(train_end),
                    "test_start": str(eval_start),
                    "test_end": str(test_end),
                    "train_rows": int(len(tr)),
                    "test_rows": int(len(te)),
                    "train_labels": tr["label"].value_counts().to_dict(),
                    "test_labels": te["label"].value_counts().to_dict(),
                }
            )

        eval_start = eval_start + step_td
        i += 1

    if not preds:
        raise SystemExit("No walk-forward folds produced predictions. Relax min rows / window sizes.")

    pred_df = pd.concat(preds, axis=0, ignore_index=True)
    pred_df = pred_df.sort_values("timestamp_utc").drop_duplicates(subset=["event_id"], keep="first").reset_index(drop=True)

    metrics = classify_metrics(pred_df["y_true"], pred_df["y_pred"])

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "predictions")
    ensure_dir(out_dir / "metrics")

    pred_path = out_dir / "predictions" / "market_only_hgb_walkforward_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    payload = {
        "model": "market_only_hgb_walkforward",
        "rows": int(len(pred_df)),
        "n_features": int(len(feature_cols)),
        "metrics_test_aggregate": metrics,
        "params": {
            "train_days": args.train_days,
            "test_days": args.test_days,
            "step_days": args.step_days,
            "gap_hours": args.gap_hours,
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "l2": args.l2,
            "class_balance": args.class_balance,
        },
        "folds": folds,
    }
    write_json(payload, out_dir / "metrics" / "market_only_hgb_walkforward_metrics.json")
    print(f"Saved predictions: {pred_path}")
    print(payload)


if __name__ == "__main__":
    main()
