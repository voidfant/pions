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
    p = argparse.ArgumentParser(description="Train market-only classifier on OHLCV features")
    p.add_argument("--input", default="market_nir/data/processed/market_only_dataset.parquet")
    p.add_argument("--model-type", choices=["hgb", "logreg"], default="hgb")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="market_nir/artifacts")
    return p.parse_args()


def build_model(model_type: str, seed: int):
    if model_type == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                        solver="lbfgs",
                        multi_class="auto",
                    ),
                ),
            ]
        )

    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=350,
        max_depth=5,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=seed,
    )


def main() -> None:
    args = parse_args()
    df = load_table(args.input).copy()
    df["timestamp_utc"] = parse_utc(df["timestamp_utc"])

    required = {"event_id", "timestamp_utc", "ticker", "ret_h", "label", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Input missing columns: {missing}")

    feature_cols = [c for c in df.columns if c.startswith(("ret_", "hl_", "oc_", "volume_", "vol_", "close_"))]
    feature_cols = [c for c in feature_cols if c not in {"ret_h"}]
    if not feature_cols:
        raise SystemExit("No feature columns found. Expected market features with prefixes.")

    df = df.dropna(subset=feature_cols + ["label", "split", "ret_h"]).copy()
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise SystemExit("Need non-empty train/val/test")

    model = build_model(args.model_type, args.seed)
    x_train = train[feature_cols].values
    y_train = train["label"].astype(str).values
    model.fit(x_train, y_train)

    def infer(part: pd.DataFrame) -> pd.DataFrame:
        x = part[feature_cols].values
        pred = model.predict(x)
        probs = model.predict_proba(x)
        classes = list(model.classes_)

        out = part[["event_id", "timestamp_utc", "ticker", "split", "label", "ret_h"]].copy()
        out = out.rename(columns={"label": "y_true"})
        out["y_pred"] = pred
        for label in LABEL_ORDER:
            if label in classes:
                j = classes.index(label)
                out[f"prob_{label}"] = probs[:, j]
            else:
                out[f"prob_{label}"] = 0.0
        out["model"] = f"market_only_{args.model_type}"
        return out

    pred_df = pd.concat([infer(train), infer(val), infer(test)], axis=0, ignore_index=True)
    metrics = {split_name: classify_metrics(g["y_true"], g["y_pred"]) for split_name, g in pred_df.groupby("split")}

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "models")
    ensure_dir(out_dir / "predictions")
    ensure_dir(out_dir / "metrics")

    model_path = out_dir / "models" / f"market_only_{args.model_type}.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)

    pred_path = out_dir / "predictions" / f"market_only_{args.model_type}_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    payload = {
        "model": f"market_only_{args.model_type}",
        "rows": int(len(pred_df)),
        "n_features": int(len(feature_cols)),
        "features": feature_cols,
        "splits": metrics,
    }
    write_json(payload, out_dir / "metrics" / f"market_only_{args.model_type}_metrics.json")

    print(f"Saved model: {model_path}")
    print(f"Saved predictions: {pred_path}")
    print(payload)


if __name__ == "__main__":
    main()
