#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import LABEL_ORDER, classify_metrics, ensure_dir, load_table, parse_utc, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TF-IDF + LogisticRegression baseline")
    p.add_argument("--input", default="market_nir/data/processed/labeled_events_split.parquet")
    p.add_argument("--max-features", type=int, default=40000)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--out-dir", default="market_nir/artifacts")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_table(args.input).copy()
    df["timestamp_utc"] = parse_utc(df["timestamp_utc"])
    df = df.dropna(subset=["text", "label", "split"]).copy()

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise SystemExit("Need non-empty train/val/test splits")

    vec = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, args.ngram_max), min_df=2)
    x_train = vec.fit_transform(train["text"].astype(str))

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed,
        solver="lbfgs",
    )
    clf.fit(x_train, train["label"].astype(str))

    def infer(part: pd.DataFrame) -> pd.DataFrame:
        x = vec.transform(part["text"].astype(str))
        probs = clf.predict_proba(x)
        pred = clf.predict(x)

        classes = list(clf.classes_)
        out = part[["event_id", "timestamp_utc", "ticker", "split", "label", "ret_h"]].copy()
        out = out.rename(columns={"label": "y_true"})
        out["y_pred"] = pred

        for label in LABEL_ORDER:
            if label in classes:
                idx = classes.index(label)
                out[f"prob_{label}"] = probs[:, idx]
            else:
                out[f"prob_{label}"] = 0.0

        out["model"] = "baseline_tfidf_lr"
        return out

    pred_df = pd.concat([infer(train), infer(val), infer(test)], axis=0).reset_index(drop=True)

    metrics = {}
    for split_name, g in pred_df.groupby("split"):
        metrics[split_name] = classify_metrics(g["y_true"], g["y_pred"])

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "models")
    ensure_dir(out_dir / "predictions")
    ensure_dir(out_dir / "metrics")

    joblib.dump(vec, out_dir / "models" / "baseline_tfidf_vectorizer.joblib")
    joblib.dump(clf, out_dir / "models" / "baseline_tfidf_lr.joblib")

    pred_path = out_dir / "predictions" / "baseline_tfidf_lr_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    metrics_payload = {
        "model": "baseline_tfidf_lr",
        "rows": int(len(pred_df)),
        "splits": metrics,
        "params": {
            "max_features": int(args.max_features),
            "ngram_max": int(args.ngram_max),
        },
    }
    write_json(metrics_payload, out_dir / "metrics" / "baseline_tfidf_lr_metrics.json")

    print(f"Saved predictions: {pred_path}")
    print(metrics_payload)


if __name__ == "__main__":
    main()
