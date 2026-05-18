#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from common import LABEL_ORDER, classify_metrics, ensure_dir, load_table, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate and compare model predictions")
    p.add_argument(
        "--predictions",
        nargs="+",
        default=[
            "market_nir/artifacts/predictions/baseline_tfidf_lr_predictions.parquet",
            "market_nir/artifacts/predictions/distilbert_predictions.parquet",
        ],
    )
    p.add_argument("--out-dir", default="market_nir/artifacts")
    return p.parse_args()


def plot_confusion(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(LABEL_ORDER)))
    ax.set_yticks(np.arange(len(LABEL_ORDER)))
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_calibration(df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    # binary projection: UP vs not-UP
    y_true = (df["y_true"] == "UP").astype(int).values
    y_prob = df["prob_UP"].values

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(mean_pred, frac_pos, marker="o", label=model_name)
    ax.set_title(f"Calibration (UP class): {model_name}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "plots")

    rows = []
    summary = {}

    for pred_path in args.predictions:
        path = Path(pred_path)
        if not path.exists():
            print(f"Skip missing predictions file: {path}")
            continue

        df = load_table(path)
        if df.empty:
            continue
        model_name = str(df["model"].iloc[0]) if "model" in df.columns else path.stem

        summary[model_name] = {}
        for split_name, g in df.groupby("split"):
            m = classify_metrics(g["y_true"], g["y_pred"])
            summary[model_name][split_name] = m
            rows.append({"model": model_name, "split": split_name, **m})

        test_df = df[df["split"] == "test"].copy()
        if len(test_df) > 0:
            plot_confusion(
                test_df["y_true"].values,
                test_df["y_pred"].values,
                f"Confusion matrix ({model_name}, test)",
                out_dir / "plots" / f"confusion_{model_name}_test.png",
            )
            plot_calibration(
                test_df,
                model_name,
                out_dir / "plots" / f"calibration_{model_name}_up_test.png",
            )

    if not rows:
        raise SystemExit("No prediction files loaded")

    table = pd.DataFrame(rows).sort_values(["split", "macro_f1"], ascending=[True, False])
    csv_path = out_dir / "metrics" / "model_comparison_table.csv"
    table.to_csv(csv_path, index=False)

    write_json(summary, out_dir / "metrics" / "model_comparison_metrics.json")
    print(f"Saved comparison table: {csv_path}")
    print(summary)


if __name__ == "__main__":
    main()
