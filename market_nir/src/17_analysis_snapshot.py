#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print compact analysis snapshot for market_nir results")
    p.add_argument("--artifacts", default="market_nir/artifacts")
    p.add_argument("--model", default="market_only_hgb", help="Model id, e.g. market_only_hgb")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--predictions", default=None, help="Optional explicit predictions parquet path")
    p.add_argument("--top-tau", type=int, default=8, help="How many top tau tuning rows to print")
    return p.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def print_section(title: str) -> None:
    print(f"\n--- {title}")


def print_json(title: str, path: Path) -> dict | None:
    print_section(f"{title}: {path}")
    obj = read_json(path)
    if obj is None:
        print("missing")
        return None
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    return obj


def main() -> None:
    args = parse_args()
    art = Path(args.artifacts)
    metrics_dir = art / "metrics"
    pred_dir = art / "predictions"

    dataset_stats_path = metrics_dir / "13_market_only_dataset_stats.json"
    model_metrics_path = metrics_dir / f"{args.model}_metrics.json"
    backtest_path = metrics_dir / f"backtest_{args.model}_{args.split}.json"
    mc_path = metrics_dir / f"monte_carlo_{args.model}_{args.split}.json"
    tau_csv_path = metrics_dir / f"tau_tuning_{args.model}_val.csv"
    tau_json_path = metrics_dir / f"tau_tuning_{args.model}_val.json"
    cmp_path = metrics_dir / "model_comparison_table.csv"

    print_json("13_market_only_dataset_stats.json", dataset_stats_path)
    model_metrics = print_json(f"{args.model}_metrics.json", model_metrics_path)
    print_json(f"backtest_{args.model}_{args.split}.json", backtest_path)
    print_json(f"monte_carlo_{args.model}_{args.split}.json", mc_path)

    print_section(f"tau_tuning_{args.model}_val.csv")
    if tau_csv_path.exists():
        tau_df = pd.read_csv(tau_csv_path)
        cols = [c for c in ["quantile", "tau", "objective", "cum_return", "event_sharpe", "trades", "turnover", "hit_rate", "eligible"] if c in tau_df.columns]
        print(tau_df[cols].head(args.top_tau).to_string(index=False))
    else:
        print("missing")

    print_section(f"tau_tuning_{args.model}_val.json")
    if tau_json_path.exists():
        print(json.dumps(read_json(tau_json_path), ensure_ascii=False, indent=2)[:6000])
    else:
        print("missing")

    if cmp_path.exists():
        print_section("model_comparison_table.csv (filtered)")
        cmp_df = pd.read_csv(cmp_path)
        if "model" in cmp_df.columns:
            cmp_df = cmp_df[cmp_df["model"] == args.model]
        if "split" in cmp_df.columns:
            cmp_df = cmp_df[cmp_df["split"] == args.split]
        print(cmp_df.head(20).to_string(index=False) if len(cmp_df) else "no rows")
    else:
        print_section("model_comparison_table.csv")
        print("missing")

    pred_path = Path(args.predictions) if args.predictions else pred_dir / f"{args.model}_predictions.parquet"
    print_section(f"predictions: {pred_path}")
    if not pred_path.exists():
        print("missing predictions parquet")
        return

    df = pd.read_parquet(pred_path)
    if "split" in df.columns:
        df = df[df["split"] == args.split].copy()
    if len(df) == 0:
        print(f"no rows for split={args.split}")
        return

    print(f"rows={len(df)}")

    if "y_pred" in df.columns:
        print_section("y_pred counts")
        print(df["y_pred"].value_counts(dropna=False).to_dict())

    if "y_true" in df.columns:
        print_section("y_true counts")
        print(df["y_true"].value_counts(dropna=False).to_dict())

    if {"y_true", "y_pred"}.issubset(df.columns):
        print_section("confusion-like crosstab")
        ct = pd.crosstab(df["y_true"], df["y_pred"], dropna=False)
        print(ct.to_string())

    if {"prob_UP", "prob_DOWN"}.issubset(df.columns):
        score = (df["prob_UP"].values - df["prob_DOWN"].values).astype(float)
        print_section("score stats")
        print(
            {
                "min": float(np.min(score)),
                "max": float(np.max(score)),
                "mean": float(np.mean(score)),
                "std": float(np.std(score)),
                "q90_abs": float(np.quantile(np.abs(score), 0.90)),
                "q95_abs": float(np.quantile(np.abs(score), 0.95)),
                "q99_abs": float(np.quantile(np.abs(score), 0.99)),
            }
        )

    if {"ret_h", "prob_UP", "prob_DOWN"}.issubset(df.columns):
        score = (df["prob_UP"].values - df["prob_DOWN"].values).astype(float)
        ret_h = df["ret_h"].values.astype(float)
        corr = float(np.corrcoef(score, ret_h)[0, 1]) if len(score) > 2 else float("nan")
        print_section("score-ret_h correlation")
        print({"corr(score, ret_h)": corr})

    if model_metrics and "splits" in model_metrics and args.split in model_metrics["splits"]:
        print_section(f"quick verdict: {args.model}/{args.split}")
        m = model_metrics["splits"][args.split]
        print(
            {
                "macro_f1": m.get("macro_f1"),
                "balanced_accuracy": m.get("balanced_accuracy"),
                "recall_DOWN": m.get("recall_DOWN"),
                "recall_FLAT": m.get("recall_FLAT"),
                "recall_UP": m.get("recall_UP"),
            }
        )


if __name__ == "__main__":
    main()
