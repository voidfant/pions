#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import read_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablations over horizon and k")
    p.add_argument("--events", default="market_nir/data/processed/text_events_prepared.parquet")
    p.add_argument("--market", required=True)
    p.add_argument("--horizons", nargs="+", default=["30m", "2h", "1d"])
    p.add_argument("--ks", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    p.add_argument("--run-distilbert", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--out-csv", default="market_nir/artifacts/metrics/ablation_summary.csv")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    rows = []
    for h, k in itertools.product(args.horizons, args.ks):
        print(f"\n=== ABLATION h={h}, k={k} ===")

        labeled_path = "market_nir/data/processed/labeled_events.parquet"
        split_path = "market_nir/data/processed/labeled_events_split.parquet"

        run(
            [
                args.python,
                str(root / "02_label_events.py"),
                "--events",
                args.events,
                "--market",
                args.market,
                "--output",
                labeled_path,
                "--horizon",
                h,
                "--k",
                str(k),
            ]
        )
        run(
            [
                args.python,
                str(root / "03_split_time_purged.py"),
                "--input",
                labeled_path,
                "--output",
                split_path,
                "--horizon",
                h,
            ]
        )
        run([args.python, str(root / "04_train_baseline_tfidf.py"), "--input", split_path])

        baseline_metrics = read_json("market_nir/artifacts/metrics/baseline_tfidf_lr_metrics.json")
        row = {
            "horizon": h,
            "k": k,
            "model": "baseline_tfidf_lr",
            "test_macro_f1": baseline_metrics["splits"]["test"]["macro_f1"],
            "test_balanced_accuracy": baseline_metrics["splits"]["test"]["balanced_accuracy"],
        }
        rows.append(row)

        if args.run_distilbert:
            run([args.python, str(root / "05_train_distilbert.py"), "--input", split_path])
            dist_metrics = read_json("market_nir/artifacts/metrics/distilbert_metrics.json")
            rows.append(
                {
                    "horizon": h,
                    "k": k,
                    "model": "distilbert",
                    "test_macro_f1": dist_metrics["splits"]["test"]["macro_f1"],
                    "test_balanced_accuracy": dist_metrics["splits"]["test"]["balanced_accuracy"],
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved ablation summary: {args.out_csv}")
    print(out.sort_values(["model", "test_macro_f1"], ascending=[True, False]).head(20))


if __name__ == "__main__":
    main()
