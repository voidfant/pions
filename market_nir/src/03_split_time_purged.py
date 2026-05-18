#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from common import load_table, parse_utc, save_table, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time-based purged split")
    p.add_argument("--input", default="market_nir/data/processed/labeled_events.parquet")
    p.add_argument("--output", default="market_nir/data/processed/labeled_events_split.parquet")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--horizon", default="2h")
    p.add_argument("--purge", default=None, help="timedelta string, default=horizon")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_table(args.input)
    df = df.copy()
    df["timestamp_utc"] = parse_utc(df["timestamp_utc"])
    df = df.dropna(subset=["timestamp_utc", "label", "text", "ret_h"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    n = len(df)
    if n < 100:
        raise SystemExit(f"Too few rows for split: {n}")

    idx1 = int(n * args.train_ratio)
    idx2 = int(n * (args.train_ratio + args.val_ratio))
    idx1 = min(max(idx1, 1), n - 2)
    idx2 = min(max(idx2, idx1 + 1), n - 1)

    t1 = df.loc[idx1, "timestamp_utc"]
    t2 = df.loc[idx2, "timestamp_utc"]

    import pandas as pd

    purge_td = pd.to_timedelta(args.purge) if args.purge else pd.to_timedelta(args.horizon)

    split = np.full(n, "", dtype=object)

    train_mask = df["timestamp_utc"] < (t1 - purge_td)
    val_mask = (df["timestamp_utc"] >= (t1 + purge_td)) & (df["timestamp_utc"] < (t2 - purge_td))
    test_mask = df["timestamp_utc"] >= (t2 + purge_td)

    split[train_mask.values] = "train"
    split[val_mask.values] = "val"
    split[test_mask.values] = "test"

    df["split"] = split
    out = df[df["split"].isin(["train", "val", "test"])].copy()

    if out["split"].nunique() < 3:
        raise SystemExit("Split failed: not all train/val/test present after purging")

    save_table(out, args.output)

    stats = {
        "rows_total": int(n),
        "rows_after_purge": int(len(out)),
        "purge": str(purge_td),
        "boundary_train_val": str(t1),
        "boundary_val_test": str(t2),
        "counts": {k: int(v) for k, v in out["split"].value_counts().to_dict().items()},
        "time_ranges": {
            split_name: {
                "min": str(g["timestamp_utc"].min()),
                "max": str(g["timestamp_utc"].max()),
            }
            for split_name, g in out.groupby("split")
        },
    }
    write_json(stats, "market_nir/artifacts/metrics/03_split_stats.json")
    print(f"Saved split dataset: {args.output}")
    print(stats)


if __name__ == "__main__":
    main()
