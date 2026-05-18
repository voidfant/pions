#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re

from common import ensure_dir, load_table, save_table, write_json, parse_utc


def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare text events table")
    p.add_argument("--input", required=True, help="Path to raw text_events.csv/parquet")
    p.add_argument("--output", default="market_nir/data/processed/text_events_prepared.parquet")
    p.add_argument("--min-text-len", type=int, default=12)
    p.add_argument("--allowed-langs", nargs="*", default=["en", "ru"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_table(args.input)

    required = ["event_id", "timestamp_utc", "ticker", "text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    df = df.copy()
    if "lang" not in df.columns:
        df["lang"] = "unknown"
    if "source" not in df.columns:
        df["source"] = "unknown"

    df["timestamp_utc"] = parse_utc(df["timestamp_utc"])
    df = df.dropna(subset=["timestamp_utc", "ticker", "text", "event_id"])

    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() >= args.min_text_len]

    if args.allowed_langs:
        df = df[df["lang"].astype(str).str.lower().isin([x.lower() for x in args.allowed_langs])]

    # Dedup exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["event_id"], keep="first")
    df = df.drop_duplicates(subset=["timestamp_utc", "ticker", "text"], keep="first")
    after = len(df)

    df = df.sort_values(["ticker", "timestamp_utc"]).reset_index(drop=True)

    out_path = args.output
    save_table(df[["event_id", "timestamp_utc", "ticker", "source", "lang", "text"]], out_path)

    stats = {
        "rows_before": int(before),
        "rows_after": int(after),
        "rows_final": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "sources": int(df["source"].nunique()),
        "time_min": str(df["timestamp_utc"].min()),
        "time_max": str(df["timestamp_utc"].max()),
    }
    write_json(stats, "market_nir/artifacts/metrics/01_prepare_events_stats.json")
    ensure_dir("market_nir/artifacts/metrics")
    print(f"Saved prepared events: {out_path}")
    print(stats)


if __name__ == "__main__":
    main()
