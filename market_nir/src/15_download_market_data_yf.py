#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download extended OHLCV dataset from Yahoo Finance")
    p.add_argument("--tickers", default="SPY,QQQ,IWM,DIA,TLT,GLD,USO")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default="1h", help="e.g. 1h, 1d")
    p.add_argument("--output", default="market_nir/data/raw/market_bars_yf.csv")
    p.add_argument("--append-to", default=None, help="Optional existing csv to append and deduplicate")
    return p.parse_args()


def _normalize_single(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    for r in required:
        if r not in cols:
            raise ValueError(f"Missing column '{r}' for ticker {ticker}")

    out = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(df.index, utc=True),
            "ticker": ticker,
            "open": pd.to_numeric(df[cols["open"]], errors="coerce"),
            "high": pd.to_numeric(df[cols["high"]], errors="coerce"),
            "low": pd.to_numeric(df[cols["low"]], errors="coerce"),
            "close": pd.to_numeric(df[cols["close"]], errors="coerce"),
            "volume": pd.to_numeric(df[cols["volume"]], errors="coerce"),
        }
    )
    return out.dropna(subset=["timestamp_utc", "open", "high", "low", "close", "volume"])


def main() -> None:
    args = parse_args()
    try:
        import yfinance as yf
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "yfinance is not installed. Install with: pip install yfinance"
        ) from exc

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("No tickers provided")

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        data = yf.download(
            tickers=ticker,
            start=args.start,
            end=args.end,
            interval=args.interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if data is None or len(data) == 0:
            print(f"Skip empty ticker: {ticker}")
            continue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        frames.append(_normalize_single(data, ticker))

    if not frames:
        raise SystemExit("No data downloaded")

    out = pd.concat(frames, axis=0, ignore_index=True)

    if args.append_to:
        old_path = Path(args.append_to)
        if old_path.exists():
            old = pd.read_csv(old_path)
            old["timestamp_utc"] = pd.to_datetime(old["timestamp_utc"], utc=True, errors="coerce")
            out = pd.concat([old, out], axis=0, ignore_index=True)

    out = out.dropna(subset=["timestamp_utc", "ticker"]).copy()
    out = out.drop_duplicates(subset=["timestamp_utc", "ticker"], keep="last")
    out = out.sort_values(["ticker", "timestamp_utc"]).reset_index(drop=True)

    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False)

    stats = {
        "rows": int(len(out)),
        "tickers": int(out["ticker"].nunique()),
        "time_min": str(out["timestamp_utc"].min()),
        "time_max": str(out["timestamp_utc"].max()),
        "interval": args.interval,
        "output": str(out_path),
    }
    write_json(stats, "market_nir/artifacts/metrics/15_download_market_data_yf_stats.json")
    print(f"Saved: {out_path}")
    print(stats)


if __name__ == "__main__":
    main()
