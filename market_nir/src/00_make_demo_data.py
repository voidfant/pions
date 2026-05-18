#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create synthetic demo data for pipeline smoke-test")
    p.add_argument("--out-dir", default="market_nir/data/raw")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-events", type=int, default=5000)
    p.add_argument("--freq", default="5min")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = ensure_dir(args.out_dir)

    tickers = ["BTCUSDT", "ETHUSDT"]
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    n_bars = 12000
    dt = pd.to_timedelta(args.freq)

    bars_rows = []
    for t in tickers:
        price = 100.0 if t == "BTCUSDT" else 60.0
        for i in range(n_bars):
            ts = start + i * dt
            noise = rng.normal(0, 0.0015)
            drift = 0.00002 * np.sin(i / 400)
            ret = drift + noise
            new_price = max(1e-3, price * (1 + ret))
            high = max(price, new_price) * (1 + abs(rng.normal(0, 0.0006)))
            low = min(price, new_price) * (1 - abs(rng.normal(0, 0.0006)))
            bars_rows.append((ts, t, price, high, low, new_price, abs(rng.normal(1000, 200))))
            price = new_price

    bars = pd.DataFrame(
        bars_rows,
        columns=["timestamp_utc", "ticker", "open", "high", "low", "close", "volume"],
    )

    # events correlated with next-step sign to make learnable synthetic setup
    event_rows = []
    keywords_up = ["strong demand", "buy pressure", "bullish setup", "breakout", "positive momentum"]
    keywords_down = ["sell pressure", "risk off", "bearish setup", "breakdown", "negative momentum"]
    neutral = ["range", "mixed signals", "sideways", "uncertain", "flat move"]

    for i in range(args.n_events):
        t = tickers[i % len(tickers)]
        # random event time but avoid first/last windows
        bar_idx = int(rng.integers(100, n_bars - 100))
        ts = start + bar_idx * dt

        # synthetic latent sentiment
        latent = rng.normal(0, 1)
        if latent > 0.45:
            text = f"{t} {rng.choice(keywords_up)} market outlook"
        elif latent < -0.45:
            text = f"{t} {rng.choice(keywords_down)} market outlook"
        else:
            text = f"{t} {rng.choice(neutral)} market outlook"

        event_rows.append((f"evt_{i:06d}", ts, t, "demo_feed", "en", text))

    events = pd.DataFrame(event_rows, columns=["event_id", "timestamp_utc", "ticker", "source", "lang", "text"])

    events.to_csv(Path(out) / "text_events.csv", index=False)
    bars.to_csv(Path(out) / "market_bars.csv", index=False)

    print(f"Saved: {Path(out) / 'text_events.csv'}")
    print(f"Saved: {Path(out) / 'market_bars.csv'}")


if __name__ == "__main__":
    main()
