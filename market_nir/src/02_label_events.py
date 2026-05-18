#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import LABEL_ORDER, load_table, parse_utc, save_table, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label text events with future returns")
    p.add_argument("--events", default="market_nir/data/processed/text_events_prepared.parquet")
    p.add_argument("--market", required=True, help="market bars csv/parquet")
    p.add_argument("--output", default="market_nir/data/processed/labeled_events.parquet")
    p.add_argument("--horizon", default="2h", help="pandas timedelta string")
    p.add_argument("--k", type=float, default=1.0, help="threshold multiplier")
    p.add_argument("--vol-window", type=int, default=96)
    return p.parse_args()


def label_one_ticker(events_t: pd.DataFrame, market_t: pd.DataFrame, horizon: pd.Timedelta, k: float) -> pd.DataFrame:
    mt = market_t.sort_values("timestamp_utc").copy()
    mt["ret_1"] = mt["close"].pct_change()
    mt["sigma_1"] = mt["ret_1"].rolling(window=args.vol_window, min_periods=max(5, args.vol_window // 4)).std()

    t_arr = mt["timestamp_utc"].values.astype("datetime64[ns]")
    close_arr = mt["close"].values.astype(float)
    sigma_arr = mt["sigma_1"].values.astype(float)

    if len(t_arr) < 2:
        return pd.DataFrame()

    dts = np.diff(t_arr).astype("timedelta64[ns]").astype(np.int64)
    median_dt_ns = int(np.median(dts[dts > 0])) if np.any(dts > 0) else 1
    h_ns = int(horizon.value)
    h_steps = max(1.0, h_ns / max(median_dt_ns, 1))

    et = events_t.sort_values("timestamp_utc").copy()
    e_arr = et["timestamp_utc"].values.astype("datetime64[ns]")

    idx_now = np.searchsorted(t_arr, e_arr, side="right") - 1
    target_times = e_arr + np.timedelta64(h_ns, "ns")
    idx_fut = np.searchsorted(t_arr, target_times, side="left")

    valid = (idx_now >= 0) & (idx_fut >= 0) & (idx_fut < len(t_arr))
    et = et.loc[valid].copy()
    idx_now = idx_now[valid]
    idx_fut = idx_fut[valid]

    p0 = close_arr[idx_now]
    p1 = close_arr[idx_fut]
    sigma = sigma_arr[idx_now]

    et["price_t"] = p0
    et["price_t_plus_h"] = p1
    et["ret_h"] = (p1 - p0) / np.maximum(np.abs(p0), 1e-12)
    et["sigma_t"] = sigma
    et["threshold"] = k * et["sigma_t"] * np.sqrt(h_steps)

    et = et.dropna(subset=["ret_h", "sigma_t", "threshold"])

    cond_up = et["ret_h"] > et["threshold"]
    cond_down = et["ret_h"] < -et["threshold"]
    et["label"] = np.where(cond_up, "UP", np.where(cond_down, "DOWN", "FLAT"))

    return et


def main() -> None:
    global args
    args = parse_args()

    events = load_table(args.events)
    market = load_table(args.market)

    for req in ["timestamp_utc", "ticker", "text", "event_id"]:
        if req not in events.columns:
            raise SystemExit(f"Events missing column: {req}")
    for req in ["timestamp_utc", "ticker", "close"]:
        if req not in market.columns:
            raise SystemExit(f"Market missing column: {req}")

    events = events.copy()
    market = market.copy()

    events["timestamp_utc"] = parse_utc(events["timestamp_utc"])
    market["timestamp_utc"] = parse_utc(market["timestamp_utc"])

    events = events.dropna(subset=["timestamp_utc", "ticker", "text", "event_id"])
    market = market.dropna(subset=["timestamp_utc", "ticker", "close"])

    horizon = pd.to_timedelta(args.horizon)

    chunks = []
    common_tickers = sorted(set(events["ticker"]) & set(market["ticker"]))
    for ticker in common_tickers:
        e_t = events[events["ticker"] == ticker]
        m_t = market[market["ticker"] == ticker]
        out_t = label_one_ticker(e_t, m_t, horizon, k=args.k)
        if not out_t.empty:
            chunks.append(out_t)

    if not chunks:
        raise SystemExit("No labeled rows produced. Check ticker overlap and timestamp coverage.")

    out = pd.concat(chunks, axis=0).sort_values("timestamp_utc").reset_index(drop=True)
    save_table(out, args.output)

    dist = out["label"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()
    stats = {
        "rows_labeled": int(len(out)),
        "horizon": args.horizon,
        "k": float(args.k),
        "vol_window": int(args.vol_window),
        "label_distribution": {k: int(v) for k, v in dist.items()},
        "ret_h_mean": float(out["ret_h"].mean()),
        "ret_h_std": float(out["ret_h"].std()),
    }
    write_json(stats, "market_nir/artifacts/metrics/02_label_events_stats.json")
    print(f"Saved labeled events: {args.output}")
    print(stats)


if __name__ == "__main__":
    main()
