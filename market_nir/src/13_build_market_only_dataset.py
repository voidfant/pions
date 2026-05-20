#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import LABEL_ORDER, load_table, parse_utc, save_table, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build powerful market-only feature dataset from OHLCV")
    p.add_argument("--market", default="market_nir/data/raw/market_bars.csv")
    p.add_argument("--output", default="market_nir/data/processed/market_only_dataset.parquet")
    p.add_argument("--horizon", default="4h", help="future horizon for label")
    p.add_argument("--k", type=float, default=0.5, help="threshold multiplier for UP/DOWN labels")
    p.add_argument("--vol-window", type=int, default=48, help="volatility window for dynamic threshold")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--purge", default=None, help="timedelta string, default=horizon")
    p.add_argument("--min-rows-per-ticker", type=int, default=500)
    p.add_argument("--feature-set", choices=["lite", "full"], default="full")
    p.add_argument(
        "--binary-mode",
        choices=["none", "drop_flat"],
        default="none",
        help="none: keep DOWN/FLAT/UP; drop_flat: keep only DOWN/UP",
    )
    return p.parse_args()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _feature_block(mt: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    mt = mt.sort_values("timestamp_utc").copy()

    mt["log_close"] = np.log(np.maximum(mt["close"], 1e-12))
    mt["ret_1"] = mt["close"].pct_change()
    mt["ret_2"] = mt["close"].pct_change(2)
    mt["ret_3"] = mt["close"].pct_change(3)
    mt["ret_6"] = mt["close"].pct_change(6)
    mt["ret_12"] = mt["close"].pct_change(12)
    mt["ret_24"] = mt["close"].pct_change(24)
    mt["log_ret_1"] = mt["log_close"].diff()

    mt["hl_spread"] = (mt["high"] - mt["low"]) / np.maximum(np.abs(mt["close"]), 1e-12)
    mt["oc_ret"] = (mt["close"] - mt["open"]) / np.maximum(np.abs(mt["open"]), 1e-12)
    mt["volume_ret_1"] = mt["volume"].pct_change()
    mt["volume_log"] = np.log1p(np.maximum(mt["volume"], 0.0))

    for lag in (1, 2, 3, 4, 5, 8, 12):
        mt[f"ret_1_lag_{lag}"] = mt["ret_1"].shift(lag)

    for w in (6, 12, 24, 48, 96):
        mp = max(3, w // 4)
        mt[f"ret_mean_{w}"] = mt["ret_1"].rolling(w, min_periods=mp).mean()
        mt[f"ret_std_{w}"] = mt["ret_1"].rolling(w, min_periods=mp).std()
        mt[f"ret_skew_{w}"] = mt["ret_1"].rolling(w, min_periods=mp).skew()
        mt[f"vol_mean_{w}"] = mt["volume"].rolling(w, min_periods=mp).mean()
        mt[f"vol_std_{w}"] = mt["volume"].rolling(w, min_periods=mp).std()

    mt["ewm_ret_fast"] = mt["ret_1"].ewm(span=8, adjust=False).mean()
    mt["ewm_ret_slow"] = mt["ret_1"].ewm(span=34, adjust=False).mean()
    mt["ewm_ret_diff"] = mt["ewm_ret_fast"] - mt["ewm_ret_slow"]

    mt["close_sma_12"] = mt["close"].rolling(12, min_periods=4).mean()
    mt["close_sma_48"] = mt["close"].rolling(48, min_periods=12).mean()
    mt["close_rel_sma_12"] = (mt["close"] / np.maximum(mt["close_sma_12"], 1e-12)) - 1.0
    mt["close_rel_sma_48"] = (mt["close"] / np.maximum(mt["close_sma_48"], 1e-12)) - 1.0
    mt["sma_cross_12_48"] = (mt["close_sma_12"] / np.maximum(mt["close_sma_48"], 1e-12)) - 1.0

    mt["rsi_14"] = rsi(mt["close"], 14) / 100.0
    mt["rsi_28"] = rsi(mt["close"], 28) / 100.0

    mt["atr_14"] = atr(mt["high"], mt["low"], mt["close"], 14)
    mt["atr_rel_14"] = mt["atr_14"] / np.maximum(np.abs(mt["close"]), 1e-12)

    ema12 = mt["close"].ewm(span=12, adjust=False).mean()
    ema26 = mt["close"].ewm(span=26, adjust=False).mean()
    mt["macd_line"] = ema12 - ema26
    mt["macd_signal"] = mt["macd_line"].ewm(span=9, adjust=False).mean()
    mt["macd_hist"] = mt["macd_line"] - mt["macd_signal"]

    bb_mid = mt["close"].rolling(20, min_periods=8).mean()
    bb_std = mt["close"].rolling(20, min_periods=8).std()
    mt["bb_z_20"] = (mt["close"] - bb_mid) / (bb_std + 1e-12)
    mt["bb_width_20"] = (2.0 * bb_std) / np.maximum(np.abs(bb_mid), 1e-12)

    ts = mt["timestamp_utc"].dt
    hour = ts.hour.astype(float)
    dow = ts.dayofweek.astype(float)
    mt["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    mt["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    mt["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    mt["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    base_features = [
        "ret_1",
        "ret_2",
        "ret_3",
        "ret_6",
        "ret_12",
        "ret_24",
        "log_ret_1",
        "hl_spread",
        "oc_ret",
        "volume_ret_1",
        "close_rel_sma_12",
        "close_rel_sma_48",
        "sma_cross_12_48",
        "rsi_14",
        "rsi_28",
        "atr_rel_14",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "bb_z_20",
        "bb_width_20",
        "ewm_ret_fast",
        "ewm_ret_slow",
        "ewm_ret_diff",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    lag_and_roll = [c for c in mt.columns if c.startswith(("ret_1_lag_", "ret_mean_", "ret_std_", "ret_skew_", "vol_mean_", "vol_std_"))]
    if feature_set == "full":
        feature_cols = base_features + lag_and_roll
    else:
        feature_cols = base_features

    feature_cols = [c for c in feature_cols if c in mt.columns]
    return mt, feature_cols


def _label_block(mt: pd.DataFrame, horizon: pd.Timedelta, k: float, vol_window: int) -> pd.DataFrame:
    mt = mt.sort_values("timestamp_utc").copy()
    mt["sigma_1"] = mt["ret_1"].rolling(window=vol_window, min_periods=max(8, vol_window // 4)).std()

    t_arr = mt["timestamp_utc"].values.astype("datetime64[ns]")
    close_arr = mt["close"].values.astype(float)
    sigma_arr = mt["sigma_1"].values.astype(float)
    if len(t_arr) < 10:
        return pd.DataFrame()

    dts = np.diff(t_arr).astype("timedelta64[ns]").astype(np.int64)
    median_dt_ns = int(np.median(dts[dts > 0])) if np.any(dts > 0) else 1
    h_ns = int(horizon.value)
    h_steps = max(1.0, h_ns / max(median_dt_ns, 1))

    idx_now = np.arange(len(mt))
    target_times = t_arr + np.timedelta64(h_ns, "ns")
    idx_fut = np.searchsorted(t_arr, target_times, side="left")
    valid = idx_fut < len(t_arr)

    mt = mt.loc[valid].copy().reset_index(drop=True)
    idx_now = idx_now[valid]
    idx_fut = idx_fut[valid]

    p0 = close_arr[idx_now]
    p1 = close_arr[idx_fut]
    sigma = sigma_arr[idx_now]

    mt["ret_h"] = (p1 - p0) / np.maximum(np.abs(p0), 1e-12)
    mt["threshold"] = k * sigma * np.sqrt(h_steps)
    mt = mt.dropna(subset=["ret_h", "threshold"]).copy()

    cond_up = mt["ret_h"] > mt["threshold"]
    cond_down = mt["ret_h"] < -mt["threshold"]
    mt["label"] = np.where(cond_up, "UP", np.where(cond_down, "DOWN", "FLAT"))
    return mt


def _time_split(df: pd.DataFrame, train_ratio: float, val_ratio: float, purge_td: pd.Timedelta) -> pd.DataFrame:
    out = df.sort_values("timestamp_utc").reset_index(drop=True).copy()
    n = len(out)
    idx1 = int(n * train_ratio)
    idx2 = int(n * (train_ratio + val_ratio))
    idx1 = min(max(idx1, 1), n - 2)
    idx2 = min(max(idx2, idx1 + 1), n - 1)

    t1 = out.loc[idx1, "timestamp_utc"]
    t2 = out.loc[idx2, "timestamp_utc"]

    split = np.full(n, "", dtype=object)
    train_mask = out["timestamp_utc"] < (t1 - purge_td)
    val_mask = (out["timestamp_utc"] >= (t1 + purge_td)) & (out["timestamp_utc"] < (t2 - purge_td))
    test_mask = out["timestamp_utc"] >= (t2 + purge_td)
    split[train_mask.values] = "train"
    split[val_mask.values] = "val"
    split[test_mask.values] = "test"
    out["split"] = split
    out = out[out["split"].isin(["train", "val", "test"])].copy()
    return out


def main() -> None:
    args = parse_args()
    market = load_table(args.market).copy()

    required = ["timestamp_utc", "ticker", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in market.columns]
    if missing:
        raise SystemExit(f"Market missing columns: {missing}")

    market["timestamp_utc"] = parse_utc(market["timestamp_utc"])
    market = market.dropna(subset=required).copy()

    horizon = pd.to_timedelta(args.horizon)
    purge_td = pd.to_timedelta(args.purge) if args.purge else horizon

    chunks = []
    feature_cols_final: list[str] | None = None
    for ticker, g in market.groupby("ticker"):
        if len(g) < args.min_rows_per_ticker:
            continue
        feat, feature_cols = _feature_block(g, feature_set=args.feature_set)
        lab = _label_block(feat, horizon=horizon, k=args.k, vol_window=args.vol_window)
        if len(lab) == 0:
            continue
        if feature_cols_final is None:
            feature_cols_final = feature_cols
        chunks.append(lab)

    if not chunks or not feature_cols_final:
        raise SystemExit("No rows built. Check market coverage and min-rows-per-ticker.")

    df = pd.concat(chunks, axis=0, ignore_index=True)
    df = _time_split(df, train_ratio=args.train_ratio, val_ratio=args.val_ratio, purge_td=purge_td)
    if df["split"].nunique() < 3:
        raise SystemExit("Split failed: not all train/val/test present")

    keep_cols = ["timestamp_utc", "ticker", "ret_h", "label", "split"] + feature_cols_final
    out = df[keep_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=feature_cols_final + ["ret_h", "label", "split"]).reset_index(drop=True)

    if args.binary_mode == "drop_flat":
        out = out[out["label"].isin(["DOWN", "UP"])].copy().reset_index(drop=True)

    if out["split"].nunique() < 3:
        raise SystemExit("Split failed after binary filtering: not all train/val/test present")
    if len(out) < 500:
        raise SystemExit(f"Too few rows after filtering: {len(out)}")

    out["event_id"] = np.array([f"market_{i}" for i in range(len(out))], dtype=object)

    save_table(out, args.output)

    label_dist = out["label"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()
    stats = {
        "rows": int(len(out)),
        "horizon": args.horizon,
        "k": float(args.k),
        "vol_window": int(args.vol_window),
        "feature_set": args.feature_set,
        "binary_mode": args.binary_mode,
        "features": feature_cols_final,
        "n_features": int(len(feature_cols_final)),
        "label_distribution": {k: int(v) for k, v in label_dist.items()},
        "split_counts": {k: int(v) for k, v in out["split"].value_counts().to_dict().items()},
        "time_min": str(out["timestamp_utc"].min()),
        "time_max": str(out["timestamp_utc"].max()),
        "tickers": int(out["ticker"].nunique()),
    }
    write_json(stats, "market_nir/artifacts/metrics/13_market_only_dataset_stats.json")
    print(f"Saved market-only dataset: {args.output}")
    print(stats)


if __name__ == "__main__":
    main()
