#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_dir, load_table, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Event-driven backtest from model probabilities")
    p.add_argument("--predictions", required=True, help="predictions parquet")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--tau", type=float, default=0.15, help="threshold on score=prob_UP-prob_DOWN")
    p.add_argument("--cost", type=float, default=0.0005, help="cost per trade")
    p.add_argument("--out-dir", default="market_nir/artifacts")
    return p.parse_args()


def max_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks)
    return float(dd.min())


def main() -> None:
    args = parse_args()
    df = load_table(args.predictions).copy()
    model_name = str(df["model"].iloc[0]) if "model" in df.columns else Path(args.predictions).stem

    df = df[df["split"] == args.split].copy()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit("No rows for requested split")

    score = df["prob_UP"].values - df["prob_DOWN"].values
    signal = np.where(score > args.tau, 1.0, np.where(score < -args.tau, -1.0, 0.0))

    ret = df["ret_h"].values.astype(float)
    pnl = signal * ret - args.cost * np.abs(signal)
    equity = np.cumsum(pnl)

    traded = np.abs(signal) > 0
    hit_rate = float((np.sign(signal[traded]) == np.sign(ret[traded])).mean()) if traded.any() else 0.0

    sharpe = float(np.mean(pnl) / (np.std(pnl) + 1e-12) * np.sqrt(max(len(pnl), 1)))

    stats = {
        "model": model_name,
        "split": args.split,
        "tau": float(args.tau),
        "cost": float(args.cost),
        "rows": int(len(df)),
        "trades": int(traded.sum()),
        "turnover": float(traded.mean()),
        "mean_pnl": float(np.mean(pnl)),
        "cum_return": float(equity[-1]),
        "event_sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "hit_rate": hit_rate,
    }

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "plots")

    bt_df = df[["event_id", "timestamp_utc", "ticker", "y_true", "y_pred", "ret_h"]].copy()
    bt_df["score"] = score
    bt_df["signal"] = signal
    bt_df["pnl"] = pnl
    bt_df["equity"] = equity
    bt_df.to_parquet(out_dir / "metrics" / f"backtest_{model_name}_{args.split}.parquet", index=False)

    write_json(stats, out_dir / "metrics" / f"backtest_{model_name}_{args.split}.json")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(equity)
    ax.set_title(f"Equity curve ({model_name}, {args.split})")
    ax.set_xlabel("Event index")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / f"equity_{model_name}_{args.split}.png", dpi=170)
    plt.close(fig)

    dd = equity - np.maximum.accumulate(equity)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(dd, color="#C44E52")
    ax.set_title(f"Drawdown curve ({model_name}, {args.split})")
    ax.set_xlabel("Event index")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / f"drawdown_{model_name}_{args.split}.png", dpi=170)
    plt.close(fig)

    print(stats)


if __name__ == "__main__":
    main()
