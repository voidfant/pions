#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_dir, load_table, parse_utc, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monte Carlo significance test for event-driven strategy")
    p.add_argument("--predictions", required=True, help="predictions parquet path")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--tau", type=float, default=0.15, help="threshold on score = prob_UP - prob_DOWN")
    p.add_argument("--cost", type=float, default=0.0005, help="cost per non-zero trade")
    p.add_argument("--n-runs", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rolling-window", type=int, default=120)
    p.add_argument("--out-dir", default="market_nir/artifacts")
    return p.parse_args()


def max_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    dd = equity - peaks
    return float(dd.min())


def event_sharpe(pnl: np.ndarray) -> float:
    return float(np.mean(pnl) / (np.std(pnl) + 1e-12) * np.sqrt(max(len(pnl), 1)))


def strategy_from_score(score: np.ndarray, ret: np.ndarray, tau: float, cost: float):
    signal = np.where(score > tau, 1.0, np.where(score < -tau, -1.0, 0.0))
    pnl = signal * ret - cost * np.abs(signal)
    equity = np.cumsum(pnl)
    traded = np.abs(signal) > 0
    hit_rate = float((np.sign(signal[traded]) == np.sign(ret[traded])).mean()) if traded.any() else 0.0
    return signal, pnl, equity, hit_rate


def main() -> None:
    args = parse_args()

    df = load_table(args.predictions).copy()
    model_name = str(df["model"].iloc[0]) if "model" in df.columns else Path(args.predictions).stem

    df = df[df["split"] == args.split].copy()
    if len(df) == 0:
        raise SystemExit("No rows for requested split")

    df["timestamp_utc"] = parse_utc(df["timestamp_utc"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    score = (df["prob_UP"].values - df["prob_DOWN"].values).astype(float)
    ret = df["ret_h"].values.astype(float)

    signal, pnl, equity, hit_rate = strategy_from_score(score, ret, tau=args.tau, cost=args.cost)
    observed = {
        "cum_return": float(equity[-1]),
        "event_sharpe": event_sharpe(pnl),
        "max_drawdown": max_drawdown(equity),
        "hit_rate": float(hit_rate),
        "trades": int((np.abs(signal) > 0).sum()),
        "turnover": float((np.abs(signal) > 0).mean()),
    }

    rng = np.random.default_rng(args.seed)
    mc_cum = np.zeros(args.n_runs, dtype=float)
    mc_sharpe = np.zeros(args.n_runs, dtype=float)
    mc_mdd = np.zeros(args.n_runs, dtype=float)

    # Null hypothesis: same trade frequency/signal distribution, randomized in time
    for i in range(args.n_runs):
        sig_rand = rng.permutation(signal)
        pnl_rand = sig_rand * ret - args.cost * np.abs(sig_rand)
        eq_rand = np.cumsum(pnl_rand)
        mc_cum[i] = eq_rand[-1]
        mc_sharpe[i] = event_sharpe(pnl_rand)
        mc_mdd[i] = max_drawdown(eq_rand)

    p_cum = float((mc_cum >= observed["cum_return"]).mean())
    p_sharpe = float((mc_sharpe >= observed["event_sharpe"]).mean())

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "plots")

    mc_stats = {
        "model": model_name,
        "split": args.split,
        "tau": float(args.tau),
        "cost": float(args.cost),
        "n_runs": int(args.n_runs),
        "observed": observed,
        "mc": {
            "cum_return_mean": float(mc_cum.mean()),
            "cum_return_std": float(mc_cum.std()),
            "sharpe_mean": float(mc_sharpe.mean()),
            "sharpe_std": float(mc_sharpe.std()),
            "mdd_mean": float(mc_mdd.mean()),
            "mdd_std": float(mc_mdd.std()),
        },
        "p_values": {
            "cum_return_right_tail": p_cum,
            "sharpe_right_tail": p_sharpe,
        },
    }
    write_json(mc_stats, out_dir / "metrics" / f"monte_carlo_{model_name}_{args.split}.json")

    mc_df = pd.DataFrame(
        {
            "mc_cum_return": mc_cum,
            "mc_sharpe": mc_sharpe,
            "mc_max_drawdown": mc_mdd,
        }
    )
    mc_df.to_parquet(out_dir / "metrics" / f"monte_carlo_{model_name}_{args.split}.parquet", index=False)

    # Histogram: cumulative return
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.hist(mc_cum, bins=50, alpha=0.75)
    ax.axvline(observed["cum_return"], color="#C44E52", linestyle="--", linewidth=2, label="observed")
    ax.set_title(f"Monte Carlo: cumulative return ({model_name}, {args.split})")
    ax.set_xlabel("Cumulative return")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / f"mc_hist_cum_return_{model_name}_{args.split}.png", dpi=170)
    plt.close(fig)

    # Histogram: Sharpe
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.hist(mc_sharpe, bins=50, alpha=0.75)
    ax.axvline(observed["event_sharpe"], color="#C44E52", linestyle="--", linewidth=2, label="observed")
    ax.set_title(f"Monte Carlo: event Sharpe ({model_name}, {args.split})")
    ax.set_xlabel("Event Sharpe")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / f"mc_hist_sharpe_{model_name}_{args.split}.png", dpi=170)
    plt.close(fig)

    # Pred vs real overlay
    roll = max(3, int(args.rolling_window))
    score_s = pd.Series(score)
    ret_s = pd.Series(ret)

    # standardized rolling means for visual comparability
    score_roll = score_s.rolling(roll, min_periods=max(3, roll // 5)).mean()
    ret_roll = ret_s.rolling(roll, min_periods=max(3, roll // 5)).mean()

    def z(x: pd.Series) -> pd.Series:
        return (x - x.mean()) / (x.std() + 1e-12)

    score_z = z(score_roll)
    ret_z = z(ret_roll)

    fig, ax = plt.subplots(figsize=(10.2, 5.1))
    ax.plot(df["timestamp_utc"], score_z, label="Predicted score (rolling z)", alpha=0.9)
    ax.plot(df["timestamp_utc"], ret_z, label="Real ret_h (rolling z)", alpha=0.85)
    ax.set_title(f"Prediction vs real market move ({model_name}, {args.split})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Standardized value")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / f"pred_vs_real_overlay_{model_name}_{args.split}.png", dpi=170)
    plt.close(fig)

    # Scatter score vs ret_h
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.scatter(score, ret, s=12, alpha=0.35)
    coef = np.polyfit(score, ret, 1)
    xx = np.linspace(score.min(), score.max(), 150)
    yy = coef[0] * xx + coef[1]
    corr = float(np.corrcoef(score, ret)[0, 1]) if len(score) > 2 else 0.0
    ax.plot(xx, yy, color="#C44E52", linewidth=2, label=f"lin fit, corr={corr:.3f}")
    ax.set_title(f"Predicted score vs real ret_h ({model_name}, {args.split})")
    ax.set_xlabel("Predicted score (prob_UP - prob_DOWN)")
    ax.set_ylabel("Real future return (ret_h)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / f"pred_vs_real_scatter_{model_name}_{args.split}.png", dpi=170)
    plt.close(fig)

    # Save aligned frame for further reporting
    out_df = df[["event_id", "timestamp_utc", "ticker", "ret_h"]].copy()
    out_df["score"] = score
    out_df["signal"] = signal
    out_df["pnl"] = pnl
    out_df["equity"] = equity
    out_df.to_parquet(out_dir / "metrics" / f"pred_vs_real_{model_name}_{args.split}.parquet", index=False)

    print(mc_stats)


if __name__ == "__main__":
    main()
