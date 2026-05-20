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
    p.add_argument(
        "--tau-quantile",
        type=float,
        default=None,
        help="If set, tau is inferred as quantile of abs(score), e.g. 0.9 for top-10%% signals",
    )
    p.add_argument(
        "--tune-on-split",
        choices=["none", "train", "val", "test"],
        default="none",
        help="Tune tau on this split, then evaluate on --split",
    )
    p.add_argument(
        "--tune-objective",
        choices=["cum_return", "event_sharpe"],
        default="event_sharpe",
        help="Objective for tau tuning",
    )
    p.add_argument(
        "--tune-quantiles",
        default="0.70,0.75,0.80,0.85,0.90,0.92,0.94,0.95,0.96,0.97,0.98,0.99",
        help="Comma-separated quantiles of abs(score) used as tau candidates for tuning",
    )
    p.add_argument(
        "--invert-signal",
        action="store_true",
        help="Invert trading direction: long<->short for all non-zero signals",
    )
    p.add_argument("--cost", type=float, default=0.0005, help="cost per trade")
    p.add_argument("--out-dir", default="market_nir/artifacts")
    return p.parse_args()


def max_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks)
    return float(dd.min())


def run_strategy(score: np.ndarray, ret: np.ndarray, tau: float, cost: float, invert_signal: bool = False):
    signal = np.where(score > tau, 1.0, np.where(score < -tau, -1.0, 0.0))
    if invert_signal:
        signal = -signal
    pnl = signal * ret - cost * np.abs(signal)
    equity = np.cumsum(pnl)
    traded = np.abs(signal) > 0
    hit_rate = float((np.sign(signal[traded]) == np.sign(ret[traded])).mean()) if traded.any() else 0.0
    sharpe = float(np.mean(pnl) / (np.std(pnl) + 1e-12) * np.sqrt(max(len(pnl), 1)))
    return {
        "signal": signal,
        "pnl": pnl,
        "equity": equity,
        "traded": traded,
        "hit_rate": hit_rate,
        "sharpe": sharpe,
    }


def parse_quantiles(spec: str) -> list[float]:
    vals = []
    for raw in spec.split(","):
        x = raw.strip()
        if not x:
            continue
        q = float(x)
        if not (0.0 < q < 1.0):
            raise ValueError(f"Bad quantile {q}, expected in (0,1)")
        vals.append(q)
    if not vals:
        raise ValueError("No quantiles provided")
    return sorted(set(vals))


def main() -> None:
    args = parse_args()
    df = load_table(args.predictions).copy()
    model_name = str(df["model"].iloc[0]) if "model" in df.columns else Path(args.predictions).stem

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    eval_df = df[df["split"] == args.split].copy()
    if len(eval_df) == 0:
        raise SystemExit("No rows for requested split")

    score = eval_df["prob_UP"].values - eval_df["prob_DOWN"].values
    tau = float(args.tau)
    tau_source = "manual_tau"
    tune_rows = []

    if args.tune_on_split != "none":
        tune_df = df[df["split"] == args.tune_on_split].copy()
        if len(tune_df) == 0:
            raise SystemExit(f"No rows for tune split: {args.tune_on_split}")
        tune_score = tune_df["prob_UP"].values - tune_df["prob_DOWN"].values
        tune_ret = tune_df["ret_h"].values.astype(float)

        q_list = parse_quantiles(args.tune_quantiles)
        best_obj = -np.inf
        best_tau = None
        best_q = None

        for q in q_list:
            tau_q = float(np.quantile(np.abs(tune_score), q))
            r = run_strategy(
                tune_score,
                tune_ret,
                tau=tau_q,
                cost=args.cost,
                invert_signal=args.invert_signal,
            )
            obj = float(r["equity"][-1]) if args.tune_objective == "cum_return" else float(r["sharpe"])
            tune_rows.append(
                {
                    "quantile": q,
                    "tau": tau_q,
                    "objective": obj,
                    "cum_return": float(r["equity"][-1]),
                    "event_sharpe": float(r["sharpe"]),
                    "trades": int(r["traded"].sum()),
                    "turnover": float(r["traded"].mean()),
                    "hit_rate": float(r["hit_rate"]),
                }
            )
            if obj > best_obj:
                best_obj = obj
                best_tau = tau_q
                best_q = q

        tau = float(best_tau)
        tau_source = f"tuned_on_{args.tune_on_split}_q{best_q:.3f}_{args.tune_objective}"
        print(
            f"[info] tau tuned on split={args.tune_on_split} "
            f"objective={args.tune_objective}: q={best_q:.3f}, tau={tau:.6f}"
        )
    elif args.tau_quantile is not None:
        q = float(args.tau_quantile)
        if not (0.0 < q < 1.0):
            raise SystemExit("--tau-quantile must be in (0, 1)")
        tau = float(np.quantile(np.abs(score), q))
        tau_source = f"eval_split_quantile_q{q:.3f}"
        print(f"[info] tau inferred from |score| quantile q={q:.3f}: tau={tau:.6f}")

    ret = eval_df["ret_h"].values.astype(float)
    rs = run_strategy(
        score=score,
        ret=ret,
        tau=tau,
        cost=args.cost,
        invert_signal=args.invert_signal,
    )
    signal = rs["signal"]
    pnl = rs["pnl"]
    equity = rs["equity"]
    traded = rs["traded"]
    hit_rate = rs["hit_rate"]
    sharpe = rs["sharpe"]

    stats = {
        "model": model_name,
        "split": args.split,
        "tau": tau,
        "tau_input": float(args.tau),
        "tau_quantile": None if args.tau_quantile is None else float(args.tau_quantile),
        "tau_source": tau_source,
        "tune_on_split": None if args.tune_on_split == "none" else args.tune_on_split,
        "tune_objective": None if args.tune_on_split == "none" else args.tune_objective,
        "invert_signal": bool(args.invert_signal),
        "cost": float(args.cost),
        "rows": int(len(eval_df)),
        "trades": int(traded.sum()),
        "turnover": float(traded.mean()),
        "score_min": float(score.min()),
        "score_max": float(score.max()),
        "score_abs_q90": float(np.quantile(np.abs(score), 0.9)),
        "score_abs_q95": float(np.quantile(np.abs(score), 0.95)),
        "score_abs_q99": float(np.quantile(np.abs(score), 0.99)),
        "mean_pnl": float(np.mean(pnl)),
        "cum_return": float(equity[-1]),
        "event_sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "hit_rate": hit_rate,
    }

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "plots")

    bt_df = eval_df[["event_id", "timestamp_utc", "ticker", "y_true", "y_pred", "ret_h"]].copy()
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

    if tune_rows:
        tune_df_out = pd.DataFrame(tune_rows).sort_values("objective", ascending=False).reset_index(drop=True)
        tune_df_out.to_csv(out_dir / "metrics" / f"tau_tuning_{model_name}_{args.tune_on_split}.csv", index=False)
        write_json(
            {
                "model": model_name,
                "eval_split": args.split,
                "tune_split": args.tune_on_split,
                "objective": args.tune_objective,
                "best_tau": tau,
                "candidates": tune_rows,
            },
            out_dir / "metrics" / f"tau_tuning_{model_name}_{args.tune_on_split}.json",
        )

    if int(traded.sum()) == 0:
        print(
            "[warning] No trades were generated. "
            "Try lower --tau or use --tau-quantile (e.g. 0.9 or 0.95)."
        )

    print(stats)


if __name__ == "__main__":
    main()
