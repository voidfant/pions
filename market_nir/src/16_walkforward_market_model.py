#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import LABEL_ORDER, classify_metrics, ensure_dir, load_table, parse_utc, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward training/evaluation for market-only model")
    p.add_argument("--input", default="market_nir/data/processed/market_only_dataset.parquet")
    p.add_argument("--out-dir", default="market_nir/artifacts")

    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--step-days", type=int, default=30)
    p.add_argument("--gap-hours", type=int, default=0)
    p.add_argument("--min-train-rows", type=int, default=3000)
    p.add_argument("--min-test-rows", type=int, default=300)
    p.add_argument(
        "--calibration-frac",
        type=float,
        default=0.2,
        help="Tail fraction of each training window used to calibrate signal direction before testing.",
    )
    p.add_argument("--min-calibration-rows", type=int, default=300)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model-type",
        choices=["classifier", "return_regressor"],
        default="classifier",
        help="classifier predicts UP/DOWN/FLAT; return_regressor predicts future return and converts it to probabilities.",
    )
    p.add_argument("--max-iter", type=int, default=600)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--min-samples-leaf", type=int, default=60)
    p.add_argument("--l2", type=float, default=0.8)
    p.add_argument("--class-balance", choices=["none", "balanced"], default="balanced")
    p.add_argument(
        "--return-weight",
        type=float,
        default=1.0,
        help="Extra sample-weight strength for larger absolute future returns in return_regressor mode.",
    )
    p.add_argument(
        "--return-weight-cap",
        type=float,
        default=5.0,
        help="Maximum return-based sample weight multiplier.",
    )
    return p.parse_args()


def class_weight_vector(y: np.ndarray) -> dict[str, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = float(counts.sum())
    k = float(len(classes))
    return {c: total / (k * float(cnt)) for c, cnt in zip(classes, counts)}


def sample_weights(y: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    wmap = class_weight_vector(y)
    return np.array([wmap[v] for v in y], dtype=float)


def build_classifier(args: argparse.Namespace, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2,
        random_state=seed,
    )


def build_regressor(args: argparse.Namespace, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2,
        random_state=seed,
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def return_sample_weights(ret_h: np.ndarray, args: argparse.Namespace) -> np.ndarray | None:
    if args.return_weight <= 0:
        return None
    scale = float(np.nanmedian(np.abs(ret_h))) + 1e-12
    w = 1.0 + args.return_weight * (np.abs(ret_h) / scale)
    return np.clip(w, 1.0, max(1.0, float(args.return_weight_cap))).astype(float)


def flat_threshold_from_train(y_label: np.ndarray, ret_h: np.ndarray) -> float:
    y_label = y_label.astype(str)
    ret_abs = np.abs(ret_h.astype(float))
    if np.any(y_label == "FLAT"):
        flat_abs = ret_abs[y_label == "FLAT"]
        if len(flat_abs) > 0:
            return float(np.quantile(flat_abs, 0.90))
    return 0.0


def probabilities_from_predicted_return(
    pred_ret: np.ndarray,
    train_ret: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_ret = pred_ret.astype(float)
    scale = float(np.quantile(np.abs(train_ret.astype(float)), 0.75)) + 1e-12
    directional_up = sigmoid(pred_ret / scale)

    has_flat = np.any(y_train.astype(str) == "FLAT")
    if has_flat:
        flat_thr = flat_threshold_from_train(y_train, train_ret)
        action_scale = max(scale * 0.5, flat_thr * 0.5, 1e-12)
        action_prob = sigmoid((np.abs(pred_ret) - flat_thr) / action_scale)
    else:
        flat_thr = 0.0
        action_prob = np.ones_like(pred_ret, dtype=float)

    prob_up = action_prob * directional_up
    prob_down = action_prob * (1.0 - directional_up)
    prob_flat = 1.0 - action_prob

    probs = np.column_stack([prob_down, prob_flat, prob_up])
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)

    labels = np.full(len(pred_ret), "FLAT", dtype=object)
    labels[pred_ret > flat_thr] = "UP"
    labels[pred_ret < -flat_thr] = "DOWN"
    return labels, probs


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {}
    corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_true) > 2 else float("nan")
    sign_hit = float((np.sign(y_pred) == np.sign(y_true)).mean())
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "corr_pred_ret_h": corr,
        "sign_hit_rate": sign_hit,
    }


def score_metrics(ret_h: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if len(ret_h) == 0:
        return {}
    corr = float(np.corrcoef(score, ret_h)[0, 1]) if len(ret_h) > 2 else float("nan")
    sign_hit = float((np.sign(score) == np.sign(ret_h)).mean())
    return {
        "corr_score_ret_h": corr,
        "score_sign_hit_rate": sign_hit,
    }


def strategy_cum_return(score: np.ndarray, ret_h: np.ndarray, quantile: float = 0.90) -> tuple[float, float, int]:
    if len(score) == 0:
        return 0.0, 0.0, 0
    abs_score = np.abs(score)
    if float(abs_score.max()) <= 0:
        return 0.0, 0.0, 0
    tau = float(np.quantile(abs_score, quantile))
    signal = np.where(score > tau, 1.0, np.where(score < -tau, -1.0, 0.0))
    pnl = signal * ret_h
    return float(pnl.sum()), tau, int(np.abs(signal).sum())


def choose_signal_direction(score: np.ndarray, ret_h: np.ndarray) -> tuple[float, dict[str, float]]:
    normal_return, tau, trades = strategy_cum_return(score, ret_h)
    inverted_return, _, inverted_trades = strategy_cum_return(-score, ret_h)
    corr = float(np.corrcoef(score, ret_h)[0, 1]) if len(score) > 2 else float("nan")
    direction = -1.0 if inverted_return > normal_return else 1.0
    return direction, {
        "calibration_corr": corr,
        "calibration_tau_q90": float(tau),
        "calibration_trades": int(max(trades, inverted_trades)),
        "calibration_return_normal": float(normal_return),
        "calibration_return_inverted": float(inverted_return),
        "signal_direction": float(direction),
    }


def maybe_split_calibration(
    tr: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frac = float(args.calibration_frac)
    if frac <= 0.0:
        return tr, tr.iloc[0:0].copy()
    frac = min(frac, 0.45)
    cal_td = (train_end - train_start) * frac
    cal_start = train_end - cal_td
    fit = tr[tr["timestamp_utc"] < cal_start].copy()
    cal = tr[tr["timestamp_utc"] >= cal_start].copy()
    if len(fit) < args.min_train_rows or len(cal) < args.min_calibration_rows:
        return tr, tr.iloc[0:0].copy()
    return fit, cal


def labels_from_probabilities(probs: np.ndarray) -> np.ndarray:
    idx = np.argmax(probs, axis=1)
    return np.array([LABEL_ORDER[int(i)] for i in idx], dtype=object)


def main() -> None:
    args = parse_args()
    df = load_table(args.input).copy()
    df["timestamp_utc"] = parse_utc(df["timestamp_utc"])
    df = df.dropna(subset=["timestamp_utc", "label", "ret_h"]).copy()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    base_cols = {"event_id", "timestamp_utc", "ticker", "ret_h", "label", "split"}
    feature_cols = [c for c in df.columns if c not in base_cols]
    if not feature_cols:
        raise SystemExit("No feature columns found")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols).copy()

    t_min = df["timestamp_utc"].min()
    t_max = df["timestamp_utc"].max()
    if pd.isna(t_min) or pd.isna(t_max):
        raise SystemExit("Bad timestamps")

    train_td = pd.Timedelta(days=args.train_days)
    test_td = pd.Timedelta(days=args.test_days)
    step_td = pd.Timedelta(days=args.step_days)
    gap_td = pd.Timedelta(hours=args.gap_hours)

    eval_start = t_min + train_td + gap_td
    folds = []
    preds = []
    i = 0

    while eval_start + test_td <= t_max:
        train_start = eval_start - gap_td - train_td
        train_end = eval_start - gap_td
        test_end = eval_start + test_td

        tr = df[(df["timestamp_utc"] >= train_start) & (df["timestamp_utc"] < train_end)].copy()
        te = df[(df["timestamp_utc"] >= eval_start) & (df["timestamp_utc"] < test_end)].copy()

        if len(tr) >= args.min_train_rows and len(te) >= args.min_test_rows:
            fit_tr, cal = maybe_split_calibration(tr, train_start, train_end, args)

            x_tr = fit_tr[feature_cols].values
            y_tr = fit_tr["label"].astype(str).values
            x_te = te[feature_cols].values
            signal_direction = 1.0
            calibration_payload: dict[str, float] = {
                "calibration_rows": int(len(cal)),
                "signal_direction": 1.0,
            }

            if args.model_type == "classifier":
                model = build_classifier(args, seed=args.seed + i * 17 + 1)
                sw = sample_weights(y_tr, args.class_balance)
                model.fit(x_tr, y_tr, sample_weight=sw)

                classes = list(model.classes_)
                if len(cal) > 0:
                    cal_prob_raw = model.predict_proba(cal[feature_cols].values)
                    cal_prob = np.zeros((len(cal), len(LABEL_ORDER)), dtype=float)
                    for label in LABEL_ORDER:
                        if label in classes:
                            cal_prob[:, LABEL_ORDER.index(label)] = cal_prob_raw[:, classes.index(label)]
                    cal_score = cal_prob[:, LABEL_ORDER.index("UP")] - cal_prob[:, LABEL_ORDER.index("DOWN")]
                    signal_direction, calibration_payload = choose_signal_direction(
                        cal_score,
                        cal["ret_h"].astype(float).values,
                    )
                    calibration_payload["calibration_rows"] = int(len(cal))

                y_pred = model.predict(x_te)
                y_prob_raw = model.predict_proba(x_te)
                y_prob = np.zeros((len(te), len(LABEL_ORDER)), dtype=float)
                for label in LABEL_ORDER:
                    if label in classes:
                        y_prob[:, LABEL_ORDER.index(label)] = y_prob_raw[:, classes.index(label)]
                if signal_direction < 0:
                    y_prob[:, [LABEL_ORDER.index("DOWN"), LABEL_ORDER.index("UP")]] = y_prob[
                        :, [LABEL_ORDER.index("UP"), LABEL_ORDER.index("DOWN")]
                    ]
                    y_pred = labels_from_probabilities(y_prob)
                pred_ret = np.sum(y_prob * np.array([-1.0, 0.0, 1.0]), axis=1)
            else:
                model = build_regressor(args, seed=args.seed + i * 17 + 1)
                train_ret = fit_tr["ret_h"].astype(float).values
                sw = return_sample_weights(train_ret, args)
                model.fit(x_tr, train_ret, sample_weight=sw)

                if len(cal) > 0:
                    cal_pred_ret = model.predict(cal[feature_cols].values).astype(float)
                    cal_labels, cal_prob = probabilities_from_predicted_return(
                        pred_ret=cal_pred_ret,
                        train_ret=train_ret,
                        y_train=y_tr,
                    )
                    cal_score = cal_prob[:, LABEL_ORDER.index("UP")] - cal_prob[:, LABEL_ORDER.index("DOWN")]
                    signal_direction, calibration_payload = choose_signal_direction(
                        cal_score,
                        cal["ret_h"].astype(float).values,
                    )
                    calibration_payload["calibration_rows"] = int(len(cal))

                pred_ret = model.predict(x_te).astype(float)
                pred_ret = pred_ret * signal_direction
                y_pred, y_prob = probabilities_from_predicted_return(
                    pred_ret=pred_ret,
                    train_ret=train_ret,
                    y_train=y_tr,
                )

            out = te[["event_id", "timestamp_utc", "ticker", "ret_h", "label"]].copy()
            out = out.rename(columns={"label": "y_true"})
            out["y_pred"] = y_pred
            out["split"] = "test"
            out["wf_fold"] = i
            out["pred_ret_h"] = pred_ret
            for j, label in enumerate(LABEL_ORDER):
                out[f"prob_{label}"] = y_prob[:, j]
            out["score"] = out["prob_UP"] - out["prob_DOWN"]
            out["model"] = "market_only_hgb_walkforward"
            preds.append(out)

            folds.append(
                {
                    "fold": i,
                    "train_start": str(train_start),
                    "train_end": str(train_end),
                    "test_start": str(eval_start),
                    "test_end": str(test_end),
                    "train_rows": int(len(tr)),
                    "test_rows": int(len(te)),
                    "fit_rows": int(len(fit_tr)),
                    **calibration_payload,
                    "train_labels": tr["label"].value_counts().to_dict(),
                    "test_labels": te["label"].value_counts().to_dict(),
                }
            )

        eval_start = eval_start + step_td
        i += 1

    if not preds:
        raise SystemExit("No walk-forward folds produced predictions. Relax min rows / window sizes.")

    pred_df = pd.concat(preds, axis=0, ignore_index=True)
    pred_df = pred_df.sort_values("timestamp_utc").drop_duplicates(subset=["event_id"], keep="first").reset_index(drop=True)

    metrics = classify_metrics(pred_df["y_true"], pred_df["y_pred"])
    ret_h_test = pred_df["ret_h"].astype(float).values
    score_test = (pred_df["prob_UP"].astype(float).values - pred_df["prob_DOWN"].astype(float).values)
    signal_metrics = score_metrics(ret_h_test, score_test)
    ret_metrics = (
        regression_metrics(ret_h_test, pred_df["pred_ret_h"].astype(float).values)
        if args.model_type == "return_regressor"
        else None
    )

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "predictions")
    ensure_dir(out_dir / "metrics")

    pred_path = out_dir / "predictions" / "market_only_hgb_walkforward_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)

    payload = {
        "model": "market_only_hgb_walkforward",
        "rows": int(len(pred_df)),
        "n_features": int(len(feature_cols)),
        "model_type": args.model_type,
        "metrics_test_aggregate": metrics,
        "signal_metrics_test_aggregate": signal_metrics,
        "return_metrics_test_aggregate": ret_metrics,
        "params": {
            "model_type": args.model_type,
            "train_days": args.train_days,
            "test_days": args.test_days,
            "step_days": args.step_days,
            "gap_hours": args.gap_hours,
            "calibration_frac": args.calibration_frac,
            "min_calibration_rows": args.min_calibration_rows,
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "l2": args.l2,
            "class_balance": args.class_balance,
            "return_weight": args.return_weight,
            "return_weight_cap": args.return_weight_cap,
        },
        "folds": folds,
    }
    write_json(payload, out_dir / "metrics" / "market_only_hgb_walkforward_metrics.json")
    print(f"Saved predictions: {pred_path}")
    print(payload)


if __name__ == "__main__":
    main()
