#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

from common import ensure_dir, read_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect artifacts into report pack")
    p.add_argument("--artifacts", default="market_nir/artifacts")
    p.add_argument("--out-dir", default="market_nir/artifacts/report_pack")
    p.add_argument(
        "--mode",
        choices=["auto", "market_only", "all"],
        default="auto",
        help="auto: prefer market_only models if present",
    )
    return p.parse_args()


def try_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def discover_models(art: Path, mode: str) -> list[str]:
    models = set()

    # 1) predictions
    pred_dir = art / "predictions"
    if pred_dir.exists():
        for p in pred_dir.glob("*_predictions.parquet"):
            stem = p.stem
            if stem.endswith("_predictions"):
                models.add(stem[: -len("_predictions")])

    # 2) metrics and plots (fallback)
    for d in [art / "metrics", art / "plots"]:
        if not d.exists():
            continue
        for p in d.glob("*"):
            if p.is_dir():
                continue
            m = infer_model_from_filename(p.name)
            if m:
                models.add(m)

    if not models:
        return []

    if mode == "market_only":
        return sorted([m for m in models if m.startswith("market_only_")])
    if mode == "all":
        return sorted(models)

    # auto
    mo = sorted([m for m in models if m.startswith("market_only_")])
    return mo if mo else sorted(models)


def infer_model_from_filename(name: str) -> str | None:
    patterns = [
        r"^confusion_(.+)_test\.png$",
        r"^calibration_(.+)_up_test\.png$",
        r"^equity_(.+)_test\.png$",
        r"^drawdown_(.+)_test\.png$",
        r"^mc_hist_cum_return_(.+)_test\.png$",
        r"^mc_hist_sharpe_(.+)_test\.png$",
        r"^pred_vs_real_overlay_(.+)_test\.png$",
        r"^pred_vs_real_scatter_(.+)_test\.png$",
        r"^backtest_(.+)_test\.json$",
        r"^backtest_(.+)_test\.parquet$",
        r"^monte_carlo_(.+)_test\.json$",
        r"^monte_carlo_(.+)_test\.parquet$",
        r"^pred_vs_real_(.+)_test\.parquet$",
        r"^(.+)_metrics\.json$",
    ]
    for pat in patterns:
        m = re.match(pat, name)
        if m:
            return m.group(1)
    return None


def main() -> None:
    args = parse_args()
    art = Path(args.artifacts)
    out = ensure_dir(args.out_dir)
    figures_dir = ensure_dir(Path(out) / "figures")
    tables_dir = ensure_dir(Path(out) / "tables")

    active_models = discover_models(art, args.mode)
    active_set = set(active_models)

    figure_candidates = []
    if (art / "plots").exists():
        for src in sorted((art / "plots").glob("*.png")):
            model = infer_model_from_filename(src.name)
            if model is None or not active_set or model in active_set:
                figure_candidates.append(src)

    copied_figs = []
    for src in figure_candidates:
        dst = figures_dir / src.name
        if try_copy(src, dst):
            copied_figs.append(dst.name)

    table_candidates = []
    global_allow = {
        "model_comparison_table.csv",
        "model_comparison_metrics.json",
        "ablation_summary.csv",
        "03_split_stats.json",
        "13_market_only_dataset_stats.json",
        "15_download_market_data_yf_stats.json",
    }
    if (art / "metrics").exists():
        for src in sorted((art / "metrics").glob("*")):
            if src.is_dir():
                continue
            if src.suffix.lower() not in {".json", ".csv", ".parquet"}:
                continue
            model = infer_model_from_filename(src.name)
            if model is None:
                if src.name in global_allow:
                    table_candidates.append(src)
                continue
            if not active_set or model in active_set:
                table_candidates.append(src)

    copied_tables = []
    for src in table_candidates:
        dst = tables_dir / src.name
        if try_copy(src, dst):
            copied_tables.append(dst.name)

    # Build markdown summary
    lines = ["# Report Pack", ""]
    if active_models:
        lines.append("## Active Models")
        lines.append("")
        for m in active_models:
            lines.append(f"- `{m}`")
        lines.append("")

    lines.extend(["## Included Figures", ""])
    for f in copied_figs:
        lines.append(f"- `{f}`")

    lines.extend(["", "## Included Tables/JSON", ""])
    for t in copied_tables:
        lines.append(f"- `{t}`")

    lines.extend(["", "## Quick Highlights", ""])

    cmp_path = art / "metrics" / "model_comparison_table.csv"
    if cmp_path.exists():
        cmp = pd.read_csv(cmp_path)
        if active_models and "model" in cmp.columns:
            cmp = cmp[cmp["model"].isin(active_models)]
        test = cmp[cmp["split"] == "test"].sort_values("macro_f1", ascending=False)
        if len(test) > 0:
            top = test.iloc[0]
            lines.append(
                f"- Best test macro F1: `{top['model']}` = `{top['macro_f1']:.4f}` (balanced_acc={top['balanced_accuracy']:.4f})"
            )

    for model in active_models:
        bt_path = art / "metrics" / f"backtest_{model}_test.json"
        if bt_path.exists():
            d = read_json(bt_path)
            lines.append(
                f"- {model} backtest(test): cum_return={d['cum_return']:.6f}, sharpe={d['event_sharpe']:.4f}, mdd={d['max_drawdown']:.6f}"
            )
        mc_path = art / "metrics" / f"monte_carlo_{model}_test.json"
        if mc_path.exists():
            m = read_json(mc_path)
            lines.append(
                f"- {model} Monte Carlo p-values: cum_return_p={m['p_values']['cum_return_right_tail']:.4f}, sharpe_p={m['p_values']['sharpe_right_tail']:.4f}"
            )

    md_path = Path(out) / "REPORT_PACK_SUMMARY.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved report pack: {out}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
