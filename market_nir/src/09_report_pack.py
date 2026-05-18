#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from common import ensure_dir, read_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect artifacts into report pack")
    p.add_argument("--artifacts", default="market_nir/artifacts")
    p.add_argument("--out-dir", default="market_nir/artifacts/report_pack")
    return p.parse_args()


def try_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    args = parse_args()
    art = Path(args.artifacts)
    out = ensure_dir(args.out_dir)
    figures_dir = ensure_dir(Path(out) / "figures")
    tables_dir = ensure_dir(Path(out) / "tables")

    figure_candidates = [
        art / "plots" / "confusion_baseline_tfidf_lr_test.png",
        art / "plots" / "confusion_distilbert_test.png",
        art / "plots" / "calibration_baseline_tfidf_lr_up_test.png",
        art / "plots" / "calibration_distilbert_up_test.png",
        art / "plots" / "equity_baseline_tfidf_lr_test.png",
        art / "plots" / "equity_distilbert_test.png",
        art / "plots" / "drawdown_baseline_tfidf_lr_test.png",
        art / "plots" / "drawdown_distilbert_test.png",
    ]

    copied_figs = []
    for src in figure_candidates:
        dst = figures_dir / src.name
        if try_copy(src, dst):
            copied_figs.append(dst.name)

    table_candidates = [
        art / "metrics" / "model_comparison_table.csv",
        art / "metrics" / "ablation_summary.csv",
        art / "metrics" / "backtest_baseline_tfidf_lr_test.json",
        art / "metrics" / "backtest_distilbert_test.json",
        art / "metrics" / "baseline_tfidf_lr_metrics.json",
        art / "metrics" / "distilbert_metrics.json",
        art / "metrics" / "03_split_stats.json",
    ]

    copied_tables = []
    for src in table_candidates:
        dst = tables_dir / src.name
        if try_copy(src, dst):
            copied_tables.append(dst.name)

    # Build markdown summary
    lines = ["# Report Pack", "", "## Included Figures", ""]
    for f in copied_figs:
        lines.append(f"- `{f}`")

    lines.extend(["", "## Included Tables/JSON", ""])
    for t in copied_tables:
        lines.append(f"- `{t}`")

    lines.extend(["", "## Quick Highlights", ""])

    cmp_path = art / "metrics" / "model_comparison_table.csv"
    if cmp_path.exists():
        cmp = pd.read_csv(cmp_path)
        test = cmp[cmp["split"] == "test"].sort_values("macro_f1", ascending=False)
        if len(test) > 0:
            top = test.iloc[0]
            lines.append(
                f"- Best test macro F1: `{top['model']}` = `{top['macro_f1']:.4f}` (balanced_acc={top['balanced_accuracy']:.4f})"
            )

    bt_dist_path = art / "metrics" / "backtest_distilbert_test.json"
    bt_base_path = art / "metrics" / "backtest_baseline_tfidf_lr_test.json"
    if bt_dist_path.exists():
        d = read_json(bt_dist_path)
        lines.append(
            f"- DistilBERT backtest(test): cum_return={d['cum_return']:.6f}, sharpe={d['event_sharpe']:.4f}, mdd={d['max_drawdown']:.6f}"
        )
    if bt_base_path.exists():
        b = read_json(bt_base_path)
        lines.append(
            f"- Baseline backtest(test): cum_return={b['cum_return']:.6f}, sharpe={b['event_sharpe']:.4f}, mdd={b['max_drawdown']:.6f}"
        )

    md_path = Path(out) / "REPORT_PACK_SUMMARY.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved report pack: {out}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
