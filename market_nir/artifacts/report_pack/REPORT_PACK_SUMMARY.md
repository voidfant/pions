# Report Pack

## Included Figures

- `confusion_baseline_tfidf_lr_test.png`
- `calibration_baseline_tfidf_lr_up_test.png`
- `equity_baseline_tfidf_lr_test.png`
- `drawdown_baseline_tfidf_lr_test.png`
- `mc_hist_cum_return_baseline_tfidf_lr_test.png`
- `mc_hist_sharpe_baseline_tfidf_lr_test.png`
- `pred_vs_real_overlay_baseline_tfidf_lr_test.png`
- `pred_vs_real_scatter_baseline_tfidf_lr_test.png`

## Included Tables/JSON

- `model_comparison_table.csv`
- `ablation_summary.csv`
- `backtest_baseline_tfidf_lr_test.json`
- `monte_carlo_baseline_tfidf_lr_test.json`
- `baseline_tfidf_lr_metrics.json`
- `distilbert_metrics.json`
- `03_split_stats.json`

## Quick Highlights

- Best test macro F1: `baseline_tfidf_lr` = `0.2766` (balanced_acc=0.3519)
- Baseline backtest(test): cum_return=0.029630, sharpe=0.4393, mdd=-0.058301
- Baseline Monte Carlo p-values: cum_return_p=0.1305, sharpe_p=0.1295
