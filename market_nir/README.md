# market_nir

DistilBERT research pipeline for market text events.

## Quick start (demo data)

Run from repo root.

```bash
./venv/bin/python market_nir/src/00_make_demo_data.py
./venv/bin/python market_nir/src/01_prepare_events.py --input market_nir/data/raw/text_events.csv
./venv/bin/python market_nir/src/02_label_events.py --market market_nir/data/raw/market_bars.csv --horizon 2h --k 1.0
./venv/bin/python market_nir/src/03_split_time_purged.py --horizon 2h
./venv/bin/python market_nir/src/04_train_baseline_tfidf.py
./venv/bin/python market_nir/src/05_train_distilbert.py --epochs 1 --batch-size 16
./venv/bin/python market_nir/src/06_eval_ml.py
./venv/bin/python market_nir/src/07_backtest_event.py --predictions market_nir/artifacts/predictions/baseline_tfidf_lr_predictions.parquet
./venv/bin/python market_nir/src/07_backtest_event.py --predictions market_nir/artifacts/predictions/distilbert_predictions.parquet
./venv/bin/python market_nir/src/10_monte_carlo_test.py --predictions market_nir/artifacts/predictions/distilbert_predictions.parquet --split test --n-runs 3000
./venv/bin/python market_nir/src/09_report_pack.py
```

## Real multi-GB dataset (GDELT)

### 1) Download about 2+ GB of news-event dumps

```bash
./venv/bin/python market_nir/src/11_download_gdelt.py \
  --dataset gkg \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --target-gb 2.0 \
  --extract
```

- Raw archives: `market_nir/data/raw/gdelt/zips`
- Extracted TSVs: `market_nir/data/raw/gdelt/extracted`
- Manifest: `market_nir/data/raw/gdelt/manifest_gkg.csv`

### 2) Build `text_events.csv` from GDELT GKG

```bash
./venv/bin/python market_nir/src/12_build_gdelt_events.py \
  --input-dir market_nir/data/raw/gdelt/extracted \
  --output market_nir/data/raw/text_events_gdelt.csv \
  --ticker SPY \
  --lang en
```

### 3) Run the existing market_nir pipeline on this dataset

```bash
./venv/bin/python market_nir/src/01_prepare_events.py --input market_nir/data/raw/text_events_gdelt.csv
./venv/bin/python market_nir/src/02_label_events.py --market market_nir/data/raw/market_bars.csv --horizon 2h --k 1.0
./venv/bin/python market_nir/src/03_split_time_purged.py --horizon 2h
./venv/bin/python market_nir/src/04_train_baseline_tfidf.py
./venv/bin/python market_nir/src/05_train_distilbert.py --epochs 6 --batch-size 16 --patience 2
```

## Market-only pipeline (no news, OHLCV only)

```bash
./venv/bin/python market_nir/src/13_build_market_only_dataset.py \
  --market market_nir/data/raw/market_bars.csv \
  --horizon 4h \
  --k 0.5 \
  --vol-window 48

./venv/bin/python market_nir/src/14_train_market_model.py --model-type hgb

./venv/bin/python market_nir/src/07_backtest_event.py \
  --predictions market_nir/artifacts/predictions/market_only_hgb_predictions.parquet \
  --split test \
  --tau-quantile 0.9

./venv/bin/python market_nir/src/10_monte_carlo_test.py \
  --predictions market_nir/artifacts/predictions/market_only_hgb_predictions.parquet \
  --split test \
  --tau 0.0 \
  --n-runs 3000
```

## Main scripts

- `01_prepare_events.py`: basic cleaning and dedup for text events
- `02_label_events.py`: future-return labeling (`UP/FLAT/DOWN`)
- `03_split_time_purged.py`: time-based train/val/test with purge window
- `04_train_baseline_tfidf.py`: baseline model
- `05_train_distilbert.py`: DistilBERT fine-tuning
- `06_eval_ml.py`: model comparison, confusion/calibration plots
- `07_backtest_event.py`: event-driven backtest on model signals
- `08_ablation_runner.py`: grid over horizon and threshold multiplier
- `10_monte_carlo_test.py`: Monte Carlo significance test + prediction-vs-real plots
- `09_report_pack.py`: collect metrics and plots for report integration
- `11_download_gdelt.py`: downloader for large GDELT corpora (target by GB)
- `12_build_gdelt_events.py`: convert extracted GDELT GKG files into `text_events.csv`
- `13_build_market_only_dataset.py`: build time-series features and labels from market OHLCV only
- `14_train_market_model.py`: train market-only classifier (`hgb` or `logreg`)

## Input contracts

- Text events: `event_id,timestamp_utc,ticker,source,lang,text`
- Market bars: `timestamp_utc,ticker,open,high,low,close,volume`
