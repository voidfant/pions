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
./venv/bin/python market_nir/src/09_report_pack.py
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
- `09_report_pack.py`: collect metrics and plots for report integration

## Input contracts

- Text events: `event_id,timestamp_utc,ticker,source,lang,text`
- Market bars: `timestamp_utc,ticker,open,high,low,close,volume`
