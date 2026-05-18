# Colab Playbook: обучение DistilBERT и выгрузка модели

Этот сценарий заточен под текущий пайплайн `market_nir/src/05_train_distilbert.py`.

## Вариант A (рекомендуется)
Локально готовим `labeled_events_split.parquet`, в Colab делаем только обучение.

## Вариант B
Грузим raw CSV в Colab и запускаем `01 -> 02 -> 03 -> 05` полностью в Colab.

---

## 0) Создай новый Colab и включи GPU
`Runtime -> Change runtime type -> T4 GPU`.

---

## 1) Установка зависимостей
```python
!pip -q install "torch>=2.2,<3.0" "transformers>=4.40,<5.0" "datasets>=2.18,<4.0" \
               "pandas>=2.0,<3.0" "pyarrow>=15,<22" "scikit-learn>=1.4,<2.0"
```

---

## 2) Клонируем репозиторий
Если репо приватное, сначала подтяни его удобным для тебя способом (token/drive upload).

```python
%cd /content
!git clone <PUT_YOUR_REPO_URL_HERE> repo
%cd /content/repo
```

Если не хочешь `git clone`, просто загрузи папку `market_nir` в `/content/repo` вручную.

---

## 3A) (Рекомендуется) Загрузить готовый split-файл
Локально подготовь:
- `market_nir/data/processed/labeled_events_split.parquet`

Загрузи в Colab:
```python
from google.colab import files
uploaded = files.upload()  # выбери labeled_events_split.parquet
```

Перемести файл в ожидаемую папку:
```python
import os, shutil
os.makedirs('/content/repo/market_nir/data/processed', exist_ok=True)
shutil.move('/content/labeled_events_split.parquet', '/content/repo/market_nir/data/processed/labeled_events_split.parquet')
```

---

## 3B) (Опционально) Полная подготовка данных в Colab
Если хочешь прогнать весь препроцесс в Colab, загрузи:
- `text_events.csv`
- `market_bars.csv`

И выполни:
```python
!python /content/repo/market_nir/src/01_prepare_events.py \
  --input /content/repo/market_nir/data/raw/text_events.csv

!python /content/repo/market_nir/src/02_label_events.py \
  --market /content/repo/market_nir/data/raw/market_bars.csv \
  --horizon 2h --k 1.0

!python /content/repo/market_nir/src/03_split_time_purged.py --horizon 2h
```

---

## 4) Запуск обучения DistilBERT в Colab
```python
!python /content/repo/market_nir/src/05_train_distilbert.py \
  --input /content/repo/market_nir/data/processed/labeled_events_split.parquet \
  --model-name distilbert-base-uncased \
  --epochs 3 \
  --batch-size 16 \
  --max-len 128 \
  --lr 2e-5
```

Для ускорения можешь попробовать:
- `--batch-size 24` (если влезает в VRAM)
- `--epochs 4..5` для более стабильной метрики

---

## 5) Проверка, что модель обучилась
```python
!ls -lah /content/repo/market_nir/artifacts/models/distilbert_market
!ls -lah /content/repo/market_nir/artifacts/metrics
```

Должны быть:
- `market_nir/artifacts/models/distilbert_market/*`
- `market_nir/artifacts/predictions/distilbert_predictions.parquet`
- `market_nir/artifacts/metrics/distilbert_metrics.json`

---

## 6) Сжать и скачать обученную модель
```python
!cd /content/repo/market_nir/artifacts/models && zip -r /content/distilbert_market.zip distilbert_market
from google.colab import files
files.download('/content/distilbert_market.zip')
```

---

## 7) (Опционально) Сохранить в Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/distilbert_market.zip /content/drive/MyDrive/
```

---

## 8) Что делать локально после скачивания
1. Распакуй архив в:
`/Users/konstantin/Documents/rtu_mirea/pions/market_nir/models/distilbert_market`

2. Дальше запускай локально:
```bash
./venv/bin/python /Users/konstantin/Documents/rtu_mirea/pions/market_nir/src/06_eval_ml.py \
  --predictions /Users/konstantin/Documents/rtu_mirea/pions/market_nir/artifacts/predictions/baseline_tfidf_lr_predictions.parquet \
               /Users/konstantin/Documents/rtu_mirea/pions/market_nir/artifacts/predictions/distilbert_predictions.parquet

./venv/bin/python /Users/konstantin/Documents/rtu_mirea/pions/market_nir/src/07_backtest_event.py \
  --predictions /Users/konstantin/Documents/rtu_mirea/pions/market_nir/artifacts/predictions/distilbert_predictions.parquet

./venv/bin/python /Users/konstantin/Documents/rtu_mirea/pions/market_nir/src/09_report_pack.py
```

---

## Быстрый чек-лист ошибок
- `No module named ...` -> не выполнилась ячейка установки зависимостей.
- `File not found labeled_events_split.parquet` -> файл загружен не туда.
- `CUDA out of memory` -> уменьшить `--batch-size` (например, до 8/12).
- Метрики плохие -> увеличить эпохи и проверить баланс классов/порог `k`/горизонт `h`.
