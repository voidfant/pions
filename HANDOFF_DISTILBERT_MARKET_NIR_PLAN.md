# Handoff Plan: DistilBERT на рыночных данных (НИР-формат)

## 1) Цель
Собрать воспроизводимый исследовательский пайплайн, который проверяет гипотезу:

**текстовые рыночные сообщения содержат сигнал, позволяющий предсказывать будущую доходность на горизонте `h`**.

Результат должен быть пригоден для включения в курсовую как НИР-раздел: методика, эксперименты, графики, ограничения, выводы.

---

## 2) Scope (что делаем)
- Обучаем `DistilBERT` на текстах, привязанных к времени и тикеру.
- Цель классификации: `UP / FLAT / DOWN`.
- Сравниваем с baseline (`TF-IDF + LogisticRegression`).
- Делаем out-of-time оценку + простой event-driven бэктест.
- Готовим артефакты для отчета: таблицы, графики, метрики, абляции.

Не делаем в первой итерации:
- мультимодальные данные (картинки, графики книги заявок);
- сложный execution engine (частичные исполнения, микроструктура L2);
- прод-оптимизации API.

---

## 3) Гипотезы и исследовательские вопросы
- **H1:** DistilBERT превосходит baseline на out-of-time `macro F1`.
- **H2:** Высокоуверенные сигналы DistilBERT дают положительный `Precision@top-k` и лучше случайного в бэктесте.
- **H3:** Устойчивость результатов сохраняется на нескольких горизонтах `h`.

Вопросы:
- Какой горизонт (`30m / 2h / 1d`) наиболее предсказуем?
- Что сильнее влияет на качество: размер данных, баланс классов или порог входа в сделку?
- Где модель систематически ошибается (ложные `UP`, ложные `DOWN`)?

---

## 4) Data Contract (обязательный формат данных)

### 4.1 `data/raw/text_events.csv`
Поля:
- `event_id` (str, уникальный id события)
- `timestamp_utc` (ISO8601, UTC)
- `ticker` (str)
- `source` (str: news/social/forum/etc)
- `text` (str)
- `lang` (str, ожидаем `en` или `ru`)

### 4.2 `data/raw/market_bars.csv`
Поля:
- `timestamp_utc` (ISO8601, UTC)
- `ticker` (str)
- `open`, `high`, `low`, `close` (float)
- `volume` (float)

Частота: единая внутри одного эксперимента (`1m` или `5m` и т.д.).

### 4.3 `data/processed/labeled_events.parquet`
Поля (минимум):
- `event_id`, `timestamp_utc`, `ticker`, `text`
- `ret_h` (float)
- `sigma_t` (float, реализованная волатильность на lookback-окне)
- `label` (`UP/FLAT/DOWN`)
- `split` (`train/val/test`) — только time-based split

---

## 5) Разметка таргета (без утечки)

### 5.1 Доходность
`ret_h = (P(t+h) - P(t)) / P(t)`

### 5.2 Пороговая схема
`thr_t = k * sigma_t`, где `sigma_t` считается только из данных **до `t`**.

Метка:
- `UP`, если `ret_h > thr_t`
- `DOWN`, если `ret_h < -thr_t`
- `FLAT` иначе

Рекомендуемые сетки для абляции:
- `h`: `[30m, 2h, 1d]`
- `k`: `[0.5, 1.0, 1.5]`

---

## 6) Анти-утечка (критично)
- Только `time-based split`, никакого random split.
- Train/Val/Test должны быть последовательными по календарю.
- Purged split: удалять события около границ, чтобы окна `(t, t+h]` не пересекались между сплитами.
- Все трансформации/балансировка/калибровка учатся только на train.
- Dedup near-duplicate текстов до split (иначе завышение качества).

---

## 7) Модели

### 7.1 Baseline
- `TF-IDF (1-2 grams) + LogisticRegression`
- Нужен для честного сравнения, обязательно в отчете.

### 7.2 Основная модель
- `distilbert-base-uncased`
- Classification head на 3 класса
- Loss: weighted CE (или focal при сильном дисбалансе)
- Early stopping по `macro F1` на val

### 7.3 Опционально
- calibration (temperature scaling / isotonic) на val
- threshold tuning для торгового сигнала

---

## 8) Метрики

### 8.1 ML-метрики
- `macro F1` (главная)
- `balanced accuracy`
- `precision/recall` по классам
- confusion matrix
- calibration curve + Brier score

### 8.2 Trading-метрики (event-driven)
- `Precision@top-k` по уверенному `UP`/`DOWN`
- cumulative return
- Sharpe
- max drawdown
- hit-rate
- turnover

С учетом комиссий и slippage (пусть даже простой фиксированный rate).

---

## 9) Пайплайн (файлы/скрипты)

Рекомендуемая структура:

```text
project/
  data/
    raw/
      text_events.csv
      market_bars.csv
    processed/
  artifacts/
    models/
    metrics/
    plots/
  src/
    01_prepare_events.py
    02_label_events.py
    03_split_time_purged.py
    04_train_baseline_tfidf.py
    05_train_distilbert.py
    06_eval_ml.py
    07_backtest_event.py
    08_ablation_runner.py
    09_report_pack.py
```

Порядок запуска:
1. `01_prepare_events.py`
2. `02_label_events.py --h ... --k ...`
3. `03_split_time_purged.py`
4. `04_train_baseline_tfidf.py`
5. `05_train_distilbert.py`
6. `06_eval_ml.py`
7. `07_backtest_event.py`
8. `08_ablation_runner.py`
9. `09_report_pack.py`

---

## 10) Обязательные артефакты для курсовой
- Таблица сплитов по датам и размеру (`train/val/test`).
- Распределение классов по сплитам.
- Сравнение baseline vs DistilBERT (таблица метрик).
- Confusion matrix (test).
- Калибровка вероятностей.
- Precision@top-k curve.
- Equity curve и drawdown curve (бэктест).
- Абляция по `h` и `k`.
- Error analysis: 20-30 примеров уверенных ошибок.

---

## 11) Формат НИР-раздела в отчете (готовый каркас)
1. Постановка задачи и гипотезы.
2. Источники данных и протокол синхронизации.
3. Разметка таргета и контроль утечки.
4. Описание моделей (baseline + DistilBERT).
5. Протокол эксперимента.
6. Результаты ML-оценки.
7. Результаты event-driven бэктеста.
8. Абляционные эксперименты.
9. Угрозы валидности и ограничения.
10. Выводы и практическая применимость.

---

## 12) Definition of Done
Считаем этап завершенным, если:
- есть воспроизводимый запуск от raw до итоговых метрик;
- DistilBERT и baseline оценены на **одном и том же** out-of-time test;
- построены ML + trading графики;
- в отчете есть секции по ограничениям/валидности;
- все результаты привязаны к конкретным версиям данных/параметрам (`h`, `k`, seed).

---

## 13) Риски и смягчение
- Дисбаланс классов -> weighted loss, stratified diagnostics по классам.
- Переобучение на коротком периоде -> rolling out-of-time проверки.
- Слабая переносимость -> тест на другом временном окне/тикерах.
- Ложная торговая прибыль -> обязательно учитывать costs/slippage.

---

## 14) Быстрый план handoff (кто что делает)
- **Data owner:** готовит и валидирует `text_events.csv` + `market_bars.csv`.
- **ML owner:** разметка, сплиты, baseline, DistilBERT, calibration.
- **Quant owner:** event-driven бэктест и риск-метрики.
- **Report owner:** упаковка таблиц/графиков и НИР-описание в курсовой.

Если исполнитель один — выполнять роли в указанном порядке.
