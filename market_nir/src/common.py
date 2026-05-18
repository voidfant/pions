from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support

LABEL_ORDER = ["DOWN", "FLAT", "UP"]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")


def write_json(obj: Dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def classify_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_ORDER, zero_division=0
    )
    for i, label in enumerate(LABEL_ORDER):
        metrics[f"precision_{label}"] = float(prec[i])
        metrics[f"recall_{label}"] = float(rec[i])
        metrics[f"f1_{label}"] = float(f1[i])
        metrics[f"support_{label}"] = float(sup[i])

    return metrics


def map_label_to_id(labels: Iterable[str]) -> np.ndarray:
    idx = {label: i for i, label in enumerate(LABEL_ORDER)}
    return np.array([idx[x] for x in labels], dtype=np.int64)


def map_id_to_label(ids: Iterable[int]) -> np.ndarray:
    arr = np.array(list(ids), dtype=np.int64)
    return np.array([LABEL_ORDER[int(x)] for x in arr], dtype=object)
