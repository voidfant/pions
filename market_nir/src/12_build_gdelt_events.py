#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path
from urllib.parse import urlparse

from common import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build text_events.csv from GDELT GKG dumps")
    p.add_argument("--input-dir", default="market_nir/data/raw/gdelt/extracted")
    p.add_argument("--pattern", default="*.gkg.csv")
    p.add_argument("--output", default="market_nir/data/raw/text_events_gdelt.csv")
    p.add_argument("--ticker", default="SPY", help="Ticker label for all records")
    p.add_argument("--lang", default="en")
    p.add_argument("--min-text-len", type=int, default=20)
    p.add_argument("--max-files", type=int, default=0, help="0 means all files")
    p.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    return p.parse_args()


def ts14_to_iso(ts14: str) -> str:
    parsed = dt.datetime.strptime(ts14, "%Y%m%d%H%M%S")
    return parsed.replace(tzinfo=dt.timezone.utc).isoformat()


def clean_field(value: str) -> str:
    return value.replace("\t", " ").replace("\n", " ").strip()


def pick_first_nonempty(row: list[str], indexes: list[int]) -> str:
    for idx in indexes:
        if idx < len(row):
            value = clean_field(row[idx])
            if value:
                return value
    return ""


def compose_text(
    themes: str,
    persons: str,
    orgs: str,
    locations: str,
    tone: str,
    url: str,
) -> str:
    domain = urlparse(url).netloc.lower() if url else ""
    chunks: list[str] = []
    if themes:
        chunks.append(f"Themes: {themes}")
    if persons:
        chunks.append(f"Persons: {persons}")
    if orgs:
        chunks.append(f"Organizations: {orgs}")
    if locations:
        chunks.append(f"Locations: {locations}")
    if tone:
        chunks.append(f"Tone: {tone}")
    if domain:
        chunks.append(f"Domain: {domain}")
    return " | ".join(chunks)


def main() -> None:
    args = parse_args()
    # GDELT rows may contain very large text-like fields; raise CSV parser limit.
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No files found: {input_dir}/{args.pattern}")

    out_path = Path(args.output)
    ensure_dir(out_path.parent)

    # GKG V2.1 common positions used here.
    # The parser includes fallback indexes because some dumps expose slightly different layouts.
    idx_date = 1
    idx_source_primary = [3, 4]
    idx_url_primary = [4, 5]
    idx_themes_primary = [7, 15]
    idx_locations_primary = [9, 17]
    idx_persons_primary = [11, 19]
    idx_orgs_primary = [13, 21]
    idx_tone_primary = [15, 23]
    max_idx_guard = max(idx_tone_primary)

    kept = 0
    seen_ids: set[str] = set()
    rows_total = 0
    files_used = 0
    time_min = None
    time_max = None

    with out_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=["event_id", "timestamp_utc", "ticker", "source", "lang", "text"],
        )
        writer.writeheader()

        for file_idx, path in enumerate(files, start=1):
            print(f"[{file_idx}/{len(files)}] parsing {path.name}")
            file_kept = 0
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f, delimiter="\t")
                for row_idx, row in enumerate(reader, start=1):
                    rows_total += 1
                    if args.max_rows > 0 and kept >= args.max_rows:
                        break
                    if len(row) <= max_idx_guard:
                        continue

                    ts_raw = clean_field(row[idx_date])
                    if len(ts_raw) != 14 or not ts_raw.isdigit():
                        continue

                    source = pick_first_nonempty(row, idx_source_primary)
                    url = pick_first_nonempty(row, idx_url_primary)
                    themes = pick_first_nonempty(row, idx_themes_primary)
                    locations = pick_first_nonempty(row, idx_locations_primary)
                    persons = pick_first_nonempty(row, idx_persons_primary)
                    orgs = pick_first_nonempty(row, idx_orgs_primary)
                    tone = pick_first_nonempty(row, idx_tone_primary)
                    source = source or (urlparse(url).netloc.lower() if url else "gdelt")
                    text = compose_text(themes, persons, orgs, locations, tone, url)
                    if len(text) < args.min_text_len:
                        continue

                    event_id = f"gdelt_{path.stem}_{row_idx}"
                    if event_id in seen_ids:
                        continue
                    seen_ids.add(event_id)

                    ts_iso = ts14_to_iso(ts_raw)
                    writer.writerow(
                        {
                            "event_id": event_id,
                            "timestamp_utc": ts_iso,
                            "ticker": args.ticker,
                            "source": source,
                            "lang": args.lang,
                            "text": text,
                        }
                    )
                    kept += 1
                    file_kept += 1
                    time_min = ts_iso if time_min is None or ts_iso < time_min else time_min
                    time_max = ts_iso if time_max is None or ts_iso > time_max else time_max

                if args.max_rows > 0 and kept >= args.max_rows:
                    files_used = file_idx
                    break

            files_used = file_idx
            print(f"  kept rows from file: {file_kept}")

    summary = {
        "input_dir": str(input_dir),
        "files_matched": len(files),
        "files_used": files_used,
        "rows_scanned": rows_total,
        "rows_kept": kept,
        "output": str(out_path),
        "ticker": args.ticker,
        "lang": args.lang,
        "time_min": time_min,
        "time_max": time_max,
    }
    write_json(summary, "market_nir/artifacts/metrics/12_build_gdelt_events_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
