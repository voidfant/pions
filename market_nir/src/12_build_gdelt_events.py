#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
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


def parse_semicolon_list(value: str, limit: int = 8) -> list[str]:
    parts = [x.strip() for x in value.split(";") if x.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for item in parts:
        if "," in item:
            item = item.split(",", 1)[0].strip()
        item = item.replace("_", " ")
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def extract_page_title(extras_xml: str) -> str:
    m = re.search(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", extras_xml, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    title = re.sub(r"\s+", " ", m.group(1)).strip()
    return title


def compose_text(
    title: str,
    themes: str,
    persons: str,
    orgs: str,
    locations: str,
    tone: str,
    url: str,
) -> str:
    domain = urlparse(url).netloc.lower() if url else ""

    themes_items = parse_semicolon_list(themes, limit=8)
    persons_items = parse_semicolon_list(persons, limit=6)
    orgs_items = parse_semicolon_list(orgs, limit=6)
    locations_items = parse_semicolon_list(locations, limit=4)

    tone_clean = clean_field(tone)
    if "," in tone_clean:
        tone_clean = tone_clean.split(",", 1)[0].strip()

    chunks: list[str] = []
    if title:
        chunks.append(f"Title: {title}")
    if domain:
        chunks.append(f"Source: {domain}")
    if themes_items:
        chunks.append("Themes: " + ", ".join(themes_items))
    if persons_items:
        chunks.append("Persons: " + ", ".join(persons_items))
    if orgs_items:
        chunks.append("Organizations: " + ", ".join(orgs_items))
    if locations_items:
        chunks.append("Locations: " + ", ".join(locations_items))
    if tone_clean:
        chunks.append(f"Tone: {tone_clean}")
    text = " | ".join(chunks)
    return re.sub(r"\s+", " ", text).strip()


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

    # GKG V2.x canonical column indexes (27 columns).
    # We intentionally avoid GCAM/very-high-dimensional columns in text composition.
    idx_date = 1
    idx_source = 3
    idx_url = 4
    idx_themes = 7
    idx_locations = 9
    idx_persons = 11
    idx_orgs = 13
    idx_tone = 15
    idx_extras = 26
    max_idx_guard = max(idx_extras, idx_tone)

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

                    source = clean_field(row[idx_source]) if idx_source < len(row) else ""
                    url = clean_field(row[idx_url]) if idx_url < len(row) else ""
                    themes = clean_field(row[idx_themes]) if idx_themes < len(row) else ""
                    locations = clean_field(row[idx_locations]) if idx_locations < len(row) else ""
                    persons = clean_field(row[idx_persons]) if idx_persons < len(row) else ""
                    orgs = clean_field(row[idx_orgs]) if idx_orgs < len(row) else ""
                    tone = clean_field(row[idx_tone]) if idx_tone < len(row) else ""
                    extras = clean_field(row[idx_extras]) if idx_extras < len(row) else ""
                    title = extract_page_title(extras)
                    source = source or (urlparse(url).netloc.lower() if url else "gdelt")
                    text = compose_text(title, themes, persons, orgs, locations, tone, url)
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
