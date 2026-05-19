#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

from common import ensure_dir, write_json

MASTERFILE_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

SUFFIX_BY_DATASET = {
    "gkg": ".gkg.csv.zip",
    "events": ".export.CSV.zip",
    "mentions": ".mentions.CSV.zip",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download multi-GB GDELT dataset snapshot")
    p.add_argument("--dataset", choices=sorted(SUFFIX_BY_DATASET), default="gkg")
    p.add_argument("--master-url", default=MASTERFILE_URL)
    p.add_argument("--start-date", help="YYYY-MM-DD (UTC), optional")
    p.add_argument("--end-date", help="YYYY-MM-DD (UTC), optional")
    p.add_argument("--target-gb", type=float, default=2.0, help="Target total zip size in GB")
    p.add_argument("--max-files", type=int, default=0, help="0 means no limit")
    p.add_argument("--order", choices=["asc", "desc"], default="desc")
    p.add_argument("--out-dir", default="market_nir/data/raw/gdelt")
    p.add_argument("--extract", action="store_true", help="Extract every downloaded zip")
    p.add_argument("--timeout-sec", type=int, default=120)
    return p.parse_args()


def parse_day(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)


def parse_master_line(line: str) -> tuple[int, str] | None:
    line = line.strip()
    if not line:
        return None
    parts = line.split(maxsplit=2)
    if len(parts) < 3:
        return None
    try:
        size = int(parts[0])
    except ValueError:
        return None
    return size, parts[2]


def timestamp_from_url(url: str) -> dt.datetime | None:
    name = url.rsplit("/", 1)[-1]
    m = re.match(r"^(\d{14})\.", name)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)


def iter_candidates(
    master_url: str,
    suffix: str,
    start: dt.datetime | None,
    end: dt.datetime | None,
    timeout_sec: int,
) -> list[tuple[dt.datetime, int, str]]:
    with urllib.request.urlopen(master_url, timeout=timeout_sec) as resp:
        payload = resp.read().decode("utf-8", errors="replace")

    out: list[tuple[dt.datetime, int, str]] = []
    for raw in payload.splitlines():
        parsed = parse_master_line(raw)
        if not parsed:
            continue
        size, url = parsed
        if not url.endswith(suffix):
            continue
        ts = timestamp_from_url(url)
        if ts is None:
            continue
        if start and ts < start:
            continue
        if end and ts >= end + dt.timedelta(days=1):
            continue
        out.append((ts, size, url))
    return out


def local_zip_bytes(download_dir: Path) -> int:
    return sum(p.stat().st_size for p in download_dir.glob("*.zip") if p.is_file())


def download_file(url: str, dst: Path, timeout_sec: int) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "market-nir-gdelt-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp, dst.open("wb") as out:
        total = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
        return total


def md5sum(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def extract_zip(zip_path: Path, extract_dir: Path) -> list[str]:
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = extract_dir / member
            if target.exists() and target.stat().st_size > 0:
                extracted.append(str(target))
                continue
            zf.extract(member, extract_dir)
            extracted.append(str(target))
    return extracted


def main() -> None:
    args = parse_args()
    suffix = SUFFIX_BY_DATASET[args.dataset]
    start = parse_day(args.start_date)
    end = parse_day(args.end_date)

    out_dir = ensure_dir(args.out_dir)
    zip_dir = ensure_dir(out_dir / "zips")
    extract_dir = ensure_dir(out_dir / "extracted")
    manifest_path = out_dir / f"manifest_{args.dataset}.csv"

    target_bytes = int(args.target_gb * (1024**3))
    existing_bytes = local_zip_bytes(zip_dir)
    print(
        {
            "dataset": args.dataset,
            "target_gb": args.target_gb,
            "existing_zip_gb": round(existing_bytes / (1024**3), 4),
            "out_dir": str(out_dir),
        }
    )

    candidates = iter_candidates(
        master_url=args.master_url,
        suffix=suffix,
        start=start,
        end=end,
        timeout_sec=args.timeout_sec,
    )
    candidates.sort(key=lambda x: x[0], reverse=(args.order == "desc"))
    print(f"Candidates selected: {len(candidates)}")
    if not candidates:
        raise SystemExit("No candidate files found for requested filters.")

    downloaded = 0
    downloaded_bytes = 0
    extracted_count = 0
    rows: list[str] = []

    if manifest_path.exists():
        rows.append(manifest_path.read_text(encoding="utf-8"))
    else:
        rows.append("timestamp_utc,size_bytes,url,zip_path,md5,extracted_files\n")

    for idx, (ts, size_hint, url) in enumerate(candidates, start=1):
        if existing_bytes >= target_bytes:
            print("Target size reached, stopping.")
            break
        if args.max_files > 0 and downloaded >= args.max_files:
            print("max-files reached, stopping.")
            break

        name = url.rsplit("/", 1)[-1]
        zip_path = zip_dir / name
        if zip_path.exists() and zip_path.stat().st_size > 0:
            existing_bytes += 0
            print(
                f"[{idx}/{len(candidates)}] skip existing {name} "
                f"({zip_path.stat().st_size / (1024**2):.2f} MB)"
            )
            continue

        print(f"[{idx}/{len(candidates)}] downloading {name} (hint {size_hint / (1024**2):.2f} MB)")
        tmp_path = zip_path.with_suffix(zip_path.suffix + ".part")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            bytes_written = download_file(url=url, dst=tmp_path, timeout_sec=args.timeout_sec)
            tmp_path.rename(zip_path)
        except Exception as exc:  # noqa: BLE001
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"Download failed for {url}: {exc}", file=sys.stderr)
            continue

        checksum = md5sum(zip_path)
        extracted_files_str = ""
        if args.extract:
            extracted_files = extract_zip(zip_path, extract_dir)
            extracted_count += len(extracted_files)
            extracted_files_str = "|".join(extracted_files)

        rows.append(
            f"{ts.isoformat()},{bytes_written},{url},{zip_path},{checksum},{extracted_files_str}\n"
        )
        downloaded += 1
        downloaded_bytes += bytes_written
        existing_bytes += bytes_written

    manifest_path.write_text("".join(rows), encoding="utf-8")

    summary = {
        "dataset": args.dataset,
        "target_gb": args.target_gb,
        "zip_dir": str(zip_dir),
        "extract_dir": str(extract_dir),
        "downloaded_files": downloaded,
        "downloaded_gb": round(downloaded_bytes / (1024**3), 6),
        "total_zip_gb_now": round(local_zip_bytes(zip_dir) / (1024**3), 6),
        "extracted_files": extracted_count,
        "manifest": str(manifest_path),
        "time_filter_start": args.start_date,
        "time_filter_end": args.end_date,
    }
    write_json(summary, out_dir / f"summary_{args.dataset}.json")
    print(summary)


if __name__ == "__main__":
    main()
