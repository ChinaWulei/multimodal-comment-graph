from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sqlite3
import time
from collections import Counter, defaultdict
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps

from common import load_config, resolve_path


FORMAT_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp"}


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator_structure.yaml"
    parser = argparse.ArgumentParser(description="Rate-limited, resumable image downloader for the selected structure subset.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--splits", nargs="+", choices=["train", "validate", "test"], default=["validate", "test"])
    parser.add_argument("--limit", type=int, default=0, help="Per selected split; 0 means all rows in the selected manifest.")
    parser.add_argument("--download", action="store_true", help="Required to issue network requests; otherwise only report existing files.")
    return parser.parse_args()


def init_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS image_status (
            news_id TEXT PRIMARY KEY, split TEXT NOT NULL, url TEXT NOT NULL, domain TEXT NOT NULL,
            status TEXT NOT NULL, http_status INTEGER, attempts INTEGER NOT NULL DEFAULT 0,
            local_path TEXT, error TEXT, updated_at REAL NOT NULL
        )"""
    )
    connection.commit()
    return connection


def find_complete_image(directory: Path, news_id: str) -> Path | None:
    for path in directory.glob(f"{news_id}.*"):
        if path.suffix == ".part" or not path.is_file() or path.stat().st_size <= 0:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            return path
        except Exception:
            continue
    return None


def retry_after_seconds(header: str | None) -> float | None:
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(header).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


def save_status(connection: sqlite3.Connection, row: dict, status: str, attempts: int, http_status: int | None, local_path: str = "", error: str = "") -> None:
    connection.execute(
        """INSERT INTO image_status(news_id,split,url,domain,status,http_status,attempts,local_path,error,updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(news_id) DO UPDATE SET split=excluded.split,url=excluded.url,domain=excluded.domain,
        status=excluded.status,http_status=excluded.http_status,attempts=excluded.attempts,
        local_path=excluded.local_path,error=excluded.error,updated_at=excluded.updated_at""",
        (row["news_id"], row["split"], row["image_url"], row["domain"], status, http_status, attempts, local_path, error[:1000], time.time()),
    )
    connection.commit()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    plan = config["official_subset"]
    data_dir = resolve_path(config_path, config["data"]["output_dir"])
    images_dir = data_dir / "official_subset_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    connection = init_db(data_dir / "official_image_download.sqlite3")
    rows = []
    for split in args.splits:
        path = data_dir / f"official_structure_subset_{split}.csv"
        if not path.is_file():
            raise RuntimeError(f"Missing selected subset manifest: {path}. Run plan_official_structure_subset.py first.")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            split_rows = list(csv.DictReader(handle))
        if args.limit > 0:
            split_rows = split_rows[: args.limit]
        for row in split_rows:
            row["split"] = split
            row["domain"] = urlparse(row["image_url"]).netloc.casefold() or "unknown"
            rows.append(row)
    session = requests.Session()
    session.headers.update({"User-Agent": "MFCL-research-image-subset/1.0 (respectful single-thread downloader)"})
    min_interval = float(plan["per_domain_request_interval_seconds"])
    timeout = float(plan["download_timeout_seconds"])
    max_attempts = int(plan["max_download_attempts"])
    last_request: dict[str, float] = defaultdict(lambda: 0.0)
    summary = Counter()
    domain_summary: dict[str, Counter] = defaultdict(Counter)
    for index, row in enumerate(rows, start=1):
        output_dir = images_dir / row["split"]
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = find_complete_image(output_dir, row["news_id"])
        if existing is not None:
            save_status(connection, row, "existing_valid", 0, None, str(existing))
            summary["existing_valid"] += 1
            domain_summary[row["domain"]]["existing_valid"] += 1
            continue
        if not args.download:
            summary["missing_not_downloaded"] += 1
            domain_summary[row["domain"]]["missing_not_downloaded"] += 1
            continue
        attempts = 0
        final_status = "failed"
        final_http = None
        final_error = ""
        local_path = ""
        while attempts < max_attempts:
            attempts += 1
            pause = min_interval - (time.monotonic() - last_request[row["domain"]])
            if pause > 0:
                time.sleep(pause)
            try:
                response = session.get(row["image_url"], timeout=timeout, stream=True)
                last_request[row["domain"]] = time.monotonic()
                final_http = response.status_code
                if response.status_code == 200:
                    content = response.content
                    with Image.open(io.BytesIO(content)) as image:
                        image = ImageOps.exif_transpose(image)
                        image.load()
                        image_format = image.format or "JPEG"
                    extension = FORMAT_EXTENSIONS.get(image_format.upper(), ".img")
                    final_path = output_dir / f"{row['news_id']}{extension}"
                    temp_path = final_path.with_suffix(final_path.suffix + ".part")
                    temp_path.write_bytes(content)
                    # Re-open temp file before making it visible as a completed image.
                    with Image.open(temp_path) as image:
                        image.verify()
                    temp_path.replace(final_path)
                    final_status, local_path = "downloaded_valid", str(final_path)
                    break
                if response.status_code in {403, 404}:
                    final_status = "forbidden" if response.status_code == 403 else "not_found"
                    final_error = f"HTTP {response.status_code}"
                    break
                if response.status_code == 429:
                    delay = retry_after_seconds(response.headers.get("Retry-After"))
                    delay = delay if delay is not None else min(60.0, 4.0 * (2 ** (attempts - 1)))
                    final_status, final_error = "rate_limited", f"HTTP 429; paused {delay:.1f}s"
                    if attempts < max_attempts:
                        time.sleep(delay)
                        continue
                    break
                if response.status_code in {500, 502, 503, 504}:
                    final_status, final_error = "server_error", f"HTTP {response.status_code}"
                    if attempts < max_attempts:
                        time.sleep(min(30.0, 2.0 * (2 ** (attempts - 1)) + random.random()))
                        continue
                    break
                final_status, final_error = "http_error", f"HTTP {response.status_code}"
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                final_status, final_error = "network_error", f"{type(exc).__name__}: {exc}"
                if attempts < max_attempts:
                    time.sleep(min(30.0, 2.0 * (2 ** (attempts - 1)) + random.random()))
                    continue
            except Exception as exc:
                final_status, final_error = "invalid_image", f"{type(exc).__name__}: {exc}"
                break
        save_status(connection, row, final_status, attempts, final_http, local_path, final_error)
        summary[final_status] += 1
        domain_summary[row["domain"]][final_status] += 1
        print(f"{index}/{len(rows)} {row['split']} {row['news_id']} {final_status} attempts={attempts}")
    report = {
        "download_enabled": bool(args.download), "splits": args.splits, "selected_rows": len(rows),
        "summary": dict(summary), "domains": {domain: dict(counts) for domain, counts in sorted(domain_summary.items())},
        "images_dir": str(images_dir), "status_db": str(data_dir / "official_image_download.sqlite3"),
    }
    report_path = data_dir / "official_image_download_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
