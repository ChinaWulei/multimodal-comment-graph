"""Download and validate main-post images from the official CHECKED dataset.

The official dataset stores image URLs rather than image binaries.  This tool
builds a reproducible manifest, downloads images conservatively, validates them
with Pillow, and records every result in SQLite so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import csv
import email.utils
import hashlib
import json
import os
import random
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "official_repo" / "dataset"
DEFAULT_IMAGES = ROOT / "images"
DEFAULT_DB = ROOT / "download_status.sqlite3"
DEFAULT_MANIFEST = ROOT / "image_manifest.csv"
DEFAULT_AUDIT = ROOT / "dataset_audit.json"
DEFAULT_SUMMARY = ROOT / "download_summary.json"

RETRYABLE_HTTP = {429, 500, 502, 503, 504}
NON_RETRYABLE_HTTP = {403, 404}
FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "GIF": ".gif",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "TIFF": ".tif",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CHECKED main-post images with resume and validation."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--status-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum image records to process; 0 means all records.",
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=45.0)
    parser.add_argument("--request-interval", type=float, default=0.5)
    parser.add_argument("--max-retry-wait", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def normalise_urls(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        url = str(item or "").strip()
        if url.startswith(("http://", "https://")) and url not in seen:
            output.append(url)
            seen.add(url)
    return output


def valid_comment_count(value: Any) -> int:
    if not isinstance(value, list):
        return 0
    count = 0
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text") or "").split()).strip("：: ")
        folded = text.casefold()
        if not text or folded in {"[deleted]", "[removed]"} or folded in seen:
            continue
        seen.add(folded)
        count += 1
    return count


def scan_dataset(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    news_counts: Counter[str] = Counter()
    image_news_counts: Counter[str] = Counter()
    comment_news_counts: Counter[str] = Counter()
    usable_news_counts: Counter[str] = Counter()
    total_comments = 0
    valid_comments = 0
    malformed = 0

    for folder_name in ("fake_news", "real_news"):
        folder = dataset_dir / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing CHECKED folder: {folder}")
        for path in sorted(folder.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                malformed += 1
                continue
            news_id = str(record.get("id") or path.stem).strip()
            label = str(record.get("label") or folder_name.removesuffix("_news")).strip()
            text = str(record.get("text") or "").strip()
            comments = record.get("comments")
            comment_count = len(comments) if isinstance(comments, list) else 0
            clean_comment_count = valid_comment_count(comments)
            urls = normalise_urls(record.get("pic_url"))
            news_counts[label] += 1
            total_comments += comment_count
            valid_comments += clean_comment_count
            if urls:
                image_news_counts[label] += 1
            if clean_comment_count:
                comment_news_counts[label] += 1
            if text and urls and clean_comment_count:
                usable_news_counts[label] += 1
            for image_index, url in enumerate(urls, start=1):
                rows.append(
                    {
                        "item_key": f"{news_id}:{image_index}",
                        "news_id": news_id,
                        "label": label,
                        "image_index": image_index,
                        "source_url": url,
                        "domain": (urlparse(url).hostname or "").lower(),
                        "comment_count": comment_count,
                        "valid_comment_count": clean_comment_count,
                        "has_text": bool(text),
                        "source_json": str(path.resolve()),
                    }
                )

    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir.resolve()),
        "news_total": sum(news_counts.values()),
        "news_by_label": dict(news_counts),
        "news_with_images": sum(image_news_counts.values()),
        "news_with_images_by_label": dict(image_news_counts),
        "news_with_valid_comments": sum(comment_news_counts.values()),
        "news_with_valid_comments_by_label": dict(comment_news_counts),
        "news_with_text_images_comments": sum(usable_news_counts.values()),
        "news_with_text_images_comments_by_label": dict(usable_news_counts),
        "main_post_image_urls": len(rows),
        "image_urls_by_domain": dict(Counter(row["domain"] for row in rows)),
        "raw_comments": total_comments,
        "valid_unique_nonempty_comments": valid_comments,
        "malformed_json_files": malformed,
    }
    return rows, audit


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["item_key"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def initialise_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS downloads (
            item_key TEXT PRIMARY KEY,
            news_id TEXT NOT NULL,
            label TEXT NOT NULL,
            image_index INTEGER NOT NULL,
            source_url TEXT NOT NULL,
            requested_url TEXT,
            domain TEXT,
            local_path TEXT,
            status TEXT NOT NULL,
            http_status INTEGER,
            attempts INTEGER NOT NULL DEFAULT 0,
            byte_count INTEGER,
            width INTEGER,
            height INTEGER,
            image_format TEXT,
            error TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def get_state(connection: sqlite3.Connection, item_key: str) -> sqlite3.Row | None:
    connection.row_factory = sqlite3.Row
    return connection.execute(
        "SELECT * FROM downloads WHERE item_key = ?", (item_key,)
    ).fetchone()


def record_state(connection: sqlite3.Connection, values: dict[str, Any]) -> None:
    columns = [
        "item_key", "news_id", "label", "image_index", "source_url",
        "requested_url", "domain", "local_path", "status", "http_status",
        "attempts", "byte_count", "width", "height", "image_format",
        "error", "updated_at",
    ]
    payload = [values.get(column) for column in columns]
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(f"{column}=excluded.{column}" for column in columns[1:])
    connection.execute(
        f"INSERT INTO downloads ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(item_key) DO UPDATE SET {updates}",
        payload,
    )
    connection.commit()


def candidate_urls(source_url: str) -> list[str]:
    parsed = urlparse(source_url)
    if not parsed.hostname or not parsed.hostname.endswith("sinaimg.cn"):
        return [source_url]
    candidates: list[str] = []
    for marker in ("/thumb150/", "/thumb180/", "/mw690/", "/orj360/"):
        if marker in source_url:
            candidates.append(source_url.replace(marker, "/large/"))
            break
    candidates.append(source_url)
    return list(dict.fromkeys(candidates))


def retry_after_seconds(value: str | None, default: float, maximum: float) -> float:
    if value:
        try:
            return min(maximum, max(0.0, float(value)))
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return min(maximum, max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds()))
            except (TypeError, ValueError, OverflowError):
                pass
    return min(maximum, default)


def validate_image(path: Path) -> tuple[str, int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            image_format = str(image.format or "").upper()
            width, height = image.size
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"Pillow validation failed: {exc}") from exc
    if width <= 0 or height <= 0 or not image_format:
        raise ValueError("Image has invalid dimensions or unknown format")
    return image_format, width, height


def existing_success(connection: sqlite3.Connection, item_key: str) -> bool:
    state = get_state(connection, item_key)
    if not state or state["status"] != "success" or not state["local_path"]:
        return False
    path = Path(state["local_path"])
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        validate_image(path)
        return True
    except ValueError:
        return False


def download_one(
    session: requests.Session,
    connection: sqlite3.Connection,
    row: dict[str, Any],
    image_dir: Path,
    args: argparse.Namespace,
    last_request: dict[str, float],
) -> str:
    if existing_success(connection, row["item_key"]):
        return "skipped_valid"

    news_dir = image_dir / row["label"] / row["news_id"]
    news_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{row['news_id']}__{int(row['image_index']):02d}"
    temp_path = news_dir / f"{stem}.part"
    total_attempts = 0
    final_status = "failed"
    final_http: int | None = None
    final_error = "No URL attempted"
    final_requested = row["source_url"]

    for requested_url in candidate_urls(row["source_url"]):
        final_requested = requested_url
        domain = (urlparse(requested_url).hostname or "").lower()
        for attempt in range(1, args.max_attempts + 1):
            total_attempts += 1
            elapsed = time.monotonic() - last_request.get(domain, 0.0)
            if elapsed < args.request_interval:
                time.sleep(args.request_interval - elapsed)
            try:
                last_request[domain] = time.monotonic()
                with session.get(
                    requested_url,
                    stream=True,
                    timeout=(args.connect_timeout, args.read_timeout),
                    allow_redirects=True,
                ) as response:
                    final_http = response.status_code
                    if response.status_code == 200:
                        hasher = hashlib.sha256()
                        byte_count = 0
                        with temp_path.open("wb") as handle:
                            for chunk in response.iter_content(chunk_size=128 * 1024):
                                if not chunk:
                                    continue
                                handle.write(chunk)
                                hasher.update(chunk)
                                byte_count += len(chunk)
                        image_format, width, height = validate_image(temp_path)
                        extension = FORMAT_EXTENSIONS.get(image_format, ".img")
                        final_path = news_dir / f"{stem}{extension}"
                        os.replace(temp_path, final_path)
                        record_state(
                            connection,
                            {
                                **row,
                                "requested_url": requested_url,
                                "local_path": str(final_path.resolve()),
                                "status": "success",
                                "http_status": 200,
                                "attempts": total_attempts,
                                "byte_count": byte_count,
                                "width": width,
                                "height": height,
                                "image_format": image_format,
                                "error": None,
                                "updated_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                        return "success"
                    if response.status_code in NON_RETRYABLE_HTTP:
                        final_status = f"http_{response.status_code}"
                        final_error = f"HTTP {response.status_code}"
                        break
                    if response.status_code in RETRYABLE_HTTP:
                        final_status = f"http_{response.status_code}"
                        final_error = f"HTTP {response.status_code}"
                        if attempt < args.max_attempts:
                            default_wait = min(args.max_retry_wait, 2 ** (attempt - 1) * 2.0)
                            wait = retry_after_seconds(
                                response.headers.get("Retry-After"), default_wait, args.max_retry_wait
                            )
                            time.sleep(wait + random.random() * 0.25)
                            continue
                        break
                    final_status = f"http_{response.status_code}"
                    final_error = f"Unexpected HTTP {response.status_code}"
                    break
            except KeyboardInterrupt:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                raise
            except (requests.RequestException, OSError, ValueError) as exc:
                temp_path.unlink(missing_ok=True)
                final_status = "network_or_invalid_image"
                final_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_attempts:
                    time.sleep(min(args.max_retry_wait, 2 ** (attempt - 1) * 2.0))
                    continue
                break

    temp_path.unlink(missing_ok=True)
    record_state(
        connection,
        {
            **row,
            "requested_url": final_requested,
            "local_path": None,
            "status": final_status,
            "http_status": final_http,
            "attempts": total_attempts,
            "byte_count": None,
            "width": None,
            "height": None,
            "image_format": None,
            "error": final_error[:1000],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return final_status


def make_summary(connection: sqlite3.Connection, selected: int, interrupted: bool) -> dict[str, Any]:
    status_counts = dict(
        connection.execute(
            "SELECT status, COUNT(*) FROM downloads GROUP BY status ORDER BY status"
        ).fetchall()
    )
    domain_rows = connection.execute(
        """
        SELECT domain, status, COUNT(*)
        FROM downloads
        GROUP BY domain, status
        ORDER BY domain, status
        """
    ).fetchall()
    by_domain: dict[str, dict[str, int]] = {}
    for domain, status, count in domain_rows:
        by_domain.setdefault(domain or "", {})[status] = count
    success_stats = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(byte_count), 0) FROM downloads WHERE status='success'"
    ).fetchone()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_manifest_rows": selected,
        "interrupted": interrupted,
        "status_counts": status_counts,
        "success_files": success_stats[0],
        "downloaded_bytes": success_stats[1],
        "downloaded_gib": round(success_stats[1] / (1024 ** 3), 4),
        "by_domain": by_domain,
    }


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    image_dir = args.image_dir.resolve()
    rows, audit = scan_dataset(dataset_dir)
    write_manifest(rows, args.manifest_csv.resolve())
    args.audit_json.resolve().write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.audit_only:
        return 0

    selected_rows = rows[: args.limit] if args.limit > 0 else rows
    image_dir.mkdir(parents=True, exist_ok=True)
    connection = initialise_db(args.status_db.resolve())
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "CHECKED-academic-dataset-downloader/1.0 (+non-commercial research)",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
    )
    counts: Counter[str] = Counter()
    interrupted = False
    try:
        last_request: dict[str, float] = {}
        for index, row in enumerate(selected_rows, start=1):
            result = download_one(session, connection, row, image_dir, args, last_request)
            counts[result] += 1
            if index == 1 or index % 50 == 0 or index == len(selected_rows):
                print(f"progress={index}/{len(selected_rows)} current={result} run_counts={dict(counts)}")
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted safely; completed status rows are preserved.")
    finally:
        session.close()
        summary = make_summary(connection, len(selected_rows), interrupted)
        args.summary_json.resolve().write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        connection.close()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
