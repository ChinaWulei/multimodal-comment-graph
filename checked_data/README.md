# CHECKED local data

This directory contains an unmodified shallow clone of the official CHECKED
repository plus a resumable downloader for the main-post images referenced by
the dataset JSON files.

## Layout

```text
checked_data/
├── official_repo/             # official cyang03/CHECKED repository
├── images/<label>/<news_id>/  # validated downloaded main-post images
├── download_checked_images.py
├── image_manifest.csv
├── download_status.sqlite3
├── dataset_audit.json
└── download_summary.json
```

## Commands (PowerShell)

Audit without network access:

```powershell
python .\checked_data\download_checked_images.py --audit-only
```

Download or resume all main-post images:

```powershell
python .\checked_data\download_checked_images.py
```

Test only the first 20 manifest entries:

```powershell
python .\checked_data\download_checked_images.py --limit 20
```

Successful files are skipped after Pillow verification. Downloads use a
temporary `.part` file and are atomically renamed only after validation. The
SQLite database records successes and failures, so the same command resumes an
interrupted run.

The downloader first tries the Sina `large` variant when the official URL is a
thumbnail, then falls back to the exact official URL. It uses one request at a
time, waits between requests, respects `Retry-After`, and does not bypass HTTP
403 or other access restrictions.
